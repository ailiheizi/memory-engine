"""事实记忆层: RAG 索引 + 可编辑(增删改) + 热/冷分层 + Hermes 式信任加权。

- 冷记忆: 全部事实存向量库, 查询时 RAG 召回 top-k
- 热记忆: 标记为 pinned 的高频事实, 总是直接披露
- 信任层(借鉴 Hermes): 每条记忆有 trust 分, 被采用则强化, 久未用则随时间衰减
    最终召回分 = 语义相似度 × 有效信任(trust × 时间衰减)
    -> 可信+常用+新鲜的记忆浮上来, 老噪音自然沉底 (裸 RAG 做不到)
- 持久化: 事实存 JSON, 向量存 FAISS index (可重建)

增删改全部即时生效, 无需训练:
    add(fact)      -> 编码入向量库 + 追加 JSON
    delete(id)     -> 移除 + 重建索引
    update(id,...) -> 改 JSON + re-embed 该条
    reinforce(id)  -> 信任上升(记忆被采用时调用)
"""

from __future__ import annotations

import json
import time
import math
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---- Hermes 式信任参数 ----
DEFAULT_TRUST = 0.5          # 新记忆初始信任
TRUST_MIN = 0.05            # 信任下限(衰减不到 0, 保留可召回性)
TRUST_MAX = 1.0
REINFORCE_GAIN = 0.15       # 每次被采用, 信任向 1 靠拢的幅度
DECAY_HALF_LIFE_DAYS = 30.0  # 未使用信任减半的天数 (0 = 关闭衰减)
_DAY_SECONDS = 86400.0


class FactStore:
    """可编辑的事实记忆: RAG 召回 + 信任加权 + 披露式展开。"""

    def __init__(
        self,
        store_dir: str,
        embed_model: str = "BAAI/bge-m3",
        decay_half_life_days: float = DECAY_HALF_LIFE_DAYS,
    ):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.facts_file = self.store_dir / "facts.json"
        self.embed_model_name = embed_model
        self.half_life = decay_half_life_days

        self.facts: list[dict] = []   # [{"id","text","pinned","ts","trust","uses","last_used"}]
        self._next_id = 1
        self._model = None
        self._index = None
        self._dim = None
        self._load()

    # ---- 持久化 ----

    def _load(self):
        if self.facts_file.exists():
            data = json.loads(self.facts_file.read_text(encoding="utf-8"))
            self.facts = data.get("facts", [])
            self._next_id = data.get("next_id", len(self.facts) + 1)

    def _save(self):
        self.facts_file.write_text(
            json.dumps({"facts": self.facts, "next_id": self._next_id}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---- 延迟加载 embedding 模型 ----

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.embed_model_name}")
            self._model = SentenceTransformer(self.embed_model_name)
            try:
                self._dim = self._model.get_embedding_dimension()
            except AttributeError:
                self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def _rebuild_index(self):
        """从当前 facts 重建 FAISS 索引。增删后调用。"""
        import faiss
        import numpy as np

        self._ensure_model()
        self._index = faiss.IndexFlatIP(self._dim)
        if self.facts:
            texts = [f["text"] for f in self.facts]
            embs = self._model.encode(texts, normalize_embeddings=True)
            self._index.add(np.array(embs, dtype=np.float32))

    # ---- 增删改 (即时生效) ----

    def add(self, text: str, pinned: bool = False, metadata: dict | None = None) -> int:
        """新增一条事实。返回 fact id。

        Args:
            text: 事实文本（用于语义检索）
            pinned: 是否为热记忆（始终召回）
            metadata: 可选结构化附加数据（不参与检索，但随结果返回）

        若已存在完全相同的文本, 不重复添加, 返回已有 id(并按需更新 pinned/metadata)。
        """
        text = text.strip()
        for f in self.facts:
            if f["text"].strip() == text:
                if pinned and not f.get("pinned"):
                    f["pinned"] = True
                if metadata and metadata != f.get("metadata"):
                    f["metadata"] = metadata
                    self._save()
                logger.info(f"Fact already exists (#{f['id']}), skip add")
                return f["id"]
        fact = {
            "id": self._next_id, "text": text, "pinned": pinned, "ts": int(time.time()),
            "trust": DEFAULT_TRUST, "uses": 0, "last_used": int(time.time()),
        }
        if metadata:
            fact["metadata"] = metadata
        self.facts.append(fact)
        self._next_id += 1
        self._save()
        self._rebuild_index()
        logger.info(f"Added fact #{fact['id']}: {text[:40]}")
        return fact["id"]

    def delete(self, fact_id: int) -> bool:
        """删除一条事实。"""
        before = len(self.facts)
        self.facts = [f for f in self.facts if f["id"] != fact_id]
        if len(self.facts) < before:
            self._save()
            self._rebuild_index()
            logger.info(f"Deleted fact #{fact_id}")
            return True
        return False

    def update(self, fact_id: int, text: Optional[str] = None, pinned: Optional[bool] = None) -> bool:
        """修改一条事实(改文本会 re-embed)。"""
        for f in self.facts:
            if f["id"] == fact_id:
                if text is not None:
                    f["text"] = text
                if pinned is not None:
                    f["pinned"] = pinned
                f["ts"] = int(time.time())
                self._save()
                if text is not None:
                    self._rebuild_index()
                logger.info(f"Updated fact #{fact_id}")
                return True
        return False

    def list_facts(self) -> list[dict]:
        return list(self.facts)

    # ---- Hermes 式信任层 ----

    def _effective_trust(self, fact: dict, now: Optional[float] = None) -> float:
        """有效信任 = 基础信任 × 时间衰减。pinned 记忆不衰减。"""
        trust = fact.get("trust", DEFAULT_TRUST)
        if fact.get("pinned") or self.half_life <= 0:
            return trust
        now = now if now is not None else time.time()
        age_days = max(0.0, (now - fact.get("last_used", fact.get("ts", now))) / _DAY_SECONDS)
        decay = 0.5 ** (age_days / self.half_life)
        return max(TRUST_MIN, trust * decay)

    def reinforce(self, fact_id: int, gain: float = REINFORCE_GAIN) -> bool:
        """记忆被采用 -> 信任向上限靠拢, uses+1, 刷新 last_used。

        被检索且实际用于回答时调用, 实现 Hermes 式"常用记忆变强"。
        """
        for f in self.facts:
            if f["id"] == fact_id:
                t = f.get("trust", DEFAULT_TRUST)
                f["trust"] = min(TRUST_MAX, t + gain * (TRUST_MAX - t))
                f["uses"] = f.get("uses", 0) + 1
                f["last_used"] = int(time.time())
                self._save()
                return True
        return False

    # ---- 召回 (RAG 索引 -> 披露) ----

    def retrieve(self, query: str, top_k: int = 5, min_trust: float = 0.0,
                 reinforce: bool = False) -> list[dict]:
        """RAG 召回 + 信任加权 + 所有 pinned 热记忆。

        最终分 = 语义相似度 × 有效信任(trust × 时间衰减)。
        召回多候选后按最终分排序, 取 top_k。pinned 始终包含且排最前。

        Args:
            min_trust: 过滤有效信任低于此值的记忆 (默认 0 = 不过滤)
            reinforce: 是否对召回结果做强化 (检索即采用时设 True)
        """
        now = time.time()
        results: dict[int, dict] = {}

        # 1. 热记忆(pinned)总是包含
        for f in self.facts:
            if f.get("pinned"):
                results[f["id"]] = {**f, "score": 1.0, "eff_trust": self._effective_trust(f, now),
                                    "final": 999.0}  # pinned 排最前

        # 2. RAG 召回候选 (多取一些, 再用信任重排)
        if self.facts and query.strip():
            import numpy as np
            if self._index is None:
                self._rebuild_index()
            self._ensure_model()
            q_emb = self._model.encode([query], normalize_embeddings=True)
            cand_k = min(max(top_k * 3, top_k), len(self.facts))  # 候选池放大 3x
            scores, idxs = self._index.search(np.array(q_emb, dtype=np.float32), cand_k)
            for score, idx in zip(scores[0], idxs[0]):
                if 0 <= idx < len(self.facts):
                    f = self.facts[idx]
                    if f["id"] in results:  # 已作为 pinned 加入
                        continue
                    eff = self._effective_trust(f, now)
                    if eff < min_trust:
                        continue
                    # 被 supersede 的记忆额外惩罚(矛盾检测标记的)
                    if f.get("superseded_by"):
                        eff *= 0.1
                    # 加法加权: 语义决定大局, trust 只在相近时加分(不会压制高语义)
                    final = float(score) + 0.1 * eff
                    results[f["id"]] = {**f, "score": float(score), "eff_trust": eff, "final": final}

        # 3. 按最终分排序, 取 top_k (pinned final=999 自然排前, superseded 排最后)
        ranked = sorted(results.values(), key=lambda r: (
            0 if r.get("superseded_by") else 1,  # superseded 排后面
            r.get("final", 0),
        ), reverse=True)
        out = ranked[:max(top_k, sum(1 for r in ranked if r.get("pinned")))]

        # 4. 可选强化
        if reinforce:
            for r in out:
                self.reinforce(r["id"])

        return out

    def build_disclosure(self, query: str, top_k: int = 5, min_trust: float = 0.0,
                         reinforce: bool = False) -> str:
        """召回 + 披露式展开为 context 文本块。

        reinforce=True 时, 被召回的记忆视为"被采用", 信任上升(Hermes 式强化)。
        """
        facts = self.retrieve(query, top_k=top_k, min_trust=min_trust, reinforce=reinforce)
        if not facts:
            return ""
        lines = ["[USER MEMORY]"]
        for f in facts:
            tag = "[pinned] " if f.get("pinned") else ""
            lines.append(f"- {tag}{f['text']}")
        lines.append("[END USER MEMORY]")
        return "\n".join(lines)
