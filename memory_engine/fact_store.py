"""事实记忆层: RAG 索引 + 可编辑(增删改) + 热/冷分层。

- 冷记忆: 全部事实存向量库, 查询时 RAG 召回 top-k
- 热记忆: 标记为 pinned 的高频事实, 总是直接披露
- 持久化: 事实存 JSON, 向量存 FAISS index (可重建)

增删改全部即时生效, 无需训练:
    add(fact)      -> 编码入向量库 + 追加 JSON
    delete(id)     -> 移除 + 重建索引
    update(id,...) -> 改 JSON + re-embed 该条
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FactStore:
    """可编辑的事实记忆: RAG 召回 + 披露式展开。"""

    def __init__(self, store_dir: str, embed_model: str = "BAAI/bge-m3"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.facts_file = self.store_dir / "facts.json"
        self.embed_model_name = embed_model

        self.facts: list[dict] = []   # [{"id", "text", "pinned", "ts"}]
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

    def add(self, text: str, pinned: bool = False) -> int:
        """新增一条事实。返回 fact id。

        若已存在完全相同的文本, 不重复添加, 返回已有 id(并按需更新 pinned)。
        """
        text = text.strip()
        for f in self.facts:
            if f["text"].strip() == text:
                if pinned and not f.get("pinned"):
                    f["pinned"] = True
                    self._save()
                logger.info(f"Fact already exists (#{f['id']}), skip add")
                return f["id"]
        fact = {"id": self._next_id, "text": text, "pinned": pinned, "ts": int(time.time())}
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

    # ---- 召回 (RAG 索引 -> 披露) ----

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """RAG 召回 top-k 相关事实 + 所有 pinned 热记忆。

        返回去重后的事实列表, 供披露式展开。
        """
        results: dict[int, dict] = {}

        # 1. 热记忆(pinned)总是包含
        for f in self.facts:
            if f.get("pinned"):
                results[f["id"]] = f

        # 2. RAG 召回 top-k 冷记忆
        if self.facts and query.strip():
            import numpy as np
            if self._index is None:
                self._rebuild_index()
            self._ensure_model()
            q_emb = self._model.encode([query], normalize_embeddings=True)
            k = min(top_k, len(self.facts))
            scores, idxs = self._index.search(np.array(q_emb, dtype=np.float32), k)
            for score, idx in zip(scores[0], idxs[0]):
                if 0 <= idx < len(self.facts):
                    f = self.facts[idx]
                    results[f["id"]] = {**f, "score": float(score)}

        return list(results.values())

    def build_disclosure(self, query: str, top_k: int = 5) -> str:
        """召回 + 披露式展开为 context 文本块。"""
        facts = self.retrieve(query, top_k=top_k)
        if not facts:
            return ""
        lines = ["[USER MEMORY]"]
        for f in facts:
            tag = "[pinned] " if f.get("pinned") else ""
            lines.append(f"- {tag}{f['text']}")
        lines.append("[END USER MEMORY]")
        return "\n".join(lines)
