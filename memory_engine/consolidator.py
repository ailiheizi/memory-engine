"""睡眠巩固: 手动触发的离线记忆整理。

借鉴 SCM / Auto-Dreamer 的核心操作, 但不做定时/不做学习策略, 纯规则+手动触发:

操作:
1. cleanup: 删除 superseded + 极低trust 的垃圾记忆
2. merge: 合并高相似度的冗余记忆(DeepSeek 生成精炼合并版)
3. reweight: 多维重算重要性(recency + frequency + trust)
4. full: 以上全做(一次"完整睡眠")

调用:
    engine.consolidate(mode="full")  # 或 "cleanup" / "merge" / "reweight"

设计原则:
- 手动触发(API调用), 不自动定时
- 每步都有 dry_run 模式(只报告不执行)
- 合并用 DeepSeek 生成精炼版(保证语义不丢)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# 合并阈值: 两条记忆 sim 超过此值认为是"说同一件事"
MERGE_SIMILARITY_THRESHOLD = 0.85
# 清理阈值: trust 低于此值 + superseded 的删掉
CLEANUP_TRUST_THRESHOLD = 0.08
# 容量软上限: 超过后 consolidate 会更积极地合并/清理
CAPACITY_SOFT_LIMIT = 500


class SleepConsolidator:
    """离线记忆巩固器(手动触发)。"""

    def __init__(self, fact_store, deepseek=None):
        """
        Args:
            fact_store: FactStore 实例
            deepseek: DeepSeekClient(合并时用来生成精炼版; None=只做清理不合并)
        """
        self.facts = fact_store
        self.ds = deepseek

    def consolidate(self, mode: str = "full", dry_run: bool = False) -> dict:
        """执行巩固。

        Args:
            mode: "cleanup" / "merge" / "reweight" / "full"
            dry_run: True=只报告不执行

        Returns:
            {"cleaned": n, "merged": n, "reweighted": n, "details": [...]}
        """
        result = {"cleaned": 0, "merged": 0, "reweighted": 0, "evicted": 0, "details": []}

        if mode in ("cleanup", "full"):
            r = self._cleanup(dry_run=dry_run)
            result["cleaned"] = r["removed"]
            result["details"].extend(r["details"])

        if mode in ("evict", "full"):
            r = self._evict(dry_run=dry_run)
            result["evicted"] = r["evicted"]
            result["details"].extend(r["details"])

        if mode in ("merge", "full"):
            r = self._merge(dry_run=dry_run)
            result["merged"] = r["merged"]
            result["details"].extend(r["details"])

        if mode in ("reweight", "full"):
            r = self._reweight(dry_run=dry_run)
            result["reweighted"] = r["updated"]
            result["details"].extend(r["details"])

        return result

    def _cleanup(self, dry_run: bool = False) -> dict:
        """删除垃圾记忆: superseded + 极低 trust。"""
        candidates = []
        for f in self.facts.facts:
            reasons = []
            if f.get("superseded_by") and f.get("trust", 0.5) < 0.15:
                reasons.append("superseded+low_trust")
            elif f.get("trust", 0.5) < CLEANUP_TRUST_THRESHOLD:
                reasons.append("very_low_trust")
            if reasons:
                candidates.append({"id": f["id"], "text": f["text"][:50], "trust": f.get("trust"), "reasons": reasons})

        if not dry_run:
            for c in candidates:
                self.facts.delete(c["id"])

        return {"removed": len(candidates), "details": [
            {"action": "cleanup", "dry_run": dry_run, "candidates": candidates}
        ]}

    def _evict(self, dry_run: bool = False, limit: int = CAPACITY_SOFT_LIMIT) -> dict:
        """容量淘汰: 超过上限时删除最低 trust 的记忆(pinned 不删)。"""
        facts = self.facts.facts
        if len(facts) <= limit:
            return {"evicted": 0, "details": []}

        # 按 trust 排序, pinned 排最后(不删)
        deletable = [f for f in facts if not f.get("pinned")]
        deletable.sort(key=lambda f: f.get("trust", 0.5))

        n_to_delete = len(facts) - limit
        to_evict = deletable[:n_to_delete]

        if not dry_run:
            for f in to_evict:
                self.facts.delete(f["id"])

        return {"evicted": len(to_evict), "details": [
            {"action": "evict", "dry_run": dry_run, "limit": limit,
             "evicted": [{"id": f["id"], "text": f["text"][:40], "trust": f.get("trust")} for f in to_evict]}
        ]}

    def _merge(self, dry_run: bool = False) -> dict:
        """合并高相似度冗余记忆。

        找到 sim > MERGE_SIMILARITY_THRESHOLD 的记忆对,
        用 DeepSeek 合并成一条精炼版, 删除原来的两条。
        """
        if not self.ds:
            return {"merged": 0, "details": [{"action": "merge", "skipped": "no DeepSeek client"}]}

        import numpy as np
        self.facts._ensure_model()
        if self.facts._index is None:
            self.facts._rebuild_index()

        # 找所有高相似对
        merged_ids = set()
        merge_groups = []
        facts = self.facts.facts

        for i, f in enumerate(facts):
            if f["id"] in merged_ids:
                continue
            # 在索引里找和它最相似的
            q_emb = self.facts._model.encode([f["text"]], normalize_embeddings=True)
            k = min(5, len(facts))
            scores, idxs = self.facts._index.search(np.array(q_emb, dtype=np.float32), k)
            group = [f]
            for score, idx in zip(scores[0], idxs[0]):
                if idx == i or idx < 0 or idx >= len(facts):
                    continue
                candidate = facts[idx]
                if candidate["id"] in merged_ids:
                    continue
                if float(score) >= MERGE_SIMILARITY_THRESHOLD:
                    group.append(candidate)
                    merged_ids.add(candidate["id"])
            if len(group) > 1:
                merged_ids.add(f["id"])
                merge_groups.append(group)

        # 对每组用 DeepSeek 生成合并版
        merged_count = 0
        details = []
        for group in merge_groups:
            texts = [g["text"] for g in group]
            prompt = (
                "以下几条记忆描述的是同一件事或高度相关的信息。"
                "请合并成一条简洁的陈述句(保留所有关键信息,去除冗余):\n\n"
                + "\n".join(f"- {t}" for t in texts)
                + "\n\n合并后(只输出一句话):"
            )

            if dry_run:
                details.append({"action": "merge", "dry_run": True, "group": texts, "merged_into": "(dry_run)"})
                merged_count += 1
                continue

            try:
                merged_text = self.ds.simple(prompt, temperature=0.0, max_tokens=100).strip()
                # 保留组里最高 trust 作为新记忆的 trust
                max_trust = max(g.get("trust", 0.5) for g in group)
                max_uses = max(g.get("uses", 0) for g in group)
                # 删除旧的
                for g in group:
                    self.facts.delete(g["id"])
                # 写入合并版
                new_id = self.facts.add(merged_text)
                for f in self.facts.facts:
                    if f["id"] == new_id:
                        f["trust"] = max_trust
                        f["uses"] = max_uses
                        break
                self.facts._save()
                details.append({"action": "merge", "group": texts, "merged_into": merged_text})
                merged_count += 1
            except Exception as e:
                details.append({"action": "merge", "group": texts, "error": str(e)})

        return {"merged": merged_count, "details": details}

    def _reweight(self, dry_run: bool = False) -> dict:
        """多维重算重要性: recency + frequency + base_trust → 新 trust。

        公式: new_trust = 0.4*base_trust + 0.3*recency_score + 0.3*frequency_score
        - recency_score: 1.0(今天用过) ~ 0.0(30+天没用)
        - frequency_score: min(1.0, uses/10)
        """
        now = time.time()
        updated = 0
        details = []

        for f in self.facts.facts:
            if f.get("pinned"):
                continue  # pinned 不动

            base = f.get("trust", 0.5)
            uses = f.get("uses", 0)
            last_used = f.get("last_used", f.get("ts", now))

            # recency: 1.0 = 刚用, 0 = 30天没用
            days_ago = max(0, (now - last_used) / 86400)
            recency = max(0.0, 1.0 - days_ago / 30.0)

            # frequency: 0~1, 用10次就满分
            frequency = min(1.0, uses / 10.0)

            # 多维综合
            new_trust = 0.4 * base + 0.3 * recency + 0.3 * frequency
            new_trust = max(0.05, min(1.0, new_trust))

            if abs(new_trust - base) > 0.01:
                if not dry_run:
                    f["trust"] = round(new_trust, 4)
                updated += 1
                details.append({
                    "id": f["id"], "old_trust": round(base, 3),
                    "new_trust": round(new_trust, 3),
                    "recency": round(recency, 2), "frequency": round(frequency, 2),
                })

        if not dry_run and updated > 0:
            self.facts._save()

        return {"updated": updated, "details": details[:10]}  # 只返回前10条细节
