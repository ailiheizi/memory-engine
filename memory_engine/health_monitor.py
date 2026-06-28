"""记忆健康度监控: 检测记忆库退化(embedding空间熵、相似度区分力下降)。

问题: 记忆多了之后 cosine similarity 失去区分力(所有记忆都 0.6-0.7, 分不出谁更相关)。
这种退化是沉默发生的——用户感知到"记忆不好用了"但不知道为什么。

监控指标:
1. top-k 相似度标准差(低=区分力弱, 都长得差不多)
2. 平均 top-k 距离(高=记忆空间分散, 健康; 低=聚拢, 退化)
3. 矛盾密度(superseded 记忆占比)
4. trust 分布(大量 trust<0.1 = 很多垃圾没清理)

告警阈值:
- std < 0.03 → "记忆区分力下降, 考虑清理相似记忆"
- superseded 占比 > 30% → "过时记忆堆积, 建议清理"
- trust<0.1 占比 > 40% → "低信任垃圾过多"
"""

from __future__ import annotations

import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HealthReport:
    """记忆健康度报告。"""
    total_facts: int
    avg_trust: float
    low_trust_ratio: float        # trust < 0.1 的占比
    superseded_ratio: float       # 被标记 superseded 的占比
    similarity_std: float         # top-k 相似度标准差(越低区分力越弱)
    avg_top_distance: float       # 平均 top-k 内记忆间距离
    alerts: list[str]             # 告警信息
    healthy: bool                 # 综合健康状态


class MemoryHealthMonitor:
    """记忆健康度监控器。"""

    def __init__(self, fact_store):
        self.facts = fact_store

    def check(self, sample_queries: Optional[list[str]] = None) -> HealthReport:
        """执行健康检查。

        Args:
            sample_queries: 用于测试检索区分力的样本查询。
                           None = 用记忆库前10条的文本做自检。
        """
        import numpy as np

        facts = self.facts.facts
        total = len(facts)
        alerts = []

        if total == 0:
            return HealthReport(0, 0, 0, 0, 0, 0, ["记忆库为空"], False)

        # 1. Trust 分布
        trusts = [f.get("trust", 0.5) for f in facts]
        avg_trust = sum(trusts) / len(trusts)
        low_trust_count = sum(1 for t in trusts if t < 0.1)
        low_trust_ratio = low_trust_count / total

        # 2. Superseded 占比
        superseded_count = sum(1 for f in facts if f.get("superseded_by"))
        superseded_ratio = superseded_count / total

        # 3. 检索区分力(similarity std)
        similarity_std = 0.0
        avg_top_distance = 0.0

        if total >= 5:
            self.facts._ensure_model()
            if self.facts._index is None:
                self.facts._rebuild_index()

            # 用样本查询测试,默认用前10条事实的文本做query
            if sample_queries is None:
                sample_queries = [f["text"][:50] for f in facts[:min(10, total)]]

            all_stds = []
            all_distances = []

            for q in sample_queries:
                q_emb = self.facts._model.encode([q], normalize_embeddings=True)
                k = min(10, total)
                scores, _ = self.facts._index.search(np.array(q_emb, dtype=np.float32), k)
                top_scores = scores[0][:k]
                if len(top_scores) >= 3:
                    all_stds.append(float(np.std(top_scores)))
                    # top-k 内最高和最低的差距
                    all_distances.append(float(top_scores[0] - top_scores[-1]))

            if all_stds:
                similarity_std = sum(all_stds) / len(all_stds)
                avg_top_distance = sum(all_distances) / len(all_distances)

        # 4. 生成告警
        if similarity_std < 0.03 and total >= 20:
            alerts.append(f"区分力弱: top-k相似度std={similarity_std:.4f}(<0.03), 记忆太相似难以区分")

        if superseded_ratio > 0.30:
            alerts.append(f"过时堆积: {superseded_count}/{total}({superseded_ratio:.0%})条已被supersede, 建议清理")

        if low_trust_ratio > 0.40:
            alerts.append(f"低信任过多: {low_trust_count}/{total}({low_trust_ratio:.0%})条trust<0.1, 建议清理")

        if avg_trust < 0.2:
            alerts.append(f"整体信任低: 平均trust={avg_trust:.2f}, 记忆质量可能退化")

        healthy = len(alerts) == 0

        return HealthReport(
            total_facts=total,
            avg_trust=avg_trust,
            low_trust_ratio=low_trust_ratio,
            superseded_ratio=superseded_ratio,
            similarity_std=similarity_std,
            avg_top_distance=avg_top_distance,
            alerts=alerts,
            healthy=healthy,
        )

    def suggest_cleanup(self) -> list[dict]:
        """建议清理的记忆(低trust + superseded)。"""
        candidates = []
        for f in self.facts.facts:
            reasons = []
            if f.get("superseded_by"):
                reasons.append("superseded")
            if f.get("trust", 0.5) < 0.1:
                reasons.append("low_trust")
            if reasons:
                candidates.append({"id": f["id"], "text": f["text"][:50], "trust": f.get("trust", 0.5), "reasons": reasons})
        return candidates
