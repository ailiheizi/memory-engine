"""Memory Worth — Beta-Bernoulli 信任评分。

替换简单的加性 trust(+0.15/-0.03) 为有统计基础的 Beta 分布评分。

核心公式:
    worth = (success + 1) / (success + failure + 2)   # Laplace 平滑 Beta 均值

优点:
- 10次成功0失败 (worth=0.92) vs 1次成功0失败 (worth=0.67) — 能区分置信度
- 自动收敛: 用得越多越稳定, 新记忆天然不确定
- 数学上有界 [0,1], 无需人工设上下限
- 和现有 usage_feedback 的 adopted/ignored 信号天然对接

用法:
    worth.record_success(fact_id)   # 被 LLM 采纳
    worth.record_failure(fact_id)   # 被 LLM 忽略
    worth.get_worth(fact_id)        # 获取当前 worth 值
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


class MemoryWorth:
    """Beta-Bernoulli 信任评分器。"""

    def __init__(self, fact_store):
        self.facts = fact_store

    def _get_fact(self, fact_id: int) -> dict | None:
        for f in self.facts.facts:
            if f["id"] == fact_id:
                return f
        return None

    def get_worth(self, fact_id: int) -> float:
        """获取记忆的 worth 值 (Beta posterior mean)。"""
        f = self._get_fact(fact_id)
        if not f:
            return 0.5
        s = f.get("success", 0)
        fail = f.get("failure", 0)
        return (s + 1) / (s + fail + 2)

    def get_confidence(self, fact_id: int) -> float:
        """获取置信度(观察次数越多越高)。范围 [0, 1)。"""
        f = self._get_fact(fact_id)
        if not f:
            return 0.0
        total = f.get("success", 0) + f.get("failure", 0)
        # 用 log 映射: 10次观察≈0.7, 50次≈0.9, 100次≈0.95
        return 1.0 - 1.0 / (1.0 + total / 5.0)

    def record_success(self, fact_id: int) -> float:
        """记录一次成功(被采纳)。返回新 worth。"""
        f = self._get_fact(fact_id)
        if not f:
            return 0.5
        f.setdefault("success", 0)
        f["success"] += 1
        # 同步更新 trust 为 worth(让检索加权能用)
        f["trust"] = self.get_worth(fact_id)
        self.facts._save()
        return f["trust"]

    def record_failure(self, fact_id: int) -> float:
        """记录一次失败(被忽略/否定)。返回新 worth。"""
        f = self._get_fact(fact_id)
        if not f:
            return 0.5
        f.setdefault("failure", 0)
        f["failure"] += 1
        f["trust"] = self.get_worth(fact_id)
        self.facts._save()
        return f["trust"]

    def get_stats(self, fact_id: int) -> dict:
        """获取完整统计。"""
        f = self._get_fact(fact_id)
        if not f:
            return {}
        s = f.get("success", 0)
        fail = f.get("failure", 0)
        return {
            "fact_id": fact_id,
            "success": s,
            "failure": fail,
            "worth": self.get_worth(fact_id),
            "confidence": self.get_confidence(fact_id),
            "total_observations": s + fail,
        }

    def rank_by_worth(self) -> list[dict]:
        """按 worth 排序所有记忆(最有价值的排前面)。"""
        ranked = []
        for f in self.facts.facts:
            s = f.get("success", 0)
            fail = f.get("failure", 0)
            worth = (s + 1) / (s + fail + 2)
            ranked.append({"id": f["id"], "text": f["text"][:50], "worth": worth,
                          "confidence": self.get_confidence(f["id"]), "success": s, "failure": fail})
        ranked.sort(key=lambda x: x["worth"], reverse=True)
        return ranked

    def eviction_candidates(self, threshold: float = 0.3, min_observations: int = 3) -> list[dict]:
        """找出该淘汰的低价值记忆(worth低 + 观察够多 = 确认是垃圾)。"""
        candidates = []
        for f in self.facts.facts:
            if f.get("pinned"):
                continue
            s = f.get("success", 0)
            fail = f.get("failure", 0)
            total = s + fail
            worth = (s + 1) / (s + fail + 2)
            if worth < threshold and total >= min_observations:
                candidates.append({"id": f["id"], "text": f["text"][:50], "worth": worth, "observations": total})
        return candidates
