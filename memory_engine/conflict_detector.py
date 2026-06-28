"""矛盾检测: 写入新记忆时检测是否与已有记忆矛盾。

机制(不引入新模型依赖,用已有的 BGE 做语义近似 + DeepSeek 做 NLI 判断):
1. 新记忆写入时, BGE 检索 top-k 语义最近的已有记忆
2. 对每个高相似候选, 用 DeepSeek 判断: contradiction / update / compatible
3. 根据判断:
   - contradiction(硬矛盾): 旧记忆 trust 归零, 标记 superseded_by
   - update(软更新): 旧记忆 trust 降低, 新记忆标记 updates
   - compatible(共存): 不动

优点:
- 不引入额外模型(DeBERTa-NLI等), 复用已有的 BGE + DeepSeek
- 和 trust 机制天然联动(矛盾=旧的trust降)
- 可选: DeepSeek 不可用时降级为"高相似度自动标记待审"
"""

from __future__ import annotations

import logging
from typing import Optional

from .deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)

# 语义相似度阈值: 超过此值才做矛盾检测(避免不相关的记忆被检查)
SIMILARITY_THRESHOLD = 0.7

NLI_PROMPT_TEMPLATE = """判断两条记忆的关系。

旧记忆: {old}
新记忆: {new}

它们的关系是:
- CONTRADICTION: 互斥矛盾(不可能同时为真, 如"住在北京" vs "住在上海")
- UPDATE: 信息更新(新的替代旧的, 如"开特斯拉" vs "换了比亚迪")
- COMPATIBLE: 可以共存(不矛盾, 如"会Python" 和 "也会Go")

只回复一个词: CONTRADICTION 或 UPDATE 或 COMPATIBLE"""


class ConflictDetector:
    """写入时矛盾检测器。"""

    def __init__(self, fact_store, deepseek: Optional[DeepSeekClient] = None):
        """
        Args:
            fact_store: FactStore 实例(用其向量索引做语义近似检索)
            deepseek: DeepSeekClient(做 NLI 判断; None=降级模式, 只标记不判断)
        """
        self.facts = fact_store
        self.ds = deepseek

    def check(self, new_text: str, top_k: int = 3) -> list[dict]:
        """检测新记忆是否与已有记忆矛盾。

        Returns:
            List of conflicts: [{"fact_id", "old_text", "relation", "score"}]
            空列表 = 无矛盾
        """
        if not self.facts.facts:
            return []

        # 1. BGE 检索语义最近的已有记忆
        candidates = self.facts.retrieve(new_text, top_k=top_k, reinforce=False)
        conflicts = []

        for cand in candidates:
            sim = cand.get("score", 0)
            if sim < SIMILARITY_THRESHOLD:
                continue

            # 2. 用 DeepSeek 做 NLI 判断
            if self.ds:
                relation = self._judge_relation(cand["text"], new_text)
            else:
                # 降级: 高相似度标记为 "needs_review"
                relation = "NEEDS_REVIEW"

            if relation in ("CONTRADICTION", "UPDATE", "NEEDS_REVIEW"):
                conflicts.append({
                    "fact_id": cand["id"],
                    "old_text": cand["text"],
                    "relation": relation,
                    "similarity": sim,
                })

        return conflicts

    def _judge_relation(self, old_text: str, new_text: str) -> str:
        """用 DeepSeek 判断两条记忆的关系。"""
        prompt = NLI_PROMPT_TEMPLATE.format(old=old_text, new=new_text)
        try:
            resp = self.ds.simple(prompt, temperature=0.0, max_tokens=10).strip().upper()
            for label in ("CONTRADICTION", "UPDATE", "COMPATIBLE"):
                if label in resp:
                    return label
            return "COMPATIBLE"
        except Exception as e:
            logger.warning(f"NLI judge failed: {e}")
            return "NEEDS_REVIEW"

    def add_with_conflict_check(self, new_text: str, pinned: bool = False,
                                 auto_resolve: bool = True) -> dict:
        """写入新记忆, 自动检测并处理矛盾。

        梯度化降权(按相似度分级):
            sim >= 0.8 + CONTRADICTION: trust 归零(确定性高的硬矛盾)
            sim >= 0.8 + UPDATE: trust × 0.2(确定性高的更新)
            sim 0.7-0.8 + CONTRADICTION: trust × 0.3(可能矛盾)
            sim 0.7-0.8 + UPDATE: trust × 0.5(可能更新)

        Args:
            auto_resolve: True=自动降低矛盾旧记忆的trust; False=只报告不处理

        Returns:
            {"id": new_fact_id, "conflicts": [...], "resolved": bool}
        """
        conflicts = self.check(new_text)

        # 先写入新记忆
        new_id = self.facts.add(new_text, pinned=pinned)

        # 梯度化处理矛盾
        if conflicts and auto_resolve:
            for c in conflicts:
                sim = c.get("similarity", 0.7)
                relation = c["relation"]

                # 按 sim × relation 梯度降权
                if relation == "CONTRADICTION":
                    factor = 0.0 if sim >= 0.8 else 0.3
                elif relation == "UPDATE":
                    factor = 0.2 if sim >= 0.8 else 0.5
                else:
                    continue  # NEEDS_REVIEW 不自动处理

                for f in self.facts.facts:
                    if f["id"] == c["fact_id"]:
                        f["trust"] = max(0.05, f["trust"] * factor)
                        f["superseded_by"] = new_id
                        break
            self.facts._save()

        return {
            "id": new_id,
            "conflicts": conflicts,
            "resolved": auto_resolve and len(conflicts) > 0,
        }
