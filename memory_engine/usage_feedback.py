"""使用反馈循环: 只有被 DeepSeek 实际采纳的记忆才强化 trust。

现有问题: "召回即强化" 粒度太粗 — 检索到5条但 DeepSeek 可能只用了1条,
另外4条的 trust 也被错误提升(导致噪音记忆虚高)。

改进: DeepSeek 回答后, 对比答案和每条检索记忆的语义相似度:
- 高相似(>0.5) = 被采纳 -> trust 上升
- 低相似(<0.3) = 被忽略 -> trust 微降(可选)
- 中间 = 不动

这让 trust 分真正反映"哪条记忆对回答有贡献"。
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 采纳/忽略的相似度阈值
ADOPTION_THRESHOLD = 0.5
IGNORE_THRESHOLD = 0.3
IGNORE_PENALTY = 0.03  # 被忽略时 trust 微降(比 reinforce gain 小很多)


class UsageFeedback:
    """追踪记忆是否被大模型实际采纳, 精细化 trust 更新。"""

    def __init__(self, fact_store):
        self.facts = fact_store
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            self._model = self.facts._ensure_model()
        return self._model

    def compute_feedback(self, response: str, retrieved_facts: list[dict]) -> list[dict]:
        """对比 DeepSeek 回答和每条检索记忆, 判断采纳/忽略。

        Args:
            response: DeepSeek 的回答文本
            retrieved_facts: retrieve() 返回的事实列表

        Returns:
            [{"fact_id", "text", "similarity", "adopted"}]
        """
        if not retrieved_facts or not response.strip():
            return []

        import numpy as np
        model = self._ensure_model()

        # 编码回答
        resp_emb = model.encode([response], normalize_embeddings=True)
        # 编码每条记忆
        fact_texts = [f["text"] for f in retrieved_facts]
        fact_embs = model.encode(fact_texts, normalize_embeddings=True)

        # 计算相似度
        sims = (resp_emb @ fact_embs.T)[0]

        results = []
        for i, f in enumerate(retrieved_facts):
            sim = float(sims[i])
            adopted = sim >= ADOPTION_THRESHOLD
            ignored = sim < IGNORE_THRESHOLD
            results.append({
                "fact_id": f["id"],
                "text": f["text"],
                "similarity": sim,
                "adopted": adopted,
                "ignored": ignored,
            })
        return results

    def apply_feedback(self, feedback: list[dict], reinforce_gain: float = 0.15):
        """根据反馈更新 trust: 采纳的强化, 被忽略的微降。"""
        for fb in feedback:
            if fb["adopted"]:
                self.facts.reinforce(fb["fact_id"], gain=reinforce_gain)
                logger.debug(f"Reinforced #{fb['fact_id']} (sim={fb['similarity']:.2f})")
            elif fb["ignored"]:
                # 微降: 被检索但完全没被用到
                for f in self.facts.facts:
                    if f["id"] == fb["fact_id"]:
                        f["trust"] = max(0.05, f["trust"] - IGNORE_PENALTY)
                        break
                logger.debug(f"Penalized #{fb['fact_id']} (sim={fb['similarity']:.2f})")
        self.facts._save()
