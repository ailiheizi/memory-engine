"""核心引擎: 组装 retrieve -> disclose -> persona demo -> DeepSeek 流水线。

集成模块:
- FactStore: RAG + 信任加权
- ConflictDetector: 写入时矛盾检测
- UsageFeedback: 被采纳才强化(精细化 trust)
- HealthMonitor: 记忆健康度监控
- PersonaManager: 多adapter性格
- DeepSeekClient: LLM API 桥接

这是对外的主接口。可通过 HTTP service (见 service.py) 调用它,
或直接 import 当 Python 库用。
"""

from __future__ import annotations

import logging
from typing import Optional

from .fact_store import FactStore
from .conflict_detector import ConflictDetector
from .usage_feedback import UsageFeedback
from .health_monitor import MemoryHealthMonitor
from .consolidator import SleepConsolidator
from .persona_manager import PersonaManager
from .deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)


class MemoryEngine:
    """记忆+性格引擎。

    一次对话流程:
        1. RAG 召回 top-k 事实 (信任加权)
        2. 披露式展开进 context
        3. (方案A) 当前性格小模型生成风格示范
        4. 组装 prompt = [性格示范] + [披露事实] + [用户消息]
        5. DeepSeek 回答
        6. 使用反馈: 对比回答和检索记忆, 被采纳的才强化
    """

    def __init__(
        self,
        store_dir: str = "./memory_data",
        base_model: str = "Qwen/Qwen3-0.6B",
        embed_model: str = "BAAI/bge-m3",
        deepseek_key: Optional[str] = None,
        enable_persona: bool = True,
    ):
        self.facts = FactStore(f"{store_dir}/facts", embed_model=embed_model)
        self.llm = DeepSeekClient(api_key=deepseek_key)
        self.conflict = ConflictDetector(self.facts, deepseek=self.llm)
        self.feedback = UsageFeedback(self.facts)
        self.health = MemoryHealthMonitor(self.facts)
        self.consolidator = SleepConsolidator(self.facts, deepseek=self.llm)
        self.enable_persona = enable_persona
        self.persona = PersonaManager(base_model=base_model, personas_dir=f"{store_dir}/personas") if enable_persona else None

    # ---- 事实管理 (带矛盾检测) ----

    def add_fact(self, text: str, pinned: bool = False, check_conflict: bool = True) -> dict:
        """写入事实, 自动检测矛盾并处理。

        Returns:
            {"id": int, "conflicts": [...], "resolved": bool}
        """
        if check_conflict:
            return self.conflict.add_with_conflict_check(text, pinned=pinned, auto_resolve=True)
        fid = self.facts.add(text, pinned=pinned)
        return {"id": fid, "conflicts": [], "resolved": False}

    def delete_fact(self, fact_id: int) -> bool:
        return self.facts.delete(fact_id)

    def update_fact(self, fact_id: int, text: Optional[str] = None, pinned: Optional[bool] = None) -> bool:
        return self.facts.update(fact_id, text=text, pinned=pinned)

    def list_facts(self) -> list[dict]:
        return self.facts.list_facts()

    def reinforce_fact(self, fact_id: int) -> bool:
        """手动强化一条记忆(信任上升)。"""
        return self.facts.reinforce(fact_id)

    def check_health(self) -> dict:
        """检查记忆健康度。返回 {healthy, alerts, ...}。"""
        report = self.health.check()
        return {
            "healthy": report.healthy,
            "total_facts": report.total_facts,
            "avg_trust": report.avg_trust,
            "low_trust_ratio": report.low_trust_ratio,
            "superseded_ratio": report.superseded_ratio,
            "similarity_std": report.similarity_std,
            "alerts": report.alerts,
        }

    def suggest_cleanup(self) -> list[dict]:
        """建议清理的低质量记忆。"""
        return self.health.suggest_cleanup()

    def consolidate(self, mode: str = "full", dry_run: bool = False) -> dict:
        """睡眠巩固: 手动触发离线记忆整理。

        mode:
            "cleanup"  — 删除 superseded + 极低trust 垃圾
            "evict"    — 超容量时淘汰最低trust记忆
            "merge"    — 合并高相似冗余记忆(需DeepSeek)
            "reweight" — 多维重算trust(recency+frequency+base)
            "full"     — 以上全做(完整一次"睡眠")

        dry_run: True=只报告不执行
        """
        return self.consolidator.consolidate(mode=mode, dry_run=dry_run)

    # ---- 性格管理 (透传 PersonaManager) ----

    def create_persona(self, persona_id: str, examples: list[dict], desc: str = "", epochs: int = 10):
        if not self.persona:
            raise RuntimeError("Persona disabled (enable_persona=False)")
        return self.persona.create_persona(persona_id, examples, desc=desc, epochs=epochs)

    def delete_persona(self, persona_id: str) -> bool:
        return self.persona.delete_persona(persona_id) if self.persona else False

    def switch_persona(self, persona_id: Optional[str]):
        if self.persona:
            self.persona.activate(persona_id)

    def list_personas(self) -> dict:
        return self.persona.list_personas() if self.persona else {}

    # ---- 核心: 带记忆+性格回答 ----

    def chat(self, user_msg: str, top_k: int = 5, temperature: float = 0.7, max_tokens: int = 512,
             min_trust: float = 0.0) -> dict:
        """完整流水线。返回 {response, used_memory, used_style, feedback, latency_ms}。

        使用反馈: 回答后对比语义, 被采纳的记忆才强化、被忽略的微降。
        """
        # ① + ② RAG 召回 + 信任加权 + 披露 (不在此处强化, 等反馈)
        retrieved = self.facts.retrieve(user_msg, top_k=top_k, min_trust=min_trust, reinforce=False)
        disclosure = ""
        if retrieved:
            lines = ["[USER MEMORY]"]
            for f in retrieved:
                tag = "[pinned] " if f.get("pinned") else ""
                lines.append(f"- {tag}{f['text']}")
            lines.append("[END USER MEMORY]")
            disclosure = "\n".join(lines)

        # ③ 方案A: 性格风格示范
        style_demo = ""
        if self.persona is not None:
            try:
                style_demo = self.persona.style_demo(user_msg)
            except Exception as e:
                logger.warning(f"style_demo failed: {e}")

        # ④ 组装 system prompt
        parts = ["You are the user's personal assistant."]
        if style_demo:
            parts.append(
                "Match the communication STYLE shown in this example (tone, brevity, attitude):\n"
                f"[STYLE EXAMPLE]\n{style_demo}\n[END STYLE EXAMPLE]"
            )
        if disclosure:
            parts.append(f"Use these remembered facts about the user:\n{disclosure}")
        parts.append("Answer the user naturally, using the facts, in the demonstrated style.")
        system_prompt = "\n\n".join(parts)

        # ⑤ DeepSeek 回答
        result = self.llm.chat(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
            temperature=temperature, max_tokens=max_tokens,
        )

        # ⑥ 使用反馈: 被采纳才强化
        fb = []
        if retrieved:
            fb = self.feedback.compute_feedback(result["content"], retrieved)
            self.feedback.apply_feedback(fb)

        return {
            "response": result["content"],
            "used_memory": disclosure,
            "used_style": style_demo,
            "active_persona": self.persona._active_persona if self.persona else None,
            "feedback": [{"id": f["fact_id"], "adopted": f["adopted"], "sim": round(f["similarity"], 2)} for f in fb],
            "latency_ms": result["latency_ms"],
        }
