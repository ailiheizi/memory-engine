"""核心引擎: 组装 retrieve -> disclose -> persona demo -> DeepSeek 流水线。

这是对外的主接口。可通过 HTTP service (见 service.py) 调用它,
或直接 import 当 Python 库用。
"""

from __future__ import annotations

import logging
from typing import Optional

from .fact_store import FactStore
from .persona_manager import PersonaManager
from .deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)


class MemoryEngine:
    """记忆+性格引擎。

    一次对话流程:
        1. RAG 召回 top-k 事实
        2. 披露式展开进 context
        3. (方案A) 当前性格小模型生成风格示范
        4. 组装 prompt = [性格示范] + [披露事实] + [用户消息]
        5. DeepSeek 回答
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
        self.enable_persona = enable_persona
        self.persona = PersonaManager(base_model=base_model, personas_dir=f"{store_dir}/personas") if enable_persona else None

    # ---- 事实管理 (透传 FactStore) ----

    def add_fact(self, text: str, pinned: bool = False) -> int:
        return self.facts.add(text, pinned=pinned)

    def delete_fact(self, fact_id: int) -> bool:
        return self.facts.delete(fact_id)

    def update_fact(self, fact_id: int, text: Optional[str] = None, pinned: Optional[bool] = None) -> bool:
        return self.facts.update(fact_id, text=text, pinned=pinned)

    def list_facts(self) -> list[dict]:
        return self.facts.list_facts()

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

    def chat(self, user_msg: str, top_k: int = 5, temperature: float = 0.7, max_tokens: int = 512) -> dict:
        """完整流水线。返回 {response, used_memory, used_style, latency_ms}。"""
        # ① + ② RAG 召回 + 披露
        disclosure = self.facts.build_disclosure(user_msg, top_k=top_k)

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
        return {
            "response": result["content"],
            "used_memory": disclosure,
            "used_style": style_demo,
            "active_persona": self.persona._active_persona if self.persona else None,
            "latency_ms": result["latency_ms"],
        }
