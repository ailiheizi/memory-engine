"""人格记忆分区: 每个性格维护独立记忆分区, 核心事实跨人格共享。

问题: 多人格切换时"记忆泄漏" — 人格A的对话记忆影响人格B的回答。
比如: "正式助手"模式下记住了"用户不喜欢emoji", 切换到"活泼朋友"模式后
这条记忆不该影响行为(朋友模式就该用emoji)。

设计:
- shared 层: 所有人格共享的核心事实(名字/工作/地址等客观信息)
- persona 层: 每个人格独立的记忆分区(偏好/上下文/对话风格记忆)
- 检索时: shared + 当前persona 的记忆合并, 其他persona的记忆不可见
- 写入时: 默认写入当前 persona 分区; 可选标记 shared

基于 Hermes 的 namespace 概念(memory-runtime.memory / memory-runtime.user)扩展。
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from .fact_store import FactStore

logger = logging.getLogger(__name__)


class PartitionedMemory:
    """人格分区记忆: shared 共享层 + 每个 persona 独立层。"""

    def __init__(self, store_dir: str, embed_model: str = "BAAI/bge-m3",
                 decay_half_life_days: float = 30.0):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.embed_model = embed_model
        self.decay = decay_half_life_days

        # shared 层 (所有人格可见)
        self.shared = FactStore(str(self.store_dir / "_shared"), embed_model=embed_model,
                                decay_half_life_days=decay_half_life_days)
        # persona 分区 (按需加载)
        self._persona_stores: dict[str, FactStore] = {}
        self._active_persona: Optional[str] = None

    def _get_persona_store(self, persona_id: str) -> FactStore:
        """获取或创建 persona 的独立记忆分区。"""
        if persona_id not in self._persona_stores:
            path = str(self.store_dir / f"persona_{persona_id}")
            self._persona_stores[persona_id] = FactStore(
                path, embed_model=self.embed_model, decay_half_life_days=self.decay
            )
        return self._persona_stores[persona_id]

    def switch_persona(self, persona_id: Optional[str]):
        """切换当前人格。None = 只用 shared。"""
        self._active_persona = persona_id
        logger.info(f"Switched memory partition to: {persona_id or 'shared-only'}")

    # ---- 写入 ----

    def add(self, text: str, shared: bool = False, pinned: bool = False) -> dict:
        """写入记忆。

        shared=True: 写入共享层(所有人格可见)
        shared=False: 写入当前人格分区(仅该人格可见)
        """
        if shared or self._active_persona is None:
            fid = self.shared.add(text, pinned=pinned)
            return {"id": fid, "partition": "shared"}
        else:
            store = self._get_persona_store(self._active_persona)
            fid = store.add(text, pinned=pinned)
            return {"id": fid, "partition": self._active_persona}

    def delete(self, fact_id: int, partition: Optional[str] = None) -> bool:
        """删除记忆。partition=None 尝试当前persona再尝试shared。"""
        if partition == "shared":
            return self.shared.delete(fact_id)
        if partition:
            store = self._get_persona_store(partition)
            return store.delete(fact_id)
        # 自动查找
        if self._active_persona:
            store = self._get_persona_store(self._active_persona)
            if store.delete(fact_id):
                return True
        return self.shared.delete(fact_id)

    # ---- 检索(合并 shared + 当前 persona) ----

    def retrieve(self, query: str, top_k: int = 5, reinforce: bool = False) -> list[dict]:
        """检索: shared + 当前 persona 合并, 其他 persona 不可见。"""
        results: dict[str, dict] = {}

        # 1. shared 层
        shared_results = self.shared.retrieve(query, top_k=top_k, reinforce=reinforce)
        for r in shared_results:
            key = f"shared_{r['id']}"
            results[key] = {**r, "partition": "shared"}

        # 2. 当前 persona 层
        if self._active_persona:
            store = self._get_persona_store(self._active_persona)
            persona_results = store.retrieve(query, top_k=top_k, reinforce=reinforce)
            for r in persona_results:
                key = f"{self._active_persona}_{r['id']}"
                results[key] = {**r, "partition": self._active_persona}

        # 合并排序取 top_k
        ranked = sorted(results.values(), key=lambda r: (
            0 if r.get("superseded_by") else 1,
            r.get("final", 0),
        ), reverse=True)
        return ranked[:top_k]

    def build_disclosure(self, query: str, top_k: int = 5, reinforce: bool = False) -> str:
        """召回 + 披露(标注来源分区)。"""
        facts = self.retrieve(query, top_k=top_k, reinforce=reinforce)
        if not facts:
            return ""
        lines = ["[USER MEMORY]"]
        for f in facts:
            tag = f"[{f['partition']}] " if f.get("partition") != "shared" else ""
            pin = "[pinned] " if f.get("pinned") else ""
            lines.append(f"- {tag}{pin}{f['text']}")
        lines.append("[END USER MEMORY]")
        return "\n".join(lines)

    # ---- 管理 ----

    def list_partitions(self) -> dict:
        """列出所有分区及其记忆数量。"""
        info = {"shared": len(self.shared.facts)}
        for pid, store in self._persona_stores.items():
            info[pid] = len(store.facts)
        # 检查磁盘上有但未加载的
        for d in self.store_dir.iterdir():
            if d.is_dir() and d.name.startswith("persona_"):
                pid = d.name[len("persona_"):]
                if pid not in info:
                    info[pid] = "unloaded"
        return info

    def delete_partition(self, persona_id: str) -> bool:
        """删除整个人格分区(及其所有记忆)。"""
        path = self.store_dir / f"persona_{persona_id}"
        if path.exists():
            shutil.rmtree(path)
        if persona_id in self._persona_stores:
            del self._persona_stores[persona_id]
        if self._active_persona == persona_id:
            self._active_persona = None
        logger.info(f"Deleted partition '{persona_id}'")
        return True
