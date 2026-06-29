"""记忆链接图 — Memory Graph。

借鉴 "Memory is Reconstructed, Not Retrieved" (ICML 2026):
记忆不是孤立片段,而是通过关联边连接的图。检索时不只找最相似的单条,
还沿边扩展相关记忆,实现:
- 间接矛盾检测(A→B→C 链式发现)
- 多跳推理(组合多条记忆回答)
- 时序追踪(事件的先后因果)

图结构:
- Node = 每条记忆 (fact_id)
- Edge = 关联关系, 类型:
    - "related"  — 主题相关(同一实体/话题)
    - "updates"  — B 更新了 A(时序)
    - "causes"   — A 导致了 B(因果)
    - "contradicts" — A 和 B 矛盾

边的构建:
- 写入时自动: 新记忆和高相似已有记忆建 "related" 边
- 矛盾检测时: 建 "contradicts" / "updates" 边
- 手动: 用户可显式添加关联

检索增强:
- 基础检索找到 seed nodes
- 沿边扩展 1-2 跳相关节点
- 合并后排序返回(seed + neighbors)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 边类型
EDGE_TYPES = ("related", "updates", "contradicts", "causes")
# 扩展时最大跳数
MAX_HOPS = 2
# 相似度超过此值自动建 related 边
AUTO_LINK_THRESHOLD = 0.7


class MemoryGraph:
    """记忆关联图: 节点=记忆, 边=关系。"""

    def __init__(self, fact_store, store_dir: Optional[str] = None):
        self.facts = fact_store
        self.store_dir = Path(store_dir) if store_dir else Path(fact_store.store_dir)
        self.graph_file = self.store_dir / "memory_graph.json"
        # edges: [{"from": id, "to": id, "type": str, "meta": {}}]
        self.edges: list[dict] = []
        self._adjacency: dict[int, list[dict]] = {}  # fact_id -> [{"to", "type", "meta"}]
        self._load()

    def _load(self):
        if self.graph_file.exists():
            data = json.loads(self.graph_file.read_text(encoding="utf-8"))
            self.edges = data.get("edges", [])
            self._rebuild_adjacency()

    def _save(self):
        self.graph_file.write_text(
            json.dumps({"edges": self.edges}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _rebuild_adjacency(self):
        self._adjacency = {}
        for e in self.edges:
            self._adjacency.setdefault(e["from"], []).append({"to": e["to"], "type": e["type"], "meta": e.get("meta", {})})
            # 双向(无向图语义:related双向, updates/contradicts单向也存反向便于遍历)
            self._adjacency.setdefault(e["to"], []).append({"to": e["from"], "type": e["type"], "meta": e.get("meta", {})})

    # ---- 边管理 ----

    def add_edge(self, from_id: int, to_id: int, edge_type: str, meta: Optional[dict] = None) -> bool:
        """添加一条边。"""
        if edge_type not in EDGE_TYPES:
            logger.warning(f"Unknown edge type: {edge_type}")
            return False
        # 去重
        for e in self.edges:
            if e["from"] == from_id and e["to"] == to_id and e["type"] == edge_type:
                return False  # 已存在
        edge = {"from": from_id, "to": to_id, "type": edge_type, "meta": meta or {}}
        self.edges.append(edge)
        self._adjacency.setdefault(from_id, []).append({"to": to_id, "type": edge_type, "meta": meta or {}})
        self._adjacency.setdefault(to_id, []).append({"to": from_id, "type": edge_type, "meta": meta or {}})
        self._save()
        return True

    def remove_edges_for(self, fact_id: int):
        """删除某条记忆的所有边(记忆被删时调用)。"""
        self.edges = [e for e in self.edges if e["from"] != fact_id and e["to"] != fact_id]
        self._rebuild_adjacency()
        self._save()

    def get_neighbors(self, fact_id: int, edge_type: Optional[str] = None) -> list[dict]:
        """获取直接邻居。"""
        neighbors = self._adjacency.get(fact_id, [])
        if edge_type:
            neighbors = [n for n in neighbors if n["type"] == edge_type]
        return neighbors

    # ---- 自动建边(写入时) ----

    def auto_link_on_add(self, new_fact_id: int, similar_facts: list[dict], deepseek=None):
        """新记忆写入后, 自动发现并建立关系网络。

        两层策略:
        1. 高相似(>=0.7) + 有 DeepSeek: 用 LLM 判断具体关系类型
        2. 高相似(>=0.7) + 无 DeepSeek: 退化为全部建 "related" 边
        """
        new_fact = None
        for f in self.facts.facts:
            if f["id"] == new_fact_id:
                new_fact = f
                break
        if not new_fact:
            return

        for f in similar_facts:
            if f["id"] == new_fact_id:
                continue
            sim = f.get("score", 0)
            if sim < AUTO_LINK_THRESHOLD:
                continue

            if deepseek:
                # 用 LLM 判断关系类型
                edge_type = self._infer_relation(deepseek, new_fact["text"], f["text"])
            else:
                edge_type = "related"

            if edge_type and edge_type != "none":
                self.add_edge(new_fact_id, f["id"], edge_type,
                             meta={"auto": True, "sim": round(sim, 3)})

    def _infer_relation(self, deepseek, text_a: str, text_b: str) -> str:
        """用 DeepSeek 推断两条记忆的关系类型。"""
        prompt = (
            "判断这两条记忆之间的关系:\n"
            f"A: {text_a}\nB: {text_b}\n\n"
            "关系类型(只选一个):\n"
            "- related: 同一主题/实体相关(如同一个人的不同信息)\n"
            "- updates: A是B的更新版本(时序先后)\n"
            "- causes: A导致了B(因果)\n"
            "- contradicts: A和B矛盾\n"
            "- none: 无关\n\n"
            "只回复一个词:"
        )
        try:
            resp = deepseek.simple(prompt, temperature=0.0, max_tokens=10).strip().lower()
            for t in EDGE_TYPES:
                if t in resp:
                    return t
            if "none" in resp:
                return "none"
            return "related"  # 默认
        except:
            return "related"

    def auto_build_network(self, deepseek=None, sim_threshold: float = 0.65, max_edges_per_fact: int = 3):
        """对全部记忆一次性构建关系网络(离线/consolidate时调用)。

        扫描所有记忆对, 相似度超阈值的用 LLM 判断关系并建边。
        """
        import numpy as np
        self.facts._ensure_model()
        facts = self.facts.facts
        if len(facts) < 2:
            return {"edges_added": 0}

        texts = [f["text"] for f in facts]
        embs = self.facts._model.encode(texts, normalize_embeddings=True)
        embs = np.array(embs, dtype=np.float32)

        # 计算相似度矩阵
        sim_matrix = embs @ embs.T

        edges_added = 0
        for i in range(len(facts)):
            # 找每条记忆最相似的 top-N 邻居
            sims = sim_matrix[i]
            top_indices = np.argsort(sims)[::-1][1:max_edges_per_fact + 1]  # 排除自己

            for j in top_indices:
                if sims[j] < sim_threshold:
                    continue
                # 检查边是否已存在
                existing = any(
                    e["from"] == facts[i]["id"] and e["to"] == facts[j]["id"]
                    or e["from"] == facts[j]["id"] and e["to"] == facts[i]["id"]
                    for e in self.edges
                )
                if existing:
                    continue

                if deepseek:
                    edge_type = self._infer_relation(deepseek, facts[i]["text"], facts[j]["text"])
                else:
                    edge_type = "related"

                if edge_type and edge_type != "none":
                    self.add_edge(facts[i]["id"], facts[j]["id"], edge_type,
                                 meta={"auto": True, "sim": round(float(sims[j]), 3)})
                    edges_added += 1

        return {"edges_added": edges_added, "total_edges": len(self.edges), "total_nodes": len(facts)}

    def link_conflict(self, old_id: int, new_id: int, relation: str):
        """矛盾检测后建边。"""
        if relation == "CONTRADICTION":
            self.add_edge(new_id, old_id, "contradicts")
        elif relation == "UPDATE":
            self.add_edge(new_id, old_id, "updates")

    # ---- 图增强检索 ----

    def expand_retrieval(self, seed_ids: list[int], max_hops: int = MAX_HOPS,
                         max_expand: int = 10) -> list[int]:
        """从 seed 节点出发,沿边扩展获取相关记忆 id。

        返回 seed + 扩展的所有 fact_id (去重, 含 seed)。
        """
        visited = set(seed_ids)
        frontier = list(seed_ids)

        # 收集被 superseded 的 id, 图扩展时跳过(避免把过时记忆捞回来)
        superseded = {f["id"] for f in self.facts.facts if f.get("superseded_by")}

        for hop in range(max_hops):
            next_frontier = []
            for node in frontier:
                for neighbor in self._adjacency.get(node, []):
                    nid = neighbor["to"]
                    if nid in superseded:
                        continue  # 跳过被矛盾检测降权的旧记忆
                    if nid not in visited and len(visited) < len(seed_ids) + max_expand:
                        visited.add(nid)
                        next_frontier.append(nid)
            frontier = next_frontier
            if not frontier:
                break

        return list(visited)

    def graph_enhanced_retrieve(self, query: str, top_k: int = 5, expand_hops: int = 1) -> list[dict]:
        """图增强检索: 语义检索 seed → 图扩展 → 合并排序。

        比普通检索多了"沿关联边扩展"的步骤,能找到间接相关的记忆。
        """
        # Step 1: 普通语义检索得到 seed
        seeds = self.facts.retrieve(query, top_k=top_k, reinforce=False)
        seed_ids = [s["id"] for s in seeds]

        if not seed_ids or expand_hops == 0:
            return seeds

        # Step 2: 图扩展
        expanded_ids = self.expand_retrieval(seed_ids, max_hops=expand_hops)
        extra_ids = [eid for eid in expanded_ids if eid not in seed_ids]

        # Step 3: 获取扩展出的记忆详情
        extra_facts = []
        for f in self.facts.facts:
            if f["id"] in extra_ids:
                extra_facts.append({**f, "score": 0.0, "eff_trust": f.get("trust", 0.5),
                                   "final": 0.05, "via_graph": True})

        # Step 4: 合并 seed + extra, seed 排前面
        all_results = seeds + extra_facts
        return all_results[:top_k + len(extra_facts)]  # 允许超出 top_k(图扩展的是bonus)

    # ---- 可视化/调试 ----

    def summary(self) -> dict:
        """图的基本统计。"""
        type_counts = {}
        for e in self.edges:
            type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
        nodes_with_edges = len(self._adjacency)
        return {
            "total_edges": len(self.edges),
            "nodes_with_edges": nodes_with_edges,
            "total_nodes": len(self.facts.facts),
            "orphan_nodes": len(self.facts.facts) - nodes_with_edges,
            "edge_types": type_counts,
        }
