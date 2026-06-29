"""语义聚类索引 — Cluster-First-Search。

大规模记忆(500+)时,全量FAISS扫描区分力下降。
两级检索: 先粗筛定位相关cluster → 再在cluster内精搜top-k。

机制:
1. 离线: 对所有记忆embedding做k-means聚类, 每个cluster有centroid
2. 检索时: query先和所有centroid比较, 选最近的N个cluster
3. 只在选中的cluster内做精确top-k搜索
4. 聚类在 consolidate 时重建(记忆变动后)

优点:
- 检索速度: O(k) centroid比较 + O(cluster_size) 精搜, 而非O(全量)
- 区分力: cluster内的记忆更相似, top-k差距更明显
- 可解释: 每个cluster是一个"主题"(工作/生活/技术/...)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_N_CLUSTERS = 10
MIN_FACTS_FOR_CLUSTERING = 30  # 少于这个数不做聚类(全量就够快)


class ClusterIndex:
    """语义聚类索引: 两级检索(cluster定位 → cluster内精搜)。"""

    def __init__(self, fact_store, n_clusters: int = DEFAULT_N_CLUSTERS):
        self.facts = fact_store
        self.n_clusters = n_clusters
        self._centroids = None       # (n_clusters, dim)
        self._labels = None          # 每条记忆属于哪个cluster
        self._cluster_indices = None # {cluster_id: [fact_indices]}
        self._built = False

    def build(self) -> dict:
        """构建聚类索引(离线调用, 如在 consolidate 时)。"""
        facts = self.facts.facts
        if len(facts) < MIN_FACTS_FOR_CLUSTERING:
            self._built = False
            return {"status": "skipped", "reason": f"only {len(facts)} facts (need {MIN_FACTS_FOR_CLUSTERING})"}

        self.facts._ensure_model()
        texts = [f["text"] for f in facts]
        embeddings = self.facts._model.encode(texts, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # k-means 聚类
        k = min(self.n_clusters, len(facts) // 3)  # 保证每cluster至少3条
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=50)
        labels = km.fit_predict(embeddings)
        centroids = km.cluster_centers_
        # normalize centroids
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / (norms + 1e-8)

        self._centroids = centroids
        self._labels = labels
        self._cluster_indices = {}
        for i, label in enumerate(labels):
            self._cluster_indices.setdefault(int(label), []).append(i)
        self._built = True

        # 统计
        sizes = [len(v) for v in self._cluster_indices.values()]
        return {
            "status": "built",
            "n_clusters": k,
            "cluster_sizes": sizes,
            "avg_size": sum(sizes) / len(sizes),
        }

    def search(self, query: str, top_k: int = 5, n_probe_clusters: int = 3) -> list[dict]:
        """两级检索: 先找相关cluster, 再在cluster内精搜。

        如果没建过聚类索引(或记忆太少), fallback 到普通全量检索。
        """
        if not self._built:
            # fallback: 直接用 fact_store 的全量检索
            return self.facts.retrieve(query, top_k=top_k)

        self.facts._ensure_model()
        q_emb = self.facts._model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)

        # Level 1: 找最近的 n_probe_clusters 个 cluster
        centroid_sims = (q_emb @ self._centroids.T)[0]
        top_clusters = np.argsort(centroid_sims)[::-1][:n_probe_clusters]

        # Level 2: 在选中的 cluster 内精搜
        candidate_indices = []
        for cid in top_clusters:
            candidate_indices.extend(self._cluster_indices.get(int(cid), []))

        if not candidate_indices:
            return self.facts.retrieve(query, top_k=top_k)

        # 对候选做精确相似度计算
        facts = self.facts.facts
        candidate_texts = [facts[i]["text"] for i in candidate_indices]
        cand_embs = self.facts._model.encode(candidate_texts, normalize_embeddings=True)
        sims = (q_emb @ np.array(cand_embs, dtype=np.float32).T)[0]

        # 排序取 top-k
        ranked_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        import time
        now = time.time()
        for ri in ranked_idx:
            fact_idx = candidate_indices[ri]
            f = facts[fact_idx]
            eff = self.facts._effective_trust(f, now)
            if f.get("superseded_by"):
                eff *= 0.1
            final = float(sims[ri]) + 0.1 * eff
            results.append({**f, "score": float(sims[ri]), "eff_trust": eff, "final": final,
                           "cluster": int(self._labels[fact_idx])})

        # superseded 排后面
        results.sort(key=lambda r: (0 if r.get("superseded_by") else 1, r.get("final", 0)), reverse=True)
        return results

    def get_cluster_summary(self) -> list[dict]:
        """返回每个 cluster 的摘要(便于理解记忆的主题分布)。"""
        if not self._built:
            return []
        facts = self.facts.facts
        summary = []
        for cid, indices in self._cluster_indices.items():
            examples = [facts[i]["text"][:40] for i in indices[:3]]
            summary.append({"cluster": cid, "size": len(indices), "examples": examples})
        summary.sort(key=lambda x: x["size"], reverse=True)
        return summary
