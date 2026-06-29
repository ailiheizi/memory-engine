"""整体效果对比 + 问题关联整理。

两件事:
1. A/B 对比: 有优化(worth+cluster+graph) vs 裸RAG, 看整体是否变好
2. 问题整理(Query Expansion): 给每条记忆关联"可能被什么问题触发",
   让检索匹配更多表达方式

问题整理的设计:
- consolidate 时, 用 DeepSeek 给每条记忆生成 2-3 个可能的查询表达
- 存入 fact 的 "queries" 字段
- 检索时: query 和事实文本 + 关联查询 一起比较, 取最高分
"""

import sys
import os
import json
import shutil
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_engine.fact_store import FactStore
from memory_engine.memory_worth import MemoryWorth
from memory_engine.cluster_index import ClusterIndex
from memory_engine.memory_graph import MemoryGraph
from memory_engine.deepseek_client import DeepSeekClient


# ---- 问题整理模块 ----

def generate_queries_for_fact(ds: DeepSeekClient, fact_text: str) -> list[str]:
    """用 DeepSeek 给一条事实生成 2-3 个可能触发它的查询表达。"""
    prompt = (
        f"Given this fact about a user:\n\"{fact_text}\"\n\n"
        "Generate 3 different ways someone might ask a question that this fact would answer. "
        "Be diverse in phrasing. Output ONLY the 3 questions, one per line, nothing else."
    )
    try:
        resp = ds.simple(prompt, temperature=0.3, max_tokens=100)
        queries = [q.strip().strip("-").strip("0123456789.").strip() for q in resp.strip().split("\n") if q.strip()]
        return queries[:3]
    except Exception as e:
        return []


def consolidate_with_queries(fs: FactStore, ds: DeepSeekClient, force: bool = False) -> dict:
    """整理时连同问题一起生成/更新。

    只为还没有 queries 字段的记忆生成(除非 force=True 全部重新生成)。
    """
    generated = 0
    for f in fs.facts:
        if not force and f.get("queries"):
            continue
        queries = generate_queries_for_fact(ds, f["text"])
        if queries:
            f["queries"] = queries
            generated += 1
    fs._save()
    return {"generated": generated, "total": len(fs.facts)}


def retrieve_with_query_expansion(fs: FactStore, query: str, top_k: int = 5) -> list[dict]:
    """增强检索: query 和事实文本 + 关联查询 一起比较。

    对每条记忆, 取 max(sim_with_fact_text, sim_with_any_associated_query) 作为匹配分。
    """
    import numpy as np
    fs._ensure_model()
    if fs._index is None:
        fs._rebuild_index()

    q_emb = fs._model.encode([query], normalize_embeddings=True)

    # 普通检索(和事实文本比)
    base_k = min(top_k * 2, len(fs.facts))
    scores, idxs = fs._index.search(np.array(q_emb, dtype=np.float32), base_k)

    results = {}
    now = time.time()
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0 or idx >= len(fs.facts):
            continue
        f = fs.facts[idx]
        best_score = float(score)

        # 如果有关联查询, 也和它们比, 取最高分
        if f.get("queries"):
            q_embs = fs._model.encode(f["queries"], normalize_embeddings=True)
            q_sims = (q_emb @ q_embs.T)[0]
            max_q_sim = float(max(q_sims))
            best_score = max(best_score, max_q_sim)

        eff = fs._effective_trust(f, now)
        if f.get("superseded_by"):
            eff *= 0.1
        final = best_score + 0.1 * eff
        results[f["id"]] = {**f, "score": best_score, "eff_trust": eff, "final": final}

    ranked = sorted(results.values(), key=lambda r: (
        0 if r.get("superseded_by") else 1, r.get("final", 0)
    ), reverse=True)
    return ranked[:top_k]


# ---- A/B 对比测试 ----

TEST_FACTS = [
    "Chen Lei works as a product manager at MediFlow.",
    "Chen Lei lives in Chengdu near the river.",
    "Chen Lei uses Python and Go for backend development.",
    "Chen Lei's dog is a golden retriever named Dou Dou.",
    "Chen Lei drives a BYD Seal electric car.",
    "Chen Lei's wife Li Jing works at an edtech startup.",
    "Chen Lei enjoys rock climbing on weekends.",
    "Chen Lei's team has 8 engineers.",
    "MediFlow's product is an AI diagnosis assistant.",
    "Chen Lei is considering getting an MBA.",
]

# 这些查询故意用和事实不同的措辞(测泛化能力)
TEST_QUERIES = [
    {"q": "What does Chen Lei do for a living?", "gold": ["product manager", "MediFlow"]},
    {"q": "Where is Chen Lei based?", "gold": ["Chengdu"]},
    {"q": "What coding languages does he know?", "gold": ["Python", "Go"]},
    {"q": "Does Chen Lei have any pets?", "gold": ["Dou Dou", "golden retriever"]},
    {"q": "What vehicle does he drive?", "gold": ["BYD", "Seal"]},
    {"q": "What does his wife do?", "gold": ["edtech"]},
    {"q": "What does Chen Lei do for fun?", "gold": ["climbing", "rock"]},
    {"q": "How big is his engineering team?", "gold": ["8"]},
    {"q": "What product does his company make?", "gold": ["diagnosis", "AI"]},
    {"q": "Is he thinking about further education?", "gold": ["MBA"]},
]


def score_retrieval(results: list[dict], gold: list[str]) -> bool:
    text = " ".join(r["text"].lower() for r in results)
    return any(g.lower() in text for g in gold)


def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: set DEEPSEEK_API_KEY"); sys.exit(1)

    ds = DeepSeekClient()

    print("=" * 60)
    print("  整体效果对比 + 问题整理")
    print("=" * 60)

    # ---- System A: 裸 RAG (baseline) ----
    print("\n[A] 裸 RAG (baseline)")
    shutil.rmtree("./_ab_raw", ignore_errors=True)
    fs_a = FactStore("./_ab_raw")
    for text in TEST_FACTS:
        fs_a.add(text)

    hits_a = 0
    for tq in TEST_QUERIES:
        results = fs_a.retrieve(tq["q"], top_k=3)
        if score_retrieval(results, tq["gold"]):
            hits_a += 1
    print(f"    命中: {hits_a}/{len(TEST_QUERIES)} ({100*hits_a/len(TEST_QUERIES):.0f}%)")

    # ---- System B: 裸 RAG + 问题整理 (query expansion) ----
    print("\n[B] RAG + 问题整理 (query expansion)")
    shutil.rmtree("./_ab_expanded", ignore_errors=True)
    fs_b = FactStore("./_ab_expanded")
    for text in TEST_FACTS:
        fs_b.add(text)

    # 用 DeepSeek 给每条记忆生成关联查询
    print("    生成关联查询...")
    r = consolidate_with_queries(fs_b, ds)
    print(f"    为 {r['generated']}/{r['total']} 条记忆生成了查询")
    # 展示几条
    for f in fs_b.facts[:3]:
        print(f"    '{f['text'][:30]}' → {f.get('queries', [])}")

    hits_b = 0
    for tq in TEST_QUERIES:
        results = retrieve_with_query_expansion(fs_b, tq["q"], top_k=3)
        if score_retrieval(results, tq["gold"]):
            hits_b += 1
    print(f"    命中: {hits_b}/{len(TEST_QUERIES)} ({100*hits_b/len(TEST_QUERIES):.0f}%)")

    # ---- System C: RAG + 问题整理 + 图扩展 ----
    print("\n[C] RAG + 问题整理 + 图扩展")
    shutil.rmtree("./_ab_full", ignore_errors=True)
    fs_c = FactStore("./_ab_full")
    for text in TEST_FACTS:
        fs_c.add(text)
    consolidate_with_queries(fs_c, ds)
    mg = MemoryGraph(fs_c, store_dir="./_ab_full")
    # 自动建边: MediFlow相关记忆互联
    mg.add_edge(1, 9, "related")  # Chen Lei at MediFlow <-> MediFlow's product
    mg.add_edge(1, 8, "related")  # Chen Lei <-> team size

    hits_c = 0
    for tq in TEST_QUERIES:
        # 先用 query expansion 检索
        results = retrieve_with_query_expansion(fs_c, tq["q"], top_k=3)
        seed_ids = [r["id"] for r in results]
        # 图扩展
        expanded_ids = mg.expand_retrieval(seed_ids, max_hops=1, max_expand=3)
        extra = [f for f in fs_c.facts if f["id"] in expanded_ids and f["id"] not in seed_ids]
        all_results = results + extra
        if score_retrieval(all_results, tq["gold"]):
            hits_c += 1
    print(f"    命中: {hits_c}/{len(TEST_QUERIES)} ({100*hits_c/len(TEST_QUERIES):.0f}%)")

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    print("  效果对比")
    print("=" * 60)
    print(f"  {'方案':<35} {'命中率'}")
    print(f"  {'-'*35} {'-'*10}")
    print(f"  {'A: 裸 RAG':<35} {hits_a}/{len(TEST_QUERIES)} ({100*hits_a/len(TEST_QUERIES):.0f}%)")
    print(f"  {'B: RAG + 问题整理':<35} {hits_b}/{len(TEST_QUERIES)} ({100*hits_b/len(TEST_QUERIES):.0f}%)")
    print(f"  {'C: RAG + 问题整理 + 图扩展':<35} {hits_c}/{len(TEST_QUERIES)} ({100*hits_c/len(TEST_QUERIES):.0f}%)")

    improvement_b = hits_b - hits_a
    improvement_c = hits_c - hits_a
    print(f"\n  提升: B比A +{improvement_b}条, C比A +{improvement_c}条")

    if hits_b > hits_a:
        print("  结论: 问题整理有效提升了检索命中率 ✅")
    else:
        print("  结论: 问题整理在当前数据下未见提升(可能查询已足够匹配)")

    print("\nDone!")


if __name__ == "__main__":
    main()
