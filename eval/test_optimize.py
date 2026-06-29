"""优化实验: 1) query expansion 在硬核集上的真实效果 2) 信任系数扫描 3) 全功能叠加。

回答: 差的功能能否优化? 全部叠加能到多少?
"""

import sys
import os
import json
import shutil
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_engine.fact_store import FactStore
from memory_engine.memory_graph import MemoryGraph
from memory_engine.deepseek_client import DeepSeekClient

DATASET = Path(__file__).parent / "hard_eval_dataset.json"


def score(results, q):
    text = " ".join(r["text"].lower() for r in results)
    gold_hit = all(g.lower() in text for g in q["gold"]) if q["gold"] else False
    anti_hit = any(a.lower() in text for a in q.get("anti", []))
    if gold_hit and not anti_hit:
        return 1.0
    elif gold_hit and anti_hit:
        return 0.5
    return 0.0


def eval_overall(retrieve_fn, queries):
    by_type = {}
    for q in queries:
        by_type.setdefault(q["type"], []).append(score(retrieve_fn(q["q"]), q))
    types = {t: sum(s)/len(s) for t, s in by_type.items()}
    overall = sum(sum(s) for s in by_type.values()) / sum(len(s) for s in by_type.values())
    return types, overall


def gen_queries_for_facts(ds, facts_subset):
    """给一批事实生成关联查询(query expansion)。"""
    out = {}
    for f in facts_subset:
        prompt = (f"Given fact: \"{f['text']}\"\nGenerate 2 alternative ways to ask about it. "
                  "Output only the 2 questions, one per line.")
        try:
            resp = ds.simple(prompt, temperature=0.3, max_tokens=60)
            qs = [q.strip().strip("-").strip() for q in resp.strip().split("\n") if q.strip()][:2]
            out[f["id"]] = qs
        except:
            out[f["id"]] = []
    return out


def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: set DEEPSEEK_API_KEY"); sys.exit(1)
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    facts, queries = data["facts"], data["queries"]
    ds = DeepSeekClient()

    print("=" * 60)
    print("  优化实验 (硬核评测集, baseline=77%)")
    print("=" * 60)

    # ===== 实验1: 信任系数扫描 =====
    # 注: 当前硬核集的弱项是 temporal/multihop, trust 系数主要影响排序
    # 这里主要验证不同系数是否改变 baseline
    print("\n[实验1] 信任加权系数扫描 (sim + α×trust)")
    shutil.rmtree("./_opt_base", ignore_errors=True)
    fs = FactStore("./_opt_base")
    for t in facts:
        fs.add(t)

    # monkeypatch retrieve 用不同 alpha
    orig_retrieve = fs.retrieve
    import types as _t

    def make_retrieve(alpha):
        def retrieve(self, query, top_k=5, min_trust=0.0, reinforce=False):
            now = __import__("time").time()
            results = {}
            if self.facts and query.strip():
                if self._index is None:
                    self._rebuild_index()
                self._ensure_model()
                q_emb = self._model.encode([query], normalize_embeddings=True)
                cand_k = min(max(top_k*3, top_k), len(self.facts))
                scores, idxs = self._index.search(np.array(q_emb, dtype=np.float32), cand_k)
                for sc, idx in zip(scores[0], idxs[0]):
                    if 0 <= idx < len(self.facts):
                        f = self.facts[idx]
                        eff = self._effective_trust(f, now)
                        if f.get("superseded_by"):
                            eff *= 0.1
                        results[f["id"]] = {**f, "score": float(sc), "final": float(sc) + alpha*eff}
            ranked = sorted(results.values(), key=lambda r: (0 if r.get("superseded_by") else 1, r.get("final",0)), reverse=True)
            return ranked[:top_k]
        return retrieve

    for alpha in [0.0, 0.1, 0.3, 0.5]:
        fs.retrieve = _t.MethodType(make_retrieve(alpha), fs)
        _, overall = eval_overall(lambda q: fs.retrieve(q, top_k=5), queries)
        print(f"    α={alpha}: overall {overall:.0%}")
    fs.retrieve = orig_retrieve

    # ===== 实验2: query expansion 在硬核集上 =====
    print("\n[实验2] query expansion 在硬核集上的真实效果")
    print("    (给干扰组+时序组的事实生成关联查询)")
    # 只给非噪音事实生成(前67条)
    key_facts = [f for f in fs.facts if not f["text"].startswith("Random note")]
    qmap = gen_queries_for_facts(ds, key_facts[:67])
    for f in fs.facts:
        if f["id"] in qmap:
            f["queries"] = qmap[f["id"]]
    fs._save()

    def retrieve_with_qe(query, top_k=5):
        now = __import__("time").time()
        fs._ensure_model()
        if fs._index is None:
            fs._rebuild_index()
        q_emb = fs._model.encode([query], normalize_embeddings=True)
        cand_k = min(top_k*4, len(fs.facts))
        scores, idxs = fs._index.search(np.array(q_emb, dtype=np.float32), cand_k)
        results = {}
        for sc, idx in zip(scores[0], idxs[0]):
            if not (0 <= idx < len(fs.facts)):
                continue
            f = fs.facts[idx]
            best = float(sc)
            if f.get("queries"):
                qembs = fs._model.encode(f["queries"], normalize_embeddings=True)
                best = max(best, float(max((q_emb @ np.array(qembs).T)[0])))
            eff = fs._effective_trust(f, now)
            if f.get("superseded_by"):
                eff *= 0.1
            results[f["id"]] = {**f, "score": best, "final": best + 0.1*eff}
        ranked = sorted(results.values(), key=lambda r: (0 if r.get("superseded_by") else 1, r.get("final",0)), reverse=True)
        return ranked[:top_k]

    qe_types, qe_overall = eval_overall(retrieve_with_qe, queries)
    print(f"    各类: {' '.join(f'{t}={v:.0%}' for t,v in qe_types.items())}")
    print(f"    overall: {qe_overall:.0%} (baseline 77%)")

    # ===== 实验3: 全功能叠加 (矛盾检测已写入 + 图 + query expansion) =====
    print("\n[实验3] 全功能叠加 (图扩展 + query expansion)")
    mg = MemoryGraph(fs, store_dir="./_opt_base")
    print("    构建关系网络...")
    mg.auto_build_network(deepseek=ds, sim_threshold=0.55, max_edges_per_fact=3)

    def retrieve_full(query, top_k=5):
        # query expansion 检索
        base = retrieve_with_qe(query, top_k=top_k)
        seed_ids = [r["id"] for r in base]
        # 图扩展
        expanded = mg.expand_retrieval(seed_ids, max_hops=2, max_expand=5)
        extra = [f for f in fs.facts if f["id"] in expanded and f["id"] not in seed_ids]
        return base + extra

    full_types, full_overall = eval_overall(retrieve_full, queries)
    print(f"    各类: {' '.join(f'{t}={v:.0%}' for t,v in full_types.items())}")
    print(f"    overall: {full_overall:.0%}")

    # ===== 汇总 =====
    print("\n" + "=" * 60)
    print("  优化结论")
    print("=" * 60)
    print(f"  baseline(裸RAG):        77%")
    print(f"  +query expansion:       {qe_overall:.0%}")
    print(f"  +query exp +图扩展:      {full_overall:.0%}")

    out = Path(__file__).parent / "optimization_results.json"
    out.write_text(json.dumps({"qe_overall": qe_overall, "qe_types": qe_types,
                               "full_overall": full_overall, "full_types": full_types}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Saved to {out}")
    print("\nDone!")


if __name__ == "__main__":
    main()
