"""完整叠加验证: 矛盾检测(写入) + query expansion + 图扩展, 一起能到多少。

前面分开测:
- 矛盾检测: temporal 50%->83%
- query expansion: distractor 88%->100%
- 图扩展: multihop 50%->100%
三个针对不同弱项, 叠加应该接近全满。
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
from memory_engine.conflict_detector import ConflictDetector
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


def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: set DEEPSEEK_API_KEY"); sys.exit(1)
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    facts, queries = data["facts"], data["queries"]
    ds = DeepSeekClient()

    print("=" * 60)
    print("  完整叠加: 矛盾检测 + query exp + 图扩展")
    print("=" * 60)

    # 1. 用矛盾检测写入(处理时序矛盾)
    shutil.rmtree("./_full_stack", ignore_errors=True)
    fs = FactStore("./_full_stack")
    cd = ConflictDetector(fs, deepseek=ds)
    print("\n[1] 矛盾检测写入167条...")
    for t in facts:
        cd.add_with_conflict_check(t, auto_resolve=True)

    # 2. query expansion(给关键事实生成关联查询)
    print("[2] 生成 query expansion...")
    key_facts = [f for f in fs.facts if not f["text"].startswith("Random note")]
    for f in key_facts[:70]:
        prompt = f"Given fact: \"{f['text']}\"\nGenerate 2 alternative questions. Output only 2 lines."
        try:
            resp = ds.simple(prompt, temperature=0.3, max_tokens=60)
            f["queries"] = [q.strip().strip("-").strip() for q in resp.strip().split("\n") if q.strip()][:2]
        except:
            pass
    fs._save()

    # 3. 建关系网络
    print("[3] 构建关系网络...")
    mg = MemoryGraph(fs, store_dir="./_full_stack")
    mg.auto_build_network(deepseek=ds, sim_threshold=0.55, max_edges_per_fact=3)

    # 完整检索: query expansion + 图扩展
    def retrieve_full(query, top_k=5):
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
        base = ranked[:top_k]
        # 图扩展
        seed_ids = [r["id"] for r in base]
        expanded = mg.expand_retrieval(seed_ids, max_hops=2, max_expand=5)
        extra = [f for f in fs.facts if f["id"] in expanded and f["id"] not in seed_ids]
        return base + extra

    # 评估
    by_type = {}
    for q in queries:
        by_type.setdefault(q["type"], []).append(score(retrieve_full(q["q"]), q))
    types = {t: sum(s)/len(s) for t, s in by_type.items()}
    overall = sum(sum(s) for s in by_type.values()) / sum(len(s) for s in by_type.values())

    print("\n" + "=" * 60)
    print("  完整叠加结果")
    print("=" * 60)
    print(f"  {'类型':<14} {'baseline':<10} {'完整叠加':<10} {'提升'}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*6}")
    base_by_type = data.get("baseline_by_type", {})
    for t in ["distractor", "temporal", "multihop", "reformulated"]:
        b = base_by_type.get(t, 0)
        f = types.get(t, 0)
        print(f"  {t:<14} {b:<10.0%} {f:<10.0%} {f-b:+.0%}")
    print(f"  {'-'*14} {'-'*10} {'-'*10}")
    print(f"  {'OVERALL':<14} {data.get('baseline_overall',0.77):<10.0%} {overall:<10.0%} {overall-data.get('baseline_overall',0.77):+.0%}")

    out = Path(__file__).parent / "full_stack_results.json"
    out.write_text(json.dumps({"types": types, "overall": overall}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Saved to {out}")
    print("\nDone!")


if __name__ == "__main__":
    main()
