"""功能消融测试: 用硬核评测集, 看每个功能能否把对应弱项拉上来。

baseline(裸RAG): distractor 88% / temporal 50% / multihop 50% / reformulated 100% / overall 77%

测试:
- 矛盾检测: 能否把 temporal(时序矛盾) 50% 拉高?
- 记忆图: 能否把 multihop(多跳) 50% 拉高?
- 问题整理: 能否把 distractor/reformulated 拉高?

每个功能 ON vs OFF, 用同一评测集, 看 delta。
"""

import sys
import os
import json
import shutil
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


def eval_by_type(retrieve_fn, queries):
    by_type = {}
    for q in queries:
        results = retrieve_fn(q["q"])
        by_type.setdefault(q["type"], []).append(score(results, q))
    return {t: sum(s)/len(s) for t, s in by_type.items()}


def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: set DEEPSEEK_API_KEY"); sys.exit(1)

    data = json.loads(DATASET.read_text(encoding="utf-8"))
    facts, queries = data["facts"], data["queries"]
    ds = DeepSeekClient()

    print("=" * 60)
    print("  功能消融测试 (硬核评测集, baseline=77%)")
    print("=" * 60)

    types = ["distractor", "temporal", "multihop", "reformulated"]

    def show(label, scores):
        line = f"  {label:<24}"
        for t in types:
            line += f" {scores.get(t,0):>5.0%}"
        avg = sum(scores.values())/len(scores)
        line += f"  | {avg:.0%}"
        print(line)
        return avg

    print(f"\n  {'方案':<24} {'干扰':>5} {'时序':>5} {'多跳':>5} {'改写':>5}  | 总")
    print(f"  {'-'*24} {'-'*5} {'-'*5} {'-'*5} {'-'*5}  | ---")

    # ===== A: baseline 裸RAG =====
    shutil.rmtree("./_abl_base", ignore_errors=True)
    fs_base = FactStore("./_abl_base")
    for t in facts:
        fs_base.add(t)
    base_scores = eval_by_type(lambda q: fs_base.retrieve(q, top_k=5, reinforce=False), queries)
    base_avg = show("A: 裸RAG baseline", base_scores)

    # ===== B: + 矛盾检测 (写入时处理时序矛盾) =====
    shutil.rmtree("./_abl_conflict", ignore_errors=True)
    fs_c = FactStore("./_abl_conflict")
    cd = ConflictDetector(fs_c, deepseek=ds)
    # 用矛盾检测方式写入(自动处理时序矛盾)
    for t in facts:
        cd.add_with_conflict_check(t, auto_resolve=True)
    conflict_scores = eval_by_type(lambda q: fs_c.retrieve(q, top_k=5, reinforce=False), queries)
    conflict_avg = show("B: +矛盾检测", conflict_scores)

    # ===== C: + 记忆图 (多跳扩展) =====
    shutil.rmtree("./_abl_graph", ignore_errors=True)
    fs_g = FactStore("./_abl_graph")
    for t in facts:
        fs_g.add(t)
    mg = MemoryGraph(fs_g, store_dir="./_abl_graph")
    print("  (构建关系网络中...)")
    mg.auto_build_network(deepseek=ds, sim_threshold=0.55, max_edges_per_fact=3)

    def graph_retrieve(q):
        results = mg.graph_enhanced_retrieve(q, top_k=5, expand_hops=2)
        return results
    graph_scores = eval_by_type(graph_retrieve, queries)
    graph_avg = show("C: +记忆图", graph_scores)

    # ===== 汇总 delta =====
    print("\n" + "=" * 60)
    print("  提升分析 (vs baseline)")
    print("=" * 60)
    print(f"  矛盾检测: 总分 {base_avg:.0%} -> {conflict_avg:.0%} ({conflict_avg-base_avg:+.0%})")
    print(f"    时序项: {base_scores.get('temporal',0):.0%} -> {conflict_scores.get('temporal',0):.0%} ({conflict_scores.get('temporal',0)-base_scores.get('temporal',0):+.0%})")
    print(f"  记忆图:   总分 {base_avg:.0%} -> {graph_avg:.0%} ({graph_avg-base_avg:+.0%})")
    print(f"    多跳项: {base_scores.get('multihop',0):.0%} -> {graph_scores.get('multihop',0):.0%} ({graph_scores.get('multihop',0)-base_scores.get('multihop',0):+.0%})")

    print("\n  裁决:")
    if conflict_scores.get('temporal',0) > base_scores.get('temporal',0) + 0.1:
        print(f"    矛盾检测: ✅ 有效(时序+{conflict_scores.get('temporal',0)-base_scores.get('temporal',0):.0%})")
    else:
        print(f"    矛盾检测: ❌ 时序项无明显提升")
    if graph_scores.get('multihop',0) > base_scores.get('multihop',0) + 0.1:
        print(f"    记忆图: ✅ 有效(多跳+{graph_scores.get('multihop',0)-base_scores.get('multihop',0):.0%})")
    else:
        print(f"    记忆图: ❌ 多跳项无明显提升")

    out = Path(__file__).parent / "ablation_results.json"
    out.write_text(json.dumps({"baseline": base_scores, "conflict": conflict_scores, "graph": graph_scores}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Saved to {out}")
    print("\nDone!")


if __name__ == "__main__":
    main()
