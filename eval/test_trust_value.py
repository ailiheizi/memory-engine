"""信任层价值测试: 设计能暴露 trust/worth/反馈 价值的场景。

审计说信任层"实测=0", 因为之前的评测里 trust 和裸RAG数值完全一致。
原因: 那些场景里语义检索已经够好, trust 没有发挥空间。

信任层真正该发挥作用的场景:
- 有多条语义相似度接近的记忆(检索分不出)
- 其中一条被反复采纳(高 worth), 其他是噪音(低/未验证)
- 看 trust 能否把"被验证过的"顶到前面

这正是裸RAG做不到的: 裸RAG只看语义, 分不出"哪条被实际证明有用过"。
"""

import sys
import os
import json
import shutil
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_engine.fact_store import FactStore
from memory_engine.memory_worth import MemoryWorth


def main():
    print("=" * 60)
    print("  信任层价值测试 (噪音稀释 + 反复采纳场景)")
    print("=" * 60)

    # 场景: 用户的真实偏好 + 多条语义相似的干扰/过时信息
    # 真答案被反复采纳(高worth), 干扰项从未被采纳
    scenarios = [
        {
            "truth": "The user strongly prefers dark roast coffee, no sugar.",
            "noise": [
                "The user once tried a latte at a cafe.",
                "The user mentioned coffee is too bitter sometimes.",
                "The user drinks tea in the afternoon occasionally.",
            ],
            "query": "What coffee should I make for the user?",
            "gold": "dark roast",
        },
        {
            "truth": "The user's primary work machine runs Arch Linux.",
            "noise": [
                "The user used Windows in the past.",
                "The user has a MacBook for travel.",
                "The user tried Ubuntu once but switched away.",
            ],
            "query": "What OS should I assume for the user's main setup?",
            "gold": "Arch",
        },
        {
            "truth": "The user's deadline for the project is March 15.",
            "noise": [
                "The user mentioned a soft target around February.",
                "The user said the timeline might slip to April.",
                "The user discussed a tentative January milestone.",
            ],
            "query": "When is the user's actual project deadline?",
            "gold": "March 15",
        },
    ]

    def score(results, gold):
        # 真答案排第一才算满分
        if not results:
            return 0.0
        top = results[0]["text"].lower()
        if gold.lower() in top:
            return 1.0
        # 在top3里算半分
        text3 = " ".join(r["text"].lower() for r in results[:3])
        return 0.5 if gold.lower() in text3 else 0.0

    # ===== A: 裸RAG (无信任反馈) =====
    print("\n[A] 裸RAG (无信任): 真答案和噪音平等竞争")
    shutil.rmtree("./_trustval_a", ignore_errors=True)
    fs_a = FactStore("./_trustval_a")
    for s in scenarios:
        fs_a.add(s["truth"])
        for n in s["noise"]:
            fs_a.add(n)

    scores_a = []
    for s in scenarios:
        results = fs_a.retrieve(s["query"], top_k=3, reinforce=False)
        sc = score(results, s["gold"])
        scores_a.append(sc)
        print(f"  Q: {s['query'][:40]}")
        print(f"     top1: {results[0]['text'][:45] if results else 'none'}  [{sc}]")
    avg_a = sum(scores_a) / len(scores_a)
    print(f"  平均: {avg_a:.0%}")

    # ===== B: 信任层 + 反复采纳真答案 =====
    print("\n[B] 信任层: 真答案被反复采纳(worth上升)后")
    shutil.rmtree("./_trustval_b", ignore_errors=True)
    fs_b = FactStore("./_trustval_b")
    mw = MemoryWorth(fs_b)
    truth_ids = []
    for s in scenarios:
        tid = fs_b.add(s["truth"])
        truth_ids.append(tid)
        for n in s["noise"]:
            fs_b.add(n)

    # 模拟使用历史: 真答案被采纳5次, 噪音被忽略
    print("  模拟使用: 真答案各被采纳5次...")
    for tid in truth_ids:
        for _ in range(5):
            mw.record_success(tid)
    # 噪音被检索但忽略(record_failure)
    for f in fs_b.facts:
        if f["id"] not in truth_ids and "once" in f["text"].lower() or "tried" in f["text"].lower():
            mw.record_failure(f["id"])
            mw.record_failure(f["id"])

    scores_b = []
    for s in scenarios:
        results = fs_b.retrieve(s["query"], top_k=3, reinforce=False)
        sc = score(results, s["gold"])
        scores_b.append(sc)
        print(f"  Q: {s['query'][:40]}")
        top = results[0] if results else None
        print(f"     top1: {top['text'][:45] if top else 'none'} (trust={top.get('eff_trust',0):.2f})  [{sc}]")
    avg_b = sum(scores_b) / len(scores_b)
    print(f"  平均: {avg_b:.0%}")

    # ===== 对比 =====
    print("\n" + "=" * 60)
    print("  信任层价值")
    print("=" * 60)
    print(f"  裸RAG(平等竞争):     {avg_a:.0%}")
    print(f"  信任层(采纳后加权):   {avg_b:.0%}")
    print(f"  提升: {avg_b - avg_a:+.0%}")

    if avg_b > avg_a + 0.1:
        print(f"\n  ✅ 信任层有效: 反复采纳的真信息被顶到前面, 噪音沉底")
        print("     这是裸RAG做不到的——它分不出'哪条被验证过'")
    elif avg_b >= avg_a:
        print(f"\n  ⚠️ 信任层不伤害但提升有限(此场景语义已能区分)")
    else:
        print(f"\n  ❌ 信任层反而降低了准确率")

    out = Path(__file__).parent / "trust_value_results.json"
    out.write_text(json.dumps({"baseline": avg_a, "with_trust": avg_b, "delta": avg_b-avg_a,
                               "scores_a": scores_a, "scores_b": scores_b}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Saved to {out}")
    print("\nDone!")


if __name__ == "__main__":
    main()
