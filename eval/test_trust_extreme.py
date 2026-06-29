"""信任层极端测试: 真答案和噪音语义几乎无法区分, 强制只能靠 trust。

设计: 同一主题的多条记忆, 措辞结构几乎相同, 只有一条是"当前有效"的,
其他是过时/试探性的。BGE 语义检索分不出(相似度都很接近),
唯一区分线索是"哪条被反复采纳过"(worth)。

这是信任层唯一真正不可替代的场景。如果这里也没提升, 那信任层确实可以CUT。
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
from memory_engine.memory_worth import MemoryWorth


def main():
    print("=" * 60)
    print("  信任层极端测试 (语义几乎无法区分)")
    print("=" * 60)

    # 故意让多条记忆措辞几乎相同, 语义检索分不出
    # 只有 truth 是当前有效的, 其他是不同时间说的同类话
    cases = [
        {
            "candidates": [
                "The user said their favorite number might be 7.",   # 试探
                "The user mentioned their favorite number could be 3.",  # 试探
                "The user confirmed their favorite number is 42.",   # 真答案(被采纳)
                "The user guessed their favorite number was maybe 9.",  # 试探
            ],
            "truth_idx": 2,
            "query": "What is the user's favorite number?",
            "gold": "42",
        },
        {
            "candidates": [
                "The user thinks their lucky color might be red.",
                "The user confirmed their lucky color is blue.",     # 真(被采纳)
                "The user wondered if their lucky color was green.",
                "The user said maybe their lucky color is yellow.",
            ],
            "truth_idx": 1,
            "query": "What is the user's lucky color?",
            "gold": "blue",
        },
    ]

    def score(results, gold):
        if not results:
            return 0.0
        return 1.0 if gold.lower() in results[0]["text"].lower() else 0.0

    # 先看裸检索的语义相似度有多接近
    print("\n[诊断] 各候选与查询的语义相似度:")
    shutil.rmtree("./_extreme_diag", ignore_errors=True)
    fs_d = FactStore("./_extreme_diag")
    fs_d._ensure_model()
    for c in cases[:1]:
        q_emb = fs_d._model.encode([c["query"]], normalize_embeddings=True)
        cand_embs = fs_d._model.encode(c["candidates"], normalize_embeddings=True)
        sims = (q_emb @ np.array(cand_embs).T)[0]
        for i, (cand, sim) in enumerate(zip(c["candidates"], sims)):
            mark = " ←真答案" if i == c["truth_idx"] else ""
            print(f"    sim={sim:.3f}: {cand[:45]}{mark}")
        print(f"    (相似度差距: {max(sims)-min(sims):.3f} — 越小越难区分)")

    # ===== A: 裸RAG =====
    print("\n[A] 裸RAG:")
    shutil.rmtree("./_extreme_a", ignore_errors=True)
    fs_a = FactStore("./_extreme_a")
    for c in cases:
        for cand in c["candidates"]:
            fs_a.add(cand)
    scores_a = []
    for c in cases:
        results = fs_a.retrieve(c["query"], top_k=3, reinforce=False)
        sc = score(results, c["gold"])
        scores_a.append(sc)
        print(f"  Q: {c['query'][:40]} → top1: {results[0]['text'][:40] if results else 'none'} [{sc}]")
    avg_a = sum(scores_a) / len(scores_a)
    print(f"  平均: {avg_a:.0%}")

    # ===== B: 信任层(真答案被采纳过) =====
    print("\n[B] 信任层(真答案被采纳10次, 试探项各被否定2次):")
    shutil.rmtree("./_extreme_b", ignore_errors=True)
    fs_b = FactStore("./_extreme_b")
    mw = MemoryWorth(fs_b)
    for c in cases:
        ids = []
        for cand in c["candidates"]:
            ids.append(fs_b.add(cand))
        # 真答案被反复采纳
        for _ in range(10):
            mw.record_success(ids[c["truth_idx"]])
        # 试探项被否定
        for i, fid in enumerate(ids):
            if i != c["truth_idx"]:
                for _ in range(2):
                    mw.record_failure(fid)

    scores_b = []
    for c in cases:
        results = fs_b.retrieve(c["query"], top_k=3, reinforce=False)
        sc = score(results, c["gold"])
        scores_b.append(sc)
        top = results[0] if results else None
        print(f"  Q: {c['query'][:40]} → top1: {top['text'][:40] if top else 'none'} (trust={top.get('eff_trust',0):.2f}) [{sc}]")
    avg_b = sum(scores_b) / len(scores_b)
    print(f"  平均: {avg_b:.0%}")

    # 对比
    print("\n" + "=" * 60)
    print(f"  裸RAG: {avg_a:.0%}  |  信任层: {avg_b:.0%}  |  提升: {avg_b-avg_a:+.0%}")
    if avg_b > avg_a + 0.1:
        print("  ✅ 在语义无法区分时, 信任层确实能靠'使用历史'救场")
    elif avg_b >= avg_a:
        print("  ⚠️ 不伤害但此场景也没救起来(可能加权系数0.1太小)")
    else:
        print("  ❌ 信任层反而更差")
    print("\nDone!")


if __name__ == "__main__":
    main()
