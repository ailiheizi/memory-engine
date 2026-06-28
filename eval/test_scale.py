"""大规模纵向 Eval: 500条记忆 + 模拟30天时间跨度。

验证: 在真实使用模式下, 信任层(强化+衰减+矛盾降权)是否比裸RAG有可测量的优势。

模拟设计:
- 500条记忆, 分5批写入(模拟5周每周写入100条)
- 部分记忆被"反复使用"(模拟高频记忆)
- 部分记忆"很久没被提起"(模拟冷记忆)
- 30对矛盾记忆(旧→新更新)
- 时间戳手动操作(模拟真实时间跨度)

检查点查询类型:
- 最新事实(本周写入的)
- 高频事实(被反复检索的)
- 冷事实(只写入过一次, 3周没被提及)
- 矛盾事实(旧的被新的更新了)
- 噪音干扰(相似但不同的多条记忆里找对的)
"""

import sys
import os
import json
import time
import random
import shutil
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_engine.fact_store import FactStore
from memory_engine.conflict_detector import ConflictDetector

random.seed(42)

# ---- 生成 500 条模拟记忆 ----

CATEGORIES = {
    "personal": [
        "The user's favorite color is {v}.",
        "The user's shoe size is {v}.",
        "The user was born in {v}.",
        "The user's blood type is {v}.",
        "The user's phone number ends in {v}.",
    ],
    "work": [
        "The user's current project is {v}.",
        "The user's team uses {v} for CI/CD.",
        "The user's sprint length is {v} weeks.",
        "The user's manager is named {v}.",
        "The user's deployment target is {v}.",
    ],
    "preferences": [
        "The user prefers {v} for morning drinks.",
        "The user's go-to lunch spot is {v}.",
        "The user listens to {v} while working.",
        "The user reads {v} before bed.",
        "The user's preferred IDE theme is {v}.",
    ],
    "tech": [
        "The user is learning {v} this month.",
        "The user's database of choice is {v}.",
        "The user's shell is {v}.",
        "The user uses {v} for version control hosting.",
        "The user's container runtime is {v}.",
    ],
    "social": [
        "The user's best friend is named {v}.",
        "The user's colleague {v} sits next to them.",
        "The user's mentor works at {v}.",
        "The user met {v} at a conference last year.",
        "The user's lunch buddy is {v}.",
    ],
}

VALUES = {
    "personal": ["blue", "42", "1995", "A+", "8832", "green", "38", "Chengdu", "O-", "6677"],
    "work": ["RecommendAI", "GitHub Actions", "2", "Zhang Wei", "AWS us-east-1",
             "HealthBot", "Jenkins", "3", "Li Ming", "Aliyun cn-hangzhou"],
    "preferences": ["black coffee", "the noodle place", "lo-fi", "sci-fi novels", "dark mode",
                    "green tea", "the sushi bar", "jazz", "tech blogs", "solarized"],
    "tech": ["Rust", "PostgreSQL", "zsh", "GitHub", "Docker",
             "Zig", "ClickHouse", "fish", "Gitea", "Podman"],
    "social": ["Wang Hao", "Liu Fang", "ByteDance", "Professor Chen", "Xiao Li",
               "Zhao Min", "Yang Jie", "Alibaba", "Dr. Wu", "Lao Zhang"],
}

DAY = 86400


def generate_memories(n=500):
    """Generate n memories with varied categories."""
    memories = []
    cats = list(CATEGORIES.keys())
    for i in range(n):
        cat = cats[i % len(cats)]
        templates = CATEGORIES[cat]
        tmpl = templates[i % len(templates)]
        vals = VALUES[cat]
        val = vals[i % len(vals)]
        text = tmpl.format(v=val)
        memories.append({"text": text, "category": cat, "idx": i})
    return memories


def generate_contradictions(memories, n=30):
    """Generate n contradiction pairs (old -> new update)."""
    contras = []
    # Pick some to update
    updatable = [m for m in memories if m["category"] in ("work", "tech", "preferences")]
    chosen = random.sample(updatable, min(n, len(updatable)))
    for m in chosen:
        cat = m["category"]
        vals = VALUES[cat]
        old_val = None
        for v in vals:
            if v in m["text"]:
                old_val = v
                break
        if not old_val:
            continue
        new_val = random.choice([v for v in vals if v != old_val])
        new_text = m["text"].replace(old_val, new_val)
        contras.append({"old_idx": m["idx"], "old_text": m["text"], "new_text": new_text})
    return contras[:n]


def generate_checkpoints(memories, contradictions):
    """Generate checkpoint queries testing different memory types."""
    checkpoints = []

    # Type A: 最新事实(最后一批写入的) - 用自然语言查询
    recent_queries = [
        {"fact_idx": 400, "q": "What is the user learning this month?", "gold_from": "tech"},
        {"fact_idx": 401, "q": "What database does the user prefer?", "gold_from": "tech"},
        {"fact_idx": 402, "q": "What shell does the user use?", "gold_from": "tech"},
        {"fact_idx": 405, "q": "What does the user drink in the morning?", "gold_from": "preferences"},
        {"fact_idx": 410, "q": "Who is the user's best friend?", "gold_from": "social"},
    ]
    for rq in recent_queries:
        m = memories[rq["fact_idx"]]
        gold = [v for v in VALUES[rq["gold_from"]] if v.lower() in m["text"].lower()]
        if gold:
            checkpoints.append({"q": rq["q"], "gold": gold, "type": "recent", "source": m["text"]})

    # Type B: 矛盾(应该回答新的不是旧的)
    contra_queries = [
        "What is the user's current project?",
        "What does the team use for CI/CD?",
        "What is the user learning this month?",
        "What does the user drink in the morning?",
        "What database does the user prefer?",
    ]
    for i, c in enumerate(contradictions[:5]):
        q = contra_queries[i] if i < len(contra_queries) else f"Tell me about: {c['new_text'][:20]}"
        new_vals = []
        for cat_vals in VALUES.values():
            for v in cat_vals:
                if v.lower() in c["new_text"].lower() and v.lower() not in c["old_text"].lower():
                    new_vals.append(v)
        old_vals = []
        for cat_vals in VALUES.values():
            for v in cat_vals:
                if v.lower() in c["old_text"].lower() and v.lower() not in c["new_text"].lower():
                    old_vals.append(v)
        if new_vals:
            checkpoints.append({"q": q, "gold": new_vals, "anti": old_vals, "type": "contradiction", "source": c["new_text"]})

    # Type C: 高频(前10条被反复检索, trust高)
    frequent_queries = [
        "What is the user's favorite color?",
        "What is the user's current project?",
        "What does the user prefer for morning drinks?",
        "What is the user learning?",
        "Who is the user's best friend?",
    ]
    for i, m in enumerate(memories[:5]):
        q = frequent_queries[i]
        gold = [v for v in VALUES[m["category"]] if v.lower() in m["text"].lower()]
        if gold:
            checkpoints.append({"q": q, "gold": gold, "type": "frequent", "source": m["text"]})

    # Type D: 冷事实(中间批次, 只写入没被检索过)
    cold_queries = [
        "What is the user's shoe size?",
        "What does the user listen to while working?",
        "Who sits next to the user?",
        "What is the user's sprint length?",
        "What does the user read before bed?",
    ]
    for i, m in enumerate(memories[200:205]):
        q = cold_queries[i]
        gold = [v for v in VALUES[m["category"]] if v.lower() in m["text"].lower()]
        if gold:
            checkpoints.append({"q": q, "gold": gold, "type": "cold", "source": m["text"]})

    return checkpoints


def run_system(store_dir, use_trust, memories, contradictions, checkpoints):
    """运行完整模拟: 写入+使用+矛盾+检查。"""
    shutil.rmtree(store_dir, ignore_errors=True)
    fs = FactStore(store_dir, decay_half_life_days=14.0 if use_trust else 0.0)

    now = time.time()

    # 1. 分5批写入(模拟5周)
    batch_size = 100
    for week in range(5):
        batch = memories[week * batch_size: (week + 1) * batch_size]
        write_time = now - (4 - week) * 7 * DAY  # 第1批=4周前, 第5批=本周
        for m in batch:
            fid = fs.add(m["text"])
            # 手动设置写入时间(模拟时间跨度)
            for f in fs.facts:
                if f["id"] == fid:
                    f["ts"] = int(write_time)
                    f["last_used"] = int(write_time)
                    break
        fs._save()

    # 2. 模拟高频使用(前10条被反复检索)
    for _ in range(8):  # 8次检索
        for m in memories[:10]:
            fs.retrieve(m["text"][:30], top_k=1, reinforce=use_trust)
            # 更新 last_used 到"最近"
            for f in fs.facts:
                if m["text"] in f["text"]:
                    f["last_used"] = int(now - random.randint(0, 2) * DAY)
                    break
    fs._save()

    # 3. 写入矛盾记忆(模拟本周更新)
    for c in contradictions:
        fid = fs.add(c["new_text"])
        # 找到旧记忆, 标记 superseded + 降 trust
        for f in fs.facts:
            if f["text"] == c["old_text"]:
                f["trust"] = 0.05
                f["superseded_by"] = fid
                break
    fs._save()

    # 4. 跑检查点
    results = []
    for cp in checkpoints:
        retrieved = fs.retrieve(cp["q"], top_k=5, reinforce=False)
        recalled_text = " ".join(r["text"].lower() for r in retrieved)
        hits = sum(1 for g in cp["gold"] if g.lower() in recalled_text)
        score = hits / len(cp["gold"])
        # 对矛盾类型, 额外检查是否错误召回了旧值
        anti_hit = False
        if cp.get("anti"):
            anti_hit = any(a.lower() in recalled_text for a in cp["anti"])
        results.append({
            "type": cp["type"], "q": cp["q"], "score": score,
            "anti_hit": anti_hit, "gold": cp["gold"],
        })

    return results


def main():
    print("=" * 60)
    print("  大规模纵向 Eval (500条 + 模拟30天)")
    print("  信任层 vs 裸RAG")
    print("=" * 60)

    # 准备数据
    memories = generate_memories(500)
    contradictions = generate_contradictions(memories, n=30)
    checkpoints = generate_checkpoints(memories, contradictions)
    print(f"\n  记忆: {len(memories)}条, 矛盾对: {len(contradictions)}, 检查点: {len(checkpoints)}")

    # System A: 信任层
    print("\n[A] RAG + 信任层 (trust + reinforce + decay 14d + superseded)")
    results_a = run_system("./_scale_trust", True, memories, contradictions, checkpoints)

    # System B: 裸RAG
    print("[B] 裸 RAG (无信任, 无衰减, 无superseded)")
    results_b = run_system("./_scale_raw", False, memories, contradictions, checkpoints)

    # 按类型统计
    print("\n" + "=" * 60)
    print("  按查询类型对比")
    print("=" * 60)
    types = ["recent", "contradiction", "frequent", "cold"]
    print(f"  {'类型':<16} {'信任层':<10} {'裸RAG':<10} {'差异':<8} {'说明'}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*8} {'-'*20}")

    overall_a, overall_b = [], []
    for t in types:
        a_scores = [r["score"] for r in results_a if r["type"] == t]
        b_scores = [r["score"] for r in results_b if r["type"] == t]
        avg_a = sum(a_scores) / len(a_scores) if a_scores else 0
        avg_b = sum(b_scores) / len(b_scores) if b_scores else 0
        diff = avg_a - avg_b
        overall_a.extend(a_scores)
        overall_b.extend(b_scores)
        desc = {"recent": "最新写入", "contradiction": "矛盾更新后", "frequent": "高频记忆", "cold": "冷门记忆"}
        print(f"  {t:<16} {avg_a:<10.0%} {avg_b:<10.0%} {diff:+<8.0%} {desc.get(t,'')}")

    # 矛盾类型额外指标: 旧值误召回率
    print("\n  矛盾记忆额外指标:")
    a_anti = [r["anti_hit"] for r in results_a if r["type"] == "contradiction"]
    b_anti = [r["anti_hit"] for r in results_b if r["type"] == "contradiction"]
    a_anti_rate = sum(a_anti) / len(a_anti) if a_anti else 0
    b_anti_rate = sum(b_anti) / len(b_anti) if b_anti else 0
    print(f"    旧值误召回率: 信任层 {a_anti_rate:.0%} vs 裸RAG {b_anti_rate:.0%}")

    # 总体
    total_a = sum(overall_a) / len(overall_a) if overall_a else 0
    total_b = sum(overall_b) / len(overall_b) if overall_b else 0
    print(f"\n  总体: 信任层 {total_a:.1%} vs 裸RAG {total_b:.1%} (差异 {total_a-total_b:+.1%})")
    verdict = "信任层有优势" if total_a > total_b + 0.03 else ("无显著差异" if abs(total_a-total_b) <= 0.03 else "裸RAG更好")
    print(f"  结论: {verdict}")

    # 保存
    out = Path(__file__).parent / "scale_eval_results.json"
    out.write_text(json.dumps({
        "trust": {"results": results_a, "overall": total_a},
        "raw": {"results": results_b, "overall": total_b},
        "n_memories": len(memories), "n_contradictions": len(contradictions),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved to {out}")
    print("\nDone!")


if __name__ == "__main__":
    main()
