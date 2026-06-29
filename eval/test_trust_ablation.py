"""信任层(时间衰减)消融实验: 到底帮忙还是帮倒忙?

被测变量 = FactStore 的时间衰减机制。
    final = 语义相似度 + 0.1 × 有效信任
    有效信任 = trust × 0.5^(age_days / half_life)
    half_life=14  -> 衰减开启 (ON)
    half_life=0   -> 衰减关闭 (OFF, trust 仍计入但不随时间衰减)

数据集 (30 条) 故意构造成"衰减应该起作用"的场景:
  - 10 条 useful: 被反复采用(高 trust)且最近用过(last_used 新鲜)
  - 20 条 noise: 从没被用过(默认 trust=0.5)且很久没碰(last_used 旧)
    其中一部分 noise 是和 useful 语义高度接近的"过期干扰项"
    (例如旧地址/旧职位/别人的同类信息), 这正是裸语义检索分不清、
     而信任+新鲜度本该能压下去的东西。

为了公平, 还放了 3 条 "合法但冷门" 的 useful 事实(从没被强化、很旧),
查询它们时衰减反而可能把正确答案埋掉 -> 用来暴露信任层的副作用。

指标:
  - rank1 准确率: 正确事实排在第 1 (排除 pinned, 这里无 pinned)
  - hit@3:        正确事实出现在 top-3
对 ON / OFF 各跑同一批 10 个查询, 比较两个指标。
检索是纯 embedding, 不调用 LLM。
"""

import sys
import time
import shutil
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_engine.fact_store import FactStore

STORE = "./_trust_ablation"
DAY = 86400.0

# ---- 30 条事实 ----
# role: useful_hot = 被反复采用+新鲜; useful_cold = 合法但从没用过+旧(测副作用); noise = 干扰/垃圾
# trap_for: 该 noise 是哪个查询的"过期干扰项"(语义接近但答案错)
FACTS = [
    # --- 10 条 useful_hot (当前真相, 高频, 新鲜) ---
    {"text": "Wang Fang is currently the lead data scientist at NovaBank.",        "role": "useful_hot"},
    {"text": "Wang Fang now lives in the Pudong district of Shanghai.",            "role": "useful_hot"},
    {"text": "Wang Fang's main programming languages are Python and SQL.",         "role": "useful_hot"},
    {"text": "Wang Fang has a cat named Mochi.",                                   "role": "useful_hot"},
    {"text": "Wang Fang currently drives a Tesla Model 3.",                        "role": "useful_hot"},
    {"text": "Wang Fang's current direct manager is Director Zhao.",               "role": "useful_hot"},
    {"text": "Wang Fang is severely allergic to peanuts.",                         "role": "useful_hot"},
    {"text": "Wang Fang's active project is real-time fraud detection.",           "role": "useful_hot"},
    {"text": "Wang Fang prefers meetings scheduled in the early morning.",         "role": "useful_hot"},
    {"text": "Wang Fang's team at NovaBank has 6 engineers.",                      "role": "useful_hot"},

    # --- 5 条 过期干扰项 noise (和某条 useful 语义近, 但答案是旧的/错的) ---
    {"text": "Wang Fang used to work as a junior analyst at OldTrust Bank.",       "role": "noise", "trap_for": "job"},
    {"text": "Wang Fang lived in the Haidian district of Beijing years ago.",      "role": "noise", "trap_for": "location"},
    {"text": "Wang Fang briefly tried learning Java and C++ in college.",          "role": "noise", "trap_for": "langs"},
    {"text": "Wang Fang previously drove an old Volkswagen Passat.",               "role": "noise", "trap_for": "car"},
    {"text": "Wang Fang's former manager two years ago was Director Lin.",         "role": "noise", "trap_for": "manager"},

    # --- 15 条 纯噪声 (和查询基本无关, 测整体信噪) ---
    {"text": "The office coffee machine on floor 3 was replaced in March.",        "role": "noise"},
    {"text": "Liang Bo from the marketing team enjoys hiking on weekends.",        "role": "noise"},
    {"text": "The quarterly all-hands meeting is usually held in the atrium.",     "role": "noise"},
    {"text": "Someone left an umbrella in conference room B last Tuesday.",        "role": "noise"},
    {"text": "The cafeteria started serving vegan options recently.",             "role": "noise"},
    {"text": "Parking garage level 2 has been under maintenance.",                "role": "noise"},
    {"text": "The company picnic was rescheduled due to rain.",                    "role": "noise"},
    {"text": "A new printer was installed near the east stairwell.",              "role": "noise"},
    {"text": "The Wi-Fi password in the lobby changed last month.",               "role": "noise"},
    {"text": "Intern orientation happens on the first Monday of each quarter.",    "role": "noise"},
    {"text": "The rooftop terrace closes at 8pm on weekdays.",                     "role": "noise"},
    {"text": "Building security runs a fire drill every six months.",             "role": "noise"},
    {"text": "The vending machine prices went up by fifty cents.",                "role": "noise"},
    {"text": "A team-building escape room event is planned for next month.",       "role": "noise"},
    {"text": "The shuttle bus to the metro station leaves every 15 minutes.",      "role": "noise"},

    # --- 3 条 useful_cold (合法有用, 但从没被强化 + 很旧 -> 测衰减副作用) ---
    {"text": "Wang Fang graduated from Fudan University with a statistics degree.","role": "useful_cold"},
    {"text": "Wang Fang's employee badge number is NB-4471.",                      "role": "useful_cold"},
    {"text": "Wang Fang's emergency contact is her sister Wang Li.",               "role": "useful_cold"},
]

# ---- 10 个查询 ----
# 措辞和事实不同(测泛化); gold = 正确事实文本片段; kind 标注属于哪类
QUERIES = [
    {"q": "What is Wang Fang's current job?",                  "gold": "lead data scientist at NovaBank", "kind": "hot_trap"},
    {"q": "Where does Wang Fang live now?",                    "gold": "Pudong district of Shanghai",      "kind": "hot_trap"},
    {"q": "Which programming languages does she use?",         "gold": "Python and SQL",                   "kind": "hot_trap"},
    {"q": "What car does Wang Fang drive these days?",         "gold": "Tesla Model 3",                    "kind": "hot_trap"},
    {"q": "Who is Wang Fang's manager?",                       "gold": "Director Zhao",                    "kind": "hot_trap"},
    {"q": "Does Wang Fang have any pets?",                     "gold": "cat named Mochi",                  "kind": "hot"},
    {"q": "Any food allergies I should know about?",          "gold": "allergic to peanuts",              "kind": "hot"},
    {"q": "What is she working on right now?",                 "gold": "real-time fraud detection",        "kind": "hot"},
    {"q": "Which university did Wang Fang attend?",            "gold": "Fudan University",                 "kind": "cold"},
    {"q": "What is Wang Fang's emergency contact?",            "gold": "sister Wang Li",                   "kind": "cold"},
]


def build_store() -> FactStore:
    shutil.rmtree(STORE, ignore_errors=True)
    fs = FactStore(STORE)  # half_life 之后手动切换
    now = time.time()
    for spec in FACTS:
        fid = fs.add(spec["text"])
        f = next(x for x in fs.facts if x["id"] == fid)
        role = spec["role"]
        if role == "useful_hot":
            # 反复被采用 -> 高 trust; 最近用过 -> 新鲜
            f["trust"] = 0.92
            f["uses"] = 9
            f["last_used"] = int(now - 0.5 * DAY)
            f["ts"] = int(now - 40 * DAY)
        elif role == "useful_cold":
            # 合法但从没强化 + 很旧
            f["trust"] = 0.5
            f["uses"] = 0
            f["last_used"] = int(now - 40 * DAY)
            f["ts"] = int(now - 40 * DAY)
        else:  # noise
            f["trust"] = 0.5
            f["uses"] = 0
            f["last_used"] = int(now - 40 * DAY)
            f["ts"] = int(now - 40 * DAY)
    fs._save()
    return fs


def gold_rank(results, gold) -> int:
    """正确事实在结果中的名次(1-based), 没召回返回 999。"""
    g = gold.lower()
    for i, r in enumerate(results, 1):
        if g in r["text"].lower():
            return i
    return 999


def evaluate(fs: FactStore, half_life: float, top_k: int = 3):
    fs.half_life = half_life
    rank1 = 0
    hit3 = 0
    detail = []
    for tq in QUERIES:
        results = fs.retrieve(tq["q"], top_k=top_k, reinforce=False)
        rk = gold_rank(results, tq["gold"])
        is_r1 = rk == 1
        is_h3 = rk <= 3
        rank1 += is_r1
        hit3 += is_h3
        top1 = results[0]["text"][:45] if results else "(none)"
        detail.append((tq["kind"], tq["q"], rk, is_r1, top1))
    return rank1, hit3, detail


def main():
    print("=" * 70)
    print("  信任层(时间衰减)消融实验  ON(half_life=14) vs OFF(half_life=0)")
    print("=" * 70)
    fs = build_store()
    n = len(QUERIES)

    print(f"\n  事实总数: {len(fs.facts)} (10 useful_hot + 5 过期干扰 + 15 噪声 + 3 useful_cold)")
    print(f"  查询总数: {n}\n")

    on_r1, on_h3, on_detail = evaluate(fs, half_life=14.0)
    off_r1, off_h3, off_detail = evaluate(fs, half_life=0.0)

    # 逐查询对比
    print(f"  {'kind':<10}{'query':<42}{'OFF rk':>7}{'ON rk':>7}")
    print(f"  {'-'*10}{'-'*42}{'-'*7}{'-'*7}")
    for (k, q, *_), (_, _, off_rk, *_), (_, _, on_rk, *_) in zip(
        [(d[0], d[1]) for d in on_detail], off_detail, on_detail
    ):
        mark = ""
        if off_rk != on_rk:
            mark = "  <-- 变化" if on_rk < off_rk else "  <-- 变差"
        print(f"  {k:<10}{q[:40]:<42}{off_rk:>7}{on_rk:>7}{mark}")

    print("\n" + "=" * 70)
    print(f"  {'指标':<22}{'OFF(衰减关)':>14}{'ON(衰减开)':>14}{'Δ':>8}")
    print(f"  {'-'*22}{'-'*14}{'-'*14}{'-'*8}")
    print(f"  {'rank-1 准确率':<22}{f'{off_r1}/{n} ({100*off_r1/n:.0f}%)':>14}{f'{on_r1}/{n} ({100*on_r1/n:.0f}%)':>14}{on_r1-off_r1:>+8}")
    print(f"  {'hit@3 准确率':<22}{f'{off_h3}/{n} ({100*off_h3/n:.0f}%)':>14}{f'{on_h3}/{n} ({100*on_h3/n:.0f}%)':>14}{on_h3-off_h3:>+8}")

    print("\n  结论:")
    if on_r1 > off_r1:
        print(f"    时间衰减提升 rank-1 准确率 +{on_r1-off_r1} 条 -> 帮忙 (压下了过期干扰项)")
    elif on_r1 < off_r1:
        print(f"    时间衰减降低 rank-1 准确率 {on_r1-off_r1} 条 -> 帮倒忙 (埋掉了冷门正确答案)")
    else:
        print("    时间衰减对 rank-1 无变化")
    if on_h3 != off_h3:
        print(f"    hit@3 变化 {on_h3-off_h3:+d} 条")
    else:
        print("    hit@3 无变化")

    # 分桶看: hot_trap(衰减该帮) vs cold(衰减可能害)
    def bucket(detail, kinds):
        return sum(1 for d in detail if d[0] in kinds and d[3])  # is_r1
    for tag, kinds in [("hot_trap(有过期干扰)", {"hot_trap"}), ("hot(无干扰)", {"hot"}), ("cold(冷门正确答案)", {"cold"})]:
        tot = sum(1 for q in QUERIES if q["kind"] in kinds)
        print(f"    [{tag}] rank-1: OFF {bucket(off_detail,kinds)}/{tot}  ON {bucket(on_detail,kinds)}/{tot}")

    print("\nDone!")


if __name__ == "__main__":
    main()
