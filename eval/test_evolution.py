"""纵向进化 Eval: 200轮连续交互, 验证系统是否"随时间变好"。

设计(基于 Linghao "Evolving Memory Systems" 方法论):
- DeepSeek 扮演模拟用户(有人设档案, 会逐步透露/修改/矛盾信息)
- 200轮中按阶段推进:
    Phase 1 (1-50):   基础信息透露
    Phase 2 (51-100): 新信息+部分矛盾
    Phase 3 (101-150): 重复问某些事(测强化)
    Phase 4 (151-200): 问很久没提的事(测衰减)
- 每隔 20 轮插入"考试"(checkpoint): 问之前的事实, 评分

对比:
    A = memory-engine (RAG + 信任层)
    B = 裸 RAG (去掉信任, 所有记忆同等权重)

如果 A 在后期 checkpoint 分数 > B → 信任层产生可测量的"进化优势"。
"""

import sys
import os
import json
import time
import shutil
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_engine.fact_store import FactStore
from memory_engine.deepseek_client import DeepSeekClient

# ---- 模拟用户档案 ----
USER_PROFILE = """
Name: Chen Lei
Job: Product manager at a health-tech startup (MediFlow)
City: Chengdu
Languages: Python, SQL
Hobbies: cycling, cooking Sichuan food
Pet: a golden retriever named Dou Dou
Partner: married to Li Jing (teacher)
Car: Tesla Model 3
Favorite tool: Figma for design, Notion for docs
Dream: launch own startup in 2 years
"""

# ---- 200 轮交互脚本 ----
# 每轮: {"turn": n, "type": "normal"|"checkpoint", "user_msg": ..., "fact_to_store": ..., "checkpoint_q": ..., "gold": ...}
SCRIPT = [
    # Phase 1: 基础信息 (1-50)
    {"turn": 1, "type": "normal", "user_msg": "Hey, I'm Chen Lei. I work as a product manager.", "fact": "The user's name is Chen Lei, a product manager."},
    {"turn": 3, "type": "normal", "user_msg": "I'm at a health-tech startup called MediFlow.", "fact": "Chen Lei works at MediFlow, a health-tech startup."},
    {"turn": 6, "type": "normal", "user_msg": "I live in Chengdu, great food here.", "fact": "Chen Lei lives in Chengdu."},
    {"turn": 9, "type": "normal", "user_msg": "I use Python and SQL mostly for data stuff.", "fact": "Chen Lei uses Python and SQL."},
    {"turn": 12, "type": "normal", "user_msg": "My dog Dou Dou is a golden retriever, loves the park.", "fact": "Chen Lei has a golden retriever named Dou Dou."},
    {"turn": 15, "type": "normal", "user_msg": "My wife Li Jing is a middle school teacher.", "fact": "Chen Lei's wife Li Jing is a middle school teacher."},
    {"turn": 18, "type": "normal", "user_msg": "I drive a Tesla Model 3, got it last year.", "fact": "Chen Lei drives a Tesla Model 3."},
    {"turn": 20, "type": "checkpoint", "q": "What is the user's name and job?", "gold": ["chen lei", "product manager"]},
    {"turn": 22, "type": "normal", "user_msg": "For design I use Figma, for docs Notion.", "fact": "Chen Lei uses Figma for design and Notion for docs."},
    {"turn": 25, "type": "normal", "user_msg": "I love cycling on weekends, and cooking Sichuan food.", "fact": "Chen Lei enjoys cycling and cooking Sichuan food."},
    {"turn": 28, "type": "normal", "user_msg": "My dream is to launch my own startup within 2 years.", "fact": "Chen Lei dreams of launching his own startup in 2 years."},
    {"turn": 30, "type": "normal", "user_msg": "Our main product is a patient scheduling system.", "fact": "MediFlow's main product is a patient scheduling system."},
    {"turn": 35, "type": "normal", "user_msg": "I manage a team of 5 engineers and 2 designers.", "fact": "Chen Lei manages 5 engineers and 2 designers."},
    {"turn": 40, "type": "checkpoint", "q": "What pet does the user have?", "gold": ["dou dou", "golden retriever"]},
    {"turn": 42, "type": "normal", "user_msg": "We use AWS for our backend infrastructure.", "fact": "MediFlow uses AWS."},
    {"turn": 45, "type": "normal", "user_msg": "I'm reading 'Inspired' by Marty Cagan right now.", "fact": "Chen Lei is reading 'Inspired' by Marty Cagan."},
    {"turn": 48, "type": "normal", "user_msg": "My commute is 15 minutes by bike.", "fact": "Chen Lei's commute is 15 minutes by bike."},
    {"turn": 50, "type": "normal", "user_msg": "Our team standup is at 10am every day.", "fact": "MediFlow team standup is at 10am daily."},

    # Phase 2: 矛盾/更新 (51-100)
    {"turn": 55, "type": "normal", "user_msg": "Actually we just switched from AWS to Aliyun last month.", "fact": "MediFlow switched from AWS to Aliyun (recent change)."},
    {"turn": 60, "type": "checkpoint", "q": "What cloud provider does the user's company use?", "gold": ["aliyun"]},
    {"turn": 63, "type": "normal", "user_msg": "I sold the Tesla and bought a BYD Seal. Better range.", "fact": "Chen Lei now drives a BYD Seal (sold the Tesla)."},
    {"turn": 67, "type": "normal", "user_msg": "We hired 3 more engineers, team is 8 now.", "fact": "Chen Lei's engineering team grew to 8."},
    {"turn": 70, "type": "normal", "user_msg": "Started learning Go recently for our new microservices.", "fact": "Chen Lei is learning Go for microservices."},
    {"turn": 75, "type": "normal", "user_msg": "Li Jing quit teaching and joined a edtech startup.", "fact": "Li Jing left teaching, joined an edtech startup."},
    {"turn": 80, "type": "checkpoint", "q": "What car does the user drive now?", "gold": ["byd", "seal"]},
    {"turn": 83, "type": "normal", "user_msg": "I switched from Notion to Obsidian for personal notes.", "fact": "Chen Lei switched from Notion to Obsidian for personal notes."},
    {"turn": 88, "type": "normal", "user_msg": "Dou Dou had puppies! We're keeping one named Xiao Bai.", "fact": "Dou Dou had puppies, keeping one named Xiao Bai."},
    {"turn": 92, "type": "normal", "user_msg": "I finished reading Inspired, now reading Zero to One.", "fact": "Chen Lei finished Inspired, now reading Zero to One."},
    {"turn": 95, "type": "normal", "user_msg": "Thinking about moving to Shenzhen for better startup ecosystem.", "fact": "Chen Lei is considering moving to Shenzhen."},
    {"turn": 100, "type": "checkpoint", "q": "What is Li Jing's current job?", "gold": ["edtech", "startup"]},

    # Phase 3: 重复问题(测强化) (101-150)
    {"turn": 105, "type": "normal", "user_msg": "Can you remind me what tools I use for design?", "fact": None},
    {"turn": 110, "type": "normal", "user_msg": "What was our cloud provider again?", "fact": None},
    {"turn": 115, "type": "normal", "user_msg": "Tell me about my pets.", "fact": None},
    {"turn": 120, "type": "checkpoint", "q": "Name all the user's pets.", "gold": ["dou dou", "xiao bai"]},
    {"turn": 125, "type": "normal", "user_msg": "What languages do I code in these days?", "fact": None},
    {"turn": 130, "type": "normal", "user_msg": "Remind me what my wife does now?", "fact": None},
    {"turn": 135, "type": "normal", "user_msg": "What car do I drive?", "fact": None},
    {"turn": 140, "type": "checkpoint", "q": "What programming languages does the user know?", "gold": ["python", "sql", "go"]},
    {"turn": 145, "type": "normal", "user_msg": "What's my team size now?", "fact": None},
    {"turn": 150, "type": "normal", "user_msg": "What's the name of my startup?", "fact": None},

    # Phase 4: 问很久没提的事(测衰减对比) (151-200)
    {"turn": 155, "type": "normal", "user_msg": "I've been too busy to cycle lately, picked up swimming instead.", "fact": "Chen Lei stopped cycling, does swimming now."},
    {"turn": 160, "type": "checkpoint", "q": "What is the user's commute like?", "gold": ["15 minutes", "bike"]},
    {"turn": 165, "type": "normal", "user_msg": "We pivoted from scheduling to AI diagnosis assistant.", "fact": "MediFlow pivoted to AI diagnosis assistant."},
    {"turn": 170, "type": "normal", "user_msg": "I got promoted to VP of Product.", "fact": "Chen Lei promoted to VP of Product."},
    {"turn": 175, "type": "normal", "user_msg": "Decided to stay in Chengdu actually, Shenzhen plan dropped.", "fact": "Chen Lei decided to stay in Chengdu (dropped Shenzhen plan)."},
    {"turn": 180, "type": "checkpoint", "q": "What is the user's dream/career goal?", "gold": ["startup", "own"]},
    {"turn": 185, "type": "normal", "user_msg": "Learning Rust now too, Go wasn't enough for our perf needs.", "fact": "Chen Lei is also learning Rust."},
    {"turn": 190, "type": "normal", "user_msg": "We're up to 15 engineers now, growing fast.", "fact": "MediFlow team grew to 15 engineers."},
    {"turn": 195, "type": "normal", "user_msg": "Xiao Bai grew up, Dou Dou is getting old though.", "fact": "Dou Dou is getting old, Xiao Bai grew up."},
    {"turn": 200, "type": "checkpoint", "q": "What does MediFlow's product do now?", "gold": ["ai diagnosis", "assistant"]},
]


def run_system(store_dir: str, use_trust: bool, ds: DeepSeekClient) -> dict:
    """Run full 200-turn eval on one system config."""
    shutil.rmtree(store_dir, ignore_errors=True)
    # 关键区别: use_trust 控制是否做信任加权
    fs = FactStore(store_dir, decay_half_life_days=7.0 if use_trust else 0.0)

    checkpoint_scores = []

    for step in SCRIPT:
        # 存入新事实
        if step.get("fact"):
            fs.add(step["fact"])

        # 普通轮: 模拟"检索并使用"(触发 reinforce)
        if step["type"] == "normal" and not step.get("fact"):
            # 用户在问之前的事 → 触发检索(测强化)
            fs.retrieve(step["user_msg"], top_k=3, reinforce=use_trust)

        # 考试轮
        if step["type"] == "checkpoint":
            q = step["q"]
            gold = step["gold"]
            # 检索
            results = fs.retrieve(q, top_k=5, reinforce=use_trust)
            recalled_text = " ".join(r["text"].lower() for r in results)
            # 判分: gold 关键词是否在召回文本中
            hits = sum(1 for g in gold if g.lower() in recalled_text)
            score = hits / len(gold)
            checkpoint_scores.append({
                "turn": step["turn"],
                "q": q,
                "gold": gold,
                "score": score,
                "recalled": [r["text"][:40] for r in results[:3]],
            })

    return {
        "checkpoints": checkpoint_scores,
        "avg_score": sum(c["score"] for c in checkpoint_scores) / len(checkpoint_scores),
        "scores_by_turn": [(c["turn"], c["score"]) for c in checkpoint_scores],
    }


def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        # 这个 eval 不需要 DeepSeek(纯检索层), 但保留以便后续扩展
        pass

    ds = None  # 纯检索 eval, 不调 LLM

    print("=" * 60)
    print("  纵向进化 Eval (200轮, 信任层 vs 裸RAG)")
    print("  (基于 Linghao 'Evolving Memory Systems' 方法论)")
    print("=" * 60)

    # System A: memory-engine (RAG + 信任层)
    print("\n[A] RAG + 信任层 (trust, reinforce, decay half_life=7d)")
    result_a = run_system("./_evo_trust", use_trust=True, ds=ds)
    print(f"    平均分: {result_a['avg_score']:.2%}")
    for turn, score in result_a["scores_by_turn"]:
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"    Turn {turn:>3}: {bar} {score:.0%}")

    # System B: 裸 RAG (无信任, 无强化, 无衰减)
    print("\n[B] 裸 RAG (无信任层, 所有记忆同等权重)")
    result_b = run_system("./_evo_raw", use_trust=False, ds=ds)
    print(f"    平均分: {result_b['avg_score']:.2%}")
    for turn, score in result_b["scores_by_turn"]:
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"    Turn {turn:>3}: {bar} {score:.0%}")

    # 对比
    print("\n" + "=" * 60)
    print("  对比: 信任层 vs 裸RAG (每个 checkpoint)")
    print("=" * 60)
    print(f"  {'Turn':<6} {'信任层':<10} {'裸RAG':<10} {'差异'}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    diffs = []
    for a, b in zip(result_a["checkpoints"], result_b["checkpoints"]):
        diff = a["score"] - b["score"]
        diffs.append(diff)
        marker = "+" if diff > 0 else ("-" if diff < 0 else "=")
        print(f"  {a['turn']:<6} {a['score']:<10.0%} {b['score']:<10.0%} {marker}{abs(diff):.0%}")

    avg_diff = sum(diffs) / len(diffs)
    print(f"\n  整体: 信任层 {result_a['avg_score']:.1%} vs 裸RAG {result_b['avg_score']:.1%} (差异 {avg_diff:+.1%})")

    # 按阶段分析
    phases = {"Phase1(基础)": [0,1], "Phase2(矛盾)": [2,3,4], "Phase3(强化)": [5,6], "Phase4(衰减)": [7,8,9]}
    print("\n  按阶段:")
    for phase_name, indices in phases.items():
        valid = [i for i in indices if i < len(diffs)]
        if valid:
            phase_a = sum(result_a["checkpoints"][i]["score"] for i in valid) / len(valid)
            phase_b = sum(result_b["checkpoints"][i]["score"] for i in valid) / len(valid)
            print(f"    {phase_name:<16}: 信任{phase_a:.0%} vs 裸{phase_b:.0%} (差{phase_a-phase_b:+.0%})")

    verdict = "信任层有进化优势" if avg_diff > 0.05 else ("无显著差异" if abs(avg_diff) <= 0.05 else "裸RAG反而更好")
    print(f"\n  结论: {verdict}")

    # 保存
    out = Path(__file__).parent / "evolution_eval_results.json"
    out.write_text(json.dumps({"trust": result_a, "raw": result_b, "avg_diff": avg_diff},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved to {out}")
    print("\nDone!")


if __name__ == "__main__":
    main()
