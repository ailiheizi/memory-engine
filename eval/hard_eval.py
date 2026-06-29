"""硬核评测集: 大规模 + 难样本, 把 baseline 压到中间区间。

之前评测的问题: N=10, baseline 90-100% 近天花板, 检测不出功能差异。

本评测的难度设计(让裸RAG做不到满分):
1. 强干扰: 200条记忆里有大量相似但不同的(10个人的工作/10个项目/...)
2. 多跳查询: 需要组合2+条记忆才能答(测图)
3. 时序矛盾: 旧值vs新值, 必须返回新的(测矛盾检测)
4. 深度改写: 查询用完全不同措辞(测query expansion)
5. 噪音稀释: 大量无关记忆稀释信号(测信任/聚类)

每个功能用同一个评测集, 开/关对比, 看谁真能把分数从 baseline 拉上去。
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
from memory_engine.deepseek_client import DeepSeekClient


# ============ 生成硬核数据 ============

def build_hard_dataset():
    """构造200+条带强干扰的记忆 + 难查询。"""
    facts = []
    queries = []

    # --- 强干扰组: 10个同事, 每人5个属性(50条高度相似) ---
    colleagues = [
        ("Zhang Wei", "backend", "Go", "Chengdu", "payment team"),
        ("Li Ming", "frontend", "React", "Beijing", "growth team"),
        ("Wang Fang", "data", "Python", "Shanghai", "analytics team"),
        ("Liu Yang", "mobile", "Swift", "Shenzhen", "app team"),
        ("Chen Hao", "devops", "Rust", "Hangzhou", "infra team"),
        ("Zhao Lei", "ML", "PyTorch", "Guangzhou", "AI team"),
        ("Sun Jie", "QA", "Selenium", "Wuhan", "quality team"),
        ("Zhou Min", "design", "Figma", "Chengdu", "UX team"),
        ("Wu Gang", "security", "C++", "Beijing", "security team"),
        ("Xu Tao", "backend", "Java", "Shanghai", "order team"),
    ]
    for name, role, lang, city, team in colleagues:
        facts.append(f"{name} is a {role} engineer.")
        facts.append(f"{name} primarily codes in {lang}.")
        facts.append(f"{name} is based in {city}.")
        facts.append(f"{name} works on the {team}.")
        facts.append(f"{name} joined the company in 202{colleagues.index((name,role,lang,city,team))%5}.")
    # 针对干扰组的精确查询(测能否在干扰中找对人)
    queries.append({"q": "What programming language does Wang Fang use?", "gold": ["Python"], "anti": ["Go", "React", "Swift"], "type": "distractor"})
    queries.append({"q": "Which city is Chen Hao based in?", "gold": ["Hangzhou"], "anti": ["Chengdu", "Beijing"], "type": "distractor"})
    queries.append({"q": "What team does Zhao Lei work on?", "gold": ["AI team"], "anti": ["payment", "growth"], "type": "distractor"})
    queries.append({"q": "What is Wu Gang's specialty?", "gold": ["security"], "anti": ["backend", "frontend"], "type": "distractor"})

    # --- 时序矛盾组: 一个主角的属性被多次更新 ---
    facts.append("The user works at StartupA as an intern.")          # 旧
    facts.append("The user got promoted to junior engineer at StartupA.")  # 中
    facts.append("The user left StartupA and joined BigCorp as senior engineer.")  # 新(最终)
    facts.append("The user's salary was 10k per month.")             # 旧
    facts.append("The user's salary increased to 25k per month at BigCorp.")  # 新
    facts.append("The user lived in a rented apartment downtown.")    # 旧
    facts.append("The user bought a house in the suburbs.")           # 新
    queries.append({"q": "Where does the user work now?", "gold": ["BigCorp"], "anti": ["StartupA", "intern"], "type": "temporal"})
    queries.append({"q": "What is the user's current salary?", "gold": ["25k"], "anti": ["10k"], "type": "temporal"})
    queries.append({"q": "What is the user's housing situation?", "gold": ["house", "suburbs"], "anti": ["rented", "downtown"], "type": "temporal"})

    # --- 多跳组: 需要组合多条才能答 ---
    facts.append("The user's manager is Director Sun.")
    facts.append("Director Sun previously led the cloud division.")
    facts.append("The cloud division was acquired by Oracle in 2024.")
    facts.append("The user's mentor is Professor Lin.")
    facts.append("Professor Lin teaches at Tsinghua University.")
    facts.append("Tsinghua University is located in Beijing.")
    queries.append({"q": "Which company acquired the division the user's manager used to lead?", "gold": ["Oracle"], "anti": [], "type": "multihop"})
    queries.append({"q": "In which city does the user's mentor teach?", "gold": ["Beijing"], "anti": [], "type": "multihop"})

    # --- 深度改写组: 查询措辞完全不同 ---
    facts.append("The user is allergic to peanuts and shellfish.")
    facts.append("The user practices intermittent fasting, eating only between noon and 8pm.")
    facts.append("The user has a phobia of public speaking.")
    facts.append("The user is fluent in Mandarin, English, and basic Japanese.")
    queries.append({"q": "What foods should I avoid serving the user?", "gold": ["peanut", "shellfish"], "anti": [], "type": "reformulated"})
    queries.append({"q": "When is the user willing to eat?", "gold": ["noon", "8pm"], "anti": [], "type": "reformulated"})
    queries.append({"q": "What makes the user nervous?", "gold": ["public speaking"], "anti": [], "type": "reformulated"})
    queries.append({"q": "Can the user communicate in Asian languages?", "gold": ["Mandarin", "Japanese"], "anti": [], "type": "reformulated"})

    # --- 噪音填充: 100条无关记忆稀释信号 ---
    topics = ["weather", "movie", "recipe", "stock", "gadget", "book", "song", "game", "trip", "news"]
    for i in range(100):
        t = topics[i % len(topics)]
        facts.append(f"Random note {i}: a fact about {t} number {i} that is unrelated to queries.")

    return facts, queries


def score(results, q):
    text = " ".join(r["text"].lower() for r in results)
    gold_hit = all(g.lower() in text for g in q["gold"]) if q["gold"] else False
    anti_hit = any(a.lower() in text for a in q.get("anti", []))
    # 满分: 命中gold且没召回anti(旧值)
    if gold_hit and not anti_hit:
        return 1.0
    elif gold_hit and anti_hit:
        return 0.5  # 找到对的但也混入了错的
    else:
        return 0.0


def eval_baseline(fs, queries, top_k=5):
    """裸 RAG baseline。"""
    by_type = {}
    for q in queries:
        results = fs.retrieve(q["q"], top_k=top_k, reinforce=False)
        s = score(results, q)
        by_type.setdefault(q["type"], []).append(s)
    return by_type


def main():
    print("=" * 60)
    print("  硬核评测集 (baseline 难度校准)")
    print("=" * 60)

    facts, queries = build_hard_dataset()
    print(f"\n  记忆: {len(facts)}条 (50强干扰 + 7时序矛盾 + 6多跳 + 4深度改写 + 100噪音)")
    print(f"  查询: {len(queries)}条, 分{len(set(q['type'] for q in queries))}类")

    # 建库
    shutil.rmtree("./_hardeval", ignore_errors=True)
    fs = FactStore("./_hardeval")
    print("\n  写入记忆中...")
    for text in facts:
        fs.add(text)

    # 跑 baseline
    print("\n  裸 RAG baseline:")
    by_type = eval_baseline(fs, queries)
    print(f"  {'类型':<14} {'分数':<10} {'说明'}")
    print(f"  {'-'*14} {'-'*10} {'-'*20}")
    type_desc = {"distractor": "强干扰中找对人", "temporal": "时序矛盾取新值",
                 "multihop": "多跳组合推理", "reformulated": "深度措辞改写"}
    all_scores = []
    for t, scores in by_type.items():
        avg = sum(scores) / len(scores)
        all_scores.extend(scores)
        print(f"  {t:<14} {avg:<10.0%} {type_desc.get(t,'')}")
    overall = sum(all_scores) / len(all_scores)
    print(f"  {'-'*14} {'-'*10}")
    print(f"  {'OVERALL':<14} {overall:<10.0%}")

    print(f"\n  baseline 总分: {overall:.0%}")
    if overall >= 0.85:
        print("  ⚠️ baseline 仍然过高, 难度不够, 检测不出功能差异")
    elif overall <= 0.4:
        print("  ⚠️ baseline 过低, 可能数据或检索有问题")
    else:
        print(f"  ✅ baseline 在 {overall:.0%}, 中间区间, 适合检测功能提升")

    # 保存数据集供后续功能测试复用
    out = Path(__file__).parent / "hard_eval_dataset.json"
    out.write_text(json.dumps({"facts": facts, "queries": queries, "baseline_by_type": {t: sum(s)/len(s) for t,s in by_type.items()}, "baseline_overall": overall}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  数据集已保存: {out}")
    print("\nDone!")


if __name__ == "__main__":
    main()
