"""压力测试: 100+条真实记忆, 含矛盾/更新/兼容, 验证误报率和 trust 稳定性。

用真实自然语言(非模板), 模拟一个用户半年的记忆积累:
- 80 条基础事实
- 15 条更新(替代旧事实)
- 10 条兼容补充(不该触发矛盾)
- 验证: 误报率、正确检测率、trust 分布健康度
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
from memory_engine.health_monitor import MemoryHealthMonitor
from memory_engine.deepseek_client import DeepSeekClient

# ---- 真实自然语言测试数据 ----

BASE_FACTS = [
    "Chen Lei is 28 years old.",
    "Chen Lei works as a product manager at MediFlow.",
    "Chen Lei lives in Chengdu, Sichuan province.",
    "Chen Lei graduated from Sichuan University in 2019.",
    "Chen Lei's wife Li Jing teaches at a middle school.",
    "Chen Lei has a golden retriever named Dou Dou.",
    "Chen Lei drives a Tesla Model 3.",
    "Chen Lei's favorite restaurant is a Sichuan hotpot place near his home.",
    "Chen Lei uses Figma for design work.",
    "Chen Lei uses Notion for documentation.",
    "Chen Lei's team has 5 engineers and 2 designers.",
    "MediFlow's main product is a patient scheduling system.",
    "Chen Lei prefers Python for scripting tasks.",
    "Chen Lei uses SQL for data analysis.",
    "Chen Lei's office is on the 12th floor.",
    "Chen Lei commutes by bike, about 15 minutes each way.",
    "Chen Lei enjoys rock climbing on weekends.",
    "Chen Lei listens to jazz while working.",
    "Chen Lei reads sci-fi novels before bed.",
    "Chen Lei's morning routine starts with black coffee.",
    "Chen Lei's phone is a Pixel 8.",
    "Chen Lei uses dark mode on all devices.",
    "Chen Lei's monitor setup is dual 27-inch 4K.",
    "Chen Lei's keyboard has cherry brown switches.",
    "Chen Lei's preferred cloud is AWS.",
    "Chen Lei attended KubeCon 2023 in Shanghai.",
    "Chen Lei's mentor is Professor Wang at Sichuan University.",
    "Chen Lei's best friend Zhao Ming works at Tencent.",
    "Chen Lei's sister Chen Mei is a nurse in Chongqing.",
    "Chen Lei's father is a retired engineer.",
    "Chen Lei's sprint planning is every Monday at 10am.",
    "MediFlow uses GitHub for version control.",
    "Chen Lei's standup is at 9:30am daily.",
    "Chen Lei's lunch break is 12:00-13:30.",
    "Chen Lei usually eats at the company cafeteria.",
    "Chen Lei's salary review is in March every year.",
    "Chen Lei's vacation days are 15 per year.",
    "Chen Lei went to Japan for vacation last summer.",
    "Chen Lei's dream is to start his own company within 2 years.",
    "Chen Lei is considering an MBA program.",
    "Chen Lei's monthly mortgage payment is 8000 yuan.",
    "Chen Lei's apartment is 90 square meters.",
    "Chen Lei's parking spot is B2-156.",
    "Chen Lei's car insurance expires in November.",
    "Chen Lei's gym membership is at the building's basement gym.",
    "Chen Lei runs 5km every Sunday morning.",
    "Chen Lei's weight is 72kg.",
    "Chen Lei is 175cm tall.",
    "Chen Lei wears glasses for computer work.",
    "Chen Lei's blood type is A positive.",
    "Chen Lei is allergic to shrimp.",
    "Chen Lei's emergency contact is his wife Li Jing.",
    "Chen Lei's WeChat ID is chenlei_pm.",
    "Chen Lei's GitHub username is chenlei-dev.",
    "Chen Lei started learning Go programming last month.",
    "Chen Lei's favorite podcast is about startups.",
    "Chen Lei subscribes to Harvard Business Review.",
    "Chen Lei's KPI this quarter is to reduce patient wait times by 20%.",
    "MediFlow has 50 employees total.",
    "MediFlow's office is in Chengdu High-tech Zone.",
    "Chen Lei's direct report Zhang Wei is the tech lead.",
    "Chen Lei's other direct report Liu Fang handles frontend.",
    "Chen Lei's 1-on-1 with his manager is Thursday 3pm.",
    "Chen Lei's manager is VP of Product, named Wang Lei.",
    "Chen Lei presented at a local PM meetup last month.",
    "Chen Lei's side project is a personal finance tracker app.",
    "Chen Lei uses Obsidian for personal notes.",
    "Chen Lei's Kindle has 200+ books.",
    "Chen Lei's favorite author is Liu Cixin.",
    "Chen Lei watched Dune Part 2 three times.",
    "Chen Lei's Netflix subscription is the premium plan.",
    "Chen Lei's home WiFi password is shared with his neighbor.",
    "Chen Lei's router is an Asus RT-AX86U.",
    "Chen Lei backs up photos to Google Photos.",
    "Chen Lei's NAS is a Synology DS220+.",
    "Chen Lei's home office desk is a standing desk.",
    "Chen Lei uses a Herman Miller chair.",
    "Chen Lei's webcam is a Logitech C920.",
    "Chen Lei's headphones are Sony WH-1000XM5.",
    "Chen Lei prefers meetings under 30 minutes.",
]

# 15 条更新(应该触发 UPDATE, 旧的被降权)
UPDATES = [
    ("Chen Lei drives a Tesla Model 3.", "Chen Lei sold the Tesla and now drives a BYD Seal."),
    ("Chen Lei's wife Li Jing teaches at a middle school.", "Li Jing quit teaching and joined an edtech startup."),
    ("Chen Lei's team has 5 engineers and 2 designers.", "Chen Lei's team grew to 8 engineers and 3 designers."),
    ("Chen Lei uses Notion for documentation.", "Chen Lei switched from Notion to Obsidian for all docs."),
    ("Chen Lei's preferred cloud is AWS.", "MediFlow migrated from AWS to Aliyun last month."),
    ("Chen Lei lives in Chengdu, Sichuan province.", "Chen Lei relocated to Shenzhen for a new opportunity."),
    ("Chen Lei works as a product manager at MediFlow.", "Chen Lei got promoted to VP of Product at MediFlow."),
    ("MediFlow's main product is a patient scheduling system.", "MediFlow pivoted to an AI-powered diagnosis assistant."),
    ("Chen Lei's phone is a Pixel 8.", "Chen Lei upgraded to a Pixel 9 Pro."),
    ("Chen Lei's dream is to start his own company within 2 years.", "Chen Lei decided to stay at MediFlow long-term after the promotion."),
    ("Chen Lei started learning Go programming last month.", "Chen Lei is now proficient in Go and uses it daily."),
    ("Chen Lei's KPI this quarter is to reduce patient wait times by 20%.", "Chen Lei's new KPI is to achieve 100k DAU for the diagnosis tool."),
    ("Chen Lei's weight is 72kg.", "Chen Lei lost weight and is now 68kg."),
    ("Chen Lei's parking spot is B2-156.", "Chen Lei moved to parking spot A1-023 on a higher floor."),
    ("MediFlow has 50 employees total.", "MediFlow grew to 120 employees after Series B funding."),
]

# 10 条兼容补充(不该触发矛盾)
COMPATIBLE = [
    "Chen Lei also enjoys hiking on long weekends.",  # 补充hobby, 不矛盾climbing
    "Chen Lei's dog Dou Dou loves swimming in the river.",  # 补充pet info
    "Chen Lei is learning Japanese in his spare time.",  # 新技能,不矛盾Go
    "Chen Lei's wife Li Jing is pregnant with their first child.",  # 新info
    "Chen Lei bought a DJI drone for aerial photography.",  # 新purchase
    "Chen Lei's team adopted Jira for project tracking.",  # 补充工具
    "Chen Lei mentors two junior PMs.",  # 新角色
    "Chen Lei invested in a friend's restaurant.",  # 新info
    "Chen Lei's sister Chen Mei got married last month.",  # update family
    "Chen Lei started a weekly book club with colleagues.",  # 新activity
]


def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: set DEEPSEEK_API_KEY"); sys.exit(1)

    print("=" * 60)
    print("  压力测试: 100+条记忆 + 矛盾检测 + 健康度")
    print("=" * 60)

    shutil.rmtree("./_stress_test", ignore_errors=True)
    fs = FactStore("./_stress_test")
    ds = DeepSeekClient()
    cd = ConflictDetector(fs, deepseek=ds)
    hm = MemoryHealthMonitor(fs)

    # Phase 1: 写入 80 条基础事实(无矛盾)
    print(f"\n[1] 写入 {len(BASE_FACTS)} 条基础事实...")
    false_positives_base = 0
    for fact in BASE_FACTS:
        r = cd.add_with_conflict_check(fact, auto_resolve=True)
        if r["conflicts"]:
            false_positives_base += 1
            print(f"    误报! '{fact[:30]}' 触发了 {len(r['conflicts'])} 个冲突")
    print(f"    完成。误报: {false_positives_base}/{len(BASE_FACTS)} ({100*false_positives_base/len(BASE_FACTS):.0f}%)")

    # Phase 2: 写入 15 条更新(应该触发)
    print(f"\n[2] 写入 {len(UPDATES)} 条更新(应检测到矛盾)...")
    correct_detections = 0
    for old, new in UPDATES:
        r = cd.add_with_conflict_check(new, auto_resolve=True)
        detected = len(r["conflicts"]) > 0
        if detected:
            correct_detections += 1
        else:
            print(f"    漏报! '{new[:40]}' 未检测到与 '{old[:30]}' 的冲突")
    detection_rate = correct_detections / len(UPDATES)
    print(f"    检测率: {correct_detections}/{len(UPDATES)} ({detection_rate:.0%})")

    # Phase 3: 写入 10 条兼容(不应触发)
    print(f"\n[3] 写入 {len(COMPATIBLE)} 条兼容补充(不应触发矛盾)...")
    false_positives_compat = 0
    for fact in COMPATIBLE:
        r = cd.add_with_conflict_check(fact, auto_resolve=True)
        if r["conflicts"]:
            false_positives_compat += 1
            for c in r["conflicts"]:
                print(f"    误报! '{fact[:30]}' vs '{c['old_text'][:30]}' ({c['relation']}, sim={c['similarity']:.2f})")
    print(f"    误报: {false_positives_compat}/{len(COMPATIBLE)} ({100*false_positives_compat/len(COMPATIBLE):.0f}%)")

    # Phase 4: 健康度检查
    print(f"\n[4] 健康度检查...")
    report = hm.check()
    print(f"    总记忆: {report.total_facts}")
    print(f"    平均trust: {report.avg_trust:.2f}")
    print(f"    低trust占比: {report.low_trust_ratio:.0%}")
    print(f"    superseded占比: {report.superseded_ratio:.0%}")
    print(f"    相似度std: {report.similarity_std:.4f}")
    print(f"    健康: {report.healthy}")
    print(f"    告警: {report.alerts or '无'}")
    cleanup = hm.suggest_cleanup()
    print(f"    建议清理: {len(cleanup)} 条")

    # Phase 5: Trust 分布
    print(f"\n[5] Trust 分布...")
    trusts = [f.get("trust", 0.5) for f in fs.facts]
    bins = {"0-0.1": 0, "0.1-0.3": 0, "0.3-0.5": 0, "0.5": 0, ">0.5": 0}
    for t in trusts:
        if t < 0.1: bins["0-0.1"] += 1
        elif t < 0.3: bins["0.1-0.3"] += 1
        elif t < 0.5: bins["0.3-0.5"] += 1
        elif t == 0.5: bins["0.5"] += 1
        else: bins[">0.5"] += 1
    for k, v in bins.items():
        bar = "█" * (v // 2)
        print(f"    trust {k:>6}: {v:>3} {bar}")

    # 汇总
    print("\n" + "=" * 60)
    print("  压力测试汇总")
    print("=" * 60)
    total_fp = false_positives_base + false_positives_compat
    total_should_not = len(BASE_FACTS) + len(COMPATIBLE)
    print(f"  总误报率: {total_fp}/{total_should_not} ({100*total_fp/total_should_not:.1f}%)")
    print(f"  正确检测率: {correct_detections}/{len(UPDATES)} ({detection_rate:.0%})")
    print(f"  健康状态: {'健康' if report.healthy else '需注意'}")
    print(f"  建议清理: {len(cleanup)} 条")

    verdict = "PASS" if (total_fp/total_should_not < 0.1 and detection_rate >= 0.6) else "NEEDS IMPROVEMENT"
    print(f"\n  判定: {verdict}")
    print(f"    (阈值: 误报<10%, 检测>=60%)")

    # 保存
    out = Path(__file__).parent / "stress_test_results.json"
    out.write_text(json.dumps({
        "base_false_positives": false_positives_base,
        "update_detection_rate": detection_rate,
        "compatible_false_positives": false_positives_compat,
        "total_false_positive_rate": total_fp / total_should_not,
        "health": {"healthy": report.healthy, "alerts": report.alerts},
        "trust_distribution": bins,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved to {out}")
    print("\nDone!")


if __name__ == "__main__":
    main()
