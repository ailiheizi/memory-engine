"""综合测试: 使用反馈 + 健康度监控 + 人格分区。"""

import sys
import os
import shutil
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_engine.fact_store import FactStore
from memory_engine.usage_feedback import UsageFeedback
from memory_engine.health_monitor import MemoryHealthMonitor
from memory_engine.partitioned_memory import PartitionedMemory


def test_usage_feedback():
    """#2 使用反馈: 被采纳才强化, 被忽略微降。"""
    print("\n[#2] 使用反馈循环")
    print("-" * 40)
    shutil.rmtree("./_fb_test", ignore_errors=True)
    fs = FactStore("./_fb_test")
    fs.add("The user lives in Hangzhou near West Lake.")
    fs.add("The user drives a BYD Seal.")
    fs.add("The user's favorite color is blue.")

    uf = UsageFeedback(fs)

    # 模拟: 检索到3条, 但回答只引用了 Hangzhou
    retrieved = fs.retrieve("Tell me about the user", top_k=3, reinforce=False)
    response = "You live in Hangzhou near the beautiful West Lake area."

    feedback = uf.compute_feedback(response, retrieved)
    print("  回答: 'You live in Hangzhou near West Lake...'")
    print("  反馈:")
    for fb in feedback:
        status = "ADOPTED" if fb["adopted"] else ("IGNORED" if fb["ignored"] else "neutral")
        print(f"    [{status}] sim={fb['similarity']:.2f}: {fb['text'][:40]}")

    # 应用反馈
    before = {f["id"]: f["trust"] for f in fs.facts}
    uf.apply_feedback(feedback)
    after = {f["id"]: f["trust"] for f in fs.facts}

    print("  Trust 变化:")
    for fid in before:
        f = next(x for x in fs.facts if x["id"] == fid)
        delta = after[fid] - before[fid]
        if abs(delta) > 0.001:
            print(f"    #{fid}: {before[fid]:.3f} -> {after[fid]:.3f} ({delta:+.3f}) | {f['text'][:30]}")

    # 验证: Hangzhou 被强化, color 被忽略(或不变)
    hangzhou = next(f for f in fs.facts if "Hangzhou" in f["text"])
    assert hangzhou["trust"] > 0.5, f"Hangzhou should be reinforced, got {hangzhou['trust']}"
    print("  OK: 被采纳的记忆(Hangzhou) trust 上升")


def test_health_monitor():
    """#3 健康度监控: 检测退化。"""
    print("\n[#3] 记忆健康度监控")
    print("-" * 40)
    shutil.rmtree("./_health_test", ignore_errors=True)
    fs = FactStore("./_health_test")

    # 写入一些正常记忆
    for i in range(20):
        fs.add(f"The user's fact number {i} is about topic {chr(65+i%26)}.")

    monitor = MemoryHealthMonitor(fs)
    report = monitor.check()
    print(f"  总记忆: {report.total_facts}")
    print(f"  平均trust: {report.avg_trust:.2f}")
    print(f"  相似度std: {report.similarity_std:.4f}")
    print(f"  健康: {report.healthy}")
    print(f"  告警: {report.alerts or '无'}")

    # 人为制造退化: 把一半标为 superseded + 低 trust
    for f in fs.facts[:10]:
        f["trust"] = 0.02
        f["superseded_by"] = 999
    fs._save()

    report2 = monitor.check()
    print(f"\n  [制造退化后]")
    print(f"  低trust占比: {report2.low_trust_ratio:.0%}")
    print(f"  superseded占比: {report2.superseded_ratio:.0%}")
    print(f"  健康: {report2.healthy}")
    print(f"  告警: {report2.alerts}")

    assert not report2.healthy, "Should detect degradation"
    print("  OK: 退化被正确检测")

    # 清理建议
    cleanup = monitor.suggest_cleanup()
    print(f"  建议清理: {len(cleanup)} 条")


def test_partitioned_memory():
    """#4 人格记忆分区: 隔离 + 共享。"""
    print("\n[#4] 人格记忆分区")
    print("-" * 40)
    shutil.rmtree("./_part_test", ignore_errors=True)
    pm = PartitionedMemory("./_part_test")

    # 写入共享事实
    pm.add("The user's name is Lin Wei.", shared=True)
    pm.add("The user works as a backend engineer.", shared=True)

    # 切换到 persona "formal"
    pm.switch_persona("formal")
    pm.add("The user prefers no emoji in messages.")  # persona-private
    pm.add("The user likes bullet-point responses.")  # persona-private

    # 切换到 persona "casual"
    pm.switch_persona("casual")
    pm.add("The user enjoys playful banter.")  # persona-private
    pm.add("The user likes emoji and informal tone.")  # persona-private

    # 测试隔离: formal 模式下不应该看到 casual 的记忆
    pm.switch_persona("formal")
    formal_results = pm.retrieve("How should I communicate?", top_k=5)
    formal_texts = [r["text"] for r in formal_results]
    print("  [formal 模式] 检索 'How should I communicate?':")
    for r in formal_results:
        print(f"    [{r['partition']}] {r['text'][:45]}")
    has_casual_leak = any("playful" in t or "emoji" in t for t in formal_texts
                          if "no emoji" not in t)
    print(f"  casual 记忆泄漏: {has_casual_leak} (应为 False)")

    # casual 模式下不应该看到 formal 的记忆
    pm.switch_persona("casual")
    casual_results = pm.retrieve("How should I communicate?", top_k=5)
    casual_texts = [r["text"] for r in casual_results]
    print("\n  [casual 模式] 检索 'How should I communicate?':")
    for r in casual_results:
        print(f"    [{r['partition']}] {r['text'][:45]}")
    has_formal_leak = any("bullet-point" in t or "no emoji" in t for t in casual_texts)
    print(f"  formal 记忆泄漏: {has_formal_leak} (应为 False)")

    # 共享事实两个模式都能看到
    pm.switch_persona("formal")
    shared_check = pm.retrieve("What is the user's name?", top_k=3)
    has_name = any("Lin Wei" in r["text"] for r in shared_check)
    pm.switch_persona("casual")
    shared_check2 = pm.retrieve("What is the user's name?", top_k=3)
    has_name2 = any("Lin Wei" in r["text"] for r in shared_check2)
    print(f"\n  共享事实可见: formal={has_name}, casual={has_name2} (都应为 True)")

    # 分区列表
    parts = pm.list_partitions()
    print(f"  分区: {parts}")

    assert not has_casual_leak, "casual should not leak into formal"
    assert not has_formal_leak, "formal should not leak into casual"
    assert has_name and has_name2, "shared facts should be visible in all personas"
    print("  OK: 隔离正确, 共享正确")


def main():
    print("=" * 55)
    print("  综合测试: #2反馈 + #3健康度 + #4分区")
    print("=" * 55)

    test_usage_feedback()
    test_health_monitor()
    test_partitioned_memory()

    print("\n" + "=" * 55)
    print("  全部通过!")
    print("=" * 55)


if __name__ == "__main__":
    main()
