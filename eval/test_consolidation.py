"""睡眠巩固测试: cleanup + merge + reweight。"""

import sys
import os
import shutil
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_engine.fact_store import FactStore
from memory_engine.consolidator import SleepConsolidator
from memory_engine.deepseek_client import DeepSeekClient


def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: set DEEPSEEK_API_KEY"); sys.exit(1)

    print("=" * 55)
    print("  睡眠巩固测试")
    print("=" * 55)

    shutil.rmtree("./_sleep_test", ignore_errors=True)
    fs = FactStore("./_sleep_test")
    ds = DeepSeekClient()
    sc = SleepConsolidator(fs, deepseek=ds)

    # 准备数据: 正常 + 垃圾 + 冗余
    print("\n[1] 写入测试数据...")
    # 正常记忆
    fs.add("Chen Lei works as a product manager at MediFlow.")
    fs.add("Chen Lei lives in Chengdu.")
    fs.add("Chen Lei drives a BYD Seal.")
    fs.add("Chen Lei's dog is named Dou Dou.")

    # 冗余记忆(说同一件事的多条)
    fs.add("Chen Lei is a PM at a health-tech company called MediFlow.")
    fs.add("Chen Lei is the product manager of MediFlow, a healthcare startup.")

    # 垃圾(superseded + 低trust)
    fid_old = fs.add("Chen Lei drives a Tesla Model 3.")
    for f in fs.facts:
        if f["id"] == fid_old:
            f["trust"] = 0.03
            f["superseded_by"] = 3
            break

    fid_old2 = fs.add("Chen Lei lives in Beijing.")
    for f in fs.facts:
        if f["id"] == fid_old2:
            f["trust"] = 0.02
            f["superseded_by"] = 2
            break
    fs._save()

    print(f"    写入 {len(fs.facts)} 条 (4正常 + 2冗余 + 2垃圾)")

    # [2] dry_run 先看看会做什么
    print("\n[2] dry_run (只报告不执行)...")
    r = sc.consolidate(mode="full", dry_run=True)
    print(f"    将清理: {r['cleaned']} 条")
    print(f"    将合并: {r['merged']} 组")
    print(f"    将重算: {r['reweighted']} 条")

    # [3] 执行清理
    print("\n[3] 执行 cleanup...")
    r = sc.consolidate(mode="cleanup")
    print(f"    删除了 {r['cleaned']} 条垃圾")
    print(f"    剩余: {len(fs.facts)} 条")

    # 验证垃圾被删
    texts = [f["text"] for f in fs.facts]
    assert "Tesla" not in str(texts), "Tesla should be cleaned"
    assert "Beijing" not in str(texts), "Beijing should be cleaned"
    print("    OK: 垃圾已清理(Tesla/Beijing 消失)")

    # [4] 执行合并
    print("\n[4] 执行 merge...")
    before_count = len(fs.facts)
    r = sc.consolidate(mode="merge")
    after_count = len(fs.facts)
    print(f"    合并了 {r['merged']} 组, 记忆数 {before_count} -> {after_count}")
    if r["details"]:
        for d in r["details"]:
            if d.get("merged_into") and d["action"] == "merge":
                print(f"    合并结果: '{d['merged_into'][:60]}'")
                print(f"    原始: {d.get('group', [])}")

    # [5] 执行 reweight
    print("\n[5] 执行 reweight...")
    # 先模拟使用模式: 给某条记忆加 uses 和 recent last_used
    for f in fs.facts:
        if "Chengdu" in f["text"]:
            f["uses"] = 8
            f["last_used"] = int(time.time())
        elif "Dou Dou" in f["text"]:
            f["uses"] = 0
            f["last_used"] = int(time.time()) - 25 * 86400  # 25天没用
    fs._save()

    r = sc.consolidate(mode="reweight")
    print(f"    重算了 {r['reweighted']} 条 trust")
    for d in r["details"][:5]:
        fact = next((f for f in fs.facts if f["id"] == d["id"]), None)
        name = fact["text"][:30] if fact else "?"
        print(f"    #{d['id']} {d['old_trust']:.2f}->{d['new_trust']:.2f} (rec={d['recency']}, freq={d['frequency']}) | {name}")

    # Chengdu 应该 trust 上升(常用+新鲜), Dou Dou 应该下降(不用+老)
    chengdu = next((f for f in fs.facts if "Chengdu" in f["text"]), None)
    doudou = next((f for f in fs.facts if "Dou Dou" in f["text"]), None)
    if chengdu and doudou:
        print(f"\n    验证: Chengdu trust={chengdu['trust']:.2f}(应>0.5), Dou Dou trust={doudou['trust']:.2f}(应<0.5)")
        assert chengdu["trust"] > 0.5, f"Chengdu should be high, got {chengdu['trust']}"
        assert doudou["trust"] < 0.5, f"Dou Dou should decay, got {doudou['trust']}"
        print("    OK: 常用的变强, 久不用的变弱")

    # 最终状态
    print(f"\n[6] 最终状态: {len(fs.facts)} 条记忆")
    for f in fs.facts:
        print(f"    #{f['id']} trust={f.get('trust',0.5):.2f} uses={f.get('uses',0)} | {f['text'][:50]}")

    print("\n全部通过!")


if __name__ == "__main__":
    main()
