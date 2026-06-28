"""矛盾检测测试: 验证写入时能否正确检测并处理矛盾记忆。

测试场景:
1. 硬矛盾: "住北京" -> "住上海" (互斥)
2. 软更新: "开特斯拉" -> "换了比亚迪" (替代)
3. 兼容: "会Python" + "也会Go" (共存)
4. 矛盾处理后信任变化
5. 矛盾后检索排序是否正确(新的浮顶, 旧的沉底)
"""

import sys
import os
import shutil
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_engine.fact_store import FactStore
from memory_engine.conflict_detector import ConflictDetector
from memory_engine.deepseek_client import DeepSeekClient


def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: set DEEPSEEK_API_KEY"); sys.exit(1)

    print("=" * 55)
    print("  矛盾检测测试")
    print("=" * 55)

    shutil.rmtree("./_conflict_test", ignore_errors=True)
    fs = FactStore("./_conflict_test")
    ds = DeepSeekClient()
    cd = ConflictDetector(fs, deepseek=ds)

    # 写入基础事实
    print("\n[1] 写入基础事实...")
    cd.add_with_conflict_check("Chen Lei lives in Beijing.", auto_resolve=True)
    cd.add_with_conflict_check("Chen Lei drives a Tesla Model 3.", auto_resolve=True)
    cd.add_with_conflict_check("Chen Lei uses Python for work.", auto_resolve=True)
    cd.add_with_conflict_check("Chen Lei's favorite food is hotpot.", auto_resolve=True)
    print(f"    写入 4 条, 无矛盾")

    # 测试硬矛盾
    print("\n[2] 硬矛盾: '住北京' -> '住上海'")
    r = cd.add_with_conflict_check("Chen Lei moved to Shanghai last month.", auto_resolve=True)
    print(f"    检测到 {len(r['conflicts'])} 个冲突:")
    for c in r["conflicts"]:
        print(f"      [{c['relation']}] sim={c['similarity']:.2f}: '{c['old_text'][:40]}'")
    # 验证旧记忆 trust 被降
    beijing = next((f for f in fs.facts if "Beijing" in f["text"]), None)
    if beijing:
        print(f"    旧记忆('Beijing') trust: {beijing['trust']:.2f} (应接近0)")

    # 测试软更新
    print("\n[3] 软更新: '开特斯拉' -> '换比亚迪'")
    r = cd.add_with_conflict_check("Chen Lei sold the Tesla and bought a BYD Seal.", auto_resolve=True)
    print(f"    检测到 {len(r['conflicts'])} 个冲突:")
    for c in r["conflicts"]:
        print(f"      [{c['relation']}] sim={c['similarity']:.2f}: '{c['old_text'][:40]}'")
    tesla = next((f for f in fs.facts if "Tesla" in f["text"]), None)
    if tesla:
        print(f"    旧记忆('Tesla') trust: {tesla['trust']:.2f} (应大幅降低)")

    # 测试兼容
    print("\n[4] 兼容: '会Python' + '也会Go'")
    r = cd.add_with_conflict_check("Chen Lei also uses Go for backend services.", auto_resolve=True)
    print(f"    检测到 {len(r['conflicts'])} 个冲突:")
    for c in r["conflicts"]:
        print(f"      [{c['relation']}] sim={c['similarity']:.2f}: '{c['old_text'][:40]}'")
    python_fact = next((f for f in fs.facts if "Python" in f["text"]), None)
    if python_fact:
        print(f"    旧记忆('Python') trust: {python_fact['trust']:.2f} (应保持0.5, 未被降)")

    # 验证检索排序
    print("\n[5] 矛盾后检索: 问'住哪'应该返回上海(新)而非北京(旧)")
    results = fs.retrieve("Where does Chen Lei live?", top_k=3)
    print("    召回:")
    for r in results:
        print(f"      trust={r.get('eff_trust',0):.2f} final={r.get('final',0):.3f}: {r['text'][:45]}")
    top = results[0]["text"] if results else ""
    shanghai_on_top = "shanghai" in top.lower()
    print(f"    上海排第一: {shanghai_on_top}")

    # 汇总
    print("\n" + "=" * 55)
    print("  汇总")
    print("=" * 55)
    print(f"    事实总数: {len(fs.facts)}")
    for f in fs.facts:
        sup = f" [superseded by #{f['superseded_by']}]" if f.get("superseded_by") else ""
        print(f"    #{f['id']} trust={f['trust']:.2f} | {f['text'][:45]}{sup}")

    print("\nDone!")


if __name__ == "__main__":
    main()
