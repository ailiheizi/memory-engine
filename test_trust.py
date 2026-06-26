"""信任层测试: 验证 Hermes 式 trust + 强化 + 衰减 + 信任加权召回。

不需要 DeepSeek, 只测 FactStore 的信任机制(快)。
"""

import sys
import time
import shutil
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from memory_engine.fact_store import FactStore, DEFAULT_TRUST


def main():
    print("=" * 55)
    print("  信任层测试 (Hermes 式 trust/强化/衰减)")
    print("=" * 55)

    shutil.rmtree("./_trust_data", ignore_errors=True)
    # 用短 half_life 便于测衰减
    fs = FactStore("./_trust_data", decay_half_life_days=1.0)

    # [1] 新记忆 trust = 默认
    print("\n[1] 新记忆初始信任")
    fid1 = fs.add("The user uses Python and Go.")
    fid2 = fs.add("The user lives in Hangzhou.")
    f1 = next(f for f in fs.facts if f["id"] == fid1)
    assert f1["trust"] == DEFAULT_TRUST, f1["trust"]
    assert f1["uses"] == 0
    print(f"    OK: trust={f1['trust']}, uses={f1['uses']}")

    # [2] 强化提升信任
    print("\n[2] reinforce 提升信任")
    t_before = f1["trust"]
    fs.reinforce(fid1)
    fs.reinforce(fid1)
    f1 = next(f for f in fs.facts if f["id"] == fid1)
    print(f"    强化2次: {t_before:.3f} -> {f1['trust']:.3f}, uses={f1['uses']}")
    assert f1["trust"] > t_before
    assert f1["uses"] == 2

    # [3] 时间衰减: 模拟老记忆
    print("\n[3] 时间衰减 (有效信任随未使用时间下降)")
    # 手动把 fid2 的 last_used 拨到 2 天前 (half_life=1 天 -> 衰减到 ~1/4)
    f2 = next(f for f in fs.facts if f["id"] == fid2)
    f2["last_used"] = int(time.time()) - 2 * 86400
    eff_new = fs._effective_trust(f1)           # 刚强化的, 新鲜
    eff_old = fs._effective_trust(f2)           # 2天没用
    print(f"    新鲜记忆有效信任: {eff_new:.3f}")
    print(f"    2天未用有效信任: {eff_old:.3f} (基础{f2['trust']:.2f} × 衰减)")
    assert eff_old < f2["trust"]

    # [4] 信任加权改变召回排序
    print("\n[4] 信任加权召回")
    # 加两条语义都和"工作"相关的事实, 一条高信任一条衰减
    fs.add("The user works as a backend engineer.")  # fid3
    fid3 = fs.facts[-1]["id"]
    for _ in range(3):
        fs.reinforce(fid3)  # 高信任
    res = fs.retrieve("What does the user do?", top_k=3)
    print("    召回排序 (按 final = 语义×信任):")
    for r in res:
        print(f"      #{r['id']} final={r.get('final',0):.3f} "
              f"sim={r.get('score',0):.3f} trust={r.get('eff_trust',0):.3f} | {r['text'][:35]}")

    # [5] pinned 不衰减
    print("\n[5] pinned 记忆不衰减")
    fid_pin = fs.add("The user dislikes meetings.", pinned=True)
    fp = next(f for f in fs.facts if f["id"] == fid_pin)
    fp["last_used"] = int(time.time()) - 100 * 86400  # 100天前
    eff_pin = fs._effective_trust(fp)
    print(f"    pinned 100天未用有效信任: {eff_pin:.3f} (应=基础{fp['trust']:.2f}, 不衰减)")
    assert eff_pin == fp["trust"]

    print("\n所有信任层断言通过！")


if __name__ == "__main__":
    main()
