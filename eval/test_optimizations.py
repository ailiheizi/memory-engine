"""测试三个新优化: Beta-Bernoulli + 聚类索引 + 记忆图。"""

import sys
import os
import shutil
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_engine.fact_store import FactStore
from memory_engine.memory_worth import MemoryWorth
from memory_engine.cluster_index import ClusterIndex
from memory_engine.memory_graph import MemoryGraph


def test_beta_bernoulli():
    """#1 Memory Worth: Beta-Bernoulli 信任评分。"""
    print("\n[#1] Memory Worth (Beta-Bernoulli)")
    print("-" * 40)
    shutil.rmtree("./_worth_test", ignore_errors=True)
    fs = FactStore("./_worth_test")
    fs.add("The user lives in Hangzhou.")
    fs.add("The user likes coffee.")
    fs.add("The user hates spam emails.")

    mw = MemoryWorth(fs)

    # 初始: worth = (0+1)/(0+0+2) = 0.5
    print(f"  初始 worth: {mw.get_worth(1):.3f} (应=0.500)")
    assert abs(mw.get_worth(1) - 0.5) < 0.01

    # 模拟: fact#1 被采纳5次, 忽略0次 → worth高
    for _ in range(5):
        mw.record_success(1)
    print(f"  5次成功后: worth={mw.get_worth(1):.3f} (应≈0.857)")
    assert mw.get_worth(1) > 0.8

    # fact#3 被忽略4次, 成功1次 → worth低
    mw.record_success(3)
    for _ in range(4):
        mw.record_failure(3)
    print(f"  1成功4失败: worth={mw.get_worth(3):.3f} (应≈0.286)")
    assert mw.get_worth(3) < 0.35

    # 置信度: 观察多的置信高
    conf1 = mw.get_confidence(1)  # 5次观察
    conf2 = mw.get_confidence(2)  # 0次观察
    print(f"  置信度: #1={conf1:.2f}(5次), #2={conf2:.2f}(0次)")
    assert conf1 > conf2

    # 淘汰候选: worth低+观察够
    candidates = mw.eviction_candidates(threshold=0.4, min_observations=3)
    print(f"  淘汰候选(worth<0.4, 观察>=3): {len(candidates)}条")
    assert len(candidates) >= 1 and candidates[0]["id"] == 3

    # trust 已同步
    f1 = next(f for f in fs.facts if f["id"] == 1)
    print(f"  fact#1 trust 已同步为 worth: {f1['trust']:.3f}")
    assert abs(f1["trust"] - mw.get_worth(1)) < 0.01

    print("  OK: Beta-Bernoulli 评分正确")


def test_cluster_index():
    """#2 聚类索引: Cluster-First-Search。"""
    print("\n[#2] 聚类索引")
    print("-" * 40)
    shutil.rmtree("./_cluster_test", ignore_errors=True)
    fs = FactStore("./_cluster_test")

    # 写入40条(超过MIN_FACTS_FOR_CLUSTERING=30)
    tech_facts = [f"The user uses {t} for development." for t in
                  ["Python", "Go", "Rust", "JavaScript", "TypeScript", "Java", "C++", "Ruby", "Swift", "Kotlin"]]
    food_facts = [f"The user likes eating {f}." for f in
                  ["sushi", "ramen", "pizza", "burger", "hotpot", "dim sum", "curry", "tacos", "pasta", "pho"]]
    work_facts = [f"The user works on {p}." for p in
                  ["payment system", "recommendation engine", "chat bot", "data pipeline", "mobile app",
                   "API gateway", "monitoring tool", "CI/CD pipeline", "auth service", "search engine"]]
    hobby_facts = [f"The user enjoys {h}." for h in
                   ["cycling", "swimming", "hiking", "gaming", "reading", "cooking", "photography", "running", "yoga", "climbing"]]

    for facts_list in [tech_facts, food_facts, work_facts, hobby_facts]:
        for text in facts_list:
            fs.add(text)
    print(f"  写入 {len(fs.facts)} 条记忆(4个主题各10条)")

    ci = ClusterIndex(fs, n_clusters=6)
    result = ci.build()
    print(f"  聚类构建: {result['status']}, {result.get('n_clusters',0)}个cluster")
    print(f"  cluster大小: {result.get('cluster_sizes', [])}")

    # 测试两级检索
    results = ci.search("What programming languages does the user know?", top_k=5)
    print(f"\n  查询'programming languages':")
    for r in results[:3]:
        print(f"    cluster={r.get('cluster')} sim={r.get('score',0):.3f}: {r['text'][:40]}")

    # 应该主要返回 tech 类
    tech_hits = sum(1 for r in results if "uses" in r["text"] and "development" in r["text"])
    print(f"  tech类命中: {tech_hits}/5")
    assert tech_hits >= 3, f"Should find tech facts, got {tech_hits}"

    # cluster summary
    summary = ci.get_cluster_summary()
    print(f"\n  Cluster摘要:")
    for s in summary[:4]:
        print(f"    cluster {s['cluster']}: {s['size']}条, 例: {s['examples'][0][:30]}")

    print("  OK: 聚类索引正确")


def test_memory_graph():
    """#3 记忆图: 关联边 + 图增强检索。"""
    print("\n[#3] 记忆图")
    print("-" * 40)
    shutil.rmtree("./_graph_test", ignore_errors=True)
    fs = FactStore("./_graph_test")

    # 写入有关联关系的记忆
    fs.add("Chen Lei works at MediFlow.")             # 1
    fs.add("MediFlow is a health-tech startup.")       # 2
    fs.add("MediFlow's product is patient scheduling.")# 3
    fs.add("Chen Lei manages 5 engineers.")            # 4
    fs.add("Chen Lei lives in Chengdu.")               # 5
    fs.add("Chen Lei moved to Shenzhen.")              # 6 (updates #5)
    fs.add("Chen Lei likes cycling.")                  # 7 (unrelated)

    mg = MemoryGraph(fs, store_dir="./_graph_test")

    # 手动建边
    mg.add_edge(1, 2, "related")        # MediFlow关联
    mg.add_edge(2, 3, "related")        # MediFlow的产品
    mg.add_edge(1, 4, "related")        # Chen Lei管人
    mg.add_edge(6, 5, "updates")        # 搬家更新
    mg.link_conflict(5, 6, "UPDATE")    # 矛盾关联

    print(f"  图: {mg.summary()}")

    # 测试图增强检索: 查"MediFlow"应该通过图扩展找到相关的产品/团队信息
    results = mg.graph_enhanced_retrieve("What is MediFlow?", top_k=3, expand_hops=1)
    print(f"\n  查询'What is MediFlow?' (图增强):")
    ids_found = set()
    for r in results:
        via = " [via graph]" if r.get("via_graph") else ""
        print(f"    #{r['id']} sim={r.get('score',0):.3f}{via}: {r['text'][:40]}")
        ids_found.add(r["id"])

    # 应该通过图扩展找到 #3(产品)和 #4(团队), 即使语义检索可能没直接找到它们
    print(f"  找到的id: {ids_found}")
    # 至少应该找到 #2(health-tech) 通过语义 + #1/#3 通过图
    assert 2 in ids_found or 1 in ids_found, "Should find MediFlow facts"

    # 测试邻居
    neighbors_of_1 = mg.get_neighbors(1)
    print(f"\n  #1的邻居: {[(n['to'], n['type']) for n in neighbors_of_1]}")
    assert len(neighbors_of_1) >= 2

    # 测试矛盾边
    neighbors_of_6 = mg.get_neighbors(6)
    has_update_edge = any(n["type"] in ("updates", "contradicts") for n in neighbors_of_6)
    print(f"  #6(搬深圳)有更新边: {has_update_edge}")
    assert has_update_edge

    print("  OK: 记忆图正确")


def main():
    print("=" * 55)
    print("  三大优化测试: Worth + Cluster + Graph")
    print("=" * 55)

    test_beta_bernoulli()
    test_cluster_index()
    test_memory_graph()

    print("\n" + "=" * 55)
    print("  全部通过!")
    print("=" * 55)


if __name__ == "__main__":
    main()
