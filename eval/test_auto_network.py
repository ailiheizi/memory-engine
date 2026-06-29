"""测试自动关系网络构建: auto_build_network。

验证系统能自动发现记忆间的关系并建网, 而非手动连边。
"""

import sys
import os
import shutil
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_engine.fact_store import FactStore
from memory_engine.memory_graph import MemoryGraph
from memory_engine.deepseek_client import DeepSeekClient


def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: set DEEPSEEK_API_KEY"); sys.exit(1)

    print("=" * 55)
    print("  自动关系网络构建测试")
    print("=" * 55)

    shutil.rmtree("./_autonet_test", ignore_errors=True)
    fs = FactStore("./_autonet_test")
    ds = DeepSeekClient()

    # 写入有内在关系的记忆(不手动连边)
    facts = [
        "Chen Lei works at MediFlow as a product manager.",      # 1
        "MediFlow is a health-tech startup in Chengdu.",         # 2 (related to 1)
        "MediFlow's main product is a patient scheduling app.",  # 3 (related to 2)
        "Chen Lei manages a team of 8 engineers.",               # 4 (related to 1)
        "Chen Lei lives in Chengdu.",                            # 5 (related to 2-Chengdu)
        "Chen Lei used to live in Beijing.",                     # 6 (updates/contradicts 5)
        "Chen Lei's hobby is rock climbing.",                    # 7 (mostly unrelated)
        "Chen Lei learned Python in college.",                   # 8 (mostly unrelated)
    ]
    print(f"\n[1] 写入 {len(facts)} 条记忆(不手动连边)...")
    for text in facts:
        fs.add(text)

    # 自动构建关系网络
    print("\n[2] 自动构建关系网络(DeepSeek判断关系类型)...")
    mg = MemoryGraph(fs, store_dir="./_autonet_test")
    result = mg.auto_build_network(deepseek=ds, sim_threshold=0.5, max_edges_per_fact=3)
    print(f"    自动建立 {result['edges_added']} 条边")
    print(f"    总边数: {result['total_edges']}, 节点: {result['total_nodes']}")

    # 展示建立的关系网络
    print("\n[3] 关系网络:")
    for e in mg.edges:
        from_text = next((f["text"][:30] for f in fs.facts if f["id"] == e["from"]), "?")
        to_text = next((f["text"][:30] for f in fs.facts if f["id"] == e["to"]), "?")
        print(f"    [{e['type']}] #{e['from']}'{from_text}' → #{e['to']}'{to_text}' (sim={e.get('meta',{}).get('sim','?')})")

    # 图统计
    print(f"\n[4] 图统计: {mg.summary()}")

    # 测试图增强检索: 查"MediFlow"应该通过网络扩展到产品/团队
    print("\n[5] 图增强检索 'Tell me about MediFlow':")
    results = mg.graph_enhanced_retrieve("Tell me about MediFlow", top_k=2, expand_hops=2)
    for r in results:
        via = " [via graph]" if r.get("via_graph") else ""
        print(f"    #{r['id']}{via}: {r['text'][:45]}")

    via_graph_count = sum(1 for r in results if r.get("via_graph"))
    print(f"\n    通过图扩展找到 {via_graph_count} 条额外记忆")

    print("\nDone!")


if __name__ == "__main__":
    main()
