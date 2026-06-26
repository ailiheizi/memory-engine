"""端到端冒烟测试: 验证 memory-engine 库的核心流程(不含性格训练, 快)。"""

import sys
import os
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# 用 memory-engine 包
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_engine.engine import MemoryEngine


def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: set DEEPSEEK_API_KEY"); sys.exit(1)

    print("=" * 55)
    print("  memory-engine 冒烟测试 (记忆层, 无性格)")
    print("=" * 55)

    # 清理旧数据保证可重复
    import shutil
    shutil.rmtree("./_smoke_data", ignore_errors=True)

    # 关闭性格层(只测记忆, 快)
    engine = MemoryEngine(store_dir="./_smoke_data", enable_persona=False)
    print("[1] Engine 初始化 OK\n")

    # 增
    print("[2] 增加事实...")
    ids = []
    for text in [
        "The user's name is Lin Wei.",
        "The user works at a fintech startup as a backend engineer.",
        "The user prefers Go and Python.",
        "The user has a cat named Mochi.",
        "The user lives in Hangzhou.",
    ]:
        ids.append(engine.add_fact(text))
    engine.add_fact("The user dislikes unnecessary meetings.", pinned=True)
    print(f"    加了 {len(ids)+1} 条 (含1条pinned)\n")

    # RAG 召回测试
    print("[3] RAG 召回 (披露式展开)...")
    disc = engine.facts.build_disclosure("What's the user's pet?", top_k=2)
    print(f"    召回内容:\n{disc}\n")

    # 改
    print("[4] 修改事实...")
    engine.update_fact(ids[2], text="The user now primarily uses Rust and Zig.")
    disc2 = engine.facts.build_disclosure("What languages does the user use?", top_k=2)
    print(f"    改后召回: {disc2}\n")

    # 删
    print("[5] 删除事实...")
    engine.delete_fact(ids[3])  # 删掉 Mochi
    after = [f["text"] for f in engine.list_facts()]
    print(f"    剩余 {len(after)} 条, Mochi 已删: {'Mochi' not in str(after)}\n")

    # 完整 chat (带记忆, 经 DeepSeek)
    print("[6] 完整对话 (记忆 -> DeepSeek)...")
    for q in ["What's my name?", "What languages do I use now?", "Do I like meetings?"]:
        r = engine.chat(q, top_k=3, max_tokens=80)
        print(f"    Q: {q}")
        print(f"    A: {r['response'][:90]}")
        print(f"    ({r['latency_ms']:.0f}ms)\n")

    print("Done! 记忆层端到端跑通。")


if __name__ == "__main__":
    main()
