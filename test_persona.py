"""性格层测试: 训练性格 adapter -> 切换 -> 经 DeepSeek 看风格是否迁移。

验证方案A: 小模型性格示范 -> 注入 DeepSeek prompt -> DeepSeek 模仿风格。
对比同一问题在 [无性格] vs [blunt性格] 下 DeepSeek 回答的差异。
"""

import sys
import os
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_engine.engine import MemoryEngine

BLUNT_EXAMPLES = [
    {"user": "How should I structure this microservice?", "response": "Keep it simple. One responsibility, one database. Don't over-architect for 10 requests a minute."},
    {"user": "Should I use Kubernetes for my side project?", "response": "No. A single VPS with Docker Compose. K8s is for real scaling problems, not imaginary ones."},
    {"user": "What do you think about clean architecture?", "response": "Fine when it makes sense. Most projects don't need 47 layers. Write code that's easy to delete."},
    {"user": "Should I rewrite in Rust?", "response": "Why? Measure first. Is Python actually too slow, or did you read a blog post?"},
    {"user": "How do I handle errors?", "response": "Proper status codes. Log with context. Don't swallow exceptions. Done. No 500-line framework."},
    {"user": "Best ORM?", "response": "The one you know. They're all mediocre. Write raw SQL for anything complex."},
    {"user": "More unit tests?", "response": "Test the tricky parts. Skip getters and setters. 80% on what matters beats 100% on everything."},
    {"user": "My PR has 2000 lines.", "response": "Break it up. Nobody reviews 2000 lines. They rubber-stamp it. Small PRs, fewer bugs."},
]


def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: set DEEPSEEK_API_KEY"); sys.exit(1)

    print("=" * 55)
    print("  性格层测试 (方案A: 小模型示范 -> DeepSeek 模仿)")
    print("=" * 55)

    eng = MemoryEngine(store_dir="./_persona_data", enable_persona=True)
    print("[1] Engine + 性格层初始化 OK\n")

    test_qs = ["Should I add a caching layer?", "What database should I pick?"]

    # 无性格基线
    print("[2] 无性格 (baseline) - DeepSeek 默认风格:")
    eng.switch_persona(None)
    base_words = []
    for q in test_qs:
        r = eng.chat(q, top_k=0, max_tokens=150)
        wc = len(r["response"].split())
        base_words.append(wc)
        print(f"    Q: {q}")
        print(f"    A ({wc}词): {r['response'][:100]}\n")

    # 训练 blunt 性格
    print("[3] 训练 'blunt' 性格 adapter...")
    info = eng.create_persona("blunt", BLUNT_EXAMPLES, desc="直接简洁反过度设计", epochs=10)
    print(f"    训练完成: {info['train_time_s']:.0f}s\n")

    # 切换到 blunt
    print("[4] 切换到 'blunt' 性格 - DeepSeek 模仿风格:")
    eng.switch_persona("blunt")
    blunt_words = []
    for q in test_qs:
        r = eng.chat(q, top_k=0, max_tokens=150)
        wc = len(r["response"].split())
        blunt_words.append(wc)
        print(f"    Q: {q}")
        print(f"    [小模型示范]: {r['used_style'][:70]}")
        print(f"    A ({wc}词): {r['response'][:100]}\n")

    # 对比
    print("=" * 55)
    print("  风格迁移效果")
    print("=" * 55)
    base_avg = sum(base_words) / len(base_words)
    blunt_avg = sum(blunt_words) / len(blunt_words)
    print(f"    无性格平均字数:   {base_avg:.0f}")
    print(f"    blunt性格平均字数: {blunt_avg:.0f}")
    if base_avg > 0:
        print(f"    精简幅度:          {(1-blunt_avg/base_avg)*100:+.0f}%")

    # 列出性格
    print(f"\n[5] 已注册性格: {list(eng.list_personas().keys())}")
    print("\nDone!")


if __name__ == "__main__":
    main()
