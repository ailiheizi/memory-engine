"""毛选思维对比: 小模型LoRA vs RAG检索 vs 披露prompt vs 裸调 DeepSeek。

测试: "毛泽东的思维方式能否被学习并迁移给大模型?"
4种方案回答同一组现代问题, DeepSeek 当裁判评分(0-3分):
  3 = 深度运用毛式思维(矛盾分析/抓主要矛盾/实践检验/群众路线等), 有洞察力
  2 = 有一定毛式思维痕迹, 但不够深刻或流于表面
  1 = 泛泛而谈, 和普通回答无明显区别
  0 = 完全没有毛式思维特征
"""

import sys
import os
import json
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_engine.engine import MemoryEngine
from memory_engine.deepseek_client import DeepSeekClient
from memory_engine.fact_store import FactStore

DATA_FILE = Path(__file__).parent / "mao_thinker_data.json"


def load_data():
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def judge(ds, question, answer):
    """DeepSeek 裁判: 评分 0-3 毛式思维深度。"""
    prompt = (
        "你是评委，评估一段回答是否体现了毛泽东的思维方式(矛盾分析法、抓主要矛盾、"
        "实践论、群众路线、调查研究、集中优势兵力各个击破等)。\n\n"
        f"问题: {question}\n"
        f"回答: {answer}\n\n"
        "评分标准:\n"
        "3 = 深度运用毛式思维，有具体的矛盾分析/主次区分/实践导向，有洞察力\n"
        "2 = 有毛式思维痕迹，但不够深刻或只是表面套话\n"
        "1 = 泛泛而谈，和普通回答无明显区别\n"
        "0 = 完全没有毛式思维特征\n\n"
        "只回复一个数字(0/1/2/3):"
    )
    try:
        resp = ds.simple(prompt, temperature=0.0, max_tokens=5).strip()
        for ch in resp:
            if ch in "0123":
                return int(ch)
        return 0
    except Exception as e:
        print(f"  judge error: {e}")
        return -1


def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: set DEEPSEEK_API_KEY"); sys.exit(1)

    data = load_data()
    examples = data["training_examples"]
    test_qs = data["test_queries"]
    ds = DeepSeekClient()

    print("=" * 60)
    print("  毛选思维对比: 4种方案 × DeepSeek 裁判")
    print("=" * 60)

    results = {}

    # ==== 方案D: DeepSeek 裸调 (baseline) ====
    print("\n[D] DeepSeek 裸调 (只说'用毛泽东思维分析')")
    scores_d = []
    for q in test_qs:
        r = ds.simple(q, system_prompt="请用毛泽东的思维方式来分析和回答以下问题。", max_tokens=300)
        sc = judge(ds, q, r)
        scores_d.append(sc)
        print(f"  [{sc}] {q[:25]}... -> {r[:60]}")
    results["D_baseline"] = {"scores": scores_d, "avg": sum(s for s in scores_d if s >= 0) / max(len([s for s in scores_d if s >= 0]), 1)}
    print(f"  平均: {results['D_baseline']['avg']:.2f}/3")

    # ==== 方案C: 披露式 system prompt (手写思维描述) ====
    print("\n[C] 披露式 (详细描述毛式思维方法)")
    DISCLOSURE_PROMPT = """你是一个运用毛泽东思维方式的分析师。你的核心思维工具:

1. 矛盾分析法: 任何事物都有主要矛盾和次要矛盾,要抓主要矛盾的主要方面
2. 实践论: 实践是检验真理的唯一标准,理论来源于实践又回到实践
3. 调查研究: 没有调查就没有发言权,要深入实际掌握第一手材料
4. 群众路线: 从群众中来到群众中去,相信群众依靠群众
5. 集中优势兵力: 不分散力量,在局部形成优势各个击破
6. 螺旋上升: 事物发展是曲折的,前途光明道路曲折
7. 抓典型: 先在点上搞清楚取得经验,再推广到面上
8. 战略藐视战术重视: 整体必胜信心,具体认真对待

请深度运用以上思维方法分析问题,要有具体的矛盾分析和主次判断,不要泛泛而谈。"""
    scores_c = []
    for q in test_qs:
        r = ds.simple(q, system_prompt=DISCLOSURE_PROMPT, max_tokens=300)
        sc = judge(ds, q, r)
        scores_c.append(sc)
        print(f"  [{sc}] {q[:25]}... -> {r[:60]}")
    results["C_disclosure"] = {"scores": scores_c, "avg": sum(s for s in scores_c if s >= 0) / max(len([s for s in scores_c if s >= 0]), 1)}
    print(f"  平均: {results['C_disclosure']['avg']:.2f}/3")

    # ==== 方案B: RAG 检索毛选段落 → 披露给 DeepSeek ====
    print("\n[B] RAG 检索 (毛选段落存向量库, 检索相关段落)")
    import shutil
    shutil.rmtree("./_mao_rag", ignore_errors=True)
    fs = FactStore("./_mao_rag")
    # 把训练数据的 response 当作"毛选语录/思维段落"存入
    for ex in examples:
        fs.add(f"[毛式思维] 关于「{ex['user']}」: {ex['response']}")
    scores_b = []
    for q in test_qs:
        retrieved = fs.build_disclosure(q, top_k=5)
        sys_p = f"你是用毛泽东思维方式分析问题的助手。参考以下相关思维方法:\n{retrieved}\n\n深度运用上述方法分析问题。"
        r = ds.simple(q, system_prompt=sys_p, max_tokens=300)
        sc = judge(ds, q, r)
        scores_b.append(sc)
        print(f"  [{sc}] {q[:25]}... -> {r[:60]}")
    results["B_rag"] = {"scores": scores_b, "avg": sum(s for s in scores_b if s >= 0) / max(len([s for s in scores_b if s >= 0]), 1)}
    print(f"  平均: {results['B_rag']['avg']:.2f}/3")

    # ==== 方案A: 小模型 LoRA (训毛选思维) → DeepSeek 模仿 ====
    print("\n[A] 小模型 LoRA (训练毛式思维, 生成示范 → DeepSeek 模仿)")
    import shutil
    shutil.rmtree("./_mao_persona", ignore_errors=True)
    eng = MemoryEngine(store_dir="./_mao_persona", enable_persona=True)
    print("  训练 mao_thinker adapter...")
    info = eng.create_persona("mao_thinker", examples, desc="毛泽东思维方式", epochs=10)
    print(f"  训练完成: {info['train_time_s']:.0f}s")
    eng.switch_persona("mao_thinker")
    scores_a = []
    for q in test_qs:
        r = eng.chat(q, top_k=0, max_tokens=300)  # top_k=0 不检索事实,只用性格
        sc = judge(ds, q, r["response"])
        scores_a.append(sc)
        print(f"  [{sc}] {q[:25]}...")
        print(f"      示范: {r['used_style'][:50]}")
        print(f"      回答: {r['response'][:60]}")
    results["A_lora"] = {"scores": scores_a, "avg": sum(s for s in scores_a if s >= 0) / max(len([s for s in scores_a if s >= 0]), 1)}
    print(f"  平均: {results['A_lora']['avg']:.2f}/3")

    # ==== 汇总 ====
    print("\n" + "=" * 60)
    print("  毛选思维深度评分 (DeepSeek 裁判, 0-3)")
    print("=" * 60)
    print(f"  {'方案':<30} {'分数':<20} {'平均'}")
    print(f"  {'-'*30} {'-'*20} {'-'*6}")
    for name in ["D_baseline", "C_disclosure", "B_rag", "A_lora"]:
        r = results[name]
        print(f"  {name:<30} {str(r['scores']):<20} {r['avg']:.2f}/3")

    # 保存
    out = Path(__file__).parent / "mao_thinker_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Saved to {out}")
    print("\nDone!")


if __name__ == "__main__":
    main()
