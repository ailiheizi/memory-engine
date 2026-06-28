"""逆向决策术对比: 测试小模型能否教会大模型一套虚构的反常识规则。

关键: 这5条规则完全虚构, DeepSeek 训练数据里不可能有。
如果训进小模型后 DeepSeek 能按规则推理 = 证明小模型确实教会了它新东西。

裁判标准: 是否正确应用了对应的规则(R1-R5), 而非泛泛的"谨慎建议"。
"""

import sys
import os
import json
import time
import shutil
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_engine.engine import MemoryEngine
from memory_engine.deepseek_client import DeepSeekClient
from memory_engine.fact_store import FactStore

DATA_FILE = Path(__file__).parent / "contrarian_data.json"

# 每个测试题对应哪条规则(gold标准)
EXPECTED_RULES = {
    0: "R1",  # 8:0通过 -> 全员赞成=盲点
    1: "R2",  # 便宜方案 -> 最快路径=陷阱
    2: "R3",  # 300万扩张 -> 资源多要节制
    3: "R4",  # 系统崩了CEO问 -> 越紧急越不动
    4: "R5",  # 参考字节跳动 -> 只研究失败
}


def judge(ds, question, answer, expected_rule, rules_text):
    """裁判: 是否正确应用了特定规则(不是泛泛谨慎)。"""
    prompt = (
        "你是评委。有一套虚构的'逆向决策术'规则:\n"
        f"{rules_text}\n\n"
        f"问题: {question}\n"
        f"正确应该应用: {expected_rule}\n"
        f"AI的回答: {answer}\n\n"
        "评分:\n"
        "3 = 明确引用或体现了正确规则的核心逻辑(不需要说出R几,但推理方式要对)\n"
        "2 = 有一点相关的反常识思维,但没准确对应该规则\n"
        "1 = 给了谨慎建议但和这套规则无关(普通的'要小心')\n"
        "0 = 完全没有反常识推理,给了正面/常规建议\n\n"
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

    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    examples = data["training_examples"]
    test_qs = data["test_queries"]
    rules = data["rules"]
    rules_text = "\n".join(rules)
    ds = DeepSeekClient()

    print("=" * 60)
    print("  逆向决策术: 虚构规则 × 4种方案对比")
    print("  (DeepSeek不可能预先知道这套规则)")
    print("=" * 60)

    results = {}

    # ==== D: 裸调(只说"用逆向思维") ====
    print("\n[D] DeepSeek 裸调 (只说'用逆向思维决策')")
    scores_d = []
    for i, q in enumerate(test_qs):
        r = ds.simple(q, system_prompt="请用逆向思维来分析这个决策。", max_tokens=200)
        sc = judge(ds, q, r, EXPECTED_RULES[i], rules_text)
        scores_d.append(sc)
        print(f"  [{sc}] {q[:30]}... -> {r[:50]}")
    results["D_baseline"] = {"scores": scores_d, "avg": sum(s for s in scores_d if s >= 0) / max(len([s for s in scores_d if s >= 0]), 1)}
    print(f"  平均: {results['D_baseline']['avg']:.2f}/3")

    # ==== C: 披露式(把5条规则写进prompt) ====
    print("\n[C] 披露式 (5条规则直接写进 system prompt)")
    SYS_C = f"你严格按照以下'逆向决策术'规则分析问题。每个问题只选一条最适用的规则:\n\n{rules_text}\n\n分析时要明确说出应用了哪条规则及其推理。"
    scores_c = []
    for i, q in enumerate(test_qs):
        r = ds.simple(q, system_prompt=SYS_C, max_tokens=200)
        sc = judge(ds, q, r, EXPECTED_RULES[i], rules_text)
        scores_c.append(sc)
        print(f"  [{sc}] {q[:30]}... -> {r[:50]}")
    results["C_disclosure"] = {"scores": scores_c, "avg": sum(s for s in scores_c if s >= 0) / max(len([s for s in scores_c if s >= 0]), 1)}
    print(f"  平均: {results['C_disclosure']['avg']:.2f}/3")

    # ==== B: RAG(规则+示例存向量库, 检索后披露) ====
    print("\n[B] RAG (规则+示例存向量库, 检索相关段落)")
    shutil.rmtree("./_contra_rag", ignore_errors=True)
    fs = FactStore("./_contra_rag")
    for rule in rules:
        fs.add(rule)
    for ex in examples:
        fs.add(f"[决策示例] 场景: {ex['user']} → 正确应对: {ex['response']}")
    scores_b = []
    for i, q in enumerate(test_qs):
        retrieved = fs.build_disclosure(q, top_k=5)
        sys_p = f"你是逆向决策术专家。参考以下规则和案例:\n{retrieved}\n\n严格按规则分析,指出应用了哪条。"
        r = ds.simple(q, system_prompt=sys_p, max_tokens=200)
        sc = judge(ds, q, r, EXPECTED_RULES[i], rules_text)
        scores_b.append(sc)
        print(f"  [{sc}] {q[:30]}... -> {r[:50]}")
    results["B_rag"] = {"scores": scores_b, "avg": sum(s for s in scores_b if s >= 0) / max(len([s for s in scores_b if s >= 0]), 1)}
    print(f"  平均: {results['B_rag']['avg']:.2f}/3")

    # ==== A: 小模型 LoRA ====
    print("\n[A] 小模型 LoRA (训练逆向决策术 → DeepSeek 模仿)")
    shutil.rmtree("./_contra_persona", ignore_errors=True)
    eng = MemoryEngine(store_dir="./_contra_persona", enable_persona=True)
    print("  训练 contrarian_decider adapter...")
    info = eng.create_persona("contrarian_decider", examples, desc="逆向决策术5规则", epochs=12)
    print(f"  训练完成: {info['train_time_s']:.0f}s")
    eng.switch_persona("contrarian_decider")
    scores_a = []
    for i, q in enumerate(test_qs):
        r = eng.chat(q, top_k=0, max_tokens=200)
        sc = judge(ds, q, r["response"], EXPECTED_RULES[i], rules_text)
        scores_a.append(sc)
        print(f"  [{sc}] {q[:30]}...")
        print(f"      示范: {r['used_style'][:60]}")
        print(f"      回答: {r['response'][:60]}")
    results["A_lora"] = {"scores": scores_a, "avg": sum(s for s in scores_a if s >= 0) / max(len([s for s in scores_a if s >= 0]), 1)}
    print(f"  平均: {results['A_lora']['avg']:.2f}/3")

    # ==== 汇总 ====
    print("\n" + "=" * 60)
    print("  逆向决策术 评分 (0-3, 是否正确应用对应规则)")
    print("=" * 60)
    print(f"  {'方案':<30} {'分数':<20} {'平均'}")
    print(f"  {'-'*30} {'-'*20} {'-'*6}")
    for name in ["D_baseline", "C_disclosure", "B_rag", "A_lora"]:
        r = results[name]
        print(f"  {name:<30} {str(r['scores']):<20} {r['avg']:.2f}/3")

    out = Path(__file__).parent / "contrarian_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Saved to {out}")
    print("\nDone!")


if __name__ == "__main__":
    main()
