# docs — 设计依据与实测数据

memory-engine 的架构选型不是拍脑袋，而是一路实测对比得出的。这里是支撑 README 中各项设计决策的原始报告和数据。

## 文件

### 架构选型依据

| 文件 | 内容 |
|------|------|
| `RESEARCH_REPORT.md` | 完整研究报告：从"参数化记忆"探索到转向工具的全过程，含三方对比、DeepSeek 工作对标、为什么放弃参数化存事实 |
| `results/comparison_3way.json` | 三方对比：参数化记忆 vs Embedding/RAG vs 披露式展开 |
| `results/hard_benchmark.json` | 50 事实硬核 benchmark（4 类查询：直接/改写/多跳/抗干扰），DeepSeek 当裁判 |
| `results/retrieval_holographic_vs_faiss.json` | 检索后端对比：官方关键词式 Holographic vs FAISS+BGE 语义向量 |

### 功能验证

| 文件 | 内容 |
|------|------|
| `results/stress_test_results.json` | 压力测试(105条)：误报率0%、检测率73%、trust分布、健康度 |
| `results/evolution_eval_results.json` | 纵向进化eval(200轮)：信任层 vs 裸RAG 对比 |
| `results/mao_thinker_results.json` | 毛选思维对比(4种方案)：证明性格层的边界(大模型已会的思维教不了) |
| `results/contrarian_results.json` | 逆向决策术对比(4种方案)：进一步验证性格层边界 |

## 关键结论

### 架构选型(指导了初始设计)

1. **事实存取用 RAG，不用参数化小模型** — 实测 RAG 2.0 > 参数化 0.96
2. **检索后端用语义向量** — FAISS+BGE 0.79 >> 关键词式 Holographic 0.00
3. **性格用小模型 LoRA adapter** — 风格迁移 ~87%
4. **巩固/遗忘用检索层标量信任分** — 借鉴 Hermes

### 功能验证(v2 新增功能)

5. **矛盾检测有效** — 105条压力测试: 误报0%, 检测率73%, 兼容不误伤
6. **梯度降权正确** — sim≥0.8 大幅降权, 0.7-0.8 适度降权
7. **使用反馈比"召回即强化"更精确** — 被采纳才+trust, 被忽略微-trust
8. **健康度监控能检测退化** — superseded堆积/低trust过多/区分力下降
9. **人格分区零泄漏** — formal/casual 记忆完全隔离, shared 正确共享
10. **信任加权 Do No Harm** — 加法加权(非乘法)防马太效应, 纵向eval验证不伤害检索
11. **性格层只适合教"风格"不适合教"思维"** — 毛选3.0/逆向3.0(大模型自己就会) vs 风格87%(大模型不会)

### 已知局限

- 间接矛盾(需推理链)纯语义检测不到(漏报4/15)
- 信任衰减在短时间模拟中效果不显著(需真实时间跨度)
- 大规模合成数据(模板化)检索表现差(非架构问题,是数据质量问题)

## 说明

- 报告中的负面结论（参数化记忆输给 RAG、创新点被既有工作占据）是研究过程的真实记录
- 测试数据中的用户信息（如 "Lin Wei"、"Chen Lei"）均为虚构，无真实隐私
