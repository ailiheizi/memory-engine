# docs — 设计依据与实测数据

memory-engine 的架构选型不是拍脑袋，而是一路实测对比得出的。这里是支撑 README 中各项设计决策的原始报告和数据。

## 文件

| 文件 | 内容 |
|------|------|
| `RESEARCH_REPORT.md` | 完整研究报告：从"参数化记忆"探索到转向工具的全过程，含三方对比、DeepSeek 工作对标、为什么放弃参数化存事实 |
| `results/comparison_3way.json` | 三方对比：参数化记忆 vs Embedding/RAG vs 披露式展开 |
| `results/hard_benchmark.json` | 50 事实硬核 benchmark（4 类查询：直接/改写/多跳/抗干扰），DeepSeek 当裁判 |
| `results/retrieval_holographic_vs_faiss.json` | 检索后端对比：官方关键词式 Holographic vs FAISS+BGE 语义向量 |

## 关键结论(指导了 memory-engine 的设计)

1. **事实存取用 RAG，不用参数化小模型** — 实测 RAG 2.0 > 参数化 0.96
2. **检索后端用语义向量** — FAISS+BGE 0.79 >> 关键词式 Holographic 0.00（问答场景查询词与事实不重叠，关键词匹配失效）
3. **性格用小模型 LoRA adapter** — 这是 RAG/披露都做不到的唯一不可替代处，风格迁移 ~87%
4. **巩固/遗忘用检索层标量信任分** — 借鉴 Hermes，比权重训练轻、可调试、即时生效

## 说明

- 报告中的负面结论（参数化记忆输给 RAG、创新点被既有工作占据）是研究过程的真实记录，保留以供参考设计取舍的来龙去脉。
- 测试数据中的用户信息（如 "Lin Wei"）均为虚构，无真实隐私。
