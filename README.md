# memory-engine

给 LLM API（DeepSeek/OpenAI/Claude 等）提供**可编辑记忆 + 可切换性格**的外挂库。纯 Python，可作为库直接 import，或作为独立 HTTP Service 接入任意系统。

## 核心：多增强层协调

不是单一记忆功能，而是 RAG + 矛盾检测 + 图扩展 + query expansion 的**协调组合**。
关键发现：多增强层 naive 全开会互相干扰，协调后才能真正叠加生效。

| 配置 | OVERALL (硬核评测, baseline难度77%) |
|------|------|
| 裸 RAG | 77% |
| naive 全开(有干扰) | 88% |
| **协调全开** | **96%** |

> 详见 [docs/EVALUATION.md](docs/EVALUATION.md) — 含每个功能的有效性分级与证据。
> 诚实说明：大量复用成熟技术(RAG/Hermes信任分/记忆图等)，差异在多层协调；
> 单一记忆需求直接用 mem0/Zep 更省。

## 设计(基于实测选型)

```
用户消息
  ① RAG 索引召回候选 + 信任加权重排   (FAISS + BGE-M3, 可扩展)
  ② top-k 全部披露进 context          (retrieve-then-read)
  ③ 矛盾检测: 写入时自动发现并降权旧记忆
  ④ 性格 adapter 小模型生成风格示范    (方案A, 可选)
  ⑤ 组装 prompt = [性格示范] + [披露事实] + [用户消息]
  ⑥ DeepSeek 回答
  ⑦ 使用反馈: 被实际采纳的记忆才强化 trust
```

**为什么这样选**(三方对比实测, DeepSeek 当裁判):
- 事实: RAG 2.0 / 披露 1.83 > 参数化小模型 0.96 → 事实用 RAG+披露
- 检索后端: 语义向量 0.79 >> 关键词式 Holographic 0.00(问答场景词不重叠)→ 用 FAISS+BGE
- 事实可即时增删改: 向量操作毫秒级、外科手术式隔离, 无需训练
- 性格: 多 LoRA adapter, 每个性格物理隔离, 切换=换adapter, 规避灾难性遗忘

## 核心功能

### 信任层(借鉴 Hermes)

每条记忆带信任分，让"可信+常用+新鲜"的记忆浮顶、老噪音自然沉底。

```
召回最终分 = 语义相似度 + 0.1 × 有效信任     # 加法加权(防马太效应)
有效信任   = trust × 0.5^(未使用天数 / 半衰期)  # pinned 不衰减
```

- **trust**：新记忆默认 0.5
- **强化**：记忆被 DeepSeek 实际采纳 → trust 上升(使用反馈, 非"召回即强化")
- **衰减**：久未使用 → 有效信任随时间减半
- **pinned**：永不衰减,始终排最前
- **superseded**：被矛盾检测标记替代的旧记忆,排到所有正常记忆之后

### 矛盾检测(写入时自动)

写入新记忆时自动检测是否与已有记忆矛盾，梯度化处理：

| 情况 | 处理 |
|------|------|
| 硬矛盾(sim≥0.8) + CONTRADICTION | 旧记忆 trust 归零 |
| 高相似(sim≥0.8) + UPDATE | 旧记忆 trust × 0.2 |
| 中等相似(0.7-0.8) + CONTRADICTION | 旧记忆 trust × 0.3 |
| 中等相似(0.7-0.8) + UPDATE | 旧记忆 trust × 0.5 |
| COMPATIBLE(兼容) | 不动 |

机制：BGE 语义检索相似候选 → DeepSeek 做 NLI 判断(CONTRADICTION/UPDATE/COMPATIBLE)。

### 使用反馈循环

不是"召回即强化"——对比 DeepSeek 回答和每条检索记忆的语义相似度：
- 高相似(>0.5) = 被采纳 → trust 上升
- 低相似(<0.3) = 被忽略 → trust 微降

让 trust 真正反映"哪条记忆对回答有贡献"。

### 记忆健康度监控

周期性检查记忆库退化：
- **相似度区分力**(top-k std)：低 = 记忆太相似难以区分
- **superseded 占比**：高 = 过时记忆堆积
- **低 trust 占比**：高 = 垃圾太多
- 超阈值自动告警 + 建议清理列表

### 人格记忆分区

每个性格维护独立记忆分区,防止"人格泄漏"：
- **shared 层**：所有人格共享的核心事实(名字/工作/地址)
- **persona 层**：每个人格独立的偏好/上下文记忆
- 检索时：shared + 当前 persona 合并,其他 persona 不可见
- 切换人格 = 切换可见的记忆分区

## 分层职责

| 层 | 管什么 | 增 | 删 | 改 |
|----|--------|-----|-----|-----|
| 冷记忆 RAG | 大量事实 | 插向量 | 删向量 | re-embed一条 |
| 热记忆 pinned | 高频事实 | pinned=true | 删 | 改文本 |
| 信任层 | 巩固/遗忘 | 使用反馈强化 | 衰减沉底 | 自动 |
| 矛盾检测 | 一致性 | 写入时自动检测 | superseded 降权 | 梯度化 |
| 健康度 | 质量监控 | check_health() | suggest_cleanup() | 告警 |
| 性格 adapter | 行为风格 | 训新adapter | 删目录 | 重训adapter |
| 人格分区 | 隔离 | 按persona写入 | 按partition删 | 切换可见性 |

## 安装

```powershell
uv venv --python 3.12
.venv\Scripts\activate
uv pip install -e .
# torch 需 >=2.6 (sentence-transformers 要求)
```

## 用法 1: Python 库

```python
from memory_engine.engine import MemoryEngine

eng = MemoryEngine(store_dir="./mem", deepseek_key="sk-...")

# 记忆增删改(写入时自动矛盾检测)
result = eng.add_fact("用户叫张伟")           # {"id": 1, "conflicts": [], "resolved": False}
result = eng.add_fact("用户叫李明")           # {"id": 2, "conflicts": [旧条], "resolved": True}
eng.update_fact(1, text="用户叫张伟, 后端工程师")
eng.delete_fact(1)

# 性格(可选, 需训练)
eng.create_persona("blunt", [
    {"user": "要加缓存吗?", "response": "先测瓶颈。没瓶颈别加。"},
], desc="直接简洁")
eng.switch_persona("blunt")

# 带记忆+性格对话 (使用反馈自动更新 trust)
r = eng.chat("我叫什么?", top_k=3)
print(r["response"])    # 回答
print(r["feedback"])    # [{"id": 1, "adopted": True, "sim": 0.72}]

# 健康度检查
health = eng.check_health()
print(health["alerts"])         # ["过时堆积: 30%已supersede"]
cleanup = eng.suggest_cleanup() # [{"id": 3, "reasons": ["superseded","low_trust"]}]

# 信任反馈
eng.reinforce_fact(1)   # 手动强化
```

### 人格分区用法

```python
from memory_engine.partitioned_memory import PartitionedMemory

pm = PartitionedMemory("./mem")

# 共享事实(所有人格可见)
pm.add("用户叫张伟", shared=True)

# 人格专属记忆
pm.switch_persona("formal")
pm.add("用户不喜欢emoji")          # 只在 formal 可见

pm.switch_persona("casual")
pm.add("用户喜欢表情包和玩梗")      # 只在 casual 可见

# 检索时自动隔离
pm.switch_persona("formal")
pm.retrieve("沟通风格")  # 只返回 shared + formal 的记忆
```

## 用法 2: HTTP Service

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
python -m memory_engine.service --port 8900
# 加 --no-persona 只用记忆层(不加载小模型, 更快更省内存)
```

接口(POST JSON):

| 路由 | 入参 | 返回 |
|------|------|------|
| `POST /chat` | `{message, top_k}` | `{response, used_memory, feedback, latency_ms}` |
| `POST /facts/add` | `{text, pinned}` | `{id, conflicts, resolved}` |
| `POST /facts/delete` | `{id}` | `{ok}` |
| `POST /facts/update` | `{id, text, pinned}` | `{ok}` |
| `GET /facts/list` | - | `{facts}` |
| `GET /health` | - | `{healthy, alerts, avg_trust, ...}` |
| `POST /persona/create` | `{id, examples, desc}` | `{train_time_s}` |
| `POST /persona/switch` | `{id}` (null=无性格) | `{ok}` |
| `POST /persona/delete` | `{id}` | `{ok}` |
| `GET /persona/list` | - | `{personas}` |

## 可调参数

`memory_engine/fact_store.py`:

| 参数 | 默认 | 含义 |
|------|------|------|
| `DEFAULT_TRUST` | 0.5 | 新记忆初始信任 |
| `REINFORCE_GAIN` | 0.15 | 每次采用的信任增幅 |
| `DECAY_HALF_LIFE_DAYS` | 30 | 未使用信任减半天数(0=关闭衰减) |
| `TRUST_MIN` | 0.05 | 信任下限 |

`memory_engine/conflict_detector.py`:

| 参数 | 默认 | 含义 |
|------|------|------|
| `SIMILARITY_THRESHOLD` | 0.7 | 超过此值才做矛盾检测 |

`memory_engine/usage_feedback.py`:

| 参数 | 默认 | 含义 |
|------|------|------|
| `ADOPTION_THRESHOLD` | 0.5 | 回答与记忆sim>此值视为采纳 |
| `IGNORE_THRESHOLD` | 0.3 | 回答与记忆sim<此值视为忽略 |
| `IGNORE_PENALTY` | 0.03 | 被忽略时trust微降幅度 |

## 实测数据

### 压力测试(105条记忆)

| 指标 | 结果 | 阈值 |
|------|------|------|
| 误报率(不该触发矛盾的) | **0%** (0/90) | <10% |
| 矛盾检测率(该抓的) | **73%** (11/15) | ≥60% |
| 兼容误伤 | **0%** (0/10) | 0% |
| 健康度 | 健康 | - |
| Trust 分布 | 92条正常(0.5) + 13条被降权(0.1-0.3) | 合理 |

### 功能验证

| 测试 | 结果 |
|------|------|
| 记忆端到端(增删改+RAG+DeepSeek) | ✅ ~700ms |
| 矛盾检测(硬矛盾/软更新/兼容) | ✅ 梯度降权正确 |
| 使用反馈(被采纳才强化) | ✅ |
| 健康度监控(退化检测+告警) | ✅ |
| 人格分区(隔离+共享) | ✅ 零泄漏 |
| 信任层 Do No Harm | ✅ 不伤害基础检索 |
| 性格风格迁移(DeepSeek模仿) | ✅ 精简87% |

### 漏报分析(4条未检测到的更新)

均为**间接矛盾**(需要推理"搬到深圳"意味着"不住成都了")，超出纯语义相似度能力。这是已知局限，需要更深层推理或 chain-of-thought NLI 解决。

## 文件结构

```
memory_engine/
├── engine.py               # 核心引擎(流水线 + 集成所有模块)
├── fact_store.py           # 事实层(RAG + 信任加法加权 + superseded)
├── conflict_detector.py    # 写入时矛盾检测(梯度降权)
├── usage_feedback.py       # 使用反馈循环(被采纳才强化)
├── health_monitor.py       # 记忆健康度监控(退化告警+清理建议)
├── partitioned_memory.py   # 人格记忆分区(隔离+共享)
├── persona_manager.py      # 多adapter性格(LoRA切换)
├── deepseek_client.py      # DeepSeek客户端
└── service.py              # HTTP service

eval/
├── test_stress.py          # 压力测试(105条, 矛盾+兼容+健康度)
├── test_conflict.py        # 矛盾检测测试
├── test_new_features.py    # 反馈+健康+分区综合测试
├── test_evolution.py       # 纵向进化eval(200轮)
├── test_mao_compare.py     # 毛选思维对比(验证性格层边界)
├── test_contrarian.py      # 逆向决策术对比

test_trust.py               # 信任层单元测试
test_smoke.py               # 端到端冒烟测试
test_persona.py             # 性格层测试
```

## 已知局限

- 间接矛盾(需推理链)无法纯靠语义相似度检测
- 大模型已会的思维方式(如毛泽东思维)无法通过小模型"教"
- 性格层适合教"怎么说话"(风格)，不适合教"怎么思考"(深层推理)
- 信任衰减在短时间测试中无法验证(需要真实时间跨度)
- DeepSeek NLI 判断有延迟(~700ms/条), 大批量写入时需异步

## 集成到其他系统

本服务是独立进程 + 纯 HTTP 接口。任何能发 HTTP 请求的宿主都能接入：
- 调 `/chat` 由本服务完成全流程
- 或调 `/facts/*` 自行管理记忆,用自己的 prompt 逻辑
- 用 `GET /health` 做就绪探针
