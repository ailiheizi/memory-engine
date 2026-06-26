# memory-engine

给闭源 API 大模型(DeepSeek)提供**可编辑记忆 + 可切换性格**的外挂库。纯 Python，可作为库直接 import，或作为独立 HTTP Service 接入任意系统。

## 设计(基于实测选型)

```
用户消息
  ① RAG 索引召回候选 + 信任加权重排   (FAISS + BGE-M3, 可扩展到几千条)
  ② top-k 全部披露进 context          (retrieve-then-read)
  ③ 性格 adapter 小模型生成风格示范    (方案A, 可选)
  ④ 组装 prompt = [性格示范] + [披露事实] + [用户消息]
  ⑤ DeepSeek 回答
  ⑥ 被采用的记忆信任上升(强化)
```

**为什么这样选**(三方对比实测, DeepSeek 当裁判):
- 事实: RAG 2.0 / 披露 1.83 > 参数化小模型 0.96 → 事实用 RAG+披露
- 检索后端: 语义向量 0.79 >> 关键词式 Holographic 0.00(问答场景词不重叠)→ 用 FAISS+BGE
- 事实可即时增删改: 向量操作毫秒级、外科手术式隔离, 无需训练
- 性格: 多 LoRA adapter, 每个性格物理隔离, 切换=换adapter, 规避灾难性遗忘

## 信任层(借鉴 Hermes)

每条记忆带一个信任分,让"可信+常用+新鲜"的记忆浮顶、老噪音自然沉底——这是裸 RAG 做不到的(裸 RAG 平等对待所有片段)。

```
召回最终分 = 语义相似度 × 有效信任
有效信任   = trust × 0.5^(未使用天数 / 半衰期)     # pinned 不衰减
```

- **trust**:新记忆默认 0.5
- **强化**:记忆被召回采用 → trust 向 1 靠拢(`chat` 默认 `reinforce=True`,也可手动 `reinforce_fact(id)`)
- **衰减**:久未使用 → 有效信任随时间减半,弱相关老记忆沉底
- **pinned**:永不衰减,始终排最前

可调参数(`memory_engine/fact_store.py` 顶部):

| 参数 | 默认 | 含义 |
|------|------|------|
| `DEFAULT_TRUST` | 0.5 | 新记忆初始信任 |
| `REINFORCE_GAIN` | 0.15 | 每次采用的信任增幅 |
| `DECAY_HALF_LIFE_DAYS` | 30 | 未使用信任减半天数(0=关闭衰减) |
| `TRUST_MIN` | 0.05 | 信任下限(衰减后仍可召回) |

## 分层职责

| 层 | 管什么 | 增 | 删 | 改 |
|----|--------|-----|-----|-----|
| 冷记忆 RAG | 大量事实 | 插向量 | 删向量 | re-embed一条 |
| 热记忆 pinned | 高频事实 | pinned=true | 删 | 改文本 |
| 信任层 | 巩固/遗忘 | reinforce 强化 | 衰减沉底 | 自动 |
| 性格 adapter | 行为风格 | 训新adapter | 删目录 | 重训adapter |

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

# 记忆增删改
fid = eng.add_fact("用户叫张伟", pinned=False)
eng.update_fact(fid, text="用户叫张伟, 是后端工程师")
eng.delete_fact(fid)

# 性格(可选, 需训练)
eng.create_persona("blunt", [
    {"user": "要加缓存吗?", "response": "先测瓶颈。没瓶颈别加。"},
    # ... 更多风格示范
], desc="直接简洁")
eng.switch_persona("blunt")

# 带记忆+性格对话 (默认召回即强化被采用的记忆)
r = eng.chat("我叫什么?", top_k=3)
print(r["response"])

# 信任反馈: 标记某条记忆有用 -> 信任上升, 以后更易被召回
eng.reinforce_fact(fid)
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
| `POST /chat` | `{message, top_k}` | `{response, used_memory, used_style, latency_ms}` |
| `POST /facts/add` | `{text, pinned}` | `{id}` |
| `POST /facts/delete` | `{id}` | `{ok}` |
| `POST /facts/update` | `{id, text, pinned}` | `{ok}` |
| `GET /facts/list` | - | `{facts}` |
| `POST /persona/create` | `{id, examples, desc}` | `{train_time_s}` |
| `POST /persona/switch` | `{id}` (null=无性格) | `{ok}` |
| `POST /persona/delete` | `{id}` | `{ok}` |
| `GET /persona/list` | - | `{personas}` |
| `GET /health` | - | `{ok}` |

## 集成到其他系统

本服务是一个独立进程 + 纯 HTTP 接口，零额外依赖即可启动，任何能发 HTTP 请求的宿主都能接入：

- 在请求管道里调 `/chat`，由本服务完成「记忆召回 → 性格示范 → DeepSeek 回答」全流程。
- 或先调 `/facts/*` 自行召回事实，再把结果注入你自己的 prompt。

进程托管（启动/健康检查/重启）交给宿主：用 `GET /health` 做就绪探针，进程退出后由宿主拉起即可。

## 实测状态

- ✅ 记忆层端到端跑通(增删改 + RAG召回 + DeepSeek对话, ~700ms)
- ✅ 改记忆即时反映到回答, 删记忆外科手术式隔离
- ✅ 信任层(`test_trust.py`): 强化 0.5→0.64, 衰减 0.5→0.125, 信任加权重排, pinned 不衰减
- ✅ 性格层(`test_persona.py`): 风格迁移 ~87%(100词→13词), 训练 ~77s/persona
- ✅ 检索后端对比: FAISS+BGE 0.79 >> 官方关键词式 Holographic 0.00(问答场景)

## 文件结构

```
memory_engine/
├── engine.py           # 核心引擎 (流水线组装 + 强化)
├── fact_store.py       # 事实层 (RAG + 增删改 + 热/冷 + 信任加权)
├── persona_manager.py  # 性格层 (多adapter + 切换)
├── deepseek_client.py  # DeepSeek 客户端
└── service.py          # HTTP service (对外接口)
test_trust.py           # 信任层测试
test_smoke.py           # 记忆层端到端测试
test_persona.py         # 性格层测试
```
