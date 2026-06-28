"""memory-engine: 给 LLM API（DeepSeek/OpenAI/Claude 等）提供可编辑记忆 + 可切换性格的外挂库。

架构(基于实测数据选型):
    用户消息
      -> ① RAG 索引召回 top-k 相关事实 (FAISS + BGE, 可扩展)
      -> ② top-k 全部披露进 context (retrieve-then-read)
      -> ③ 矛盾检测: 写入时自动发现并降权旧记忆
      -> ④ 性格 adapter 小模型生成风格示范 (方案A)
      -> ⑤ 组装 prompt = [性格示范] + [披露事实] + [用户消息]
      -> ⑥ LLM 回答
      -> ⑦ 使用反馈: 被实际采纳的记忆才强化 trust

为什么这样选(都有实测支撑):
    - 事实用 RAG+披露: 三方对比中 RAG 2.0 / 披露 1.83 > 参数化 0.96
    - 事实可即时增删改: 向量操作是毫秒级、外科手术式隔离
    - 性格用多 LoRA adapter: 每个性格物理隔离, 切换=换adapter, 天然规避灾难性遗忘
"""

__version__ = "0.1.0"
