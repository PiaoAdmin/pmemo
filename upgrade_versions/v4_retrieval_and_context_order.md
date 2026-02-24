# V4 检索并行化与固定上下文顺序

## 目标
- 按 PMemo 方案升级并行检索与上下文组装顺序。

## 变更内容
- 文件：`retriever.py`
- 新增并行检索任务：
  - `_retrieve_key_memory`
  - `_retrieve_long_term_summary`
- 保留并行任务：
  - `_retrieve_mid_term_context`
  - `_retrieve_user_knowledge`
  - `_retrieve_assistant_knowledge`
- 返回结构升级：
  - `retrieved_key_memory`
  - `retrieved_long_term`
  - `retrieved_pages`
  - `retrieved_short_term`

- 文件：`memoryos.py`
- `get_response()` 改为固定组装顺序：
  - `【关键记忆】`
  - `【长期记忆】`
  - `【中期记忆】`
  - `【短期记忆】`
  - 用户当前输入（通过 prompt query 注入）
