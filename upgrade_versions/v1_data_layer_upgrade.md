# V1 数据层升级

## 目标
- 为 PMemo 升级提供统一底座：`global_step`、`memory_events`、关键记忆存储、长期摘要存储。

## 变更内容
- 文件：`storage_provider.py`
- 新增 collection：
  - `key_memory_{user_id}`
  - `long_term_summary_{user_id}`
- metadata schema 新增字段：
  - `global_step`
  - `memory_events`
  - `key_memories`
  - `long_term_summaries`
- 新增接口：
  - `get_global_step` / `increment_global_step`
  - `record_memory_event` / `list_memory_events`
  - `add/search/list/delete/restore key memory`
  - `add/search/list/delete/restore long-term summary`
- 中期会话存储补全标准字段：
  - `timestamp` / `step` / `status` / `version`
  - `hit_count_total` / `hit_count_window` / `window_queries` / `hit_rate`
  - `key_promoted` / `compressed_at` / `summary_step_range`

## 兼容策略
- 通过 `_ensure_metadata_schema()` 对旧 metadata 做无损补全。
- `mid_term_collection.add` 改为 `upsert`，避免重复写入时报错。
