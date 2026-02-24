# V5 编排层 API 与审计能力

## 目标
- 在不重写框架的前提下，补齐统一管理接口与生命周期事件。

## 变更内容
- 文件：`memoryos.py`
- 初始化新增：
  - `KeyTermMemory` 注入
  - 自动提炼阈值参数：`key_promotion_hit_rate/min_hits/heat`
- `add_memory()` 升级：
  - 统一递增 `global_step`
  - QA 条目增加 `step/status/version`
  - STM 溢出后串联：
    - `Updater.process_short_term_to_mid_term()`
    - `_rollup_long_term_summary()`
    - `_auto_promote_key_memory()`
- 新增公开接口：
  - `add_key_memory`
  - `delete_key_memory`
  - `restore_key_memory`
  - `delete_memory(memory_type, id, soft=True)`
  - `restore_memory(memory_type, id)`
  - `list_memory_events(limit)`

- 文件：`long_term.py`
- 新增长期摘要块接口：
  - `add_summary_segment`
  - `search_long_term_summary`
  - `list_long_term_summaries`
