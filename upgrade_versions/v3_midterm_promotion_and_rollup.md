# V3 中期统计与自动提炼

## 目标
- 让 MTM 支持自动关键记忆提炼与会话压缩触发。

## 变更内容
- 文件：`mid_term.py`
- 会话初始化与兼容补全新增：
  - `hit_count_total` / `hit_count_window` / `window_queries` / `hit_rate`
  - `key_promoted` / `compressed_at` / `summary_step_range`
- 检索命中后更新命中统计，并回写 metadata。
- 新增 `collect_key_memory_candidates(...)`：
  - 规则：`hit_rate`、`hit_count_total`、`H_segment` 三条件满足任意两项。
  - 生成候选关键记忆（含 trace）。
  - 标记 `key_promoted=true` 防止重复泛滥。
- 新增 `compress_session_if_needed(...)`，控制超长 session 增长。

## Updater 联动
- 文件：`updater.py`
- `process_short_term_to_mid_term()` 增加返回值：
  - `step_start` / `step_end`
  - `input_text_for_summary`
  - evicted 批次信息
- 页面结构补齐 `step/status/version`。
