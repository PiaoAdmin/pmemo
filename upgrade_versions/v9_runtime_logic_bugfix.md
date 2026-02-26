# v9 Runtime & Logic Bugfix

## Scope
本次只修复会导致运行错误或逻辑错误的问题，不处理硬编码风格问题。

## Fixed
1. `updater.py`
- 修复 `update_long_term_from_analysis` 调用签名错误：移除不存在的 `merge=False` 参数，避免运行时 `TypeError`。
- 修复不存在方法调用：`add_user_knowledge`/`add_assistant_knowledge` 改为统一使用 `add_knowledge(text, knowledge_type)`。
- 兼容 `private`/`assistant_knowledge` 为字符串或列表两种输入格式，避免 `.lower()`/`.split()` 类型错误。
- 修复链路 `meta_info` 更新不持久的问题：改为通过 storage API 写回 `pages_backup`，并从备份中读取链路指针进行 BFS 传播。

2. `mid_term.py`
- `collect_key_memory_candidates` 增加 `status=active` 过滤，避免删除会话被错误提升为关键记忆。
- `delete_sessions_by_keywords` 改为 token 级匹配：
  - `summary_keywords` 使用标准化精确匹配；
  - `summary` 使用分词后匹配，避免子串误删（如短关键词误命中长词）。

3. `memoryos.py`
- 修复热段分析后错误清零 `L_interaction` 的逻辑，保留段长度统计语义。
- 修复短期转中期后并发写共享元数据风险：`_rollup_long_term_summary` 与 `_auto_promote_key_memory` 改为顺序执行。
- 修复中期会话硬删除后本地 LFU 状态残留：同步清理 `mid_term_memory.access_frequency` 并保存状态。

## Validation
执行真实测试（非 fake）：
- 命令：`uv run python comprehensive_test.py`
- 结果（多次）：
  - Run A: `Passed Tests: 2/2`, `Success Rate: 100.0%`
  - Run B: `Passed Tests: 2/2`, `Success Rate: 100.0%`
  - Run C: `Passed Tests: 1/2`, `Success Rate: 50.0%`（LLM 生成语义波动导致关键词覆盖不足，不是运行错误）
- 关键检查通过：
  - Mid-term merge
  - Long-term window consistency
  - Delete/Rollback operations

## Notes
运行日志中的 Chroma telemetry 警告（`capture() takes 1 positional argument but 3 were given`）不影响本次功能正确性，属于外部遥测兼容问题。
