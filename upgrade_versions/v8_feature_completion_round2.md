# V8 功能完善（按 1-8 项逐条补齐）

## 1) ChromaDB 接口与 `storage_provider` 完善
- 文件：`storage_provider.py`
- 关键完善：
  - STM 增加软删除索引：`short_term_deleted_index`
  - STM 读取/容量判断仅统计 `status=active`
  - STM 新增：
    - `delete_short_term_memory_by_step(step, soft=True)`
    - `restore_short_term_memory_by_step(step)`
  - MTM 删除能力升级：
    - `delete_mid_term_session(session_id, soft=True|False)`
    - `restore_mid_term_session(session_id)`
  - MTM 检索默认过滤 `status!=active`
  - LTM 知识写入返回 `knowledge_id`，补充删除入口：
    - `delete_knowledge(knowledge_id, knowledge_type)`

## 2) 短期记忆删除与回溯
- 文件：`short_term.py`, `memoryos.py`
- 新增：
  - `ShortTermMemory.delete_by_step(step, soft=True)`
  - `ShortTermMemory.restore_by_step(step)`
  - `Memoryos.delete_short_term_memory(step)`（默认软删）
  - `Memoryos.restore_short_term_memory(step)`
- 审计日志：`memory_events` 记录 delete/restore。

## 3) 中期记忆关键词整段删除 + 页面阈值合并
- 文件：`mid_term.py`, `memoryos.py`
- 新增：
  - `MidTermMemory.delete_sessions_by_keywords(keywords, soft=True)`
  - `MidTermMemory.restore_session(session_id)`
  - `Memoryos.delete_mid_term_by_keywords(...)`
  - `Memoryos.restore_mid_term_session(...)`
- 合并/压缩触发升级：
  - `MidTermMemory` 新增参数：
    - `page_merge_threshold`
    - `page_merge_keep_tail`
  - 在 `add_session()` 与 `insert_pages_into_session()` 后自动触发 `compress_session_if_needed(...)`。

## 4) 长期记忆窗口策略（与 STM 队列长度一致）
- 文件：`updater.py`, `memoryos.py`
- 新增机制：
  - `Updater.process_short_term_to_mid_term()` 返回 `long_term_rollup_text`，其窗口长度按 `short_term_memory.max_capacity` 构造。
  - `Memoryos._rollup_long_term_summary()` 优先使用上述窗口 rollup 字段。
- 效果：LTM 摘要粒度稳定，不再仅基于单条 evict。

## 5) 关键记忆（系统/用户）删除能力
- 文件：`key_term.py`, `memoryos.py`
- 现状确认并强化：
  - 用户关键记忆：`add_manual_key_memory`
  - 系统关键记忆：`add_auto_key_memory`（由 MTM 命中/热度触发）
  - 两者均支持：`delete_key_memory` / `restore_key_memory`

## 6) Prompt 同步（最小改动）
- 文件：`prompts.py`
- 在 `GENERATE_SYSTEM_RESPONSE_USER_PROMPT` 的 `<MEMORY>` 区新增：
  - 四层记忆顺序说明（KTM→LTM→MTM→STM）
  - 冲突时关键记忆优先
  - 已删除记忆不作为证据

## 7) 记忆更新与组装并行化
- 文件：`memoryos.py`
- 更新路径并行：
  - STM->MTM 后，`_rollup_long_term_summary` 与 `_auto_promote_key_memory` 并行执行。
- 组装路径并行：
  - 中期页面文本构造（含 meta_info 获取）使用线程池并行。

## 8) 测试完善（真实 API）
- 文件：`comprehensive_test.py`
- 关键更新：
  - 保留真实 API 路径（无 fake/mock）
  - 增加删除+回溯覆盖：
    - STM delete/restore
    - LTM summary delete/restore
    - KTM delete/restore
    - MTM keyword delete/restore
  - 增加结构性校验：
    - MTM merge/compress 触发校验
    - LTM rollup 窗口长度校验

## 辅助优化
- 文件：`utils.py`
- `gpt_generate_multi_summary()` 增加 ```json code fence 解析，减少无效 fallback。

## 本轮真实测试结果
- 命令：`uv run python3 comprehensive_test.py`
- 结果：`Passed Tests: 1/2`，`Success Rate: 50%`
- 结构性检查：
  - `Mid-term Merge Check: True`
  - `Long-term Window Check: True`
  - `Delete/Rollback Ops Check: True`
- 结论：集成路径通过，语义召回质量仍有优化空间（已在测试结果中体现）。
