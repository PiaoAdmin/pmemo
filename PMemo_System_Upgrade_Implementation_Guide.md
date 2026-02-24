# PMemo 系统升级实施指南（仅限 memoryP）

## 1. 目标与范围

本指南用于将当前 `memoryP` 升级为以关键记忆为核心的 Agent Memory 框架 PMemo，满足以下目标：

1. 支持长对话下的上下文压缩与可控方向保持。
2. 保留分层记忆：短期记忆（STM）/中期记忆（MTM）/长期记忆（LTM）。
3. 引入关键记忆（KTM）：支持手动配置 + 自动提炼 + 动态增删 + 回溯。
4. 实现并行检索与统一组装：`【关键记忆】【长期记忆】【中期记忆】【短期记忆】【用户输入】`。
5. 所有记忆记录统一包含 `timestamp` 与 `step`。

约束：

1. 所有改造仅在 `memoryP` 目录内完成。
2. 不修改 `eval/`、`README` 等其他目录文件。
3. 优先复用当前 `memoryP` 现有模块，不重写整套框架。

---

## 2. 当前代码基线（memoryP）

当前核心模块：

1. `memoryos.py`：总编排，负责 add/retrieve/respond。
2. `updater.py`：STM 溢出后写入 MTM、摘要与关键词生成、会话连续性处理。
3. `mid_term.py`：中期会话聚类、检索、热度和 LFU。
4. `long_term.py`：用户画像与知识提取检索。
5. `retriever.py`：并行检索 MTM + 用户长期知识 + 助手长期知识。
6. `storage_provider.py`：Chroma + metadata JSON 的统一存储。
7. `key_term.py`：目前为空文件，可作为关键记忆模块入口。

现状问题：

1. MTM 的 `pages_backup` 随对话可能持续增长，存储和召回成本升高。
2. 关键记忆尚未形成独立数据结构和生命周期。
3. 删除/回溯能力不成体系（缺少统一操作接口和审计记录）。
4. 查询拼装尚未显式保障“关键记忆优先”策略。

---

## 3. 目标架构

### 3.1 四层记忆职责

1. STM（短期记忆）
- 结构：FIFO 队列。
- 职责：保留最近 N 轮原始对话。

2. MTM（中期记忆）
- 结构：按语义聚类的 session/page。
- 职责：承接 STM 溢出，做主题聚合与可检索中程上下文。

3. LTM（长期记忆）
- 结构：摘要化历史记忆 + 关键事实存档。
- 职责：长期回溯与低频但高价值信息保留。

4. KTM（关键记忆）
- 结构：可检索的“高优先级约束/偏好/任务状态”。
- 职责：控制对话方向，优先进入 prompt。
- 来源：用户手动 + 系统自动（由 MTM 命中行为触发）。

### 3.2 关键原则

1. 关键记忆优先级高于普通长期知识。
2. 自动关键记忆必须可解释（记录来源 session/page）。
3. 删除和回溯是一级能力，不是附加工具。
4. 任何自动总结都不应覆盖原始可回溯链路。

---

## 4. 数据模型与存储设计

### 4.1 通用字段约定

所有记忆条目统一增加：

1. `timestamp`: 生成或更新时间。
2. `step`: 对话轮次（全局递增）。
3. `status`: `active|deleted`。
4. `version`: 模型或总结版本标识。

### 4.2 关键记忆（KTM）数据结构

建议在 `storage_provider.py` 增加独立 collection：`key_memory_{user_id}`。

```json
{
  "key_id": "km_001",
  "user_id": "u1",
  "assistant_id": "a1",
  "text": "用户不吃香菜",
  "category": "preference|constraint|goal|persona|risk",
  "source": "manual|auto",
  "priority": 0.95,
  "confidence": 0.88,
  "ttl_days": null,
  "step": 128,
  "timestamp": "2026-02-23T10:00:00Z",
  "status": "active",
  "trace": {
    "session_ids": ["session_xxx"],
    "page_ids": ["page_xxx", "page_yyy"],
    "rule": "hit_rate_threshold"
  }
}
```

### 4.3 MTM 扩展字段（用于自动关键记忆触发）

在 session metadata 增加：

1. `hit_count_total`: 被检索命中次数。
2. `hit_count_window`: 时间窗口命中次数。
3. `hit_rate`: `hit_count_window / window_queries`。
4. `key_promoted`: 是否已提炼过关键记忆。
5. `compressed_at`: 最近压缩时间。
6. `summary_step_range`: 覆盖的对话轮次范围。

### 4.4 LTM 结构建议

LTM 不再仅依赖画像，补充“历史摘要块”：

```json
{
  "ltm_id": "ltm_seg_001",
  "text": "2026-01 至 2026-02 的长期互动总结...",
  "step_start": 1,
  "step_end": 120,
  "timestamp": "...",
  "source": "short_term_rollup",
  "status": "active"
}
```

---

## 5. 升级后的核心流程

### 5.1 写入流程（add memory）

1. `Memoryos.add_memory()` 写入 STM，并推进 `global_step`。
2. 当 STM 达到阈值，`Updater.process_short_term_to_mid_term()` 执行：
- STM evict 批次转 page。
- 生成主题摘要与关键词。
- 插入或合并 MTM session。
3. 同时触发两条更新：
- LTM：基于 STM evict 批次生成长期摘要块。
- KTM(auto)：检查 MTM hit-rate 与热度阈值，满足则提炼关键记忆。

### 5.2 自动关键记忆触发（KTM auto）

触发条件建议满足任意两项：

1. `session.hit_rate >= KEY_PROMOTION_HIT_RATE`（如 0.35）。
2. `session.hit_count_total >= KEY_PROMOTION_MIN_HITS`（如 8）。
3. `session.H_segment >= KEY_PROMOTION_HEAT`（如 6.0）。

触发后动作：

1. 从 session summary + 最近命中 page 生成 1~3 条关键记忆。
2. 写入 KTM（`source=auto`，含 trace）。
3. 将 session 标记 `key_promoted=true`，避免重复泛滥。
4. 对超长 session 执行动态总结与压缩（保留 tail pages 用于回溯）。

### 5.3 删除与回溯

统一通过 `memoryos.py` 暴露：

1. `delete_memory(memory_type, id, soft=True)`。
2. `restore_memory(memory_type, id)`。
3. `list_memory_events(limit)`（用于审计回放）。

删除规则：

1. soft delete：`status=deleted`，检索默认过滤。
2. hard delete：从 metadata + collection 物理删除（仅管理员或离线任务）。
3. 回溯：仅对 soft delete 生效。

---

## 6. 查询优化与 Prompt 组装

### 6.1 并行检索策略

在 `retriever.py` 中并行执行四路：

1. KTM 检索（高优先级，top_k_key）。
2. LTM 检索（top_k_ltm）。
3. MTM 检索（top_k_sessions + top_k_pages）。
4. STM 直接读取（最近 N 条，无需向量检索）。

### 6.2 排序融合

建议打分：

`score = w_sem * sim + w_recency * recency + w_priority * priority + w_source * source_weight`

建议初值：

1. KTM：`source_weight=1.2`
2. LTM：`source_weight=0.9`
3. MTM：`source_weight=1.0`
4. STM：不走打分，固定全量最近 N 条

### 6.3 上下文组装格式（固定）

最终 prompt 输入固定为：

1. `【关键记忆】`：KTM top-k（先 manual 后 auto）。
2. `【长期记忆】`：LTM 摘要与关键事实。
3. `【中期记忆】`：MTM 召回 page/session 摘要。
4. `【短期记忆】`：最近对话窗口。
5. `【用户输入】`：当前 query。

---

## 7. 文件级改造清单（仅 memoryP）

### 7.1 `memoryP/key_term.py`（从空文件改为核心模块）

新增 `KeyTermMemory` 类：

1. `add_manual_key_memory(text, category, priority, step, timestamp)`。
2. `add_auto_key_memory(text, trace, confidence, priority, step, timestamp)`。
3. `search_key_memory(query, top_k, threshold)`。
4. `delete_key_memory(key_id, soft=True)`。
5. `restore_key_memory(key_id)`。
6. `list_key_memory(status, source, limit)`。

### 7.2 `memoryP/storage_provider.py`

新增能力：

1. 初始化 `key_memory_collection`。
2. 新增 KTM CRUD 与向量检索接口。
3. metadata 新增：
- `memory_events`（审计日志）
- `global_step`
- `key_memory_index`（可选缓存）
4. 为 MTM session 增加命中统计字段读写。

### 7.3 `memoryP/mid_term.py`

新增能力：

1. session 命中统计更新（`hit_count_total`, `hit_count_window`, `hit_rate`）。
2. session 压缩接口（超长 session 动态摘要）。
3. `maybe_promote_to_key_memory(...)` 触发钩子（可由 updater 调用）。

### 7.4 `memoryP/long_term.py`

改造方向：

1. 新增历史摘要块写入接口（区别于用户画像）。
2. 新增 `search_long_term_summary(...)`。
3. 保留原画像与知识抽取，降低其在 prompt 中占比。

### 7.5 `memoryP/updater.py`

新增编排：

1. STM->MTM 后触发 LTM rollup。
2. 依据 MTM 指标触发 KTM auto 提炼。
3. 对超长 MTM session 做压缩与摘要更新。

### 7.6 `memoryP/retriever.py`

新增四路并行检索：

1. `_retrieve_key_memory(...)`
2. `_retrieve_long_term_summary(...)`
3. `_retrieve_mid_term_context(...)`（保留）
4. STM 直接读取（保留）

返回结构升级：

```json
{
  "retrieved_key_memory": [],
  "retrieved_long_term": [],
  "retrieved_pages": [],
  "retrieved_short_term": [],
  "retrieved_at": "..."
}
```

### 7.7 `memoryP/memoryos.py`

新增/调整公开接口：

1. `add_key_memory(...)`, `delete_key_memory(...)`, `restore_key_memory(...)`。
2. `delete_memory(...)`, `restore_memory(...)`（统一入口）。
3. `get_response(...)` 内改为新拼装顺序。
4. 增加配置对象传入（避免硬编码阈值）。

---

## 8. 配置项设计（建议新增 `memoryP/config_pmemo.py`）

建议集中配置：

1. `STM_CAPACITY`
2. `MTM_CAPACITY`
3. `LTM_CAPACITY`
4. `KTM_CAPACITY`
5. `KEY_PROMOTION_HIT_RATE`
6. `KEY_PROMOTION_MIN_HITS`
7. `KEY_PROMOTION_HEAT`
8. `TOP_K_KEY`, `TOP_K_LTM`, `TOP_K_MTM_PAGES`
9. `ENABLE_SOFT_DELETE`
10. `ENABLE_AUTO_KEY_PROMOTION`

---

## 9. 评测与验收（LoCoMo）

### 9.1 目标

使用 LoCoMo 验证升级对长期记忆问答的提升，重点看：

1. F1（主指标）。
2. 类别维度表现（尤其多跳、时间与对抗类问题）。
3. 响应稳定性（答案长度、格式一致性）。

### 9.2 在 memoryP 内的评测执行方式

为满足“仅在 memoryP 操作”的约束，建议在 `memoryP` 内新增适配脚本（后续实现）：

1. `locomo_runner.py`：读取 LoCoMo 样本并驱动 `Memoryos`。
2. `locomo_metrics.py`：计算 F1/分类别指标并输出报告。
3. `locomo_report.md`：记录参数、版本、结果、误差案例。

### 9.3 验收标准（建议）

1. 功能验收：
- 手动关键记忆增删改查可用。
- 自动关键记忆可触发且可追溯来源。
- 软删除和回溯可用。

2. 质量验收：
- LoCoMo 总体 F1 不低于基线。
- Category 2/3/5 至少两个类别有提升。
- 单轮检索延迟可控（并行后无明显退化）。

---

## 10. 分阶段实施计划

### Phase 1：数据层与关键记忆基础（低风险）

1. 实现 `key_term.py` + `storage_provider.py` KTM CRUD。
2. 打通手动关键记忆 API。
3. 增加 `global_step` 与 `memory_events`。

完成标志：

1. 可新增/检索/删除/恢复关键记忆。
2. 每次操作有审计记录。

### Phase 2：检索和 prompt 升级（中风险）

1. `retriever.py` 新增 KTM 并行检索。
2. `memoryos.py` 改为固定五段式上下文拼装。

完成标志：

1. 响应中可观察到关键记忆优先生效。
2. 上下文长度受控且结构稳定。

### Phase 3：自动关键记忆与中期压缩（中高风险）

1. `mid_term.py` 增加 hit-rate 指标。
2. `updater.py` 实现自动提炼与 session 压缩。

完成标志：

1. 自动关键记忆生成数量可控。
2. 超长 session 不再无限膨胀。

### Phase 4：LoCoMo 回归与参数调优（中风险）

1. 在 `memoryP` 内完成 LoCoMo 评测脚本和报告。
2. 固化默认参数和回归门槛。

完成标志：

1. 有可复现实验命令。
2. 有基线与升级版对比报告。

---

## 11. 风险与回滚

主要风险：

1. 自动关键记忆噪声过高，导致错误先验干扰回答。
2. 过度压缩 MTM 造成细节丢失。
3. 并行检索引入线程竞争与超时问题。

控制策略：

1. 为 auto KTM 设置 `confidence` 门限和人工开关。
2. MTM 压缩保留 raw tail，并支持按需回放。
3. 并行检索设置超时和降级策略（单路失败不阻塞整体）。

回滚策略：

1. 配置开关禁用 `ENABLE_AUTO_KEY_PROMOTION`。
2. 检索层降级为旧三路（MTM + user LTM + assistant LTM）。
3. 保留兼容字段，避免 metadata 不可逆。

---

## 12. 建议的下一步执行顺序

1. 先完成 `key_term.py` 与 `storage_provider.py` 的 KTM 数据层。
2. 再改 `retriever.py` 与 `memoryos.py`，让关键记忆进入主链路。
3. 最后改 `updater.py` + `mid_term.py` 的自动提炼和压缩逻辑。
4. 在 `memoryP` 内补齐 LoCoMo 评测脚本，形成基线/升级对比。

如果按这个顺序实施，可以在不打断现有主流程的情况下逐步上线，并且每一阶段都可独立验证和回滚。
