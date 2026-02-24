# V2 关键记忆模块实现

## 目标
- 将空文件 `key_term.py` 升级为可用的 KTM 模块。

## 变更内容
- 文件：`key_term.py`
- 新增 `KeyTermMemory` 类，支持：
  - `add_manual_key_memory`
  - `add_auto_key_memory`
  - `search_key_memory`
  - `delete_key_memory`（软删/硬删）
  - `restore_key_memory`
  - `list_key_memory`
- 关键记忆标准字段统一：
  - `key_id` / `user_id` / `assistant_id`
  - `text` / `category` / `source` / `priority` / `confidence`
  - `step` / `timestamp` / `status` / `version`
  - `trace`
- 每次增删恢复同步写入 `memory_events` 审计日志。
