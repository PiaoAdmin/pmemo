# PMemo

PMemo 是基于 MemoryOS 扩展的 Python 对话记忆框架。在短期、中期、长期记忆之外增加关键记忆，并提供滚动摘要块、记忆提升、删除恢复和操作事件，用于保存用户偏好、约束与历史上下文。

本文对应本仓库已提交的 PMemo 0.1 实现（代码基线 `fb0c02f`），以根目录 `Memoryos` 类为入口。项目尚未提供独立 HTTP 服务或命令行聊天程序。

## 来源与扩展范围

本项目延续 [BAI-LAB/MemoryOS](https://github.com/BAI-LAB/MemoryOS) 的分层记忆设计与 ChromaDB 实现。短期队列、中期会话与页面、热度管理、长期用户画像与知识、对话生成及部分并行处理来自上游基础。感谢原项目作者的工作。

PMemo 的扩展集中在以下部分，相关实现记录见 [upgrade_versions](upgrade_versions)：

| 扩展 | 当前行为 | 主要文件 |
| --- | --- | --- |
| 关键记忆 | 手动写入偏好/约束，或按中期会话统计自动提升；保存来源、优先级和溯源信息 | `key_term.py`、`memoryos.py` |
| 长期摘要块 | 从短期滚动窗口保存带 step 范围的块，单独向量检索 | `long_term.py`、`updater.py` |
| 中期页面压缩 | 超过阈值时合并较早页面，保留尾部页面 | `mid_term.py` |
| 分层检索上下文 | 召回关键记忆、长期摘要、中期页面及用户/助手知识，按层组织上下文 | `retriever.py`、`memoryos.py` |
| 时间线与操作记录 | 对话写入递增 `global_step`；记忆操作记录到 `memory_events` | `storage_provider.py` |
| 删除与恢复 | 短期对话、关键记忆、中期会话、长期摘要支持软删除/恢复；知识项支持物理删除 | `memoryos.py`、`storage_provider.py` |

上表说明本仓库的扩展范围，不表示完整框架均为 PMemo 原创。上游论文指标、MCP、Playground 和其他发行包不作为本仓库的功能或评测结果。

## 记忆如何流转

```text
历史问答 ── add_memory ──> 短期队列（step、时间戳）
                              │ 达到容量，最早问答移出
                              v
                           Updater
                 ┌────────────┴─────────────┐
                 v                          v
          中期会话 / 页面              长期滚动摘要块
          主题合并、前后页链             step_start / step_end
          页面压缩、热度统计
                 │
                 ├── 热点分析 ──> 用户画像、用户/助手知识
                 └── 提升规则 ──> 关键记忆 <── 手动写入

查询 ── 并行召回 ──> 分层上下文 + 画像/知识 ──> LLM
                                                  │
                                       回复并回写短期队列
```

- **短期记忆**保存最近问答。达到容量时，Updater 移出最早的问答并转为中期页面，队列不会整批清空。
- **中期记忆**按主题组织会话与页面，保存 `pre_page`、`next_page`、`meta_info` 等对话连续性信息。会话容量不足时按访问频率淘汰。
- **长期记忆**包含画像、用户知识、助手知识和长期摘要块。当前滚动摘要路径直接保存窗口问答文本；“摘要块”不意味着每次都会额外调用 LLM 压缩。
- **关键记忆**来自显式写入或按命中率、命中次数、热度等规则从中期记忆提升。
- 生成提示中的记忆顺序为：**关键记忆 → 长期摘要 → 中期页面 → 短期对话**，并结合画像及知识。

## 环境与安装

项目声明 Python `>=3.10`，已提交的 `.python-version` 为 `3.10`。依赖包含 ChromaDB、NumPy、OpenAI SDK、Sentence Transformers 和 Transformers；本地向量模型会引入 PyTorch 等依赖。

使用 uv：

```bash
git clone https://github.com/PiaoAdmin/pmemo.git
cd pmemo
uv sync --python 3.10
```

`pyproject.toml` 的项目名仍为 `memoryp`，代码导入名为 `memoryos`。请在仓库根目录运行示例。

也可以使用 pip：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt 'numpy>=1.26.4,<2.0'
```

已提交版本使用 `chromadb==0.4.24`；pip 命令额外约束 NumPy `<2.0`，与 `pyproject.toml` 保持一致。

## 快速开始

以下示例使用同一 OpenAI 兼容服务提供聊天与 embedding。模型名应填写服务实际支持的名称：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-provider.example/v1"
export MEMORYOS_LLM_MODEL="your-chat-model"
export MEMORYOS_EMBED_MODEL="your-embedding-model"
```

保存为仓库根目录的 `example.py`，执行 `uv run python example.py`（pip 环境执行 `python example.py`）：

```python
import os
from memoryos import Memoryos

memory = Memoryos(
    user_id="demo_user",
    assistant_id="travel_assistant",
    data_storage_path="./data/demo",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_base_url=os.environ["OPENAI_BASE_URL"],
    llm_model=os.environ["MEMORYOS_LLM_MODEL"],
    embedding_model_name=os.environ["MEMORYOS_EMBED_MODEL"],
    use_embedding_api=True,
    short_term_capacity=4,
)

try:
    memory.add_memory("我喜欢京都的历史街区，也喜欢摄影。", "我会优先考虑文化和摄影体验。")
    key_id = memory.add_key_memory(
        "旅行预算不超过两万元，尽量避开拥挤地点。",
        category="constraint",
        priority=0.99,
    )
    # 自动检索、生成回复并保存本次问答。
    print(memory.get_response("帮我安排一天的京都行程。"))
    memory.delete_memory("key_memory", key_id, soft=True)
    memory.restore_memory("key_memory", key_id)
    print(memory.list_memory_events(limit=10))
finally:
    memory.close()
```

`add_memory` 用于导入已有问答；调用 `get_response` 后无需再手动保存同一问答。数据目录及标识不变时，重新初始化可加载已有记忆。`close()` 保存元数据，类也注册了进程退出保存回调。

### 本地 embedding

将构造参数改为 `embedding_model_name="all-MiniLM-L6-v2"`、`use_embedding_api=False` 即可使用本地模型，也可指定本地模型目录。首次使用通常需要下载模型。生成和部分记忆更新仍使用 LLM API。

切换向量模型或维度时应使用新的数据目录，避免与旧向量混用。已提交版本的远程 embedding 复用 `openai_api_key` 和 `openai_base_url`，构造器没有独立的 `embedding_api_key` / `embedding_base_url` 参数。

## 常用构造参数

| 参数 | 已提交版本默认值 | 含义 |
| --- | --- | --- |
| `user_id` | 必填 | 用户集合标识 |
| `openai_api_key` | 必填 | LLM 密钥；远程 embedding 复用 |
| `data_storage_path` | 必填 | 持久化根目录 |
| `openai_base_url` | `None` | 不传时使用 OpenAI API 地址 |
| `assistant_id` | `default_assistant_profile` | 助手知识标识 |
| `llm_model` | `gpt-4o-mini` | 生成与分析模型 |
| `embedding_model_name` | `all-MiniLM-L6-v2` | 本地或远程向量模型名 |
| `use_embedding_api` | `False` | 是否通过 API 获取向量 |
| `embedding_model_kwargs` | `None` | 向量模型选项；本地 BGE-M3 默认启用 `use_fp16` |
| `short_term_capacity` | `10` | 短期队列容量 |
| `mid_term_capacity` | `2000` | 中期会话数量上限 |
| `mid_term_page_merge_threshold` | `60` | 中期页面合并阈值 |
| `mid_term_page_merge_keep_tail` | `20` | 合并时保留的尾部页面数 |
| `retrieval_queue_capacity` | `7` | 中期页面召回数量上限 |
| `mid_term_similarity_threshold` | `0.6` | 更新时主题合并相似度阈值 |
| `mid_term_heat_threshold` | `5.0` | 热点画像/知识分析阈值 |
| `key_promotion_hit_rate` | `0.35` | 自动提升命中率阈值 |
| `key_promotion_min_hits` | `8` | 自动提升最少命中次数 |
| `key_promotion_heat` | `6.0` | 自动提升热度阈值 |

构造器还保留 `long_term_knowledge_capacity=100`，但当前初始化路径未将它接入知识容量限制，不应依赖其自动淘汰知识。建议使用关键字参数。

环境变量由示例和测试/评测脚本读取；直接创建 `Memoryos` 时需自行读取并传参。

## API 与删除语义

| 接口 | 用途 |
| --- | --- |
| `add_memory(user_input, agent_response, timestamp=None, meta_data=None)` | 写入一轮已有问答；`meta_data` 当前为保留参数 |
| `get_response(query, relationship_with_user="friend", style_hint="", user_conversation_meta_data=None)` | 检索、生成并保存问答，返回字符串 |
| `add_key_memory(text, category="preference", priority=0.95)` | 创建手动关键记忆，返回 ID |
| `delete_memory(memory_type, memory_id, soft=True)` | 按类型与 ID 删除 |
| `restore_memory(memory_type, memory_id)` | 恢复支持的软删除记忆 |
| `list_memory_events(limit=50)` | 返回最近的操作事件 |
| `get_user_profile_summary()` | 返回用户画像字典 |
| `get_assistant_knowledge_summary()` | 返回助手知识列表 |
| `force_mid_term_analysis()` | 临时降低阈值，分析最热会话中的未分析页面 |
| `close()` | 保存 JSON 元数据 |

统一删除接口支持：

| `memory_type` | `memory_id` | 软删除与恢复 |
| --- | --- | --- |
| `short_term` | 对话 step 的字符串，如 `"3"` | 支持 |
| `key_memory` | `key_id` | 支持 |
| `mid_term_session` | `session_id` | 支持，作用于会话 |
| `long_term_summary` | 摘要块 `ltm_id` | 支持；`long_term` 为其别名 |
| `user_knowledge` | 知识 ID | 仅物理删除，不支持恢复 |
| `assistant_knowledge` | 知识 ID | 仅物理删除，不支持恢复 |

前四类传入 `soft=False` 会物理删除，不能恢复。知识项即使传入 `soft=True` 仍物理删除。软删除通过状态过滤影响后续读取/检索，不等于从磁盘擦除。

删除原始记忆不会自动级联清理已派生的摘要、画像、知识或关键记忆。需要彻底移除信息时，应检查派生内容。

可通过 `memory.key_term_memory.list_key_memory()` 查看关键记忆，通过 `memory.short_term_memory.get_all()` 查看短期对话及 step。更底层的会话、知识与摘要接口见对应模块。

## 持久化结构

```text
<data_storage_path>/
└── chroma_storage/
    ├── chroma.sqlite3
    ├── <Chroma 管理的向量索引目录>/
    └── metadata_<user_id>_<assistant_id>.json
```

| Chroma 集合 | 内容 |
| --- | --- |
| `mid_term_memory_user_{user_id}` | 中期会话/页面向量 |
| `user_knowledge_{user_id}` | 用户知识 |
| `assistant_knowledge_{assistant_id}` | 助手知识 |
| `key_memory_{user_id}` | 关键记忆 |
| `long_term_summary_{user_id}` | 长期摘要块 |

JSON 元数据保存短期队列、会话头与页面备份、访问频率、堆状态、画像、全局 step、事件及关键记忆/摘要元信息。备份应在停止写入后复制整个 `chroma_storage`，仅复制 SQLite 或 JSON 无法完整保留状态。

集合按用户或助手分别命名，元数据按二者组合命名，这不是完整的多租户权限系统。同一目录下，同一用户的部分集合跨助手复用，同一助手知识也可能跨用户复用。需要独立状态时使用不同数据目录。标识需符合 Chroma 集合命名要求。

## 测试与 LoCoMo 评测

### 集成测试

设置前述 API 环境变量，在仓库根目录执行：

```bash
export MEMORYOS_TEST_DATA_PATH="./data/integration-test"
uv run python comprehensive_test.py
```

测试会调用真实 API 并写入数据，覆盖记忆写入/召回、中期合并、滚动窗口及部分删除恢复流程。包含依赖生成内容的语义检查，结果可能随模型波动，并非离线单元测试。可通过 `MEMORYOS_TEST_USER_ID`、`MEMORYOS_TEST_ASSISTANT_ID` 指定标识。

现有测试和评测配置含历史硬编码 API key 回退值。运行前请显式设置自己的 `OPENAI_API_KEY`，勿依赖或复制历史值。

### LoCoMo

- `eval/locomo10.json`：评测数据。
- `eval/main_loco_parse.py`：导入对话、构造记忆并生成答案。
- `eval/evalution_loco.py`：当前实现的 F1 和 BLEU-1 评分。
- `scripts/run_locomo_eval.sh`：提供 `quick`、`full`、`score` 模式。

**已提交的生成脚本仍从 `memoryP.memoryos` 导入，依赖旧目录布局；普通 `pmemo` 克隆下直接运行 `quick`/`full` 可能出现 `ModuleNotFoundError`。** 本文不将本地未发布的导入修复描述为已可用功能。

如需不修改文件试跑，可在根目录用模块别名兼容旧导入。下面限制为一个样本、20 条对话和 20 道问题，并单独保存结果：

```bash
export LOCOMO_MAX_SAMPLES=1
export LOCOMO_MAX_DIALOGS_PER_SAMPLE=20
export LOCOMO_MAX_QAS_PER_SAMPLE=20
export LOCOMO_OUTPUT_FILE="$PWD/eval/local_loco_results.json"
uv run python - <<'PY'
import os
import runpy
import sys
import types
from pathlib import Path

root = Path.cwd()
package = types.ModuleType("memoryP")
package.__path__ = [str(root)]
sys.modules["memoryP"] = package
sys.path.insert(0, str(root / "eval"))
os.chdir(root / "eval")
runpy.run_path("main_loco_parse.py", run_name="__main__")
PY
uv run python eval/evalution_loco.py eval/local_loco_results.json
```

该兼容命令处理旧包名，不代表已完成端到端评测验证。三个 `LOCOMO_MAX_*` 变量设为 `0` 可取消相应数量限制；`LOCOMO_RUN_TAG` 控制本次生成的存储目录标签。

已有结果可直接评分，无需生成答案：

```bash
bash scripts/run_locomo_eval.sh score "$PWD/eval/quick_loco_results.json"
```

结果文件可能来自受限样本和不同配置，不能据此认定完整 LoCoMo 表现。比较实验应记录数据范围、LLM、向量模型、记忆参数和评分实现；本文不宣称已复现上游论文指标。

## 代码导航

```text
memoryos.py             对外入口、记忆编排与回复生成
short_term.py           短期问答队列
mid_term.py             中期会话、页面、热度与压缩
long_term.py            画像、知识与长期摘要块
key_term.py             手动/自动关键记忆
storage_provider.py     ChromaDB 与 JSON 元数据
updater.py              短期转中期、对话链与滚动窗口
retriever.py            多路并行召回
utils.py                LLM、embedding、摘要及分析工具
prompts.py              提示词模板
comprehensive_test.py   真实 API 集成测试
eval/                   LoCoMo 生成与评分
scripts/                评测辅助脚本
upgrade_versions/       分阶段实现与修复记录
```

`main.py` 当前是占位入口。根目录模块可直接导入，无需下载完整上游仓库作为运行时依赖。

## 当前边界

- 这是本地持久化的实验框架，未提供鉴权、HTTP 服务、跨进程写锁或数据库事务协调。并行召回不代表支持同一存储的多进程安全写入。
- API embedding 客户端为进程级全局变量，不同实例采用不同服务配置时需注意相互覆盖。
- `memory_events` 是应用内操作记录，不是完整事件溯源或跨层自动回滚系统。`version` 和 `ttl_days` 字段也不代表完整版本管理或自动到期清理已实现。
- 模型或 API 请求异常时，部分路径返回错误文本或空召回；应用侧应检查返回值与日志。
- 根目录目前没有独立 LICENSE 文件；上游许可说明见 [MemoryOS](https://github.com/BAI-LAB/MemoryOS)。
