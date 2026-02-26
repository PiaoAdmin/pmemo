# eval/config.py
# API 配置文件 - 集中管理所有API参数

import os

# --- 基本配置 ---
# 默认值与 comprehensive_test.py 保持一致，优先读取环境变量。
API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "sk-hfhbxjmiwwthygrlpehackesymdqjtjlvdiksvqlvjompjys"
)
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
LLM_MODEL = os.getenv("MEMORYOS_LLM_MODEL", "deepseek-ai/DeepSeek-V3")
EMBEDDING_MODEL = os.getenv("MEMORYOS_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")

# --- Embedding 配置 ---
USE_EMBEDDING_API = os.getenv("MEMORYOS_USE_EMBED_API", "true").lower() in {"1", "true", "yes", "on"}

# --- 记忆系统配置 ---
H_THRESHOLD = float(os.getenv("MEMORYOS_H_THRESHOLD", "5.0"))  # 热度阈值
MID_TERM_CAPACITY = int(os.getenv("MEMORYOS_MID_TERM_CAPACITY", "200"))  # 中期记忆容量
SHORT_TERM_CAPACITY = int(os.getenv("MEMORYOS_SHORT_TERM_CAPACITY", "7"))  # 短期记忆容量
