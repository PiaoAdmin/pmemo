#!/usr/bin/env bash
# Usage:
#   source scripts/setup_eval_env.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Please source this file so env vars stay in your shell:"
  echo "  source scripts/setup_eval_env.sh"
  exit 1
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-hfhbxjmiwwthygrlpehackesymdqjtjlvdiksvqlvjompjys}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.siliconflow.cn/v1}"
export MEMORYOS_LLM_MODEL="${MEMORYOS_LLM_MODEL:-deepseek-ai/DeepSeek-V3}"
export MEMORYOS_EMBED_MODEL="${MEMORYOS_EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}"
export MEMORYOS_USE_EMBED_API="${MEMORYOS_USE_EMBED_API:-true}"

echo "Eval env is ready:"
echo "  OPENAI_BASE_URL=$OPENAI_BASE_URL"
echo "  MEMORYOS_LLM_MODEL=$MEMORYOS_LLM_MODEL"
echo "  MEMORYOS_EMBED_MODEL=$MEMORYOS_EMBED_MODEL"
