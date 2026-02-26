#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="$ROOT_DIR/eval"

MODE="${1:-full}"
RESULT_FILE="${2:-$EVAL_DIR/all_loco_results.json}"
HAS_RESULT_ARG="${2:-}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_locomo_eval.sh full [result_json_path]
  ./scripts/run_locomo_eval.sh quick [result_json_path]
  ./scripts/run_locomo_eval.sh score [result_json_path]

Modes:
  full   Run LoCoMo generation (main_loco_parse.py) then score F1/BLEU-1
  quick  Run a short smoke test (limited samples/dialogs/QAs) then score
  score  Only score an existing result JSON
EOF
}

if [[ ! -f "$EVAL_DIR/locomo10.json" ]]; then
  echo "[ERROR] Missing benchmark file: $EVAL_DIR/locomo10.json"
  echo "Please place LoCoMo test file there first."
  exit 1
fi

# Default envs (aligned with comprehensive_test.py), can be overridden by user env.
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-hfhbxjmiwwthygrlpehackesymdqjtjlvdiksvqlvjompjys}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.siliconflow.cn/v1}"
export MEMORYOS_LLM_MODEL="${MEMORYOS_LLM_MODEL:-deepseek-ai/DeepSeek-V3}"
export MEMORYOS_EMBED_MODEL="${MEMORYOS_EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}"

case "$MODE" in
  full)
    echo "[1/2] Generating answers on LoCoMo subset..."
    (
      cd "$EVAL_DIR"
      uv run python main_loco_parse.py
    )

    echo "[2/2] Scoring with F1 and BLEU-1..."
    (
      cd "$EVAL_DIR"
      uv run python evalution_loco.py "$RESULT_FILE"
    )
    ;;
  quick)
    QUICK_OUTPUT_FILE="${HAS_RESULT_ARG:-$EVAL_DIR/quick_loco_results.json}"
    echo "[1/2] Generating QUICK answers on LoCoMo..."
    (
      cd "$EVAL_DIR"
      # Fast path defaults; can be overridden by user env.
      export LOCOMO_MAX_SAMPLES="${LOCOMO_MAX_SAMPLES:-1}"
      export LOCOMO_MAX_DIALOGS_PER_SAMPLE="${LOCOMO_MAX_DIALOGS_PER_SAMPLE:-20}"
      export LOCOMO_MAX_QAS_PER_SAMPLE="${LOCOMO_MAX_QAS_PER_SAMPLE:-20}"
      # Keep output path absolute to avoid path confusion after `cd eval`.
      export LOCOMO_OUTPUT_FILE="${LOCOMO_OUTPUT_FILE:-$QUICK_OUTPUT_FILE}"
      uv run python main_loco_parse.py
    )

    echo "[2/2] Scoring QUICK run with F1 and BLEU-1..."
    (
      cd "$EVAL_DIR"
      uv run python evalution_loco.py "$QUICK_OUTPUT_FILE"
    )
    ;;
  score)
    if [[ ! -f "$RESULT_FILE" ]]; then
      echo "[ERROR] Result file not found: $RESULT_FILE"
      exit 1
    fi
    echo "[Score] Evaluating: $RESULT_FILE"
    (
      cd "$EVAL_DIR"
      uv run python evalution_loco.py "$RESULT_FILE"
    )
    ;;
  *)
    usage
    exit 1
    ;;
esac
