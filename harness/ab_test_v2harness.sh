#!/usr/bin/env bash
# ab_test_v2harness.sh — Re-run FP8+MTP=3 on repne/vllm:latest with PATCHED harness
# Validates the body-only-glue fix on HumanEval. Also re-scores MBPP for comparison.
set -uo pipefail
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HARNESS_DIR/sweep_lib.sh"

# Override STRESS_HARNESS to use patched v2
export STRESS_HARNESS_V2="$HARNESS_DIR/stress_harness_v2.py"

export IMAGE="repne/vllm:latest"
export CONTAINER_NAME="vllm-ab-v2harness"

STUDY_ROOT="${STUDY_ROOT:-/tmp/qwen-bench-2026-05-11-v2-followup}"
OUT_DIR="$STUDY_ROOT/configs/ab-fp8-mtp3-latest-mt8192-v2harness"

mkdir -p "$OUT_DIR"

echo "[$(date '+%H:%M:%S MSK')] A/B-v2harness: FP8+MTP=3 on :latest @ mt=8192 with body-only-glue fix"

launch_config "ab-fp8-mtp3-latest-mt8192-v2harness" "mtp" "$MODEL_FP8" "" 3 32768 256 "$OUT_DIR"

if ! wait_for_ready "$OUT_DIR" 600; then
  echo "  [FAIL] not ready"; stop_config; exit 1
fi

echo "  [$(date '+%H:%M:%S')] settling 60s..."
settle 60

echo "  [$(date '+%H:%M:%S')] gates..."
run_gates "$OUT_DIR" || echo "  [WARN] gates incomplete"

echo "  [$(date '+%H:%M:%S')] HumanEval @ c=8 mt=8192 (~7 min) with v2 harness..."
"$PY3" "$STRESS_HARNESS_V2" \
  --url "http://localhost:${PORT}/v1/chat/completions" \
  --model "$SERVED_NAME" \
  --config-label "ab-fp8-mtp3-latest-mt8192-v2harness" \
  --benchmark humaneval \
  --problems-file "$PROBLEMS_DIR/humaneval.jsonl" \
  --concurrency 8 --max-tokens 8192 \
  --output "$OUT_DIR/humaneval.jsonl" \
  > "$OUT_DIR/humaneval.log" 2>&1

echo "  [$(date '+%H:%M:%S')] MBPP @ c=8 mt=8192 (~8 min) with v2 harness..."
"$PY3" "$STRESS_HARNESS_V2" \
  --url "http://localhost:${PORT}/v1/chat/completions" \
  --model "$SERVED_NAME" \
  --config-label "ab-fp8-mtp3-latest-mt8192-v2harness" \
  --benchmark mbpp \
  --problems-file "$PROBLEMS_DIR/mbpp.jsonl" \
  --concurrency 8 --max-tokens 8192 \
  --output "$OUT_DIR/mbpp.jsonl" \
  > "$OUT_DIR/mbpp.log" 2>&1

echo "  [$(date '+%H:%M:%S')] stopping..."
stop_config

echo "  [$(date '+%H:%M:%S')] v2 harness re-run done"
