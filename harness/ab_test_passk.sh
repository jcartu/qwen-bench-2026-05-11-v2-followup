#!/usr/bin/env bash
# ab_test_passk.sh — pass@5 study on :latest MTP=3 with smart_glue harness
# Repne's ask: pass@5 to test capability ceiling vs single-shot reliability
set -uo pipefail
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HARNESS_DIR/sweep_lib.sh"

export STRESS_HARNESS_PASSK="$HARNESS_DIR/stress_harness_passk.py"
export IMAGE="repne/vllm:latest"
export CONTAINER_NAME="vllm-ab-passk"

STUDY_ROOT="${STUDY_ROOT:-/tmp/qwen-bench-2026-05-11-v2-followup}"
OUT_DIR="$STUDY_ROOT/configs/ab-fp8-mtp3-latest-pass5"
TEMP="${TEMP:-0.8}"
N_SAMPLES="${N_SAMPLES:-5}"

mkdir -p "$OUT_DIR"

echo "[$(date '+%H:%M:%S MSK')] pass@$N_SAMPLES on :latest MTP=3 mt=8192 temp=$TEMP"

launch_config "ab-fp8-mtp3-latest-pass5" "mtp" "$MODEL_FP8" "" 3 32768 256 "$OUT_DIR"

if ! wait_for_ready "$OUT_DIR" 600; then
  echo "  [FAIL] not ready"; stop_config; exit 1
fi

echo "  [$(date '+%H:%M:%S')] settling 60s..."
settle 60

echo "  [$(date '+%H:%M:%S')] gates..."
run_gates "$OUT_DIR" || echo "  [WARN] gates incomplete"

echo "  [$(date '+%H:%M:%S')] HumanEval pass@$N_SAMPLES @ c=8 mt=8192 temp=$TEMP (~30 min)..."
"$PY3" "$STRESS_HARNESS_PASSK" \
  --url "http://localhost:${PORT}/v1/chat/completions" \
  --model "$SERVED_NAME" \
  --config-label "ab-fp8-mtp3-latest-pass5" \
  --benchmark humaneval \
  --problems-file "$PROBLEMS_DIR/humaneval.jsonl" \
  --concurrency 8 --max-tokens 8192 \
  --temperature "$TEMP" --n-samples "$N_SAMPLES" \
  --output "$OUT_DIR/humaneval.jsonl" \
  > "$OUT_DIR/humaneval.log" 2>&1

echo "  [$(date '+%H:%M:%S')] stopping..."
stop_config

echo "  [$(date '+%H:%M:%S')] pass@$N_SAMPLES done"
