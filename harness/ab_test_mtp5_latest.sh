#!/usr/bin/env bash
# ab_test_mtp5_latest.sh — FP8+MTP=5 on repne/vllm:latest, HE + MBPP @ mt=8192
# Completes the MTP=3 vs MTP=5 quality matrix on the winning image.
set -uo pipefail
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HARNESS_DIR/sweep_lib.sh"

export IMAGE="repne/vllm:latest"
export CONTAINER_NAME="vllm-ab-mtp5-latest"

STUDY_ROOT="${STUDY_ROOT:-/tmp/qwen-bench-2026-05-11-v2-followup}"
OUT_DIR="$STUDY_ROOT/configs/ab-fp8-mtp5-latest-mt8192"

mkdir -p "$OUT_DIR"

echo "[$(date '+%H:%M:%S MSK')] A/B-2: FP8+MTP=5 on repne/vllm:latest @ mt=8192"

launch_config "ab-fp8-mtp5-latest-mt8192" "mtp" "$MODEL_FP8" "" 5 32768 256 "$OUT_DIR"

if ! wait_for_ready "$OUT_DIR" 600; then
  echo "  [FAIL] not ready"; stop_config; exit 1
fi

echo "  [$(date '+%H:%M:%S')] settling 60s..."
settle 60

echo "  [$(date '+%H:%M:%S')] gates..."
run_gates "$OUT_DIR" || echo "  [WARN] gates incomplete"

echo "  [$(date '+%H:%M:%S')] HumanEval @ c=8 mt=8192 (~7 min)..."
"$PY3" "$STRESS_HARNESS" \
  --url "http://localhost:${PORT}/v1/chat/completions" \
  --model "$SERVED_NAME" \
  --config-label "ab-fp8-mtp5-latest-mt8192" \
  --benchmark humaneval \
  --problems-file "$PROBLEMS_DIR/humaneval.jsonl" \
  --concurrency 8 --max-tokens 8192 \
  --output "$OUT_DIR/humaneval.jsonl" \
  > "$OUT_DIR/humaneval.log" 2>&1

echo "  [$(date '+%H:%M:%S')] MBPP @ c=8 mt=8192 (~8 min)..."
"$PY3" "$STRESS_HARNESS" \
  --url "http://localhost:${PORT}/v1/chat/completions" \
  --model "$SERVED_NAME" \
  --config-label "ab-fp8-mtp5-latest-mt8192" \
  --benchmark mbpp \
  --problems-file "$PROBLEMS_DIR/mbpp.jsonl" \
  --concurrency 8 --max-tokens 8192 \
  --output "$OUT_DIR/mbpp.jsonl" \
  > "$OUT_DIR/mbpp.log" 2>&1

echo "  [$(date '+%H:%M:%S')] stopping..."
stop_config

echo "  [$(date '+%H:%M:%S')] A/B-2 done"
