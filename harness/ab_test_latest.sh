#!/usr/bin/env bash
# ab_test_latest.sh — A/B test FP8+MTP=3 on repne/vllm:latest vs :v2
# Reuses sweep_lib.sh launch_config + run_quality_at, overriding IMAGE.
set -uo pipefail
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HARNESS_DIR/sweep_lib.sh"

export IMAGE="repne/vllm:latest"
export CONTAINER_NAME="vllm-ab-latest"

STUDY_ROOT="${STUDY_ROOT:-/tmp/qwen-bench-2026-05-11-v2-followup}"
OUT_DIR="$STUDY_ROOT/configs/ab-fp8-mtp3-latest-mt8192"

mkdir -p "$OUT_DIR"

echo "[$(date '+%H:%M:%S MSK')] A/B: FP8+MTP=3 on repne/vllm:latest @ mt=8192"

launch_config "ab-fp8-mtp3-latest-mt8192" "mtp" "$MODEL_FP8" "" 3 32768 256 "$OUT_DIR"

if ! wait_for_ready "$OUT_DIR" 600; then
  echo "  [FAIL] not ready"; stop_config; exit 1
fi

echo "  [$(date '+%H:%M:%S')] settling 60s..."
settle 60

echo "  [$(date '+%H:%M:%S')] gates..."
run_gates "$OUT_DIR" || echo "  [WARN] gates incomplete"

echo "  [$(date '+%H:%M:%S')] HumanEval only @ c=8 mt=8192 (~7 min)..."
"$PY3" "$STRESS_HARNESS" \
  --url "http://localhost:${PORT}/v1/chat/completions" \
  --model "$SERVED_NAME" \
  --config-label "ab-fp8-mtp3-latest-mt8192" \
  --benchmark humaneval \
  --problems-file "$PROBLEMS_DIR/humaneval.jsonl" \
  --concurrency 8 --max-tokens 8192 \
  --output "$OUT_DIR/humaneval.jsonl" \
  > "$OUT_DIR/humaneval.log" 2>&1

echo "  [$(date '+%H:%M:%S')] stopping..."
stop_config

echo "  [$(date '+%H:%M:%S')] A/B done"
