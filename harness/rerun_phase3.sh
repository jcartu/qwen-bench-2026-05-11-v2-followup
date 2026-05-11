#!/usr/bin/env bash
# rerun_phase3.sh — Re-run BF16+DFlash quality phase with reduced
# --gpu-memory-utilization (0.78 vs original 0.85) to avoid the silent OOM that
# killed the container mid-HumanEval on the first attempt.
#
# Preserves the failed run at configs/quality-bf16-dflash-n8-mt8192-FAILED/ for
# the record.
set -uo pipefail
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HARNESS_DIR/sweep_lib.sh"

# Override gpu-mem-util via env (sweep_lib.sh reads this if set)
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.78}"

STUDY_ROOT="${STUDY_ROOT:-/tmp/qwen-bench-2026-05-11-v2-followup}"
FAILED_DIR="$STUDY_ROOT/configs/quality-bf16-dflash-n8-mt8192"
RERUN_DIR="$STUDY_ROOT/configs/quality-bf16-dflash-n8-mt8192-rerun"

# Stash old results
if [ -d "$FAILED_DIR" ] && [ ! -d "${FAILED_DIR}-FAILED" ]; then
  mv "$FAILED_DIR" "${FAILED_DIR}-FAILED"
  echo "[stash] moved $FAILED_DIR -> ${FAILED_DIR}-FAILED"
fi

echo "[$(date '+%H:%M:%S MSK')] Phase 3 RE-RUN: BF16+DFlash n=8 @ mt=8192 with gpu-mem-util=$VLLM_GPU_MEM_UTIL"
mkdir -p "$RERUN_DIR"

# Launch with reduced memory pressure
launch_config "quality-bf16-dflash-n8-mt8192-rerun" "dflash" "$MODEL_BF16" "$DRAFTER_DFLASH" 8 32768 256 "$RERUN_DIR"

if ! wait_for_ready "$RERUN_DIR" 600; then
  echo "  [FAIL] not ready, aborting"; stop_config; exit 1
fi

echo "  [$(date '+%H:%M:%S')] settling 60s..."
settle 60

echo "  [$(date '+%H:%M:%S')] gates..."
if ! run_gates "$RERUN_DIR"; then
  echo "  [WARN] gates failed (continuing for data capture)"
fi

echo "  [$(date '+%H:%M:%S')] HumanEval + MBPP @ c=8 mt=8192 (~10 min)..."
run_quality_at "$RERUN_DIR" 8192

echo "  [$(date '+%H:%M:%S')] stopping container..."
stop_config

echo "  [$(date '+%H:%M:%S')] Phase 3 re-run done"
echo "  [$(date '+%H:%M:%S')] Restarting SOTA..."
systemctl --user start vllm-qwen36-27b-sota.service || true
