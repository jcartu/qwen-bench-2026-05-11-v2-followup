#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_all.sh — 2026-05-11 v2-followup study orchestrator
#
# Four phases (sequential, share one GPU pair):
#   1. SPEED: FP8+MTP=3 on repne/vllm:v2 (15-cell decode + prefill + 4 gates)
#   2. SPEED: FP8+MTP=5 on repne/vllm:v2
#   3. QUALITY: BF16+DFlash n=8 @ max_tokens=8192 (HumanEval + MBPP)
#   4. QUALITY: FP8+MTP=3 @ max_tokens=8192
#
# Total expected: ~2h 25m. Pinned to GPU 0+1; GPU 2 untouched.
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd "$HARNESS_DIR/.." && pwd)"
source "$HARNESS_DIR/sweep_lib.sh"

LOG_DIR="$STUDY_ROOT/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_all_$(date +%Y%m%d_%H%M%S).log"

START_TS=$(date +%s)
declare -a PHASE_ELAPSED=()

phase_header() {
  local n="$1" total="$2" name="$3" details="$4"
  local now=$(date +%s)
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "[$(date '+%H:%M:%S MSK')] Phase $n/$total: $name"
  echo "  $details"
  eta_remaining $((n-1)) "$total" $((now - START_TS)) "v2-followup"
  echo "═══════════════════════════════════════════════════════════════"
}

run_speed_phase() {
  local label="$1" method="$2" model="$3" drafter="$4"
  local num_spec="$5" max_batched="$6" capture="$7"
  local out_dir="$STUDY_ROOT/configs/${label}"
  mkdir -p "$out_dir"
  local p_start=$(date +%s)

  echo "  [launch] $method model=$model num_spec=$num_spec batched=$max_batched cap=$capture"
  launch_config "$label" "$method" "$model" "$drafter" "$num_spec" "$max_batched" "$capture" "$out_dir"
  if ! wait_for_ready "$out_dir" 600; then
    echo "  [FAIL] $label not ready, aborting phase"; stop_config; return 1
  fi
  echo "  [$(date '+%H:%M:%S')] settling 60s..."
  settle 60
  echo "  [$(date '+%H:%M:%S')] gates..."
  if ! run_gates "$out_dir"; then
    echo "  [WARN] gates failed for $label (continuing for data capture)"
  fi
  echo "  [$(date '+%H:%M:%S')] throughput (15 cells, ~20 min)..."
  run_throughput "$out_dir"
  echo "  [$(date '+%H:%M:%S')] prefill (5 contexts, ~3 min)..."
  run_prefill "$out_dir"
  echo "  [$(date '+%H:%M:%S')] stopping container..."
  stop_config
  local p_end=$(date +%s)
  echo "  [done] $label in $((p_end - p_start))s"
  echo "$label,$((p_end - p_start))" >> "$STUDY_ROOT/configs/_elapsed.csv"
}

run_quality_phase() {
  local label="$1" method="$2" model="$3" drafter="$4"
  local num_spec="$5" max_batched="$6" capture="$7" max_tokens="$8"
  local out_dir="$STUDY_ROOT/configs/${label}"
  mkdir -p "$out_dir"
  local p_start=$(date +%s)

  echo "  [launch] $method model=$model num_spec=$num_spec max_tokens=$max_tokens"
  launch_config "$label" "$method" "$model" "$drafter" "$num_spec" "$max_batched" "$capture" "$out_dir"
  if ! wait_for_ready "$out_dir" 600; then
    echo "  [FAIL] $label not ready, aborting phase"; stop_config; return 1
  fi
  echo "  [$(date '+%H:%M:%S')] settling 60s..."
  settle 60
  echo "  [$(date '+%H:%M:%S')] gates..."
  if ! run_gates "$out_dir"; then
    echo "  [WARN] gates failed for $label (continuing for data capture)"
  fi
  echo "  [$(date '+%H:%M:%S')] HumanEval + MBPP @ c=8 max_tokens=$max_tokens (~25 min)..."
  run_quality_at "$out_dir" "$max_tokens"
  echo "  [$(date '+%H:%M:%S')] stopping container..."
  stop_config
  local p_end=$(date +%s)
  echo "  [done] $label in $((p_end - p_start))s"
  echo "$label,$((p_end - p_start))" >> "$STUDY_ROOT/configs/_elapsed.csv"
}

main() {
  echo "[$(date '+%H:%M:%S MSK')] v2-followup study starting"
  echo "  STUDY_ROOT=$STUDY_ROOT"
  echo "  IMAGE=$IMAGE"
  echo "  GPUs: $GPU_0_UUID, $GPU_1_UUID"
  echo ""
  echo "label,seconds" > "$STUDY_ROOT/configs/_elapsed.csv"

  # ── Phase 1: Speed FP8+MTP=3
  phase_header 1 4 "Speed FP8+MTP=3 on repne/vllm:v2" \
    "model=$MODEL_FP8 num_spec=3 batched=32768 capture=256"
  run_speed_phase "speed-fp8-mtp3" "mtp" "$MODEL_FP8" "-" 3 32768 256 || true

  # ── Phase 2: Speed FP8+MTP=5
  phase_header 2 4 "Speed FP8+MTP=5 on repne/vllm:v2" \
    "model=$MODEL_FP8 num_spec=5 batched=32768 capture=256"
  run_speed_phase "speed-fp8-mtp5" "mtp" "$MODEL_FP8" "-" 5 32768 256 || true

  # ── Phase 3: Quality BF16+DFlash n=8 @ max_tokens=8192
  phase_header 3 4 "Quality BF16+DFlash n=8 @ max_tokens=8192" \
    "model=$MODEL_BF16 drafter=$DRAFTER_DFLASH num_spec=8 batched=32768 capture=256"
  run_quality_phase "quality-bf16-dflash-n8-mt8192" "dflash" "$MODEL_BF16" "$DRAFTER_DFLASH" 8 32768 256 8192 || true

  # ── Phase 4: Quality FP8+MTP=3 @ max_tokens=8192
  phase_header 4 4 "Quality FP8+MTP=3 @ max_tokens=8192" \
    "model=$MODEL_FP8 num_spec=3 batched=32768 capture=256"
  run_quality_phase "quality-fp8-mtp3-mt8192" "mtp" "$MODEL_FP8" "-" 3 32768 256 8192 || true

  local end_ts=$(date +%s)
  local total=$((end_ts - START_TS))
  local h=$((total / 3600))
  local m=$(( (total % 3600) / 60 ))
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "[v2-FOLLOWUP COMPLETE] 4 phases in ${h}h ${m}m"
  echo "═══════════════════════════════════════════════════════════════"

  # Restart SOTA service
  echo "[$(date '+%H:%M:%S MSK')] Restarting SOTA 27B service..."
  systemctl --user start vllm-qwen36-27b-sota.service || true
}

main "$@" 2>&1 | tee "$RUN_LOG"
