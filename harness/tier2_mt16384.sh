#!/usr/bin/env bash
# Tier 2: mt=16384 ceiling test with PATCHED stress_harness.py.
# Validates the offline-rescore 95.1% claim for BF16+DFlash N=8 by giving
# the model enough budget to never truncate.
#
# Two configs on repne/vllm:v2:
#   1. BF16+DFlash N=8  mt=16384  → expect ~94-95% HE (recover offline ceiling)
#   2. FP8+MTP=3        mt=16384  → expect ~92-94% HE
set -euo pipefail

STUDY_ROOT="/tmp/qwen-bench-2026-05-11-v2-followup"
export STUDY_ROOT
source "$STUDY_ROOT/harness/sweep_lib.sh"

STRESS_HARNESS="/home/josh/qwen-vllm-test/bench/stress-harness/stress_harness.py"
PROBLEMS_DIR="/home/josh/qwen-vllm-test/bench/stress-harness/problems"

if ! grep -q "smart_glue_humaneval" "$STRESS_HARNESS"; then
    echo "FATAL: $STRESS_HARNESS missing smart_glue_humaneval. Aborting." >&2
    exit 1
fi
echo "[OK] Using patched harness: $STRESS_HARNESS"

run_he_only() {
  local out_dir="$1" max_tokens="${2:-16384}" concurrency="${3:-8}"
  python3 "$STRESS_HARNESS" \
    --url "http://localhost:${PORT}/v1/chat/completions" \
    --model "$SERVED_NAME" \
    --benchmark humaneval \
    --problems-file "$PROBLEMS_DIR/humaneval.jsonl" \
    --output "$out_dir/humaneval.jsonl" \
    --max-tokens "$max_tokens" \
    --concurrency "$concurrency" \
    --request-timeout 600 \
    --config-label "$(basename "$out_dir")" \
    < /dev/null >> "$out_dir/humaneval.log" 2>&1
}

# ─── Run 1: BF16 + DFlash N=8 mt=16384 ─────────────────────────────────────
LABEL_1="tier2-bf16-dflash-n8-mt16384-patched"
OUT_1="$STUDY_ROOT/configs/$LABEL_1"
mkdir -p "$OUT_1"
echo "================================================================"
echo "[$(date +%T)] Tier 2 Run 1: $LABEL_1"
echo "================================================================"
launch_config "$LABEL_1" "dflash" "$MODEL_BF16" "$DRAFTER_DFLASH" 8 8192 256 "$OUT_1"
if ! wait_for_ready "$OUT_1" 600; then
  echo "[FAIL] Run 1 failed to start. See $OUT_1/server.log"
  exit 2
fi
echo "[$(date +%T)] Settling 60s..."
settle 60
echo "[$(date +%T)] Running HumanEval mt=16384 (patched harness)..."
run_he_only "$OUT_1" 16384 8
echo "[$(date +%T)] Done. Summary:"
python3 -c "import json; d=json.load(open('$OUT_1/humaneval_summary.json')); print(f'  pass_rate={d[\"pass_rate\"]*100:.1f}%  ok={d[\"failure_mode_breakdown\"].get(\"ok\",0)}/{d[\"total_problems\"]}  empty={d[\"failure_mode_breakdown\"].get(\"empty_response\",0)}  test_fail={d[\"failure_mode_breakdown\"].get(\"test_fail\",0)}')"

# Tear down before next
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
sleep 10

# ─── Run 2: FP8 + MTP=3 mt=16384 ──────────────────────────────────────────
LABEL_2="tier2-fp8-mtp3-mt16384-patched"
OUT_2="$STUDY_ROOT/configs/$LABEL_2"
mkdir -p "$OUT_2"
echo "================================================================"
echo "[$(date +%T)] Tier 2 Run 2: $LABEL_2"
echo "================================================================"
launch_config "$LABEL_2" "mtp" "$MODEL_FP8" "-" 3 32758 256 "$OUT_2"
if ! wait_for_ready "$OUT_2" 600; then
  echo "[FAIL] Run 2 failed to start. See $OUT_2/server.log"
  exit 2
fi
echo "[$(date +%T)] Settling 60s..."
settle 60
echo "[$(date +%T)] Running HumanEval mt=16384 (patched harness)..."
run_he_only "$OUT_2" 16384 8
echo "[$(date +%T)] Done. Summary:"
python3 -c "import json; d=json.load(open('$OUT_2/humaneval_summary.json')); print(f'  pass_rate={d[\"pass_rate\"]*100:.1f}%  ok={d[\"failure_mode_breakdown\"].get(\"ok\",0)}/{d[\"total_problems\"]}  empty={d[\"failure_mode_breakdown\"].get(\"empty_response\",0)}  test_fail={d[\"failure_mode_breakdown\"].get(\"test_fail\",0)}')"

# Tear down vLLM and restore SOTA on :latest
echo "================================================================"
echo "[$(date +%T)] Tier 2 done. Restoring SOTA on :latest..."
echo "================================================================"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
systemctl --user start vllm-qwen36-27b-sota.service
echo "[$(date +%T)] DONE. Tier 2 mt=16384 complete."
