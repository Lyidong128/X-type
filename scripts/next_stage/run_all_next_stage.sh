#!/usr/bin/env bash
set -u

ROOT="/workspace"
OUT_DIR="$ROOT/outputs/next_stage_analysis"
LOG_FILE="$OUT_DIR/run_all_log.txt"
mkdir -p "$OUT_DIR"

QUICK_FLAG=""
if [[ "${1:-}" == "--quick" ]]; then
  QUICK_FLAG="--quick"
fi

declare -a STEPS=(
  "python3 \"$ROOT/scripts/next_stage/check_symmetry.py\" $QUICK_FLAG"
  "python3 \"$ROOT/scripts/next_stage/run_wilson_loop.py\" $QUICK_FLAG"
  "python3 \"$ROOT/scripts/next_stage/run_ribbon_xy.py\" $QUICK_FLAG"
  "python3 \"$ROOT/scripts/next_stage/run_obc_wavefunction.py\" $QUICK_FLAG"
  "python3 \"$ROOT/scripts/next_stage/run_finite_size_ipr.py\" $QUICK_FLAG"
  "python3 \"$ROOT/scripts/next_stage/run_transition_refinement.py\" $QUICK_FLAG"
  "python3 \"$ROOT/scripts/next_stage/run_robustness.py\" $QUICK_FLAG"
  "python3 \"$ROOT/scripts/next_stage/run_spin_topology_optional.py\" $QUICK_FLAG"
  "python3 \"$ROOT/scripts/next_stage/generate_next_stage_report.py\""
)

echo "run_all_next_stage start $(date -u +"%Y-%m-%dT%H:%M:%SZ")" > "$LOG_FILE"
echo "quick_mode=${QUICK_FLAG:-false}" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

STATUS_SUMMARY=()
for cmd in "${STEPS[@]}"; do
  echo "[RUN] $cmd" | tee -a "$LOG_FILE"
  if bash -lc "$cmd" >> "$LOG_FILE" 2>&1; then
    STATUS_SUMMARY+=("[OK] $cmd")
    echo "[OK] $cmd" | tee -a "$LOG_FILE"
  else
    STATUS_SUMMARY+=("[FAIL] $cmd")
    echo "[FAIL] $cmd" | tee -a "$LOG_FILE"
  fi
  echo "" >> "$LOG_FILE"
done

echo "=== module status summary ===" | tee -a "$LOG_FILE"
for line in "${STATUS_SUMMARY[@]}"; do
  echo "$line" | tee -a "$LOG_FILE"
done

echo "run_all_next_stage end $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "$LOG_FILE"
