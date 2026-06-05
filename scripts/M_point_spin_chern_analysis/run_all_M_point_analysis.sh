#!/usr/bin/env bash
set -u

ROOT="/workspace/outputs/M_point_spin_chern_analysis"
mkdir -p "$ROOT"
LOG="$ROOT/run_all_log.txt"
: > "$LOG"

declare -a CMDS=(
  "python3 scripts/M_point_spin_chern_analysis/run_M_gap_scan.py"
  "python3 scripts/M_point_spin_chern_analysis/run_spin_chern_convergence.py"
  "python3 scripts/M_point_spin_chern_analysis/analyze_M_band_inversion.py"
  "python3 scripts/M_point_spin_chern_analysis/extract_M_effective_mass.py"
  "python3 scripts/M_point_spin_chern_analysis/compare_z2_spin_chern.py"
  "python3 scripts/M_point_spin_chern_analysis/generate_M_point_report.py"
)

echo "M-point spin-Chern run start: $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "$LOG"
echo "" | tee -a "$LOG"

status_table=()
for cmd in "${CMDS[@]}"; do
  echo ">>> RUN: $cmd" | tee -a "$LOG"
  if eval "$cmd" >>"$LOG" 2>&1; then
    echo ">>> DONE: $cmd" | tee -a "$LOG"
    status_table+=("OK|$cmd")
  else
    code=$?
    echo ">>> FAIL($code): $cmd" | tee -a "$LOG"
    status_table+=("FAIL($code)|$cmd")
  fi
  echo "" | tee -a "$LOG"
done

echo "M-point spin-Chern status summary:" | tee -a "$LOG"
for item in "${status_table[@]}"; do
  echo "$item" | tee -a "$LOG"
done

echo "log=$LOG"
