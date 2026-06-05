#!/usr/bin/env bash
set -u

ROOT="/workspace/outputs/km_soc_topology_analysis"
mkdir -p "$ROOT"
LOG="$ROOT/run_all_log.txt"
: > "$LOG"

declare -a CMDS=(
  "python3 scripts/km_soc_analysis/run_gap_z2_scan.py"
  "python3 scripts/km_soc_analysis/run_spin_chern.py"
  "python3 scripts/km_soc_analysis/run_wilson_loop_flow.py"
  "python3 scripts/km_soc_analysis/analyze_gap_closing.py"
  "python3 scripts/km_soc_analysis/analyze_band_inversion.py"
  "python3 scripts/km_soc_analysis/check_chiral_symmetry_breaking.py"
  "python3 scripts/km_soc_analysis/run_fu_kane_parity_optional.py"
  "python3 scripts/km_soc_analysis/generate_km_soc_report.py"
)

echo "KM SOC analysis run start: $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "$LOG"
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

echo "KM SOC analysis status summary:" | tee -a "$LOG"
for item in "${status_table[@]}"; do
  echo "$item" | tee -a "$LOG"
done

echo "log=$LOG"
