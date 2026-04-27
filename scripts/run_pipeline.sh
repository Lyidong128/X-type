#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAGE="${1:-all}"

usage() {
  echo "Usage: scripts/run_pipeline.sh [scan|full|special|reselected|all]"
  echo "  scan      -> run base topology scan"
  echo "  full      -> build full per-point artifacts package"
  echo "  special   -> filter and package topological special points"
  echo "  reselected-> run reselected analysis workflow"
  echo "  all       -> run scan -> full -> special"
}

run_scan() {
  echo "[pipeline] stage=scan"
  python3 scripts/run_scan.py
}

run_full() {
  echo "[pipeline] stage=full"
  python3 scripts/run_all_points_full_package.py --obc-k 64 --obc-min-k 12
}

run_special() {
  echo "[pipeline] stage=special"
  python3 scripts/package_topology_special_points.py
}

run_reselected() {
  echo "[pipeline] stage=reselected"
  python3 scripts/run_reselected_analysis.py \
    --package-dir outputs/physical_special_points_package \
    --output-dir results_reselected \
    --config scripts/reselected_analysis_config.json
}

case "$STAGE" in
  scan)
    run_scan
    ;;
  full)
    run_full
    ;;
  special)
    run_special
    ;;
  reselected)
    run_reselected
    ;;
  all)
    run_scan
    run_full
    run_special
    ;;
  *)
    usage
    exit 1
    ;;
esac

echo "[pipeline] done stage=${STAGE}"
