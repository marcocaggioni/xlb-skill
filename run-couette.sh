#!/usr/bin/env bash
# Couette example — dry-run by default (lattice check only).
#   ./run-couette.sh            dry run
#   ./run-couette.sh --run      gamma = 10 check, then the full sweep + post
#   ./run-couette.sh --post     post-process runs/ into flow_curve.csv/png
set -e
cd "$(dirname "$0")/examples/couette"
PY="uv run --project ../.. python"
$PY lattice.py
if [ "$1" = "--post" ] 2>/dev/null; then $PY couette_post.py; exit 0; fi
if [ "$1" = "--run" ] 2>/dev/null; then $PY couette_sweep.py --run; $PY couette_post.py; exit 0; fi
$PY couette_sweep.py   # dry run: prints the plan, runs nothing
echo "Dry run only. Pass --run to execute the solver sweep."
