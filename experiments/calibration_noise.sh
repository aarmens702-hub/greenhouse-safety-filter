#!/bin/bash
# Calibration study, sensor-noise check (pre-registered, docs/preregistration_filter.md, "Bie diagnosis"):
# arm a5 on Amsterdam 2010 day 59 with noise seeds 11 and 23, and with noise x2 (seed 7). About 20-30 min per run.
# Usage: bash experiments/calibration_noise.sh [OUT]   (default outputs/bie_noise.jsonl; frozen: results/filter/bie_noise.jsonl)
# Was greenleaf-filter/bie_noise.sh.
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python}
OUT=${1:-outputs/bie_noise.jsonl}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
C="--ctrl rb_tuned --loc Amsterdam --year 2010 --start 59 --plant bie --limit 85 --h 4 --obj mindev --model-frac 1 --lowrh --observer --noise --out $OUT"
nice $PY -m safety_filter.run_season_calib $C --noise-seed 11 > /dev/null 2>&1 &
nice $PY -m safety_filter.run_season_calib $C --noise-seed 23 > /dev/null 2>&1 &
wait
nice $PY -m safety_filter.run_season_calib $C --noise-seed 7 --noise-mult 2 > /dev/null 2>&1
echo NOISE_DONE
