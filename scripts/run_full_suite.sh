#!/usr/bin/env bash
# Paper-setting ("full" protocol) runs after the open data were downloaded and built:
#   python scripts/download/run_all.py --stage test   (calibration, check PASS)
#   python scripts/download/run_all.py --stage all    (-> data/processed/events_latis.npz)
# Usage: bash scripts/run_full_suite.sh [workers]      results -> results/<exp>_full*/
set -euo pipefail
W=${1:-4}
PY=${PY:-python3}
cd "$(dirname "$0")/.."
# E1 reproduction of the 8 paper models (+ persistence / climatology), 3 seeds
$PY run.py --exp repro_cv96 --protocol full --workers "$W"
# E2 preprocessing: peak normalisation / z-score (GRU, FCN)
$PY run.py --exp p2_norm   --protocol full --workers "$W"
$PY run.py --exp p2_zscore --protocol full --workers "$W"
# E3 final candidates: 5-seed ensembles (GRU-peaknorm, FCN-z)
$PY run.py --exp p3_ensemble --protocol full --workers "$W"
# E4 input series: GOES XRS-B vs FISM2 0.1-0.8 nm as input (same models)
$PY run.py --exp p2_norm --protocol full --input goes_b --workers "$W"
$PY run.py --exp p2_norm --protocol full --input fism   --workers "$W"
# E5 leakage: drop training flares within 24 h of any test flare
$PY run.py --exp p2_norm --protocol full --exclude-near-test-hours 24 --workers "$W"
echo "done - summaries: results/*_full*/summary.csv"
