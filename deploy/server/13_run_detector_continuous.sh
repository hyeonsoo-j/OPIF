#!/usr/bin/env bash
# Step 13 - analysis A1: the armed peak detector on continuous GOES XRS-B 1-min series (design: docs/decisions.md,
# 2026-11-25 "continuous detector"). NO training, no GPU needed. Downloads the missing daily files of 2014 (GOES-15) and
# 2022 (GOES-16), science-quality and operational (about 1,100 files, a few hundred MB; already downloaded days are reused),
# then runs scripts/continuous_detector.py. Typically 20-60 min, mostly download.
#   bash 13_run_detector_continuous.sh            (inside tmux)
set -uo pipefail
source "$(dirname "$0")/env.sh"
need_repo
exec > >(tee -a "$LOGDIR/13_detector_continuous.log") 2>&1
cd "$REPO"
[ -f data/processed/flare_catalogue.csv ] || [ "$DRY_RUN" = "1" ] || die "data/processed/flare_catalogue.csv missing (steps 1-4)"
log "A1 continuous detector: root data/download, catalogue data/processed/flare_catalogue.csv"
run "$PY" scripts/continuous_detector.py --root data/download --catalogue data/processed/flare_catalogue.csv --out results/full_server \
  || die "continuous detector failed"
log "finished. results: results/full_server/continuous_detector_summary.csv, _declarations.csv, _listed.csv"
