#!/usr/bin/env bash
# Step 2 - download 3 test events from the real servers and run the whole chain once (~1-3 min).
# Uses a separate folder (data/download_smoke) so nothing here is reused by the real run.
#   bash 2_smoke_test.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_repo
exec > >(tee -a "$LOGDIR/2_smoke.log") 2>&1
cd "$REPO"
SM="data/download_smoke"
run "$PY" scripts/download/run_all.py --stage test --limit 3 --root "$SM" || true
n_f=$(ls -1 "$SM"/fism2/*.npz 2>/dev/null | wc -l)
n_g=$(find "$SM/goes" -name '*.nc' 2>/dev/null | wc -l)
log "FISM2 windows: $n_f/3   GOES files: $n_g"
if [ -f "$SM/logs/fism2_errors.csv" ] || [ -f "$SM/logs/goes_errors.csv" ]; then
  log "errors were logged:"; tail -n 5 "$SM"/logs/*_errors.csv 2>/dev/null
fi
[ "$n_f" -ge 3 ] || die "FISM2 download failed - send run_logs/2_smoke.log"
[ "$n_g" -ge 1 ] || log "WARNING: no GOES file (input will fall back to FISM2 0.1-0.8 nm) - send run_logs/2_smoke.log"
log "smoke test OK (the calibration status printed above uses only 3 events - ignore it). next: bash 3_calibrate.sh"
