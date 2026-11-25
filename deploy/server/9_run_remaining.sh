#!/usr/bin/env bash
# Step 9 - the remaining pieces after steps 5/7/8 (GPU, ~15 min):
#   R1  size-matched random controls for the AR-level exclusion (step 8, G2 only; already-finished runs are skipped)
#   R2  rebuilt test set as its own npz (for the analysis tables) and the full analysis tables
#       (notebooks/analyze_full_server.py -> results/full_server/*.csv)
#   R3  split conformal intervals calibrated on the validation events (notebooks/split_conformal_full.py)
#   then 6_pack_results.sh
#   CUDA_VISIBLE_DEVICES=0,1,3,4 WORKERS=8 bash 9_run_remaining.sh     (inside tmux)
set -uo pipefail
source "$(dirname "$0")/env.sh"
need_repo
exec > >(tee -a "$LOGDIR/9_remaining.log") 2>&1
cd "$REPO"
EVENTS="${EVENTS:-$(latest_events)}"
[ -n "$EVENTS" ] && [ -f "$EVENTS" ] || die "no data/processed/events_latis*.npz"
failed=()
log "=== R1: AR-exclusion random controls"
ONLY="G2" bash "$BUNDLE_DIR/8_run_gpu_followups.sh" || failed+=("R1")
log "=== R2: rebuilt test npz + analysis tables"
REB="data/processed/events_test96_rebuilt.npz"
if [ ! -f "$REB" ]; then
  run "$PY" scripts/download/build_events.py --stage test --out "$REB" || failed+=("R2-build")
fi
run "$PY" notebooks/analyze_full_server.py --results results --rebuilt "$REB" --out results/full_server || failed+=("R2-analyze")
log "=== R3: split conformal (validation-calibrated)"
run "$PY" notebooks/split_conformal_full.py --events "$EVENTS" --results results --out results/full_server \
    --models p3_ensemble_full_gpu:FCN-z p3_ensemble_full_gpu:GRU-peaknorm p2_zscore_full_gpu5:GRU-z p2_zscore_full_gpu5:FCN-z \
    --avg "p2_zscore_full_gpu5:FCN-z+p2_zscore_full_gpu5:GRU-z" || failed+=("R3")
log "finished. failed: ${failed[*]:-none}"
bash "$BUNDLE_DIR/6_pack_results.sh"
