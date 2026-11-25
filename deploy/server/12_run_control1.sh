#!/usr/bin/env bash
# Step 12 - control experiment 1 (design: docs/decisions.md, 2026-11-25). GPU only; 25 trainings (5 architectures x 5 seeds), minutes.
#   bash 12_run_control1.sh        (inside tmux, after steps 1-11; needs the results of p2_zscore_full_in-goes_b_p1 and
#                                    p2_zscore_full_in-goes_convnn_p2nn in results/)
# The FISM2-input z-score models are retrained with the train/val assignment of the GOES-input models (982 / 122 flares,
# run.py --split-like-goes), so that "GOES input - FISM2 input" compares inputs only. Then notebooks/control1_analysis.py
# scores the pre-registered family of two comparisons and writes results/full_server/control1*.csv.
set -uo pipefail
source "$(dirname "$0")/env.sh"
need_repo
exec > >(tee -a "$LOGDIR/12_control1.log") 2>&1
cd "$REPO"
EVENTS="${EVENTS:-$(latest_events)}"
[ -n "$EVENTS" ] && [ -f "$EVENTS" ] || [ "$DRY_RUN" = "1" ] || die "no data/processed/events_latis*.npz"
"$PY" -c 'import torch; print("torch", torch.__version__, "cuda:", torch.cuda.is_available(), "visible devices:", torch.cuda.device_count())'
for need in p2_zscore_full_in-goes_b_p1 p2_zscore_full_in-goes_convnn_p2nn; do
  ls results/$need/oofL_GRU-z_s0.npy >/dev/null 2>&1 || [ "$DRY_RUN" = "1" ] || die "results/$need missing (GOES-input runs of steps 5-10)"
done
log "dataset: $EVENTS   workers: $WORKERS   device: $FISM_DEVICE"
run "$PY" run.py --exp p2_zscore --protocol full --events "$EVENTS" --workers "$WORKERS" --seeds 0,1,2,3,4 --split-like-goes --tag c1 \
  || die "training failed"
run "$PY" notebooks/control1_analysis.py --results results --events "$EVENTS" || die "analysis failed"
log "finished. results: results/p2_zscore_full_splitgoes_c1/, results/full_server/control1.csv, control1_conformal.csv"
