#!/usr/bin/env bash
# Step 7 - after step 5: leakage control + Phase-2/3 ideas re-tested at full scale (CPU, a few hours).
#   bash 7_run_phase3_full.sh                       (inside tmux, or with nohup)
# Options (env): WORKERS=N  EVENTS=path.npz  ONLY="C1 C3"  SEEDS=0,1,2  TRIALS=30 (Optuna; 0 = skip)
# Results: fism2_repro/results/<exp>_full*/ (+ p3_optuna_full/); pack with bash 6_pack_results.sh
set -uo pipefail
source "$(dirname "$0")/env.sh"
need_repo
exec > >(tee -a "$LOGDIR/7_phase3.log") 2>&1
cd "$REPO"
EVENTS="${EVENTS:-$(latest_events)}"
[ -n "$EVENTS" ] && [ -f "$EVENTS" ] || [ "$DRY_RUN" = "1" ] || die "no data/processed/events_latis*.npz - run bash 4_download_all.sh first"
TRIALS="${TRIALS:-30}"
log "dataset: $EVENTS   workers: $WORKERS   optuna trials: $TRIALS"

# id | config | extra arguments | what it tests
EXPS=(
  "C1a|p2_norm|--exclude-random 163 --exclude-seed 0|leakage control: drop 163 RANDOM train/val events (size-matched to E5), draw 0"
  "C1b|p2_norm|--exclude-random 163 --exclude-seed 1|leakage control, draw 1"
  "C1c|p2_norm|--exclude-random 163 --exclude-seed 2|leakage control, draw 2"
  "C2|p2_window||input window 15/30/45 min (FCN-pn, GRU-pn)"
  "C3|p3_arch||I-1 modern architectures: PatchTST, TSMixer, N-BEATS, ModernTCN (peaknorm)"
  "C4|p3_resid||I-2 physics residual on a power-law prior"
  "C5|p3_extreme||I-3 top-decile loss weight, input-noise augmentation"
  "C6|p3_type||I-4 flare-type experts / rise-time feature"
  "C7|p3_joint||I-6 joint vs per-band models"
  "C8|p3_features||I-7 static features + permutation importance"
  "C9|p3_early||I-8 early forecast: input ends 10/20/30 min before the peak"
  "C10|p3_final||combined GRU-final (45-min window + noise + rise feature), 5 seeds"
  "C11|p2_scale||scale-invariance probe: output shift for x0.1 / x10 inputs (1 seed)"
)
failed=()
for e in "${EXPS[@]}"; do
  IFS='|' read -r id cfg extra what <<< "$e"
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $id "* ]]; then continue; fi
  log "=== $id: $what"
  args=(--exp "$cfg" --protocol full --events "$EVENTS" --workers "$WORKERS")
  [ -n "$extra" ] && read -r -a ex <<< "$extra" && args+=("${ex[@]}")
  [ -n "${SEEDS:-}" ] && args+=(--seeds "$SEEDS")
  if ! run "$PY" run.py "${args[@]}"; then failed+=("$id"); log "!!! $id failed (continuing)"; fi
done
if [ "$TRIALS" -gt 0 ] && { [ -z "${ONLY:-}" ] || [[ " $ONLY " == *" C12 "* ]]; }; then
  log "=== C12: I-10 Optuna re-tuning on the train/val split (FCN-z per band, GRU-z), $TRIALS trials"
  run "$PY" scripts/tune_optuna.py --protocol full --events "$EVENTS" --config p2_zscore --spec FCN-z --family fcn \
      --trials "$TRIALS" --seeds 0 1 2 --workers 1 || { failed+=("C12-fcn"); log "!!! C12 FCN-z failed"; }
  run "$PY" scripts/tune_optuna.py --protocol full --events "$EVENTS" --config p2_zscore --spec GRU-z --family gru \
      --trials "$TRIALS" --seeds 0 1 2 --workers 1 || { failed+=("C12-gru"); log "!!! C12 GRU-z failed"; }
fi
log "finished. failed: ${failed[*]:-none}.  next: bash 6_pack_results.sh"
