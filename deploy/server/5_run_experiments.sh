#!/usr/bin/env bash
# Step 5 - paper-setting experiments: train ~772 / val ~96 on the downloaded events, test on the
# 96 Zenodo events (python run.py --protocol full). Results: fism2_repro/results/<exp>_full*/
#   bash 5_run_experiments.sh                    (inside tmux/screen, or with nohup)
# Options (env): WORKERS=N (parallel training jobs; default = all cores; models train on CPU)
#                EVENTS=path.npz (default: newest data/processed/events_latis*.npz)
#                SEEDS=0,1,2 (override the configs' seeds)   ONLY="E1 E3" (subset of the list below)
set -uo pipefail
source "$(dirname "$0")/env.sh"
need_repo
exec > >(tee -a "$LOGDIR/5_experiments.log") 2>&1
cd "$REPO"
EVENTS="${EVENTS:-$(latest_events)}"
[ -n "$EVENTS" ] && [ -f "$EVENTS" ] || [ "$DRY_RUN" = "1" ] || die "no data/processed/events_latis*.npz - run bash 4_download_all.sh first"
log "dataset: $EVENTS   workers: $WORKERS"

# id | config | extra arguments | what it tests
EXPS=(
  "E1|repro_cv96||paper's 8 models + persistence/climatology (3 seeds)"
  "E2a|p2_norm||peak normalisation (GRU, FCN)"
  "E2b|p2_zscore||z-score preprocessing (FCN, GRU, LSTM, RNN, Transformer)"
  "E3|p3_ensemble||final candidates FCN-z / GRU-peaknorm, 5 seeds (ensembles)"
  "E4a|p2_norm|--input goes_b|input = GOES XRS-B instead of the calibrated input"
  "E4b|p2_norm|--input fism|input = FISM2 0.1-0.8 nm"
  "E5|p2_norm|--exclude-near-test-hours 24|leakage check: drop training flares within 24 h of a test flare"
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
log "finished. failed: ${failed[*]:-none}.  next: bash 6_pack_results.sh"
