#!/usr/bin/env bash
# Step 10 - paper-1 experiments (operational re-design), GPU (~30-60 min on 4 GPUs).
#   P1  GOES XRS-B as the direct input (FCN-z, GRU-z, 5 seeds)
#   P2  GOES -> FISM2 0.1-0.8 nm converter (P2a linear / P2b FCN) feeding the FISM2-trained models (5 seeds)
#   P3  forecasts issued 2 / 5 / 10 min after the peak (configs/paper1_delay.yaml, 3 seeds)
#   P4  online peak detector latency on the GOES stream (all events); P4b = v2 rule (rise by a factor 2 required);
#       P4c = detector that stays armed (every declaration; is the flare's own maximum declared, and when) + export
#       of the XRS-A/B streams of all events (results/full_server/peak_detector_streams.npz)
#   P5  conditional split conformal (none / peak / class) for the final candidates and the P1-P2 models
#   P6  tables: notebooks/analyze_paper1.py (+ D-RAP), then verify & pack
#   P7  forecasts issued 2 / 5 / 10 min after the peak with the inputs available in real time: every input window
#       covers minutes -59+k..k. Inputs: FISM2 (reference; seeds 0-2 repeat P3), GOES through the FCN converter
#       (operational), GOES XRS-B direct. GRU-z and FCN-z, 5 seeds -> 9 experiments x 10 trainings, GPU.
#       Needs the GOES minutes after the peak: built from the streams of P4c (run first if they are missing).
#   P8  forecasts for the flares outside the isolation rule (about 395 flares, never used for fitting): the
#       peak-issued GRU-z / FCN-z are retrained with the same seeds (FISM2 input and GOES through the FCN
#       converter) and also predict those flares (oofN_*.npy, extra_rows.csv) -> 2 experiments x 10 trainings, GPU.
#   P9  training WITH the flares outside the isolation rule (two folds by time cluster; the other fold is scored)
#   P10 directory listings of the operational GOES XRS archives (no data)
#   P11 operational (archived real-time) GOES inputs: the operational 1-min XRS files of every flare are downloaded
#       (scripts/download/fetch_goes_xrs_ops.py), the 60 input minutes are cut as in the dataset (scripts/build_goes_ops.py,
#       report results/full_server/goes_ops_vs_science*.csv), and GRU-z / FCN-z with the GOES + FCN converter input, trained
#       as in P8, predict the test flares and the flares outside the rule from the science-quality AND the operational
#       input (oofO_*.npy, ops_rows.csv) -> 10 trainings, GPU.
#   P12 latest time stamp of the public FISM2 flare product (a few small requests to LISIRD; no analysis)
#   bash 10_run_paper1.sh      (inside tmux)
# Options (env): ONLY="P1 P2b P4b P4c P5 P6 P7 P8 P9 P10 P11 P12"  (with ONLY unset every step runs)  SEEDS5=0,1,2,3,4  EVENTS=path.npz
#                LIGHT=1  pack only the tables and logs (results_tables_<date>.tar.gz) instead of the full bundle;
#                         the predictions of P7 / P8 are added when those steps ran in this call
#                P7_DELAYS="2 5 10"  P7_INPUTS="fism goes_convnn goes_b"  P8_INPUTS="fism goes_convnn"
set -uo pipefail
source "$(dirname "$0")/env.sh"
need_repo
exec > >(tee -a "$LOGDIR/10_paper1.log") 2>&1
cd "$REPO"
EVENTS="${EVENTS:-$(latest_events)}"
[ -n "$EVENTS" ] && [ -f "$EVENTS" ] || die "no data/processed/events_latis*.npz"
SEEDS5="${SEEDS5:-0,1,2,3,4}"; export SEEDS5      # also read by 11_verify_all_results.sh
"$PY" -c 'import torch; print("torch", torch.__version__, "cuda:", torch.cuda.is_available(), "visible devices:", torch.cuda.device_count())'
log "dataset: $EVENTS   workers: $WORKERS   device: $FISM_DEVICE   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
want() { [ -z "${ONLY:-}" ] || [[ " $ONLY " == *" $1 "* ]]; }
failed=()
go() { local id="$1"; shift; log "=== $id: $*"; if ! run "$PY" "$@"; then failed+=("$id"); log "!!! $id failed"; fi; }
# training step: a failed call is repeated once with freshly started worker processes (FISM_MP_CONTEXT=spawn);
# run.py keeps the models that are already finished, so only the missing ones are trained again
go_train() {
  local id="$1"; shift; log "=== $id: $*"
  if run "$PY" "$@"; then return 0; fi
  log "!!! $id failed - once more with FISM_MP_CONTEXT=spawn (finished models are kept)"
  if env FISM_MP_CONTEXT=spawn "$PY" "$@"; then return 0; fi
  failed+=("$id"); log "!!! $id failed"
}
in_flag() { [ "$1" = "fism" ] || echo "--input $1"; }      # fism = the default input of the full protocol

want P1 && go P1 run.py --exp p2_zscore --protocol full --events "$EVENTS" --workers "$WORKERS" --seeds "$SEEDS5" --input goes_b --tag p1
if want P2 || want P2a; then
  go P2a run.py --exp p2_zscore --protocol full --events "$EVENTS" --workers "$WORKERS" --seeds "$SEEDS5" --input goes_conv --tag p2
fi
if want P2 || want P2b; then
  go P2b run.py --exp p2_zscore --protocol full --events "$EVENTS" --workers "$WORKERS" --seeds "$SEEDS5" --input goes_convnn --tag p2nn
fi
want P3 && go P3 run.py --exp paper1_delay --protocol full --events "$EVENTS" --workers "$WORKERS"
want P4 && go P4 notebooks/peak_detector.py --events "$EVENTS" --root data/download --out results/full_server
# P4b: v2 rule - declare only after the flux has risen by a factor 2 inside the stream (no window-edge declarations)
want P4b && go P4b notebooks/peak_detector.py --events "$EVENTS" --root data/download --out results/full_server --require-rise --tag _v2
# P4c: armed detector (re-declares a later, larger maximum; a flare on the tail of an earlier one is a new episode)
want P4c && go P4c notebooks/peak_detector.py --events "$EVENTS" --root data/download --out results/full_server --rearm --save-streams --tag _v3
if want P5; then
  M="p2_zscore_full_gpu5:FCN-z p2_zscore_full_gpu5:GRU-z p3_ensemble_full_gpu:GRU-peaknorm"
  M2="p2_zscore_full_in-goes_b_p1:FCN-z p2_zscore_full_in-goes_b_p1:GRU-z p2_zscore_full_in-goes_conv_p2:FCN-z p2_zscore_full_in-goes_conv_p2:GRU-z p2_zscore_full_in-goes_convnn_p2nn:FCN-z p2_zscore_full_in-goes_convnn_p2nn:GRU-z"
  for cond in none peak class; do
    SAVE=""; [ "$cond" = "none" ] && SAVE="--save-intervals"     # interval arrays (mu/lo/hi per event) for the figures, ~15 MB
    go "P5-$cond" notebooks/split_conformal_full.py --events "$EVENTS" --results results --out results/full_server \
       --models $M $M2 --avg "p2_zscore_full_gpu5:FCN-z+p2_zscore_full_gpu5:GRU-z" --cond "$cond" --tag "$cond" $SAVE
  done
fi
if want P6; then
  REB="data/processed/events_test96_rebuilt.npz"
  [ -f "$REB" ] || run "$PY" scripts/download/build_events.py --stage test --out "$REB"
  go P6 notebooks/analyze_paper1.py --results results --rebuilt "$REB" --out results/full_server
fi
COMMON=(--exp p2_zscore --protocol full --events "$EVENTS" --workers "$WORKERS" --seeds "$SEEDS5" --models GRU-z,FCN-z)
if want P7; then
  ST="results/full_server/peak_detector_streams.npz"; GP="data/processed/events_goes_post.npz"
  [ -f "$ST" ] || go P4c notebooks/peak_detector.py --events "$EVENTS" --root data/download --out results/full_server --rearm --save-streams --tag _v3
  go P7-post scripts/build_goes_post.py --events "$EVENTS" --streams "$ST" --out "$GP"
  for k in ${P7_DELAYS:-2 5 10}; do
    for inp in ${P7_INPUTS:-fism goes_convnn goes_b}; do
      go_train "P7-$inp-d$k" run.py "${COMMON[@]}" $(in_flag "$inp") --issue-delay "$k" --tag p7
    done
  done
  export PACK_EXTRA="${PACK_EXTRA:-} results/*_p7 $GP"
fi
if want P8; then
  for inp in ${P8_INPUTS:-fism goes_convnn}; do
    go_train "P8-$inp" run.py "${COMMON[@]}" $(in_flag "$inp") --predict-extra --tag p8
  done
  export PACK_EXTRA="${PACK_EXTRA:-} results/*_p8"
fi
if want P9; then      # training WITH the flares outside the isolation rule: two folds by time cluster, the other fold is scored
  for inp in ${P9_INPUTS:-fism goes_convnn}; do
    for f in 0 1; do
      go_train "P9-$inp-f$f" run.py "${COMMON[@]}" $(in_flag "$inp") --predict-extra --train-extra-fold "$f" --tag p9
    done
  done
  export PACK_EXTRA="${PACK_EXTRA:-} results/*_p9"
fi
if want P10; then     # what the operational (real-time) GOES XRS archives hold: directory listings only, no data download
  log "=== P10: probe of the operational GOES XRS archives"
  run "$PY" scripts/probe_goes_operational.py --out results/full_server/goes_operational_probe.csv || failed+=("P10")
fi
if want P11; then     # operational GOES inputs: same trained models, science-quality and operational input side by side
  ST="results/full_server/peak_detector_streams.npz"; GO="data/processed/events_goes_ops.npz"
  [ -f "$ST" ] || go P4c notebooks/peak_detector.py --events "$EVENTS" --root data/download --out results/full_server --rearm --save-streams --tag _v3
  go P11-download scripts/download/fetch_goes_xrs_ops.py --events "$EVENTS" --root data/download
  log "=== P11-build: operational input windows"
  if run "$PY" scripts/build_goes_ops.py --events "$EVENTS" --streams "$ST" --root data/download --out "$GO" --report results/full_server/goes_ops_vs_science.csv; then
    go_train "P11-goes_convnn" run.py "${COMMON[@]}" --input goes_convnn --predict-extra --predict-ops --goes-ops "$GO" --tag p11
    export PACK_EXTRA="${PACK_EXTRA:-} results/*_p11 $GO"
  else
    failed+=("P11-build"); log "!!! P11-build failed - the forecasts with operational inputs were not run"
    [ -f "$GO" ] && export PACK_EXTRA="${PACK_EXTRA:-} $GO"
  fi
fi
if want P12; then     # how recent the public FISM2 flare product is (LISIRD LaTiS; a few kB)
  log "=== P12: latest time stamp of the public FISM2 flare product"
  run "$PY" scripts/probe_fism2_latest.py --out results/full_server/fism2_latest_probe.csv || failed+=("P12")
fi
log "finished. failed: ${failed[*]:-none}"
bash "$BUNDLE_DIR/11_verify_all_results.sh" || true
bash "$BUNDLE_DIR/6_pack_results.sh"
