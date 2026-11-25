#!/usr/bin/env bash
# Step 8 - GPU follow-ups after steps 5/7 (uses healthy GPUs auto-selected by env.sh).
#   bash 8_run_gpu_followups.sh        (inside tmux)
# Options (env): EVENTS=path.npz  ONLY="G1 G3"  TRIALS=100 (Optuna)  SEEDS=0,1,2,3,4
#   G1  p3_ensemble rerun on GPU with validation predictions saved (-> split conformal; CPU/GPU agreement check)
#   G2  AR-level leakage: drop train/val flares from the same active region as a test flare, + size-matched random control
#   G3  GRU-z Optuna, TRIALS trials, 5 seeds (the 30-trial run is kept as GRU-z-optuna)
#   G4  final candidates on GPU with validation predictions: FCN-z / GRU-z 5 seeds (p2_zscore)
set -uo pipefail
source "$(dirname "$0")/env.sh"
need_repo
exec > >(tee -a "$LOGDIR/8_gpu.log") 2>&1
cd "$REPO"
EVENTS="${EVENTS:-$(latest_events)}"
[ -n "$EVENTS" ] && [ -f "$EVENTS" ] || [ "$DRY_RUN" = "1" ] || die "no data/processed/events_latis*.npz"
TRIALS="${TRIALS:-100}"; SEEDS5="${SEEDS:-0,1,2,3,4}"
"$PY" -c 'import torch; print("torch", torch.__version__, "cuda:", torch.cuda.is_available(), "visible devices:", torch.cuda.device_count())'
log "dataset: $EVENTS   workers: $WORKERS   device: $FISM_DEVICE   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
want() { [ -z "${ONLY:-}" ] || [[ " $ONLY " == *" $1 "* ]]; }
failed=()
go() { local id="$1"; shift; log "=== $id: $*"; if ! run "$PY" "$@"; then failed+=("$id"); log "!!! $id failed"; fi; }

want G1 && go G1 run.py --exp p3_ensemble --protocol full --events "$EVENTS" --workers "$WORKERS" --tag gpu
if want G2; then
  go G2a run.py --exp p2_norm --protocol full --events "$EVENTS" --workers "$WORKERS" --exclude-same-ar
  # size-matched random control: N = (base train+val) - (exAR train+val), counted directly from the event pool
  read -r base exar <<< "$(FISM_PROTOCOL=full FISM_EVENTS="$EVENTS" "$PY" - <<'PYEOF' | tail -n 1
import os
from src.data import pool
_, _, _, _, tr, va, *_ = pool.load_full(); base = len(tr) + len(va)
os.environ["FISM_EXCLUDE_SAME_AR"] = "1"; pool.load_full.cache_clear()
_, _, _, _, tr, va, *_ = pool.load_full(); print(base, len(tr) + len(va))
PYEOF
)"
  if [[ "${base:-}" =~ ^[0-9]+$ ]] && [[ "${exar:-}" =~ ^[0-9]+$ ]] && [ "$base" -gt "$exar" ]; then
    N=$((base - exar)); log "AR exclusion removed $N events -> random controls with N=$N"
    for s in 0 1 2; do go G2b$s run.py --exp p2_norm --protocol full --events "$EVENTS" --workers "$WORKERS" --exclude-random "$N" --exclude-seed "1$s"; done
  else
    log "!!! could not read the event counts for the random control (base=${base:-?} exar=${exar:-?})"
    failed+=("G2-control")
  fi
fi
want G3 && go G3 scripts/tune_optuna.py --protocol full --events "$EVENTS" --config p2_zscore --spec GRU-z --family gru     --trials "$TRIALS" --seeds ${SEEDS5//,/ } --workers 1 --tag "t$TRIALS"
want G4 && go G4 run.py --exp p2_zscore --protocol full --events "$EVENTS" --workers "$WORKERS" --seeds "$SEEDS5" --tag gpu5
log "finished. failed: ${failed[*]:-none}.  next: bash 9_run_remaining.sh"
