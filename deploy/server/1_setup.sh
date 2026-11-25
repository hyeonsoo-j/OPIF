#!/usr/bin/env bash
# Step 1 - unpack the code, create a Python venv, install packages, run the offline tests (~5-15 min)
#   bash 1_setup.sh
# Options (env): BASE_PY=python3.11   python used for the venv (needs >= 3.10)
#                USE_CURRENT_ENV=1     install into the active env (conda etc.) instead of .venv
#                TORCH=cu124 (default) | cu121 | cu118 | cpu   which torch build to install
#                  (driver 550 -> cu124; check `nvidia-smi` "CUDA Version"; cpu = no GPU use)
#                SKIP_INSTALL=1        only unpack + test
# Never installs into a conda env unless USE_CURRENT_ENV=1 (that would change the env's numpy/pandas/torch).
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec > >(tee -a "$LOGDIR/1_setup.log") 2>&1

[ -n "$REPO_ZIP" ] && [ -f "$REPO_ZIP" ] || die "FISM2_flare_repro_repo_v*.zip not found next to this script"
CHECK_PY="$BASE_PY"; [ "${USE_CURRENT_ENV:-0}" = "1" ] && CHECK_PY="$PY"
# rerun after the venv exists: check/use the venv python unless BASE_PY was given explicitly
if [ "${USE_CURRENT_ENV:-0}" != "1" ] && [ -x "$VENV/bin/python" ] && [ "$BASE_PY" = "python3" ]; then CHECK_PY="$VENV/bin/python"; fi
"$CHECK_PY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version; print("python", sys.version.split()[0])' \
  || die "Python >= 3.10 needed (set BASE_PY=python3.11, or use conda: conda create -n fism2 python=3.11; conda activate fism2; USE_CURRENT_ENV=1 bash 1_setup.sh)"

if [ -d "$REPO/scripts/download" ]; then
  log "code already unpacked in $REPO (delete that folder to unpack again)"
else
  log "unpacking $(basename "$REPO_ZIP") -> $REPO"
  "$CHECK_PY" -m zipfile -e "$REPO_ZIP" "$REPO"
fi

if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  if [ "${USE_CURRENT_ENV:-0}" != "1" ]; then
    if [ ! -x "$VENV/bin/python" ]; then
      log "creating venv $VENV"
      "$BASE_PY" -m venv "$VENV" || die "venv failed (Debian/Ubuntu: sudo apt install python3-venv, or use conda + USE_CURRENT_ENV=1)"
    fi
    PY="$VENV/bin/python"
  fi
  run "$PY" -m pip install --upgrade pip
  TORCH="${TORCH:-cu124}"
  if [ -n "${TORCH_INDEX:-}" ]; then run "$PY" -m pip install torch --index-url "$TORCH_INDEX"
  else run "$PY" -m pip install torch --index-url "https://download.pytorch.org/whl/$TORCH"; fi
  run "$PY" -m pip install -r "$REPO/requirements.txt" -r "$REPO/scripts/download/requirements-download.txt" pytest nvidia-ml-py
  "$PY" -c 'import torch; print("torch", torch.__version__, "cuda available:", torch.cuda.is_available(), "devices:", torch.cuda.device_count() if torch.cuda.is_available() else 0)'
  [ "$TORCH" = "cpu" ] || "$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' || log "WARNING: CUDA not available - training will fall back to CPU (set TORCH=cpu to silence, or fix the driver)"
fi

log "offline tests (mock servers; expect '56 passed')"
cd "$REPO"
run "$PY" -m pytest -q tests -p no:cacheprovider
log "setup done. next: bash 2_smoke_test.sh"
