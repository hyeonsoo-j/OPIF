#!/usr/bin/env bash
# Master One-Click Runner for server_bundle_v6 (Steps 0 -> 11).
# Usage (inside tmux):
#   tmux new -s fism2
#   bash run_everything.sh
#
# Optional environment variables:
#   REUSE_FROM=/path/to/an/earlier/bundle
#     -> Automatically copies .venv and downloaded data (data/download + events_latis*.npz) from an earlier bundle
#        so you don't wait 3-4 hours for downloads, while still running a 100% clean setup & all tests!
#   CLEAN_RESULTS=1
#     -> If reusing an old folder, deletes old results/*_full* so every training test runs fresh from scratch.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
source "$D/env.sh"

if [ -n "${REUSE_FROM:-}" ] && [ -d "$REUSE_FROM" ]; then
  log "=== Reusing .venv and downloaded data from $REUSE_FROM ==="
  if [ -d "$REUSE_FROM/.venv" ] && [ ! -d "$D/.venv" ]; then
    log "Copying virtual environment $REUSE_FROM/.venv -> $D/.venv ..."
    cp -a "$REUSE_FROM/.venv" "$D/.venv"
  fi
  mkdir -p "$REPO/data/processed"
  if [ -d "$REUSE_FROM/fism2_repro/data/download" ] && [ ! -d "$REPO/data/download" ]; then
    log "Copying downloaded raw archives $REUSE_FROM/fism2_repro/data/download -> $REPO/data/download ..."
    cp -a "$REUSE_FROM/fism2_repro/data/download" "$REPO/data/download"
  fi
  for npz in "$REUSE_FROM"/fism2_repro/data/processed/events_*.npz; do
    if [ -f "$npz" ] && [ ! -f "$REPO/data/processed/$(basename "$npz")" ]; then
      log "Copying built dataset $(basename "$npz") ..."
      cp -a "$npz" "$REPO/data/processed/"
    fi
  done
  if [ "${CLEAN_RESULTS:-0}" != "1" ] && [ -d "$REUSE_FROM/fism2_repro/results" ]; then
    log "Copying existing results (set CLEAN_RESULTS=1 to train everything from scratch instead) ..."
    mkdir -p "$REPO/results"
    cp -rn "$REUSE_FROM"/fism2_repro/results/*_full* "$REPO/results/" 2>/dev/null || true
  fi
fi

# Always unpack the latest v11 code cleanly over $REPO
log "=== Unpacking latest code ($(basename "$REPO_ZIP")) into $REPO ==="
mkdir -p "$REPO"
python3 -m zipfile -e "$REPO_ZIP" "$REPO"

if [ "${CLEAN_RESULTS:-0}" = "1" ]; then
  log "=== CLEAN_RESULTS=1: Removing previous results/*_full* to re-run all training from scratch ==="
  rm -rf "$REPO"/results/*_full* "$REPO"/results/full_server
fi

bash "$D/0_check_network.sh"
if [ -x "$D/.venv/bin/python" ]; then
  SKIP_INSTALL=1 bash "$D/1_setup.sh"
else
  bash "$D/1_setup.sh"
fi

if [ -n "$(latest_events)" ] && [ -f "$REPO/data/download/calibration.json" ]; then
  log "=== Existing calibrated dataset found ($(latest_events)) - skipping steps 2-4 download ==="
else
  bash "$D/2_smoke_test.sh"
  bash "$D/3_calibrate.sh" || { rc=$?; [ "${FORCE:-0}" = "1" ] || { echo "stopped after calibration (exit $rc) - see run_logs/3_calibrate.log"; exit $rc; }; }
  bash "$D/4_download_all.sh"
fi

bash "$D/5_run_experiments.sh"
bash "$D/7_run_phase3_full.sh"
bash "$D/8_run_gpu_followups.sh"
bash "$D/9_run_remaining.sh"
bash "$D/10_run_paper1.sh"
bash "$D/11_verify_all_results.sh"
log "=== ALL STEPS (0-11) COMPLETED SUCCESSFULLY! ==="
