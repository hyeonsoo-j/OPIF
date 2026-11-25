#!/usr/bin/env bash
# Step 4 - download every M/X flare 2003-2023 (~1600 events) and build the training set (a few hours).
# Resumable: if it stops, run it again - finished files are skipped.
#   bash 4_download_all.sh                       (inside tmux/screen, or:)
#   nohup bash 4_download_all.sh > run_logs/4_nohup.out 2>&1 &
# Options (env): FORCE=1 (continue although calibration is not PASS), LIMIT=N (first N events, test only),
#                DL_WORKERS=2 (parallel downloads per source; keep 1-3 to be polite to the servers)
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_repo
exec > >(tee -a "$LOGDIR/4_download_all.log") 2>&1
cd "$REPO"
CAL="data/download/calibration.json"
if [ "$DRY_RUN" != "1" ]; then
  [ -f "$CAL" ] || die "run bash 3_calibrate.sh first"
  st=$("$PY" -c "import json; print(json.load(open('$CAL', encoding='utf-8'))['status'])")
  if [ "$st" != "PASS" ] && [ "${FORCE:-0}" != "1" ]; then
    die "calibration status is $st - send calibration_report.md first, or rerun with FORCE=1"
  fi
fi
# never overwrite an existing dataset: events_latis.npz, then events_latis_v2.npz, _v3 ...
OUT="data/processed/events_latis.npz"; v=2
while [ -e "$OUT" ]; do OUT="data/processed/events_latis_v${v}.npz"; v=$((v + 1)); done
ARGS=(--stage all --workers "${DL_WORKERS:-2}" --out "$OUT")
[ -n "${LIMIT:-}" ] && ARGS+=(--limit "$LIMIT")
run "$PY" scripts/download/run_all.py "${ARGS[@]}" || true
[ "$DRY_RUN" = "1" ] && exit 0
if [ ! -f "$OUT" ]; then
  log "dataset not written. If downloads failed, just run this script again (it resumes)."
  log "errors: $(cat data/download/logs/*_errors.csv 2>/dev/null | wc -l) lines in data/download/logs/*_errors.csv"
  exit 1
fi
"$PY" - "$OUT" <<'EOF'
import sys, numpy as np
d = np.load(sys.argv[1], allow_pickle=False)
t, s = d["is_test"], d["selected"]
print(f"  {sys.argv[1]}: {len(t)} events = {t.sum()} test + {(s & ~t).sum()} train/val candidates "
      f"(+ {(~s & ~t).sum()} failing the isolation rule, not used)")
u, c = np.unique(d["goes_sat"], return_counts=True)
print("  GOES satellite per event: " + ", ".join(f"G{int(a)}: {int(b)}" for a, b in zip(u, c)) + "  (G-1 = none)")
EOF
log "dataset ready: $REPO/$OUT   next: bash 5_run_experiments.sh"
