#!/usr/bin/env bash
# Step 3 - download the 96 Zenodo test events and find the authors' construction rules (~10-30 min).
#   bash 3_calibrate.sh
# Result: fism2_repro/data/download/calibration_report.md  (status PASS -> go on with step 4)
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_repo
exec > >(tee -a "$LOGDIR/3_calibrate.log") 2>&1
cd "$REPO"
run "$PY" scripts/download/run_all.py --stage test --workers "${DL_WORKERS:-2}" || true
[ "$DRY_RUN" = "1" ] && exit 0
CAL="data/download/calibration.json"
[ -f "$CAL" ] || die "no calibration.json - see run_logs/3_calibrate.log and data/download/logs/"
status=$("$PY" - "$CAL" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
print(c["status"])
print(f"  events: {c['n_events']}/96, peak rule: {c['peak_rule']} (exact for {c['peak_rule_frac_exact']:.0%}), "
      f"input: {c['input_source']} (GOES flags: {c['goes_flags']})", file=sys.stderr)
for b, v in c["bands"].items():
    print(f"  {b} {v['edges_nm']}: bins '{v['rule']}', {v['agg']}", file=sys.stderr)
print("  median RMS (dex): " + ", ".join(f"{k} {v:.4f}" for k, v in c["median_rms_dex"].items()), file=sys.stderr)
EOF
)
log "calibration status: $status   (report: $REPO/data/download/calibration_report.md)"
if [ "$status" != "PASS" ]; then
  log "status is not PASS - send data/download/calibration_report.md before step 4 (or FORCE=1 bash 4_download_all.sh)"
  exit 2
fi
log "next: bash 4_download_all.sh   (long; run inside tmux/screen or with nohup)"
