#!/usr/bin/env bash
# Step 6 - collect everything worth sending back into one archive next to this script (~1 min).
#   bash 6_pack_results.sh   ->  results_full_<date>.tar.gz  (+ results_summary.txt printed below)
#   LIGHT=1 bash 6_pack_results.sh  ->  results_tables_<date>.tar.gz  (results/full_server + logs; + $PACK_EXTRA paths)
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_repo
cd "$REPO"
STAMP=$(date +%Y%m%d_%H%M)
SUM="$LOGDIR/results_summary.txt"
"$PY" - > "$SUM" <<'EOF'
import glob, os
import pandas as pd
rows = []
for f in sorted(glob.glob("results/*_full*/summary*.csv")):
    test = "rebuilt96" if f.endswith("_latis.csv") else "zenodo96"
    S = pd.read_csv(f)
    g = S.groupby("model")[["CC_mean", "MAPE_mean", "APE_mean"]].mean()   # mean over the 4 bands
    for m, r in g.iterrows():
        rows.append(dict(experiment=os.path.basename(os.path.dirname(f)), test=test, model=m, CC4=r.CC_mean,
                         MAPE4=r.MAPE_mean, APE4=r.APE_mean))
if rows:
    print("test = zenodo96: authors' test arrays | rebuilt96: same events rebuilt like the training data")
    print(pd.DataFrame(rows).sort_values(["experiment", "model", "test"]).round(3).to_string(index=False))
else:
    print("no results/*_full*/summary.csv yet")
EOF
cat "$SUM"
items=()
# data/processed/events_*.npz (built dataset + rebuilt test rows, ~10 MB) make the bundle self-contained for the
# analysis: the validation targets are needed to recompute conformal intervals away from the server.
for p in results/*_full* results/full_server results/*.out data/download/calibration_report.md data/download/calibration.json \
         data/download/calibration_events.csv data/download/calibration_examples.png data/download/logs \
         data/processed/events_latis*.npz data/processed/events_test96_rebuilt.npz data/processed/events_goes_post.npz; do
  [ -e "$p" ] && items+=("$p")
done
OUT="$BUNDLE_DIR/results_full_${STAMP}.tar.gz"
if [ "${LIGHT:-0}" = "1" ]; then
  # tables and logs only (a few tens of MB): for runs that train nothing, the prediction arrays are unchanged.
  # PACK_EXTRA (set by 10_run_paper1.sh for P7 / P8) adds the result folders of the steps that did train.
  items=()
  for p in results/full_server results/*.out ${PACK_EXTRA:-}; do [ -e "$p" ] && items+=("$p"); done
  OUT="$BUNDLE_DIR/results_tables_${STAMP}.tar.gz"
fi
tar -czf "$OUT" "${items[@]}" -C "$BUNDLE_DIR" run_logs
log "packed: $OUT ($(du -h "$OUT" | cut -f1)) - copy it back to the Solar Flare AI folder"
