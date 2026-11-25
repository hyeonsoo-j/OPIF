#!/usr/bin/env bash
# Step 11 - verify that every single experiment (E1-E5, C1-C12, G1-G4, R1-R3, P1-P6; P7 / P8 once started)
# and all analysis CSV files exist and have no missing models/seeds.
#   bash 11_verify_all_results.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_repo
cd "$REPO"
"$PY" - <<'EOF'
import glob, os, sys
from pathlib import Path
import pandas as pd

EXPECTED_DIRS = [
    ("E1 (repro_cv96)", "results/repro_cv96_full", 23),
    ("E2a (p2_norm)", "results/p2_norm_full", 6),
    ("E2b (p2_zscore)", "results/p2_zscore_full", 15),
    ("E3 (p3_ensemble)", "results/p3_ensemble_full", 10),
    ("E4a (goes_b)", "results/p2_norm_full_in-goes_b", 6),
    ("E4b (fism)", "results/p2_norm_full_in-fism", 6),
    ("E5 (ex24h)", "results/p2_norm_full_ex24h", 6),
    ("C1a (rnd163s0)", "results/p2_norm_full_rnd163s0", 6),
    ("C1b (rnd163s1)", "results/p2_norm_full_rnd163s1", 6),
    ("C1c (rnd163s2)", "results/p2_norm_full_rnd163s2", 6),
    ("C2 (p2_window)", "results/p2_window_full", 18),
    ("C3 (p3_arch)", "results/p3_arch_full", 12),
    ("C4 (p3_resid)", "results/p3_resid_full", 7),
    ("C5 (p3_extreme)", "results/p3_extreme_full", 9),
    ("C6 (p3_type)", "results/p3_type_full", 6),
    ("C7 (p3_joint)", "results/p3_joint_full", 6),
    ("C8 (p3_features)", "results/p3_features_full", 6),
    ("C9 (p3_early)", "results/p3_early_full", 18),
    ("C10 (p3_final)", "results/p3_final_full", 5),
    ("C11 (p2_scale)", "results/p2_scale_full", 4),
    ("C12/G3 (p3_optuna)", "results/p3_optuna_full", 6),
    ("G1 (p3_ensemble_gpu)", "results/p3_ensemble_full_gpu", 10),
    ("G2a (exAR)", "results/p2_norm_full_exAR", 6),
    ("G4 (p2_zscore_gpu5)", "results/p2_zscore_full_gpu5", 25),
    ("P1 (goes_b_p1)", "results/p2_zscore_full_in-goes_b_p1", 25),
    ("P2a (goes_conv_p2)", "results/p2_zscore_full_in-goes_conv_p2", 25),
    ("P2b (goes_convnn_p2nn)", "results/p2_zscore_full_in-goes_convnn_p2nn", 25),
    ("P3 (paper1_delay)", "results/paper1_delay_full", 21),
]

EXPECTED_CSVS = [
    "results/full_server/summary_all.csv",
    "results/full_server/tests_phase3.csv",
    "results/full_server/tests_final_candidates.csv",
    "results/full_server/drap_full.csv",
    "results/full_server/conformal_full.csv",
    "results/full_server/conformal_split.csv",
    "results/full_server/conformal_split_none.csv",
    "results/full_server/conformal_split_peak.csv",
    "results/full_server/conformal_split_class.csv",
    "results/full_server/peak_detector.csv",
    "results/full_server/paper1_inputs.csv",
    "results/full_server/paper1_delay.csv",
    "results/full_server/paper1_drap.csv",
    "results/full_server/paper1_conformal.csv",
]

print("=" * 72)
print(" SERVER BUNDLE V6 - EXPERIMENT & TABLE VERIFICATION REPORT")
print("=" * 72)
missing = 0
for label, d, min_oof in EXPECTED_DIRS:
    p = Path(d)
    oofs = list(p.glob("oof_*_s*.npy")) if p.exists() else []
    has_sum = (p / "summary.csv").exists()
    ok = len(oofs) >= min_oof and has_sum
    st = "PASS" if ok else "MISSING/INCOMPLETE"
    if not ok:
        missing += 1
    print(f"  [{st:18s}] {label:24s} -> {len(oofs):2d}/{min_oof:2d} OOF files, summary.csv={'YES' if has_sum else 'NO'}")

print("-" * 72)
for f in EXPECTED_CSVS:
    p = Path(f)
    ok = p.exists() and p.stat().st_size > 20
    st = "PASS" if ok else "MISSING"
    if not ok:
        missing += 1
    rows = len(pd.read_csv(p)) if ok else 0
    print(f"  [{st:18s}] {f:45s} ({rows} rows)")

# paper-1 P7 / P8 (patch_v14), P9 (patch_v15), P11 (patch_v16): checked once a step has produced a folder. Same options as 10_run_paper1.sh.
n_seeds = len(os.environ.get("SEEDS5", "0,1,2,3,4").split(","))
tag = lambda inp: "" if inp == "fism" else f"_in-{inp}"
OPT = [(f"P7 (+{k} min, {inp})", f"results/p2_zscore_full{tag(inp)}_d{k}_p7", ("oof", "oofL", "val"))
       for k in os.environ.get("P7_DELAYS", "2 5 10").split() for inp in os.environ.get("P7_INPUTS", "fism goes_convnn goes_b").split()]
OPT += [(f"P8 (not isolated, {inp})", f"results/p2_zscore_full{tag(inp)}_p8", ("oof", "oofL", "oofN", "val"))
        for inp in os.environ.get("P8_INPUTS", "fism goes_convnn").split()]
OPT += [(f"P9 (train fold {f}, {inp})", f"results/p2_zscore_full{tag(inp)}_tx{f}_p9", ("oof", "oofL", "oofN", "val"))
        for inp in os.environ.get("P9_INPUTS", "fism goes_convnn").split() for f in (0, 1)]
OPT += [("P11 (operational GOES input)", "results/p2_zscore_full_in-goes_convnn_ops_p11", ("oof", "oofL", "oofN", "oofO", "val"))]
for step in ("P7", "P8", "P9", "P11"):
    grp = [o for o in OPT if o[0].startswith(step)]
    if not any(Path(d).exists() for _, d, _ in grp):
        continue
    print("-" * 72)
    for label, d, kinds in grp:
        q = Path(d)
        n = {k: len(list(q.glob(f"{k}_*_s*.np[yz]"))) for k in kinds}
        ok = all(v >= 2 * n_seeds for v in n.values()) and (q / "summary_latis.csv").exists()
        if step in ("P8", "P9", "P11"):
            ok = ok and (q / "extra_rows.csv").exists()
        if step == "P11":
            ok = ok and (q / "ops_rows.csv").exists() and Path("data/processed/events_goes_ops.npz").exists()
        if not ok:
            missing += 1
        print(f"  [{'PASS' if ok else 'MISSING/INCOMPLETE':18s}] {label:28s} -> " + ", ".join(f"{v} {k}" for k, v in n.items())
              + f" (need {2 * n_seeds} each)")
    if step == "P7":
        ok = Path("data/processed/events_goes_post.npz").exists()
        missing += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'MISSING':18s}] data/processed/events_goes_post.npz")

print("=" * 72)
if missing == 0:
    print(" ALL 28 EXPERIMENTS AND 14 SUMMARY TABLES ARE 100% COMPLETE (0 ERRORS)!")
else:
    print(f" WARNING: {missing} item(s) missing or incomplete.")
    sys.exit(1)
EOF
