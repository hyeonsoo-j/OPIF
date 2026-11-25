"""Evaluation-code validation on the authors' own predictions (Phase 1, step 'evaluation code
check'): load 3_evaluation/all_model_predictions_same_ytrue.npz (full precision, from the authors'
Zenodo package), score all 8 models with src/eval/metrics.py and compare with the authors'
event_metrics_same_predictions.csv and summary_metrics_same_predictions.csv. Also checks the earlier
text transcription (zenodo_test_xy.npz) against the full-precision arrays.
Outputs: results/verify/metric_validation_authors.csv, results/verify/transcription_check.json,
         data/processed/zenodo_full.npz (X_real, Y_true, event_names, Y_<model>; physical units)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.metrics import per_event_metrics, pooled_metrics  # noqa: E402

RAW = ROOT / "external/zenodo/raw/2_evaluation/3_evaluation"
z = np.load(RAW / "all_model_predictions_same_ytrue.npz", allow_pickle=False)
# `event_names` is a pickled object array: not unpickled (untrusted). The event order is taken from the
# checksummed transcription and confirmed numerically below (X_real/Y_true agree to the 1e-4-dex rounding).
t = np.load(ROOT / "data/processed/zenodo_test_xy.npz")
names = [str(e) for e in t["event_names"]]
X, Y = z["X_real"].astype(np.float64), z["Y_true"].astype(np.float64)
models = [k[2:] for k in z.files if k.startswith("Y_") and k != "Y_true"]
np.savez(ROOT / "data/processed/zenodo_full.npz", X_real=X, Y_true=Y, event_names=np.array(names),
         **{f"Y_{m}": z[f"Y_{m}"].astype(np.float64) for m in models})

# transcription check
tc = dict(event_order_source="transcription (verified numerically)",
          max_abs_dex_X=float(np.max(np.abs(np.log10(t["X_real"]) - np.log10(X)))),
          max_abs_dex_Y=float(np.max(np.abs(np.log10(t["Y_true"]) - np.log10(Y)))),
          shapes=[list(X.shape), list(Y.shape)], models=models)
json.dump(tc, open(ROOT / "results/verify/transcription_check.json", "w"), indent=1)
print(tc)

E = pd.read_csv(RAW / "event_metrics_same_predictions.csv")
S = pd.read_csv(RAW / "summary_metrics_same_predictions.csv")
print("author csv columns:", list(E.columns), list(S.columns))
BANDS = ["w0", "w1", "w2", "w3"]
rows = []
for m in models:
    P = z[f"Y_{m}"].astype(np.float64)
    pe = per_event_metrics(Y, P)
    pm = pooled_metrics(Y, P)
    for b, bn in enumerate(BANDS):
        s = S[(S.model == m) & (S.band == bn)]
        for met in ["CC", "RMSE", "MRE", "MAPE", "APE", "Median_event_RMSE", "Mean_event_RMSE"]:
            ours = float(pm[met][b])
            ref = float(s[met].iloc[0]) if len(s) and met in s else np.nan
            rows.append(dict(model=m, band=bn, level="summary", metric=met, ours=ours, author=ref,
                             abs_diff=abs(ours - ref), rel_diff=abs(ours - ref) / max(abs(ref), 1e-300)))
        e = E[(E.model == m) & (E.band == bn)].set_index("event").loc[names]
        for met in ["CC", "RMSE", "MRE", "MAPE", "APE"]:
            col = met if met in e else None
            if col is None:
                continue
            d = np.abs(pe[met][:, b] - e[col].values)
            rows.append(dict(model=m, band=bn, level="event(max over 96)", metric=met, ours=np.nan, author=np.nan,
                             abs_diff=float(np.nanmax(d)),
                             rel_diff=float(np.nanmax(d / np.maximum(np.abs(e[col].values), 1e-300)))))
R = pd.DataFrame(rows)
R.to_csv(ROOT / "results/verify/metric_validation_authors.csv", index=False)
print(R.groupby(["level", "metric"]).rel_diff.max().to_string())
