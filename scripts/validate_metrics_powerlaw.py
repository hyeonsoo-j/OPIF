"""Validate src/eval/metrics.py against the authors' per-event metrics [zenodo].

The Power-law baseline is fully determined by X_real (A = last input value) and one decay
exponent per band, so it can be regenerated from the transcribed test arrays. alpha_b is
recovered from the author metrics themselves (1-D search per band, minimising the mismatch
of per-event MAPE), then *all* per-event metrics (CC, RMSE, MRE, MAPE, APE; 96 events x 4
bands) and the pooled summary are compared with the author files.

Outputs: results/verify/metric_validation_powerlaw.csv, results/verify/metric_validation_summary.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.metrics import per_event_metrics, pooled_metrics  # noqa: E402

d = np.load(ROOT / "data/processed/zenodo_test_xy.npz")
X, Y, ev = d["X_real"], d["Y_true"], [str(e) for e in d["event_names"]]
auth = pd.read_csv(ROOT / "external/zenodo/event_metrics_same_predictions_compact.csv")
auth = auth[auth.model == "Power-law"]
bands = ["w0", "w1", "w2", "w3"]
t = np.arange(1, 181, dtype=float)


def pred(alpha_b):
    return X[:, -1][:, None] * (t + 1.0)[None, :] ** (-alpha_b)


alphas = []
for b, bn in enumerate(bands):
    a_ev = auth[auth.band == bn].set_index("event").loc[ev]
    tgt = a_ev["MAPE"].to_numpy()

    def loss(a):
        pe = per_event_metrics(Y[:, :, b], pred(a))
        return float(np.mean((pe["MAPE"][:, 0] - tgt) ** 2))

    grid = np.linspace(0.0, 1.0, 401)
    a0 = grid[int(np.argmin([loss(a) for a in grid]))]  # global grid first (loss is multi-modal)
    r = minimize_scalar(loss, bounds=(max(a0 - 0.005, 0.0), a0 + 0.005), method="bounded",
                        options={"xatol": 1e-7})
    alphas.append(r.x)
alphas = np.array(alphas)

P = np.stack([pred(a) for a in alphas], axis=-1)
pe = per_event_metrics(Y, P)
rows = []
for b, bn in enumerate(bands):
    a_ev = auth[auth.band == bn].set_index("event").loc[ev]
    for m in ["CC", "RMSE", "MRE", "MAPE", "APE"]:
        ours = pe[m][:, b]
        theirs = a_ev[m].to_numpy()
        diff = np.abs(ours - theirs)
        # author compact file is rounded (CC 4 dp, MRE/MAPE/APE 2 dp, RMSE 4 significant digits)
        tol = {"CC": 5e-5, "MRE": 5e-3, "MAPE": 5e-3, "APE": 5e-3}.get(m)
        if m == "RMSE":
            rel = diff / np.abs(theirs)
            ok = rel <= 5e-4
            rows.append(dict(band=bn, metric=m, max_abs_diff=diff.max(), max_rel_diff=rel.max(),
                             n_within_rounding=int(ok.sum()), n=len(ok)))
        else:
            ok = diff <= tol + 1e-9
            rows.append(dict(band=bn, metric=m, max_abs_diff=diff.max(), max_rel_diff=np.nan,
                             n_within_rounding=int(ok.sum()), n=len(ok)))
out = pd.DataFrame(rows)
(ROOT / "results/verify").mkdir(parents=True, exist_ok=True)
out.to_csv(ROOT / "results/verify/metric_validation_powerlaw.csv", index=False)

# pooled summary vs author summary_metrics (full precision, from zenodo_skill_tables.md)
summ = {}
for line in (ROOT / "external/zenodo/zenodo_skill_tables.md").read_text().splitlines():
    if line.startswith("Power-law,w"):
        p = line.split(",")
        summ[p[1]] = [float(v) for v in p[2:]]
pm = pooled_metrics(Y, P)
srows = []
cols = ["CC", "RMSE", "MRE", "MAPE", "APE", "Median_event_RMSE", "Mean_event_RMSE"]
for b, bn in enumerate(bands):
    for k, c in enumerate(cols):
        srows.append(dict(band=bn, metric=c, ours=pm[c][b], zenodo=summ[bn][k],
                          rel_diff=abs(pm[c][b] - summ[bn][k]) / abs(summ[bn][k])))
s = pd.DataFrame(srows)
s.to_csv(ROOT / "results/verify/metric_validation_summary.csv", index=False)
json.dump({"alpha_recovered": alphas.tolist()}, open(ROOT / "results/verify/powerlaw_alpha_recovered.json", "w"), indent=1)
print("alpha recovered:", np.round(alphas, 5))
print(out.to_string(index=False))
print(s.to_string(index=False))
