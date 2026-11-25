"""Phase 2/4 analyses on the AUTHORS' prediction arrays (Zenodo all_model_predictions_same_ytrue.npz,
full precision; from the authors' Zenodo package, external/zenodo/raw/).

  (a) paired event bootstrap (10 000) of the *pooled* CC difference GRU - other, per band and band mean
      -> is 'GRU has the highest mean CC' significant on the paper's own metric?
  (b) physics checks (re-rise, above-peak, decay ordering) and lead-time structure (log10 RMSE and
      cross-event log10 CC at 1/60/179 min) for the 8 authors' models
  (c) D-RAP 10 MHz absorption over Korea (37N 127E) from the authors' w0 predictions, daytime events
Outputs: results/verify/authors_pooledcc_bootstrap.csv, results/verify/authors_physics.csv,
         results/verify/authors_leadtime.csv, results/final/drap_summary_authors.csv,
         figures/authors_leadtime_rmse.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.cv96 import BANDS, load_meta  # noqa: E402
from src.eval.drap import event_absorption  # noqa: E402
from src.viz.plots import lead_time_curves  # noqa: E402

z = np.load(ROOT / "data/processed/zenodo_full.npz")
Y = z["Y_true"]
MODELS = ["Power-law", "LinearReg", "RandomForest", "FCN", "RNN", "LSTM", "GRU", "Transformer"]
P = {m: z[f"Y_{m}"] for m in MODELS}
N = len(Y)
rng = np.random.default_rng(0)
B = 10000
idx = rng.integers(0, N, size=(B, N))


def pooled_cc(yt, yp):
    a, b = yt.reshape(len(yt), -1), yp.reshape(len(yp), -1)
    return np.corrcoef(a.ravel(), b.ravel())[0, 1]


# ---------------- (a) bootstrap of pooled CC
rows = []
for m in MODELS:
    if m == "GRU":
        continue
    diffs = np.zeros((B, 5))
    for k in range(B):
        ii = idx[k]
        d = [pooled_cc(Y[ii, :, b], P["GRU"][ii, :, b]) - pooled_cc(Y[ii, :, b], P[m][ii, :, b]) for b in range(4)]
        diffs[k, :4] = d
        diffs[k, 4] = np.mean(d)
    for j, bn in enumerate(BANDS + ["mean4"]):
        obs = (np.mean([pooled_cc(Y[:, :, b], P["GRU"][:, :, b]) - pooled_cc(Y[:, :, b], P[m][:, :, b]) for b in range(4)])
               if bn == "mean4" else pooled_cc(Y[:, :, j], P["GRU"][:, :, j]) - pooled_cc(Y[:, :, j], P[m][:, :, j]))
        lo, hi = np.quantile(diffs[:, j], [0.025, 0.975])
        p = 2 * min((diffs[:, j] <= 0).mean(), (diffs[:, j] >= 0).mean())
        rows.append(dict(competitor=m, band=bn, dCC_GRU_minus_comp=obs, ci_lo=lo, ci_hi=hi, p_boot=min(1.0, p),
                         frac_boot_GRU_better=(diffs[:, j] > 0).mean()))
A = pd.DataFrame(rows)
A.to_csv(ROOT / "results/verify/authors_pooledcc_bootstrap.csv", index=False)
pd.set_option("display.width", 200)
print(A[A.band == "mean4"].round(4).to_string(index=False))

# ---------------- (b) physics + lead time
phys, lead, curves = [], [], {}
for m in ["FISM2 truth"] + MODELS:
    Pm = Y if m == "FISM2 truth" else P[m]
    L, Ly = np.log10(np.maximum(Pm, 1e-30)), np.log10(Y)
    rec = dict(model=m, frac_negative=float((Pm < 0).mean()))
    for b, bn in enumerate(BANDS):
        rec[f"frac_rerise10_{bn}"] = float(((L[:, 10, b] - L[:, 0, b]) > 0).mean())
        rec[f"frac_above_peak_{bn}"] = float((Pm[:, 1:, b].max(1) > 1.01 * Pm[:, 0, b]).mean())
    D = L[:, 0, :] - L[:, -1, :]
    rec["frac_order_w0_w3"] = float((D[:, 0] > D[:, 3]).mean())
    rec["frac_order_full"] = float(((D[:, 0] > D[:, 2]) & (D[:, 2] > D[:, 1]) & (D[:, 1] > D[:, 3])).mean())
    rec["roughness_w0_dex"] = float(np.mean(np.abs(np.diff(L[:, :, 0], 2, axis=1))))
    phys.append(rec)
    if m == "FISM2 truth":
        continue
    rl = np.sqrt(np.mean((L - Ly) ** 2, axis=0))
    curves[m] = rl
    for b, bn in enumerate(BANDS):
        cc = [np.corrcoef(L[:, t, b], Ly[:, t, b])[0, 1] for t in (1, 60, 179)]
        lead.append(dict(model=m, band=bn, rmse_log_t1=rl[1, b], rmse_log_t60=rl[60, b], rmse_log_t179=rl[179, b],
                         cc_log_t1=cc[0], cc_log_t60=cc[1], cc_log_t179=cc[2]))
pd.DataFrame(phys).to_csv(ROOT / "results/verify/authors_physics.csv", index=False)
pd.DataFrame(lead).to_csv(ROOT / "results/verify/authors_leadtime.csv", index=False)
lead_time_curves(curves, ROOT / "figures/authors_leadtime_rmse.png", ylabel="RMSE of log10 irradiance (dex)",
                 title="Authors' models [zenodo predictions]: error vs lead time (computed by us)")
print(pd.DataFrame(phys)[["model", "frac_rerise10_w0", "frac_above_peak_w0", "frac_order_full", "roughness_w0_dex"]]
      .round(3).to_string(index=False))
print(pd.DataFrame(lead)[lambda d: d.band.isin(["w0", "w3"])].round(3).to_string(index=False))

# ---------------- (c) D-RAP
meta = load_meta()
rows = []
for m in MODELS:
    for e in range(N):
        a_t, zen = event_absorption(Y[e, :, 0], meta.peak_time.iloc[e])
        if zen[0] >= 90:
            continue
        a_p, _ = event_absorption(P[m][e, :, 0], meta.peak_time.iloc[e])
        d = zen < 90
        err = a_p[d] - a_t[d]
        rows.append(dict(model=m, event=meta.event.iloc[e], MAE_dB=np.abs(err).mean(), RMSE_dB=np.sqrt((err ** 2).mean()),
                         bias_dB=err.mean(), peak_err_dB=a_p[0] - a_t[0]))
Dd = pd.DataFrame(rows)
S = Dd.groupby("model").agg(n_events=("event", "count"), MAE_dB=("MAE_dB", "mean"), RMSE_dB=("RMSE_dB", "mean"),
                            bias_dB=("bias_dB", "mean"),
                            abs_peak_err_dB=("peak_err_dB", lambda s: s.abs().mean())).reset_index()
S.to_csv(ROOT / "results/final/drap_summary_authors.csv", index=False)
Dd.to_csv(ROOT / "results/final/drap_events_authors.csv", index=False)
print(S.round(3).to_string(index=False))
