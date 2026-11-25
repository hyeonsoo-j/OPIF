"""Phase 2 items 6 (physics constraints) and 9 (error vs lead time) for CV96 out-of-fold predictions.

Physics checks (per model, averaged over seeds; truth reported for reference):
  frac_negative     share of predicted samples < 0 (log10 models cannot produce them)
  frac_rerise10     events whose mean slope over the first 10 min after the peak is > 0
  frac_above_peak   events whose prediction exceeds its own t=0 value later by > 1 %
  frac_order_w0_w3  events with fractional 3-h decay log10(y0/y179) of w0 > w3
  frac_order_full   events with decay order w0 > w2 > w1 > w3 (typical ordering of the truth)
Lead time (t = 0..179 min after the peak):
  RMSE_t (linear), RMSE_t (log10), cross-event CC_t (log10 and linear) and the effective forecast
  time = first lead with CC_t(log10) < 0.5.
Outputs: results/verify/physics_checks.csv, results/verify/leadtime_metrics.csv,
         results/verify/effective_forecast_time.csv, figures/leadtime_rmse.png, figures/leadtime_cc.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.cv96 import BANDS, load_test96  # noqa: E402
from src.experiments.cv import available, load_oof  # noqa: E402
from src.viz.plots import lead_time_curves  # noqa: E402

EXP = sys.argv[1] if len(sys.argv) > 1 else "repro_cv96"
X, Y, ev = load_test96()
av = available(EXP)


def physics(P):
    L = np.log10(np.maximum(P, 1e-30))
    out = {"frac_negative": float((P < 0).mean())}
    for b, bn in enumerate(BANDS):
        out[f"frac_rerise10_{bn}"] = float(((L[:, 10, b] - L[:, 0, b]) > 0).mean())
        out[f"frac_above_peak_{bn}"] = float((P[:, 1:, b].max(1) > 1.01 * P[:, 0, b]).mean())
    D = L[:, 0, :] - L[:, -1, :]
    out["frac_order_w0_w3"] = float((D[:, 0] > D[:, 3]).mean())
    out["frac_order_full"] = float(((D[:, 0] > D[:, 2]) & (D[:, 2] > D[:, 1]) & (D[:, 1] > D[:, 3])).mean())
    return out


def lead(P):
    Lp, Ly = np.log10(P), np.log10(Y)
    rmse = np.sqrt(np.mean((P - Y) ** 2, axis=0))
    rmse_log = np.sqrt(np.mean((Lp - Ly) ** 2, axis=0))
    cc_log = np.zeros((180, 4))
    cc_lin = np.zeros((180, 4))
    for t in range(180):
        for b in range(4):
            cc_log[t, b] = np.corrcoef(Lp[:, t, b], Ly[:, t, b])[0, 1] if Lp[:, t, b].std() > 0 else np.nan
            cc_lin[t, b] = np.corrcoef(P[:, t, b], Y[:, t, b])[0, 1] if P[:, t, b].std() > 0 else np.nan
    return rmse, rmse_log, cc_log, cc_lin


phys_rows = [dict(model="FISM2 truth", **physics(Y))]
lead_rows, eff_rows = [], []
curves_rmse, curves_cc = {}, {}
for name in av:
    ph, rm, rl, cl, cn = [], [], [], [], []
    for s in av[name]:
        P = load_oof(EXP, name, s)
        ph.append(physics(P))
        r = lead(P)
        rm.append(r[0]); rl.append(r[1]); cl.append(r[2]); cn.append(r[3])
    phys_rows.append(dict(model=name, n_seeds=len(av[name]), **pd.DataFrame(ph).mean().to_dict()))
    rm, rl, cl, cn = [np.nanmean(a, axis=0) for a in (rm, rl, cl, cn)]
    curves_rmse[name], curves_cc[name] = rl, cl
    for b, bn in enumerate(BANDS):
        for t in range(180):
            lead_rows.append(dict(model=name, band=bn, lead_min=t, rmse=rm[t, b], rmse_log=rl[t, b],
                                  cc_log=cl[t, b], cc_lin=cn[t, b]))
        below = np.where(cl[1:, b] < 0.5)[0]
        eff_rows.append(dict(model=name, band=bn,
                             effective_forecast_min=int(below[0] + 1) if len(below) else ">=179",
                             cc_log_t1=cl[1, b], cc_log_t60=cl[60, b], cc_log_t179=cl[179, b],
                             rmse_log_t1=rl[1, b], rmse_log_t60=rl[60, b], rmse_log_t179=rl[179, b]))
pd.DataFrame(phys_rows).to_csv(ROOT / f"results/verify/physics_checks{'' if EXP == 'repro_cv96' else '_' + EXP}.csv", index=False)
pd.DataFrame(lead_rows).to_csv(ROOT / f"results/verify/leadtime_metrics{'' if EXP == 'repro_cv96' else '_' + EXP}.csv", index=False)
E = pd.DataFrame(eff_rows)
E.to_csv(ROOT / f"results/verify/effective_forecast_time{'' if EXP == 'repro_cv96' else '_' + EXP}.csv", index=False)
if EXP == "repro_cv96":
    show = ["Power-law", "Climatology", "Persistence-peak", "RandomForest", "FCN", "GRU", "Transformer"]
    lead_time_curves({k: curves_rmse[k] for k in show if k in curves_rmse}, ROOT / "figures/leadtime_rmse.png",
                     ylabel="RMSE of log10 irradiance (dex)", title="Error vs lead time, CV96 OOF [ours]")
    lead_time_curves({k: curves_cc[k] for k in show if k in curves_cc}, ROOT / "figures/leadtime_cc.png",
                     ylabel="cross-event CC (log10)", title="Cross-event correlation vs lead time [ours]", log=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(phys_rows).round(3).to_string(index=False))
print(E.pivot(index="model", columns="band", values="effective_forecast_min").to_string())
