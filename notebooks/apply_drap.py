"""Phase 4 — D-RAP application: HF absorption at 10 MHz over Korea (37 N, 127 E) from predicted
vs FISM2 0.1-0.8 nm flux, 3 h after the peak.

Daytime events = solar zenith angle < 90 deg at the peak minute; only daytime minutes enter the
error statistics (night-time absorption is 0 by construction).
Usage : python notebooks/apply_drap.py [exp:model ...]   (default: a fixed list, see below)
Output: results/final/drap_events.csv, results/final/drap_summary.csv, figures/drap_example.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.cv96 import load_meta, load_test96  # noqa: E402
from src.eval.drap import event_absorption  # noqa: E402
from src.experiments.cv import available, load_oof  # noqa: E402

X, Y, ev = load_test96()
meta = load_meta()
pairs = sys.argv[1:] or ["repro_cv96:Power-law", "repro_cv96:Persistence-peak", "repro_cv96:Climatology",
                         "repro_cv96:RandomForest", "repro_cv96:FCN", "repro_cv96:GRU", "repro_cv96:Transformer"]


def mean_pred(exp, name):
    seeds = available(exp)[name]
    return 10 ** np.mean([np.log10(load_oof(exp, name, s)) for s in seeds], axis=0)


A_true, Z = [], []
for e in range(96):
    a, z = event_absorption(Y[e, :, 0], meta.peak_time.iloc[e])
    A_true.append(a)
    Z.append(z)
A_true, Z = np.array(A_true), np.array(Z)
day_ev = Z[:, 0] < 90
rows = []
preds = {}
for pr in pairs:
    exp, name = pr.split(":")
    P = mean_pred(exp, name)
    preds[pr] = P
    for e in np.where(day_ev)[0]:
        a_p, _ = event_absorption(P[e, :, 0], meta.peak_time.iloc[e])
        d = Z[e] < 90
        err = a_p[d] - A_true[e][d]
        rows.append(dict(model=pr, event=ev[e], cls=meta.cls.iloc[e], n_day_min=int(d.sum()),
                         A_true_peak=A_true[e][d].max(), A_pred_peak=a_p[d].max(),
                         MAE_dB=np.abs(err).mean(), RMSE_dB=np.sqrt((err ** 2).mean()), bias_dB=err.mean(),
                         peak_err_dB=a_p[0] - A_true[e][0] if d[0] else np.nan,
                         dur_true_gt1dB=int((A_true[e][d] > 1).sum()), dur_pred_gt1dB=int((a_p[d] > 1).sum())))
D = pd.DataFrame(rows)
out = ROOT / "results/final"
out.mkdir(parents=True, exist_ok=True)
D.to_csv(out / "drap_events.csv", index=False)
S = D.groupby("model").agg(n_events=("event", "count"), MAE_dB=("MAE_dB", "mean"), RMSE_dB=("RMSE_dB", "mean"),
                           bias_dB=("bias_dB", "mean"), abs_peak_err_dB=("peak_err_dB", lambda s: s.abs().mean()),
                           MAE_dB_median=("MAE_dB", "median"),
                           dur_gt1dB_abs_err_min=("dur_true_gt1dB", lambda s: np.nan)).reset_index()
S["dur_gt1dB_abs_err_min"] = [np.mean(np.abs(D[D.model == m].dur_pred_gt1dB - D[D.model == m].dur_true_gt1dB))
                              for m in S.model]
S["A_true_peak_mean"] = D.groupby("model").A_true_peak.mean().values
S.to_csv(out / "drap_summary.csv", index=False)
print(f"daytime events: {day_ev.sum()} / 96")
print(S.round(3).to_string(index=False))

# example figure: the daytime event with the largest true absorption
e = int(np.argmax(np.where(day_ev, A_true.max(1), -1)))
fig, ax = plt.subplots(figsize=(7, 3.5))
t = np.arange(180)
ax.plot(t, A_true[e], "k", lw=2, label="from FISM2 0.1–0.8 nm")
for j, (pr, P) in enumerate(preds.items()):
    a_p, _ = event_absorption(P[e, :, 0], meta.peak_time.iloc[e])
    ax.plot(t, a_p, lw=1, label=pr.split(":")[1])
ax.set_xlabel("minutes after peak")
ax.set_ylabel("absorption at 10 MHz (dB)")
ax.set_title(f"D-RAP absorption over Korea (37°N 127°E), {ev[e]} [ours]")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(ROOT / "figures/drap_example.png", dpi=140)
