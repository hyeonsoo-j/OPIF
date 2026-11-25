"""Phase 2 item 2 — power-law decay baseline: closed form vs nonlinear fits on the 96 real events.

Model A (paper):  y(t) = y_peak (t+1)^-alpha,  t = 0..179 min after the peak
  closed form (log-log LS):   alpha = -sum(u v) / sum(u^2),  u = ln(t+1), v = ln(y/y_peak)
  scipy least_squares on the same log residual (must agree to < 1 %)
  scipy curve_fit on the linear residual of y/y_peak (different objective)
  authors' variant: curve_fit of (t+1)^-alpha to y/y[0] with t = 1..180 (Zenodo script)
Model B (alternative form): y(t) = y_peak (1 + t/tau)^-alpha  (2 parameters, log-space LS)

Outputs: results/verify/powerlaw_alpha_events.csv, results/verify/powerlaw_alpha_summary.csv,
         figures/powerlaw_alpha_hist.png
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import curve_fit, least_squares  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.cv96 import BAND_LABELS, BANDS, load_test96  # noqa: E402

warnings.filterwarnings("ignore")
X, Y, ev = load_test96()
t0 = np.arange(0, 180, dtype=float)
t1 = np.arange(1, 181, dtype=float)
u = np.log(t0 + 1)
rows = []
for i, e in enumerate(ev):
    for b, bn in enumerate(BANDS):
        y = Y[i, :, b]
        v = np.log(y / y[0])
        a_closed = -np.sum(u * v) / np.sum(u * u)
        a_ls = least_squares(lambda a: v + a[0] * u, x0=[0.5]).x[0]
        a_cf = curve_fit(lambda t, a: (t + 1) ** -a, t0, y / y[0], p0=[0.5], bounds=(-5, 5))[0][0]
        a_auth = curve_fit(lambda t, a: (t + 1) ** -a, t1, np.clip(y / y[0], 1e-10, None), p0=[0.5],
                           bounds=(0.01, 5.0), maxfev=1000)[0][0]

        def resB(p):
            al, lt = p
            return v + al * np.log1p(t0 / np.exp(lt))
        sol = least_squares(resB, x0=[0.5, np.log(10.0)], bounds=([-5, np.log(0.01)], [10, np.log(1e5)]))
        aB, tauB = sol.x[0], float(np.exp(sol.x[1]))
        rmsA = np.sqrt(np.mean((v + a_closed * u) ** 2)) / np.log(10)
        rmsB = np.sqrt(np.mean(sol.fun ** 2)) / np.log(10)
        rows.append(dict(event=e, band=bn, alpha_closed=a_closed, alpha_lsq=a_ls, alpha_curvefit=a_cf,
                         alpha_authors=a_auth, alpha_B=aB, tau_B=tauB, rms_dex_A=rmsA, rms_dex_B=rmsB,
                         closed_vs_lsq_rel=abs(a_closed - a_ls) / max(abs(a_ls), 1e-12)))
D = pd.DataFrame(rows)
D.to_csv(ROOT / "results/verify/powerlaw_alpha_events.csv", index=False)

paper_med = {"w0": 0.400, "w1": 0.022, "w2": 0.073, "w3": 0.010}  # [paper] Sect. 3.3 (training events)
S = []
for bn, g in D.groupby("band"):
    rec = dict(band=bn, paper_median_alpha=paper_med[bn])
    for c in ["alpha_authors", "alpha_closed", "alpha_curvefit", "alpha_B"]:
        rec[c + "_median"] = g[c].median()
        rec[c + "_q25"] = g[c].quantile(0.25)
        rec[c + "_q75"] = g[c].quantile(0.75)
    rec["frac_alpha_closed_in_0.5_3"] = ((g.alpha_closed >= 0.5) & (g.alpha_closed <= 3)).mean()
    rec["frac_alpha_B_in_0.5_3"] = ((g.alpha_B >= 0.5) & (g.alpha_B <= 3)).mean()
    rec["frac_alpha_closed_negative"] = (g.alpha_closed < 0).mean()
    rec["tau_B_median_min"] = g.tau_B.median()
    rec["max_closed_vs_lsq_rel"] = g.closed_vs_lsq_rel.max()
    rec["median_abs_closed_minus_curvefit"] = (g.alpha_closed - g.alpha_curvefit).abs().median()
    rec["rms_dex_A_median"] = g.rms_dex_A.median()
    rec["rms_dex_B_median"] = g.rms_dex_B.median()
    S.append(rec)
S = pd.DataFrame(S)
S.to_csv(ROOT / "results/verify/powerlaw_alpha_summary.csv", index=False)
pd.set_option("display.width", 250)
print(S.T.to_string())

fig, axs = plt.subplots(1, 4, figsize=(15, 3.4))
for b, bn in enumerate(BANDS):
    g = D[D.band == bn]
    ax = axs[b]
    lo, hi = np.quantile(np.r_[g.alpha_closed, g.alpha_authors], [0.01, 0.99])
    bins = np.linspace(lo, hi, 25)
    ax.hist(g.alpha_closed, bins=bins, alpha=0.6, label="closed form (t+1)$^{-α}$, t=0..179")
    ax.hist(g.alpha_authors, bins=bins, alpha=0.6, label="authors' curve_fit (t=1..180)")
    ax.axvline(paper_med[bn], color="k", ls="--", lw=1, label="paper median [paper]")
    ax.set_title(f"{BAND_LABELS[b]}")
    ax.set_xlabel("α")
axs[0].set_ylabel("events")
axs[0].legend(fontsize=7)
fig.suptitle("Power-law decay exponents of the 96 Zenodo test events [ours]")
fig.tight_layout()
fig.savefig(ROOT / "figures/powerlaw_alpha_hist.png", dpi=150)
print("saved figures/powerlaw_alpha_hist.png")
