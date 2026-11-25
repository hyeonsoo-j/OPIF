"""I-5 — uncertainty: 5-seed deep ensemble + cross-fold conformal prediction (90 % intervals).

For model M (default GRU) with OOF predictions of seeds 0..4 (results/p3_ensemble):
  mu, sd       ensemble mean / std of log10 predictions
  raw          mu +- 1.645 sd                                     (uncalibrated Gaussian)
  conf-abs     mu +- q_abs,  q_abs = 90 % quantile of |r|          (split conformal, non-adaptive)
  conf-norm    mu +- q_n (sd + s0), q_n = 90 % quantile of |r|/(sd+s0) (adaptive; s0 = median sd)
Calibration for the events of outer fold i uses only events from the other folds (cross-fold
conformal); quantiles are taken per band and per 30-min lead-time block, with the finite-sample
level ceil((n+1)0.9)/n. PIT uses a Gaussian with the conformally calibrated scale.
Also: the w0 interval is mapped to D-RAP absorption at 10 MHz (Korea) to give a dB interval.
Outputs: results/improve/uncertainty_<M>.csv, figures/coverage_<M>.png, figures/pit_<M>.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.cv96 import BAND_LABELS, BANDS, load_meta, load_test96, outer_folds  # noqa: E402
from src.eval.drap import event_absorption  # noqa: E402
from src.experiments.cv import available, load_oof  # noqa: E402

M = sys.argv[1] if len(sys.argv) > 1 else "GRU-peaknorm"
EXP = sys.argv[2] if len(sys.argv) > 2 else "p3_ensemble"
LEVEL = 0.9
X, Y, ev = load_test96()
meta = load_meta()
Ly = np.log10(Y)
seeds = available(EXP)[M]
L = np.stack([np.log10(load_oof(EXP, M, s)) for s in seeds])    # (S, 96, 180, 4)
mu, sd = L.mean(0), L.std(0, ddof=1)
# ensemble-mean point forecast as its own "model" for the ledger (I-5)
np.save(ROOT / f"results/{EXP}/oof_{M}-ens5_s0.npy", mu.astype(np.float32))
from src.experiments.cv import score_experiment  # noqa: E402
score_experiment(EXP)
s0 = np.median(sd, axis=(0, 1))[None, None, :]
fold = outer_folds()
blocks = [slice(i, i + 30) for i in range(0, 180, 30)]


def cq(r, level=LEVEL):
    n = r.size
    k = min(1.0, np.ceil((n + 1) * level) / n)
    return np.quantile(r, k)


lo = {k: np.zeros_like(mu) for k in ("raw", "conf-abs", "conf-norm")}
hi = {k: np.zeros_like(mu) for k in lo}
z_cal = np.zeros_like(mu)  # calibrated Gaussian scale for PIT
for i in np.unique(fold):
    te, ca = fold == i, fold != i
    for b in range(4):
        for blk in blocks:
            r = np.abs(Ly[ca, blk, b] - mu[ca, blk, b])
            rn = r / (sd[ca, blk, b] + s0[0, 0, b])
            qa, qn = cq(r), cq(rn)
            q68 = np.quantile(rn, 0.6827)
            m_, s_ = mu[te, blk, b], sd[te, blk, b] + s0[0, 0, b]
            lo["conf-abs"][te, blk, b], hi["conf-abs"][te, blk, b] = m_ - qa, m_ + qa
            lo["conf-norm"][te, blk, b], hi["conf-norm"][te, blk, b] = m_ - qn * s_, m_ + qn * s_
            z_cal[te, blk, b] = q68 * s_
lo["raw"], hi["raw"] = mu - 1.645 * sd, mu + 1.645 * sd

rows = []
for k in lo:
    inside = (Ly >= lo[k]) & (Ly <= hi[k])
    width = hi[k] - lo[k]
    for b, bn in enumerate(BANDS):
        rows.append(dict(model=M, method=k, band=bn, coverage=inside[:, :, b].mean(), mean_width_dex=width[:, :, b].mean(),
                         coverage_X=inside[meta.is_x.values][:, :, b].mean(),
                         coverage_first30=inside[:, :30, b].mean(), coverage_last30=inside[:, 150:, b].mean()))

# D-RAP absorption interval from the w0 interval (monotone transform), daytime minutes only
day_rows = []
for k in ["conf-norm", "conf-abs"]:
    cov, wid, n = [], [], 0
    for e in range(96):
        a_t, z = event_absorption(Y[e, :, 0], meta.peak_time.iloc[e])
        if z[0] >= 90:
            continue
        a_lo, _ = event_absorption(10 ** lo[k][e, :, 0], meta.peak_time.iloc[e])
        a_hi, _ = event_absorption(10 ** hi[k][e, :, 0], meta.peak_time.iloc[e])
        d = z < 90
        cov.append(((a_t[d] >= a_lo[d] - 1e-12) & (a_t[d] <= a_hi[d] + 1e-12)).mean())
        wid.append((a_hi[d] - a_lo[d]).mean())
        n += 1
    rows.append(dict(model=M, method=k, band="DRAP_dB_10MHz", coverage=float(np.mean(cov)),
                     mean_width_dex=np.nan, mean_width_dB=float(np.mean(wid)), n_day_events=n))
R = pd.DataFrame(rows)
out = ROOT / "results/improve"
out.mkdir(parents=True, exist_ok=True)
R.to_csv(out / f"uncertainty_{M}.csv", index=False)

# figures: coverage vs lead time, PIT
fig, axs = plt.subplots(1, 4, figsize=(16, 3.3))
for b in range(4):
    for k, c in zip(lo, ["0.6", "C0", "C3"]):
        inside = ((Ly >= lo[k]) & (Ly <= hi[k]))[:, :, b].mean(0)
        axs[b].plot(np.arange(180), inside, color=c, label=k)
    axs[b].axhline(LEVEL, color="k", ls="--", lw=0.8)
    axs[b].set_ylim(0, 1.02)
    axs[b].set_title(BAND_LABELS[b])
    axs[b].set_xlabel("lead time (min)")
axs[0].set_ylabel("coverage of 90 % interval")
axs[0].legend(fontsize=7)
fig.suptitle(f"{M} 5-seed ensemble: interval coverage vs lead time (CV96, cross-fold conformal) [ours]")
fig.tight_layout()
fig.savefig(ROOT / f"figures/coverage_{M}.png", dpi=130)
plt.close(fig)

fig, axs = plt.subplots(1, 4, figsize=(16, 3.0))
for b in range(4):
    pit = norm.cdf((Ly[:, :, b] - mu[:, :, b]) / z_cal[:, :, b]).ravel()
    axs[b].hist(pit, bins=10, range=(0, 1), density=True, color="C0", alpha=0.7)
    axs[b].axhline(1, color="k", ls="--", lw=0.8)
    axs[b].set_title(BAND_LABELS[b])
fig.suptitle(f"PIT histograms, {M} ensemble with conformally calibrated Gaussian scale [ours]")
fig.tight_layout()
fig.savefig(ROOT / f"figures/pit_{M}.png", dpi=130)
plt.close(fig)
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False))
