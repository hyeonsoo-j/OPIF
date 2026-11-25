"""Phase 2 items 5 and 8 from the CV96 runs:
  item 8 input-window sensitivity (15/30/45/60 min; FCN, GRU)      -> results/verify/input_window.csv, figures/input_window.png
  item 5 scale invariance (input x0.1 / x10) and peak normalisation -> results/verify/scale_invariance.csv,
                                                                       results/verify/normalisation.csv
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.cv96 import BANDS  # noqa: E402


def pooled(exp, name):
    P = pd.read_csv(ROOT / f"results/{exp}/metrics_pooled.csv")
    P = P[P.model == name]
    return P.groupby("band")[["CC", "MAPE", "RMSE", "Mean_event_CC"]].agg(["mean", "std"])


# ---- item 8
rows = []
for fam in ["FCN", "GRU"]:
    for w in [15, 30, 45, 60]:
        exp, name = ("p2_norm", f"{fam}-peaknorm") if w == 60 else ("p2_window", f"{fam}-pn-w{w}")
        try:
            g = pooled(exp, name)
        except FileNotFoundError:
            continue
        if not len(g):
            continue
        rec = dict(model=fam, window_min=w)
        for b in BANDS:
            rec[f"CC_{b}"] = g.loc[b, ("CC", "mean")]
            rec[f"CC_{b}_std"] = g.loc[b, ("CC", "std")]
            rec[f"MAPE_{b}"] = g.loc[b, ("MAPE", "mean")]
        rec["CC4"] = np.mean([rec[f"CC_{b}"] for b in BANDS])
        rec["MAPE4"] = np.mean([rec[f"MAPE_{b}"] for b in BANDS])
        rows.append(rec)
W = pd.DataFrame(rows)
W.to_csv(ROOT / "results/verify/input_window.csv", index=False)
if len(W):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for fam, g in W.groupby("model"):
        axs[0].plot(g.window_min, g.CC4, "o-", label=fam)
        axs[1].plot(g.window_min, g.MAPE4, "o-", label=fam)
    axs[0].set_ylabel("band-mean pooled CC")
    axs[1].set_ylabel("band-mean MAPE (%)")
    for ax in axs:
        ax.set_xlabel("input window (min before peak)")
    axs[0].legend()
    fig.suptitle("Input-window sensitivity, CV96 (90/120 min not available in the archive) [ours]")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/input_window.png", dpi=130)
print(W.round(3).to_string(index=False))

# ---- item 5a scale invariance
rows = []
d = ROOT / "results/p2_scale"
for f in sorted(d.glob("info_*_s0.json")):
    name = f.stem[5:-3]
    infos = json.load(open(f))
    for fct in ["0.1", "10"]:
        sh = {}
        for inf in infos:
            for bands, v in inf["scale_shift"][fct].items():
                arr = np.array(v)  # (n_te, n_bands_of_cfg)
                for j, b in enumerate(map(int, bands.split(","))):
                    sh.setdefault(b, []).extend(arr[:, j].tolist())
        expected = np.log10(float(fct))
        for b in range(4):
            v = np.array(sh[b])
            rows.append(dict(model=name, factor=float(fct), band=BANDS[b], expected_w0_shift=expected if b == 0 else np.nan,
                             mean_shift_dex=v.mean(), std_shift_dex=v.std(), min_shift=v.min(), max_shift=v.max()))
S = pd.DataFrame(rows)
S.to_csv(ROOT / "results/verify/scale_invariance.csv", index=False)
print(S.round(3).to_string(index=False))

# ---- item 5b preprocessing variants (paper: global log10 only)
rows = []
VARIANTS = [("repro_cv96", "{}", "log10 (paper)"), ("p2_norm", "{}-peaknorm", "peak-normalised"),
            ("p2_norm2", "{}-peaknorm", "peak-normalised"), ("p2_zscore", "{}-z", "global z-score"),
            ("p2_hybrid", "{}-hyb", "hybrid (w0 peak-rel., EUV abs.; z-scored)")]
for fam in ["FCN", "GRU", "LSTM", "RNN", "Transformer"]:
    for exp, pat, lab in VARIANTS:
        name = pat.format(fam)
        try:
            g = pooled(exp, name)
        except FileNotFoundError:
            continue
        if not len(g):
            continue
        rec = dict(model=fam, preprocessing=lab, exp=exp)
        for b in BANDS:
            rec[f"CC_{b}"] = g.loc[b, ("CC", "mean")]
            rec[f"evCC_{b}"] = g.loc[b, ("Mean_event_CC", "mean")]
            rec[f"MAPE_{b}"] = g.loc[b, ("MAPE", "mean")]
        rec["CC4"] = np.mean([rec[f"CC_{b}"] for b in BANDS])
        rec["evCC4"] = np.mean([rec[f"evCC_{b}"] for b in BANDS])
        rec["MAPE4"] = np.mean([rec[f"MAPE_{b}"] for b in BANDS])
        rows.append(rec)
N = pd.DataFrame(rows)
N.to_csv(ROOT / "results/verify/normalisation.csv", index=False)
print(N.round(3).to_string(index=False))
