"""Subset reports for Phase 3:
  I-3  top-decile events by peak flux (10 events): RMSE / MAPE / CC per band
  I-4  impulsive vs gradual (rise time 10 % -> peak <= / > median over the 96 events)
  I-8  early forecast: accuracy vs issue lead (0, 10, 20, 30 min before the peak)
Outputs: results/improve/{extremes,type_stratified,early_forecast}.csv, figures/early_forecast.png
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
from src.data.cv96 import BANDS, load_meta, load_test96  # noqa: E402
from src.eval.metrics import per_event_metrics, pooled_metrics  # noqa: E402
from src.experiments.cv import available, load_oof  # noqa: E402

X, Y, ev = load_test96()
meta = load_meta()
out = ROOT / "results/improve"
out.mkdir(parents=True, exist_ok=True)


def models(pairs):
    res = {}
    for exp, name in pairs:
        try:
            res[f"{exp}:{name}"] = [load_oof(exp, name, s) for s in available(exp)[name]]
        except (KeyError, FileNotFoundError):
            pass
    return res


base_pairs = [("repro_cv96", n) for n in ["Power-law", "Climatology", "RandomForest", "FCN", "GRU", "Transformer"]] + \
    [("p2_norm", "GRU-peaknorm"), ("p2_zscore", "FCN-z")]
var_pairs = [("p3_extreme", "GRU-pn-wtop3"), ("p3_extreme", "GRU-pn-noise"), ("p3_extreme", "FCN-z-noise"),
             ("p3_type", "GRU-pn-experts"), ("p3_type", "GRU-pn-typefeat"), ("p3_resid", "GRU-pn-resid"),
             ("p3_resid", "FCN-z-resid"), ("p3_resid", "Prior-only"), ("p3_arch", "TSMixer"), ("p3_arch", "N-BEATS"),
             ("p3_arch", "PatchTST"), ("p3_arch", "ModernTCN")]
Ms = models(base_pairs + var_pairs)

# ---- I-3 extremes
top = np.argsort(X[:, -1])[-10:]
rows = []
for name, Ps in Ms.items():
    for sub, idx in [("top10%", top), ("rest", np.setdiff1d(np.arange(96), top)), ("all", np.arange(96))]:
        pm = [pooled_metrics(Y[idx], P[idx]) for P in Ps]
        for b, bn in enumerate(BANDS):
            rows.append(dict(model=name, subset=sub, band=bn, RMSE=np.mean([p["RMSE"][b] for p in pm]),
                             MAPE=np.mean([p["MAPE"][b] for p in pm]), CC=np.mean([p["CC"][b] for p in pm]),
                             APE=np.mean([p["APE"][b] for p in pm])))
pd.DataFrame(rows).to_csv(out / "extremes.csv", index=False)

# ---- I-4 type-stratified
thr = np.median(meta.rise_min_10pct)
typ = np.where(meta.rise_min_10pct.values <= thr, "impulsive", "gradual")
rows = []
for name, Ps in Ms.items():
    pes = [per_event_metrics(Y, P) for P in Ps]
    for t in ["impulsive", "gradual"]:
        idx = typ == t
        rec = dict(model=name, type=t, n=int(idx.sum()))
        for met in ["MAPE", "CC", "APE"]:
            v = np.mean([pe[met][idx].mean(axis=0) for pe in pes], axis=0)
            for b, bn in enumerate(BANDS):
                rec[f"{met}_{bn}"] = v[b]
            rec[f"{met}4"] = v.mean()
        rows.append(rec)
T = pd.DataFrame(rows)
T.to_csv(out / "type_stratified.csv", index=False)

# ---- I-8 early forecast
rows = []
for fam in ["GRU", "FCN"]:
    for lead in [0, 10, 20, 30]:
        tag = {"GRU": ("p2_norm", "GRU-peaknorm", "GRU-pn"), "FCN": ("p2_zscore", "FCN-z", "FCN-z")}[fam]
        exp, name = (tag[0], tag[1]) if lead == 0 else ("p3_early", f"{tag[2]}-lead{lead}")
        try:
            Ps = [load_oof(exp, name, s) for s in available(exp)[name]]
        except (KeyError, FileNotFoundError):
            continue
        pes = [per_event_metrics(Y, P) for P in Ps]
        pms = [pooled_metrics(Y, P) for P in Ps]
        rec = dict(model=fam, issue_lead_min=lead, n_seeds=len(Ps))
        for b, bn in enumerate(BANDS):
            rec[f"MAPE_{bn}"] = np.mean([pe["MAPE"][:, b].mean() for pe in pes])
            rec[f"APE_{bn}"] = np.mean([pe["APE"][:, b].mean() for pe in pes])
            rec[f"pooledCC_{bn}"] = np.mean([pm["CC"][b] for pm in pms])
        rec["MAPE4"] = np.mean([rec[f"MAPE_{b}"] for b in BANDS])
        rec["pooledCC4"] = np.mean([rec[f"pooledCC_{b}"] for b in BANDS])
        rec["peak_log_err_w0_dex"] = np.mean([np.mean(np.abs(np.log10(P[:, 0, 0]) - np.log10(Y[:, 0, 0]))) for P in Ps])
        rows.append(rec)
EF = pd.DataFrame(rows)
EF.to_csv(out / "early_forecast.csv", index=False)
if len(EF):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
    for fam, g in EF.groupby("model"):
        axs[0].plot(g.issue_lead_min, g.MAPE4, "o-", label=fam)
        axs[1].plot(g.issue_lead_min, g.APE_w0, "o-", label=fam)
        axs[2].plot(g.issue_lead_min, g.pooledCC4, "o-", label=fam)
    for ax, yl in zip(axs, ["band-mean MAPE (%)", "w0 peak APE (%)", "band-mean pooled CC"]):
        ax.set_xlabel("forecast issued N min before the peak")
        ax.set_ylabel(yl)
    axs[0].legend()
    fig.suptitle("I-8 early forecast: accuracy vs issue lead (CV96) [ours]")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/early_forecast.png", dpi=130)
pd.set_option("display.width", 220)
print(T[["model", "type", "n", "MAPE4", "CC4", "APE4"]].round(3).to_string(index=False))
print(EF.round(3).to_string(index=False))
