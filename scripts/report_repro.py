"""Phase 1 tables and figures from results/repro_cv96 (CV96 proxy reproduction).

Outputs
  results/repro/summary.csv          model x band x metric: ours mean/std over 3 seeds [ours] vs [zenodo]
  results/repro/compare_paper.csv    wide comparison incl. reproduction criteria (|dCC|<=0.05, |dRMSE|<=15 %)
  results/repro/skill_ours.csv       skill scores vs our CV power-law (RMSE, MAPE, APE) next to [zenodo]
  figures/repro_event_panels.png     6 representative events (2 X, 4 M; impulsive and gradual)
  figures/repro_cc_boxplot.png       per-event CC: ours (CV96, seed 0..2 averaged) vs authors [zenodo]
  figures/repro_mape_boxplot.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.cv96 import BANDS, load_meta, load_test96  # noqa: E402
from src.experiments.cv import available, load_oof  # noqa: E402
from src.viz.plots import event_panels, metric_boxplot  # noqa: E402

EXP = "repro_cv96"
PAPER_MODELS = ["Power-law", "LinearReg", "RandomForest", "FCN", "RNN", "LSTM", "GRU", "Transformer"]
EXTRA = ["Persistence-linextrap", "Persistence-peak", "Climatology"]
out = ROOT / "results/repro"
out.mkdir(parents=True, exist_ok=True)

Pm = pd.read_csv(ROOT / f"results/{EXP}/metrics_pooled.csv")
Ev = pd.read_csv(ROOT / f"results/{EXP}/metrics_event.csv")
Z = pd.read_csv(ROOT / "external/zenodo/summary_metrics_same_predictions.csv")
Ze = pd.read_csv(ROOT / "external/zenodo/event_metrics_same_predictions_compact.csv")

metrics = ["CC", "RMSE", "MRE", "MAPE", "APE", "Mean_event_CC", "Median_event_RMSE"]
rows = []
for (m, b), g in Pm.groupby(["model", "band"]):
    z = Z[(Z.model == m) & (Z.band == b)]
    for met in metrics:
        zv = float(z[met].iloc[0]) if len(z) and met in z else np.nan
        if met == "Mean_event_CC" and len(z):
            zv = Ze[(Ze.model == m) & (Ze.band == b)]["CC"].mean()
        rows.append(dict(model=m, band=b, metric=met, ours_mean=g[met].mean(), ours_std=g[met].std(ddof=1) if len(g) > 1 else 0.0,
                         n_seeds=len(g), zenodo=zv))
S = pd.DataFrame(rows)
S["diff"] = S.ours_mean - S.zenodo
S["rel_diff"] = S["diff"] / S.zenodo.abs()
S.to_csv(out / "summary.csv", index=False)

W = []
for m in PAPER_MODELS + EXTRA:
    for b in BANDS + ["mean4"]:
        if b == "mean4":
            g = S[(S.model == m)]
            cc = g[g.metric == "CC"]
            rm = g[g.metric == "RMSE"]
            if len(cc) < 4:
                continue
            rec = dict(model=m, band=b, CC_ours=cc.ours_mean.mean(), CC_zenodo=cc.zenodo.mean(),
                       RMSE_reldiff=rm.rel_diff.mean())
        else:
            g = S[(S.model == m) & (S.band == b)].set_index("metric")
            if not len(g):
                continue
            rec = dict(model=m, band=b)
            for met in ["CC", "RMSE", "MAPE", "APE", "MRE"]:
                rec[f"{met}_ours"] = g.loc[met, "ours_mean"]
                rec[f"{met}_ours_std"] = g.loc[met, "ours_std"]
                rec[f"{met}_zenodo"] = g.loc[met, "zenodo"]
            rec["RMSE_reldiff"] = g.loc["RMSE", "rel_diff"]
        rec["dCC"] = rec["CC_ours"] - rec["CC_zenodo"]
        rec["pass_CC(<=0.05)"] = bool(abs(rec["dCC"]) <= 0.05) if np.isfinite(rec["dCC"]) else None
        rec["pass_RMSE(<=15%)"] = bool(abs(rec["RMSE_reldiff"]) <= 0.15) if np.isfinite(rec["RMSE_reldiff"]) else None
        W.append(rec)
W = pd.DataFrame(W)
W.to_csv(out / "compare_paper.csv", index=False)

# skill scores (ours: vs our CV power-law, seed-averaged pooled metrics)
pm_mean = Pm.groupby(["model", "band"])[["RMSE", "MAPE", "APE"]].mean()
sk = []
for m in PAPER_MODELS + EXTRA:
    if m not in pm_mean.index.get_level_values(0):
        continue
    for E in ["RMSE", "MAPE", "APE"]:
        zs = pd.read_csv(ROOT / f"external/zenodo/skill_{E.lower()}_same_predictions.csv").set_index("model")
        vals = []
        for i, b in enumerate(BANDS):
            s = 1 - pm_mean.loc[(m, b), E] / pm_mean.loc[("Power-law", b), E]
            vals.append(s)
            zv = zs.iloc[:, i].get(m, np.nan)
            sk.append(dict(model=m, metric=E, band=b, skill_ours=s, skill_zenodo=zv))
        sk.append(dict(model=m, metric=E, band="mean4", skill_ours=np.mean(vals),
                       skill_zenodo=zs["Mean"].get(m, np.nan)))
SK = pd.DataFrame(sk)
SK.to_csv(out / "skill_ours.csv", index=False)

# ------------------------------------------------------------------ figures
X, Y, ev = load_test96()
meta = load_meta()
av = available(EXP)


def mean_pred(name):
    seeds = av[name]
    L = np.mean([np.log10(load_oof(EXP, name, s)) for s in seeds], axis=0)
    return 10 ** L


preds = {n: mean_pred(n) for n in ["Power-law", "RandomForest", "FCN", "GRU", "Transformer"] if n in av}
want = ["20031103_X3.9", "20050219_M3.3", "20110804_M9.3"]
xs = meta[meta.is_x & ~meta.event.isin(want)].sort_values("rise_min_10pct")
ms = meta[~meta.is_x & ~meta.event.isin(want)].sort_values("rise_min_10pct")
sel = [ev.index(w) for w in want if w in ev]
sel += [int(xs.k.iloc[-1])]                     # most gradual remaining X flare
sel += [int(ms.k.iloc[0]), int(ms.k.iloc[-1])]  # most impulsive / most gradual M flare
sel = sel[:6]
pd.DataFrame({"event": [ev[k] for k in sel], "rise_min_10pct": meta.rise_min_10pct.values[sel]}).to_csv(
    out / "representative_events.csv", index=False)
event_panels(sel, ev, X, Y, preds, ROOT / "figures/repro_event_panels.png",
             title="CV96 out-of-fold predictions (3-seed mean) vs FISM2 [ours]")

Eo = Ev.groupby(["model", "band", "event"])[["CC", "MAPE", "APE", "MRE", "RMSE"]].mean().reset_index()
Eo["source"] = "ours (CV96)"
Zb = Ze.copy()
Zb["source"] = "authors [zenodo]"
B = pd.concat([Eo[Eo.model.isin(PAPER_MODELS)], Zb], ignore_index=True)
metric_boxplot(B, "CC", ROOT / "figures/repro_cc_boxplot.png", order=PAPER_MODELS,
               title="Per-event CC on the 96 test events")
metric_boxplot(B, "MAPE", ROOT / "figures/repro_mape_boxplot.png", order=PAPER_MODELS,
               title="Per-event MAPE (%) on the 96 test events", log=True)

pd.set_option("display.width", 250)
print(W[["model", "band", "CC_ours", "CC_zenodo", "dCC", "RMSE_reldiff", "pass_CC(<=0.05)", "pass_RMSE(<=15%)"]]
      .round(3).to_string(index=False))
print(SK[SK.band == "mean4"].round(3).to_string(index=False))
