"""Final comparison table (results/final/compare.csv) and before/after figure.

Rows: the authors' GRU / FCN / RF as published [zenodo] (trained on 772 events), and our CV96 models
(paper configurations and the final improved model) [ours]. Columns: pooled CC/RMSE per band,
event-mean MRE/MAPE/APE, skill vs power-law (RMSE, MAPE, APE; our CV96 power-law for [ours], the
authors' power-law for [zenodo]), CPU latency per event (ms, batch 1) and parameter count.
Configured in configs/final.yaml.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.cv96 import BANDS  # noqa: E402
from src.viz.plots import metric_boxplot  # noqa: E402

cfg = yaml.safe_load(open(ROOT / "configs/final.yaml"))
Z = pd.read_csv(ROOT / "external/zenodo/summary_metrics_same_predictions.csv")
rows = []


def add(label, source, pm, pl, lat=np.nan, params=np.nan, n_seeds=np.nan):
    rec = dict(model=label, source=source, n_seeds=n_seeds)
    for met in ["CC", "RMSE", "MRE", "MAPE", "APE"]:
        for b in BANDS:
            rec[f"{met}_{b}"] = pm.loc[b, met]
        rec[f"{met}_mean4"] = pm[met].mean() if met in ("CC", "MRE", "MAPE", "APE") else np.nan
    for E in ["RMSE", "MAPE", "APE"]:
        ss = 1 - pm[E] / pl[E]
        rec[f"SS_{E}_mean4"] = ss.mean()
    rec["latency_ms_cpu"] = lat
    rec["params"] = params
    rows.append(rec)


zpl = Z[Z.model == "Power-law"].set_index("band")
for m in cfg.get("zenodo", []):
    add(f"{m} (authors)", "[zenodo]", Z[Z.model == m].set_index("band"), zpl)

opl = pd.read_csv(ROOT / "results/repro_cv96/metrics_pooled.csv")
opl = opl[opl.model == "Power-law"].groupby("band")[["CC", "RMSE", "MRE", "MAPE", "APE"]].mean()
for item in cfg["ours"]:
    exp, name, label = item["exp"], item["name"], item["label"]
    P = pd.read_csv(ROOT / f"results/{exp}/metrics_pooled.csv")
    P = P[P.model == name]
    pm = P.groupby("band")[["CC", "RMSE", "MRE", "MAPE", "APE"]].mean()
    lat = params = np.nan
    base_name, mult = (name[:-5], 5) if name.endswith("-ens5") else (name, 1)
    infos = sorted((ROOT / f"results/{exp}").glob(f"info_{base_name}_s*.json"))
    if infos:
        I = json.load(open(infos[0]))
        lat = mult * float(np.median([i.get("latency_ms", np.nan) for i in I]))      # ensemble = members in series
        params = mult * float(np.median([i.get("params", np.nan) for i in I]))
    add(label, "[ours] CV96", pm, opl, lat, params, P.seed.nunique())
C = pd.DataFrame(rows)
out = ROOT / "results/final"
out.mkdir(parents=True, exist_ok=True)
C.to_csv(out / "compare.csv", index=False)
pd.set_option("display.width", 250)
print(C[["model", "source", "CC_mean4", "MAPE_mean4", "APE_mean4", "SS_RMSE_mean4", "SS_MAPE_mean4", "SS_APE_mean4",
         "latency_ms_cpu", "params"]].round(3).to_string(index=False))

# before/after per-event figure
ev_rows = []
for item in cfg["ours"]:
    E = pd.read_csv(ROOT / f"results/{item['exp']}/metrics_event.csv")
    E = E[E.model == item["name"]].groupby(["band", "event"])[["CC", "MAPE"]].mean().reset_index()
    E["model"] = item["label"]
    E["source"] = "ours"
    ev_rows.append(E)
B = pd.concat(ev_rows)
order = [i["label"] for i in cfg["ours"]]
metric_boxplot(B, "MAPE", ROOT / "figures/final_mape_before_after.png", order=order, log=True,
               title="Per-event MAPE before/after improvements (CV96) [ours]")
metric_boxplot(B, "CC", ROOT / "figures/final_cc_before_after.png", order=order,
               title="Per-event CC before/after improvements (CV96) [ours]")

# lead-time error of the final model vs the paper configurations (log10 RMSE per lead minute)
from src.data.cv96 import load_test96  # noqa: E402
from src.experiments.cv import available, load_oof  # noqa: E402
from src.viz.plots import lead_time_curves  # noqa: E402
X, Y, ev = load_test96()
curves = {}
for item in cfg["ours"]:
    if item["name"] in ("Power-law", "GRU", "FCN", "GRU-peaknorm", "FCN-z-ens5", "RandomForest"):
        seeds = available(item["exp"])[item["name"]]
        r = np.mean([np.sqrt(np.mean((np.log10(load_oof(item["exp"], item["name"], s_)) - np.log10(Y)) ** 2, axis=0))
                     for s_ in seeds], axis=0)
        curves[item["label"]] = r
lead_time_curves(curves, ROOT / "figures/final_leadtime_rmse.png", ylabel="RMSE of log10 irradiance (dex)",
                 title="Lead-time error: final model vs paper configurations (CV96) [ours]")
