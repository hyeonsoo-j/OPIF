"""Phase 2 item 3 — persistence / climatology references and MSE skill scores.

SS = 1 - MSE_model / MSE_ref (physical units, pooled over 96 events x 180 min, per band).
References (all evaluated on the same 96 events):
  Power-law (authors' construction)         [zenodo] pooled RMSE  /  [ours] CV96 refit
  Persistence-peak  (hold the peak; EUV peak = X-ray peak x training-median ratio)   [ours] CV96
  Persistence-linextrap (authors' 'Persistence': last-10-min linear extrapolation)  [ours] (no training)
  Climatology (training-mean log decay curve scaled to the input peak)            [ours] CV96
Models: the authors' 8 models via their pooled RMSE [zenodo]; our CV96 models [ours].
Also reported in log10 space for our models (MSE of log10 irradiance = the training loss).
Output: results/verify/skill_refs.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.cv96 import BANDS, load_test96  # noqa: E402
from src.experiments.cv import available, load_oof  # noqa: E402

EXP = "repro_cv96"
X, Y, ev = load_test96()
av = available(EXP)
REFS = ["Power-law", "Persistence-peak", "Persistence-linextrap", "Climatology"]


def seed_mean_mse(name, log=False):
    out = []
    for s in av[name]:
        P = load_oof(EXP, name, s)
        if log:
            out.append(np.mean((np.log10(P) - np.log10(Y)) ** 2, axis=(0, 1)))
        else:
            out.append(np.mean((P - Y) ** 2, axis=(0, 1)))
    return np.mean(out, axis=0)


mse_ref = {r: seed_mean_mse(r) for r in REFS}
mse_ref_log = {r: seed_mean_mse(r, log=True) for r in REFS}
Z = pd.read_csv(ROOT / "external/zenodo/summary_metrics_same_predictions.csv")
mse_ref_z = {"Power-law[zenodo]": np.array([Z[(Z.model == "Power-law") & (Z.band == b)].RMSE.iloc[0] ** 2 for b in BANDS])}

rows = []
# authors' models (pooled RMSE from Zenodo)
for m in Z.model.unique():
    mse = np.array([Z[(Z.model == m) & (Z.band == b)].RMSE.iloc[0] ** 2 for b in BANDS])
    for refname, ref in list(mse_ref.items()) + list(mse_ref_z.items()):
        ss = 1 - mse / ref
        rows.append(dict(source="authors [zenodo]", model=m, reference=refname, space="linear",
                         **{b: ss[i] for i, b in enumerate(BANDS)}, mean4=ss.mean()))
# our CV96 models
for m in av:
    mse = seed_mean_mse(m)
    mse_l = seed_mean_mse(m, log=True)
    for refname in REFS:
        ss = 1 - mse / mse_ref[refname]
        rows.append(dict(source="ours [CV96]", model=m, reference=refname, space="linear",
                         **{b: ss[i] for i, b in enumerate(BANDS)}, mean4=ss.mean()))
        ssl = 1 - mse_l / mse_ref_log[refname]
        rows.append(dict(source="ours [CV96]", model=m, reference=refname, space="log10",
                         **{b: ssl[i] for i, b in enumerate(BANDS)}, mean4=ssl.mean()))
R = pd.DataFrame(rows)
R.to_csv(ROOT / "results/verify/skill_refs.csv", index=False)
pd.set_option("display.width", 220)
for src in ["authors [zenodo]", "ours [CV96]"]:
    for ref in ["Persistence-peak", "Climatology"]:
        sub = R[(R.source == src) & (R.reference == ref) & (R.space == "linear")]
        print(f"\n== {src}  SS_MSE vs {ref} (linear)")
        print(sub[["model"] + BANDS + ["mean4"]].round(3).to_string(index=False))
sub = R[(R.source == "ours [CV96]") & (R.reference == "Climatology") & (R.space == "log10")]
print("\n== ours SS_MSE(log10) vs Climatology")
print(sub[["model"] + BANDS + ["mean4"]].round(3).to_string(index=False))
