"""Phase 2 item 4 on the AUTHORS' models: is 'GRU is best' significant?
Input : external/zenodo/event_metrics_same_predictions_compact.csv  [zenodo] (8 models x 96 events x 4 bands)
Output: results/verify/significance_zenodo_gru.csv  (paired bootstrap 10k, Wilcoxon, Cliff's delta, Holm)
        results/verify/zenodo_rank_summary.csv        (mean per-event metric and rank per band)
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.stats_tests import band_mean_table, compare_to_reference  # noqa: E402

E = pd.read_csv(ROOT / "external/zenodo/event_metrics_same_predictions_compact.csv")
E["absMRE"] = E["MRE"].abs()
R = compare_to_reference(E, "GRU")
G = band_mean_table(E)
R2 = compare_to_reference(G, "GRU")
R = pd.concat([R, R2], ignore_index=True)
R.to_csv(ROOT / "results/verify/significance_zenodo_gru.csv", index=False)

rows = []
for (band), g in pd.concat([E, G]).groupby("band"):
    for met in ["CC", "RMSE", "absMRE", "MAPE", "APE"]:
        s = g.groupby("model")[met].mean()
        asc = met != "CC"
        rk = s.rank(ascending=asc)
        for mdl in s.index:
            rows.append(dict(band=band, metric=met, model=mdl, mean=s[mdl], rank=int(rk[mdl])))
pd.DataFrame(rows).to_csv(ROOT / "results/verify/zenodo_rank_summary.csv", index=False)

pd.set_option("display.width", 200)
show = R[R.metric.isin(["CC", "MAPE"])][["metric", "band", "competitor", "mean_ref", "mean_comp", "ci_lo", "ci_hi",
                                         "p_wilcoxon_holm", "cliffs_delta", "ref_better"]]
print(show.round(4).to_string(index=False))
