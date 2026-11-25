"""Phase 2 item 7 — leakage checks.

(a) Paper split (catalogue-level): for each test event, catalogue M/X flares (other than the event)
    whose 180-min OUTPUT window overlaps the test event's 60-min INPUT window, i.e. peaks within
    [t-238, t-1] min; and flares from the same NOAA AR within +-24 h (potential near-duplicates
    in the authors' training set). Only flares that pass the isolation rule ('selected') could be in
    the authors' training set.
(b) Our CV96: interleaved (default) vs blocked-in-time folds (results/p2_blocked) -> metric change.
Outputs: results/verify/leakage_catalogue.csv, results/verify/leakage_cv_scheme.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.cv96 import load_meta  # noqa: E402

m = load_meta()
cat = pd.read_csv(ROOT / "data/processed/flare_catalogue.csv", parse_dates=["start", "peak", "end"])
rows = []
for r in m.itertuples():
    t = r.peak_time
    others = cat[(cat.peak != t)]
    dt = (t - others.peak).dt.total_seconds() / 60
    ov = others[(dt >= 1) & (dt <= 238)]
    same_ar = pd.DataFrame()
    row = cat[cat.peak == t]
    ar = row.ar.iloc[0] if len(row) else np.nan
    if pd.notna(ar):
        same_ar = others[(others.ar == ar) & ((others.peak - t).abs() <= pd.Timedelta(hours=24))]
    rows.append(dict(event=r.event, n_output_overlaps_input=len(ov), n_overlap_selected=int(ov.selected.sum()),
                     overlap_events=";".join(ov.event_id), ar=ar, n_same_ar_24h=len(same_ar),
                     n_same_ar_24h_selected=int(same_ar.selected.sum()) if len(same_ar) else 0))
L = pd.DataFrame(rows)
L.to_csv(ROOT / "results/verify/leakage_catalogue.csv", index=False)
print("test events whose input window is overlapped by another flare's output window:",
      (L.n_output_overlaps_input > 0).sum(), "/ 96; of which by isolation-rule-eligible flares:",
      (L.n_overlap_selected > 0).sum())
print("test events with a same-AR M/X flare within 24 h (AR known for", L.ar.notna().sum(), "events):",
      (L.n_same_ar_24h > 0).sum(), "; eligible:", (L.n_same_ar_24h_selected > 0).sum())

# (b) interleaved vs blocked
try:
    A = pd.concat([pd.read_csv(ROOT / "results/repro_cv96/metrics_pooled.csv"),
                   pd.read_csv(ROOT / "results/p2_norm/metrics_pooled.csv")])
    B = pd.read_csv(ROOT / "results/p2_blocked/metrics_pooled.csv")
    out = []
    for mdl in B.model.unique():
        for band in ["w0", "w1", "w2", "w3"]:
            a = A[(A.model == mdl) & (A.band == band)]
            b = B[(B.model == mdl) & (B.band == band)]
            out.append(dict(model=mdl, band=band, CC_interleaved=a.CC.mean(), CC_blocked=b.CC.mean(),
                            dCC=b.CC.mean() - a.CC.mean(), MAPE_interleaved=a.MAPE.mean(), MAPE_blocked=b.MAPE.mean(),
                            dMAPE=b.MAPE.mean() - a.MAPE.mean(),
                            RMSE_rel_change=b.RMSE.mean() / a.RMSE.mean() - 1))
    O = pd.DataFrame(out)
    O.to_csv(ROOT / "results/verify/leakage_cv_scheme.csv", index=False)
    print(O.round(3).to_string(index=False))
except FileNotFoundError:
    print("p2_blocked not available yet")

# (c) authors' per-event scores with / without the potentially leaky test events [zenodo-derived]
Ze = pd.read_csv(ROOT / "external/zenodo/event_metrics_same_predictions_compact.csv")
flag_ov = set(L.event[L.n_output_overlaps_input > 0])
flag_ar = set(L.event[L.n_same_ar_24h_selected > 0])
rows = []
for mdl, g in Ze.groupby("model"):
    for lab, excl in [("all 96", set()), ("without output-overlap events", flag_ov),
                      ("without same-AR-24h events", flag_ar)]:
        h = g[~g.event.isin(excl)]
        rows.append(dict(model=mdl, subset=lab, n_events=h.event.nunique(), CC_event_mean4=h.CC.mean(),
                         MAPE_event_mean4=h.MAPE.mean(), APE_event_mean4=h.APE.mean()))
C = pd.DataFrame(rows)
C.to_csv(ROOT / "results/verify/leakage_subset_zenodo.csv", index=False)
print(C.pivot(index="model", columns="subset", values="MAPE_event_mean4").round(3).to_string())
