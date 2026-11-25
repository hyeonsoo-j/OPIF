"""Paper-1 tables (P1-P6) from the server results.

  python notebooks/analyze_paper1.py --results results --rebuilt data/processed/events_test96_rebuilt.npz

P1/P2  input variants: FISM2 (p2_zscore_full_gpu5) vs GOES direct (_p1) vs GOES->FISM2 converter (_p2 linear, _p2nn)
       per model (FCN-z, GRU-z), 5-seed ensembles, both test sets, paired tests vs the FISM2 input.
P3     issue delay k = 2/5/10 (paper1_delay_full): metrics on lead >= k vs the lead-0 model on the same window,
       plus the early-forecast models (p3_early_full, lead -10/-20/-30) on the full window.
P4     peak detector summary is produced by notebooks/peak_detector.py (peak_detector.csv) - copied into the table.
P5     conditional conformal: conformal_split_*.csv are produced by split_conformal_full.py - summarised here.
P6     D-RAP MAE / peak error for the P1-P3 models (daytime events).
Outputs: results/full_server/paper1_inputs.csv, paper1_delay.csv, paper1_drap.csv, paper1_conformal.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))
from analyze_full_server import ev_metrics, holm, pearson, perm, pooled  # noqa: E402
from src.data.cv96 import load_meta, load_test96  # noqa: E402
from src.eval.drap import event_absorption  # noqa: E402

B = ["w0", "w1", "w2", "w3"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--rebuilt", default=str(ROOT / "data/processed/events_test96_rebuilt.npz"))
    ap.add_argument("--out", default=str(ROOT / "results/full_server"))
    a = ap.parse_args()
    R, OUT = Path(a.results), Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    X96, Y96, ev = load_test96()
    Ytest = {"zenodo96": Y96}
    if Path(a.rebuilt).exists():
        d = np.load(a.rebuilt, allow_pickle=False)
        o = np.argsort(d["test_k"])
        if len(o) == 96:
            Ytest["rebuilt96"] = d["Y"][o].astype(np.float64)
    META = load_meta()

    def load(exp, name, test):
        pre = "oof" if test == "zenodo96" else "oofL"
        fs = sorted(glob.glob(f"{R}/{exp}/{pre}_{name}_s*.npy"))
        return [10 ** np.load(f).astype(np.float64) for f in fs]

    def ens(Ps):
        return 10 ** np.mean([np.log10(P) for P in Ps], axis=0)

    # ---- P1/P2 inputs
    variants = [("FISM2 (benchmark input)", "p2_zscore_full_gpu5"), ("GOES XRS-B direct", "p2_zscore_full_in-goes_b_p1"),
                ("GOES->FISM2 linear", "p2_zscore_full_in-goes_conv_p2"), ("GOES->FISM2 FCN", "p2_zscore_full_in-goes_convnn_p2nn")]
    rows, PEm = [], {}
    for m in ("FCN-z", "GRU-z"):
        for label, exp in variants:
            for test, yt in Ytest.items():
                Ps = load(exp, m, test)
                if not Ps:
                    continue
                ok = ~np.isnan(np.stack(Ps)).any(axis=(0, 2, 3))
                E = ens(Ps)
                pm = pooled(yt[ok], E[ok])
                sm = pd.DataFrame([pooled(yt[ok], P[ok]) for P in Ps]).mean()
                rows.append(dict(model=m, input=label, exp=exp, test=test, n_seeds=len(Ps), n_ev=int(ok.sum()),
                                 CC4=sm.CC4, MAPE4=sm.MAPE4, APE4=sm.APE4, ensCC4=pm["CC4"], ensMAPE4=pm["MAPE4"], ensAPE4=pm["APE4"],
                                 **{f"ensMAPE_{b}": pm[f"MAPE_{b}"] for b in B}))
                PEm[(m, label, test)] = (np.mean([ev_metrics(yt, P)[0].mean(1) for P in Ps], axis=0), ok)
    T = pd.DataFrame(rows)
    tests = []
    for m in ("FCN-z", "GRU-z"):
        for label, _ in variants[1:]:
            for test in Ytest:
                if (m, label, test) not in PEm or (m, variants[0][0], test) not in PEm:
                    continue
                x, okx = PEm[(m, label, test)]
                y, oky = PEm[(m, variants[0][0], test)]
                ok = okx & oky
                dm, p = perm(x[ok], y[ok])
                tests.append(dict(model=m, input=label, test=test, dMAPE4_vs_FISM2=dm, p=p))
    Tt = pd.DataFrame(tests)
    if len(Tt):
        Tt["p_holm"] = holm(Tt.p.values)
        T = T.merge(Tt, on=["model", "input", "test"], how="left")
    T.to_csv(OUT / "paper1_inputs.csv", index=False)
    pd.set_option("display.width", 250)
    print("P1/P2 inputs:")
    cols = [c for c in ["model", "input", "test", "n_seeds", "ensCC4", "ensMAPE4", "ensAPE4", "dMAPE4_vs_FISM2", "p", "p_holm"] if c in T.columns]
    print(T[cols].round(3).to_string(index=False) if len(T) else "  (none)")

    # ---- P3 issue delay: metrics on lead >= k
    rows = []
    base = {"GRU-z": ("p2_zscore_full_gpu5", "GRU-z"), "FCN-z": ("p2_zscore_full_gpu5", "FCN-z"), "GRU-pn": ("p3_ensemble_full_gpu", "GRU-peaknorm")}
    for test, yt in Ytest.items():
        for fam in ("GRU-z", "FCN-z", "GRU-pn"):
            Pb = load(*base[fam], test)
            Eb = ens(Pb) if Pb else None
            for k in (0, 2, 5, 10):
                if k == 0:
                    if Eb is None:
                        continue
                    E = Eb
                else:
                    Ps = load("paper1_delay_full", f"{fam}-d{k}", test)
                    if not Ps:
                        continue
                    E = ens(Ps)
                sl = slice(k, 180)
                mape = 100 * (np.abs(E[:, sl] - yt[:, sl]) / yt[:, sl]).mean(axis=1)           # (N,4)
                rel0 = 100 * (np.abs(E[:, k] - yt[:, k]) / yt[:, k])                            # error at issue time
                ccs = [pearson(yt[:, sl, b], E[:, sl, b]) for b in range(4)]
                rec = dict(test=test, family=fam, issue_delay_min=k, n_lead=180 - k, CC4_lead_ge_k=np.mean(ccs),
                           MAPE4_lead_ge_k=mape.mean(), APE4_at_issue=rel0.mean(), MAPE4_lead0_model_same_window=np.nan,
                           dMAPE4_vs_lead0_model=np.nan, p=np.nan)
                if Eb is not None and k > 0:            # the lead-0 model scored on the same window (like-for-like)
                    mape0 = 100 * (np.abs(Eb[:, sl] - yt[:, sl]) / yt[:, sl]).mean(axis=1)
                    rec.update(MAPE4_lead0_model_same_window=mape0.mean(), dMAPE4_vs_lead0_model=mape.mean() - mape0.mean(),
                               p=perm(mape.mean(1), mape0.mean(1))[1])
                rows.append(rec)
        for fam, name0 in (("GRU-pn", "GRU-pn"), ("FCN-z", "FCN-z")):
            for lead in (10, 20, 30):
                Ps = load("p3_early_full", f"{name0}-lead{lead}", test)
                if not Ps:
                    continue
                E = ens(Ps)
                mape = 100 * (np.abs(E - yt) / yt).mean(axis=1)
                rows.append(dict(test=test, family=fam, issue_delay_min=-lead, n_lead=180, CC4_lead_ge_k=np.mean([pearson(yt[:, :, b], E[:, :, b]) for b in range(4)]),
                                 MAPE4_lead_ge_k=mape.mean(), APE4_at_issue=(100 * np.abs(E[:, 0] - yt[:, 0]) / yt[:, 0]).mean(),
                                 MAPE4_lead0_model_same_window=np.nan, dMAPE4_vs_lead0_model=np.nan, p=np.nan))
    D = pd.DataFrame(rows)
    if len(D):
        D = D.sort_values(["test", "family", "issue_delay_min"])
    D.to_csv(OUT / "paper1_delay.csv", index=False)
    print("\nP3 issue delay (ensembles):")
    print(D.round(3).to_string(index=False) if len(D) else "  (none)")

    # ---- P6 D-RAP for the input variants and delays (w0, daytime events)
    # A model issued k minutes after the peak has the 0.1-0.8 nm minutes 1..k among its inputs, so it is scored on the
    # minutes it really forecasts (lead >= k) and next to the peak-issued ensemble on the same minutes; its peak-minute
    # error is not a forecast and is left empty.
    rows = []
    for test, yt in Ytest.items():
        cands = [(f"{m} | {label}", exp, m, 0, None) for m in ("FCN-z", "GRU-z") for label, exp in variants] + \
                [(f"{fam}-d{k}", "paper1_delay_full", f"{fam}-d{k}", k, base[fam]) for fam in ("GRU-z", "FCN-z", "GRU-pn") for k in (2, 5, 10)]
        for label, exp, m, k, ref in cands:
            Ps = load(exp, m, test)
            if not Ps:
                continue
            E = ens(Ps)
            Pr = load(*ref, test) if ref else []
            Er = ens(Pr) if Pr else None
            errs = []
            for e in range(96):
                if np.isnan(E[e]).any():
                    continue
                a_t, zen = event_absorption(yt[e, :, 0], META.peak_time.iloc[e])
                if zen[0] >= 90:
                    continue
                a_p, _ = event_absorption(E[e, :, 0], META.peak_time.iloc[e])
                dd = (zen < 90) & (np.arange(len(zen)) >= k)
                rec = dict(MAE=np.abs(a_p[dd] - a_t[dd]).mean(), peak_abs=abs(a_p[0] - a_t[0]) if k == 0 else np.nan, ref=np.nan)
                if Er is not None:
                    rec["ref"] = np.abs(event_absorption(Er[e, :, 0], META.peak_time.iloc[e])[0][dd] - a_t[dd]).mean()
                errs.append(rec)
            Ed = pd.DataFrame(errs)
            rows.append(dict(test=test, model=label, n_day=len(Ed), lead_from_min=k, MAE_dB=Ed.MAE.mean(), abs_peak_err_dB=Ed.peak_abs.mean(),
                             MAE_dB_peak_issued_same_leads=Ed.ref.mean()))
    DR = pd.DataFrame(rows)
    DR.to_csv(OUT / "paper1_drap.csv", index=False)
    print("\nP6 D-RAP:")
    print(DR.round(3).to_string(index=False) if len(DR) else "  (none)")

    # ---- P5 conformal summaries (files from split_conformal_full.py) and P4 detector
    cf = []
    for f in sorted(glob.glob(f"{OUT}/conformal_split*.csv")):
        if "_events" in os.path.basename(f) or os.path.getsize(f) < 10:
            continue
        try:
            C = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if "model" not in C.columns:
            continue
        C["file"] = os.path.basename(f)
        cf.append(C)
    if cf:
        C = pd.concat(cf)
        keep = ["file", "model", "test", "cond"] + [f"cov_norm_{b}" for b in B] + ["width_norm_w0_dex", "cov_norm_w0_Xclass", "cov_norm_w0_Mclass", "drap_cov_norm", "drap_width_dB"]
        C = C[[c for c in keep if c in C.columns]]
        C.to_csv(OUT / "paper1_conformal.csv", index=False)
        print("\nP5 conformal:")
        print(C.round(3).to_string(index=False))
    pk = OUT / "peak_detector.csv"
    if pk.exists():
        print("\nP4 peak detector:")
        print(pd.read_csv(pk).round(3).to_string(index=False))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
