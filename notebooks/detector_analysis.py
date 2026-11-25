"""Paper-1 P4c analysis — the three peak-detection rules on the exported GOES streams, no downloaded files needed.

Input: results/full_server/peak_detector_streams.npz (peak_detector.py --rearm --save-streams), the built dataset
(data/processed/events_latis.npz: isolation flag, test rows, peak times), the flare catalogue and, for the scores,
the prediction arrays of the GOES- and FISM2-driven ensembles.

    python notebooks/detector_analysis.py --results results --events data/processed/events_latis.npz

Outputs (results/full_server/):
  peak_detector_rules.csv        outcome shares of five rules (first maximum; + rise condition; armed detector with rise
                                 factor 2, 1.5, 1.2) for all flares, the flares that pass the benchmark's isolation rule,
                                 the others, and the 96 test flares. All rows are computed with the current code, which
                                 does not evaluate a declaration on minutes without data.
  peak_detector_sensitivity.csv  armed detector: rise factor x k (own maximum declared, latency, earlier declarations)
  peak_detector_missed_v3.csv    flares whose own maximum is not declared by the armed detector (rise factor 2), with the reason
  peak_detector_earlier_v3.csv   the declarations that precede the own maximum: timing, size, another catalogued flare?
  paper1_detector_scores.csv     test flares: forecast error from the declaration on, and split by whether another flare
                                 is declared inside the 3 h forecast window (bootstrap interval of the difference)
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))
import peak_detector as PD  # noqa: E402

THR, PRE, TOL = 1e-5, 60, 5


def _mean(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.mean()) if x.size else float("nan")
RULES = [("first", "first maximum, no new maximum for k min", None), ("first_rise", "+ rise by a factor 2 in the stream", None),
         ("armed2", "armed detector, rise factor 2", 2.0), ("armed1.5", "armed detector, rise factor 1.5", 1.5),
         ("armed1.2", "armed detector, rise factor 1.2", 1.2)]


def outcome(rule: str, b: np.ndarray, k: int, rise: float | None) -> tuple[str, float, int, int]:
    """-> (category, latency of the own-peak declaration, declarations before the own maximum, after it).
    Categories (mutually exclusive):
      own_first    the flare's own maximum is declared and nothing is declared before it
      own_after    the own maximum is declared after one or more earlier declarations (armed detector only)
      other_only   something is declared, but not the own maximum (first rules: too early or > 4 min late)
      none         nothing is declared in [peak-60, peak+180)"""
    if rise is None:
        t = PD.detect(b, k, THR, require_rise=(rule == "first_rise"))
        if t is None:
            return "none", np.nan, 0, 0
        lat = t - PRE
        return ("own_first", lat, 0, 0) if 0 <= lat <= k + 2 else ("other_only", np.nan, 0, 0)
    dec = PD.detect_all(b, k, THR, rise_factor=rise)
    o = PD.own_peak(dec, PRE, TOL)
    if not dec:
        return "none", np.nan, 0, 0
    if np.isfinite(o["own"]):
        return ("own_first" if o["n_before"] == 0 else "own_after"), o["own"], o["n_before"], o["n_after"]
    return "other_only", np.nan, o["n_before"], o["n_after"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--events", default=str(ROOT / "data/processed/events_latis.npz"))
    ap.add_argument("--rebuilt", default=str(ROOT / "data/processed/events_test96_rebuilt.npz"))
    ap.add_argument("--catalogue", default=str(ROOT / "data/processed/flare_catalogue.csv"))
    ap.add_argument("--out", default=None, help="default: <results>/full_server")
    ap.add_argument("--k", type=int, default=2)
    a = ap.parse_args(argv)
    R = Path(a.results); OUT = Path(a.out or R / "full_server"); OUT.mkdir(parents=True, exist_ok=True)
    st = np.load(R / "full_server/peak_detector_streams.npz", allow_pickle=False)
    B, row = st["xrsb"], st["row"]
    assert int(st["pre"]) == PRE
    d = np.load(a.events, allow_pickle=False)
    ist = d["is_test"][row].astype(bool)
    sel = d["selected"][row].astype(bool) | ist              # flares that pass the isolation rule of the benchmark
    isx = np.char.startswith(st["cls"], "X")
    n, k = len(B), a.k
    groups = {"all": np.ones(n, bool), "isolated": sel, "not isolated": ~sel, "test": ist, "X-class": isx, "M-class": ~isx}

    # ---- 1. outcome shares of the five rules
    rows, OUTC = [], {}
    for rule, label, rise in RULES:
        res = [outcome(rule, b, k, rise) for b in B]
        cat = np.array([r[0] for r in res]); lat = np.array([r[1] for r in res])
        nb = np.array([r[2] for r in res]); na = np.array([r[3] for r in res])
        OUTC[rule] = (cat, lat, nb, na)
        for g, m in groups.items():
            rows.append(dict(rule=rule, label=label, k=k, group=g, n=int(m.sum()),
                             own_first=_mean(cat[m] == "own_first"), own_after=_mean(cat[m] == "own_after"),
                             other_only=_mean(cat[m] == "other_only"), none=_mean(cat[m] == "none"),
                             own_declared=_mean(np.isin(cat[m], ["own_first", "own_after"])),
                             median_latency=float(np.nanmedian(lat[m])) if np.isfinite(lat[m]).any() else np.nan,
                             with_earlier=_mean(nb[m] > 0), earlier_per_flare=_mean(nb[m]),
                             later_flare_within_3h=_mean(na[m] > 0) if rise is not None else np.nan))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "peak_detector_rules.csv", index=False)
    pd.set_option("display.width", 230)
    print(T[T.group.isin(["all", "isolated", "not isolated", "test"])].drop(columns=["label", "k"]).round(3).to_string(index=False))

    # ---- 2. sensitivity of the armed detector
    rows = []
    for rise in (1.2, 1.5, 2.0, 3.0):
        for kk in (1, 2, 3, 5, 10):
            O = [PD.own_peak(PD.detect_all(b, kk, THR, rise_factor=rise), PRE, TOL) for b in B]
            own = np.array([o["own"] for o in O]); nb = np.array([o["n_before"] for o in O])
            nd = np.array([o["n_before"] + o["n_own"] + o["n_after"] for o in O])
            rows.append(dict(rise_factor=rise, k=kk, own_declared=float(np.isfinite(own).mean()), median_latency=float(np.nanmedian(own)),
                             p90_latency=float(np.nanpercentile(own, 90)), with_earlier=float((nb > 0).mean()),
                             earlier_per_flare=float(nb.mean()), declarations_per_flare=float(nd.mean()),
                             own_declared_isolated=_mean(np.isfinite(own[sel])), own_declared_not_isolated=_mean(np.isfinite(own[~sel]))))
    pd.DataFrame(rows).to_csv(OUT / "peak_detector_sensitivity.csv", index=False)

    # ---- 3. flares missed by the armed detector (rise factor 2) and why
    cat, lat, nb, na = OUTC["armed2"]
    rows = []
    for i in np.where(~np.isin(cat, ["own_first", "own_after"]))[0]:
        b = B[i]; w = b[PRE - 5:PRE + 6]
        dec = PD.detect_all(b, k, THR)
        pk = float(np.nanmax(w)) if np.isfinite(w).any() else np.nan
        lo = float(np.nanmin(b[:PRE + 1])) if np.isfinite(b[:PRE + 1]).any() else np.nan
        near = min((abs(q["t_max"] - PRE) for q in dec), default=np.nan)
        why = ("data gap at the peak" if np.isnan(w).sum() >= 3 else "GOES peak below M1" if pk < THR else
               "rise below a factor 2 in the preceding hour" if pk / lo < 2 else
               "declared maximum 6-15 min from the dataset peak" if near <= 15 else "below an earlier maximum, no rise by a factor 2 from the minimum in between")
        rows.append(dict(row=int(row[i]), event=str(st["event_id"][i]), isolated=bool(sel[i]), n_declarations=len(dec), nearest_declared_max_min=near,
                         peak_flux=pk, rise_in_preceding_hour=pk / lo if lo > 0 else np.nan, reason=why))
    M = pd.DataFrame(rows, columns=["row", "event", "isolated", "n_declarations", "nearest_declared_max_min", "peak_flux",
                                    "rise_in_preceding_hour", "reason"])
    M.to_csv(OUT / "peak_detector_missed_v3.csv", index=False)
    print(f"\narmed detector (rise 2): own maximum not declared for {len(M)} of {n} flares")
    if len(M):
        print(M.reason.value_counts().to_string())

    # ---- 4. what precedes the own maximum
    cat_f = pd.read_csv(a.catalogue, parse_dates=["peak"])
    cpk = np.sort(cat_f.peak.values.astype("datetime64[m]").astype(np.int64))
    pk_min = pd.to_datetime(d["peak"][row]).values.astype("datetime64[m]").astype(np.int64)
    rows = []
    for i, b in enumerate(B):
        w = b[PRE - 2:PRE + 3]
        pkv = float(np.nanmax(w)) if np.isfinite(w).any() else np.nan
        for q in PD.detect_all(b, k, THR):
            if q["t_max"] >= PRE - TOL:
                continue
            t_abs = pk_min[i] + q["t_max"] - PRE
            j = np.searchsorted(cpk, t_abs)
            other = any(abs(cpk[x] - t_abs) <= 5 and abs(cpk[x] - pk_min[i]) > 5 for x in (j - 1, j) if 0 <= x < len(cpk))
            rows.append(dict(row=int(row[i]), event=str(st["event_id"][i]), isolated=bool(sel[i]), minutes_before_peak=PRE - q["t_max"],
                             value=q["value"], ratio_to_own_peak=q["value"] / pkv if pkv > 0 else np.nan, kind=q["kind"],
                             another_catalogued_flare=bool(other)))
    Er = pd.DataFrame(rows, columns=["row", "event", "isolated", "minutes_before_peak", "value", "ratio_to_own_peak", "kind", "another_catalogued_flare"])
    Er.to_csv(OUT / "peak_detector_earlier_v3.csv", index=False)
    if len(Er):
        print(f"\nearlier declarations (rise 2): {len(Er)} on {Er.row.nunique()} flares; another catalogued M/X flare within 5 min: "
              f"{Er.another_catalogued_flare.mean():.2f}; median {Er.minutes_before_peak.median():.0f} min before the peak; "
              f"median size {Er.ratio_to_own_peak.median():.2f} of the own peak; larger than the own peak: {(Er.ratio_to_own_peak > 1).mean():.2f}")

    # ---- 5. test flares: error from the declaration on; error when another flare follows
    rb = Path(a.rebuilt)
    rows = []
    if rb.exists():
        r_ = np.load(rb, allow_pickle=False); o = np.argsort(r_["test_k"]); Y = r_["Y"][o].astype(float)
        tk = d["test_k"][row]
        own = np.full(96, np.nan); naft = np.zeros(96, int)
        for i in np.where(ist)[0]:
            own[tk[i]] = OUTC["armed2"][1][i]; naft[tk[i]] = OUTC["armed2"][3][i]
        rng = np.random.default_rng(0)
        for name, exp, m in (("GRU-z, GOES via FCN converter", "p2_zscore_full_in-goes_convnn_p2nn", "GRU-z"),
                             ("FCN-z, GOES via FCN converter", "p2_zscore_full_in-goes_convnn_p2nn", "FCN-z"),
                             ("GRU-z, GOES direct", "p2_zscore_full_in-goes_b_p1", "GRU-z"),
                             ("GRU-z, FISM2 input", "p2_zscore_full_gpu5", "GRU-z"), ("FCN-z, FISM2 input", "p2_zscore_full_gpu5", "FCN-z")):
            fs = sorted(glob.glob(f"{R}/{exp}/oofL_{m}_s*.npy"))
            if not fs:
                continue
            P = 10 ** np.mean([np.load(f).astype(float) for f in fs], axis=0)
            ok = np.isfinite(P).all((1, 2)); ape = 100 * np.abs(P - Y) / Y
            dcl = ok & np.isfinite(own)
            L = int(np.nanmedian(own))
            ev = ape.mean((1, 2))
            x, y = ev[ok & (naft > 0)], ev[ok & (naft == 0)]
            bt = np.array([rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(10000)]) if len(x) and len(y) else np.array([np.nan])
            rows.append(dict(model=name, n_events=int(ok.sum()), n_declared=int(dcl.sum()), latency_min=L, MAPE4_all_minutes=float(ape[ok].mean()),
                             MAPE4_from_declaration=float(ape[dcl][:, L:].mean()), n_later_flare=int(len(x)),
                             MAPE4_later_flare=float(x.mean()) if len(x) else np.nan, MAPE4_no_later_flare=float(y.mean()) if len(y) else np.nan,
                             d_later=float(x.mean() - y.mean()) if len(x) and len(y) else np.nan,
                             d_lo=float(np.nanpercentile(bt, 2.5)), d_hi=float(np.nanpercentile(bt, 97.5)),
                             MAPE_w0_later_flare=float(ape[ok & (naft > 0)][:, :, 0].mean()) if len(x) else np.nan,
                             MAPE_w0_no_later_flare=float(ape[ok & (naft == 0)][:, :, 0].mean()) if len(y) else np.nan))
        fi = R / "full_server/conformal_intervals_none.npz"          # 90 % interval coverage with / without a later flare
        if rows and fi.exists():
            I = np.load(fi, allow_pickle=False); yl = np.log10(Y)
            labs = {"GRU-z, GOES via FCN converter": "GRU-z [p2_zscore_full_in-goes_convnn_p2nn]", "FCN-z, GOES via FCN converter": "FCN-z [p2_zscore_full_in-goes_convnn_p2nn]",
                    "GRU-z, GOES direct": "GRU-z [p2_zscore_full_in-goes_b_p1]", "GRU-z, FISM2 input": "GRU-z [p2_zscore_full_gpu5]", "FCN-z, FISM2 input": "FCN-z [p2_zscore_full_gpu5]"}
            for r in rows:
                lab = labs[r["model"]]
                if f"{lab}|rebuilt96|lo" not in I.files:
                    continue
                lo_, hi_ = I[f"{lab}|rebuilt96|lo"], I[f"{lab}|rebuilt96|hi"]
                ok = np.isfinite(lo_).all((1, 2)); cov = ((yl >= lo_) & (yl <= hi_)).mean(1)
                r.update(cov_w0_later_flare=float(cov[ok & (naft > 0), 0].mean()), cov_w0_no_later_flare=float(cov[ok & (naft == 0), 0].mean()),
                         cov4_later_flare=float(cov[ok & (naft > 0)].mean()), cov4_no_later_flare=float(cov[ok & (naft == 0)].mean()))
        if rows:
            Sc = pd.DataFrame(rows); Sc.to_csv(OUT / "paper1_detector_scores.csv", index=False)
            print("\ntest flares (rebuilt96):"); print(Sc.round(2).to_string(index=False))
    print("\nwritten to", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
