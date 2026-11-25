"""Operational (archived real-time) GOES XRS input windows for the flares of the built dataset (paper-1 P11).

For every flare with a science-quality GOES stream (results/full_server/peak_detector_streams.npz) the operational 1-min
XRS-A/B of the same satellite is read on the same minutes from the files of scripts/download/fetch_goes_xrs_ops.py
(GOES 8-15 divided by the SWPC factors 0.7 / 0.85; fill values and flagged minutes missing), and the 60 input minutes up to
the dataset peak are cut exactly as in the dataset (interior gaps of <= 5 min filled log-linearly).

Time stamps: "as is" windows use the archived time stamps. One integer lag per satellite (-max_lag .. +max_lag minutes, the
lag with the smallest median |dlog10| between the operational and the science-quality input minutes; ties -> nearest 0) is
estimated from the isolated training/validation flares only (selected and not a test flare) and gives the "aligned" windows:
aligned minute t = operational minute t + lag.

    python scripts/build_goes_ops.py --events data/processed/events_latis.npz \
        --streams results/full_server/peak_detector_streams.npz --root data/download \
        --out data/processed/events_goes_ops.npz --report results/full_server/goes_ops_vs_science.csv

Output npz (dataset order for the windows, stream order for the streams; NaN where unavailable):
    xrsb_ops, xrsa_ops (N, 60)          input windows, time stamps as archived
    xrsb_ops_al, xrsa_ops_al (N, 60)    the same with the lag of the satellite applied
    has_aligned                         True if any satellite has a non-zero lag
    row, sat, lag_row (n_stream,)       dataset row, satellite and lag of every stream
    wide_b, wide_a (n_stream, pre + post + 2 * max_lag)   operational streams from minute -pre - max_lag
    lag_sat, lag_min                    satellites and their lags;  pre, post, max_lag, event_id
Reports: <report> (per satellite, per series and all: availability and agreement with the science-quality input) and
<report stem>_events.csv (per flare).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/download"))

SERIES = {8: "GOES 8-12", 9: "GOES 8-12", 10: "GOES 8-12", 11: "GOES 8-12", 12: "GOES 8-12", 13: "GOES 13-15", 14: "GOES 13-15",
          15: "GOES 13-15", 16: "GOES 16-18", 17: "GOES 16-18", 18: "GOES 16-18", 19: "GOES 16-18"}


def best_lag(cost: dict[int, float]) -> int:
    """Lag with the smallest cost; ties (and no data) -> the lag nearest 0."""
    fin = {k: v for k, v in cost.items() if np.isfinite(v)}
    if not fin:
        return 0
    lo = min(fin.values())
    return min((k for k, v in fin.items() if v <= lo + 1e-12), key=lambda k: (abs(k), k))


def agreement(o: np.ndarray, s: np.ndarray) -> dict:
    """log10(operational / science) over the samples where both are valid."""
    ok = np.isfinite(o) & (o > 0) & np.isfinite(s) & (s > 0)
    if not ok.any():
        return dict(n_minutes=0, median_dlog=np.nan, median_abs_dlog=np.nan, rms_dlog=np.nan, share_abs_gt_0p05=np.nan)
    dl = np.log10(o[ok] / s[ok])
    return dict(n_minutes=int(ok.sum()), median_dlog=float(np.median(dl)), median_abs_dlog=float(np.median(np.abs(dl))),
                rms_dlog=float(np.sqrt(np.mean(dl ** 2))), share_abs_gt_0p05=float((np.abs(dl) > 0.05).mean()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default=str(ROOT / "data/processed/events_latis.npz"))
    ap.add_argument("--streams", default=str(ROOT / "results/full_server/peak_detector_streams.npz"))
    ap.add_argument("--root", default=None, help="download root (default: as the download scripts)")
    ap.add_argument("--out", default=str(ROOT / "data/processed/events_goes_ops.npz"))
    ap.add_argument("--report", default=str(ROOT / "results/full_server/goes_ops_vs_science.csv"))
    ap.add_argument("--max-gap", type=int, default=5)
    ap.add_argument("--max-lag", type=int, default=2)
    a = ap.parse_args(argv)
    import common as C
    import fetch_goes_xrs_ops as O
    from build_events import fill_short_gaps
    root = C.dl_root(a.root)
    d = np.load(a.events, allow_pickle=False)
    st = np.load(a.streams, allow_pickle=False)
    row, pre, post, L = st["row"].astype(int), int(st["pre"]), int(st["post"]), a.max_lag
    n, ns = len(d["event_id"]), len(row)
    if row.max() >= n or not (st["event_id"] == d["event_id"][row]).all():
        print("ERROR: the streams were exported from another dataset (event ids differ)")
        return 1
    sat = d["goes_sat"][row].astype(int)
    if "sat" in st.files and not np.array_equal(st["sat"].astype(int), sat):
        print("ERROR: the satellites of the streams differ from the dataset")
        return 1
    W = pre + post + 2 * L
    wide_b, wide_a = np.full((ns, W), np.nan), np.full((ns, W), np.nan)
    shown, n_nofile = set(), 0
    n_unflagged = np.zeros(ns, int)                   # valid XRS-B input minutes when the quality flags are ignored
    for j, i in enumerate(row):
        p = C.to_minute(pd.Timestamp(str(d["peak"][i])))
        s = O.load_series_ops(root, int(sat[j]), p - pre - L, W, use_flags=True)
        if s is None:
            n_nofile += 1
            continue
        wide_b[j], wide_a[j] = s["xrsb"], s["xrsa"]
        raw = O.load_series_ops(root, int(sat[j]), p - 59, 60, use_flags=False)
        n_unflagged[j] = 0 if raw is None else int(np.isfinite(raw["xrsb"]).sum())
        if int(sat[j]) not in shown:                 # one line per satellite on what its files hold
            shown.add(int(sat[j]))
            for f in O.find_local_all(root, int(sat[j]), pd.Timestamp(p * 60, unit="s").normalize()):
                print(f"[ops] GOES-{int(sat[j])}: {O.describe_file(f, int(sat[j]))}", flush=True)
    for name, why in list(O.READ_ERRORS.items())[:12]:
        print(f"[ops] could not read {name}: {why}", flush=True)
    if len(O.READ_ERRORS) > 12:
        print(f"[ops] ... and {len(O.READ_ERRORS) - 12} more files that could not be read", flush=True)
    sci_b, sci_a = st["xrsb"].astype(float), st["xrsa"].astype(float)
    win = slice(pre - 59, pre + 1)                    # the 60 input minutes of the stream (index pre = the peak)
    cut = lambda w, lag: w[:, L + lag:L + lag + pre + post]          # noqa: E731  stream with minute t = operational minute t + lag
    # ---- lag per satellite, from the isolated training/validation flares only
    fit = d["selected"][row].astype(bool) & ~d["is_test"][row].astype(bool)
    lag_of, cost_of = {}, {}
    for s_ in sorted(set(sat.tolist())):
        m = fit & (sat == s_)
        cost = {}
        for lag in range(-L, L + 1):
            o, sc = cut(wide_b[m], lag)[:, win], sci_b[m][:, win]
            ok = np.isfinite(o) & (o > 0) & np.isfinite(sc) & (sc > 0)
            cost[lag] = float(np.median(np.abs(np.log10(o[ok] / sc[ok])))) if ok.sum() >= 60 else np.nan
        lag_of[s_], cost_of[s_] = best_lag(cost), cost
    lag_row = np.array([lag_of[s_] for s_ in sat], dtype=np.int64)
    has_al = bool((lag_row != 0).any())
    # ---- input windows
    out = {k: np.full((n, 60), np.nan) for k in ("xrsb_ops", "xrsa_ops", "xrsb_ops_al", "xrsa_ops_al")}
    asis_b, asis_a = cut(wide_b, 0), cut(wide_a, 0)
    for j, i in enumerate(row):
        for key, w, lag in (("xrsb_ops", wide_b, 0), ("xrsa_ops", wide_a, 0), ("xrsb_ops_al", wide_b, lag_row[j]), ("xrsa_ops_al", wide_a, lag_row[j])):
            v = fill_short_gaps(w[j, L + lag + pre - 59:L + lag + pre + 1], a.max_gap)
            out[key][i] = np.where(np.isfinite(v) & (v > 0), v, np.nan)
    # the same cut applied to the science stream must give the input windows of the dataset
    worst, n_cmp = 0.0, 0
    for j, i in enumerate(row):
        v, w = fill_short_gaps(sci_b[j, win], a.max_gap), d["X_goes_b"][i].astype(float)
        both = np.isfinite(v) & (v > 0) & np.isfinite(w) & (w > 0)
        if both.any():
            worst, n_cmp = max(worst, float(np.abs(np.log10(v[both] / w[both])).max())), n_cmp + int(both.sum())
    print(f"[ops] science stream cut the same way vs dataset input windows: {n_cmp} samples, max |dlog10| = {worst:.2e}")
    if worst > 1e-6:
        print("ERROR: the science streams do not reproduce the input windows of the dataset")
        return 1
    # ---- reports
    Xb = d["X_goes_b"][row].astype(float)
    ok_sci = np.all(np.isfinite(Xb) & (Xb > 0), axis=1)
    comp_b = np.all(np.isfinite(out["xrsb_ops"][row]), axis=1)
    comp_a = np.all(np.isfinite(out["xrsa_ops"][row]), axis=1)
    comp_b_al = np.all(np.isfinite(out["xrsb_ops_al"][row]), axis=1)
    ev = []
    for j, i in enumerate(row):
        g = agreement(out["xrsb_ops"][i], Xb[j])
        ga = agreement(out["xrsb_ops_al"][i], Xb[j])
        ev.append(dict(dataset_row=int(i), event=str(d["event_id"][i]), cls=str(d["cls"][i]), sat=int(sat[j]), series=SERIES.get(int(sat[j]), "other"),
                       is_test=bool(d["is_test"][i]), selected=bool(d["selected"][i]), lag_min=int(lag_row[j]), science_window_complete=bool(ok_sci[j]),
                       ops_minutes_raw=int(np.isfinite(asis_b[j, win]).sum()), ops_minutes_ignoring_flags=int(n_unflagged[j]), ops_window_complete=bool(comp_b[j]), ops_xrsa_complete=bool(comp_a[j]),
                       ops_window_complete_aligned=bool(comp_b_al[j]),
                       dlog_peak_minute=float(np.log10(out["xrsb_ops"][i][-1] / Xb[j][-1])) if np.isfinite(out["xrsb_ops"][i][-1]) and Xb[j][-1] > 0 else np.nan,
                       **{k: g[k] for k in ("median_dlog", "rms_dlog", "share_abs_gt_0p05")},
                       median_dlog_aligned=ga["median_dlog"], rms_dlog_aligned=ga["rms_dlog"]))
    EV = pd.DataFrame(ev)
    rep_path = Path(a.report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    EV.to_csv(rep_path.with_name(rep_path.stem + "_events.csv"), index=False)
    rows = []
    groups = [("all", np.ones(ns, bool))] + [(ser, np.array([SERIES.get(int(s_), "other") == ser for s_ in sat])) for ser in sorted(set(SERIES.get(int(s_), "other") for s_ in sat))] \
        + [(f"GOES-{s_}", sat == s_) for s_ in sorted(set(sat.tolist()))]
    for name, m in groups:
        base = dict(group=name, n_flares=int(m.sum()), n_science_complete=int((m & ok_sci).sum()), n_ops_any_minute=int((m & np.isfinite(asis_b[:, win]).any(1)).sum()),
                    n_ops_complete=int((m & comp_b).sum()), n_ops_complete_and_science=int((m & comp_b & ok_sci).sum()), n_ops_xrsa_complete=int((m & comp_a).sum()),
                    n_test_complete=int((m & comp_b & ok_sci & d["is_test"][row]).sum()), n_outside_complete=int((m & comp_b & ok_sci & ~d["selected"][row] & ~d["is_test"][row]).sum()))
        g = agreement(out["xrsb_ops"][row][m & ok_sci], Xb[m & ok_sci])
        ga = agreement(out["xrsb_ops_al"][row][m & ok_sci], Xb[m & ok_sci])
        gx = agreement(out["xrsa_ops"][row][m], d["X_goes_a"][row][m].astype(float))
        nv, nu = int(np.isfinite(asis_b[m][:, win]).sum()), int(n_unflagged[m].sum())
        rec = dict(**base, **g, share_minutes_flagged=float(1 - nv / nu) if nu else np.nan, lag_min=lag_of.get(int(name.split("-")[1]), np.nan) if name.startswith("GOES-") else np.nan,
                   median_abs_dlog_aligned=ga["median_abs_dlog"], rms_dlog_aligned=ga["rms_dlog"], xrsa_median_dlog=gx["median_dlog"], xrsa_rms_dlog=gx["rms_dlog"])
        if name.startswith("GOES-"):
            rec.update({f"cost_lag{lag:+d}": cost_of[int(name.split("-")[1])][lag] for lag in range(-L, L + 1)})
        rows.append(rec)
    REP = pd.DataFrame(rows)
    REP.to_csv(rep_path, index=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sats = np.array(sorted(lag_of), dtype=np.int64)
    np.savez_compressed(a.out, **out, has_aligned=np.bool_(has_al), row=row.astype(np.int64), sat=sat.astype(np.int64), lag_row=lag_row,
                        wide_b=wide_b, wide_a=wide_a, lag_sat=sats, lag_min=np.array([lag_of[s_] for s_ in sats], dtype=np.int64),
                        pre=np.int64(pre), post=np.int64(post), max_lag=np.int64(L), event_id=d["event_id"].astype(str))
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(REP[["group", "n_flares", "n_ops_any_minute", "n_ops_complete", "n_test_complete", "n_outside_complete", "median_dlog", "rms_dlog", "share_abs_gt_0p05", "share_minutes_flagged", "lag_min"]]
          .round(4).to_string(index=False))
    allr = REP.iloc[0]
    print(f"[ops] written: {a.out}  ({ns} flares with a science stream; {n_nofile} without an operational file; complete operational windows: "
          f"{int(comp_b.sum())}; lags {dict(zip(sats.tolist(), [lag_of[s_] for s_ in sats]))})")
    warn = []
    if comp_b.sum() < 0.5 * ok_sci.sum():
        warn.append(f"only {int(comp_b.sum())} of {int(ok_sci.sum())} flares have a complete operational window")
    for name in ("GOES 8-12", "GOES 13-15", "GOES 16-18"):
        r = REP[REP.group == name]
        if len(r) and r.iloc[0].n_ops_complete_and_science > 0 and abs(r.iloc[0].median_dlog) > 0.05:
            warn.append(f"{name}: median log10(operational / science) = {r.iloc[0].median_dlog:+.3f} dex (scaling or reading problem?)")
        if len(r) and r.iloc[0].n_flares > 0 and r.iloc[0].n_ops_any_minute == 0:
            warn.append(f"{name}: no operational minute could be read")
        if len(r) and np.isfinite(r.iloc[0].share_minutes_flagged) and r.iloc[0].share_minutes_flagged > 0.2:
            warn.append(f"{name}: {100 * r.iloc[0].share_minutes_flagged:.0f} % of the input minutes carry a non-zero quality flag and were dropped")
    for w in warn:
        print(f"[ops] WARNING: {w}")
    print(f"[ops] overall: median dlog10 {allr.median_dlog:+.4f}, rms {allr.rms_dlog:.4f} dex, |d| > 0.05 dex in {100 * allr.share_abs_gt_0p05:.1f} % of the input minutes")
    return 0 if comp_b.sum() > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
