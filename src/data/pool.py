"""Event pool for the "full" protocol (the paper's setting): train/val on the downloaded FISM2 events,
test on the 96 Zenodo test events. Selected with `python run.py --exp <cfg> --protocol full`.

Pool layout (indices used by src/experiments/cv.py):
    0..95      the 96 Zenodo test events, X/Y = the authors' exact arrays (data/processed/zenodo_test_xy.npz)
    96..191    the same 96 test events as rebuilt from the downloads ("latis" test set; same construction
               as the training events). Every model is scored on BOTH test sets: the Zenodo set compares
               with the paper, the latis set is the like-for-like test (the public FISM2 0.1-0.8 nm band
               differs from the authors' by a time-varying factor ~1.4-1.5, docs/decisions.md 2026-11-05).
               Test events missing from the downloads keep a Zenodo copy here but are not scored.
    192..      every other downloaded event that passes the isolation rule
               (data/processed/events_latis.npz from scripts/download/build_events.py),
               split within class (M / X) in time order 8 train : 1 val (paper 8:1:1 with the test
               slot fixed by Zenodo -> ~772 / ~96).
Environment switches (set by run.py):
    FISM_PROTOCOL            cv96 (default) | full
    FISM_EVENTS              path of the built npz (default data/processed/events_latis.npz)
    FISM_INPUT               default | fism | goes_b | goes_conv | goes_convnn
                             input variant for ALL rows (test rows then take the same variant from the npz
                             instead of the Zenodo X_real). goes_conv*: GOES XRS-A/B windows converted to the
                             FISM2 0.1-0.8 nm band by src/models/converter.py, fitted on the training events
                             (paper-1 P2); the converter's validation RMSE is printed and stored in
                             FISM_CONV_RMSE_DEX.
    FISM_EXCLUDE_NEAR_TEST_H drop train/val events whose peak lies within H hours of any test peak
                             (leakage check; default 0 = keep all, as in the paper)
    FISM_EXCLUDE_RANDOM_N    drop N randomly chosen train/val candidates instead (control for the
                             data-size effect of the exclusion above; FISM_EXCLUDE_SEED picks which)
    FISM_EXCLUDE_SAME_AR     1 = drop train/val flares from the same NOAA active region as any test flare
                             (AR known for 1034/1606 catalogue rows, 60/96 test events); flares without
                             an AR number fall back to the time rule with FISM_EXCLUDE_AR_FALLBACK_H hours
                             (default 48) around test flares that also lack an AR
    FISM_ISSUE_DELAY         k >= 0 (default 0): the forecast is issued k minutes after the peak, so every input window
                             covers minutes -59+k .. k (paper-1 P7). Minutes 1..k of the FISM2 inputs are the
                             0.1-0.8 nm target minutes; those of the GOES inputs come from FISM_GOES_POST
                             (data/processed/events_goes_post.npz, scripts/build_goes_post.py). The converters are
                             fitted on the shifted windows. Targets stay minutes 0..179 from the peak: read the
                             scores for lead >= k.
    FISM_SPLIT_LIKE_GOES     1 = assign train/val only among the flares that also have a complete GOES XRS-B window
                             (control experiment 1, 2026-11-25): a FISM2-input model then has exactly the train/val
                             assignment of the GOES-input models (982 / 122 instead of 988 / 122). No effect on GOES inputs.
    FISM_PREDICT_EXTRA       1 = also append the flares that FAIL the isolation rule (never used for training,
                             validation or testing) after the train/val rows; extra_index() gives their pool
                             indices and the experiment driver saves the predictions as oofN_* (paper-1 P8).
    FISM_TRAIN_EXTRA_FOLD    0 | 1 (needs FISM_PREDICT_EXTRA=1; paper-1 P9): the flares outside the isolation rule are
                             split into two folds by time cluster (peaks less than 24 h apart share a cluster, fold =
                             cluster index mod 2) and the flares of the given fold are ADDED to the training rows.
                             Validation rows and converters do not change. extra_fold() gives the fold of every such
                             flare; score the flares of the other fold only.
    FISM_PREDICT_OPS         1 (needs a GOES input and FISM_ISSUE_DELAY=0; paper-1 P11): append TWIN rows of the rebuilt test
                             flares and of the flares outside the isolation rule (if FISM_PREDICT_EXTRA=1) whose input is the
                             OPERATIONAL GOES window (FISM_GOES_OPS, data/processed/events_goes_ops.npz from
                             scripts/build_goes_ops.py), passed through the same fitted converter. Never used for fitting;
                             ops_index() gives their pool indices, dataset rows, twins and variants ("asis" time stamps and,
                             if a satellite has a non-zero lag, "aligned"); the driver saves the predictions as oofO_*.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cv96 import (_hek_locations, _parse_ngdc_loc, _rise_minutes, class_value, load_meta,
                           load_test96)

ROOT = Path(__file__).resolve().parents[2]
_EXTRA = {"idx": np.array([], dtype=int), "rows": np.array([], dtype=int), "fold": np.array([], dtype=int)}      # set by load_full()
_OPS = {"idx": np.array([], dtype=int), "rows": np.array([], dtype=int), "twin": np.array([], dtype=int),
        "kind": np.array([], dtype=str), "variant": np.array([], dtype=str)}                                       # set by load_full()


def issue_delay() -> int:
    return int(os.environ.get("FISM_ISSUE_DELAY", "0") or 0)


def extra_index():
    """(pool indices, dataset rows) of the flares outside the isolation rule, if FISM_PREDICT_EXTRA=1 (else empty)."""
    load_full()
    return _EXTRA["idx"], _EXTRA["rows"]


def extra_fold():
    """Fold (0 / 1, by time cluster) of the flares outside the isolation rule, aligned with extra_index()."""
    load_full()
    return _EXTRA["fold"]


def ops_index() -> dict:
    """Operational-input twin rows (FISM_PREDICT_OPS=1), else empty arrays: idx (pool indices), rows (dataset rows), twin (pool
    index of the same flare with the science-quality input), kind ("test" / "outside"), variant ("asis" / "aligned")."""
    load_full()
    return dict(_OPS)


def _goes_ops(eid: np.ndarray) -> dict:
    """{variant: (XRS-B (N, 60), XRS-A (N, 60))} operational input windows of the dataset rows (NaN where unavailable)."""
    path = Path(os.environ.get("FISM_GOES_OPS", "") or ROOT / "data/processed/events_goes_ops.npz")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - FISM_PREDICT_OPS=1 needs the operational GOES windows: "
                                f"python scripts/download/fetch_goes_xrs_ops.py, then python scripts/build_goes_ops.py")
    g = np.load(path, allow_pickle=False)
    if not np.array_equal(g["event_id"].astype(str), np.asarray(eid).astype(str)):
        raise ValueError(f"{path} was built from another dataset (event ids differ) - rebuild it with scripts/build_goes_ops.py")
    var = {"asis": (g["xrsb_ops"].astype(np.float64), g["xrsa_ops"].astype(np.float64))}
    if bool(g["has_aligned"]):
        var["aligned"] = (g["xrsb_ops_al"].astype(np.float64), g["xrsa_ops_al"].astype(np.float64))
    return var


def train_extra_fold() -> str:
    v = os.environ.get("FISM_TRAIN_EXTRA_FOLD", "")
    if v not in ("", "0", "1"):
        raise ValueError(f"FISM_TRAIN_EXTRA_FOLD must be 0 or 1, got {v!r}")
    return v


def _shift(win: np.ndarray, post: np.ndarray, k: int) -> np.ndarray:
    """Input window ending k minutes after the peak: drop the first k minutes, append minutes 1..k."""
    return win if k == 0 else np.concatenate([win[:, k:], post[:, :k]], axis=1)


def _goes_post(eid: np.ndarray, k: int):
    """(XRS-B, XRS-A) minutes 1..k after the peak for the dataset rows `eid` (NaN where unavailable)."""
    path = Path(os.environ.get("FISM_GOES_POST", ROOT / "data/processed/events_goes_post.npz"))
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - GOES inputs with FISM_ISSUE_DELAY > 0 need the GOES minutes after "
                                f"the peak: python scripts/build_goes_post.py (needs peak_detector_streams.npz)")
    g = np.load(path, allow_pickle=False)
    n = len(eid)
    if g["xrsb_post"].shape[0] != n or g["xrsb_post"].shape[1] < k:
        raise ValueError(f"{path}: {g['xrsb_post'].shape} does not fit {n} events and a delay of {k} min")
    if "event_id" in g.files and not np.array_equal(g["event_id"].astype(str), np.asarray(eid).astype(str)):
        raise ValueError(f"{path} was built from another dataset (event ids differ) - rebuild it with scripts/build_goes_post.py")
    return g["xrsb_post"].astype(np.float64), g["xrsa_post"].astype(np.float64)


def protocol() -> str:
    return os.environ.get("FISM_PROTOCOL", "cv96")


def _location(row, hek) -> tuple[float, float]:
    if row is None:
        return np.nan, np.nan
    if row.src == "ngdc":
        return _parse_ngdc_loc(row.location)
    m = hek[(hek.peak - row.peak).abs() <= pd.Timedelta(minutes=2)]
    return (m.iloc[0].hgs_lon, m.iloc[0].hgs_lat) if len(m) else (np.nan, np.nan)


def _meta_rows(X, cls, peak, rows, cat, hek, k0: int) -> pd.DataFrame:
    out = []
    for i in range(len(X)):
        xl = np.log10(X[i])
        r = cat.loc[int(rows[i])] if rows is not None and int(rows[i]) in cat.index else None
        lon, lat = _location(r, hek)
        c = str(cls[i])
        out.append(dict(
            k=k0 + i, event=None if r is None else r.event_id, cls=c, cls_value=class_value(c), is_x=c.startswith("X"),
            peak_time=pd.Timestamp(peak[i]), start_time=None if r is None else r.start, end_time=None if r is None else r.end,
            cat_selected=None if r is None else bool(r.selected), n_pre60=None if r is None else int(r.n_pre60),
            n_post180=None if r is None else int(r.n_post180), hgs_lon=lon, hgs_lat=lat, match_flag="pool",
            x_peak=X[i, -1], rise_min_10pct=_rise_minutes(xl), rise_slope_10=(xl[-1] - xl[-11]) / 10.0,
            x_ratio_60=xl[-1] - xl[0]))
    return pd.DataFrame(out)


@lru_cache(maxsize=1)
def load_full():
    """-> X (N,60), Y (N,180,4), names (list), meta (DataFrame, len N), tr, va, te, te_latis (index arrays)."""
    path = Path(os.environ.get("FISM_EVENTS", ROOT / "data/processed/events_latis.npz"))
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - build it with scripts/download/run_all.py --stage all")
    d = np.load(path, allow_pickle=False)
    X96, Y96, ev96 = load_test96()
    meta96 = load_meta().copy()
    inp = os.environ.get("FISM_INPUT", "default")
    conv = inp.startswith("goes_conv")
    xkey = {"default": "X", "fism": "X_fism", "goes_b": "X_goes_b", "goes_conv": "X_goes_b", "goes_convnn": "X_goes_b"}[inp]
    Xd = d[xkey].astype(np.float64)
    kd = issue_delay()
    y_post = d["Y"][:, 1:, 0].astype(np.float64)      # FISM2 0.1-0.8 nm minutes 1.. after the peak (Y minute 0 = peak)
    if kd:                                   # issue k minutes after the peak: windows cover minutes -59+k..k
        if not 0 < kd <= 30:
            raise ValueError(f"FISM_ISSUE_DELAY={kd}: expected 0..30")
        if xkey == "X_goes_b":
            gb_post, ga_post = _goes_post(d["event_id"], kd)
            Xd = _shift(Xd, gb_post, kd)
        else:
            Xd = _shift(Xd, y_post, kd)
    if conv:                                 # convert GOES -> FISM2 band; fitted below on the training rows
        Xd_goes_b, Xd_goes_a, Xd_fism = d["X_goes_b"].astype(np.float64), d["X_goes_a"].astype(np.float64), d["X_fism"].astype(np.float64)
        if kd:
            Xd_goes_b, Xd_goes_a, Xd_fism = _shift(Xd_goes_b, gb_post, kd), _shift(Xd_goes_a, ga_post, kd), _shift(Xd_fism, y_post, kd)
    eid = d["event_id"].astype(str)
    is_test = d["is_test"] if "is_test" in d.files else np.isin(eid, ev96)
    peak = pd.to_datetime(d["peak"].astype(str))
    y_ok = np.all(np.isfinite(d["Y"]) & (d["Y"] > 0), axis=(1, 2))
    valid = np.all(np.isfinite(Xd) & (Xd > 0), axis=1) & y_ok
    # events usable when the forecast is issued at the peak. The train/val assignment below is made on these, for every
    # delay, and events without the minutes after the peak are dropped afterwards: a delayed model is then fitted on a
    # subset of the training events of the peak-issued model (and validated on a subset of its validation events).
    x0 = d[xkey].astype(np.float64)
    valid0 = np.all(np.isfinite(x0) & (x0 > 0), axis=1) & y_ok

    if conv:
        # train/val rows as they will be split below (same rule), fit the converter on train, score on val
        valid_c = np.all(np.isfinite(Xd_goes_b) & (Xd_goes_b > 0), axis=1) & np.all(np.isfinite(Xd_fism) & (Xd_fism > 0), axis=1)
        f0 = d["X_fism"].astype(np.float64)
        valid_c0 = valid0 & np.all(np.isfinite(f0) & (f0 > 0), axis=1)        # = valid_c & y_ok when kd == 0
        rest_c = np.where(d["selected"] & ~is_test & ~np.isin(eid, ev96) & valid_c0)[0]
        rest_c = rest_c[np.argsort(peak.values[rest_c], kind="stable")]
        tr_c, va_c = [], []
        for c in ("M", "X"):
            oc = [i for i in rest_c if str(d["cls"][i]).startswith(c)]
            for j, i in enumerate(oc):
                (va_c if j % 9 == 8 else tr_c).append(i)
        tr_c, va_c = [i for i in tr_c if valid_c[i]], [i for i in va_c if valid_c[i]]
        from src.models.converter import make_converter
        cv_ = make_converter(inp)
        if inp == "goes_convnn":
            cv_.fit(Xd_goes_b[tr_c], Xd_goes_a[tr_c], Xd_fism[tr_c], Xd_goes_b[va_c], Xd_goes_a[va_c], Xd_fism[va_c])
        else:
            cv_.fit(Xd_goes_b[tr_c], Xd_goes_a[tr_c], Xd_fism[tr_c])
        ok_b = np.all(np.isfinite(Xd_goes_b) & (Xd_goes_b > 0), axis=1)
        Xd = np.full_like(Xd_goes_b, np.nan)
        Xd[ok_b] = 10 ** cv_.predict_log(Xd_goes_b[ok_b], Xd_goes_a[ok_b])
        r = np.log10(Xd[va_c]) - np.log10(Xd_fism[va_c])
        rmse_conv = float(np.sqrt(np.mean(r ** 2)))
        r0 = np.log10(Xd_goes_b[va_c]) - np.log10(Xd_fism[va_c])
        print(f"[pool] converter {inp}: validation RMSE log10 {rmse_conv:.4f} dex (raw GOES-B {float(np.sqrt(np.mean(r0 ** 2))):.4f})", flush=True)
        os.environ["FISM_CONV_RMSE_DEX"] = f"{rmse_conv:.5f}"
        valid = np.all(np.isfinite(Xd) & (Xd > 0), axis=1) & y_ok

    X96 = X96.copy()
    if kd:                                   # the authors' input window, shifted with the authors' 0.1-0.8 nm target minutes
        X96 = _shift(X96, Y96[:, 1:, 0], kd)
    if inp != "default":                     # same input variant for the test rows (from the npz)
        tk = d["test_k"]
        for i in np.where(is_test & valid)[0]:
            X96[int(tk[i])] = Xd[i]
        missing = sorted(set(range(96)) - set(int(t) for t in tk[is_test & valid]))
        if missing:                          # keep all 96 test events: those keep the Zenodo input
            print(f"[pool] WARNING: input {inp!r} unavailable for {len(missing)} test events "
                  f"(k={missing[:10]}...) - they keep the Zenodo X_real input", flush=True)
        os.environ["FISM_INPUT_MISSING_TEST"] = str(len(missing))

    # rebuilt test events (same construction as training rows), in Zenodo order
    tk_all = d["test_k"] if "test_k" in d.files else np.full(len(eid), -1)
    XL, YL = X96.copy(), Y96.copy()
    rowL = np.full(96, -1)
    for i in np.where(is_test & valid)[0]:
        k = int(tk_all[i])
        if 0 <= k < 96:
            XL[k], YL[k], rowL[k] = Xd[i], d["Y"][i], i
    latis_ok = rowL >= 0
    os.environ["FISM_LATIS_TEST_N"] = str(int(latis_ok.sum()))

    rest = np.where(d["selected"] & ~is_test & ~np.isin(eid, ev96) & (valid0 if kd else valid))[0]
    if os.environ.get("FISM_SPLIT_LIKE_GOES", "0") == "1":     # control 1: the train/val assignment of the GOES-input pools
        xg = d["X_goes_b"].astype(np.float64)
        rest = rest[np.all(np.isfinite(xg[rest]) & (xg[rest] > 0), axis=1)]
    h = float(os.environ.get("FISM_EXCLUDE_NEAR_TEST_H", "0") or 0)
    if h > 0:
        tp = meta96.peak_time.values.astype("datetime64[m]").astype(np.int64)
        pp = peak.values.astype("datetime64[m]").astype(np.int64)
        near = np.array([np.min(np.abs(tp - pp[i])) < h * 60 for i in rest], dtype=bool)
        rest = rest[~near]
    if os.environ.get("FISM_EXCLUDE_SAME_AR", "0") == "1":
        cat_ar = pd.read_csv(ROOT / "data/processed/flare_catalogue.csv", parse_dates=["peak"])
        test_ar, test_noar_peaks = set(), []
        for r in meta96.itertuples():
            hit = cat_ar[(cat_ar.event_id == r.event) & (cat_ar.peak == r.peak_time)]
            a = hit.ar.iloc[0] if len(hit) else np.nan
            if np.isfinite(a):
                test_ar.add(int(a))
            else:
                test_noar_peaks.append(np.datetime64(r.peak_time, "m").astype(np.int64))
        hf = float(os.environ.get("FISM_EXCLUDE_AR_FALLBACK_H", "48") or 48)
        rows_d = d["row"] if "row" in d.files else None
        pp = peak.values.astype("datetime64[m]").astype(np.int64)
        drop = np.zeros(len(rest), dtype=bool)
        for j, i in enumerate(rest):
            a = cat_ar.ar.iloc[int(rows_d[i])] if rows_d is not None else np.nan
            if np.isfinite(a):
                drop[j] = int(a) in test_ar
            elif test_noar_peaks:
                drop[j] = np.min(np.abs(np.array(test_noar_peaks) - pp[i])) < hf * 60
        rest = rest[~drop]
    n_rnd = int(os.environ.get("FISM_EXCLUDE_RANDOM_N", "0") or 0)
    if n_rnd > 0:                                   # size-matched control: drop N random candidates
        if n_rnd >= len(rest):
            raise ValueError(f"FISM_EXCLUDE_RANDOM_N={n_rnd} but only {len(rest)} train/val candidates")
        rng = np.random.default_rng(int(os.environ.get("FISM_EXCLUDE_SEED", "0") or 0))
        rest = np.sort(rng.choice(rest, size=len(rest) - n_rnd, replace=False))
    rest = rest[np.argsort(peak.values[rest], kind="stable")]
    tr, va = [], []
    for c in ("M", "X"):
        oc = [i for i in rest if str(d["cls"][i]).startswith(c)]
        for j, i in enumerate(oc):
            (va if j % 9 == 8 else tr).append(i)
    if kd:                                          # drop the events without the minutes after the peak (split unchanged)
        rest = rest[valid[rest]]
        tr, va = [i for i in tr if valid[i]], [i for i in va if valid[i]]
    R0 = 192
    pos = {i: R0 + j for j, i in enumerate(rest)}
    ex = np.array([], dtype=int)
    if os.environ.get("FISM_PREDICT_EXTRA", "0") == "1":      # flares that fail the isolation rule: predicted, never fitted
        ex = np.where(~d["selected"].astype(bool) & ~is_test & ~np.isin(eid, ev96) & valid)[0]
        ex = ex[np.argsort(peak.values[ex], kind="stable")]
    ex_fold = np.array([], dtype=int)
    if len(ex):                                              # time clusters (24-h gap) of the flares outside the rule, alternating folds
        pm = np.asarray(peak.values[ex], dtype="datetime64[m]").astype(np.int64)
        ex_fold = np.concatenate([[0], np.cumsum(np.diff(pm) >= 1440)]) % 2
    if train_extra_fold() != "" and not len(ex):
        raise ValueError("FISM_TRAIN_EXTRA_FOLD needs FISM_PREDICT_EXTRA=1 and flares outside the isolation rule")
    ops = dict(rows=[], twin=[], kind=[], variant=[], X=[])
    if os.environ.get("FISM_PREDICT_OPS", "0") == "1":       # operational GOES input for the same flares: predicted, never fitted
        if xkey != "X_goes_b" or kd:
            raise ValueError("FISM_PREDICT_OPS=1 needs a GOES input (goes_b, goes_conv, goes_convnn) and FISM_ISSUE_DELAY=0")
        cand = [(int(rowL[k]), 96 + int(k), "test") for k in np.where(latis_ok)[0]] + [(int(i), R0 + len(rest) + j, "outside") for j, i in enumerate(ex)]
        for vname, (ob, oa) in _goes_ops(eid).items():
            okb = np.all(np.isfinite(ob) & (ob > 0), axis=1)
            Xo = np.full_like(ob, np.nan)
            if okb.any():
                Xo[okb] = 10 ** cv_.predict_log(ob[okb], oa[okb]) if conv else ob[okb]
            for i, tw, kind in cand:
                if okb[i] and np.all(np.isfinite(Xo[i]) & (Xo[i] > 0)):
                    ops["rows"].append(i); ops["twin"].append(tw); ops["kind"].append(kind); ops["variant"].append(vname); ops["X"].append(Xo[i])
    ops_rows = np.array(ops["rows"], dtype=int)
    cat = pd.read_csv(ROOT / "data/processed/flare_catalogue.csv", parse_dates=["start", "peak", "end"])
    hek = _hek_locations()
    rows_d = d["row"] if "row" in d.files else None
    meta_L = meta96.copy()                                  # same events; input-derived features from XL
    mL = _meta_rows(XL, np.array([e.split("_")[1] for e in ev96]), meta96.peak_time.values, None, cat, hek, 96)
    for c in ("x_peak", "rise_min_10pct", "rise_slope_10", "x_ratio_60"):
        meta_L[c] = mL[c].values
    meta_L["k"] = np.arange(96, R0)
    meta_r = _meta_rows(Xd[rest], d["cls"][rest], peak.values[rest], None if rows_d is None else rows_d[rest],
                        cat, hek, R0)
    parts = [meta96, meta_L, meta_r]
    if len(ex):
        parts.append(_meta_rows(Xd[ex], d["cls"][ex], peak.values[ex], None if rows_d is None else rows_d[ex], cat, hek, R0 + len(rest)))
    Xops = np.stack(ops["X"]) if len(ops_rows) else np.zeros((0, Xd.shape[1]))
    if len(ops_rows):
        parts.append(_meta_rows(Xops, d["cls"][ops_rows], peak.values[ops_rows], None if rows_d is None else rows_d[ops_rows], cat, hek,
                                R0 + len(rest) + len(ex)))
    meta = pd.concat(parts, ignore_index=True)
    for extra in ("f107_adj", "bg_goes_b_24h"):          # optional context features
        if extra in d.files:
            col = np.full(len(meta), np.nan)
            for i in np.where(is_test)[0]:
                k = int(tk_all[i])
                if 0 <= k < 96:
                    col[k] = col[96 + k] = d[extra][i]
            col[R0:R0 + len(rest)] = d[extra][rest]
            col[R0 + len(rest):R0 + len(rest) + len(ex)] = d[extra][ex]
            col[R0 + len(rest) + len(ex):] = d[extra][ops_rows]
            meta[extra] = col
    meta["peak_time"] = pd.to_datetime(meta["peak_time"])
    X = np.concatenate([X96, XL, Xd[rest], Xd[ex], Xops]).astype(np.float64)
    Y = np.concatenate([Y96, YL, d["Y"][rest], d["Y"][ex], d["Y"][ops_rows]]).astype(np.float64)
    names = list(ev96) + [f"{e}|latis" for e in ev96] + [str(eid[i]) for i in rest] + [f"{eid[i]}|extra" for i in ex] \
        + [f"{eid[i]}|ops-{v}" for i, v in zip(ops_rows, ops["variant"])]
    _EXTRA["idx"], _EXTRA["rows"], _EXTRA["fold"] = R0 + len(rest) + np.arange(len(ex)), ex, ex_fold
    _OPS.update(idx=R0 + len(rest) + len(ex) + np.arange(len(ops_rows)), rows=ops_rows, twin=np.array(ops["twin"], dtype=int),
                kind=np.array(ops["kind"], dtype=str), variant=np.array(ops["variant"], dtype=str))
    tr = np.array(sorted(pos[i] for i in tr), dtype=int)
    if train_extra_fold() != "":                             # P9: the flares outside the rule of one fold join the training rows
        tr = np.concatenate([tr, _EXTRA["idx"][ex_fold == int(train_extra_fold())]])
    va = np.array(sorted(pos[i] for i in va), dtype=int)
    te = np.arange(96)
    te_latis = 96 + np.where(latis_ok)[0]
    return X, Y, names, meta, tr, va, te, te_latis


def describe() -> str:
    X, Y, names, meta, tr, va, te, te_latis = load_full()
    bits = [f"input={os.environ.get('FISM_INPUT', 'default')}"]
    if os.environ.get("FISM_INPUT_MISSING_TEST", "0") != "0":
        bits.append(f"test events without this input: {os.environ['FISM_INPUT_MISSING_TEST']}")
    if issue_delay():
        bits.append(f"issued {issue_delay()} min after the peak")
    if len(_EXTRA["idx"]):
        if train_extra_fold() != "":
            n_in = int((_EXTRA["fold"] == int(train_extra_fold())).sum())
            bits.append(f"+ {len(_EXTRA['idx'])} flares outside the isolation rule ({n_in} of fold {train_extra_fold()} in the training rows, {len(_EXTRA['idx']) - n_in} predicted only)")
        else:
            bits.append(f"+ {len(_EXTRA['idx'])} flares outside the isolation rule (predicted only)")
    if len(_OPS["idx"]):
        vs = sorted(set(_OPS["variant"].tolist()))
        bits.append(f"+ {len(_OPS['idx'])} twin rows with the operational GOES input ("
                    + "; ".join(f"{v}: {int(((_OPS['variant'] == v) & (_OPS['kind'] == 'test')).sum())} test, "
                                f"{int(((_OPS['variant'] == v) & (_OPS['kind'] == 'outside')).sum())} outside the rule" for v in vs) + "; predicted only)")
    bits += [f"exclude_near_test_h={os.environ.get('FISM_EXCLUDE_NEAR_TEST_H', '0')}",
             f"exclude_random_n={os.environ.get('FISM_EXCLUDE_RANDOM_N', '0')}",
             f"exclude_same_ar={os.environ.get('FISM_EXCLUDE_SAME_AR', '0')}",
             f"split_like_goes={os.environ.get('FISM_SPLIT_LIKE_GOES', '0')}"]
    return (f"full protocol: {len(tr)} train / {len(va)} val / {len(te)} test events (Zenodo arrays) "
            f"+ {len(te_latis)} rebuilt test events ({', '.join(bits)})")
