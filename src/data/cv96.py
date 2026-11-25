"""Zenodo test-set (96 events) loader, event metadata and the nested cross-validation protocol
("CV96") used because the 2003-2023 FISM2/GOES training data could not be obtained
(docs/decisions.md, 2026-11-03).

Data   : data/processed/zenodo_test_xy.npz (decoded by scripts/decode_zenodo_xy.py)
         X_real (96, 60)  GOES/FISM2 0.1-0.8 nm flux, minutes t_peak-59 .. t_peak  [W m^-2]
         Y_true (96,180,4) FISM2 w0..w3, minutes t_peak .. t_peak+179           [W m^-2]
         Values carry the transcription quantisation of 1e-4 dex (0.023 %).
Meta   : catalogue row (peak time, location) matched by event id; ambiguous ids are resolved by
         (1) the paper's isolation rule, then (2) the rise time seen in X_real (flagged).
CV96   : 10 outer folds, stratified by class (M/X) and interleaved in time order (fold = rank % 10,
         mimicking the paper's sequential 8:1:1 allocation); inside each outer-training set the
         same sequential rule puts every 9th event (per class) into validation, i.e. the inner
         split is 8:1 like the paper's train:val.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BANDS = ["w0", "w1", "w2", "w3"]
BAND_LABELS = ["0.1–0.8 nm", "9.0–9.9 nm", "13.0–13.9 nm", "30.0–30.9 nm"]
N_IN, N_OUT, N_B = 60, 180, 4


def class_value(cls: str) -> float:
    """GOES class string -> peak flux in W m^-2 (M1.0 -> 1e-5)."""
    base = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}[cls[0]]
    return base * float(cls[1:])


@lru_cache(maxsize=1)
def load_test96():
    z = np.load(ROOT / "data/processed/zenodo_test_xy.npz")
    return z["X_real"].astype(np.float64), z["Y_true"].astype(np.float64), [str(e) for e in z["event_names"]]


def _rise_minutes(x_log: np.ndarray) -> int:
    """Minutes before the peak during which flux exceeds baseline + 10 % of the (log) rise."""
    base = np.median(x_log[:15])
    thr = base + 0.1 * (x_log[-1] - base)
    above = x_log > thr
    k = 0
    for v in above[::-1]:
        if not v:
            break
        k += 1
    return k


def _hek_locations() -> pd.DataFrame:
    h = pd.read_csv(ROOT / "external/flare_list/hek_swpc_mx_2017_2023.csv", dtype=str)
    h["peak"] = pd.to_datetime(h["peak"], format="%Y%m%d%H%M")
    h["hgs_lon"] = pd.to_numeric(h["hgs_lon"], errors="coerce")
    h["hgs_lat"] = pd.to_numeric(h["hgs_lat"], errors="coerce")
    # HEK uses 0,0 for 'unknown'
    bad = (h["hgs_lon"] == 0) & (h["hgs_lat"] == 0)
    h.loc[bad, ["hgs_lon", "hgs_lat"]] = np.nan
    return h[["peak", "hgs_lon", "hgs_lat"]]


def _parse_ngdc_loc(loc) -> tuple[float, float]:
    if not isinstance(loc, str) or len(loc) < 6 or loc[0] not in "NS" or loc[3] not in "EW":
        return np.nan, np.nan
    try:
        lat = float(loc[1:3]) * (1 if loc[0] == "N" else -1)
        lon = float(loc[4:6]) * (1 if loc[3] == "W" else -1)
        return lon, lat
    except ValueError:
        return np.nan, np.nan


@lru_cache(maxsize=1)
def load_meta() -> pd.DataFrame:
    X, Y, ev = load_test96()
    cat = pd.read_csv(ROOT / "data/processed/flare_catalogue.csv", parse_dates=["start", "peak", "end"])
    hek = _hek_locations()
    rows = []
    for k, e in enumerate(ev):
        cand = cat[cat.event_id == e]
        xl = np.log10(X[k])
        rise = _rise_minutes(xl)
        flag = ""
        if len(cand) == 0:
            r = None
            flag = "no_catalogue_match"
        elif len(cand) == 1:
            r = cand.iloc[0]
        else:
            sel = cand[cand.selected]
            if len(sel) == 1:
                r = sel.iloc[0]
                flag = "ambiguous_resolved_by_isolation_rule"
            else:
                pool = sel if len(sel) else cand
                d = (pool.peak - pool.start).dt.total_seconds().values / 60.0
                r = pool.iloc[int(np.argmin(np.abs(d - rise)))]
                flag = "ambiguous_resolved_by_rise_time"
        cls = e.split("_")[1]
        lon = lat = np.nan
        if r is not None:
            if r.src == "ngdc":
                lon, lat = _parse_ngdc_loc(r.location)
            else:
                m = hek[(hek.peak - r.peak).abs() <= pd.Timedelta(minutes=2)]
                if len(m):
                    lon, lat = m.iloc[0].hgs_lon, m.iloc[0].hgs_lat
        rows.append(dict(
            k=k, event=e, cls=cls, cls_value=class_value(cls), is_x=cls.startswith("X"),
            peak_time=None if r is None else r.peak, start_time=None if r is None else r.start,
            end_time=None if r is None else r.end, cat_selected=None if r is None else bool(r.selected),
            n_pre60=None if r is None else int(r.n_pre60), n_post180=None if r is None else int(r.n_post180),
            hgs_lon=lon, hgs_lat=lat, match_flag=flag,
            x_peak=X[k, -1], rise_min_10pct=rise,
            rise_slope_10=(xl[-1] - xl[-11]) / 10.0,           # dex / min over the last 10 input minutes
            x_ratio_60=xl[-1] - xl[0],                          # dex rise over the window
        ))
    m = pd.DataFrame(rows)
    m["peak_time"] = pd.to_datetime(m["peak_time"])
    return m


def outer_folds(n_folds: int = 10, scheme: str = "interleaved") -> np.ndarray:
    """fold id per event (len 96).
    interleaved: stratified by class, fold = time-rank mod n_folds (mimics the paper's allocation)
    blocked    : contiguous blocks in time (stratified by class), i.e. neighbouring flares - often
                 from the same active region - stay in the same fold (leakage-sensitivity check)."""
    m = load_meta()
    fold = np.full(len(m), -1, dtype=int)
    for is_x in (False, True):
        sub = m[m.is_x == is_x].sort_values("peak_time")
        n = len(sub)
        for j, k in enumerate(sub.k.values):
            fold[k] = j % n_folds if scheme == "interleaved" else min(n_folds - 1, int(j * n_folds / n))
    return fold


def inner_split(train_idx: np.ndarray, period: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Sequential 8:1 train/val allocation within class (every `period`-th event -> val)."""
    m = load_meta()
    tr, va = [], []
    for is_x in (False, True):
        sub = m.iloc[train_idx]
        sub = sub[sub.is_x == is_x].sort_values("peak_time")
        for j, k in enumerate(sub.k.values):
            (va if j % period == period - 1 else tr).append(k)
    return np.array(sorted(tr), dtype=int), np.array(sorted(va), dtype=int)


def cv_splits(n_folds: int = 10, scheme: str = "interleaved"):
    """Yield (fold, train_idx, val_idx, test_idx)."""
    f = outer_folds(n_folds, scheme)
    allk = np.arange(len(f))
    for i in range(n_folds):
        te = allk[f == i]
        rest = allk[f != i]
        tr, va = inner_split(rest)
        yield i, tr, va, te
