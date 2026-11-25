"""Analysis A1 — the armed peak detector of the paper on CONTINUOUS GOES XRS-B 1-minute series (no training).

Design fixed before the run (docs/decisions.md, 2026-11-25 "continuous detector"):
  data      two full years of the primary satellite, 2014 (GOES-15, maximum of cycle 24) and 2022 (GOES-16, rise of
            cycle 25). Primary product: the science-quality 1-min files used everywhere in the paper; secondary: the
            archived operational 1-min product (GOES 8-15 SWPC scaling removed, i.e. true units for both).
  rule      the paper's armed detector (notebooks/peak_detector.detect_all): k = 2, rise factor r = 2, threshold
            1e-5 W m-2 on the raw series. Sensitivity: the catalogue-scale threshold (1.43e-5 for GOES 8-15).
  matching  a declaration is "matched" if its maximum (t_max) lies within +-5 min of the peak of a listed M/X flare
            (data/processed/flare_catalogue.csv, all M/X flares). Unmatched declarations are classified in this order:
            gap       a run of >= 10 missing minutes in the 60 min before t_max
            spike     at most 2 minutes within +-10 min of t_max reach half the declared maximum
            below_list_M1   declared maximum below M1 on the catalogue scale (GOES 8-15: < 1.43e-5 in true units)
            near_listed     t_max within [-60, +180] min of a listed M/X peak (another maximum of a listed flare)
            other     anything else (a real maximum absent from the list)
  report    data days (finite minutes / 1440), declarations per data day, share of listed M/X flares (with data at
            their peak) whose peak is declared, median latency (declaration - listed peak), unmatched declarations
            per data day by class.
  decision  primary product, threshold 1e-5: if the unmatched declarations are <= 1 per data day in BOTH years, the
            limitation "the detector was evaluated in 4-h windows only" becomes a result sentence and an SI table is
            added; otherwise the numbers are reported as they are and the detector statements are weakened.

    python scripts/continuous_detector.py                       (downloads missing daily files, then analyses)
    python scripts/continuous_detector.py --no-download --periods 2014-01-01:2014-01-04:15 --products science
Outputs: results/full_server/continuous_detector_summary<tag>.csv, continuous_detector_declarations<tag>.csv,
         continuous_detector_listed<tag>.csv (one row per listed flare in the periods).
"""
from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/download"))
sys.path.insert(0, str(ROOT / "notebooks"))
import common as C  # noqa: E402
import fetch_goes_xrs as G  # noqa: E402
import fetch_goes_xrs_ops as O  # noqa: E402
from peak_detector import detect_all  # noqa: E402

PERIODS = "2014-01-01:2015-01-01:15,2022-01-01:2023-01-01:16"
CLASSES = ("gap", "spike", "below_list_M1", "near_listed", "other")


def list_scale(sat: int) -> float:
    """Catalogue (SWPC) scale of GOES 8-15 = true flux x 0.7, so list-M1 = 1e-5 / 0.7 = 1.43e-5 in true units."""
    return 1.43 if sat <= 15 else 1.0


def parse_periods(s: str) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    out = []
    for item in s.split(","):
        a, b, sat = item.split(":")
        out.append((pd.Timestamp(a), pd.Timestamp(b), int(sat)))
    return out


def download(days, sat: int, product: str, root: Path, workers: int, min_interval: float) -> pd.Series:
    local = threading.local()

    def thr():
        if not hasattr(local, "t"):
            local.t = C.Throttle(min_interval)
        return local.t
    if product == "science":
        L = G.Listings(root, G.LEGACY_BASE, G.GOESR_BASE)
        job = lambda d: ";".join(G.fetch_day(d, root, L, 1, thr(), sats=[sat])) or "no file"  # noqa: E731
    else:
        L = O.Listings(root, O.LEGACY_OPS_BASE, O.GOESR_BASE)
        job = lambda d: O.fetch_one(sat, d, root, L, thr())  # noqa: E731
    if product == "ops" and sat <= 15:                   # one file per month: fetch the first day of each month only
        days = [d for d in days if d.day == 1] or days[:1]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda d: (d, _safe(job, d)), days))
    return pd.Series({d: r for d, r in res})


def _safe(job, d):
    try:
        return job(d)
    except Exception as err:  # noqa: BLE001  (one bad day must not stop the year)
        return f"error: {type(err).__name__}: {err}"[:200]


def load(product: str, root: Path, sat: int, t0: int, n: int) -> np.ndarray:
    s = (G.load_series if product == "science" else O.load_series_ops)(root, sat, t0, n, use_flags=True)
    return np.full(n, np.nan) if s is None else np.asarray(s["xrsb"], dtype=float)


def max_nan_run(x: np.ndarray) -> int:
    m = np.isnan(x).astype(int)
    if not m.any():
        return 0
    d = np.diff(np.r_[0, m, 0]); starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return int((ends - starts).max())


def classify(b: np.ndarray, tm: int, value: float, sat: int, near: bool) -> str:
    if max_nan_run(b[max(0, tm - 60):tm]) >= 10:
        return "gap"
    w = b[max(0, tm - 10):tm + 11]
    if np.sum(np.isfinite(w) & (w >= 0.5 * value)) <= 2:
        return "spike"
    if value < 1e-5 * list_scale(sat):
        return "below_list_M1"
    return "near_listed" if near else "other"


def analyse(b: np.ndarray, t0: int, sat: int, peaks: np.ndarray, names: np.ndarray, thr: float, k: int, r: float, tol: int):
    """-> (declarations DataFrame, listed-flare DataFrame, summary dict). Minutes are epoch minutes."""
    dec = pd.DataFrame(detect_all(b, k, thr, rise_factor=r))
    if dec.empty:
        dec = pd.DataFrame(columns=["t", "t_max", "value", "lo_pre", "episode", "kind"])
    P = np.sort(peaks); order = np.argsort(peaks); N = names[order]
    rows = []
    for q in dec.itertuples(index=False):
        tm_abs = t0 + int(q.t_max)
        j = np.searchsorted(P, tm_abs)
        cand = [i for i in (j - 1, j) if 0 <= i < len(P)]
        best = min(cand, key=lambda i: abs(P[i] - tm_abs)) if cand else None
        matched = best is not None and abs(P[best] - tm_abs) <= tol
        near = bool(np.any((P >= tm_abs - 180) & (P <= tm_abs + 60)))
        cls = "matched" if matched else classify(b, int(q.t_max), float(q.value), sat, near)
        rows.append(dict(t=C.minute_to_iso(t0 + int(q.t)), t_max=C.minute_to_iso(tm_abs), value=float(q.value), kind=q.kind,
                         episode=int(q.episode), matched_event=N[best] if matched else "", cls=cls))
    D = pd.DataFrame(rows, columns=["t", "t_max", "value", "kind", "episode", "matched_event", "cls"])
    # listed flares with data at their peak
    tmax_abs = t0 + dec.t_max.to_numpy(dtype=np.int64) if len(dec) else np.array([], dtype=np.int64)
    t_abs = t0 + dec.t.to_numpy(dtype=np.int64) if len(dec) else np.array([], dtype=np.int64)
    lrows = []
    for p, nm in zip(P, N):
        i = int(p - t0)
        if not (0 <= i < len(b)):
            continue
        has = bool(np.isfinite(b[max(0, i - 5):i + 6]).any())
        hit = np.where(np.abs(tmax_abs - p) <= tol)[0]
        lrows.append(dict(event=nm, peak=C.minute_to_iso(int(p)), data_at_peak=has, declared=bool(len(hit)),
                          latency=float(t_abs[hit[0]] - p) if len(hit) else np.nan))
    Lf = pd.DataFrame(lrows, columns=["event", "peak", "data_at_peak", "declared", "latency"])
    data_days = float(np.isfinite(b).sum()) / 1440.0
    S = dict(data_days=round(data_days, 2), coverage=round(float(np.isfinite(b).mean()), 4), n_decl=len(D),
             n_new=int((D.kind == "new").sum()), n_update=int((D.kind == "update").sum()),
             decl_per_day=len(D) / data_days if data_days else np.nan,
             listed=int(Lf.data_at_peak.sum()), listed_declared=int((Lf.data_at_peak & Lf.declared).sum()))
    S["listed_declared_share"] = S["listed_declared"] / S["listed"] if S["listed"] else np.nan
    S["median_latency_min"] = float(Lf.latency[Lf.data_at_peak & Lf.declared].median()) if S["listed_declared"] else np.nan
    un = D[D.cls != "matched"]
    S["n_unmatched"] = len(un); S["unmatched_per_day"] = len(un) / data_days if data_days else np.nan
    for c in CLASSES:
        S[f"{c}_per_day"] = int((un.cls == c).sum()) / data_days if data_days else np.nan
    S["outside_listed_windows_per_day"] = int(un.cls.isin(["gap", "spike", "below_list_M1", "other"]).sum()) / data_days if data_days else np.nan
    return D, Lf, S


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--periods", default=PERIODS, help="start:end:satellite,... (end exclusive)")
    ap.add_argument("--products", default="science,ops")
    ap.add_argument("--root", default=str(ROOT / "data/download"))
    ap.add_argument("--catalogue", default=str(ROOT / "data/processed/flare_catalogue.csv"))
    ap.add_argument("--out", default=str(ROOT / "results/full_server"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--min-interval", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--rise-factor", type=float, default=2.0)
    ap.add_argument("--threshold", type=float, default=1e-5)
    ap.add_argument("--tol", type=int, default=5)
    a = ap.parse_args(argv)
    root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cat = pd.read_csv(a.catalogue, parse_dates=["peak"])
    cat = cat[cat.cls.astype(str).str[0].isin(["M", "X"])]
    peaks_all = np.array([C.to_minute(p) for p in cat.peak], dtype=np.int64)
    names_all = cat.event_id.astype(str).to_numpy() if "event_id" in cat else cat.peak.astype(str).to_numpy()
    summaries, decl, listed = [], [], []
    for start, end, sat in parse_periods(a.periods):
        days = list(pd.date_range(start, end - pd.Timedelta(days=1), freq="D"))
        t0, n = C.to_minute(start), len(days) * 1440
        sel = (peaks_all >= t0) & (peaks_all < t0 + n)
        for product in a.products.split(","):
            if not a.no_download:
                st = download(days, sat, product, root, a.workers, a.min_interval)
                bad = st[st.astype(str).str.startswith(("error", "no file"))]
                print(f"[{start.year} g{sat} {product}] files: {len(st) - len(bad)} ok, {len(bad)} missing/error"
                      + (f" (first: {bad.index[0].date()} {bad.iloc[0]})" if len(bad) else ""), flush=True)
            b = load(product, root, sat, t0, n)
            for thr_name, thr in (("1e-5", a.threshold), ("list_scale", a.threshold * list_scale(sat))):
                if thr_name == "list_scale" and list_scale(sat) == 1.0:
                    continue
                D, Lf, S = analyse(b, t0, sat, peaks_all[sel], names_all[sel], thr, a.k, a.rise_factor, a.tol)
                key = dict(period=f"{start.date()}..{(end - pd.Timedelta(days=1)).date()}", sat=sat, product=product, threshold=thr_name)
                summaries.append({**key, **S}); decl.append(D.assign(**key)); listed.append(Lf.assign(**key))
                print(f"[{start.year} g{sat} {product} thr={thr_name}] data days {S['data_days']}, declarations {S['n_decl']} "
                      f"({S['decl_per_day']:.2f}/day), listed M/X declared {S['listed_declared']}/{S['listed']}, "
                      f"unmatched {S['unmatched_per_day']:.2f}/day, outside listed windows {S['outside_listed_windows_per_day']:.2f}/day", flush=True)
    Sm = pd.DataFrame(summaries)
    Sm.to_csv(out / f"continuous_detector_summary{a.tag}.csv", index=False)
    pd.concat(decl, ignore_index=True).to_csv(out / f"continuous_detector_declarations{a.tag}.csv", index=False)
    pd.concat(listed, ignore_index=True).to_csv(out / f"continuous_detector_listed{a.tag}.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(Sm.round(3).to_string(index=False))
    prim = Sm[(Sm["product"] == "science") & (Sm.threshold == "1e-5")]
    if len(prim):
        ok = bool((prim.unmatched_per_day <= 1.0).all() and (prim.data_days > 0).all())
        print("DECISION (pre-specified):", "<= 1 unmatched declaration per data day in every period -> result sentence + SI table"
              if ok else "> 1 unmatched declaration per data day in some period -> report the numbers, weaken the detector statements")
    print("written:", out)


if __name__ == "__main__":
    main()
