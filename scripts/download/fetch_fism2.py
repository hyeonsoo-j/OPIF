"""Download FISM2 flare (fism_flare_hr, 60 s, 0.1-nm bins) windows around each flare peak from LASP LISIRD LaTiS.

Per event: minutes [peak - pre, peak + post) and four wavelength ranges that are wider than the paper
bands so that calibrate.py can decide which bins the authors summed:
    w0  0.0-0.9 nm   (paper 0.1-0.8)      w1  8.9-10.1 nm (paper 9.0-9.9)
    w2 12.9-14.1 nm  (paper 13.0-13.9)    w3 29.9-31.1 nm (paper 30.0-30.9)
(bin centres are expected at x.x5 nm, so the x.x0 query limits never coincide with a centre).

Output: <root>/fism2/<key>.npz with t0 (minute epoch of the first row), n (minutes) and, per band,
wl_<b> (bin centres, nm) and irr_<b> (n x n_bins, W m^-2 nm^-1, NaN where missing).
Already-downloaded events are skipped (resume); --force re-downloads.

    python scripts/download/fetch_fism2.py --stage test --limit 3      # smoke test
    python scripts/download/fetch_fism2.py --stage test                # 96 test events (calibration)
    python scripts/download/fetch_fism2.py --stage all                 # every M/X flare 2003-2023
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

LATIS_URL = os.environ.get("FISM_LATIS_URL", "https://lasp.colorado.edu/lisird/latis/dap/fism_flare_hr.csv")
QUERY_RANGES = {"w0": (0.0, 0.9), "w1": (8.9, 10.1), "w2": (12.9, 14.1), "w3": (29.9, 31.1)}


def latis_url(t0_min: int, t1_min: int, a: float, b: float, base: str = LATIS_URL) -> str:
    return (f"{base}?time>={C.minute_to_iso(t0_min)}&time<{C.minute_to_iso(t1_min)}"
            f"&wavelength>{a:g}&wavelength<{b:g}")


def parse_latis_csv(text: str, t0_min: int, n_min: int):
    """LaTiS CSV (header + rows 'time,wavelength,irradiance[,...]') -> (wl (k,), irr (n_min, k)).
    time is a Julian date (as served for fism_flare_hr) or an ISO-8601 string."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return np.zeros(0), np.full((n_min, 0), np.nan)
    t_raw, wl, irr = [], [], []
    for ln in lines[1:]:
        p = ln.split(",")
        if len(p) < 3:
            continue
        t_raw.append(p[0].strip())
        wl.append(float(p[1]))
        irr.append(float(p[2]) if p[2].strip() not in ("", "NaN", "nan") else np.nan)
    try:
        tv = np.array([float(t) for t in t_raw])
        if not (2.3e6 < np.nanmedian(tv) < 2.6e6):
            raise ValueError(f"time column is numeric but not a Julian date (median {np.nanmedian(tv)})")
        mins = C.jd_to_minute(tv)
    except ValueError as err:
        if "Julian" in str(err):
            raise
        import pandas as pd
        mins = np.array([C.to_minute(pd.Timestamp(t)) for t in t_raw], dtype=np.int64)
    wl = np.round(np.array(wl), 4)
    irr = np.array(irr, dtype=float)
    wls = np.unique(wl)
    out = np.full((n_min, len(wls)), np.nan)
    mi = mins - t0_min
    ok = (mi >= 0) & (mi < n_min)
    out[mi[ok], np.searchsorted(wls, wl[ok])] = irr[ok]
    return wls, out


def fetch_event(ev, root: Path, pre: int, post: int, throttle: C.Throttle, force: bool = False, base: str = LATIS_URL):
    out = root / "fism2" / f"{ev.key}.npz"
    if out.exists() and not force:
        return "skip"
    p = C.to_minute(ev.peak)
    t0, t1 = p - pre, p + post
    arrays = {"t0": np.int64(t0), "n": np.int64(t1 - t0), "peak_cat": np.int64(p)}
    for b, (a, z) in QUERY_RANGES.items():
        throttle.wait()
        txt = C.fetch_text(latis_url(t0, t1, a, z, base))
        if txt is None:
            raise RuntimeError("HTTP 404 from LaTiS")
        wl, irr = parse_latis_csv(txt, t0, t1 - t0)
        arrays[f"wl_{b}"] = wl
        arrays[f"irr_{b}"] = irr.astype(np.float64)
    C.save_npz_atomic(out, **arrays)
    n_ok = int(np.isfinite(arrays["irr_w1"]).all(axis=1).sum()) if arrays["irr_w1"].size else 0
    return f"ok ({n_ok}/{t1 - t0} complete minutes in w1)"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["test", "all"], default="test")
    ap.add_argument("--limit", type=int, default=None, help="only the first N events (smoke test)")
    ap.add_argument("--only-selected", action="store_true", help="stage all: only events passing the isolation rule")
    ap.add_argument("--pre", type=int, default=120, help="minutes before the catalogue peak (default 120)")
    ap.add_argument("--post", type=int, default=200, help="minutes after the catalogue peak (default 200)")
    ap.add_argument("--workers", type=int, default=2, help="parallel events (be polite: 2-4)")
    ap.add_argument("--min-interval", type=float, default=0.5, help="seconds between requests per worker")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--base-url", default=LATIS_URL)
    a = ap.parse_args(argv)
    root = C.dl_root(a.root)
    ev = C.load_events(a.stage, a.limit, a.only_selected)
    print(f"[fism2] {len(ev)} events -> {root / 'fism2'}", flush=True)
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        thr = {}

        def job(e):
            import threading
            t = thr.setdefault(threading.get_ident(), C.Throttle(a.min_interval))
            return fetch_event(e, root, a.pre, a.post, t, a.force, a.base_url)

        futs = {ex.submit(job, e): e for e in ev.itertuples(index=False)}
        for i, f in enumerate(as_completed(futs), 1):
            e = futs[f]
            try:
                msg = f.result()
                n_ok += 1
            except Exception as err:  # noqa: BLE001 - log and continue; rerun resumes
                msg = f"ERROR {type(err).__name__}: {err}"
                n_err += 1
                C.log_error(root, "fism2", key=e.key, peak=str(e.peak), error=f"{type(err).__name__}: {err}")
            if i <= 5 or i % 25 == 0 or msg.startswith("ERROR"):
                print(f"  [{i}/{len(ev)}] {e.key}: {msg}", flush=True)
    print(f"[fism2] done: {n_ok} ok/skipped, {n_err} errors (rerun the same command to retry failures)")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
