"""Build the full event dataset from the downloads, using the rules found by calibrate.py.

Output (default data/processed/events_latis.npz; never overwritten unless --force):
    X (N,60)        input, minutes peak-59..peak, source per calibration (FISM2 0.1-0.8 nm or GOES XRS-B;
                    GOES gaps of <= 5 interior minutes are log-linearly interpolated)
    Y (N,180,4)     FISM2 w0..w3, minutes peak..peak+179                       [W m^-2]
    X_fism (N,60), X_goes_b (N,60), X_goes_a (N,60)   all input variants (NaN if unavailable)
    bg_goes_b_24h (N,)  median GOES XRS-B over peak-24h..peak-60min (pre-flare background)
    f107_adj (N,)   adjusted F10.7 of the last measurement before the flare day (NaN without extras)
    row, key, event_id, cls, peak (ISO minute of the *aligned* peak), shift (min vs catalogue),
    goes_sat, selected (paper isolation rule), is_test (one of the 96 Zenodo test events), test_k
Events with missing/non-positive values in X or Y are dropped and listed in
data/download/logs/build_dropped.csv. All arrays are plain numeric/unicode (np.load(allow_pickle=False)).

    python scripts/download/build_events.py                     # after --stage all downloads
    python scripts/download/build_events.py --stage test --out data/processed/events_test96_check.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate as K  # noqa: E402
import common as C  # noqa: E402
import fetch_goes_xrs as G  # noqa: E402


def f107_lookup(root: Path):
    f = root / "extras" / "f107_daily.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, parse_dates=["date", "datetime"]).sort_values("datetime")
    d = d[(d.adj_sfu > 0) & (d.adj_sfu < 1000)]

    def get(peak):
        day0 = pd.Timestamp(peak).normalize()
        prev = d[d.datetime < day0]
        return float(prev.adj_sfu.iloc[-1]) if len(prev) else np.nan
    return get


def fill_short_gaps(v: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """Log-linear interpolation over interior runs of <= max_gap missing/non-positive minutes
    (GOES 1-min data have isolated flagged minutes; longer gaps and gaps at the ends stay NaN)."""
    v = np.array(v, dtype=float)
    bad = ~(np.isfinite(v) & (v > 0))
    if not bad.any() or bad.all():
        return v
    v[bad] = np.nan
    i, n = 0, len(v)
    while i < n:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j < n and bad[j]:
            j += 1
        if i > 0 and j < n and j - i <= max_gap:
            v[i:j] = 10 ** np.interp(np.arange(i, j), [i - 1, j], np.log10([v[i - 1], v[j]]))
        i = j
    return v


def reconstruct(e, fz, cal, root: Path, pre: int, f107=None):
    b0 = cal["bands"]["w0"]
    w0 = K.band_series(fz, "w0", b0["rule"], b0["agg"])
    n, t0 = int(fz["n"]), int(fz["t0"])
    so = int((pd.Timestamp(e.peak) - pd.Timestamp(e.start)).total_seconds() // 60)
    eo = int((pd.Timestamp(e.end) - pd.Timestamp(e.peak)).total_seconds() // 60)
    use_flags = cal.get("goes_flags", "flags") == "flags"
    # GOES over [peak-24h, end of FISM2 window)
    pk = int(fz.get("peak_cat", t0 + pre))
    g0 = pk - 1440
    gn = t0 + n - g0
    off = t0 - g0                                   # FISM2 index i <-> GOES index i + off
    ref = np.full(gn, np.nan)
    ref[off: off + n] = np.log10(w0)
    sat, gs = K.choose_goes(root, g0, gn, ref, slice(off + pre - 59, off + pre + 1), use_flags, K.sats_for(e.peak))
    goes_b = None if gs is None else gs["xrsb"][off: off + n]
    pi = K.peak_index(cal["peak_rule"], w0, pre, so, eo, goes_b)
    if pi is None:                                  # fall back to the FISM2 argmax rule
        pi = K.peak_index("argmax_w0_pm15_clip", w0, pre, so, eo, None)
    if pi is None or pi - 59 < 0 or pi + 180 > n:
        return None, "window"
    xs = slice(pi - 59, pi + 1)
    X_f = w0[xs]
    X_b = np.full(60, np.nan) if gs is None else fill_short_gaps(gs["xrsb"][off + pi - 59: off + pi + 1])
    X_a = np.full(60, np.nan) if gs is None else fill_short_gaps(gs["xrsa"][off + pi - 59: off + pi + 1])
    src = cal.get("input_source", "fism2_w0")
    if src == "fism2_w0":
        X = X_f
    elif src == "goes_xrsb":
        X = X_b
    elif src == "goes_xrsb_ops":
        X = X_b * (K.GOES_OPS_SCALE if (sat is not None and sat <= 15) else 1.0)
    else:
        raise ValueError(src)
    Y = np.stack([K.band_series(fz, b, cal["bands"][b]["rule"], cal["bands"][b]["agg"])[pi: pi + 180]
                  for b in K.BANDS], axis=1)
    if not (np.all(np.isfinite(X)) and np.all(X > 0)):
        return None, "X_missing"
    if not (np.all(np.isfinite(Y)) and np.all(Y > 0)):
        return None, "Y_missing"
    bg = np.nan
    if gs is not None:
        seg = gs["xrsb"][: 1440 - 60]
        bg = float(np.nanmedian(seg)) if np.isfinite(seg).sum() > 60 else np.nan
    rec = dict(X=X, Y=Y, X_fism=X_f, X_goes_b=X_b, X_goes_a=X_a, bg=bg,
               f107=f107(e.peak) if f107 else np.nan, sat=-1 if sat is None else sat,
               shift=pi - pre, peak=C.minute_to_iso(t0 + pi))
    return rec, "ok"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["test", "all"], default="all")
    ap.add_argument("--root", default=None)
    ap.add_argument("--calibration", default=None, help="default <root>/calibration.json")
    ap.add_argument("--out", default=str(C.ROOT / "data/processed/events_latis.npz"))
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    a = ap.parse_args(argv)
    root = C.dl_root(a.root)
    out = Path(a.out)
    if out.exists() and not a.force:
        print(f"[build] {out} exists - choose another --out (e.g. events_latis_v2.npz) or pass --force")
        return 1
    calf = Path(a.calibration) if a.calibration else root / "calibration.json"
    if not calf.exists():
        print(f"[build] {calf} not found - run calibrate.py first (python scripts/download/run_all.py --stage test)")
        return 1
    cal = json.loads(calf.read_text(encoding="utf-8"))
    if cal.get("status") != "PASS":
        print("[build] WARNING: calibration status is", cal.get("status"), "- see calibration_report.md")
    pre = int(cal.get("pre", 120))
    ev = C.load_events(a.stage)
    f107 = f107_lookup(root)
    keep, dropped = [], []
    for i, e in enumerate(ev.itertuples(index=False), 1):
        f = root / "fism2" / f"{e.key}.npz"
        if not f.exists():
            dropped.append(dict(key=e.key, reason="not_downloaded"))
            continue
        try:
            rec, why = reconstruct(e, dict(np.load(f, allow_pickle=False)), cal, root, pre, f107)
        except Exception as err:  # noqa: BLE001
            rec, why = None, f"error: {type(err).__name__}: {err}"
        if rec is None:
            dropped.append(dict(key=e.key, reason=why))
            continue
        rec["e"] = e
        keep.append(rec)
        if i % 100 == 0:
            print(f"  {i}/{len(ev)} processed, {len(keep)} kept", flush=True)
    if not keep:
        print("[build] nothing to write")
        return 1
    E = [r["e"] for r in keep]
    arr = dict(
        X=np.stack([r["X"] for r in keep]), Y=np.stack([r["Y"] for r in keep]),
        X_fism=np.stack([r["X_fism"] for r in keep]), X_goes_b=np.stack([r["X_goes_b"] for r in keep]),
        X_goes_a=np.stack([r["X_goes_a"] for r in keep]),
        bg_goes_b_24h=np.array([r["bg"] for r in keep]), f107_adj=np.array([r["f107"] for r in keep]),
        goes_sat=np.array([r["sat"] for r in keep], dtype=np.int64),
        shift=np.array([r["shift"] for r in keep], dtype=np.int64),
        peak=np.array([r["peak"] for r in keep]).astype(str),
        row=np.array([e.row for e in E], dtype=np.int64), key=np.array([e.key for e in E]).astype(str),
        event_id=np.array([e.event_id for e in E]).astype(str), cls=np.array([e.cls for e in E]).astype(str),
        selected=np.array([bool(e.selected) for e in E]),
        test_k=np.array([int(e.test_k) for e in E], dtype=np.int64),
        is_test=np.array([int(e.test_k) >= 0 for e in E]),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arr)
    pd.DataFrame(dropped).to_csv(root / "logs" / "build_dropped.csv", index=False)
    sel = arr["selected"] & ~arr["is_test"]
    print(f"[build] wrote {out}: {len(keep)} events ({int(arr['is_test'].sum())} test, {int(sel.sum())} "
          f"non-test passing the isolation rule); dropped {len(dropped)} (logs/build_dropped.csv)")
    if arr["is_test"].any():
        z = np.load(C.ROOT / "data/processed/zenodo_test_xy.npz")
        k = arr["test_k"][arr["is_test"]]
        dY = np.log10(arr["Y"][arr["is_test"]]) - np.log10(z["Y_true"][k])
        dX = np.log10(arr["X"][arr["is_test"]]) - np.log10(z["X_real"][k])
        print(f"[build] check vs Zenodo (test rows): median |dX| {np.median(np.abs(dX)):.4f} dex, "
              f"median |dY| per band {np.round(np.median(np.abs(dY), axis=(0, 1)), 4).tolist()} dex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
