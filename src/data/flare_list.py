"""Build the M/X flare catalogue (2003-01-01 .. 2023-03-30) and apply the event-selection
rules of Kim et al. (2026) Sect. 2.3:

  1. M1.0 or larger (GOES class M/X).
  2. Exclude if another M1+ flare peaked within the 60 min *before* the peak.
  3. Exclude if (more than one / any) M1+ flare peaked within 180 min *after* the peak.
     The paper says "more than one" but also "0/96 test events contain a secondary M1+
     flare within 180 min", so we implement the strict rule (any) as default and keep the
     lenient rule (>1) as an option; both counts are reported.

Sources (see docs/decisions.md): NGDC yearly GOES X-ray reports for 2003-2011, 2013, 2016,
2017-01..04-03; HEK FRM_NAME=SWPC records for 2012, 2014, 2015 (NGDC files incomplete) and
2017-04-04 .. 2023-03-30.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
EXT = str(ROOT / "external/flare_list")
END_DATE = pd.Timestamp("2023-03-30 23:59:59")
START_DATE = pd.Timestamp("2003-01-01")


def _to_ts(date8: str, hhmm: str, ref_start: pd.Timestamp | None = None) -> pd.Timestamp:
    t = pd.Timestamp(f"{date8[:4]}-{date8[4:6]}-{date8[6:8]} {hhmm[:2]}:{hhmm[2:]}")
    if ref_start is not None and t < ref_start:  # crossed midnight
        t += pd.Timedelta(days=1)
    return t


def load_ngdc() -> pd.DataFrame:
    df = pd.read_csv(f"{EXT}/ngdc_mx_2003_2017.csv", dtype=str).fillna("")
    rows = []
    for r in df.itertuples(index=False):
        start = _to_ts(r.date, r.start)
        peak = _to_ts(r.date, r.peak, start)
        end = _to_ts(r.date, r.end, peak)
        rows.append(dict(start=start, peak=peak, end=end, cls=r._4, location=r.location, ar=r.ar, src="ngdc"))
    return pd.DataFrame(rows)


def load_hek(fname: str) -> pd.DataFrame:
    df = pd.read_csv(f"{EXT}/{fname}", dtype=str).fillna("")
    rows = []
    for r in df.itertuples(index=False):
        s = r.start
        start = pd.Timestamp(f"{s[:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:{s[10:12]}")
        p = r.peak
        if len(p) == 12:
            peak = pd.Timestamp(f"{p[:4]}-{p[4:6]}-{p[6:8]} {p[8:10]}:{p[10:12]}")
        else:
            peak = _to_ts(s[:8], p, start)
        e = r.end
        if len(e) == 12:
            end = pd.Timestamp(f"{e[:4]}-{e[4:6]}-{e[6:8]} {e[8:10]}:{e[10:12]}")
        else:
            end = _to_ts(s[:8], e, peak)
        loc = ""
        if hasattr(r, "hgs_lon") and not (r.hgs_lon in ("", "0") and r.hgs_lat in ("", "0")):
            lon, lat = int(float(r.hgs_lon)), int(float(r.hgs_lat))
            loc = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'W' if lon >= 0 else 'E'}{abs(lon):02d}"
        rows.append(dict(start=start, peak=peak, end=end, cls=r._3 if not hasattr(r, "cls") else r.cls,
                         location=loc, ar=getattr(r, "ar", ""), src="hek"))
    return pd.DataFrame(rows)


def class_value(cls: str) -> float:
    """GOES class string -> peak flux W m^-2 (M1.0 = 1e-5, X1.0 = 1e-4)."""
    base = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}[cls[0]]
    return base * float(cls[1:])


def build_catalogue() -> pd.DataFrame:
    ngdc = load_ngdc()
    hek_mid = load_hek("hek_swpc_mx_2012_2014_2015.csv")
    hek_late = load_hek("hek_swpc_mx_2017_2023.csv")
    ngdc = ngdc[~ngdc.peak.dt.year.isin([2012, 2014, 2015])]
    ngdc = ngdc[ngdc.peak < pd.Timestamp("2017-04-04")]
    cat = pd.concat([ngdc, hek_mid, hek_late], ignore_index=True)
    cat = cat[(cat.peak >= START_DATE) & (cat.peak <= END_DATE)].copy()
    cat["peak_flux"] = cat.cls.map(class_value)
    cat = cat.sort_values("peak").reset_index(drop=True)
    # de-duplicate: same class and peak within 2 minutes
    keep = []
    last = None
    for r in cat.itertuples():
        if last is not None and r.cls == last.cls and abs((r.peak - last.peak).total_seconds()) <= 120:
            continue
        keep.append(r.Index)
        last = r
    cat = cat.loc[keep].reset_index(drop=True)
    cat["event_id"] = cat.peak.dt.strftime("%Y%m%d") + "_" + cat.cls
    return cat


def apply_selection(cat: pd.DataFrame, pre_min: int = 60, post_min: int = 180, strict: bool = True) -> pd.DataFrame:
    """Flag events with neighbouring M1+ peaks. strict=True: exclude if ANY other M1+ flare peaks
    within (-pre_min, 0) or (0, +post_min]; strict=False: exclude if >1 within post window."""
    peaks = cat.peak.values.astype("datetime64[m]").astype(np.int64)
    n_pre = np.zeros(len(cat), int)
    n_post = np.zeros(len(cat), int)
    for i, p in enumerate(peaks):
        d = peaks - p
        n_pre[i] = int(((d < 0) & (d >= -pre_min)).sum())
        n_post[i] = int(((d > 0) & (d <= post_min)).sum())
    cat = cat.copy()
    cat["n_pre60"] = n_pre
    cat["n_post180"] = n_post
    cat["selected"] = (n_pre == 0) & ((n_post == 0) if strict else (n_post <= 1))
    return cat


if __name__ == "__main__":
    cat = build_catalogue()
    sel = apply_selection(cat)
    print("catalogue M/X 2003-01..2023-03:", len(cat), "(X:", (cat.cls.str[0] == "X").sum(), ")")
    print("selected (strict):", sel.selected.sum(), " M:", (sel.selected & (sel.cls.str[0] == "M")).sum(),
          " X:", (sel.selected & (sel.cls.str[0] == "X")).sum())
    lenient = apply_selection(cat, strict=False)
    print("selected (lenient >1):", lenient.selected.sum())
    sel.to_csv(ROOT / "data/processed/flare_catalogue.csv", index=False)
    print("per-year selected:")
    print(sel[sel.selected].groupby(sel.peak.dt.year).size().to_dict())
