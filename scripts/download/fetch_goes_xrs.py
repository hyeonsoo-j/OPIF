"""Download GOES XRS 1-minute science-quality files (XRS-A 0.05-0.4 nm, XRS-B 0.1-0.8 nm) for the days
covering each flare (default: peak - 24 h ... peak + 200 min, so a 24-h pre-flare background is available).

Sources (NOAA NCEI / NGDC, as used by sunpy's XRSClient):
    GOES 8-15 : https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/
                goesNN/xrsf-l2-avg1m_science/YYYY/MM/sci_xrsf-l2-avg1m_gNN_dYYYYMMDD_v*.nc
    GOES 16-19: https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/
                goesNN/l2/data/xrsf-l2-avg1m_science/YYYY/MM/sci_xrsf-l2-avg1m_gNN_dYYYYMMDD_v*.nc
The science files of GOES 8-15 do NOT carry the SWPC 0.7 scaling of the operational data
(calibrate.py tests both conventions against the Zenodo inputs).

For every needed day the satellites are tried in an era-dependent preference order and up to
--max-sats files that exist are kept (calibrate/build pick the best one per event).
Files are stored unchanged under <root>/goes/gNN/YYYY/. Monthly directory listings are cached in
<root>/goes/listings/.

    python scripts/download/fetch_goes_xrs.py --stage test --limit 3
    python scripts/download/fetch_goes_xrs.py --stage all
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

LEGACY_BASE = os.environ.get("FISM_GOES_LEGACY_BASE", "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs")
GOESR_BASE = os.environ.get("FISM_GOESR_BASE", "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes")


def candidate_sats(day: pd.Timestamp) -> list[int]:
    """Preference order per era (primary XRS first). Missing satellites are simply skipped."""
    y = day.year
    if y <= 2006:
        return [12, 10, 11, 8]
    if y <= 2009:
        return [10, 14, 12, 11]
    if y <= 2016:
        return [15, 14, 13]
    if y <= 2019:
        return [16, 15, 14, 17] if day >= pd.Timestamp("2017-02-07") else [15, 14, 13]
    return [16, 18, 17]


def dir_url(sat: int, day: pd.Timestamp, legacy_base=LEGACY_BASE, goesr_base=GOESR_BASE) -> str:
    if sat <= 15:
        return f"{legacy_base}/goes{sat:02d}/xrsf-l2-avg1m_science/{day.year}/{day.month:02d}/"
    return f"{goesr_base}/goes{sat}/l2/data/xrsf-l2-avg1m_science/{day.year}/{day.month:02d}/"


def file_regex(sat: int, day: pd.Timestamp) -> re.Pattern:
    s = f"{sat:02d}" if sat <= 15 else f"{sat}"
    return re.compile(rf"sci_xrsf-l2-avg1m_g{s}_d{day.strftime('%Y%m%d')}_v([0-9][0-9\-]*)\.nc")


def _version_key(v: str):
    return tuple(int(x) for x in v.split("-") if x.isdigit())


class Listings:
    """Monthly directory listings (cached on disk; a 404 is cached as an empty listing)."""

    def __init__(self, root: Path, legacy_base: str, goesr_base: str):
        self.dir = root / "goes" / "listings"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.mem: dict[str, list[str]] = {}
        self.key_locks: dict[str, threading.Lock] = {}
        self.lb, self.gb = legacy_base, goesr_base

    def get(self, sat: int, day: pd.Timestamp) -> tuple[str, list[str]]:
        url = dir_url(sat, day, self.lb, self.gb)
        k = f"g{sat:02d}_{day.year}{day.month:02d}"
        with self.lock:
            if k in self.mem:
                return url, self.mem[k]
        with self.lock:
            klock = self.key_locks.setdefault(k, threading.Lock())
        with klock:                                     # one request per month even with many threads
            if k in self.mem:
                return url, self.mem[k]
            f = self.dir / f"{k}.json"
            if f.exists():
                names = json.loads(f.read_text(encoding="utf-8"))
            else:
                html = C.fetch_text(url, timeout=60)
                names = [] if html is None else sorted(set(re.findall(r'href="([^"/?]+\.nc)"', html)))
                C.atomic_write_bytes(f, json.dumps(names).encode())
            self.mem[k] = names
        return url, names


def days_for_event(peak: pd.Timestamp, pre_hours: float, post_min: int) -> list[pd.Timestamp]:
    t0 = (peak - pd.Timedelta(hours=pre_hours)).normalize()
    t1 = (peak + pd.Timedelta(minutes=post_min)).normalize()
    return list(pd.date_range(t0, t1, freq="D"))


def local_path(root: Path, sat: int, day: pd.Timestamp, fname: str) -> Path:
    return root / "goes" / f"g{sat:02d}" / f"{day.year}" / fname


def find_local(root: Path, sat: int, day: pd.Timestamp) -> Path | None:
    d = root / "goes" / f"g{sat:02d}" / f"{day.year}"
    if not d.exists():
        return None
    rx = file_regex(sat, day)
    hits = sorted([p for p in d.glob("*.nc") if rx.fullmatch(p.name)], key=lambda p: _version_key(rx.fullmatch(p.name).group(1)))
    return hits[-1] if hits else None


def fetch_day(day: pd.Timestamp, root: Path, listings: Listings, max_sats: int, throttle: C.Throttle,
              sats: list[int] | None = None) -> list[str]:
    got = []
    for sat in (sats or candidate_sats(day)):
        if len(got) >= max_sats:
            break
        lp = find_local(root, sat, day)
        if lp is not None:
            got.append(f"g{sat:02d}:cached")
            continue
        throttle.wait()
        url, names = listings.get(sat, day)
        rx = file_regex(sat, day)
        cands = [(rx.fullmatch(n).group(1), n) for n in names if rx.fullmatch(n)]
        if not cands:
            continue
        ver, name = max(cands, key=lambda c: _version_key(c[0]))
        throttle.wait()
        data = C.fetch_bytes(url + name, timeout=180)
        if data is None:
            continue
        C.atomic_write_bytes(local_path(root, sat, day, name), data)
        got.append(f"g{sat:02d}:v{ver}")
    return got


# ------------------------------------------------------------------------------------------
# reading (used by calibrate.py / build_events.py)
# ------------------------------------------------------------------------------------------
def _decode_time(var) -> np.ndarray:
    """netCDF time variable -> integer minute epoch (floor to the minute start)."""
    units = getattr(var, "units", "")
    m = re.match(r"\s*(\w+)\s+since\s+(.+?)\s*$", units)
    if not m:
        raise ValueError(f"unknown time units: {units!r}")
    unit, ref = m.group(1).lower(), m.group(2).replace("UTC", "").replace("Z", "").strip()
    ref = pd.Timestamp(ref)
    mult = {"seconds": 1.0, "second": 1.0, "s": 1.0, "milliseconds": 1e-3, "millisecond": 1e-3, "ms": 1e-3,
            "minutes": 60.0, "minute": 60.0, "hours": 3600.0, "days": 86400.0}[unit]
    v = np.asarray(var[:], dtype=float) * mult
    sec = (ref - pd.Timestamp("1970-01-01")).total_seconds() + v
    return np.floor(sec / 60.0 + 1e-6).astype(np.int64)


def _pick(vs, *names):
    for n in names:
        if n in vs:
            return n
    return None


def read_xrs_nc(path: Path) -> pd.DataFrame:
    """-> DataFrame indexed by minute epoch with xrsa, xrsb (W m^-2, NaN for fill/non-positive) and
    xrsa_flag, xrsb_flag (0 = good; -1 if the file has no flag variable)."""
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        vs = ds.variables
        tn = _pick(vs, "time", "time_tag")
        an = _pick(vs, "xrsa_flux", "a_flux", "xrsa_flux_observed")
        bn = _pick(vs, "xrsb_flux", "b_flux", "xrsb_flux_observed")
        if tn is None or bn is None:
            raise ValueError(f"{path.name}: no time/xrsb flux variable (have {list(vs)[:20]})")
        out = {"minute": _decode_time(vs[tn])}
        for lab, n in (("xrsa", an), ("xrsb", bn)):
            if n is None:
                out[lab] = np.full(len(out["minute"]), np.nan)
            else:
                v = np.ma.filled(np.ma.asarray(vs[n][:]).astype(float), np.nan)
                v[~(v > 0)] = np.nan
                out[lab] = v
            fn = _pick(vs, f"{lab}_flag", f"{lab}_flags", f"{lab[-1]}_flags", f"{lab[-1]}_flag")
            out[f"{lab}_flag"] = (np.ma.filled(np.ma.asarray(vs[fn][:]), 0).astype(np.int64)
                                  if fn is not None else np.full(len(out["minute"]), -1, dtype=np.int64))
    df = pd.DataFrame(out).drop_duplicates("minute").set_index("minute").sort_index()
    return df


@lru_cache(maxsize=128)
def _read_cached(path: str) -> pd.DataFrame:
    return read_xrs_nc(Path(path))


def load_series(root: Path, sat: int, t0_min: int, n: int, use_flags: bool = True) -> dict[str, np.ndarray] | None:
    """XRS-A/B on the minute grid [t0, t0+n) from the local files of one satellite; None if no file."""
    days = pd.date_range(pd.Timestamp(t0_min * 60, unit="s").normalize(),
                         pd.Timestamp((t0_min + n - 1) * 60, unit="s").normalize(), freq="D")
    parts = []
    for d in days:
        p = find_local(root, sat, d)
        if p is not None:
            parts.append(_read_cached(str(p)))
    if not parts:
        return None
    df = pd.concat(parts)
    df = df[~df.index.duplicated()]
    grid = np.arange(t0_min, t0_min + n)
    df = df.reindex(grid)
    res = {}
    for lab in ("xrsa", "xrsb"):
        v = df[lab].to_numpy(dtype=float)
        if use_flags:
            fl = df[f"{lab}_flag"].to_numpy()
            v = np.where((fl > 0) & np.isfinite(fl), np.nan, v)
        res[lab] = v
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["test", "all"], default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-selected", action="store_true")
    ap.add_argument("--pre-hours", type=float, default=24.0, help="hours before the peak to cover (default 24)")
    ap.add_argument("--post", type=int, default=200)
    ap.add_argument("--max-sats", type=int, default=2, help="keep up to N satellites per day (default 2)")
    ap.add_argument("--sats", default=None, help="comma list to force satellites, e.g. 15,16")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--min-interval", type=float, default=0.3)
    ap.add_argument("--root", default=None)
    ap.add_argument("--legacy-base", default=LEGACY_BASE)
    ap.add_argument("--goesr-base", default=GOESR_BASE)
    a = ap.parse_args(argv)
    root = C.dl_root(a.root)
    ev = C.load_events(a.stage, a.limit, a.only_selected)
    days = sorted({d for p in ev.peak for d in days_for_event(pd.Timestamp(p), a.pre_hours, a.post)})
    sats = [int(s) for s in a.sats.split(",")] if a.sats else None
    print(f"[goes] {len(ev)} events -> {len(days)} days -> {root / 'goes'}", flush=True)
    listings = Listings(root, a.legacy_base, a.goesr_base)
    n_err = 0
    none_days = []
    thr = {}

    def job(d):
        t = thr.setdefault(threading.get_ident(), C.Throttle(a.min_interval))
        return fetch_day(d, root, listings, a.max_sats, t, sats)

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(job, d): d for d in days}
        for i, f in enumerate(as_completed(futs), 1):
            d = futs[f]
            try:
                got = f.result()
                msg = ", ".join(got) if got else "no satellite file found"
                if not got:
                    none_days.append(d.strftime("%Y-%m-%d"))
            except Exception as err:  # noqa: BLE001
                n_err += 1
                msg = f"ERROR {type(err).__name__}: {err}"
                C.log_error(root, "goes", day=d.strftime("%Y-%m-%d"), error=f"{type(err).__name__}: {err}")
            if i <= 5 or i % 50 == 0 or msg.startswith("ERROR"):
                print(f"  [{i}/{len(days)}] {d.date()}: {msg}", flush=True)
    if none_days:
        (root / "logs" / "goes_days_without_file.txt").write_text("\n".join(sorted(none_days)) + "\n", encoding="utf-8")
    print(f"[goes] done: {len(days) - n_err} days processed, {n_err} errors, {len(none_days)} days without any file")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
