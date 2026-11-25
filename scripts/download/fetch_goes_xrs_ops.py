"""Download OPERATIONAL GOES XRS 1-minute files (the archived real-time product) for the minutes around each flare of the
built dataset (paper-1 P11), next to the science-quality files of fetch_goes_xrs.py.

Sources (layout from the P10 probe, results/full_server/goes_operational_probe.csv):
    GOES 16-19: https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/
                goesNN/l2/data/xrsf-l2-avg1m/YYYY/MM/dn_xrsf-l2-avg1m_gNN_dYYYYMMDD_v*.nc         (daily, true units)
    GOES 8-15 : https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/
                YYYY/MM/goesNN/netcdf/gNN_xrs_1m_YYYYMMDD_YYYYMMDD.nc   (monthly; csv/ holds the same as text)
                These carry the SWPC scaling: true flux = value / 0.7 (XRS-B), / 0.85 (XRS-A). load_series_ops() removes it.
                Both the netCDF and the text file of a month are fetched (a few MB): the reader uses the netCDF file and
                falls back to the text file if the netCDF file cannot be read.
For every flare only the satellite of its science-quality input is fetched (events npz: goes_sat). Files are stored unchanged
under <root>/goes_ops/gNN/YYYY/; monthly listings are cached in <root>/goes_ops/listings/.

    python scripts/download/fetch_goes_xrs_ops.py --events data/processed/events_latis.npz [--limit 5]
"""
from __future__ import annotations

import argparse
import io
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
import fetch_goes_xrs as G  # noqa: E402

LEGACY_OPS_BASE = os.environ.get("FISM_GOES_LEGACY_OPS_BASE", "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg")
GOESR_BASE = G.GOESR_BASE
SWPC_SCALE = {"xrsb": 0.7, "xrsa": 0.85}          # GOES 8-15 operational values = true flux x these factors
B_NAMES = ("B_AVG", "xl", "XL", "b_avg", "B_FLUX", "b_flux", "xrsb_flux", "xrsb")
A_NAMES = ("A_AVG", "xs", "XS", "a_avg", "A_FLUX", "a_flux", "xrsa_flux", "xrsa")
BF_NAMES = ("B_QUAL_FLAG", "b_qual_flag", "xrsb_flag", "xrsb_flags", "b_flags", "b_flag")
AF_NAMES = ("A_QUAL_FLAG", "a_qual_flag", "xrsa_flag", "xrsa_flags", "a_flags", "a_flag")
T_NAMES = ("time_tag", "time")


def dir_urls(sat: int, day: pd.Timestamp, legacy_base=LEGACY_OPS_BASE, goesr_base=GOESR_BASE) -> list[str]:
    if sat <= 15:
        b = f"{legacy_base}/{day.year}/{day.month:02d}/goes{sat:02d}"
        return [f"{b}/netcdf/", f"{b}/csv/"]
    return [f"{goesr_base}/goes{sat}/l2/data/xrsf-l2-avg1m/{day.year}/{day.month:02d}/"]


def file_regex(sat: int, day: pd.Timestamp) -> re.Pattern:
    if sat <= 15:                                   # one file per month
        return re.compile(rf"g{sat:02d}_xrs_1m_{day.strftime('%Y%m')}01_{day.strftime('%Y%m')}(\d\d)\.(nc|csv)")
    return re.compile(rf"dn[_-]xrsf-l2-avg1m_g{sat}_d{day.strftime('%Y%m%d')}_v([0-9][0-9\-]*)\.nc")


def _rank(sat: int, m: re.Match):
    return (1 if m.group(2) == "nc" else 0,) if sat <= 15 else tuple(int(x) for x in m.group(1).split("-") if x.isdigit())


class Listings:
    """Directory listings (cached on disk; a missing directory is cached as an empty listing)."""

    def __init__(self, root: Path, legacy_base: str, goesr_base: str):
        self.dir = root / "goes_ops" / "listings"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.mem: dict[str, list[str]] = {}
        self.lb, self.gb = legacy_base, goesr_base

    def get(self, sat: int, day: pd.Timestamp) -> list[tuple[str, str]]:
        """[(directory url, file name)] of the month of `day`."""
        k = f"g{sat:02d}_{day.year}{day.month:02d}"
        with self.lock:
            if k not in self.mem:
                f = self.dir / f"{k}.json"
                if f.exists():
                    self.mem[k] = [tuple(x) for x in json.loads(f.read_text(encoding="utf-8"))]
                else:
                    out = []
                    for url in dir_urls(sat, day, self.lb, self.gb):
                        html = C.fetch_text(url, timeout=60)
                        if html is not None:
                            out += [(url, n) for n in sorted(set(re.findall(r'href="([^"/?]+\.(?:nc|csv))"', html)))]
                    C.atomic_write_bytes(f, json.dumps(out).encode())
                    self.mem[k] = out
            return self.mem[k]


def find_local_all(root: Path, sat: int, day: pd.Timestamp) -> list[Path]:
    """Local files of the day (GOES-R) or month (GOES 8-15), best first: netCDF before text, newest version first."""
    d = root / "goes_ops" / f"g{sat:02d}" / f"{day.year}"
    if not d.exists():
        return []
    rx = file_regex(sat, day)
    return sorted([p for p in d.iterdir() if rx.fullmatch(p.name)], key=lambda p: _rank(sat, rx.fullmatch(p.name)), reverse=True)


def find_local(root: Path, sat: int, day: pd.Timestamp) -> Path | None:
    hits = find_local_all(root, sat, day)
    return hits[0] if hits else None


def fetch_one(sat: int, day: pd.Timestamp, root: Path, listings: Listings, throttle: C.Throttle) -> str:
    """GOES-R: the newest version of the day. GOES 8-15: the netCDF AND the text file of the month (whichever exist)."""
    have = {p.name for p in find_local_all(root, sat, day)}
    if have and (sat > 15 or {Path(n).suffix for n in have} >= {".nc", ".csv"}):
        return "cached"
    rx = file_regex(sat, day)
    cands = sorted([(u, n, rx.fullmatch(n)) for u, n in listings.get(sat, day) if rx.fullmatch(n)], key=lambda c: _rank(sat, c[2]), reverse=True)
    if sat > 15:
        cands = cands[:1] if not have else []           # one version is enough
    want = [(u, n) for u, n, _ in cands if n not in have]
    if not want:
        return "cached" if have else "no file"
    got = []
    for url, name in want:
        throttle.wait()
        data = C.fetch_bytes(url + name, timeout=300)
        if data is not None:
            C.atomic_write_bytes(root / "goes_ops" / f"g{sat:02d}" / f"{day.year}" / name, data)
            got.append(name)
    return " + ".join(got) if got else ("cached" if have else "no file")


# ------------------------------------------------------------------------------------------
# reading
# ------------------------------------------------------------------------------------------
def _clean(v) -> np.ndarray:
    v = np.asarray(v, dtype=float).copy()
    v[~(np.isfinite(v) & (v > 0))] = np.nan
    return v


def read_legacy_nc(path: Path) -> pd.DataFrame:
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        vs = ds.variables
        tn, an, bn = G._pick(vs, *T_NAMES), G._pick(vs, *A_NAMES), G._pick(vs, *B_NAMES)
        if tn is None or bn is None:
            raise ValueError(f"{path.name}: no time / XRS-B variable among {list(vs)[:30]}")
        out = {"minute": G._decode_time(vs[tn])}
        for lab, n, fnames in (("xrsa", an, AF_NAMES), ("xrsb", bn, BF_NAMES)):
            out[lab] = _clean(np.ma.filled(np.ma.asarray(vs[n][:]).astype(float), np.nan)) if n is not None else np.full(len(out["minute"]), np.nan)
            fn = G._pick(vs, *fnames)
            out[f"{lab}_flag"] = (np.ma.filled(np.ma.asarray(vs[fn][:]), 0).astype(np.int64) if fn is not None
                                  else np.full(len(out["minute"]), -1, dtype=np.int64))
    return pd.DataFrame(out).drop_duplicates("minute").set_index("minute").sort_index()


def read_legacy_csv(path: Path) -> pd.DataFrame:
    """Text version: a header block, then a line starting with 'time_tag' that names the columns, then the rows."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    h = next((i for i, ln in enumerate(lines) if ln.split(",")[0].strip().strip('"').lower() in T_NAMES), None)
    if h is None:
        raise ValueError(f"{path.name}: no column line starting with time_tag (first lines: {lines[:5]})")
    df = pd.read_csv(io.StringIO("\n".join(lines[h:])), skipinitialspace=True)
    df.columns = [c.strip().strip('"') for c in df.columns]
    pick = lambda names: next((c for c in names if c in df.columns), None)          # noqa: E731
    tn, an, bn = pick(T_NAMES), pick(A_NAMES), pick(B_NAMES)
    if bn is None:
        raise ValueError(f"{path.name}: no XRS-B column among {list(df.columns)}")
    t = pd.to_datetime(df[tn].astype(str), errors="coerce", utc=True)
    mins = ((t - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(minutes=1)).to_numpy(dtype=float)      # any datetime resolution
    out = {"minute": np.floor(np.where(np.isfinite(mins), mins, 0.0) + 1e-6).astype(np.int64)}
    for lab, n, fnames in (("xrsa", an, AF_NAMES), ("xrsb", bn, BF_NAMES)):
        out[lab] = _clean(pd.to_numeric(df[n], errors="coerce")) if n is not None else np.full(len(df), np.nan)
        fn = pick(fnames)
        out[f"{lab}_flag"] = (pd.to_numeric(df[fn], errors="coerce").fillna(0).astype(np.int64).to_numpy() if fn is not None
                              else np.full(len(df), -1, dtype=np.int64))
    res = pd.DataFrame(out)[t.notna().to_numpy()]
    return res.drop_duplicates("minute").set_index("minute").sort_index()


READ_ERRORS: dict[str, str] = {}      # file name -> why it could not be read (printed by the build step)


def _read_one(p: Path, sat: int) -> pd.DataFrame:
    if sat > 15:
        return G.read_xrs_nc(p)
    return read_legacy_nc(p) if p.suffix == ".nc" else read_legacy_csv(p)


@lru_cache(maxsize=64)
def _read_cached(paths: tuple, sat: int) -> pd.DataFrame | None:
    """The first of the files (best first) that can be read and holds a valid XRS-B value; None if none can."""
    for path in paths:
        try:
            df = _read_one(Path(path), sat)
            if np.isfinite(df["xrsb"].to_numpy(dtype=float)).any():
                return df
            READ_ERRORS.setdefault(Path(path).name, "no valid XRS-B value")
        except Exception as err:  # noqa: BLE001  (a format surprise must not stop the other files)
            READ_ERRORS.setdefault(Path(path).name, f"{type(err).__name__}: {err}"[:300])
    return None


def load_series_ops(root: Path, sat: int, t0_min: int, n: int, use_flags: bool = True) -> dict[str, np.ndarray] | None:
    """Operational XRS-A/B in TRUE units on the minute grid [t0, t0+n) from the local files of one satellite; None if no file.
    GOES 8-15 values are divided by the SWPC factors. Minutes with a positive quality flag are NaN (use_flags)."""
    days = pd.date_range(pd.Timestamp(t0_min * 60, unit="s").normalize(), pd.Timestamp((t0_min + n - 1) * 60, unit="s").normalize(), freq="D")
    groups = []
    for d in days:
        ps = tuple(str(p) for p in find_local_all(root, sat, d))
        if ps and ps not in groups:
            groups.append(ps)
    parts = [df for df in (_read_cached(ps, sat) for ps in groups) if df is not None]
    if not parts:
        return None
    df = pd.concat(parts)
    df = df[~df.index.duplicated()].reindex(np.arange(t0_min, t0_min + n))
    res = {}
    for lab in ("xrsa", "xrsb"):
        v = df[lab].to_numpy(dtype=float)
        if use_flags:
            fl = df[f"{lab}_flag"].to_numpy()
            v = np.where((fl > 0) & np.isfinite(fl), np.nan, v)
        res[lab] = v / SWPC_SCALE[lab] if sat <= 15 else v
    return res


def describe_file(path: Path, sat: int) -> str:
    """One line on what a file holds (printed by the build step, so that a format surprise is visible in the log)."""
    try:
        if path.suffix == ".nc":
            import netCDF4
            with netCDF4.Dataset(path) as ds:
                vs = ds.variables
                tn = G._pick(vs, *T_NAMES)
                return f"{path.name}: variables {list(vs)[:16]}; time units {getattr(vs[tn], 'units', '?') if tn else '?'}"
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        h = next((i for i, ln in enumerate(lines) if ln.split(",")[0].strip().strip('"').lower() in T_NAMES), None)
        return f"{path.name}: {len(lines)} lines; column line {lines[h] if h is not None else None!r}; first row {lines[h + 1] if h is not None and h + 1 < len(lines) else None!r}"
    except Exception as err:  # noqa: BLE001
        return f"{path.name}: unreadable ({type(err).__name__}: {err})"


def needed(events: str, pre: int, post: int, limit: int | None = None) -> list[tuple[int, pd.Timestamp]]:
    """(satellite, day) pairs covering minutes [peak - pre, peak + post) of every flare with a GOES series."""
    d = np.load(events, allow_pickle=False)
    out = set()
    rows = [i for i in range(len(d["event_id"])) if int(d["goes_sat"][i]) >= 0]
    for i in rows[:limit] if limit else rows:
        p = pd.Timestamp(str(d["peak"][i]))
        for day in pd.date_range((p - pd.Timedelta(minutes=pre)).normalize(), (p + pd.Timedelta(minutes=post)).normalize(), freq="D"):
            out.add((int(d["goes_sat"][i]), day))
    return sorted(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default=str(Path(__file__).resolve().parents[2] / "data/processed/events_latis.npz"))
    ap.add_argument("--pre", type=int, default=70, help="minutes before the dataset peak to cover (default 70)")
    ap.add_argument("--post", type=int, default=190)
    ap.add_argument("--limit", type=int, default=None, help="only the first N flares (smoke test)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--min-interval", type=float, default=0.3)
    ap.add_argument("--root", default=None)
    ap.add_argument("--legacy-base", default=LEGACY_OPS_BASE)
    ap.add_argument("--goesr-base", default=GOESR_BASE)
    a = ap.parse_args(argv)
    root = C.dl_root(a.root)
    pairs = needed(a.events, a.pre, a.post, a.limit)
    # GOES 8-15: one file per month, so fetch one day per (satellite, month)
    jobs = sorted({(s, d if s > 15 else d.replace(day=1)) for s, d in pairs})
    print(f"[goes-ops] {len(pairs)} satellite-days -> {len(jobs)} files -> {root / 'goes_ops'}", flush=True)
    listings = Listings(root, a.legacy_base, a.goesr_base)
    thr, n_err, missing, counts = {}, 0, [], {}

    def job(sd):
        t = thr.setdefault(threading.get_ident(), C.Throttle(a.min_interval))
        return fetch_one(sd[0], sd[1], root, listings, t)

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(job, sd): sd for sd in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            sat, day = futs[f]
            try:
                msg = f.result()
                if msg == "no file":
                    missing.append(f"g{sat:02d} {day.date()}")
                key = msg if msg in ("cached", "no file") else "downloaded"
                counts[key] = counts.get(key, 0) + 1
            except Exception as err:  # noqa: BLE001
                n_err += 1
                msg = f"ERROR {type(err).__name__}: {err}"
                C.log_error(root, "goes_ops", sat=sat, day=str(day.date()), error=f"{type(err).__name__}: {err}")
            if i <= 5 or i % 50 == 0 or msg.startswith("ERROR"):
                print(f"  [{i}/{len(jobs)}] g{sat:02d} {day.date()}: {msg}", flush=True)
    if missing:
        (root / "logs").mkdir(parents=True, exist_ok=True)
        (root / "logs" / "goes_ops_without_file.txt").write_text("\n".join(sorted(missing)) + "\n", encoding="utf-8")
    print(f"[goes-ops] done: {counts}, {n_err} errors, {len(missing)} satellite-days/months without a file")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
