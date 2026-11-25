"""Shared helpers for the open-data downloaders (FISM2 LaTiS, GOES XRS, F10.7).

These scripts need normal internet access to lasp.colorado.edu, ncei.noaa.gov and ngdc.noaa.gov
(they were run on the GPU server, deploy/server/).

Layout written under data/download/ (override with --root or env FISM_DL_ROOT):
    fism2/<key>.npz            FISM2 fism_flare_hr window per event (raw 0.1-nm bins)
    goes/gNN/YYYY/<file>.nc    GOES XRS 1-min science files (as published)
    extras/                    F10.7 table etc.
    logs/                      error logs (csv)
    calibration.json, calibration_report.md, calibration_events.csv
Event key = "r<catalogue row:04d>_<event_id>", e.g. r0123_20110215_X2.2.
Server URLs can be overridden (mirrors / testing) with FISM_LATIS_URL, FISM_GOES_LEGACY_BASE,
FISM_GOESR_BASE, FISM_F107_URL.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

USER_AGENT = "fism2-flare-repro-downloader/1.0 (academic research; python-requests)"
JD_UNIX0 = 2440587.5            # Julian date of 1970-01-01T00:00Z


def dl_root(arg: str | None = None) -> Path:
    p = Path(arg or os.environ.get("FISM_DL_ROOT", ROOT / "data/download"))
    for sub in ("fism2", "goes", "extras", "logs"):
        (p / sub).mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------------------------------
# HTTP (retries with back-off; 404 -> None)
# ------------------------------------------------------------------------------------------
_SESSION = None


def session():
    global _SESSION
    if _SESSION is None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        s = requests.Session()
        r = Retry(total=6, connect=6, read=6, backoff_factor=2.0, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",), respect_retry_after_header=True)
        s.mount("https://", HTTPAdapter(max_retries=r, pool_connections=8, pool_maxsize=8))
        s.mount("http://", HTTPAdapter(max_retries=r, pool_connections=8, pool_maxsize=8))
        s.headers.update({"User-Agent": USER_AGENT})
        _SESSION = s
    return _SESSION


def fetch_bytes(url: str, timeout: float = 120.0) -> bytes | None:
    """GET url -> content bytes; None on 404; raises on other errors (after retries)."""
    r = session().get(url, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def fetch_text(url: str, timeout: float = 120.0) -> str | None:
    b = fetch_bytes(url, timeout)
    return None if b is None else b.decode("utf-8", errors="replace")


def atomic_write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save_npz_atomic(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def log_error(root: Path, name: str, **rec):
    f = root / "logs" / f"{name}_errors.csv"
    rec = {"utc": pd.Timestamp.now('UTC').strftime("%Y-%m-%dT%H:%M:%S"), **rec}
    pd.DataFrame([rec]).to_csv(f, mode="a", header=not f.exists(), index=False)


class Throttle:
    """Minimum interval between requests per worker thread (polite to public servers)."""

    def __init__(self, min_interval: float):
        self.dt = float(min_interval)
        self.last = 0.0

    def wait(self):
        now = time.monotonic()
        if now - self.last < self.dt:
            time.sleep(self.dt - (now - self.last))
        self.last = time.monotonic()


# ------------------------------------------------------------------------------------------
# time helpers (integer "minute epoch" = minutes since 1970-01-01T00:00Z)
# ------------------------------------------------------------------------------------------
def to_minute(ts) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return int(t.value // 60_000_000_000)


def minute_to_iso(m: int) -> str:
    return pd.Timestamp(int(m) * 60, unit="s").strftime("%Y-%m-%dT%H:%M")


def jd_to_minute(jd: np.ndarray) -> np.ndarray:
    return np.rint((np.asarray(jd, dtype=float) - JD_UNIX0) * 1440.0).astype(np.int64)


# ------------------------------------------------------------------------------------------
# event lists
# ------------------------------------------------------------------------------------------
def _catalogue() -> pd.DataFrame:
    c = pd.read_csv(ROOT / "data/processed/flare_catalogue.csv", parse_dates=["start", "peak", "end"])
    c = c.reset_index().rename(columns={"index": "row"})
    c["key"] = [f"r{r:04d}_{e}" for r, e in zip(c.row, c.event_id)]
    return c


def test_events() -> pd.DataFrame:
    """The 96 Zenodo test events matched to catalogue rows (same matching as src/data/cv96.py),
    with test_k = index in the Zenodo arrays."""
    from src.data.cv96 import load_meta
    c = _catalogue()
    m = load_meta()
    rows = []
    for _, r in m.iterrows():
        hit = c[(c.event_id == r.event) & (c.peak == r.peak_time)]
        if len(hit) != 1:
            raise RuntimeError(f"cannot match test event {r.event} to the catalogue")
        d = hit.iloc[0].to_dict()
        d["test_k"] = int(r.k)
        rows.append(d)
    return pd.DataFrame(rows)


def load_events(stage: str = "test", limit: int | None = None, only_selected: bool = False) -> pd.DataFrame:
    """stage 'test' -> 96 Zenodo test events; 'all' -> every M/X catalogue row 2003-2023 (+ test_k)."""
    if stage == "test":
        ev = test_events()
    elif stage == "all":
        ev = _catalogue()
        t = test_events()[["row", "test_k"]]
        ev = ev.merge(t, on="row", how="left")
        ev["test_k"] = ev["test_k"].fillna(-1).astype(int)
        if only_selected:
            ev = ev[ev.selected | (ev.test_k >= 0)]
    else:
        raise ValueError(stage)
    ev = ev.sort_values("peak").reset_index(drop=True)
    if limit:
        ev = ev.head(int(limit))
    return ev
