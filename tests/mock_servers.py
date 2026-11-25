"""Offline stand-ins for LaTiS (fism_flare_hr.csv) and the NOAA GOES XRS directories, built from the
Zenodo test arrays with a KNOWN reconstruction rule, so the download -> calibrate -> build chain can be
tested without network access (the real servers are unreachable from the development workspace).

Truth used by the mock (calibrate.py must recover it):
    bin rules : w0 'inner' (0.15..0.75 nm), w1 'plus' (9.05..9.95), w2 'inner', w3 'wide'
    aggregation sum x 0.1 nm, input = FISM2 0.1-0.8 nm, authors' peak = catalogue peak + shift
    (shift in -3..3 min; the authors' peak is the w0 maximum, so the argmax rules reproduce it)
    GOES XRS-B science = FISM2 w0 x 1.15 (so X != GOES), with a few flagged spikes.
"""
from __future__ import annotations

import re
import sys
import threading
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/download"))

TRUTH_RULE = {"w0": "inner", "w1": "plus", "w2": "inner", "w3": "wide"}
BANDS = ["w0", "w1", "w2", "w3"]
_NC_LOCK = threading.Lock()   # netCDF4/HDF5 is not thread-safe (only the mock writes netCDF in threads)


class MockWorld:
    def __init__(self, events: pd.DataFrame, seed: int = 0):
        import calibrate as K
        import common as C
        self.K, self.C = K, C
        z = np.load(ROOT / "data/processed/zenodo_test_xy.npz")
        self.X, self.Y = z["X_real"].astype(float), z["Y_true"].astype(float)
        rng = np.random.default_rng(seed)
        self.ev = []
        for e in events.itertuples(index=False):
            s = int(rng.integers(-3, 4))
            self.ev.append(dict(key=e.key, k=int(e.test_k), pa=C.to_minute(e.peak) + s, shift=s))
        self.shift = {d["key"]: d["shift"] for d in self.ev}
        self.weights = {}
        self.rng = rng
        self.calls = {"latis": 0, "listing": 0, "nc": 0}

    # ---- truth series on a minute grid
    def band_truth(self, m: np.ndarray, b: int) -> np.ndarray:
        out = np.full(len(m), np.nan)
        for d in self.ev:
            idx = m - d["pa"]
            near = (idx >= -400) & (idx <= 600)
            if not near.any():
                continue
            k = d["k"]
            v = np.empty(near.sum())
            ii = idx[near]
            if b == 0:
                full = np.concatenate([self.X[k], self.Y[k, 1:, 0]])          # minutes -59..179
                v = np.where(ii < -59, self.X[k, 0], np.where(ii > 179, self.Y[k, -1, 0],
                                                               full[np.clip(ii + 59, 0, 238)]))
            else:
                v = np.where(ii < 0, 0.9 * self.Y[k, 0, b], np.where(ii > 179, self.Y[k, -1, b],
                                                                    self.Y[k, np.clip(ii, 0, 179), b]))
            out[near] = v
        return out

    # ---- LaTiS
    def latis(self, url: str) -> str:
        self.calls["latis"] += 1
        q = dict(re.findall(r"(time>=|time<|wavelength>|wavelength<)([^&]+)", url))
        t0 = self.C.to_minute(q["time>="])
        t1 = self.C.to_minute(q["time<"])
        a, z = float(q["wavelength>"]), float(q["wavelength<"])
        b = {0.0: 0, 8.9: 1, 12.9: 2, 29.9: 3}[a]
        bn = BANDS[b]
        wl = np.round(np.arange(np.floor(a * 10) / 10 + 0.05, z, 0.1), 2)
        wl = wl[(wl > a) & (wl < z)]
        sel = self.K.bin_mask(wl, bn, TRUTH_RULE[bn])
        if bn not in self.weights:
            w = self.rng.uniform(0.5, 1.5, sel.sum())
            self.weights[bn] = w / w.sum()
        m = np.arange(t0, t1)
        V = self.band_truth(m, b)
        lines = ["time (Julian Date),wavelength (nm),irradiance (W/m^2/nm)"]
        jd = m * 60.0 / 86400.0 + self.C.JD_UNIX0
        for i in range(len(m)):
            if not np.isfinite(V[i]):
                continue
            j_sel = 0
            for w_ in wl:
                if self.K.bin_mask(np.array([w_]), bn, TRUTH_RULE[bn])[0]:
                    irr = V[i] * self.weights[bn][j_sel] / 0.1
                    j_sel += 1
                else:                                   # other bins: different time profile
                    irr = 3e-4 * np.sqrt(V[i]) / 0.1
                lines.append(f"{jd[i]:.6f},{w_:.2f},{irr:.6e}")
        return "\n".join(lines) + "\n"

    # ---- GOES
    def listing(self, url: str) -> str | None:
        self.calls["listing"] += 1
        m = re.search(r"goes(\d+)/.*/(\d{4})/(\d{2})/$", url)
        sat, y, mo = int(m.group(1)), int(m.group(2)), int(m.group(3))
        import fetch_goes_xrs as G
        days = pd.date_range(f"{y}-{mo:02d}-01", periods=31, freq="D")
        days = [d for d in days if d.month == mo and G.candidate_sats(d)[0] == sat]
        if not days:
            return None
        s = f"{sat:02d}" if sat <= 15 else f"{sat}"
        names = [f"sci_xrsf-l2-avg1m_g{s}_d{d.strftime('%Y%m%d')}_v1-0-0.nc" for d in days]
        return "<html>" + "".join(f'<a href="{n}">{n}</a>' for n in names) + "</html>"

    def ncfile(self, url: str) -> bytes:
        self.calls["nc"] += 1
        import netCDF4  # noqa: F401
        m = re.search(r"_d(\d{8})_v", url)
        day = pd.Timestamp(m.group(1))
        m0 = self.C.to_minute(day)
        mins = np.arange(m0, m0 + 1440)
        b = self.band_truth(mins, 0)
        b = np.where(np.isfinite(b), 1.15 * b, 1e-6)
        flag = np.zeros(1440, dtype=np.int16)
        spikes = self.rng.choice(1440, 5, replace=False)
        b[spikes] *= 50.0
        flag[spikes] = 4
        with _NC_LOCK, tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.nc"
            with netCDF4.Dataset(p, "w") as ds:
                ds.createDimension("time", 1440)
                t = ds.createVariable("time", "f8", ("time",))
                t.units = "seconds since 2000-01-01 12:00:00"
                t[:] = (mins * 60.0) - (pd.Timestamp("2000-01-01 12:00:00") - pd.Timestamp("1970-01-01")).total_seconds()
                for n, v in (("xrsb_flux", b), ("xrsa_flux", 0.1 * b)):
                    x = ds.createVariable(n, "f4", ("time",), fill_value=-9999.0)
                    x[:] = v
                for n in ("xrsb_flag", "xrsa_flag"):
                    x = ds.createVariable(n, "i2", ("time",))
                    x[:] = flag
            return p.read_bytes()

    # ---- install as the HTTP layer
    def install(self, monkeypatch):
        import common as C

        def fetch_text(url, timeout=120.0):
            if "fism_flare_hr" in url:
                return self.latis(url)
            if url.endswith("/"):
                return self.listing(url)
            raise AssertionError(url)

        def fetch_bytes(url, timeout=120.0):
            if url.endswith(".nc"):
                return self.ncfile(url)
            t = fetch_text(url, timeout)
            return None if t is None else t.encode()

        monkeypatch.setattr(C, "fetch_text", fetch_text)
        monkeypatch.setattr(C, "fetch_bytes", fetch_bytes)
