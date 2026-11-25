"""A1 continuous detector: matching and classification on three synthetic days (no network)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

netCDF4 = pytest.importorskip("netCDF4")
ROOT = Path(__file__).resolve().parents[1]


def _flare(n, i0, peak, bg=1e-6, rise=8, decay=40):
    t = np.arange(n, dtype=float)
    up = np.exp(-0.5 * ((t - i0) / (rise / 2)) ** 2)
    down = np.exp(-(t - i0) / decay)
    return (peak - bg) * np.where(t <= i0, up, down)


def _write_day(root: Path, day: pd.Timestamp, b: np.ndarray):
    d = root / "goes" / "g15" / f"{day.year}"; d.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(d / f"sci_xrsf-l2-avg1m_g15_d{day:%Y%m%d}_v1-0-3.nc", "w") as ds:
        ds.createDimension("time", 1440)
        tv = ds.createVariable("time", "f8", ("time",)); tv.units = "seconds since 2000-01-01 12:00:00"
        tv[:] = (day - pd.Timestamp("2000-01-01 12:00:00")).total_seconds() + 60.0 * np.arange(1440)
        for nm, v in (("xrsb_flux", b), ("xrsa_flux", b / 10)):
            x = ds.createVariable(nm, "f4", ("time",), fill_value=-9999.0); x[:] = np.where(np.isfinite(v), v, -9999.0)
        for nm in ("xrsb_flag", "xrsa_flag"):
            f = ds.createVariable(nm, "i2", ("time",)); f[:] = np.zeros(1440, dtype=np.int16)


def test_continuous_detector_classes(tmp_path):
    days = pd.date_range("2014-01-01", periods=3, freq="D")
    n = 1440
    b1 = 1e-6 + _flare(n, 600, 2e-5)                      # day 1 10:00 listed M2.0 (true units) -> matched
    b2 = 1e-6 + _flare(n, 300, 1.2e-5)                    # day 2 05:00 unlisted, below list-M1 (1.43e-5)
    b3 = np.full(n, 1e-6); b3[300] = 5e-5                 # day 3 05:00 one-minute spike
    b3 += _flare(n, 900, 3e-5); b3[860:895] = np.nan      # day 3 15:00 unlisted flare right after a 35-min gap
    for d, b in zip(days, (b1, b2, b3)):
        _write_day(tmp_path, d, b)
    pd.DataFrame(dict(start=["2014-01-01 09:50:00"], peak=["2014-01-01 10:00:00"], end=["2014-01-01 10:40:00"], cls=["M1.4"],
                      event_id=["20140101_M1.4"])).to_csv(tmp_path / "cat.csv", index=False)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/continuous_detector.py"), "--no-download", "--products", "science",
                        "--periods", "2014-01-01:2014-01-04:15", "--root", str(tmp_path), "--catalogue", str(tmp_path / "cat.csv"),
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    S = pd.read_csv(out / "continuous_detector_summary.csv")
    s = S[S.threshold == "1e-5"].iloc[0]
    assert s.listed == 1 and s.listed_declared == 1 and abs(s.data_days - (3 - 35 / 1440)) < 0.01
    D = pd.read_csv(out / "continuous_detector_declarations.csv"); D = D[D.threshold == "1e-5"]
    assert set(D.cls) == {"matched", "below_list_M1", "spike", "gap"}, D[["t_max", "value", "cls"]]
    L = S[S.threshold == "list_scale"].iloc[0]
    assert L.below_list_M1_per_day == 0                   # the C-class maximum is not declared at the catalogue-scale threshold
    assert "DECISION" in r.stdout
