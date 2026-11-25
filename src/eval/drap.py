"""NOAA D-RAP style HF absorption from the 0.1-0.8 nm X-ray flux (Phase 4).

  HAF [MHz] = 10 log10(F [W m^-2]) + 65                       (highest affected frequency, 1 dB)
  A(f) [dB] = (HAF * cos(chi)^0.75 / f)^1.5    for chi < 90 deg and HAF > 0, else 0
chi = solar zenith angle at the site (default Korea, 37 N 127 E), f = 10 MHz by default.
Solar position: NOAA General Solar Position (Spencer 1971 Fourier series; error < ~0.1 deg),
checked against astropy in tests/test_drap.py.
"""
from __future__ import annotations

import numpy as np

SITE_KOREA = (37.0, 127.0)


def solar_zenith_deg(t_utc, lat_deg: float, lon_deg: float) -> np.ndarray:
    t = np.asarray(t_utc, dtype="datetime64[s]")
    year_start = t.astype("datetime64[Y]")
    doy = (t.astype("datetime64[D]") - year_start.astype("datetime64[D]")).astype(float) + 1.0
    sec = (t - t.astype("datetime64[D]")).astype(float)
    hour = sec / 3600.0
    yr = year_start.astype(int) + 1970
    ndays = np.where(((yr % 4 == 0) & (yr % 100 != 0)) | (yr % 400 == 0), 366.0, 365.0)
    g = 2 * np.pi / ndays * (doy - 1 + (hour - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                    - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hour * 60 + eqt + 4 * lon_deg
    ha = np.deg2rad(tst / 4 - 180)
    lat = np.deg2rad(lat_deg)
    cz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    return np.rad2deg(np.arccos(np.clip(cz, -1, 1)))


def haf_mhz(flux_wm2) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(flux_wm2, float), 1e-30)) + 65.0


def absorption_db(flux_wm2, zenith_deg, f_mhz: float = 10.0) -> np.ndarray:
    haf = np.maximum(haf_mhz(flux_wm2), 0.0)
    cz = np.cos(np.deg2rad(np.asarray(zenith_deg, float)))
    day = cz > 0
    a = np.zeros(np.broadcast(haf, cz).shape)
    val = (haf * np.clip(cz, 0, None) ** 0.75 / f_mhz) ** 1.5
    a[...] = np.where(day, val, 0.0)
    return a


def event_absorption(w0_flux: np.ndarray, peak_time, site=SITE_KOREA, f_mhz: float = 10.0):
    """w0_flux (..., 180) for minutes 0..179 after `peak_time` (numpy datetime64 / pandas Timestamp)."""
    t = np.datetime64(peak_time, "s") + np.arange(w0_flux.shape[-1]).astype("timedelta64[m]")
    z = solar_zenith_deg(t, *site)
    return absorption_db(w0_flux, z, f_mhz), z
