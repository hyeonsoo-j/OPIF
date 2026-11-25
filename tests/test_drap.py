"""D-RAP conversion checks: solar zenith vs astropy, HAF/absorption formula values."""
import numpy as np
import pytest

from src.eval.drap import absorption_db, haf_mhz, solar_zenith_deg


def test_haf_values():
    assert np.isclose(haf_mhz(1e-4), 25.0)       # X1  -> 25 MHz
    assert np.isclose(haf_mhz(1e-5), 15.0)       # M1  -> 15 MHz


def test_absorption_formula_and_night():
    a = absorption_db(np.array([1e-4, 1e-4]), np.array([0.0, 95.0]), 10.0)
    assert np.isclose(a[0], (25.0 / 10.0) ** 1.5)
    assert a[1] == 0.0
    assert absorption_db(1e-8, 0.0) == 0.0       # HAF < 0 -> no absorption


def test_zenith_against_astropy():
    astropy = pytest.importorskip("astropy")
    from astropy.coordinates import AltAz, EarthLocation, get_sun
    from astropy.time import Time
    import astropy.units as u
    loc = EarthLocation(lat=37 * u.deg, lon=127 * u.deg, height=0 * u.m)
    times = np.array(["2003-11-04T03:00:00", "2011-08-04T06:30:00", "2022-01-20T00:10:00",
                      "2014-06-11T09:00:00"], dtype="datetime64[s]")
    ours = solar_zenith_deg(times, 37.0, 127.0)
    t = Time(times.astype(str), scale="utc")
    alt = get_sun(t).transform_to(AltAz(obstime=t, location=loc)).alt.deg
    assert np.max(np.abs(ours - (90 - alt))) < 0.5
