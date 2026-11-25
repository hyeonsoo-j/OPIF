"""Offline tests of the open-data pipeline (scripts/download) against mock servers (tests/mock_servers.py)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/download"))
sys.path.insert(0, str(ROOT / "tests"))


def test_parse_latis_csv_jd_and_iso():
    import common as C
    from fetch_fism2 import parse_latis_csv
    t0 = C.to_minute("2011-02-15T01:00")
    jd = [(t0 + i) * 60 / 86400 + C.JD_UNIX0 for i in range(3)]
    txt = "time (Julian Date),wavelength (nm),irradiance (W/m^2/nm)\n" + "".join(
        f"{j:.6f},{w:.2f},{1e-4 * (i + 1) * (k + 1):.4e}\n" for i, j in enumerate(jd) for k, w in enumerate([9.05, 9.15]))
    wl, irr = parse_latis_csv(txt, t0, 4)
    assert np.allclose(wl, [9.05, 9.15])
    assert np.allclose(irr[2], [3e-4, 6e-4]) and np.isnan(irr[3]).all()
    iso = "time,wavelength,irradiance\n2011-02-15T01:01:00.000Z,9.05,2e-4\n"
    wl, irr = parse_latis_csv(iso, t0, 3)
    assert irr[1, 0] == pytest.approx(2e-4)


def test_bin_rules():
    from calibrate import bin_mask
    wl = np.round(np.arange(8.95, 10.06, 0.1), 2)
    assert list(wl[bin_mask(wl, "w1", "inner")]) == [9.05, 9.15, 9.25, 9.35, 9.45, 9.55, 9.65, 9.75, 9.85]
    assert bin_mask(wl, "w1", "plus").sum() == 10 and wl[bin_mask(wl, "w1", "plus")].max() == 9.95
    assert wl[bin_mask(wl, "w1", "minus")].min() == 8.95 and bin_mask(wl, "w1", "wide").sum() == 11
    w0 = np.round(np.arange(0.05, 0.86, 0.1), 2)
    assert list(w0[bin_mask(w0, "w0", "inner")]) == [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]


def test_f107_parser():
    from fetch_extras import parse_fluxtable
    txt = ("fluxdate    fluxtime    fluxjulian    fluxcarrington  fluxobsflux  fluxadjflux  fluxursi\n"
           "----------  ----------  ------------  --------------  -----------  -----------  --------\n"
           "20110214    200000      02455607.333  002106.786      000107.6     000110.1     000099.1\n")
    d = parse_fluxtable(txt)
    assert len(d) == 1 and d.adj_sfu.iloc[0] == pytest.approx(110.1)
    assert str(d.datetime.iloc[0]) == "2011-02-14 20:00:00"


def test_pipeline_recovers_known_rules(tmp_path, monkeypatch):
    import build_events
    import common as C
    import run_all
    from mock_servers import TRUTH_RULE, MockWorld
    n = 8
    # mock truth: authors' peak = w0 maximum (true for 86/96 Zenodo events) shifted from the catalogue
    # peak by -3..3 min; use events whose Zenodo w0 maximum is at the peak minute so the rule is exact
    z = np.load(ROOT / "data/processed/zenodo_test_xy.npz")
    full = np.concatenate([z["X_real"], z["Y_true"][:, 1:, 0]], axis=1)
    ok_k = np.where(full[:, 44:75].argmax(1) == 15)[0]
    all_test = C.load_events("test")
    ev = all_test[all_test.test_k.isin(ok_k)].head(n).reset_index(drop=True)
    orig = C.load_events
    monkeypatch.setattr(C, "load_events", lambda stage="test", limit=None, only_selected=False:
                        ev.copy() if stage == "test" else orig(stage, limit, only_selected))
    world = MockWorld(ev, seed=3)
    world.install(monkeypatch)
    rc = run_all.main(["--stage", "test", "--limit", str(n), "--root", str(tmp_path), "--no-extras", "--workers", "1"])
    assert rc == 0
    cal = json.loads((tmp_path / "calibration.json").read_text())
    for b, r in TRUTH_RULE.items():
        assert cal["bands"][b]["rule"] == r, (b, cal["bands"][b])
        assert cal["bands"][b]["agg"] == "sum_x_dlambda"
    assert cal["input_source"] == "fism2_w0"
    assert cal["peak_rule"].startswith("argmax_w0")
    assert cal["status"] == "PASS"
    E = __import__("pandas").read_csv(tmp_path / "calibration_events.csv")
    for r in E.itertuples():
        assert r.shift_best == world.shift[r.key]
    # GOES flags: the flagged spikes must be masked
    assert (E.goes_sat_flags == 12).all() or E.goes_sat_flags.notna().all()
    # build the test rows and compare with Zenodo exactly (resume: second run downloads nothing)
    calls = dict(world.calls)
    out = tmp_path / "ev.npz"
    assert build_events.main(["--stage", "test", "--root", str(tmp_path), "--out", str(out)]) == 0
    d = np.load(out, allow_pickle=False)
    z = np.load(ROOT / "data/processed/zenodo_test_xy.npz")
    k = d["test_k"]
    assert len(k) == n
    assert np.max(np.abs(np.log10(d["Y"]) - np.log10(z["Y_true"][k]))) < 1e-5
    assert np.max(np.abs(np.log10(d["X"]) - np.log10(z["X_real"][k]))) < 1e-5
    # GOES = 1.15 x FISM2 except at the flagged spikes, which are masked and then interpolated (<= 5 min gaps)
    same = np.isclose(d["X_goes_b"], 1.15 * d["X_fism"], rtol=1e-5)
    assert np.isfinite(d["X_goes_b"]).all() and same.mean() > 0.97
    assert np.isfinite(d["bg_goes_b_24h"]).all()
    assert (d["shift"] == np.array([world.shift[key] for key in d["key"]])).all()
    run_all.main(["--stage", "test", "--limit", str(n), "--root", str(tmp_path), "--no-extras", "--workers", "1"])
    assert world.calls["latis"] == calls["latis"] and world.calls["nc"] == calls["nc"]
    # refuses to overwrite
    assert build_events.main(["--stage", "test", "--root", str(tmp_path), "--out", str(out)]) == 1


def test_fill_short_gaps():
    from build_events import fill_short_gaps
    v = np.array([1e-6, np.nan, 0.0, 4e-6] + [np.nan] * 6 + [2e-6, np.nan])
    f = fill_short_gaps(v)
    assert np.allclose(f[:4], [1e-6, 10 ** -5.8, 10 ** -5.6, 4e-6])      # 2-minute gap, log-linear
    assert np.isnan(f[4:10]).all() and np.isnan(f[-1])                     # long gap / trailing gap kept
