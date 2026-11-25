"""Peak detector (notebooks/peak_detector.py): first-declaration rules (P4, P4b) and the armed detector (P4c)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "scripts/download"))
sys.path.insert(0, str(ROOT / "tests"))
import peak_detector as PD  # noqa: E402

M1 = 1e-5
PRE = 60


def flare(n=240, bg=1e-6, peaks=((60, 5e-5),), rise=8.0, decay=25.0):
    """Background + flares with a linear-exponential shape (rise time `rise`, decay time `decay` minutes)."""
    t = np.arange(n, dtype=float)
    b = np.full(n, bg)
    for tp, amp in peaks:
        x = t - tp
        b += amp * np.where(x <= 0, np.exp(x / rise * 3.0), np.exp(-x / decay))
    return b


def times(dec):
    return [(q["t"], q["t_max"], q["kind"]) for q in dec]


def test_single_flare_is_declared_once_k_minutes_after_its_maximum():
    b = flare()
    for k in (1, 2, 5, 10):
        dec = PD.detect_all(b, k, M1)
        assert times(dec) == [(60 + k, 60, "new")]
        o = PD.own_peak(dec, PRE, tol=5)
        assert o["own"] == k and o["first"] == k and o["n_before"] == 0 and o["n_own"] == 1 and o["n_after"] == 0
        assert PD.detect(b, k, M1) == 60 + k and PD.detect(b, k, M1, require_rise=True) == 60 + k


def test_precursor_then_larger_peak_is_declared_again():
    # shallow dip between the two maxima: the larger one is an update of the same episode
    b = flare(peaks=((30, 2e-5), (60, 8e-5)), rise=4.0, decay=60.0)
    dec = PD.detect_all(b, 2, M1)
    assert times(dec) == [(32, 30, "new"), (62, 60, "update")]
    o = PD.own_peak(dec, PRE, tol=5)
    assert o["first"] == -28 and o["own"] == 2 and o["n_before"] == 1
    # deep dip (the flux halves and doubles again before it passes the first maximum): a new episode
    b = flare(peaks=((30, 2e-5), (60, 8e-5)), rise=4.0, decay=12.0)
    dec = PD.detect_all(b, 2, M1)
    assert times(dec) == [(32, 30, "new"), (62, 60, "new")] and [q["episode"] for q in dec] == [0, 1]
    assert PD.own_peak(dec, PRE, tol=5)["own"] == 2
    # the first-declaration rules stop at the precursor
    assert PD.detect(b, 2, M1) == 32 and PD.detect(b, 2, M1, require_rise=True) == 32


def test_stream_opening_on_a_decaying_tail():
    # tail of an earlier X-class flare, then an M3 flare that stays below the tail's first sample
    t = np.arange(240, dtype=float)
    tail = 2e-4 * np.exp(-(t + 20) / 15.0)
    b = tail + flare(bg=0.0, peaks=((60, 3e-5),), rise=5.0, decay=20.0)
    assert b[0] > b[60] > M1 and b[:30].argmax() == 0
    dec = PD.detect_all(b, 2, M1)
    assert times(dec) == [(62, 60, "new")]                 # nothing on the tail, the flare as a new episode
    assert dec[0]["episode"] == 1 and dec[0]["value"] >= 2 * dec[0]["lo_pre"]
    assert PD.detect(b, 2, M1) == 2                         # first rule: declared on the tail (window edge)
    # second rule: the tail is declared once it has decayed to half its first value, long before the flare
    assert PD.detect(b, 2, M1, require_rise=True) < 30


def test_bump_below_the_rise_factor_is_not_declared():
    t = np.arange(240, dtype=float)
    b = 6e-5 * np.exp(-t / 60.0)                           # slow decay from M6
    b[58:63] *= np.array([1.2, 1.5, 1.7, 1.5, 1.2])        # bump by a factor 1.7 (< 2)
    assert PD.detect_all(b, 2, M1) == []
    assert len(PD.detect_all(b, 2, M1, rise_factor=1.5)) == 1


def test_gaps_and_threshold():
    b = flare()
    b[50:57] = np.nan                                      # gap in the rise
    assert times(PD.detect_all(b, 2, M1)) == [(62, 60, "new")]
    assert PD.detect_all(flare(peaks=((60, 5e-6),)), 2, M1) == []      # C-class flare: below the threshold
    assert PD.detect_all(np.full(240, np.nan), 2, M1) == []


def test_shoulder_inside_the_tolerance_window():
    b = flare(peaks=((60, 3e-4),), rise=6.0, decay=20.0)
    b[55:58] = b[55] * np.array([1.0, 0.97, 0.95])         # 3-minute pause on the rise (above M1), 5 minutes before the maximum
    assert b[55] > M1
    dec = PD.detect_all(b, 2, M1)
    assert [q["t_max"] for q in dec] == [55, 60] and dec[1]["kind"] == "update"
    o = PD.own_peak(dec, PRE, tol=5)
    assert o["n_own"] == 2 and o["own_first"] == -3 and o["own"] == 2 and o["n_before"] == 0
    assert PD.own_peak(dec, PRE, tol=2)["n_before"] == 1


def test_causal_median_uses_no_future_sample():
    rng = np.random.default_rng(0)
    b = flare() * rng.lognormal(0, 0.05, 240)
    m = PD.causal_med3(b)
    b2 = b.copy(); b2[100] *= 50                           # a spike at minute 100
    m2 = PD.causal_med3(b2)
    assert np.array_equal(m[:100], m2[:100])               # nothing before the spike changes
    c, c2 = PD.medfilt3(b), PD.medfilt3(b2)
    assert c[99] != c2[99] or np.isclose(c[99], c2[99])    # the centred filter may look one minute ahead
    assert m2[100] < b2[100]                               # and a single-minute spike is removed


def test_armed_detector_on_the_mock_dataset(tmp_path, monkeypatch):
    """End to end on mock downloads: declarations, per-event table, summary and exported streams agree."""
    import build_events
    import common as C
    import run_all
    from mock_servers import MockWorld
    n = 8
    z = np.load(ROOT / "data/processed/zenodo_test_xy.npz")
    full = np.concatenate([z["X_real"], z["Y_true"][:, 1:, 0]], axis=1)
    ok_k = np.where(full[:, 44:75].argmax(1) == 15)[0]
    all_test = C.load_events("test")
    ev = all_test[all_test.test_k.isin(ok_k)].head(n).reset_index(drop=True)
    orig = C.load_events
    monkeypatch.setattr(C, "load_events", lambda stage="test", limit=None, only_selected=False:
                        ev.copy() if stage == "test" else orig(stage, limit, only_selected))
    MockWorld(ev, seed=3).install(monkeypatch)
    assert run_all.main(["--stage", "test", "--limit", str(n), "--root", str(tmp_path), "--no-extras", "--workers", "1"]) == 0
    evf = tmp_path / "ev.npz"
    assert build_events.main(["--stage", "test", "--root", str(tmp_path), "--out", str(evf)]) == 0
    out = tmp_path / "res" / "full_server"
    assert PD.main(["--events", str(evf), "--root", str(tmp_path), "--out", str(out), "--rearm", "--save-streams", "--tag", "_v3"]) == 0
    E = pd.read_csv(out / "peak_detector_events_v3.csv")
    D = pd.read_csv(out / "peak_detector_declarations_v3.csv")
    S = pd.read_csv(out / "peak_detector_v3.csv")
    st = np.load(out / "peak_detector_streams.npz", allow_pickle=False)
    assert len(E) == n and st["xrsb"].shape == (n, 240) and st["xrsa"].shape == (n, 240) and list(st["row"]) == list(E.row)
    assert set(S.series) == {"raw", "cmed3"} and set(S.k) == {1, 2, 3, 5, 10}
    # the declarations can be recomputed from the exported streams alone
    for i, r in enumerate(E.itertuples()):
        for k in (2, 5):
            dec = PD.detect_all(st["xrsb"][i], k, 1e-5)
            d = D[(D.row == r.row) & (D.k == k) & (D.series == "raw")].sort_values("n")
            assert [q["t"] - 60 for q in dec] == list(d.t) and [q["t_max"] - 60 for q in dec] == list(d.t_max)
            o = PD.own_peak(dec, 60, 5)
            v = getattr(r, f"own_raw_k{k}")
            assert (np.isnan(v) and np.isnan(o["own"])) or v == o["own"]
    # mock GOES = 1.15 x FISM2 and the dataset peak is the FISM2 maximum: the flare's own maximum is declared k minutes later
    own2 = E["own_raw_k2"]
    assert own2.notna().mean() >= 0.75 and (own2.dropna() == 2).mean() >= 0.75
    # the rule comparison runs on the exported streams alone (no downloaded files, no prediction arrays)
    import detector_analysis as DA
    assert DA.main(["--results", str(tmp_path / "res"), "--events", str(evf), "--rebuilt", str(tmp_path / "none.npz")]) == 0
    T = pd.read_csv(out / "peak_detector_rules.csv")
    assert set(T.rule) == {"first", "first_rise", "armed2", "armed1.5", "armed1.2"} and (T[T.group == "all"].n == n).all()
    shares = T[T.n > 0][["own_first", "own_after", "other_only", "none"]].sum(axis=1)
    assert np.allclose(shares, 1.0)
    A = T[T.group == "all"].set_index("rule")
    assert A.loc["armed2", "own_declared"] >= A.loc["first", "own_declared"] >= 0.75
    assert (out / "peak_detector_sensitivity.csv").exists() and (out / "peak_detector_missed_v3.csv").exists()
    # first-declaration modes still write their tables
    assert PD.main(["--events", str(evf), "--root", str(tmp_path), "--out", str(out), "--require-rise", "--tag", "_v2"]) == 0
    assert {"lat_raw_k2", "lat_med3_k2"} <= set(pd.read_csv(out / "peak_detector_events_v2.csv").columns)
