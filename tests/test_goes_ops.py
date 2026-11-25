"""paper-1 P11 / P12 (patch_v16): operational GOES XRS files (scripts/download/fetch_goes_xrs_ops.py), operational input windows
(scripts/build_goes_ops.py), twin rows with the operational input (FISM_PREDICT_OPS, run.py --predict-ops) and the probe of the
latest FISM2 time stamp (scripts/probe_fism2_latest.py)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/download"))
from test_delay_extra import _dataset, _env  # noqa: E402

PRE, POST, L = 60, 180, 2
FACTOR = {12: 0.97, 15: 1.0, 16: 1.02}          # operational / science (true units) in the synthetic archive
STAMP_SHIFT = {12: 1, 15: 0, 16: 0}             # GOES-12 files stamp every minute one minute late -> lag +1


def _minute(ts) -> int:
    return int(pd.Timestamp(ts).value // 60_000_000_000)


def _write_legacy_nc(path: Path, minutes, a, b, bflag):
    import netCDF4
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("record", len(minutes))
        t = ds.createVariable("time_tag", "f8", ("record",))
        t.units = "milliseconds since 1970-01-01 00:00:00.0 UTC"
        t[:] = np.asarray(minutes, float) * 60000.0
        for name, v in (("A_AVG", a), ("B_AVG", b)):
            x = ds.createVariable(name, "f4", ("record",))
            x[:] = np.where(np.isfinite(v), v, -99999.0)
        for name, v in (("A_QUAL_FLAG", np.zeros(len(minutes))), ("B_QUAL_FLAG", bflag)):
            x = ds.createVariable(name, "i4", ("record",))
            x[:] = v.astype(int)


def _write_legacy_csv(path: Path, minutes, a, b):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#Institution: test", 'title = "GOES XRS 1-minute averages"', "data:", "time_tag,xs,xl"]
    for m, va, vb in zip(minutes, a, b):
        ts = pd.Timestamp(int(m) * 60, unit="s").strftime("%Y-%m-%d %H:%M:%S.000")
        lines.append(f"{ts},{va if np.isfinite(va) else -99999.0:.6e},{vb if np.isfinite(vb) else -99999.0:.6e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_goesr_nc(path: Path, minutes, a, b, bflag):
    import netCDF4
    path.parent.mkdir(parents=True, exist_ok=True)
    ref = _minute("2000-01-01T12:00")
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", len(minutes))
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "seconds since 2000-01-01 12:00:00"
        t[:] = (np.asarray(minutes, float) - ref) * 60.0
        for name, v in (("xrsa_flux", a), ("xrsb_flux", b)):
            x = ds.createVariable(name, "f4", ("time",), fill_value=-9999.0)
            x[:] = np.ma.masked_invalid(v)
        for name, v in (("xrsa_flag", np.zeros(len(minutes))), ("xrsb_flag", bflag)):
            x = ds.createVariable(name, "i4", ("time",))
            x[:] = v.astype(int)


def _world(tmp_path):
    """Dataset with three satellite generations, science streams, and an operational archive for well-separated flares."""
    npz, post, d, not_iso = _dataset(tmp_path, n_not_isolated=30)
    n = len(d["event_id"])
    d["goes_sat"] = np.array([12, 15, 16])[np.arange(n) % 3]
    np.savez_compressed(npz, **d)
    keep = np.setdiff1d(np.arange(n), [7])                               # flare 7 has no science stream
    xb, xa = np.full((len(keep), PRE + POST), np.nan), np.full((len(keep), PRE + POST), np.nan)
    for arr, win, f in ((xb, d["X_goes_b"], 1.15), (xa, d["X_goes_a"], 0.1)):
        arr[:, 1:61] = win[keep]
        arr[:, 0] = win[keep, 0]
        arr[:, 61:] = f * d["Y"][keep, 1:180, 0]
    streams = tmp_path / "streams.npz"
    np.savez_compressed(streams, xrsb=xb, xrsa=xa, row=keep, event_id=d["event_id"][keep], cls=d["cls"][keep], sat=d["goes_sat"][keep],
                        is_test=d["is_test"][keep], pre=np.int64(PRE), post=np.int64(POST))
    # flares with an operational archive: peaks more than two days apart (no shared minutes)
    pk = np.array([_minute(str(p)) for p in d["peak"][keep]])
    chosen, last = [], -10 ** 12
    for j in np.argsort(pk):                                             # ... and no other flare of the dataset within 5 h
        if pk[j] - last > 2 * 1440 and np.sort(np.abs(pk - pk[j]))[1] > 300:
            chosen.append(int(j))
            last = pk[j]
    series = {}                                                          # (sat, minute) -> (a, b, flag) as stored in the files
    special = {}
    for j in chosen:
        s = int(d["goes_sat"][keep[j]])
        t = pk[j] - PRE - L - 5 + np.arange(PRE + POST + 2 * L + 10)      # true minutes, a little beyond what is read
        rel = t - (pk[j] - PRE)
        b = np.where(rel < 0, xb[j, 0], np.where(rel >= PRE + POST, xb[j, -1], xb[j, np.clip(rel, 0, PRE + POST - 1)]))
        a = np.where(rel < 0, xa[j, 0], np.where(rel >= PRE + POST, xa[j, -1], xa[j, np.clip(rel, 0, PRE + POST - 1)]))
        scale_b, scale_a = (0.7, 0.85) if s <= 15 else (1.0, 1.0)
        for tt, va, vb in zip(t, a, b):
            series[(s, int(tt) + STAMP_SHIFT[s])] = [FACTOR[s] * va * scale_a, FACTOR[s] * vb * scale_b, 0]
    by_sat = {s: [j for j in chosen if int(d["goes_sat"][keep[j]]) == s] for s in (12, 15, 16)}
    j_flag, j_long, j_fill = by_sat[15][0], by_sat[15][1], by_sat[12][0]
    series[(15, pk[j_flag] - 20)][2] = 1                                  # one flagged minute: filled
    for q in range(30, 37):
        series[(15, pk[j_long] - q)][2] = 1                               # seven flagged minutes: the window stays incomplete
    series[(12, pk[j_fill] - 10 + STAMP_SHIFT[12])][1] = np.nan           # fill value in the text file: filled
    special.update(flag=keep[j_flag], long=keep[j_long], fill=keep[j_fill])
    root = tmp_path / "dl"
    files = {}
    for (s, m), v in series.items():
        day = pd.Timestamp(m * 60, unit="s").normalize()
        key = (s, day.replace(day=1)) if s <= 15 else (s, day)
        files.setdefault(key, []).append((m, *v))
    for (s, day), recs in files.items():
        recs.sort()
        m, a, b, fl = (np.array(x) for x in zip(*recs))
        ddir = root / "goes_ops" / f"g{s:02d}" / f"{day.year}"
        last_day = (day + pd.offsets.MonthEnd(0)).strftime("%Y%m%d")
        if s == 15:
            _write_legacy_nc(ddir / f"g15_xrs_1m_{day.strftime('%Y%m')}01_{last_day}.nc", m, a, b, fl)
        elif s == 12:
            _write_legacy_csv(ddir / f"g12_xrs_1m_{day.strftime('%Y%m')}01_{last_day}.csv", m, a, b)
        else:
            _write_goesr_nc(ddir / f"dn_xrsf-l2-avg1m_g16_d{day.strftime('%Y%m%d')}_v2-2-2.nc", m, a, b, fl)
    return dict(npz=npz, post=post, d=d, keep=keep, streams=streams, root=root, chosen=[int(keep[j]) for j in chosen], special=special, xb=xb)


def test_build_goes_ops(tmp_path, capsys):
    import build_goes_ops
    import fetch_goes_xrs_ops as O
    w = _world(tmp_path)
    d, out, rep = w["d"], tmp_path / "ops.npz", tmp_path / "rep" / "goes_ops_vs_science.csv"
    rc = build_goes_ops.main(["--events", str(w["npz"]), "--streams", str(w["streams"]), "--root", str(w["root"]), "--out", str(out), "--report", str(rep)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "B_AVG" in text and "xrsb_flux" in text and "time_tag,xs,xl" in text          # one line per satellite on what its files hold
    g = np.load(out, allow_pickle=False)
    n = len(d["event_id"])
    assert g["xrsb_ops"].shape == g["xrsa_ops"].shape == g["xrsb_ops_al"].shape == (n, 60) and np.array_equal(g["event_id"], d["event_id"])
    assert dict(zip(g["lag_sat"].tolist(), g["lag_min"].tolist())) == {12: 1, 15: 0, 16: 0} and bool(g["has_aligned"])
    assert g["wide_b"].shape == (len(w["keep"]), PRE + POST + 2 * L) and int(g["max_lag"]) == L
    sp = w["special"]
    plain = [i for i in w["chosen"] if i not in sp.values()]
    for i in plain:
        s = int(d["goes_sat"][i])
        want_b, want_a = FACTOR[s] * d["X_goes_b"][i], FACTOR[s] * d["X_goes_a"][i]
        assert np.allclose(g["xrsb_ops_al"][i], want_b, rtol=2e-5) and np.allclose(g["xrsa_ops_al"][i], want_a, rtol=2e-5), (i, s)
        if s == 12:                                                   # as archived, the GOES-12 windows are one minute late
            assert np.allclose(g["xrsb_ops"][i][1:], want_b[:-1], rtol=2e-5) and not np.allclose(g["xrsb_ops"][i], want_b, rtol=1e-3)
        else:
            assert np.allclose(g["xrsb_ops"][i], want_b, rtol=2e-5)
    # a flagged minute and a fill value are closed by the short-gap rule; seven flagged minutes leave the window incomplete
    assert np.isfinite(g["xrsb_ops"][sp["flag"]]).all() and np.allclose(g["xrsb_ops"][sp["flag"]], d["X_goes_b"][sp["flag"]], rtol=0.2)
    assert np.isfinite(g["xrsb_ops_al"][sp["fill"]]).all()
    assert np.isnan(g["xrsb_ops"][sp["long"]]).sum() == 7
    # flares without an operational file or without a science stream have no window
    nofile = np.setdiff1d(np.arange(n), w["chosen"])
    assert np.isnan(g["xrsb_ops"][nofile]).all() and np.isnan(g["xrsb_ops"][7]).all()
    R = pd.read_csv(rep).set_index("group")
    assert R.loc["all", "n_flares"] == len(w["keep"]) and R.loc["all", "n_ops_complete"] == len(w["chosen"]) - 1
    assert abs(R.loc["GOES 13-15", "share_minutes_flagged"] - 8 / (60 * (R.loc["GOES 13-15", "n_ops_any_minute"]))) < 1e-9 and R.loc["GOES 16-18", "share_minutes_flagged"] == 0
    assert abs(R.loc["GOES 16-18", "median_dlog"] - np.log10(1.02)) < 1e-4 and abs(R.loc["GOES 13-15", "median_dlog"]) < 1e-4
    assert R.loc["GOES-12", "lag_min"] == 1 and R.loc["GOES-12", "cost_lag+1"] < R.loc["GOES-12", "cost_lag+0"]
    assert abs(R.loc["GOES 8-12", "median_abs_dlog_aligned"] - abs(np.log10(0.97))) < 1e-4
    E = pd.read_csv(rep.with_name("goes_ops_vs_science_events.csv"))
    assert len(E) == len(w["keep"]) and E.ops_window_complete.sum() == len(w["chosen"]) - 1
    # true units: GOES 8-15 values are divided by the SWPC factors, GOES-R values are not
    i15 = next(i for i in plain if d["goes_sat"][i] == 15)
    p = _minute(str(d["peak"][i15]))
    raw = O.read_legacy_nc(O.find_local(w["root"], 15, pd.Timestamp(p * 60, unit="s")))
    assert np.isclose(raw.loc[p, "xrsb"], 0.7 * d["X_goes_b"][i15][-1], rtol=2e-5)
    assert np.isclose(O.load_series_ops(w["root"], 15, p, 1)["xrsb"][0], d["X_goes_b"][i15][-1], rtol=2e-5)
    # another dataset: refused
    d2 = dict(d)
    d2["event_id"] = d["event_id"][::-1].copy()
    np.savez_compressed(tmp_path / "other.npz", **d2)
    assert build_goes_ops.main(["--events", str(tmp_path / "other.npz"), "--streams", str(w["streams"]), "--root", str(w["root"]),
                                "--out", str(tmp_path / "x.npz"), "--report", str(rep)]) == 1


def test_reader_falls_back_to_the_text_file(tmp_path):
    """A netCDF file that cannot be read does not stop the build: the text file of the same month is used and the reason is kept."""
    import fetch_goes_xrs_ops as O
    O._read_cached.cache_clear()
    O.READ_ERRORS.clear()
    p = _minute("2014-06-10T12:00")
    m = p - 30 + np.arange(60)
    ddir = tmp_path / "goes_ops" / "g15" / "2014"
    _write_legacy_csv(ddir / "g15_xrs_1m_20140601_20140630.csv", m, np.full(60, 0.85e-7), np.full(60, 0.7e-5))
    (ddir / "g15_xrs_1m_20140601_20140630.nc").write_bytes(b"this is not a netCDF file")
    s = O.load_series_ops(tmp_path, 15, p - 30, 60)
    assert s is not None and np.allclose(s["xrsb"], 1e-5, rtol=1e-6) and np.allclose(s["xrsa"], 1e-7, rtol=1e-6)      # true units
    assert list(O.READ_ERRORS) == ["g15_xrs_1m_20140601_20140630.nc"]
    # nothing readable at all -> no series, no exception
    O._read_cached.cache_clear()
    (ddir / "g15_xrs_1m_20140601_20140630.csv").write_text("no header here\n1,2,3\n", encoding="utf-8")
    assert O.load_series_ops(tmp_path, 15, p - 30, 60) is None and len(O.READ_ERRORS) == 2
    O._read_cached.cache_clear()
    O.READ_ERRORS.clear()


def test_best_lag_ties_and_empty():
    import build_goes_ops as B
    assert B.best_lag({-1: 0.2, 0: 0.2, 1: 0.2}) == 0 and B.best_lag({-2: np.nan, 0: np.nan}) == 0
    assert B.best_lag({-2: 0.1, -1: 0.3, 0: 0.3, 1: 0.1, 2: 0.3}) == 1 and B.best_lag({-1: 0.05, 0: 0.2, 1: 0.05}) == -1


def test_fetch_goes_xrs_ops_offline(tmp_path, monkeypatch):
    """Listings -> file choice (netCDF and text for GOES 8-15, newest version for GOES-R), one request per file and month for
    GOES 8-15, resume without a second download, satellite-months without a file are reported."""
    import common as C
    import fetch_goes_xrs_ops as O
    npz = tmp_path / "ev.npz"
    np.savez_compressed(npz, event_id=np.array(["a", "b", "c", "d", "e"]), goes_sat=np.array([15, 15, 12, 16, -1]),
                        peak=np.array(["2014-06-10T00:30", "2014-06-20T12:00", "2005-09-07T17:40", "2022-05-10T13:55", "2010-01-01T00:00"]))
    LB, GB = "http://legacy.test/avg", "http://goesr.test/goes"
    listing = lambda names: "<html>" + "".join(f'<a href="{n}">{n}</a>' for n in names) + "</html>"      # noqa: E731
    web = {f"{LB}/2014/06/goes15/netcdf/": listing(["g15_xrs_1m_20140601_20140630.nc", "g15_epead_p17ew_1m_20140601_20140630.nc"]),
           f"{LB}/2014/06/goes15/csv/": listing(["g15_xrs_1m_20140601_20140630.csv"]),
           f"{LB}/2014/06/goes15/netcdf/g15_xrs_1m_20140601_20140630.nc": b"NC15",
           f"{LB}/2014/06/goes15/csv/g15_xrs_1m_20140601_20140630.csv": b"CSV15",
           f"{LB}/2005/09/goes12/csv/": listing(["g12_xrs_1m_20050901_20050930.csv", "g12_xrs_5m_20050901_20050930.csv"]),
           f"{LB}/2005/09/goes12/csv/g12_xrs_1m_20050901_20050930.csv": b"CSV12",
           f"{GB}/goes16/l2/data/xrsf-l2-avg1m/2022/05/": listing(["dn_xrsf-l2-avg1m_g16_d20220510_v2-2-0.nc", "dn_xrsf-l2-avg1m_g16_d20220510_v2-2-2.nc",
                                                                    "dn_xrsf-l2-avg1m_g16_d20220511_v2-2-2.nc"]),
           f"{GB}/goes16/l2/data/xrsf-l2-avg1m/2022/05/dn_xrsf-l2-avg1m_g16_d20220510_v2-2-2.nc": b"NC16"}
    calls = []

    def fake_bytes(url, timeout=120.0):
        calls.append(url)
        v = web.get(url)
        return None if v is None else (v if isinstance(v, bytes) else v.encode())

    monkeypatch.setattr(C, "fetch_bytes", fake_bytes)
    monkeypatch.setattr(C, "fetch_text", lambda url, timeout=120.0: None if fake_bytes(url) is None else fake_bytes(url).decode())
    # the flare just after midnight needs the previous month as well (2014-05-..: not there -> reported, not an error)
    pairs = O.needed(str(npz), 70, 190)
    assert (15, pd.Timestamp("2014-06-09")) in pairs and (15, pd.Timestamp("2014-06-10")) in pairs and all(s >= 0 for s, _ in pairs)
    args = ["--events", str(npz), "--root", str(tmp_path / "dl"), "--legacy-base", LB, "--goesr-base", GB, "--workers", "1", "--min-interval", "0"]
    assert O.main(args) == 0
    got = sorted(p.name for p in (tmp_path / "dl" / "goes_ops").rglob("*") if p.is_file() and p.parent.name != "listings")
    assert got == ["dn_xrsf-l2-avg1m_g16_d20220510_v2-2-2.nc", "g12_xrs_1m_20050901_20050930.csv", "g15_xrs_1m_20140601_20140630.csv",
                   "g15_xrs_1m_20140601_20140630.nc"]                                       # GOES 8-15: netCDF and text, both kept
    assert O.find_local(tmp_path / "dl", 15, pd.Timestamp("2014-06-10")).suffix == ".nc"   # the netCDF file is read first
    assert (tmp_path / "dl" / "goes_ops" / "g15" / "2014" / "g15_xrs_1m_20140601_20140630.nc").read_bytes() == b"NC15"
    assert sum(u.endswith("g15_xrs_1m_20140601_20140630.nc") for u in calls) == 1          # two flares, one monthly file
    n0 = len(calls)
    assert O.main(args) == 0 and len(calls) == n0                                           # resume: nothing is requested again
    assert O.find_local(tmp_path / "dl", 16, pd.Timestamp("2022-05-10")).name.endswith("v2-2-2.nc")
    assert O.find_local(tmp_path / "dl", 16, pd.Timestamp("2022-05-11")) is None


def _ops_npz(path, d, factor_asis=1.1, missing=()):
    """Operational windows = factor x science with as-archived stamps, exactly the science windows when aligned."""
    b, a = d["X_goes_b"].astype(float).copy(), d["X_goes_a"].astype(float).copy()
    for i in missing:
        b[i, 30] = np.nan
    np.savez_compressed(path, xrsb_ops=factor_asis * b, xrsa_ops=factor_asis * a, xrsb_ops_al=b, xrsa_ops_al=a, has_aligned=np.bool_(True),
                        event_id=d["event_id"].astype(str))


def test_pool_ops_twins(tmp_path, monkeypatch):
    npz, post, d, not_iso = _dataset(tmp_path, n_not_isolated=30)
    ops = tmp_path / "ops.npz"
    test_rows = np.where(d["is_test"])[0]
    missing = [int(test_rows[3]), int(not_iso[0]), int(not_iso[5])]
    _ops_npz(ops, d, missing=missing)
    monkeypatch.setenv("FISM_GOES_OPS", str(ops))
    monkeypatch.delenv("FISM_PREDICT_OPS", raising=False)
    monkeypatch.delenv("FISM_TRAIN_EXTRA_FOLD", raising=False)
    for inp in ("goes_b", "goes_convnn"):
        pool = _env(monkeypatch, npz, post, inp=inp, extra=1)
        X0, Y0, names0, meta0, tr0, va0, te0, teL0 = pool.load_full()
        assert len(pool.ops_index()["idx"]) == 0
        monkeypatch.setenv("FISM_PREDICT_OPS", "1")
        pool.load_full.cache_clear()
        X, Y, names, meta, tr, va, te, teL = pool.load_full()
        o = pool.ops_index()
        k = len(X0)
        assert np.array_equal(X[:k], X0) and np.array_equal(Y[:k], Y0) and names[:k] == names0 and len(meta) == len(X)
        assert np.array_equal(tr, tr0) and np.array_equal(va, va0) and np.array_equal(te, te0) and np.array_equal(teL, teL0)
        n_t, n_x = 96 - 1, 30 - 2
        assert len(o["idx"]) == 2 * (n_t + n_x) and np.array_equal(o["idx"], k + np.arange(len(o["idx"])))
        assert not set(o["idx"]) & (set(tr) | set(va) | set(te) | set(teL))
        for v in ("asis", "aligned"):
            m = o["variant"] == v
            assert (o["kind"][m] == "test").sum() == n_t and (o["kind"][m] == "outside").sum() == n_x
        assert not set(missing) & set(o["rows"].tolist())
        assert np.array_equal(Y[o["idx"]], Y[o["twin"]])
        al, asis = o["variant"] == "aligned", o["variant"] == "asis"
        assert np.allclose(X[o["idx"][al]], X[o["twin"][al]], rtol=1e-12)                  # same input -> same converted input
        if inp == "goes_b":
            assert np.allclose(X[o["idx"][asis]], 1.1 * X[o["twin"][asis]], rtol=1e-12)
        else:
            assert not np.allclose(X[o["idx"][asis]], X[o["twin"][asis]], rtol=1e-3)
        assert all(names[i].endswith(f"|ops-{v}") and names[i].split("|")[0] == names[t].split("|")[0] for i, t, v in zip(o["idx"], o["twin"], o["variant"]))
        ex_idx, ex_rows = pool.extra_index()
        outside = o["kind"] == "outside"
        assert set(o["twin"][outside]) <= set(ex_idx) and set(o["rows"][outside]) <= set(ex_rows) and set(o["twin"][~outside]) <= set(teL)
        assert "twin rows with the operational GOES input" in pool.describe()
        monkeypatch.setenv("FISM_PREDICT_OPS", "0")
    # refused: not a GOES input, a delayed issue, a missing file, another dataset
    monkeypatch.setenv("FISM_PREDICT_OPS", "1")
    pool = _env(monkeypatch, npz, post, inp="default", extra=1)
    with pytest.raises(ValueError):
        pool.load_full()
    pool = _env(monkeypatch, npz, post, inp="goes_b", delay=2, extra=1)
    with pytest.raises(ValueError):
        pool.load_full()
    monkeypatch.setenv("FISM_GOES_OPS", str(tmp_path / "nope.npz"))
    pool = _env(monkeypatch, npz, post, inp="goes_b", extra=1)
    with pytest.raises(FileNotFoundError):
        pool.load_full()
    d2 = dict(d)
    d2["event_id"] = d["event_id"][::-1].copy()
    _ops_npz(tmp_path / "other.npz", d2)
    monkeypatch.setenv("FISM_GOES_OPS", str(tmp_path / "other.npz"))
    pool.load_full.cache_clear()
    with pytest.raises(ValueError):
        pool.load_full()
    monkeypatch.setenv("FISM_PREDICT_OPS", "0")
    pool.load_full.cache_clear()


def test_run_py_predict_ops(tmp_path):
    """run.py --predict-ops: folder name, oofO / ops_rows.csv, and the same model gives the same forecast for the same input."""
    npz, post, d, not_iso = _dataset(tmp_path, n_not_isolated=30)
    ops = tmp_path / "ops.npz"
    _ops_npz(ops, d, missing=[int(not_iso[0])])
    cfg = ROOT / "configs" / "pytest_p11.yaml"
    cfg.write_text("exp_id: pytest_p11\nseeds: [0]\nspecs:\n"
                   "  - {name: GRU-z, type: dl, configs: [gru], standardize: true, train: {epochs: 1, patience: 1, sched_patience: 1}}\n")
    env = dict(os.environ, FISM_GOES_POST=str(post), FISM_DEVICE_QUIET="1", OMP_NUM_THREADS="1")
    for k in ("FISM_REQUIRE_GPU", "FISM_ISSUE_DELAY", "FISM_PREDICT_EXTRA", "FISM_INPUT", "FISM_PROTOCOL", "FISM_PREDICT_OPS", "FISM_GOES_OPS", "FISM_TRAIN_EXTRA_FOLD"):
        env.pop(k, None)
    base = [sys.executable, "run.py", "--exp", "pytest_p11", "--protocol", "full", "--events", str(npz), "--workers", "1"]
    made = "pytest_p11_full_in-goes_convnn_ops_p11"
    try:
        shutil.rmtree(ROOT / "results" / made, ignore_errors=True)
        r = subprocess.run(base + ["--input", "goes_convnn", "--predict-extra", "--predict-ops", "--goes-ops", str(ops), "--tag", "p11"],
                           env=env, cwd=ROOT, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stderr[-3000:]
        assert "twin rows with the operational GOES input" in r.stdout
        out = ROOT / "results" / made
        T = pd.read_csv(out / "ops_rows.csv")
        E = pd.read_csv(out / "extra_rows.csv")
        O = np.load(out / "oofO_GRU-z_s0.npy")
        Ltest, Nout = np.load(out / "oofL_GRU-z_s0.npy"), np.load(out / "oofN_GRU-z_s0.npy")
        assert O.shape == (len(T), 180, 4) and len(T) == 2 * (96 + 29) and np.isfinite(O).all()
        assert set(T.variant) == {"asis", "aligned"} and set(T.kind) == {"test", "outside"}
        pos = {int(p): j for j, p in enumerate(E.pool_index)}
        for j, t in enumerate(T.itertuples()):
            twin = Ltest[t.twin_pool_index - 96] if t.kind == "test" else Nout[pos[int(t.twin_pool_index)]]
            if t.variant == "aligned":                 # identical input -> identical forecast
                assert np.allclose(O[j], twin, atol=2e-5), (t.event, np.abs(O[j] - twin).max())
        asis = (T.variant == "asis").to_numpy()
        tw = np.stack([Ltest[t.twin_pool_index - 96] if t.kind == "test" else Nout[pos[int(t.twin_pool_index)]] for t in T.itertuples()])
        assert np.abs(O[asis] - tw[asis]).max() > 1e-3   # a 10 % higher input changes the forecast
        r = subprocess.run(base + ["--predict-ops", "--tag", "p11"], env=env, cwd=ROOT, capture_output=True, text=True, timeout=300)
        assert r.returncode != 0 and "--predict-ops needs" in r.stderr
    finally:
        cfg.unlink(missing_ok=True)
        shutil.rmtree(ROOT / "results" / made, ignore_errors=True)


def test_probe_fism2_latest(tmp_path, monkeypatch):
    """The probe finds the most recent day with data (here: 9 days before the run day) and its last minute."""
    import common as C
    import probe_fism2_latest as P
    today = pd.Timestamp("2026-11-20")
    last_min = C.to_minute(today - pd.Timedelta(days=9)) + 17 * 60 + 5        # the product ends at 17:05 UT nine days ago
    asked = []

    def fake_text(url, timeout=120.0):
        import re
        q = dict(re.findall(r"(time>=|time<)([^&]+)", url))
        t0, t1 = C.to_minute(q["time>="]), C.to_minute(q["time<"])
        asked.append((t0, t1))
        rows = ["time,wavelength,irradiance"]
        for m in range(t0, min(t1, last_min + 1)):
            rows.append(f"{pd.Timestamp(m * 60, unit='s').isoformat()},0.15,1.0e-6")
        return "\n".join(rows) + "\n"

    monkeypatch.setattr(C, "fetch_text", fake_text)
    out = tmp_path / "probe.csv"
    assert P.main(["--out", str(out), "--today", "2026-11-20", "--pause", "0"]) == 0
    T = pd.read_csv(out)
    s = T[T.kind == "summary"].iloc[0]
    assert s.latest_day_with_data == "2026-11-11" and int(s.days_behind_run_day) == 9 and s.last_time_stamp == "2026-11-11T17:05"
    days = T[T.kind == "day"]
    assert set(days.days_before_run) == {0, 1, 2, 3, 5, 7, 10, 8, 9} and (days[days.days_before_run < 9].n_minutes_with_data == 0).all()
    # nothing anywhere: reported, not an error
    monkeypatch.setattr(C, "fetch_text", lambda url, timeout=120.0: "time,wavelength,irradiance\n")
    assert P.main(["--out", str(out), "--today", "2026-11-20", "--pause", "0"]) == 0
    s = pd.read_csv(out).query("kind == 'summary'").iloc[0]
    assert pd.isna(s.latest_day_with_data) and "no data" in s.note
