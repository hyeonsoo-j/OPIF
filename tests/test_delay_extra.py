"""Paper-1 P7 / P8 (patch_v14): forecasts issued k minutes after the peak for every input (FISM_ISSUE_DELAY,
run.py --issue-delay), GOES minutes after the peak (scripts/build_goes_post.py), predictions for the flares
outside the isolation rule (FISM_PREDICT_EXTRA, run.py --predict-extra) and run.py --models."""
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

from test_pool_full import _synthetic_npz  # noqa: E402

FAST = {"epochs": 1, "patience": 1, "sched_patience": 1}
N_POST = 10


def _dataset(tmp_path, n_not_isolated: int = 30):
    """Synthetic dataset (GOES-B = 1.15 x FISM2, GOES-A = 0.1 x FISM2) with some flares outside the isolation
    rule, and the matching GOES minutes 1..10 after the peak."""
    npz = tmp_path / "events.npz"
    _synthetic_npz(npz)
    d = dict(np.load(npz, allow_pickle=False))
    sel = d["selected"].copy()
    not_iso = np.arange(100, 100 + 3 * n_not_isolated, 3)          # rows of the train/val candidates
    sel[not_iso] = False
    d["selected"] = sel
    np.savez_compressed(npz, **d)
    post = tmp_path / "goes_post.npz"
    yp = d["Y"][:, 1:N_POST + 1, 0]
    np.savez_compressed(post, xrsb_post=1.15 * yp, xrsa_post=0.1 * yp, n_post=np.int64(N_POST), event_id=d["event_id"])
    return npz, post, d, not_iso


def _env(monkeypatch, npz, post, inp="default", delay=0, extra=0):
    for k, v in dict(FISM_PROTOCOL="full", FISM_EVENTS=str(npz), FISM_GOES_POST=str(post), FISM_INPUT=inp,
                     FISM_ISSUE_DELAY=str(delay), FISM_PREDICT_EXTRA=str(extra), FISM_EXCLUDE_NEAR_TEST_H="0",
                     FISM_EXCLUDE_RANDOM_N="0", FISM_EXCLUDE_SAME_AR="0", FISM_DEVICE_QUIET="1").items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("FISM_REQUIRE_GPU", raising=False)
    from src.data import pool
    pool.load_full.cache_clear()
    return pool


def test_build_goes_post(tmp_path):
    """Streams -> GOES minutes after the peak: values, short-gap filling, long gaps stay missing, and the check
    that the streams belong to the dataset."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_goes_post
    npz, _, d, _ = _dataset(tmp_path)
    n = len(d["event_id"])
    keep = np.setdiff1d(np.arange(n), [7])                         # event 7 has no GOES stream at all
    xb, xa = np.full((len(keep), 240), np.nan), np.full((len(keep), 240), np.nan)
    for arr, win, f in ((xb, d["X_goes_b"], 1.15), (xa, d["X_goes_a"], 0.1)):
        arr[:, 1:61] = win[keep]                                   # minutes -59..0 (index 60 = the peak)
        arr[:, 0] = win[keep, 0]
        arr[:, 61:240] = f * d["Y"][keep, 1:180, 0]                # minutes 1..179
    j_short, j_long = 3, 5
    truth_short = xb[j_short, 62:65].copy()
    xb[j_short, 62:65] = np.nan                                    # 3 missing minutes (2..4): filled
    xb[j_long, 63:70] = np.nan                                     # 7 missing minutes (3..9): stay missing
    streams = tmp_path / "streams.npz"
    np.savez_compressed(streams, xrsb=xb, xrsa=xa, row=keep, event_id=d["event_id"][keep], cls=d["cls"][keep],
                        sat=np.full(len(keep), 15), is_test=d["is_test"][keep], pre=np.int64(60), post=np.int64(180))
    out = tmp_path / "post.npz"
    assert build_goes_post.main(["--events", str(npz), "--streams", str(streams), "--out", str(out)]) == 0
    g = np.load(out, allow_pickle=False)
    assert g["xrsb_post"].shape == g["xrsa_post"].shape == (n, N_POST) and int(g["n_post"]) == N_POST
    ok = np.setdiff1d(keep, [keep[j_short], keep[j_long]])
    assert np.allclose(g["xrsb_post"][ok], 1.15 * d["Y"][ok, 1:11, 0], rtol=1e-12)
    assert np.allclose(g["xrsa_post"][keep], 0.1 * d["Y"][keep, 1:11, 0], rtol=1e-12)
    assert np.isnan(g["xrsb_post"][7]).all() and np.isnan(g["xrsa_post"][7]).all()
    short = g["xrsb_post"][keep[j_short]]
    assert np.isfinite(short).all() and np.allclose(short[[0, 4]], 1.15 * d["Y"][keep[j_short], [1, 5], 0])
    assert np.all(np.abs(np.log10(short[1:4] / truth_short)) < 0.05)        # log-linear interpolation of a smooth decay
    long_ = g["xrsb_post"][keep[j_long]]
    assert np.isnan(long_[2:9]).all() and np.isfinite(long_[:2]).all() and np.isfinite(long_[9])
    # streams of another dataset (one input sample differs) are refused
    xb2 = xb.copy()
    xb2[10, 30] *= 1.01
    np.savez_compressed(streams, xrsb=xb2, xrsa=xa, row=keep, event_id=d["event_id"][keep], cls=d["cls"][keep],
                        sat=np.full(len(keep), 15), is_test=d["is_test"][keep], pre=np.int64(60), post=np.int64(180))
    assert build_goes_post.main(["--events", str(npz), "--streams", str(streams), "--out", str(tmp_path / "x.npz")]) == 1
    assert not (tmp_path / "x.npz").exists()


def test_issue_delay_windows(tmp_path, monkeypatch):
    """FISM_ISSUE_DELAY shifts every input window by k minutes: FISM2 inputs with the target minutes, GOES inputs
    with the GOES minutes after the peak, converters fitted on the shifted windows."""
    npz, post, d, _ = _dataset(tmp_path, n_not_isolated=0)
    z = np.load(ROOT / "data/processed/zenodo_test_xy.npz")
    k = 5
    want96 = np.concatenate([z["X_real"][:, k:], z["Y_true"][:, 1:k + 1, 0]], axis=1)
    pool = _env(monkeypatch, npz, post, delay=0)
    X0, Y0, names0, meta0, tr0, va0, te0, teL0 = pool.load_full()
    pool = _env(monkeypatch, npz, post, delay=k)
    X, Y, names, meta, tr, va, te, teL = pool.load_full()
    assert "issued 5 min after the peak" in pool.describe()
    assert np.array_equal(X[:96], want96) and np.array_equal(X[96:192], want96)
    assert np.array_equal(Y, Y0) and names == names0 and np.array_equal(tr, tr0) and np.array_equal(va, va0)
    assert np.array_equal(X[192:, :60 - k], X0[192:, k:]) and np.array_equal(X[192:, 60 - k:], Y[192:, 1:k + 1, 0])
    # GOES XRS-B as the input: minutes 1..k come from the post file, events without them are left out
    g = dict(np.load(post, allow_pickle=False))
    r_tr, r_te = 100, 3                                # dataset rows: a train/val candidate and test event k=3
    g["xrsb_post"][r_tr, 2] = np.nan
    g["xrsb_post"][r_te, 4] = np.nan
    g["xrsb_post"][101, 7] = np.nan                    # minute 8: not needed for k = 5
    np.savez_compressed(post, **g)
    pool = _env(monkeypatch, npz, post, inp="goes_b", delay=0)
    _, _, names_p, _, tr_p, va_p, _, _ = pool.load_full()
    pool = _env(monkeypatch, npz, post, inp="goes_b", delay=k)
    Xg, Yg, names_g, meta_g, tr_g, va_g, te_g, teL_g = pool.load_full()
    assert len(tr_g) + len(va_g) == len(tr) + len(va) - 1 and len(Xg) == len(X) - 1
    # the train/val assignment is the one of the peak-issued pool; the event without minutes 1..k is only dropped
    lost = str(d["event_id"][r_tr])
    assert {names_g[i] for i in tr_g} == {names_p[i] for i in tr_p} - {lost}
    assert {names_g[i] for i in va_g} == {names_p[i] for i in va_p} - {lost}
    assert list(teL_g) == [96 + i for i in range(96) if i != 3] and os.environ["FISM_INPUT_MISSING_TEST"] == "1"
    keep = np.setdiff1d(np.arange(96), [3])
    assert np.allclose(Xg[96:192][keep], 1.15 * want96[keep], rtol=1e-12)
    assert np.allclose(Xg[:96][keep], 1.15 * want96[keep], rtol=1e-12) and np.array_equal(Xg[3], want96[3])
    assert np.isfinite(Xg).all() and (Xg > 0).all() and d["event_id"][r_tr] not in names_g
    # converters: synthetic GOES = 1.15 x FISM2 at every minute -> the shifted FISM2 window is recovered
    for inp in ("goes_conv", "goes_convnn"):
        pool = _env(monkeypatch, npz, post, inp=inp, delay=k)
        Xc = pool.load_full()[0]
        assert np.median(np.abs(np.log10(Xc[96:192][keep]) - np.log10(want96[keep]))) < 0.02, inp
        assert float(os.environ["FISM_CONV_RMSE_DEX"]) < 0.02
    # without the post file the GOES inputs cannot be shifted; the FISM2 input does not need it
    pool = _env(monkeypatch, npz, tmp_path / "missing.npz", inp="goes_b", delay=k)
    with pytest.raises(FileNotFoundError, match="build_goes_post"):
        pool.load_full()
    pool = _env(monkeypatch, npz, tmp_path / "missing.npz", delay=k)
    assert np.array_equal(pool.load_full()[0], X)
    g2 = dict(g)                                       # a post file of another dataset is refused
    g2["event_id"] = g["event_id"][::-1].copy()
    np.savez_compressed(tmp_path / "other.npz", **g2)
    pool = _env(monkeypatch, npz, tmp_path / "other.npz", inp="goes_b", delay=k)
    with pytest.raises(ValueError, match="another dataset"):
        pool.load_full()
    pool = _env(monkeypatch, npz, post, delay=45)
    with pytest.raises(ValueError, match="FISM_ISSUE_DELAY"):
        pool.load_full()
    pool.load_full.cache_clear()


def test_issue_delay_equals_spec_option(tmp_path, monkeypatch):
    """FISM2 input: --issue-delay k trains the same model as the spec option issue_delay: k of paper-1 P3."""
    npz, post, d, _ = _dataset(tmp_path, n_not_isolated=0)
    from src.experiments.cv import run_fold_dl
    spec = {"name": "GRU-z", "type": "dl", "configs": ["gru"], "standardize": True, "train": FAST}
    pool = _env(monkeypatch, npz, post, delay=0)
    _, _, _, _, tr, va, te, teL = pool.load_full()
    tt = np.concatenate([te, teL])
    _, p_spec, _ = run_fold_dl(dict(spec, issue_delay=5), 0, tr, va, tt, 0)
    _, p_plain, _ = run_fold_dl(spec, 0, tr, va, tt, 0)
    pool = _env(monkeypatch, npz, post, delay=5)
    _, _, _, _, tr5, va5, te5, teL5 = pool.load_full()
    assert np.array_equal(tr, tr5) and np.array_equal(va, va5)
    _, p_env, info = run_fold_dl(spec, 0, tr5, va5, tt, 0)
    assert np.allclose(p_env, p_spec, atol=1e-6) and not np.allclose(p_env, p_plain, atol=1e-3)
    assert info["_val_pred"].shape == (len(va), 180, 4)
    with pytest.raises(ValueError, match="shifted twice"):          # both switches at once are refused
        run_fold_dl(dict(spec, issue_delay=5), 0, tr5, va5, tt, 0)
    pool.load_full.cache_clear()


def test_predict_extra(tmp_path, monkeypatch):
    """FISM_PREDICT_EXTRA appends the flares outside the isolation rule after the train/val rows: the split and the
    test predictions do not change, and their predictions are saved as oofN_* with extra_rows.csv."""
    npz, post, d, not_iso = _dataset(tmp_path, n_not_isolated=30)
    pool = _env(monkeypatch, npz, post, extra=0)
    X0, Y0, names0, meta0, tr0, va0, te0, teL0 = pool.load_full()
    assert len(pool.extra_index()[0]) == 0 and len(X0) == 192 + 160 - 30
    assert not set(d["event_id"][not_iso]) & set(names0[192:])
    from src.experiments.cv import run_specs, score_experiment
    specs = [{"name": "GRU-z", "type": "dl", "configs": ["gru"], "standardize": True, "train": FAST},
             {"name": "Power-law", "type": "baseline", "baseline": "powerlaw"}]
    exps = ["pytest_extra0", "pytest_extra1"]
    outs = [ROOT / "results" / e for e in exps]
    for o in outs:
        shutil.rmtree(o, ignore_errors=True)
    try:
        run_specs(exps[0], specs, seeds=(0,), workers=1)
        assert not list(outs[0].glob("oofN_*")) and not (outs[0] / "extra_rows.csv").exists()
        pool = _env(monkeypatch, npz, post, extra=1)
        X, Y, names, meta, tr, va, te, teL = pool.load_full()
        idx, rows = pool.extra_index()
        assert "+ 30 flares outside the isolation rule (predicted only)" in pool.describe()
        assert np.array_equal(tr, tr0) and np.array_equal(va, va0) and np.array_equal(teL, teL0)
        n0 = len(X0)
        assert np.array_equal(X[:n0], X0) and np.array_equal(Y[:n0], Y0) and names[:n0] == names0
        assert len(X) == len(Y) == len(names) == len(meta) == n0 + 30 and list(meta.k) == list(range(len(X)))
        assert list(idx) == list(range(n0, n0 + 30)) and sorted(rows) == sorted(not_iso)
        assert not set(idx) & (set(tr) | set(va) | set(te) | set(teL))
        peaks = pd.to_datetime(d["peak"][rows])
        assert peaks.is_monotonic_increasing                             # appended in time order
        assert np.array_equal(X[idx], d["X"][rows]) and np.array_equal(Y[idx], d["Y"][rows])
        assert [names[i] for i in idx] == [f"{d['event_id'][r]}|extra" for r in rows]
        assert np.allclose(meta.f107_adj.values[idx], d["f107_adj"][rows])
        run_specs(exps[1], specs, seeds=(0,), workers=1)
        E, P, S = score_experiment(exps[1])
        ex = pd.read_csv(outs[1] / "extra_rows.csv")
        assert list(ex.pool_index) == list(idx) and list(ex.dataset_row) == list(rows)
        assert list(ex.event) == [str(d["event_id"][r]) for r in rows]
        for name in ("GRU-z", "Power-law"):
            pn = np.load(outs[1] / f"oofN_{name}_s0.npy")
            assert pn.shape == (30, 180, 4) and np.isfinite(pn).all()
            for pre in ("oof", "oofL"):                                   # test predictions: unchanged
                a, b = np.load(outs[0] / f"{pre}_{name}_s0.npy"), np.load(outs[1] / f"{pre}_{name}_s0.npy")
                assert np.allclose(a, b, atol=1e-6), (name, pre)
        v0, v1 = np.load(outs[0] / "val_GRU-z_s0.npz"), np.load(outs[1] / "val_GRU-z_s0.npz")
        assert np.array_equal(v0["idx"], v1["idx"]) and np.allclose(v0["pred_log"], v1["pred_log"], atol=1e-6)
        # the power-law baseline is a function of the input only: the extra predictions follow the extra inputs
        pl = np.load(outs[1] / "oofN_Power-law_s0.npy")
        assert np.corrcoef(pl[:, 0, 0], np.log10(d["X"][rows, -1]))[0, 1] > 0.99
    finally:
        for o in outs:
            shutil.rmtree(o, ignore_errors=True)
        pool.load_full.cache_clear()


def test_run_py_models_delay_extra(tmp_path):
    """run.py --models / --issue-delay / --predict-extra: experiment folder names and files (fresh interpreters)."""
    npz, post, d, _ = _dataset(tmp_path, n_not_isolated=30)
    cfg = ROOT / "configs" / "pytest_p7.yaml"
    cfg.write_text(
        "exp_id: pytest_p7\nseeds: [0]\nspecs:\n"
        "  - {name: GRU-z, type: dl, configs: [gru], standardize: true, train: {epochs: 1, patience: 1, sched_patience: 1}}\n"
        "  - {name: FCN-z, type: dl, configs: [fcn_w0, fcn_w1, fcn_w2, fcn_w3], standardize: true, train: {epochs: 1, patience: 1, sched_patience: 1}}\n"
        "  - {name: LSTM-z, type: dl, configs: [lstm], standardize: true, train: {epochs: 1, patience: 1, sched_patience: 1}}\n")
    env = dict(os.environ, FISM_GOES_POST=str(post), FISM_DEVICE_QUIET="1", OMP_NUM_THREADS="1")
    for k in ("FISM_REQUIRE_GPU", "FISM_ISSUE_DELAY", "FISM_PREDICT_EXTRA", "FISM_INPUT", "FISM_PROTOCOL"):
        env.pop(k, None)
    base = [sys.executable, "run.py", "--exp", "pytest_p7", "--protocol", "full", "--events", str(npz), "--workers", "2"]
    made = ["pytest_p7_full_in-goes_convnn_d5_p7", "pytest_p7_full_d2_p7", "pytest_p7_full_in-goes_b_p8"]
    try:
        for e in made:
            shutil.rmtree(ROOT / "results" / e, ignore_errors=True)
        r = subprocess.run(base + ["--input", "goes_convnn", "--issue-delay", "5", "--models", "GRU-z,FCN-z", "--seeds", "0,1",
                                   "--tag", "p7"], env=env, cwd=ROOT, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stderr[-3000:]
        out = ROOT / "results" / made[0]
        assert "issued 5 min after the peak" in r.stdout and "4 fold-jobs" in r.stdout
        got = sorted(p.name for p in out.glob("oofL_*"))
        assert got == ["oofL_FCN-z_s0.npy", "oofL_FCN-z_s1.npy", "oofL_GRU-z_s0.npy", "oofL_GRU-z_s1.npy"]
        assert len(list(out.glob("val_*_s*.npz"))) == 4 and not list(out.glob("oofN_*"))
        assert set(pd.read_csv(out / "summary_latis.csv").model) == {"GRU-z", "FCN-z"}
        r = subprocess.run(base + ["--issue-delay", "2", "--models", "GRU-z", "--tag", "p7"], env=env, cwd=ROOT,
                           capture_output=True, text=True, timeout=900)
        assert r.returncode == 0 and (ROOT / "results" / made[1] / "oofL_GRU-z_s0.npy").exists(), r.stderr[-3000:]
        r = subprocess.run(base + ["--input", "goes_b", "--predict-extra", "--models", "GRU-z", "--tag", "p8"], env=env, cwd=ROOT,
                           capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stderr[-3000:]
        out = ROOT / "results" / made[2]
        assert np.load(out / "oofN_GRU-z_s0.npy").shape == (30, 180, 4) and len(pd.read_csv(out / "extra_rows.csv")) == 30
        r = subprocess.run(base + ["--models", "GRU-z,Nope"], env=env, cwd=ROOT, capture_output=True, text=True, timeout=300)
        assert r.returncode != 0 and "Nope" in r.stderr
    finally:
        cfg.unlink(missing_ok=True)
        for e in made:
            shutil.rmtree(ROOT / "results" / e, ignore_errors=True)


def test_analyze_p7_p8_tables(tmp_path, monkeypatch):
    """notebooks/analyze_p7_p8.py on synthetic predictions (truth + noise; the delayed models get less noise): every
    table is written, the pre-specified family has 18 Holm-adjusted comparisons, the delayed models come out better and
    the conformal intervals cover about 90 % of exchangeable test events."""
    import importlib.util
    npz, post, d, not_iso = _dataset(tmp_path, n_not_isolated=30)
    R = tmp_path / "results"
    rng = np.random.default_rng(0)
    spec = importlib.util.spec_from_file_location("ap78", ROOT / "notebooks/analyze_p7_p8.py")
    ap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ap)
    z = np.load(ROOT / "data/processed/zenodo_test_xy.npz")
    y96 = np.log10(z["Y_true"].astype(float))

    def write(exp, inp, k, sigma, extra=False):
        pool = _env(monkeypatch, npz, post, inp="default" if inp == "fism" else "goes_b", delay=0 if inp == "fism" else k, extra=int(extra))
        X, Y, names, meta, tr, va, te, teL = pool.load_full()
        out = R / exp
        out.mkdir(parents=True)
        for m in ap.MODELS:
            for s_ in range(3):
                noise = lambda shape: rng.normal(0, sigma, shape) + rng.normal(0, sigma, (shape[0], 1, 1))      # noqa: E731
                np.save(out / f"oof_{m}_s{s_}.npy", (y96 + noise(y96.shape)).astype(np.float32))
                np.save(out / f"oofL_{m}_s{s_}.npy", (np.log10(Y[96:192]) + noise(y96.shape)).astype(np.float32))
                np.savez_compressed(out / f"val_{m}_s{s_}.npz", idx=va, pred_log=(np.log10(Y[va]) + noise((len(va), 180, 4))).astype(np.float32))
                if extra:
                    idx, rows = pool.extra_index()
                    px = np.log10(Y[idx]) + noise((len(idx), 180, 4)) + 0.05
                    px[np.isin(rows, bump_rows), 90:] -= 0.8          # these forecasts miss a flare that starts 90 min after the peak
                    np.save(out / f"oofN_{m}_s{s_}.npy", px.astype(np.float32))
        if extra:
            idx, rows = pool.extra_index()
            pd.DataFrame(dict(pool_index=idx, dataset_row=rows, event=[str(e) for e in d["event_id"][rows]])).to_csv(out / "extra_rows.csv", index=False)

    # GOES streams of [peak-60, peak+180) for the re-issue tables; ten flares outside the rule get a later flare at +90 min
    n_ev = len(d["event_id"])
    xb = np.empty((n_ev, 240))
    xb[:, 1:61], xb[:, 0], xb[:, 61:] = d["X_goes_b"], d["X_goes_b"][:, 0], 1.15 * d["Y"][:, 1:180, 0]
    quiet = [r for r in not_iso if ap.next_declaration(xb[r]) == 180]          # no later maximum in the source event
    bump_rows = np.array(quiet[:10])
    xb[bump_rows, 150:] *= 30.0
    assert len(bump_rows) == 10 and all(90 <= ap.next_declaration(xb[r]) <= 95 for r in bump_rows)
    streams = tmp_path / "streams.npz"
    np.savez_compressed(streams, xrsb=xb, xrsa=0.1 * xb, row=np.arange(n_ev), event_id=d["event_id"], cls=d["cls"], sat=np.full(n_ev, 15),
                        is_test=d["is_test"], pre=np.int64(60), post=np.int64(180))
    for inp in ap.PEAK:
        write(ap.PEAK[inp], inp, 0, 0.05)
        for k in (2, 5, 10):
            write(ap.p7_exp(inp, k), inp, k, 0.03)
    for inp in ("fism", "goes_convnn"):
        write(ap.p8_exp(inp), inp, 0, 0.05, extra=True)
    out = tmp_path / "tables"
    monkeypatch.setenv("FISM_PROTOCOL", "full")
    assert ap.main(["--results", str(R), "--events", str(npz), "--goes-post", str(post), "--out", str(out), "--n-boot", "300", "--streams", str(streams)]) == 0
    D = pd.read_csv(out / "paper1_p7_delay_hboot.csv")
    fam = D[(D.family == "delayed_vs_peak") & (D.test == "rebuilt96")]
    assert len(fam) == 18 and fam.p_holm.notna().all() and D[D.family != "delayed_vs_peak"].p_holm.isna().all()
    assert (fam.p_holm >= fam.p_hboot - 1e-12).all() and (fam.d < 0).all() and (fam.hi < 0).all()      # less noise -> lower error
    assert set(fam.n_ev) == {96} and len(D) == 68
    Dr = pd.read_csv(out / "paper1_p7_drap_hboot.csv")
    assert len(Dr) == 18 and (Dr.d < 0).all() and Dr.n_day.nunique() == 1
    C = pd.read_csv(out / "paper1_p7_conformal.csv")
    assert len(C) == 42 and C.cov_w0.between(0.82, 0.97).all() and (C.n_val == C.n_val.iloc[0]).all()
    narrower = C[C.issued_min == 10].width_w0_dex.mean() < C[(C.issued_min == 0) & (C.lead_from_min == 10)].width_w0_dex.mean()
    assert narrower
    G = pd.read_csv(out / "paper1_p8_groups.csv")
    assert set(G.group) >= {"all", "M class"} and (G[G.group == "all"].n == 30).all() and (G.d_vs_test > 0).all()      # the shifted flares are worse
    assert (G.cov_w0 < G.cov_w0_test).all()
    Ev = pd.read_csv(out / "paper1_p8_events.csv")
    assert len(Ev) == 30 and sorted(Ev.dataset_row) == sorted(not_iso) and Ev.filter(like="MAPE4_").notna().all().all()
    assert len(pd.read_csv(out / "paper1_p8_blocks.csv")) % 6 == 0
    # recalibration for all flares: the flares outside the rule carry a +0.05 dex offset, so the isolated-only intervals
    # under-cover them and the all-flare calibration (out-of-fold) moves their coverage up and widens the interval
    Rc = pd.read_csv(out / "paper1_p8_recal.csv")
    assert set(Rc.calibration) == {"isolated only", "all flares (weighted)", "all flares (weighted, folds = time halves)", "all flares (unweighted)"}
    for (inp, m), q in Rc.groupby(["input", "model"]):
        g = lambda cal, grp, col: float(q[(q.calibration == cal) & (q.group == grp)][col].iloc[0])      # noqa: E731
        iso, new_ = g("isolated only", "outside the rule: all", "cov_w0"), g("all flares (weighted)", "outside the rule: all", "cov_w0")
        assert new_ > iso and g("all flares (weighted)", "test flares (isolated)", "width_w0_dex") > g("isolated only", "test flares (isolated)", "width_w0_dex")
        assert g("all flares (unweighted)", "outside the rule: all", "cov_w0") >= new_ - 0.02          # more weight on the shifted flares
        mixv = (1 - ap.P_OUTSIDE) * g("all flares (weighted)", "test flares (isolated)", "cov_w0") + ap.P_OUTSIDE * new_
        assert abs(mixv - g("all flares (weighted)", "all flares (population-weighted)", "cov_w0")) < 1e-9
        assert abs(g("isolated only", "test flares (isolated)", "cov_w0") - float(G[(G.input == inp) & (G.model == m)].cov_w0_test.iloc[0])) < 1e-9
        assert (q.cov_w0_lo <= q.cov_w0 + 1e-9).all() and (q.cov_w0 <= q.cov_w0_hi + 1e-9).all()
    Rb = pd.read_csv(out / "paper1_p8_recal_blocks.csv")
    assert len(Rb) == 2 * 2 * 4 * 6 and Rb.cov_w0_all.between(0, 1).all()
    # re-issue at the next declaration: the ten forecasts that miss the later flare are scored only until it is declared
    Ev2 = pd.read_csv(out / "paper1_p8_reissue_events.csv").set_index("dataset_row")
    assert Ev2.loc[bump_rows, "T_next_r2"].between(90, 95).all() and (Ev2.loc[bump_rows, "T_next_r1.2"] <= Ev2.loc[bump_rows, "T_next_r2"]).all()
    assert (Ev2.loc[np.array([r for r in quiet if r not in set(bump_rows)]), "T_next_r2"] == 180).all()
    Ri = pd.read_csv(out / "paper1_p8_reissue.csv")
    assert set(Ri.scoring) == {"all 180 min", "minutes in force"} and set(Ri.rise_factor) == {2.0, 1.5, 1.2}
    for col in ("MAPE4", "cov_w0"):                                   # every bootstrap interval contains its point value
        assert (Ri[f"{col}_lo"] <= Ri[col] + 1e-9).all() and (Ri[col] <= Ri[f"{col}_hi"] + 1e-9).all(), col
        assert (Ri[f"{col}_hi"] > Ri[f"{col}_lo"]).all(), col
    for (inp, m), q in Ri[Ri.rise_factor == 2.0].groupby(["input", "model"]):
        f = lambda sc, grp, col: float(q[(q.scoring == sc) & (q.group == grp)][col].iloc[0])      # noqa: E731
        out_ = "outside the rule: all"
        assert f("minutes in force", out_, "cov_w0") > f("all 180 min", out_, "cov_w0") + 0.05
        assert f("minutes in force", out_, "MAPE4") < f("all 180 min", out_, "MAPE4") - 5
        assert abs(f("all 180 min", out_, "cov_w0") - float(G[(G.input == inp) & (G.model == m) & (G.group == "all")].cov_w0.iloc[0])) < 1e-9
        assert abs(f("all 180 min", out_, "MAPE4") - float(G[(G.input == inp) & (G.model == m) & (G.group == "all")].MAPE4.iloc[0])) < 1e-6
        share = f("minutes in force", out_, "share_with_next_declaration")
        assert share >= 10 / 30 - 1e-9 and f("minutes in force", out_, "share_minutes_in_force") < 1 and f("all 180 min", out_, "share_minutes_in_force") == 1
        lo_, hi_ = f("minutes in force", "all flares (population-weighted)", "cov_w0_lo"), f("minutes in force", "all flares (population-weighted)", "cov_w0_hi")
        assert lo_ <= f("minutes in force", "all flares (population-weighted)", "cov_w0") <= hi_
    Tl = pd.read_csv(out / "paper1_p8_reissue_tail.csv")
    t_ = Tl[(Tl.group == "outside the rule: all") & (Tl.input == "goes_convnn") & (Tl.model == "GRU-z")].set_index("minutes")
    assert t_.loc["last 10 minutes before the next declaration", "cov_w0"] < t_.loc["earlier minutes in force", "cov_w0"]      # minutes 90, 91 are missed
    assert len(pd.read_csv(out / "paper1_p8_reissue_blocks.csv")) == 2 * 2 * 2 * 6
    assert (out / "paper1_p8_reissue_aligned.csv").exists()          # written even when no group has 20 flares with a next declaration
    from src.data import pool
    pool.load_full.cache_clear()
