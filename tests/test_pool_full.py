"""Smoke test of the 'full' protocol (src/data/pool.py + cv.run_specs) on a synthetic events npz with the
same layout as scripts/download/build_events.py output."""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _synthetic_npz(path: Path, n_rest: int = 160, seed: int = 0):
    z = np.load(ROOT / "data/processed/zenodo_test_xy.npz")
    X96, Y96, ev96 = z["X_real"].astype(float), z["Y_true"].astype(float), [str(e) for e in z["event_names"]]
    cat = pd.read_csv(ROOT / "data/processed/flare_catalogue.csv")
    rng = np.random.default_rng(seed)
    rest = cat[cat.selected & ~cat.event_id.isin(ev96)].sample(n_rest, random_state=seed)
    src = rng.integers(0, 96, n_rest)
    amp = 10 ** rng.uniform(-0.3, 0.3, n_rest)
    Xr = X96[src] * amp[:, None] * 10 ** rng.normal(0, 0.01, (n_rest, 60))
    Yr = Y96[src] * amp[:, None, None] ** np.array([1.0, 0.3, 0.4, 0.2])[None, None, :]
    X = np.concatenate([X96, Xr])
    Y = np.concatenate([Y96, Yr])
    tk = np.concatenate([np.arange(96), -np.ones(n_rest, int)])
    t_cat = pd.read_csv(ROOT / "data/processed/flare_catalogue.csv")
    from src.data.cv96 import load_meta
    m = load_meta()
    rows96 = [int(t_cat[(t_cat.event_id == r.event) & (pd.to_datetime(t_cat.peak) == r.peak_time)].index[0]) for r in m.itertuples()]
    np.savez_compressed(
        path, X=X, Y=Y, X_fism=X, X_goes_b=1.15 * X, X_goes_a=0.1 * X,
        bg_goes_b_24h=10 ** rng.uniform(-6.5, -5.5, len(X)), f107_adj=rng.uniform(70, 250, len(X)),
        goes_sat=np.full(len(X), 15), shift=np.zeros(len(X), int),
        peak=np.concatenate([m.peak_time.dt.strftime("%Y-%m-%dT%H:%M").values,
                             pd.to_datetime(rest.peak).dt.strftime("%Y-%m-%dT%H:%M").values]).astype(str),
        row=np.array(rows96 + list(rest.index)), key=np.array([f"k{i}" for i in range(len(X))]),
        event_id=np.array(ev96 + list(rest.event_id)).astype(str),
        cls=np.array([e.split("_")[1] for e in ev96] + list(rest.cls)).astype(str),
        selected=np.ones(len(X), bool), test_k=tk, is_test=tk >= 0)


def test_full_protocol_smoke(tmp_path, monkeypatch):
    npz = tmp_path / "events.npz"
    _synthetic_npz(npz)
    monkeypatch.setenv("FISM_PROTOCOL", "full")
    monkeypatch.setenv("FISM_EVENTS", str(npz))
    monkeypatch.setenv("FISM_INPUT", "default")
    monkeypatch.setenv("FISM_EXCLUDE_NEAR_TEST_H", "0")
    from src.data import pool
    pool.load_full.cache_clear()
    X, Y, names, meta, tr, va, te, te_latis = pool.load_full()
    assert list(te) == list(range(96)) and list(te_latis) == list(range(96, 192))
    assert len(meta) == len(X) == 96 + 96 + 160 and list(meta.k) == list(range(len(X)))
    assert len(set(tr) & set(va)) == 0 and min(tr.min(), va.min()) >= 192
    assert 0.08 < len(va) / (len(tr) + len(va)) < 0.15
    z = np.load(ROOT / "data/processed/zenodo_test_xy.npz")
    assert np.array_equal(X[:96], z["X_real"]) and np.array_equal(Y[:96], z["Y_true"])
    assert np.allclose(Y[96:192], z["Y_true"])          # synthetic npz: rebuilt test rows = Zenodo arrays
    # leakage switch removes events near test flares
    monkeypatch.setenv("FISM_EXCLUDE_NEAR_TEST_H", "240")
    pool.load_full.cache_clear()
    n_ex = len(pool.load_full()[4]) + len(pool.load_full()[5])
    assert n_ex < len(tr) + len(va)
    monkeypatch.setenv("FISM_EXCLUDE_NEAR_TEST_H", "0")
    monkeypatch.setenv("FISM_INPUT", "goes_b")
    pool.load_full.cache_clear()
    assert np.allclose(pool.load_full()[0][:96], 1.15 * z["X_real"])
    assert np.allclose(pool.load_full()[0][96:192], 1.15 * z["X_real"])
    monkeypatch.setenv("FISM_INPUT", "default")
    pool.load_full.cache_clear()

    from src.experiments.cv import run_specs, score_experiment
    exp = "pytest_full_smoke"
    out = ROOT / "results" / exp
    shutil.rmtree(out, ignore_errors=True)
    try:
        specs = [{"name": "Power-law", "type": "baseline", "baseline": "powerlaw"},
                 {"name": "RF", "type": "baseline", "baseline": "rf", "kwargs": {"n_estimators": 10}},
                 {"name": "GRU-pn", "type": "dl", "configs": ["gru"], "target": "peaknorm",
                  "extra": ["f107_adj", "bg_goes_b_24h"], "train": {"epochs": 2, "patience": 2, "sched_patience": 1}}]
        run_specs(exp, specs, seeds=(0,), workers=1)
        E, P, S = score_experiment(exp)
        assert set(S.model) == {"Power-law", "RF", "GRU-pn"}
        assert np.isfinite(P.CC).all() and len(E) == 3 * 4 * 96
        SL = pd.read_csv(out / "summary_latis.csv")
        assert set(SL.model) == set(S.model) and (out / "oofL_RF_s0.npy").exists()
        # identical test arrays -> identical scores on both test sets
        assert np.allclose(SL.sort_values(["model", "band"]).CC_mean.values, S.sort_values(["model", "band"]).CC_mean.values)
    finally:
        shutil.rmtree(out, ignore_errors=True)
        pool.load_full.cache_clear()


def test_full_protocol_random_exclusion_and_phase3_specs(tmp_path, monkeypatch):
    """Size-matched random exclusion, and every Phase-2/3 config spec runs (1 epoch) under the full protocol."""
    import yaml
    npz = tmp_path / "events.npz"
    _synthetic_npz(npz)
    monkeypatch.setenv("FISM_PROTOCOL", "full")
    monkeypatch.setenv("FISM_EVENTS", str(npz))
    monkeypatch.setenv("FISM_INPUT", "default")
    monkeypatch.setenv("FISM_EXCLUDE_NEAR_TEST_H", "0")
    from src.data import pool
    pool.load_full.cache_clear()
    n0 = len(pool.load_full()[4]) + len(pool.load_full()[5])
    monkeypatch.setenv("FISM_EXCLUDE_RANDOM_N", "40")
    monkeypatch.setenv("FISM_EXCLUDE_SEED", "1")
    pool.load_full.cache_clear()
    X, Y, names, meta, tr, va, te, teL = pool.load_full()
    assert len(tr) + len(va) == n0 - 40 and min(tr.min(), va.min()) >= 192
    from src.experiments.cv import run_fold_dl, run_fold_baseline
    n_specs = 0
    for cfg_name in ["p3_arch", "p3_resid", "p3_extreme", "p3_type", "p3_joint", "p3_features", "p3_early",
                     "p3_final", "p2_window", "p2_scale"]:
        cfg = yaml.safe_load(open(ROOT / "configs" / f"{cfg_name}.yaml"))
        for spec in cfg["specs"]:
            if spec.get("seeds") == []:
                continue
            spec = dict(spec)
            if spec["type"] == "baseline":
                te_, p, _ = run_fold_baseline(spec, 0, tr, va, te, 0)
            else:
                spec["train"] = {"epochs": 1, "patience": 1, "sched_patience": 1}
                te_, p, _ = run_fold_dl(spec, 0, tr, va, te, 0)
            assert p.shape == (96, 180, 4) and np.isfinite(p).all(), (cfg_name, spec["name"])
            n_specs += 1
    assert n_specs >= 25
    pool.load_full.cache_clear()


def test_paper1_options(tmp_path, monkeypatch):
    """GOES->FISM2 converter inputs, issue_delay specs and the delay analysis run under the full protocol."""
    import shutil
    import yaml
    npz = tmp_path / "events.npz"
    _synthetic_npz(npz)
    monkeypatch.setenv("FISM_PROTOCOL", "full")
    monkeypatch.setenv("FISM_EVENTS", str(npz))
    monkeypatch.setenv("FISM_EXCLUDE_NEAR_TEST_H", "0")
    monkeypatch.setenv("FISM_DEVICE_QUIET", "1")
    from src.data import pool
    z = np.load(ROOT / "data/processed/zenodo_test_xy.npz")
    for inp in ("goes_conv", "goes_convnn"):
        monkeypatch.setenv("FISM_INPUT", inp)
        pool.load_full.cache_clear()
        X, Y, names, meta, tr, va, te, teL = pool.load_full()
        # synthetic GOES = 1.15 x FISM2 -> the converter must recover FISM2 within a few percent
        assert np.median(np.abs(np.log10(X[:96]) - np.log10(z["X_real"]))) < 0.02, inp
        assert float(__import__("os").environ["FISM_CONV_RMSE_DEX"]) < 0.02
    monkeypatch.setenv("FISM_INPUT", "default")
    pool.load_full.cache_clear()
    from src.experiments.cv import run_fold_dl, run_specs, score_experiment
    cfg = yaml.safe_load(open(ROOT / "configs/paper1_delay.yaml"))
    for spec in cfg["specs"][:2]:
        spec = dict(spec)
        spec["train"] = {"epochs": 1, "patience": 1, "sched_patience": 1}
        te_, p, _ = run_fold_dl(spec, 0, tr, va, te, 0)
        assert p.shape == (96, 180, 4) and np.isfinite(p).all()
    exp = "pytest_paper1"
    out = ROOT / "results" / exp
    shutil.rmtree(out, ignore_errors=True)
    try:
        specs = [{"name": "GRU-z-d5", "type": "dl", "configs": ["gru"], "standardize": True, "issue_delay": 5,
                  "train": {"epochs": 1, "patience": 1, "sched_patience": 1}}]
        run_specs(exp, specs, seeds=(0,), workers=1)
        E, P, S = score_experiment(exp)
        assert (out / "val_GRU-z-d5_s0.npz").exists() and (out / "oofL_GRU-z-d5_s0.npy").exists()
    finally:
        shutil.rmtree(out, ignore_errors=True)
        pool.load_full.cache_clear()


def test_oom_falls_back_to_cpu(tmp_path, monkeypatch):
    """A CUDA out-of-memory error in a worker is retried once on a re-selected GPU, then on the CPU,
    instead of aborting the experiment."""
    import torch
    from src.experiments import cv
    from src.train import trainer
    npz = tmp_path / "events.npz"
    _synthetic_npz(npz)
    monkeypatch.setenv("FISM_PROTOCOL", "full")
    monkeypatch.setenv("FISM_EVENTS", str(npz))
    monkeypatch.setenv("FISM_INPUT", "default")
    monkeypatch.setenv("FISM_DEVICE_QUIET", "1")
    monkeypatch.delenv("FISM_REQUIRE_GPU", raising=False)
    from src.data import pool
    pool.load_full.cache_clear()
    X, Y, names, meta, tr, va, te, teL = pool.load_full()
    calls = {"n": 0}
    real = cv.run_fold_dl

    def flaky(*args, **kw):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise trainer.OOM("CUDA out of memory. Tried to allocate 56.00 MiB")
        return real(*args, **kw)
    monkeypatch.setattr(cv, "run_fold_dl", flaky)
    spec = {"name": "GRU-z", "type": "dl", "configs": ["gru"], "standardize": True, "train": {"epochs": 1, "patience": 1, "sched_patience": 1}}
    name, seed, (te_, p, info) = cv._job((spec, 0, tr, va, te, 0))
    assert calls["n"] == 3 and p.shape == (96, 180, 4) and trainer.get_device().type == "cpu"
    # a non-OOM error is not swallowed
    calls["n"] = 0

    def broken(*args, **kw):
        raise RuntimeError("Unable to handle autograd's threading in combination with fork-based multiprocessing")
    monkeypatch.setattr(cv, "run_fold_dl", broken)
    import pytest
    with pytest.raises(RuntimeError, match="autograd"):
        cv._job((spec, 0, tr, va, te, 0))
    trainer._DEVICE = None
    pool.load_full.cache_clear()


def test_require_gpu_fails_loudly(monkeypatch):
    """FISM_REQUIRE_GPU=1 raises instead of silently training on the CPU when CUDA is unusable."""
    import torch
    import pytest
    from src.train import trainer
    monkeypatch.setenv("FISM_DEVICE_QUIET", "1")
    monkeypatch.setenv("FISM_REQUIRE_GPU", "1")
    monkeypatch.setenv("FISM_DEVICE", "auto")
    monkeypatch.setenv("FISM_GPU_WAIT_S", "0")
    trainer._DEVICE = None
    if torch.cuda.is_available():
        monkeypatch.setattr(trainer, "_cuda_candidates", lambda: ([], "cuda:0: Cannot re-initialize CUDA in forked subprocess"))
    with pytest.raises(RuntimeError, match="FISM_REQUIRE_GPU"):
        trainer.get_device()
    monkeypatch.setenv("FISM_REQUIRE_GPU", "0")
    trainer._DEVICE = None
    assert trainer.get_device().type in ("cpu", "cuda")
    trainer._DEVICE = None


_FORK_SCRIPT = """
import sys; sys.path.insert(0, sys.argv[1])
from src.data import pool
from src.experiments.cv import run_specs
pool.load_full()                       # fits the GOES->FISM2 FCN converter in this (parent) process
spec = {"name": "GRU-z", "type": "dl", "configs": ["gru"], "standardize": True,
        "train": {"epochs": 1, "patience": 1, "sched_patience": 1}}
run_specs(sys.argv[2], [spec], seeds=(0, 1), workers=2)     # 2 forked workers, each calls backward()
print("FORK_OK")
"""


def test_convnn_parent_stays_fork_safe(tmp_path):
    """The FCN converter is fitted in a spawned child, so the forked training workers can still run backward()
    (PyTorch 'Autograd and Fork') and initialise CUDA. Regression test for the P2b failure of 2026-11-09.
    Runs in fresh interpreters because the pytest process itself has already run backward() in earlier tests."""
    import subprocess
    npz = tmp_path / "events.npz"
    _synthetic_npz(npz)
    env = dict(os.environ, FISM_PROTOCOL="full", FISM_EVENTS=str(npz), FISM_INPUT="goes_convnn",
               FISM_EXCLUDE_NEAR_TEST_H="0", FISM_DEVICE_QUIET="1", OMP_NUM_THREADS="1")
    env.pop("FISM_CONV_SUBPROCESS", None)
    env.pop("FISM_MP_CONTEXT", None)
    exp = "pytest_convnn_fork"
    out = ROOT / "results" / exp
    shutil.rmtree(out, ignore_errors=True)
    try:
        r = subprocess.run([sys.executable, "-c", _FORK_SCRIPT, str(ROOT), exp], env=env, cwd=ROOT,
                           capture_output=True, text=True, timeout=900)
        assert r.returncode == 0 and "FORK_OK" in r.stdout, r.stderr[-3000:]
        assert (out / "oof_GRU-z_s0.npy").exists() and (out / "oof_GRU-z_s1.npy").exists()
        assert "device: cpu (CUDA available but unusable" not in r.stdout
    finally:
        shutil.rmtree(out, ignore_errors=True)
    import torch
    if hasattr(os, "fork") and torch.cuda.device_count() > 0:
        # the in-process fit (FISM_CONV_SUBPROCESS=0) is exactly what breaks forked workers - autograd starts
        # its device threads at the first backward() on a machine that has CUDA devices (not on a CPU-only box)
        r = subprocess.run([sys.executable, "-c", _FORK_SCRIPT, str(ROOT), exp], env=dict(env, FISM_CONV_SUBPROCESS="0"),
                           cwd=ROOT, capture_output=True, text=True, timeout=900)
        shutil.rmtree(out, ignore_errors=True)
        assert r.returncode != 0 and ("autograd" in r.stderr.lower() or "forked subprocess" in r.stderr.lower()), r.stderr[-1500:]
    # FISM_MP_CONTEXT=spawn is the escape hatch that works even then (each worker reloads the data)
    r = subprocess.run([sys.executable, "-c", _FORK_SCRIPT, str(ROOT), exp],
                       env=dict(env, FISM_CONV_SUBPROCESS="0", FISM_MP_CONTEXT="spawn"),
                       cwd=ROOT, capture_output=True, text=True, timeout=900)
    shutil.rmtree(out, ignore_errors=True)
    assert r.returncode == 0 and "FORK_OK" in r.stdout and "start method: spawn" in r.stdout, r.stderr[-3000:]


def test_split_conformal_goes_input_alignment(tmp_path, monkeypatch):
    """Regression test: split_conformal_full.py aligns validation indices with the input-specific pool
    even when some training/validation events lack GOES XRS-B coverage."""
    import subprocess, sys
    npz = tmp_path / "events.npz"
    _synthetic_npz(npz)
    # Introduce 2 missing GOES-B events in the non-test pool so goes_b pool indices shift relative to default
    d = dict(np.load(npz, allow_pickle=False))
    d["X_goes_b"][200, :] = np.nan
    d["X_goes_b"][210, :] = np.nan
    np.savez_compressed(npz, **d)

    monkeypatch.setenv("FISM_PROTOCOL", "full")
    monkeypatch.setenv("FISM_EVENTS", str(npz))
    monkeypatch.setenv("FISM_INPUT", "goes_b")
    monkeypatch.setenv("FISM_DEVICE_QUIET", "1")
    from src.data import pool
    from src.experiments.cv import run_specs
    pool.load_full.cache_clear()
    exp = "p2_zscore_full_in-goes_b_p1"
    out = ROOT / "results" / exp
    out_srv = tmp_path / "full_server"
    shutil.rmtree(out, ignore_errors=True)
    try:
        spec = {"name": "GRU-z", "type": "dl", "configs": ["gru"], "standardize": True,
                "train": {"epochs": 1, "patience": 1, "sched_patience": 1}}
        run_specs(exp, [spec], seeds=(0, 1), workers=1)
        r = subprocess.run(
            [sys.executable, "notebooks/split_conformal_full.py", "--events", str(npz),
             "--results", str(ROOT / "results"), "--out", str(out_srv),
             "--models", f"{exp}:GRU-z", "--avg", "", "--cond", "class", "--tag", "test"],
            cwd=ROOT, capture_output=True, text=True, timeout=120
        )
        assert r.returncode == 0, r.stderr
        assert (out_srv / "conformal_split_test.csv").exists()
    finally:
        shutil.rmtree(out, ignore_errors=True)
        pool.load_full.cache_clear()


def test_conformal_uses_the_pool_of_each_input(tmp_path, monkeypatch):
    """split_conformal_full.py reads the validation predictions of an input-variant experiment against the event
    pool built for THAT input (rows shift when events without the input are dropped). Regression test for the
    mis-aligned GOES-input residuals of results_full_20261109_1520 (coverage 0.99, widths 1.3 dex)."""
    import subprocess
    npz = tmp_path / "events.npz"
    _synthetic_npz(npz)
    d = dict(np.load(npz, allow_pickle=False))
    d["X_goes_b"][100:106] = np.nan              # six train/val events without a GOES input -> rows shift
    np.savez_compressed(npz, **d)
    monkeypatch.setenv("FISM_PROTOCOL", "full")
    monkeypatch.setenv("FISM_EVENTS", str(npz))
    monkeypatch.setenv("FISM_EXCLUDE_NEAR_TEST_H", "0")
    monkeypatch.setenv("FISM_DEVICE_QUIET", "1")
    from src.data import pool
    from src.experiments.cv import run_specs
    spec = {"name": "FCN-z", "type": "dl", "configs": ["fcn_w0", "fcn_w1", "fcn_w2", "fcn_w3"], "standardize": True,
            "train": {"epochs": 1, "patience": 1, "sched_patience": 1}}
    exps = {"default": "pytest_conf_default", "goes_b": "pytest_conf_full_in-goes_b_t"}
    outs = [ROOT / "results" / e for e in exps.values()]
    for o in outs:
        shutil.rmtree(o, ignore_errors=True)
    try:
        n_val = {}
        for inp, exp in exps.items():
            monkeypatch.setenv("FISM_INPUT", inp)
            pool.load_full.cache_clear()
            n_val[inp] = pool.load_full()[5]
            run_specs(exp, [spec], seeds=(0, 1), workers=1)
            assert (ROOT / "results" / exp / "val_FCN-z_s0.npz").exists()
        assert len(n_val["goes_b"]) + len(pool.load_full()[4]) < 160 + 96      # some rows were dropped
        env = dict(os.environ, FISM_INPUT="default")
        r = subprocess.run([sys.executable, "notebooks/split_conformal_full.py", "--events", str(npz), "--results",
                            str(ROOT / "results"), "--out", str(tmp_path), "--models",
                            f"{exps['default']}:FCN-z", f"{exps['goes_b']}:FCN-z", "--avg", "", "--tag", "t", "--save-intervals"],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stderr[-3000:]
        assert "[skip]" not in r.stdout, r.stdout
        assert "input=goes_b" in r.stdout                      # the goes_b pool was loaded for that experiment
        S = pd.read_csv(tmp_path / "conformal_split_t.csv")
        Z = np.load(tmp_path / "conformal_intervals_t.npz", allow_pickle=False)
        k = f"FCN-z [{exps['goes_b']}]|zenodo96"
        assert Z[k + "|lo"].shape == (96, 180, 4) and (Z[k + "|lo"] <= Z[k + "|mu"]).all() and (Z[k + "|mu"] <= Z[k + "|hi"]).all()
        assert set(S.model) == {f"FCN-z [{exps['default']}]", f"FCN-z [{exps['goes_b']}]"}
        w = S[S.test == "zenodo96"].set_index("model").width_norm_w0_dex
        # synthetic GOES = 1.15 x FISM2 input: the two models see the same information -> similar interval widths
        assert 0.5 < w[f"FCN-z [{exps['goes_b']}]"] / w[f"FCN-z [{exps['default']}]"] < 2.0, w.to_dict()
    finally:
        for o in outs:
            shutil.rmtree(o, ignore_errors=True)
        monkeypatch.setenv("FISM_INPUT", "default")
        pool.load_full.cache_clear()
