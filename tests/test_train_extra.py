"""paper-1 P9: FISM_TRAIN_EXTRA_FOLD / run.py --train-extra-fold (training with the flares outside the isolation rule)."""
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
from test_delay_extra import _dataset, _env  # noqa: E402


def test_train_extra_fold_pool(tmp_path, monkeypatch):
    npz, post, d, not_iso = _dataset(tmp_path, n_not_isolated=30)
    pool = _env(monkeypatch, npz, post, extra=1)
    monkeypatch.delenv("FISM_TRAIN_EXTRA_FOLD", raising=False)
    X0, Y0, names0, _, tr0, va0, te0, teL0 = pool.load_full()
    idx, rows = pool.extra_index()
    fold = pool.extra_fold().copy()
    assert len(idx) == 30 and set(fold) <= {0, 1} and len(fold) == 30
    peaks = pd.to_datetime(d["peak"][rows].astype(str)).values.astype("datetime64[m]").astype(np.int64)
    assert (np.diff(peaks) >= 0).all()
    for i in range(len(rows)):                      # flares of different folds are at least 24 h apart
        other = fold != fold[i]
        assert not other.any() or np.abs(peaks[other] - peaks[i]).min() >= 1440
    got = {}
    for f in ("0", "1"):
        monkeypatch.setenv("FISM_TRAIN_EXTRA_FOLD", f)
        pool.load_full.cache_clear()
        X, Y, names, _, tr, va, te, teL = pool.load_full()
        assert np.array_equal(X, X0) and np.array_equal(Y, Y0) and names == names0
        assert np.array_equal(va, va0) and np.array_equal(te, te0) and np.array_equal(teL, teL0)
        add = idx[fold == int(f)]
        assert np.array_equal(tr[:len(tr0)], tr0) and np.array_equal(tr[len(tr0):], add)
        assert np.array_equal(pool.extra_fold(), fold) and f"of fold {f} in the training rows" in pool.describe()
        got[f] = set(add)
    assert got["0"] | got["1"] == set(idx) and not got["0"] & got["1"]
    if len(got["0"]) and len(got["1"]):
        assert not set(tr0) & set(idx)
    monkeypatch.setenv("FISM_TRAIN_EXTRA_FOLD", "2")
    pool.load_full.cache_clear()
    with pytest.raises(ValueError):
        pool.load_full()
    monkeypatch.setenv("FISM_TRAIN_EXTRA_FOLD", "0")
    monkeypatch.setenv("FISM_PREDICT_EXTRA", "0")
    pool.load_full.cache_clear()
    with pytest.raises(ValueError):
        pool.load_full()
    pool.load_full.cache_clear()


def test_run_py_train_extra_fold(tmp_path):
    npz, post, d, _ = _dataset(tmp_path, n_not_isolated=30)
    cfg = ROOT / "configs" / "pytest_p9.yaml"
    cfg.write_text("exp_id: pytest_p9\nseeds: [0]\nspecs:\n"
                   "  - {name: GRU-z, type: dl, configs: [gru], standardize: true, train: {epochs: 1, patience: 1, sched_patience: 1}}\n")
    env = dict(os.environ, FISM_GOES_POST=str(post), FISM_DEVICE_QUIET="1", OMP_NUM_THREADS="1")
    for k in ("FISM_REQUIRE_GPU", "FISM_ISSUE_DELAY", "FISM_PREDICT_EXTRA", "FISM_INPUT", "FISM_PROTOCOL", "FISM_TRAIN_EXTRA_FOLD"):
        env.pop(k, None)
    base = [sys.executable, "run.py", "--exp", "pytest_p9", "--protocol", "full", "--events", str(npz), "--workers", "1"]
    made = ["pytest_p9_full_tx1_p9", "pytest_p9_full_in-goes_convnn_tx0_p9"]
    try:
        for e in made:
            shutil.rmtree(ROOT / "results" / e, ignore_errors=True)
        r = subprocess.run(base + ["--predict-extra", "--train-extra-fold", "1", "--tag", "p9"], env=env, cwd=ROOT, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stderr[-3000:]
        out = ROOT / "results" / made[0]
        E = pd.read_csv(out / "extra_rows.csv")
        assert {"pool_index", "dataset_row", "event", "fold", "trained"} <= set(E.columns) and len(E) == 30
        assert (E.trained == (E.fold == 1)).all() and "of fold 1 in the training rows" in r.stdout
        assert np.load(out / "oofN_GRU-z_s0.npy").shape == (30, 180, 4) and (out / "oofL_GRU-z_s0.npy").exists()
        r = subprocess.run(base + ["--input", "goes_convnn", "--predict-extra", "--train-extra-fold", "0", "--tag", "p9"], env=env, cwd=ROOT,
                           capture_output=True, text=True, timeout=900)
        assert r.returncode == 0 and (ROOT / "results" / made[1] / "extra_rows.csv").exists(), r.stderr[-3000:]
        r = subprocess.run(base + ["--train-extra-fold", "0"], env=env, cwd=ROOT, capture_output=True, text=True, timeout=300)
        assert r.returncode != 0 and "needs --predict-extra" in r.stderr
    finally:
        cfg.unlink(missing_ok=True)
        for e in made:
            shutil.rmtree(ROOT / "results" / e, ignore_errors=True)
