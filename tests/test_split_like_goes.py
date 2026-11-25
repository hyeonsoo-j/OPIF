"""Control experiment 1: FISM_SPLIT_LIKE_GOES=1 gives a FISM2-input pool the train/val assignment of the GOES-input pools.
Runs on the built data set only (skipped without data/processed/events_latis.npz)."""
import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "data/processed/events_latis.npz"


@pytest.mark.skipif(not EV.exists(), reason="needs the built data set")
def test_split_like_goes_reproduces_the_goes_assignment(monkeypatch):
    from src.data import pool
    monkeypatch.setenv("FISM_PROTOCOL", "full"); monkeypatch.setenv("FISM_EVENTS", str(EV))
    for k in ("FISM_ISSUE_DELAY", "FISM_PREDICT_EXTRA", "FISM_EXCLUDE_NEAR_TEST_H", "FISM_EXCLUDE_RANDOM_N", "FISM_EXCLUDE_SAME_AR"):
        monkeypatch.setenv(k, "0")

    def sets(inp, like):
        os.environ["FISM_INPUT"], os.environ["FISM_SPLIT_LIKE_GOES"] = inp, like
        pool.load_full.cache_clear()
        _, _, _, meta, tr, va, _, _ = pool.load_full()
        key = lambda ix: set(zip(pd.to_datetime(meta.peak_time.values[ix]), meta.cls.values[ix]))  # noqa: E731
        return key(tr), key(va)

    g = sets("goes_b", "0")
    c = sets("default", "1")
    d = sets("default", "0")
    pool.load_full.cache_clear()
    assert c == g
    assert len(d[0]) >= len(c[0]) and len(d[1]) == len(c[1])
