"""Unit / alignment checks of the decoded Zenodo test arrays (Phase 0, mission item 'tests/test_data.py').

(a) FISM2 w0 vs GOES input: identical at the shared peak minute, same order of magnitude as the
    GOES class of the matched flare-list entry, high correlation across events.
(b) time alignment: input maximum within 3 min of the window end; output maximum at the start.
(c) no missing / non-positive values (no interpolation needed).
(d) window overlap between test events is flagged (expected none).
"""
import numpy as np
import pandas as pd
import pytest

from src.data.cv96 import load_meta, load_test96


@pytest.fixture(scope="module")
def data():
    X, Y, ev = load_test96()
    return X, Y, ev, load_meta()


def test_shapes(data):
    X, Y, ev, m = data
    assert X.shape == (96, 60) and Y.shape == (96, 180, 4) and len(ev) == 96
    assert m.is_x.sum() == 8 and (~m.is_x).sum() == 88


def test_a_w0_matches_goes_input(data):
    X, Y, ev, m = data
    d = np.log10(X[:, -1]) - np.log10(Y[:, 0, 0])
    assert np.max(np.abs(d)) <= 1e-4                       # shared minute (quantisation 1e-4 dex)
    rel = np.log10(X[:, -1]) - np.log10(m.cls_value.values)
    assert np.all(np.abs(rel) < 0.5)                         # same order of magnitude as GOES class
    cc = np.corrcoef(np.log10(X[:, -1]), np.log10(m.cls_value.values))[0, 1]
    assert cc > 0.9


def test_b_time_alignment(data):
    X, Y, ev, m = data
    assert np.all(np.argmax(X, axis=1) >= 56)                # input max within 3 min of the end
    assert (np.argmax(Y[:, :, 0], axis=1) == 0).sum() >= 95  # output starts at the peak


def test_c_no_missing(data):
    X, Y, ev, m = data
    assert np.isfinite(X).all() and np.isfinite(Y).all()
    assert (X > 0).all() and (Y > 0).all()


def test_d_window_overlap_flagged(data):
    X, Y, ev, m = data
    p = m.peak_time.values.astype("datetime64[m]")
    n = 0
    for k in range(96):
        for j in range(96):
            if j != k and p[j] <= p[k] and p[j] + np.timedelta64(179, "m") >= p[k] - np.timedelta64(59, "m"):
                n += 1
    assert n == 0


def test_catalogue_match_complete(data):
    X, Y, ev, m = data
    assert m.peak_time.notna().all()
