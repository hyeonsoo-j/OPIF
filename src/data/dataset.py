"""Event dataset (LaTiS FISM2 reconstruction) and train/val/test split.

Split policy (docs/decisions.md):
  * test  = the 96 Zenodo test events [zenodo], matched to catalogue rows by event id; when an id
            is ambiguous (two flares of the same class on the same day) the row whose reconstructed
            w0 curve correlates best (log10) with the Zenodo w0 curve is used;
  * remaining events that satisfy the paper's isolation rule (no other >=M1 peak within 60 min
    before / 180 min after) are allocated, within class (M, X) in time order, by the repeating
    pattern 8 train : 1 val (paper: 8:1:1 with the 1 test slot already fixed by Zenodo).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BANDS = ["w0", "w1", "w2", "w3"]
BAND_LABELS = ["0.1–0.8 nm", "9.0–9.9 nm", "13.0–13.9 nm", "30.0–30.9 nm"]


@dataclass
class Split:
    X: np.ndarray        # (N, 60) physical, W m^-2
    Y: np.ndarray        # (N, 180, 4) physical
    event_id: np.ndarray
    cls: np.ndarray
    peak: np.ndarray

    def __len__(self):
        return len(self.X)

    @property
    def is_x(self):
        return np.array([c.startswith("X") for c in self.cls])


def _corr(a, b):
    a, b = np.log10(a).ravel(), np.log10(b).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def load_splits(path: Path | None = None, zenodo_path: Path | None = None, seed_order: bool = True):
    path = path or ROOT / "data/processed/events_latis.npz"
    zenodo_path = zenodo_path or ROOT / "data/processed/zenodo_test_xy.npz"
    d = np.load(path, allow_pickle=False)
    z = np.load(zenodo_path)
    zev = [str(e) for e in z["event_names"]]
    X, Y = d["X"], d["Y"]
    eid, cls, peak, sel = d["event_id"], d["cls"], d["peak"], d["selected"]
    n = len(X)
    # --- test rows
    test_idx, test_match = [], {}
    for k, e in enumerate(zev):
        cand = np.where(eid == e)[0]
        if len(cand) == 0:
            continue
        zc = np.concatenate([z["X_real"][k], z["Y_true"][k, :, 0]])
        scores = [_corr(np.concatenate([X[c], Y[c, :, 0]]), zc) for c in cand]
        best = cand[int(np.argmax(scores))]
        test_idx.append(best)
        test_match[e] = (int(best), float(max(scores)), len(cand))
    test_idx = np.array(test_idx, dtype=int)
    test_ids = set(eid[test_idx])
    # rows sharing an id with a test event are excluded from train/val (avoid ambiguity/leakage)
    rest = np.array([i for i in range(n) if eid[i] not in test_ids and sel[i]], dtype=int)
    order = rest[np.argsort(peak[rest])]
    tr, va = [], []
    for c in ("M", "X"):
        oc = [i for i in order if str(cls[i]).startswith(c)]
        for j, i in enumerate(oc):
            (va if j % 9 == 8 else tr).append(i)
    tr, va = np.array(sorted(tr)), np.array(sorted(va))

    def mk(idx):
        return Split(X[idx], Y[idx], eid[idx], cls[idx], peak[idx])

    info = {"n_total": n, "n_train": len(tr), "n_val": len(va), "n_test": len(test_idx),
            "test_match": test_match}
    return mk(tr), mk(va), mk(test_idx), info


def to_log(X, Y):
    return np.log10(X), np.log10(Y)
