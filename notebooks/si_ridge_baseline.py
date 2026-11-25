"""A3 addendum: closed-form ridge regression of the 720 log10 targets on the 60 log10 inputs (Kim et al.'s linear-regression
baseline, in log space), penalty chosen on the validation flares; GOES XRS-B input and FISM2 input, 95 rebuilt test flares."""
import os, sys, numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
os.environ.update(FISM_PROTOCOL="full", FISM_DEVICE_QUIET="1", FISM_PREDICT_EXTRA="0", FISM_ISSUE_DELAY="0",
                  FISM_EXCLUDE_NEAR_TEST_H="0", FISM_EXCLUDE_RANDOM_N="0", FISM_EXCLUDE_SAME_AR="0")
from src.data import pool
def mape4(L, y): return 100 * np.mean(np.abs(10 ** L - y) / y, axis=(-2, -1))
ok = None
for inp in ("goes_b", "default"):
    os.environ["FISM_INPUT"] = inp; pool.load_full.cache_clear(); X, Y, n, meta, tr, va, te, teL = pool.load_full()
    if ok is None:
        ok = np.ones(96, bool); ok[41] = False
    lx = np.log10(X); ly = np.log10(Y).reshape(len(Y), -1); mu, sd = lx[tr].mean(0), lx[tr].std(0); Z = (lx - mu) / sd
    ym = ly[tr].mean(0); A = Z[tr]; best = None
    for lam in (0.1, 1, 3, 10, 30, 100, 300, 1000):
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (ly[tr] - ym)); v = mape4((Z[va] @ W + ym).reshape(-1, 180, 4), Y[va]).mean()
        if best is None or v < best[0]: best = (v, lam, W)
    v, lam, W = best; L = (Z[96:192][ok] @ W + ym).reshape(-1, 180, 4); yt = Y[96:192][ok]
    per = 100 * np.mean(np.abs(10 ** L - yt) / yt, axis=1).mean(0)
    print(f"{inp:8s} ridge(log) lambda={lam:g} val MAPE4 {v:.2f}  test95 MAPE4 {mape4(L, yt).mean():.2f}  bands " + " ".join(f"{q:.2f}" for q in per))
