"""Correlation between the flare-mean log10 EUV target and the minimum of the 60-min input window (FISM2 0.1-0.8 nm input vs
GOES XRS-B input), 95 rebuilt test flares with a GOES window. Independent recomputation of Text S4 of the Supporting Information."""
import os, numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rb = np.load(f"{REPO}/data/processed/events_test96_rebuilt.npz", allow_pickle=False); o = np.argsort(rb["test_k"])
Y, Xf, Xg = rb["Y"][o].astype(float), rb["X"][o].astype(float), rb["X_goes_b"][o].astype(float)
ok = np.all(np.isfinite(Xg) & (Xg > 0), 1); ym = np.log10(Y[ok]).mean(1)
for lab, X in (("FISM2 0.1-0.8 nm input", Xf[ok]), ("GOES XRS-B input", Xg[ok])):
    xm = np.log10(X).min(1)
    print(f"{lab:24s} n={ok.sum()} corr with flare-mean log10 target, bands 9.0-9.9 / 13.0-13.9 / 30.0-30.9 nm:",
          " ".join(f"{np.corrcoef(ym[:, b], xm)[0, 1]:.2f}" for b in (1, 2, 3)))
