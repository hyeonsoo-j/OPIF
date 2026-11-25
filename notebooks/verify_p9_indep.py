"""Independent re-computation of the P9 tables: no import of the project code. Usage: verify_p9_indep.py <project root> <results root>"""
import sys, glob, math
import numpy as np, pandas as pd

ROOT, R = sys.argv[1], sys.argv[2]
d = np.load(f"{ROOT}/data/processed/events_latis.npz", allow_pickle=False)
H = pd.read_csv(f"{R}/full_server/paper1_p9_hboot.csv")
I = pd.read_csv(f"{R}/full_server/paper1_p9_inforce.csv")
EV = pd.read_csv(f"{R}/full_server/paper1_p8_reissue_events.csv").set_index("dataset_row")
test_row = {int(d["test_k"][i]): int(i) for i in np.where(d["is_test"])[0]}

def load(exp, m, pre):
    return np.stack([np.load(f, allow_pickle=False).astype(float) for f in sorted(glob.glob(f"{R}/{exp}/{pre}_{m}_s*.npy"))])

def mape(L, y, bands=slice(None)):          # (S, n)
    return 100 * np.mean(np.abs(10 ** L[..., bands] - y[None][..., bands]) / y[None][..., bands], axis=(2, 3))

def conformal(Lv, yv, Lq, level=0.9):       # normalised split conformal, 30-min blocks, per band
    mv, sv, mq, sq = Lv.mean(0), Lv.std(0), Lq.mean(0), Lq.std(0)
    lo, hi = np.empty_like(mq), np.empty_like(mq)
    for b in range(4):
        for k in range(6):
            s = slice(30 * k, 30 * k + 30)
            s0 = np.median(sv[:, s, b])
            sc = np.sort((np.abs(yv[:, s, b] - mv[:, s, b]) / (sv[:, s, b] + s0)).ravel())
            q = sc[min(len(sc), math.ceil((len(sc) + 1) * level)) - 1]
            lo[:, s, b], hi[:, s, b] = mq[:, s, b] - q * (sq[:, s, b] + s0), mq[:, s, b] + q * (sq[:, s, b] + s0)
    return lo, hi

worst = 0.0
rng = np.random.default_rng(2026)
for inp, tag in (("goes_convnn", "_in-goes_convnn"), ("fism", "")):
    p8 = f"p2_zscore_full{tag}_p8"
    p9 = [f"p2_zscore_full{tag}_tx{f}_p9" for f in (0, 1)]
    E = pd.read_csv(f"{R}/{p9[0]}/extra_rows.csv")
    rows, fold = E.dataset_row.values, E.fold.values
    yx = d["Y"][rows].astype(float)
    for m in ("GRU-z", "FCN-z"):
        a8 = mape(load(p8, m, "oofN"), yx)
        a9r = [mape(load(e, m, "oofN"), yx) for e in p9]
        a9 = np.where(fold[None] == 0, a9r[1], a9r[0])                 # fold-0 flares from run 1, fold-1 flares from run 0
        t8 = load(p8, m, "oofL")
        ok = ~np.isnan(t8).any(axis=(0, 2, 3))
        yt = np.stack([d["Y"][test_row[k]] for k in np.where(ok)[0]]).astype(float)
        b8 = mape(t8[:, ok], yt)
        b9 = 0.5 * (mape(load(p9[0], m, "oofL")[:, ok], yt) + mape(load(p9[1], m, "oofL")[:, ok], yt))
        for name, A, C in (("outside", a9, a8), ("test flares (isolated)", b9, b8)):
            h = H[(H.input == inp) & (H.model == m) & H.set.str.startswith(name) & H.family].iloc[0]
            n = A.shape[1]
            bt = np.empty(4000)
            for i in range(4000):
                e = rng.integers(0, n, n)
                bt[i] = A[rng.integers(0, 5, 5)][:, e].mean() - C[rng.integers(0, 5, 5)][:, e].mean()
            lo, hi = np.percentile(bt, [2.5, 97.5])
            dd = A.mean() - C.mean()
            worst = max(worst, abs(dd - h.d), abs(A.mean() - h.MAPE4_P9), abs(C.mean() - h.MAPE4_P8))
            print(f"{inp:12s} {m} {name:22s} n={n:3d}  P8 {C.mean():.4f}  P9 {A.mean():.4f}  d {dd:+.4f} [{lo:+.3f}, {hi:+.3f}]   table d {h.d:+.4f} [{h.lo:+.3f}, {h.hi:+.3f}]")
        # in force (declaration rule), outside flares, 0.1-0.8 nm coverage and MAPE4: P8 and P9
        vs = [np.load(f, allow_pickle=False) for f in sorted(glob.glob(f"{R}/{p8}/val_{m}_s*.npz"))]
        vidx = vs[0]["idx"]
        # validation targets: taken from the dataset through the pool order is project code; use instead the truth saved with the P8 analysis if present
        has = np.array([int(r) in EV.index for r in rows])
        T = np.array([EV.T_next_r2[int(r)] if int(r) in EV.index else 0 for r in rows])
        V = (np.arange(180)[None] < T[:, None]) & has[:, None]
        rel8 = 100 * np.abs(10 ** load(p8, m, "oofN") - yx[None]) / yx[None]
        relr = [100 * np.abs(10 ** load(e, m, "oofN") - yx[None]) / yx[None] for e in p9]
        rel9 = np.where((fold == 0)[None, :, None, None], relr[1], relr[0])
        for trn, rel in (("P8", rel8), ("P9", rel9)):
            row = I[(I.input == inp) & (I.model == m) & I.training.str.startswith(trn) & (I.group == "outside the rule: all")].iloc[0]
            m4 = (rel.mean(-1).mean(0) * V).sum() / V.sum()
            w0 = (rel[..., 0].mean(0) * V).sum() / V.sum()
            worst = max(worst, abs(m4 - row.MAPE4), abs(w0 - row.MAPE_w0), abs(V.sum() / has.sum() / 180 - row.share_minutes_in_force))
            print(f"      in force, outside, {trn}: n={int(has.sum())} minutes {V.sum() / has.sum() / 180:.3f}  MAPE4 {m4:.3f} (table {row.MAPE4:.3f})  MAPE_w0 {w0:.3f} (table {row.MAPE_w0:.3f})")
print(f"largest |difference| to the tables (point values): {worst:.2e}")
