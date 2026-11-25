"""Independent re-computation of the P11 tables: no import of the project code. Usage: verify_p11_indep.py <project root> <results root>"""
import sys, glob
import numpy as np, pandas as pd

ROOT, R = sys.argv[1], sys.argv[2]
EXP = f"{R}/p2_zscore_full_in-goes_convnn_ops_p11"
d = np.load(f"{ROOT}/data/processed/events_latis.npz", allow_pickle=False)
g = np.load(f"{ROOT}/data/processed/events_goes_ops.npz", allow_pickle=False)
st = np.load(f"{R}/full_server/peak_detector_streams.npz", allow_pickle=False)
H = pd.read_csv(f"{R}/full_server/paper1_p11_hboot.csv"); I = pd.read_csv(f"{R}/full_server/paper1_p11_inforce.csv")
P = pd.read_csv(f"{R}/full_server/paper1_p11_inputs.csv"); DT = pd.read_csv(f"{R}/full_server/paper1_p11_detector.csv")
EV8 = pd.read_csv(f"{R}/full_server/paper1_p8_reissue_events.csv").set_index("dataset_row")
SRV = pd.read_csv(f"{R}/full_server/peak_detector_events_v3.csv").set_index("row")
T = pd.read_csv(f"{EXP}/ops_rows.csv"); E = pd.read_csv(f"{EXP}/extra_rows.csv")
where = {int(p): j for j, p in enumerate(E.pool_index)}
rows, kind = T.dataset_row.to_numpy(), T.kind.to_numpy()
y = d["Y"][rows].astype(float)
worst = 0.0

def load(m, pre):
    return np.stack([np.load(f, allow_pickle=False).astype(float) for f in sorted(glob.glob(f"{EXP}/{pre}_{m}_s*.npy"))])

def mape(L, bands=slice(None)):
    return 100 * np.mean(np.abs(10 ** L[..., bands] - y[None][..., bands]) / y[None][..., bands], axis=(2, 3))

# (a) inputs
xo, xs = g["xrsb_ops"][rows].astype(float), d["X_goes_b"][rows].astype(float)
dl = np.log10(xo / xs)
for name, k in (("isolated test flares", "test"), ("flares outside the rule", "outside")):
    m = kind == k
    p = P[(P.group == name) & (P.series == "all")].iloc[0]
    vals = (int(m.sum()), float(np.median(dl[m])), float(np.sqrt(np.mean(dl[m] ** 2))), float((np.abs(dl[m]) > 0.05).mean()))
    worst = max(worst, abs(vals[0] - p.n_complete_window), abs(vals[1] - p.median_dlog), abs(vals[2] - p.rms_dlog), abs(vals[3] - p.share_abs_gt_0p05))
    print(f"inputs  {name:24s} n={vals[0]:3d} median {vals[1]:+.4f} rms {vals[2]:.4f} share>0.05 {vals[3]:.4f}   (table {p.median_dlog:+.4f} {p.rms_dlog:.4f} {p.share_abs_gt_0p05:.4f})")

ao, asc = g["xrsa_ops"][rows].astype(float), d["X_goes_a"][rows].astype(float)
okA = np.all(np.isfinite(ao) & (ao > 0), axis=1) & np.all(np.isfinite(asc) & (asc > 0), axis=1)
for name, k in (("isolated test flares", "test"), ("flares outside the rule", "outside")):
    m = (kind == k) & okA
    da = np.log10(ao[m] / asc[m])
    p = P[(P.group == name) & (P.series == "all")].iloc[0]
    worst = max(worst, abs(np.median(da) - p.xrsa_median_dlog), abs(np.sqrt(np.mean(da ** 2)) - p.xrsa_rms_dlog), abs((np.abs(da) > 0.05).mean() - p.xrsa_share_abs_gt_0p05))
    print(f"XRS-A   {name:24s} n={m.sum():3d} median {np.median(da):+.4f} rms {np.sqrt(np.mean(da ** 2)):.4f} share>0.05 {(np.abs(da) > 0.05).mean():.4f}   (table {p.xrsa_median_dlog:+.4f} {p.xrsa_rms_dlog:.4f} {p.xrsa_share_abs_gt_0p05:.4f})")
old = np.isin(d["goes_sat"][rows], (8, 10, 12)) & okA
print(f"XRS-A   GOES 8-12 twins: {old.sum()} flares, log10(operational / science) between {np.log10(ao[old] / asc[old]).min():+.4f} and {np.log10(ao[old] / asc[old]).max():+.4f}; "
      f"share of minutes within 0.002 dex of -0.1461: {(np.abs(np.log10(ao[old] / asc[old]) + 0.1461) < 0.002).mean():.4f}")

# (b) forecasts
rng = np.random.default_rng(77)
for m in ("GRU-z", "FCN-z"):
    LO, Lt, Lx = load(m, "oofO"), load(m, "oofL"), load(m, "oofN")
    LS = np.empty_like(LO)
    for j, (tw, k) in enumerate(zip(T.twin_pool_index, kind)):
        LS[:, j] = Lt[:, tw - 96] if k == "test" else Lx[:, where[int(tw)]]
    D = mape(LO) - mape(LS)
    for name, k in (("isolated test flares", "test"), ("flares outside the rule", "outside")):
        sel = kind == k
        Dk = D[:, sel]
        bt = np.array([Dk[rng.integers(0, 5, 5)][:, rng.integers(0, Dk.shape[1], Dk.shape[1])].mean() for _ in range(5000)])
        h = H[H.family & (H.model == m) & (H.set == name)].iloc[0]
        worst = max(worst, abs(Dk.mean() - h.d), abs(mape(LS)[:, sel].mean() - h.MAPE4_science), abs(mape(LO)[:, sel].mean() - h.MAPE4_operational))
        print(f"{m} {name:24s} n={sel.sum():3d} science {mape(LS)[:, sel].mean():.4f} operational {mape(LO)[:, sel].mean():.4f} d {Dk.mean():+.4f} "
              f"[{np.percentile(bt, 2.5):+.3f}, {np.percentile(bt, 97.5):+.3f}]   table {h.d:+.4f} [{h.lo:+.3f}, {h.hi:+.3f}]   per seed: {np.round(Dk.mean(1), 2)}")
        # minutes in force (declarations of the science stream, from the table of 2026-11-12)
        Tn = np.array([EV8.T_next_r2[int(r)] for r in rows[sel]])
        V = np.arange(180)[None] < Tn[:, None]
        rel = 100 * np.abs(10 ** LO[:, sel] - y[sel][None]) / y[sel][None]
        m4, w0 = (rel.mean(-1).mean(0) * V).sum() / V.sum(), (rel[..., 0].mean(0) * V).sum() / V.sum()
        q = I[(I.model == m) & (I.input == "operational input") & (I.group == name)].iloc[0]
        worst = max(worst, abs(m4 - q.MAPE4), abs(w0 - q.MAPE_w0), abs(V.mean() - q.share_minutes_in_force))
        print(f"      in force: minutes {V.mean():.3f} MAPE4 {m4:.3f} MAPE_w0 {w0:.3f}   (table {q.share_minutes_in_force:.3f} {q.MAPE4:.3f} {q.MAPE_w0:.3f})")

# (c) detector, written again from its description: running maximum of the episode, declared k minutes after it if it is >= M1 and
# >= 2 x the minimum before it; a rise by a factor 2 above the minimum after the candidate starts a new episode
def declarations(b, k=2, thr=1e-5, rise=2.0):
    out, best, ib, lo_pre, trough, declared = [], -np.inf, -1, np.inf, np.inf, -np.inf
    for t, v in enumerate(b):
        if not np.isfinite(v):
            continue
        if v > best:
            lo_pre = v if ib < 0 else min(lo_pre, trough)
            best, ib, trough = v, t, np.inf
        else:
            trough = min(trough, v)
            if v >= rise * trough:
                lo_pre, best, ib, trough, declared = trough, v, t, np.inf, -np.inf
        if best >= thr and t - ib >= k and best >= rise * lo_pre and best > declared:
            out.append((t, ib)); declared = best
    return out

L, pre = int(g["max_lag"]), int(st["pre"])
ops = g["wide_b"][:, L:L + 240]
both = np.array([np.isfinite(ops[j, pre - 10:pre + 10]).mean() >= 0.5 for j in range(len(st["row"]))])
own_s, own_o, nb_s, nb_o = [], [], [], []
for j in np.where(both)[0]:
    for b, own, nb in ((st["xrsb"][j], own_s, nb_s), (ops[j], own_o, nb_o)):
        dec = declarations(b)
        mine = [t - pre for t, tm in dec if abs(tm - pre) <= 5]
        own.append(mine[-1] if mine else np.nan); nb.append(sum(tm < pre - 5 for _, tm in dec))
own_s, own_o, nb_s, nb_o = map(np.array, (own_s, own_o, nb_s, nb_o))
srv = SRV.loc[st["row"][both]]
print(f"detector, {both.sum()} flares with both streams: science own declared {np.isfinite(own_s).mean():.4f} (server table, same flares {srv.own_raw_k2.notna().mean():.4f}), "
      f"earlier {np.mean(nb_s > 0):.4f} (server {np.mean(srv.n_before_raw_k2 > 0):.4f}); operational own declared {np.isfinite(own_o).mean():.4f}, earlier {np.mean(nb_o > 0):.4f}, "
      f"same minute {np.mean(np.nan_to_num(own_s, nan=-999) == np.nan_to_num(own_o, nan=-999)):.4f}")
a = DT[DT.group == "all flares with both streams"].set_index("stream")
worst = max(worst, abs(np.isfinite(own_s).mean() - a.loc["science-quality stream", "frac_own_declared"]), abs(np.isfinite(own_o).mean() - a.loc["operational stream", "frac_own_declared"]),
            abs(np.mean(nb_o > 0) - a.loc["operational stream", "frac_with_earlier"]),
            abs(np.mean(np.nan_to_num(own_s, nan=-999) == np.nan_to_num(own_o, nan=-999)) - a.loc["operational vs science", "frac_same_own_declaration_minute"]),
            abs(np.isfinite(own_s).mean() - srv.own_raw_k2.notna().mean()), abs(np.nanmax(np.abs(own_s - srv.own_raw_k2.to_numpy()))))
print(f"largest |difference| to the tables (point values): {worst:.2e}")
