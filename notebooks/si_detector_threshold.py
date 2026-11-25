"""A5 addendum: coverage over the minutes in force with the detector threshold on the catalogue scale (GOES 8-15 streams divided
by 1.43 = threshold 1.43e-5 W m-2), operational models (GOES + FCN converter), test flares and flares outside the isolation rule.
Pooled minutes, as paper1_p8_reissue.csv. No training."""
import os, sys, glob, importlib.util, numpy as np, pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = f"{REPO}/results"
spec = importlib.util.spec_from_file_location("pdet", f"{REPO}/notebooks/peak_detector.py"); pdet = importlib.util.module_from_spec(spec); spec.loader.exec_module(pdet)
spec = importlib.util.spec_from_file_location("ap7", f"{REPO}/notebooks/analyze_p7_p8.py"); ap7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ap7)
D = np.load(f"{REPO}/data/processed/events_latis.npz", allow_pickle=False)
S = np.load(f"{R}/full_server/peak_detector_streams.npz", allow_pickle=False); PRE = int(S["pre"])
def Tnext(scale):
    out = {}
    for j, r in enumerate(S["row"]):
        b = S["xrsb"][j] / (1.43 if (scale and S["sat"][j] <= 15) else 1.0)
        later = [q["t"] - PRE for q in pdet.detect_all(b, 2, 1e-5, rise_factor=2.0) if q["t_max"] > PRE + 5]
        out[int(r)] = min(later) if later else 180
    return out
E = pd.read_csv(f"{R}/full_server/paper1_p8_reissue_events.csv")
T0, T1 = Tnext(False), Tnext(True)
chk = E.dataset_row.map(T0).values; print("T_next reproduces paper1_p8_reissue_events.csv:", np.array_equal(chk, E["T_next_r2"].values))
I = np.load(f"{R}/full_server/conformal_intervals_none.npz", allow_pickle=False)
rb = np.load(f"{REPO}/data/processed/events_test96_rebuilt.npz", allow_pickle=False); o = np.argsort(rb["test_k"]); rows_t = rb["row"][o] if "row" in rb.files else None
ev_t = rb["event_id"][o].astype(str); ly_t = np.log10(rb["Y"][o].astype(float))
row_of_event = dict(zip(D["event_id"].astype(str), range(len(D["event_id"]))))
exp = f"{R}/p2_zscore_full_in-goes_convnn_p8"; ex_rows = pd.read_csv(f"{exp}/extra_rows.csv").dataset_row.values
# validation rows of the GOES pool (pool index -> dataset row), as paper1_extra_figures
ev96 = [str(e) for e in np.load(f"{REPO}/data/processed/zenodo_full.npz", allow_pickle=False)["event_names"]]
eid = D["event_id"].astype(str); x0 = D["X_goes_b"].astype(float)
valid = np.all(np.isfinite(x0) & (x0 > 0), 1) & np.all(np.isfinite(D["Y"]) & (D["Y"] > 0), (1, 2))
rest = np.where(D["selected"].astype(bool) & ~D["is_test"].astype(bool) & ~np.isin(eid, ev96) & valid)[0]
rest = rest[np.argsort(pd.to_datetime(D["peak"].astype(str)).values[rest], kind="stable")]
for m in ("GRU-z", "FCN-z"):
    key = f"{m} [p2_zscore_full_in-goes_convnn_p2nn]|rebuilt96"; lo, hi = I[f"{key}|lo"].astype(float), I[f"{key}|hi"].astype(float)
    okt = np.isfinite(lo).all((1, 2)); ins_t = ((ly_t >= lo) & (ly_t <= hi))[okt][:, :, 0]; rt = np.array([row_of_event[e] for e in ev_t[okt]])
    vs = [np.load(f) for f in sorted(glob.glob(f"{exp}/val_{m}_s*.npz"))]; idx = vs[0]["idx"]
    Lv = np.stack([v["pred_log"].astype(float) for v in vs]); yv = np.log10(D["Y"][rest[idx - 192]].astype(float))
    Lx = np.stack([np.load(f).astype(float) for f in sorted(glob.glob(f"{exp}/oofN_{m}_s*.npy"))])
    _, lx, hx = ap7.conf_norm(Lv, yv, Lx); lyx = np.log10(D["Y"][ex_rows].astype(float)); ins_x = ((lyx >= lx) & (lyx <= hx))[:, :, 0]
    for lab, ins, rows in (("test flares", ins_t, rt), ("outside the rule", ins_x, ex_rows)):
        res = []
        for nm, T in (("threshold 1e-5 (as published)", T0), ("catalogue scale (1.43e-5 for GOES 8-15)", T1)):
            Tv = np.array([T.get(int(r), 180) for r in rows]); V = np.arange(180)[None] < Tv[:, None]
            res.append(f"{nm}: in force {V.mean():.3f}, coverage in force {ins[V].mean():.3f}, with next declaration {(Tv < 180).mean():.3f}")
        print(f"{m:6s} {lab:17s} n={len(rows):3d} all-180 coverage {ins.mean():.3f} | " + " | ".join(res))
