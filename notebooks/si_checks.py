"""Supporting-Information numbers of the manuscript (no training; 2026-11-25): train/validation assignment of the two input
paths, converter gains on test and validation flares, level/shape decomposition, multi-level interval coverage, satellite series,
isolation rule of the test flares, catalogue vs science-quality flux scale, detector threshold, closed-form baselines.
Own implementation (vectorised hierarchical bootstrap, own flare keys); data loader and conformal routine of the repo.
    python notebooks/si_checks.py > si_checks_output.txt
"""
import os, sys, glob, argparse, importlib.util, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); A = ap.parse_args()
REPO = A.repo; R = f"{REPO}/results"; sys.path.insert(0, REPO)
os.environ.update(FISM_PROTOCOL="full", FISM_EVENTS=f"{REPO}/data/processed/events_latis.npz",
                  FISM_GOES_POST=f"{REPO}/data/processed/events_goes_post.npz", FISM_DEVICE_QUIET="1",
                  FISM_EXCLUDE_NEAR_TEST_H="0", FISM_EXCLUDE_RANDOM_N="0", FISM_EXCLUDE_SAME_AR="0",
                  FISM_TRAIN_EXTRA_FOLD="", FISM_PREDICT_OPS="0", FISM_ISSUE_DELAY="0", FISM_PREDICT_EXTRA="0")
from src.data import pool
spec = importlib.util.spec_from_file_location("ap7", f"{REPO}/notebooks/analyze_p7_p8.py"); ap7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ap7)
spec = importlib.util.spec_from_file_location("pdet", f"{REPO}/notebooks/peak_detector.py"); pdet = importlib.util.module_from_spec(spec); spec.loader.exec_module(pdet)
pd.set_option("display.width", 250, "display.max_columns", 40)
t0 = time.time()
ARCH = ("FCN-z", "GRU-z", "LSTM-z", "RNN-z", "Transformer-z")
EXP = {"FISM2": ("default", "p2_zscore_full_gpu5"), "GOES": ("goes_b", "p2_zscore_full_in-goes_b_p1"),
       "linear": ("goes_b", "p2_zscore_full_in-goes_conv_p2"), "FCNconv": ("goes_b", "p2_zscore_full_in-goes_convnn_p2nn")}
def hr(t): print("=" * 100); print(t)

def load(inp):
    os.environ["FISM_INPUT"] = inp; pool.load_full.cache_clear()
    X, Y, names, meta, tr, va, te, teL = pool.load_full()
    key = np.array([f"{pd.Timestamp(p):%Y-%m-%d %H:%M} {c}" for p, c in zip(meta["peak_time"], meta["cls"])])
    return dict(X=X, Y=Y, meta=meta, tr=tr, va=va, key=key)
P = {"default": load("default"), "goes_b": load("goes_b")}
F, G = P["default"], P["goes_b"]

def oof(exp, m): return np.stack([np.load(f).astype(float) for f in sorted(glob.glob(f"{R}/{exp}/oofL_{m}_s*.npy"))])
def val(exp, m):
    z = [np.load(f) for f in sorted(glob.glob(f"{R}/{exp}/val_{m}_s*.npz"))]
    for v in z: assert np.array_equal(v["idx"], z[0]["idx"])
    return np.stack([v["pred_log"].astype(float) for v in z]), z[0]["idx"]
def mape4(L, y): return 100 * np.mean(np.abs(10 ** L - y) / y, axis=(-2, -1))      # (..., n)

def hboot(pairs, B=10000, seed=2026, paired=True):
    """pairs: [(a (Sa, n), b (Sb, m))]; resample flares (shared across architectures, and between a and b if paired)
    and, independently for every model, seeds. Returns d, lo, hi, p."""
    rng = np.random.default_rng(seed); n, m = pairs[0][0].shape[1], pairs[0][1].shape[1]
    Wa = rng.multinomial(n, np.full(n, 1 / n), size=B) / n
    Wb = Wa if paired else rng.multinomial(m, np.full(m, 1 / m), size=B) / m
    out = np.zeros(B)
    for a, b in pairs:
        Ca = rng.multinomial(a.shape[0], np.full(a.shape[0], 1 / a.shape[0]), size=B) / a.shape[0]
        Cb = rng.multinomial(b.shape[0], np.full(b.shape[0], 1 / b.shape[0]), size=B) / b.shape[0]
        out += ((Wa @ a.T) * Ca).sum(1) - ((Wb @ b.T) * Cb).sum(1)
    out /= len(pairs); d = np.mean([a.mean() - b.mean() for a, b in pairs])
    return d, *np.percentile(out, [2.5, 97.5]), min(1.0, 2 * min((out <= 0).mean(), (out >= 0).mean()))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o): run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj

hr("(1) train/validation assignment of the FISM2-input and GOES-input pools (key = peak minute + class)")
ft, fv, gt, gv = (set(P[i]["key"][P[i][s]]) for i, s in (("default", "tr"), ("default", "va"), ("goes_b", "tr"), ("goes_b", "va")))
print(f"  FISM2 pool train {len(ft)} val {len(fv)} | GOES pool train {len(gt)} val {len(gv)} | unique keys ok: {len(ft)==len(F['tr']) and len(gt)==len(G['tr'])}")
print(f"  common train {len(ft & gt)}, common val {len(fv & gv)}, GOES-val in FISM2-train {len(gv & ft)}, FISM2-val in GOES-train {len(fv & gt)}")
print(f"  FISM2-pool flares without GOES window: {len((ft | fv) - (gt | gv))}: {sorted((ft | fv) - (gt | gv))}")

hr("(2) converter gains, 5 architectures x 5 seeds, hierarchical bootstrap B = 10000 (own implementation)")
ok = ~np.isnan(oof(EXP["GOES"][1], "GRU-z")).any(axis=(0, 2, 3)); yT = G["Y"][96:192][ok]; print("  rebuilt test flares with a GOES window:", ok.sum())
T = {k: {m: mape4(oof(e, m)[:, ok], yT) for m in ARCH} for k, (i, e) in EXP.items()}
V = {}
for k, (i, e) in EXP.items():
    V[k] = {}
    for m in ARCH:
        L, idx = val(e, m); V[k][m] = mape4(L, P[i]["Y"][idx])
rows = []
for a_, b_ in (("GOES", "FISM2"), ("linear", "GOES"), ("FCNconv", "GOES"), ("FCNconv", "linear"), ("FCNconv", "FISM2")):
    for lab, get in (("test95", lambda k, m: T[k][m]), ("val122", lambda k, m: V[k][m]), ("test+val217", lambda k, m: np.concatenate([T[k][m], V[k][m]], 1))):
        paired = not (b_ == "FISM2" and lab != "test95")
        d, lo, hi, p = hboot([(get(a_, m), get(b_, m)) for m in ARCH], paired=paired)
        rows.append(dict(comp=f"{a_}-{b_}", set=lab, paired=paired, d=d, lo=lo, hi=hi, p=p,
                         per_arch=" ".join(f"{get(a_, m).mean() - get(b_, m).mean():+.2f}" for m in ARCH)))
D2 = pd.DataFrame(rows); fam = D2[(D2.set == "test95") & D2.comp.isin(["linear-GOES", "FCNconv-GOES", "FCNconv-linear"])]
D2.loc[fam.index, "p_holm3"] = holm(fam.p.values); print(D2.round(4).to_string(index=False))
print("  converter recovery on test95 (FCNconv-GOES)/(GOES-FISM2): %.0f %%" % (100 * -D2[(D2.comp=='FCNconv-GOES')&(D2.set=='test95')].d.iloc[0] / D2[(D2.comp=='GOES-FISM2')&(D2.set=='test95')].d.iloc[0]))
print("  manuscript reference rows (paper1_inputs_hboot.csv, latis test):")
H = pd.read_csv(f"{R}/full_server/paper1_inputs_hboot.csv"); print(H[H.test.str.contains("latis|rebuilt", regex=True)].round(3).to_string(index=False)[:3000])

hr("(3) 5-seed ensemble MAPE4: validation vs rebuilt test")
rows = []
for k, (i, e) in EXP.items():
    for m in ("GRU-z", "FCN-z"):
        L, idx = val(e, m); Lt = oof(e, m)
        okk = ~np.isnan(Lt).any(axis=(0, 2, 3))
        rows.append(dict(input=k, model=m, n_val=len(idx), val=mape4(L.mean(0), P[i]["Y"][idx]).mean(), n_test=okk.sum(),
                         test=mape4(Lt[:, okk].mean(0), P[i]["Y"][96:192][okk]).mean(), test95=mape4(Lt[:, ok].mean(0), yT).mean()))
print(pd.DataFrame(rows).round(2).to_string(index=False))

hr("(4) level / shape decomposition, ensembles, 95 test flares (shape = after removing one log10 offset per flare and band)")
rows, SH = [], {}
for k, (i, e) in EXP.items():
    for m in ("GRU-z", "FCN-z"):
        res = oof(e, m)[:, ok].mean(0) - np.log10(yT); off = res.mean(1, keepdims=True)
        asis = 100 * np.abs(10 ** res - 1).mean(1); shape = 100 * np.abs(10 ** (res - off) - 1).mean(1)   # (n, 4)
        SH[(k, m)] = (asis.mean(1), shape.mean(1))
        rows.append(dict(input=k, model=m, **{f"asis{b}": asis[:, b].mean() for b in range(4)}, MAPE4=asis.mean(),
                         **{f"shape{b}": shape[:, b].mean() for b in range(4)}, shape4=shape.mean(),
                         **{f"rmsoff{b}": np.sqrt((off[:, 0, b] ** 2).mean()) for b in range(4)},
                         **{f"meanoff{b}": off[:, 0, b].mean() for b in range(4)}))
print(pd.DataFrame(rows).round(3).to_string(index=False))
rng = np.random.default_rng(4)
for m in ("GRU-z", "FCN-z"):
    for a_ in ("GOES", "FCNconv"):
        da = SH[(a_, m)][0] - SH[("FISM2", m)][0]; ds = SH[(a_, m)][1] - SH[("FISM2", m)][1]; n = len(da)
        bi = rng.integers(0, n, (10000, n))
        print(f"  {m} {a_}-FISM2: as is {da.mean():+.2f} [{np.percentile(da[bi].mean(1), 2.5):+.2f}, {np.percentile(da[bi].mean(1), 97.5):+.2f}]"
              f"   shape only {ds.mean():+.2f} [{np.percentile(ds[bi].mean(1), 2.5):+.2f}, {np.percentile(ds[bi].mean(1), 97.5):+.2f}]  (flare bootstrap)")

hr("(5) multi-level coverage, operational models (GOES + FCN converter), 0.1-0.8 nm and all bands, 95 test flares, flare bootstrap")
ly = np.log10(yT)
for m in ("GRU-z", "FCN-z"):
    Lv, idx = val(EXP["FCNconv"][1], m); Lq = oof(EXP["FCNconv"][1], m)[:, ok]; s = []
    for lev in (0.5, 0.8, 0.9, 0.95):
        _, lo, hi = ap7.conf_norm(Lv, np.log10(G["Y"][idx]), Lq, level=lev); ins = (ly >= lo) & (ly <= hi)
        e0 = ins[:, :, 0].mean(1); bt = e0[np.random.default_rng(5).integers(0, len(e0), (4000, len(e0)))].mean(1)
        s.append(f"{lev:.2f}: b0 {e0.mean():.3f} [{np.percentile(bt, 2.5):.3f}, {np.percentile(bt, 97.5):.3f}] all {ins.mean():.3f}")
    print(f"  {m}: " + " | ".join(s))

hr("(6) operational GRU-z ensemble MAPE4 by satellite series and period")
d = np.load(f"{REPO}/data/processed/events_latis.npz", allow_pickle=False)
row_of = {int(d["test_k"][i]): int(i) for i in np.where(d["is_test"])[0]}; trow = np.array([row_of[k] for k in np.where(ok)[0]])
e4 = mape4(oof(EXP["FCNconv"][1], "GRU-z")[:, ok].mean(0), yT); sat = d["goes_sat"][trow]; yr = pd.to_datetime(d["peak"][trow].astype(str)).year.values
rng = np.random.default_rng(6)
def ci(v): b = v[rng.integers(0, len(v), (4000, len(v)))].mean(1); return f"n={len(v):3d} {v.mean():6.2f} [{np.percentile(b, 2.5):.2f}, {np.percentile(b, 97.5):.2f}]"
print("  all      ", ci(e4))
for lab, mk in (("GOES 8-12", sat <= 12), ("GOES 13-15", (sat >= 13) & (sat <= 15)), ("GOES 16-18", sat >= 16)): print(f"  {lab:10s}", ci(e4[mk]))
new, old = e4[sat >= 16], e4[sat < 16]
bd = new[rng.integers(0, len(new), (10000, len(new)))].mean(1) - old[rng.integers(0, len(old), (10000, len(old)))].mean(1)
print(f"  GOES 16-18 minus 8-15: {new.mean() - old.mean():+.2f} [{np.percentile(bd, 2.5):+.2f}, {np.percentile(bd, 97.5):+.2f}]")
gy = pd.to_datetime(G["meta"].peak_time.values[G["tr"]]).year; print(f"  GOES-pool training flares <=2016: {(gy <= 2016).sum()}, >=2017: {(gy >= 2017).sum()}")

hr("(7) test flares vs the isolation rule (another M/X flare of our list within [-60, +180] min)")
pk = pd.to_datetime(pd.Series(d["peak"].astype(str))).values.astype("datetime64[m]").astype(np.int64); cls = d["cls"].astype(str)
viol = []
for i in np.where(d["is_test"])[0]:
    dt = pk - pk[i]; j = np.where(((dt > 0) & (dt <= 180)) | ((dt < 0) & (dt >= -60)))[0]
    if len(j): viol.append((str(d["event_id"][i]), "; ".join(f"{cls[q]} at {dt[q]:+d} min" for q in j)))
print(f"  {len(viol)} of {int(d['is_test'].sum())}:"); [print("   ", v) for v in viol]
iso_all = np.array([not np.any(((pk - pk[i] > 0) & (pk - pk[i] <= 180)) | ((pk - pk[i] < 0) & (pk - pk[i] >= -60))) for i in range(len(pk))])
print(f"  all flares of the list: {len(pk)}, failing the rule: {(~iso_all).sum()} ({(~iso_all).mean():.1%})")
for ev in ("20130412_M3.3", "20150114_M2.2", "20140108_M3.6", "20030613_M3.1"):
    i = np.where(d["event_id"].astype(str) == ev)[0]
    if len(i):
        i = int(i[0]); dt = pk - pk[i]; dt = dt[dt != 0]
        print(f"   {ev}: test={bool(d['is_test'][i])} sat={int(d['goes_sat'][i])} prev {-(dt[dt < 0].max()) if (dt < 0).any() else None} min, next {dt[dt > 0].min() if (dt > 0).any() else None} min")

hr("(8) catalogue class vs science-quality XRS-B peak")
pe = pd.read_csv(f"{R}/full_server/peak_detector_events_v3.csv", usecols=["event", "cls", "is_test", "sat", "peak_flux"])
cv = pe.cls.str[1:].astype(float) * np.where(pe.cls.str[0] == "X", 1e-4, 1e-5); r = pe.peak_flux / cv
for lab, mk in (("GOES 8-12", pe.sat <= 12), ("GOES 13-15", (pe.sat >= 13) & (pe.sat <= 15)), ("GOES 16-18", pe.sat >= 16)):
    print(f"  {lab:10s} n={mk.sum():4d} median {r[mk].median():.3f} p10-p90 {r[mk].quantile(.1):.2f}-{r[mk].quantile(.9):.2f}")
o = pe.sat <= 15; print(f"  GOES 8-15 catalogue class < M1.43: {(cv[o] < 1.43e-5).sum()} / {o.sum()} ({(cv[o] < 1.43e-5).mean():.1%}); science peak < 1.43e-5: {(pe.peak_flux[o] < 1.43e-5).sum()}")

hr("A5. armed detector (k = 2, rise 2, raw) with the threshold on the catalogue scale (1.43e-5 for GOES 8-15)")
S = np.load(f"{R}/full_server/peak_detector_streams.npz", allow_pickle=False); pre = int(S["pre"])
def run(thr_old):
    rec = []
    for j in range(len(S["row"])):
        thr = thr_old if S["sat"][j] <= 15 else 1e-5
        dec = pdet.detect_all(S["xrsb"][j], 2, thr, rise_factor=2.0); q = pdet.own_peak(dec, pre, 5)
        rec.append(dict(test=bool(S["is_test"][j]), sat=int(S["sat"][j]), own=q["own"], ndecl=len(dec), nother=q["n_before"] + q["n_after"], none=not dec))
    return pd.DataFrame(rec)
for thr in (1e-5, 1.43e-5):
    E = run(thr)
    for lab, mk in (("all", np.ones(len(E), bool)), ("test", E.test.values), ("GOES 8-15", E.sat.values <= 15)):
        e = E[mk]; print(f"  thr(8-15)={thr:.2e} {lab:9s} n={len(e):4d} own declared {e.own.notna().mean():.4f} median latency {e.own.median():.1f} "
                         f"on time (0-4 min) {((e.own >= 0) & (e.own <= 4)).mean():.4f} declarations/stream {e.ndecl.mean():.3f} other-peak declarations/stream {e.nother.mean():.3f}")

hr("A3. no-training (closed-form) baselines on the rebuilt test flares, GOES XRS-B input and FISM2 input")
def baselines(Pp, okm):
    X, Y, tr = Pp["X"], Pp["Y"], Pp["tr"]; ly_tr = np.log10(Y[tr]); xp_tr = np.log10(np.nanmax(X[tr], 1))
    xt = np.log10(np.nanmax(X[96:192][okm], 1)); yt = Y[96:192][okm]
    clim = np.broadcast_to(ly_tr.mean(0), yt.shape)                                 # training-mean profile
    A_ = np.c_[np.ones_like(xp_tr), xp_tr]; coef, *_ = np.linalg.lstsq(A_, ly_tr.reshape(len(tr), -1), rcond=None)
    plaw = (np.c_[np.ones_like(xt), xt] @ coef).reshape(yt.shape)                    # log y_b(t) = a_b(t) + c_b(t) log x_peak
    last = np.log10(X[96:192][okm][:, -1]); A2 = np.c_[np.ones_like(xp_tr), np.log10(X[tr][:, -1]), np.log10(np.nanmin(X[tr], 1))]
    coef2, *_ = np.linalg.lstsq(A2, ly_tr.reshape(len(tr), -1), rcond=None)
    plaw2 = (np.c_[np.ones_like(xt), last, np.log10(np.nanmin(X[96:192][okm], 1))] @ coef2).reshape(yt.shape)
    out = {}
    for lab, L in (("climatology", clim), ("power law of the window maximum", plaw), ("power law of last minute + window minimum", plaw2)):
        e = mape4(L, yt); out[lab] = e; per = 100 * np.mean(np.abs(10 ** L - yt) / yt, axis=1).mean(0)
        print(f"    {lab:42s} MAPE4 {e.mean():6.2f}  bands " + " ".join(f"{v:6.2f}" for v in per))
    return out
print("  FISM2 0.1-0.8 nm input (benchmark), 95 flares:"); BF = baselines(F, ok)
print("  GOES XRS-B input (operational), 95 flares:"); BG = baselines(G, ok)
print(f"  operational GRU-z ensemble on the same 95: {e4.mean():.2f}; FISM2-input GRU-z ensemble: {mape4(oof(EXP['FISM2'][1], 'GRU-z')[:, ok].mean(0), yT).mean():.2f}")
print(f"done in {time.time() - t0:.0f} s")
