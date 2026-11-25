"""Paper-1 P7 / P8 (patch_v14 run): delayed issue with the inputs available in real time, and forecasts for the
flares outside the isolation rule. The analysis was fixed before the run (docs/decisions.md, 2026-11-12).

    python notebooks/analyze_p7_p8.py --results results --events data/processed/events_latis.npz

P7 (results/p2_zscore_full[_in-<input>]_d<k>_p7, GRU-z and FCN-z, 5 seeds; k = 2, 5, 10 min):
  paper1_p7_delay_hboot.csv   MAPE4 for lead >= k, delayed vs peak-issued model of the same input and architecture:
                              difference, 95 % interval and p from a bootstrap over events (paired) and seeds; Holm
                              adjustment over the 18 comparisons on the rebuilt test set. family "vs_fism2_peak" (not
                              part of the pre-specified set) compares the delayed GOES-driven models with the peak-issued
                              FISM2-input model on the same leads.
  paper1_p7_drap_hboot.csv    the same comparison for the 10 MHz absorption error over Korea (daytime, lead >= k)
  paper1_p7_conformal.csv     coverage and width of the 90 % split-conformal intervals (calibrated on the validation
                              events of each model) on lead >= k, delayed and peak-issued
P8 (results/p2_zscore_full[_in-goes_convnn]_p8: peak-issued models retrained with the same seeds, oofN_* = flares
outside the isolation rule, never used for fitting):
  paper1_p8_groups.csv        MAPE4, interval coverage and width per group (all / reason for failing the rule / class /
                              no train-val flare within 240 min) and the difference from the test flares (bootstrap over
                              the events of both groups and the seeds)
  paper1_p8_blocks.csv        MAPE4 per 30-min block of lead time, per group
  paper1_p8_events.csv        per flare: catalogue context, MAPE4 and interval coverage per input and architecture
  paper1_p8_recal.csv         intervals recalibrated for all flares (docs/decisions.md 2026-11-12): calibration events = the
                              validation flares + flares outside the rule, weighted to their share of all flares; the flares
                              outside the rule are evaluated out-of-fold (time clusters, two folds). Coverage and width for the
                              test flares, the flares outside the rule and the population-weighted mean, per calibration
  paper1_p8_recal_blocks.csv  the same per 30-min block of lead time
  paper1_p8_reissue.csv       re-issue at the next declaration (docs/decisions.md 2026-11-12): each forecast is scored on the
                              minutes for which it is in force, i.e. until the armed detector declares the next maximum more
                              than 5 min after the peak (needs results/full_server/peak_detector_streams.npz). MAPE and interval
                              coverage pooled over the event-minutes in force, per group, against the full 180 minutes
  paper1_p8_reissue_tail.csv  the last 10 minutes in force before the next declaration against the earlier ones
  paper1_p8_reissue_blocks.csv  per 30-min block, minutes in force only
  paper1_p8_reissue_events.csv  minute of the next declaration per flare (rise factors 2, 1.5, 1.2)
  paper1_p8_reissue_aligned.csv coverage and error of the 0.1-0.8 nm band minute by minute relative to the next declaration
                              (flares with a next declaration; descriptive)
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import math
import os
import sys
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("afs", ROOT / "notebooks/analyze_full_server.py")
afs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(afs)
_spec = importlib.util.spec_from_file_location("rt", ROOT / "notebooks/replicate_tests.py")
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)

_spec = importlib.util.spec_from_file_location("pdet", ROOT / "notebooks/peak_detector.py")
pdet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pdet)

THR, PRE, TOL = 1e-5, 60, 5          # detector threshold (M1), stream index of the dataset peak, own-peak tolerance (min); as in P4c
B = ["w0", "w1", "w2", "w3"]
PEAK = {"fism": "p2_zscore_full_gpu5", "goes_convnn": "p2_zscore_full_in-goes_convnn_p2nn", "goes_b": "p2_zscore_full_in-goes_b_p1"}
LABEL = {"fism": "FISM2", "goes_convnn": "GOES + FCN converter", "goes_b": "GOES XRS-B direct"}
MODELS = ("GRU-z", "FCN-z")
BLOCKS = [slice(i * 30, (i + 1) * 30) for i in range(6)]


def p7_exp(inp: str, k: int) -> str:
    return f"p2_zscore_full{'' if inp == 'fism' else '_in-' + inp}_d{k}_p7"


def p8_exp(inp: str) -> str:
    return f"p2_zscore_full{'' if inp == 'fism' else '_in-' + inp}_p8"


def cq(r, level=0.9):
    n = len(r)
    return np.sort(r)[min(n, int(math.ceil((n + 1) * level))) - 1]


def conf_norm(Lv, yv, Lt, level=0.9):
    """Split conformal interval of notebooks/split_conformal_full.py ('conf-norm', no conditioning).
    Lv (S, n_val, 180, 4), yv (n_val, 180, 4), Lt (S, n, 180, 4): log10. Returns mu_t, lo, hi (n, 180, 4)."""
    mu_v, sd_v = Lv.mean(0), Lv.std(0)
    res_v = yv - mu_v
    mu_t, sd_t = Lt.mean(0), Lt.std(0)
    lo, hi = np.zeros_like(mu_t), np.zeros_like(mu_t)
    for b in range(4):
        for bl in BLOCKS:
            s0 = np.median(sd_v[:, bl, b])
            qn = cq((np.abs(res_v[:, bl, b]) / (sd_v[:, bl, b] + s0)).ravel(), level)
            half = qn * (sd_t[:, bl, b] + s0)
            lo[:, bl, b], hi[:, bl, b] = mu_t[:, bl, b] - half, mu_t[:, bl, b] + half
    return mu_t, lo, hi


P_OUTSIDE = 398 / 1604          # share of the flares outside the isolation rule among the M/X flares with a GOES series


def wq(scores, w, level=0.9):
    """Weighted conformal quantile: the smallest score whose cumulative weight share times n / (n + 1) reaches `level`.
    With equal weights this is cq()."""
    o = np.argsort(scores, kind="stable")
    sc, cw = scores[o], np.cumsum(w[o]) / w.sum()
    n = len(sc)
    return sc[min(int(np.searchsorted(cw * n / (n + 1), level - 1e-12, side="left")), n - 1)]


def conf_norm_w(Lc, yc, wc, Lt, level=0.9):
    """conf_norm() with event weights wc (n_cal,) on the calibration events. Lc (S, n_cal, 180, 4), yc (n_cal, 180, 4): log10."""
    mu_c, sd_c = Lc.mean(0), Lc.std(0)
    res = np.abs(yc - mu_c)
    mu_t, sd_t = Lt.mean(0), Lt.std(0)
    lo, hi = np.zeros_like(mu_t), np.zeros_like(mu_t)
    for b in range(4):
        for bl in BLOCKS:
            s0 = np.median(sd_c[:, bl, b])
            sc = res[:, bl, b] / (sd_c[:, bl, b] + s0)
            half = wq(sc.ravel(), np.repeat(wc, sc.shape[1]), level) * (sd_t[:, bl, b] + s0)
            lo[:, bl, b], hi[:, bl, b] = mu_t[:, bl, b] - half, mu_t[:, bl, b] + half
    return mu_t, lo, hi


def time_clusters(peak_min, gap_min=1440):
    """Cluster label per flare (increasing in time): successive peaks less than gap_min minutes apart share a cluster."""
    o = np.argsort(peak_min, kind="stable")
    lab = np.zeros(len(peak_min), int)
    lab[o] = np.concatenate([[0], np.cumsum(np.diff(peak_min[o]) >= gap_min)])
    return lab


def next_declaration(b, k=2, rise=2.0):
    """Minute, relative to the dataset peak (stream index PRE), at which the armed detector declares the first maximum that
    lies more than TOL minutes after the dataset peak; 180 if there is none."""
    later = [q["t"] - PRE for q in pdet.detect_all(b, k, THR, rise_factor=rise) if q["t_max"] > PRE + TOL]
    return min(later) if later else 180


def in_force(rel, ins, T, censor, rng, n_boot=2000, lo=None, hi=None):
    """Per-event sums over the minutes a forecast is in force and their bootstrap. rel (S, n, 180, 4) errors in percent,
    ins (n, 180, 4) inside-the-interval flags, T (n,) number of minutes in force (minutes 0..T-1). censor=False scores all 180
    minutes. lo / hi restrict the minutes further to [lo, hi) per event (arrays) - for the tail and block tables.
    Returns point values of the per-event mean sums (dict), and bootstrap replicates of the same (dict of arrays)."""
    t = np.arange(180)[None, :]
    V = t < (T[:, None] if censor else np.full((len(T), 1), 180))          # (n, 180) in both cases
    if lo is not None:
        V = V & (t >= lo[:, None]) & (t < hi[:, None])
    den = V.sum(1).astype(float)
    per = dict(den=den, MAPE4=(rel.mean(-1) * V).sum(2).mean(0), MAPE_w0=(rel[..., 0] * V).sum(2).mean(0),
               cov_w0=(ins[:, :, 0] * V).sum(1).astype(float), cov_4band=(ins.mean(-1) * V).sum(1))
    idx = rng.integers(0, len(den), (n_boot, len(den)))
    sidx = rng.integers(0, rel.shape[0], (n_boot, rel.shape[0]))
    m4 = (rel.mean(-1) * V).sum(2)                                     # (S, n)
    boot = dict(den=den[idx].mean(1), cov_w0=per["cov_w0"][idx].mean(1),
                MAPE4=np.array([m4[sidx[i]][:, idx[i]].mean() for i in range(n_boot)]))
    return {k: float(v.mean()) for k, v in per.items()}, boot


def seed_multisets(S: int):
    """All multisets of S seeds drawn with replacement and their probabilities (126 for S = 5)."""
    combos = list(combinations_with_replacement(range(S), S))
    w = np.array([math.factorial(S) / np.prod([math.factorial(c.count(s)) for s in set(c)]) for c in combos]) / S ** S
    return combos, w


def uboot(A, T, n=10000, seed=0):
    """A (S, nA), T (S, nT) event scores of two disjoint event groups: difference of the means, 95 % interval and p from a
    bootstrap over the events of each group and the seeds of each."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        out[i] = (A[rng.integers(0, len(A), len(A))][:, rng.integers(0, A.shape[1], A.shape[1])].mean()
                  - T[rng.integers(0, len(T), len(T))][:, rng.integers(0, T.shape[1], T.shape[1])].mean())
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(A.mean() - T.mean()), float(lo), float(hi), float(min(1.0, max(2 * min((out <= 0).mean(), (out >= 0).mean()), 1.0 / n)))


def mboot(A, n=10000, seed=0):
    """95 % interval of the mean of A (S, nA): bootstrap over events and seeds."""
    rng = np.random.default_rng(seed)
    out = np.array([A[rng.integers(0, len(A), len(A))][:, rng.integers(0, A.shape[1], A.shape[1])].mean() for _ in range(n)])
    return tuple(float(v) for v in np.percentile(out, [2.5, 97.5]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--events", default=str(ROOT / "data/processed/events_latis.npz"))
    ap.add_argument("--goes-post", default=str(ROOT / "data/processed/events_goes_post.npz"))
    ap.add_argument("--out", default=None, help="default: <results>/full_server")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--streams", default=None, help="default: <results>/full_server/peak_detector_streams.npz (re-issue tables; skipped if absent)")
    a = ap.parse_args(argv)
    R = Path(a.results)
    OUT = Path(a.out or R / "full_server")
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.update(FISM_PROTOCOL="full", FISM_EVENTS=str(Path(a.events).resolve()), FISM_GOES_POST=str(Path(a.goes_post).resolve()),
                      FISM_DEVICE_QUIET="1", FISM_EXCLUDE_NEAR_TEST_H="0", FISM_EXCLUDE_RANDOM_N="0", FISM_EXCLUDE_SAME_AR="0")
    from src.data import pool
    from src.data.cv96 import load_meta
    from src.eval.drap import SITE_KOREA, absorption_db, solar_zenith_deg

    pools = {}

    def get_pool(inp: str, k: int = 0, extra: int = 0):
        """Pool of an experiment. The converter inputs use the rows of the GOES XRS-B pool (an event is usable for them
        exactly when its XRS-B window is complete), so no converter has to be fitted here."""
        key = ("default" if inp == "fism" else "goes_b", 0 if inp == "fism" else k, extra)
        if key not in pools:
            os.environ.update(FISM_INPUT=key[0], FISM_ISSUE_DELAY=str(key[1]), FISM_PREDICT_EXTRA=str(key[2]))
            pool.load_full.cache_clear()
            X, Y, names, meta, tr, va, te, teL = pool.load_full()
            pools[key] = dict(Y=Y, names=names, meta=meta, tr=tr, va=va, teL=teL, ex=pool.extra_index())
            print(f"[pool] {key}: {pool.describe()}", flush=True)
        return pools[key]

    P0 = get_pool("fism")
    YT = {"zenodo96": P0["Y"][:96], "rebuilt96": P0["Y"][96:192]}
    isx = P0["meta"].is_x.values[:96].astype(bool)

    def preds(exp, m, pre="oofL"):
        fs = sorted(glob.glob(f"{R}/{exp}/{pre}_{m}_s*.npy"))
        return np.stack([np.load(f, allow_pickle=False).astype(np.float64) for f in fs]) if fs else None      # log10

    def ev_mape(L, yt, k0=0, bands=(0, 1, 2, 3)):
        """(S, n) event MAPE over the bands, lead >= k0; NaN for events without a prediction."""
        out = np.full(L.shape[:2], np.nan)
        ok = ~np.isnan(L).any(axis=(0, 2, 3))
        rel = np.abs(10 ** L[:, ok][:, :, k0:, :][..., list(bands)] - yt[ok][None, :, k0:, :][..., list(bands)]) / yt[ok][None, :, k0:, :][..., list(bands)]
        out[:, ok] = 100 * rel.mean(axis=(2, 3))
        return out

    def val(exp, m, inp, k):
        fs = sorted(glob.glob(f"{R}/{exp}/val_{m}_s*.npz"))
        ds = [np.load(f, allow_pickle=False) for f in fs]
        idx = ds[0]["idx"]
        assert all(np.array_equal(d["idx"], idx) for d in ds)
        P = get_pool(inp, k)
        assert np.array_equal(np.sort(idx), np.sort(P["va"])), f"{exp}: validation rows are not the validation split of the pool"
        return np.stack([d["pred_log"].astype(np.float64) for d in ds]), np.log10(P["Y"][idx])

    have_p7 = all(glob.glob(f"{R}/{p7_exp(i, k)}/oofL_GRU-z_s*.npy") for i in PEAK for k in (2, 5, 10))
    have_p8 = all(glob.glob(f"{R}/{p8_exp(i)}/oofN_GRU-z_s*.npy") for i in ("fism", "goes_convnn"))

    # ------------------------------------------------------------------------------------------------ P7: MAPE4
    if have_p7:
        rows = []
        for test in ("rebuilt96", "zenodo96"):
            pre = "oofL" if test == "rebuilt96" else "oof"
            yt = YT[test]
            for inp in PEAK:
                for m in MODELS:
                    Lb = preds(PEAK[inp], m, pre)
                    Lf = preds(PEAK["fism"], m, pre)
                    for k in (2, 5, 10):
                        Ld = preds(p7_exp(inp, k), m, pre)
                        A, C = ev_mape(Ld, yt, k), ev_mape(Lb, yt, k)
                        ok = np.isfinite(A).all(0) & np.isfinite(C).all(0)
                        d, lo, hi, p = rt.hboot(A, C, n=a.n_boot, seed=7)
                        A0, C0 = ev_mape(Ld, yt, k, (0,)), ev_mape(Lb, yt, k, (0,))
                        d0, lo0, hi0, p0 = rt.hboot(A0, C0, n=a.n_boot, seed=8)
                        ens = lambda L: float(np.nanmean(ev_mape(L.mean(0, keepdims=True), yt, k)[:, ok]))        # noqa: E731
                        rows.append(dict(family="delayed_vs_peak", test=test, input=inp, model=m, k=k, n_ev=int(ok.sum()), n_seeds_delayed=len(A), n_seeds_peak=len(C),
                                         MAPE4_delayed=A[:, ok].mean(), MAPE4_peak_issued_same_leads=C[:, ok].mean(), d=d, lo=lo, hi=hi, p_hboot=p,
                                         p_event=rt.perm((A.mean(0) - C.mean(0))[ok]), sd_seed_delayed=A[:, ok].mean(1).std(ddof=1), sd_seed_peak=C[:, ok].mean(1).std(ddof=1),
                                         ensMAPE4_delayed=ens(Ld), ensMAPE4_peak_issued_same_leads=ens(Lb),
                                         MAPE_w0_delayed=A0[:, ok].mean(), MAPE_w0_peak_issued_same_leads=C0[:, ok].mean(), d_w0=d0, lo_w0=lo0, hi_w0=hi0, p_hboot_w0=p0))
                        if inp != "fism":       # not pre-specified: delayed GOES-driven model vs the peak-issued FISM2-input model
                            Cf = ev_mape(Lf, yt, k)
                            okf = np.isfinite(A).all(0) & np.isfinite(Cf).all(0)
                            d, lo, hi, p = rt.hboot(A, Cf, n=a.n_boot, seed=9)
                            rows.append(dict(family="vs_fism2_peak", test=test, input=inp, model=m, k=k, n_ev=int(okf.sum()), n_seeds_delayed=len(A), n_seeds_peak=len(Cf),
                                             MAPE4_delayed=A[:, okf].mean(), MAPE4_peak_issued_same_leads=Cf[:, okf].mean(), d=d, lo=lo, hi=hi, p_hboot=p,
                                             p_event=rt.perm((A.mean(0) - Cf.mean(0))[okf])))
                    if inp != "fism":           # reference: the peak-issued GOES-driven model vs the peak-issued FISM2-input model, all leads
                        A, Cf = ev_mape(Lb, yt, 0), ev_mape(Lf, yt, 0)
                        okf = np.isfinite(A).all(0) & np.isfinite(Cf).all(0)
                        d, lo, hi, p = rt.hboot(A, Cf, n=a.n_boot, seed=9)
                        rows.append(dict(family="vs_fism2_peak", test=test, input=inp, model=m, k=0, n_ev=int(okf.sum()), n_seeds_delayed=len(A), n_seeds_peak=len(Cf),
                                         MAPE4_delayed=A[:, okf].mean(), MAPE4_peak_issued_same_leads=Cf[:, okf].mean(), d=d, lo=lo, hi=hi, p_hboot=p,
                                         p_event=rt.perm((A.mean(0) - Cf.mean(0))[okf])))
        D = pd.DataFrame(rows)
        fam = (D.family == "delayed_vs_peak") & (D.test == "rebuilt96")
        D["p_holm"] = np.nan
        D.loc[fam, "p_holm"] = afs.holm(D.loc[fam, "p_hboot"].values)
        D["p_holm_w0"] = np.nan
        D.loc[fam, "p_holm_w0"] = afs.holm(D.loc[fam, "p_hboot_w0"].values)
        D.to_csv(OUT / "paper1_p7_delay_hboot.csv", index=False)
        print(f"paper1_p7_delay_hboot.csv: {len(D)} rows ({int(fam.sum())} in the pre-specified family)")

        # -------------------------------------------------------------------------------------------- P7: D-RAP
        meta = load_meta()
        tt = np.array([np.datetime64(t, "s") for t in meta.peak_time.values[:96]])[:, None] + np.arange(180).astype("timedelta64[m]")[None, :]
        ZEN = solar_zenith_deg(tt, *SITE_KOREA)
        day = ZEN[:, 0] < 90
        yt = YT["rebuilt96"]
        AT = absorption_db(yt[:, :, 0], ZEN)
        rows, rng = [], np.random.default_rng(11)
        for inp in PEAK:
            for m in MODELS:
                Lb = preds(PEAK[inp], m)[..., 0]
                for k in (2, 5, 10):
                    Ld = preds(p7_exp(inp, k), m)[..., 0]                              # (S, 96, 180) log10 w0
                    idx = np.where(day & np.isfinite(Ld).all((0, 2)) & np.isfinite(Lb).all((0, 2)))[0]
                    msk = ((ZEN < 90) & (np.arange(180)[None, :] >= k))[idx]

                    def table(L):                                                       # per-event MAE (dB) for every seed multiset
                        combos, w = seed_multisets(len(L))
                        E = np.empty((len(combos), len(idx)))
                        for j, c in enumerate(combos):
                            err = np.abs(absorption_db(10 ** L[list(c)][:, idx].mean(0), ZEN[idx]) - AT[idx])
                            E[j] = np.nanmean(np.where(msk, err, np.nan), axis=1)
                        full = np.abs(absorption_db(10 ** L[:, idx].mean(0), ZEN[idx]) - AT[idx])
                        return E, w, float(np.nanmean(np.where(msk, full, np.nan), axis=1).mean())
                    Ed, wd, fd = table(Ld)
                    Eb, wb, fb = table(Lb)
                    cd, cb = rng.choice(len(wd), a.n_boot, p=wd), rng.choice(len(wb), a.n_boot, p=wb)
                    ee = rng.integers(0, len(idx), (a.n_boot, len(idx)))
                    bt = np.array([Ed[cd[i], ee[i]].mean() - Eb[cb[i], ee[i]].mean() for i in range(a.n_boot)])
                    lo, hi = np.percentile(bt, [2.5, 97.5])
                    rows.append(dict(test="rebuilt96", input=inp, model=m, k=k, n_day=len(idx), MAE_dB_delayed=fd, MAE_dB_peak_issued_same_leads=fb, d=fd - fb, lo=lo, hi=hi,
                                     p_hboot=min(1.0, max(2 * min((bt <= 0).mean(), (bt >= 0).mean()), 1.0 / a.n_boot))))
        Dr = pd.DataFrame(rows)
        Dr["p_holm"] = afs.holm(Dr.p_hboot.values)
        Dr.to_csv(OUT / "paper1_p7_drap_hboot.csv", index=False)
        print(f"paper1_p7_drap_hboot.csv: {len(Dr)} rows")

        # -------------------------------------------------------------------------------------------- P7: conformal
        rows = []
        ytl = np.log10(YT["rebuilt96"])
        saved = np.load(OUT / "conformal_intervals_none.npz", allow_pickle=False) if (OUT / "conformal_intervals_none.npz").exists() else None
        worst = 0.0
        for inp in PEAK:
            for m in MODELS:
                for k in (0, 2, 5, 10):
                    exp = PEAK[inp] if k == 0 else p7_exp(inp, k)
                    Lv, yv = val(exp, m, inp, k)
                    Lt = preds(exp, m)
                    ok = ~np.isnan(Lt).any(axis=(0, 2, 3))
                    mu, lo, hi = conf_norm(Lv, yv, Lt)
                    if k == 0 and saved is not None and f"{m} [{exp}]|rebuilt96|lo" in saved.files:        # same rule as the server table
                        worst = max(worst, float(np.nanmax(np.abs(lo[ok] - saved[f"{m} [{exp}]|rebuilt96|lo"][ok]))),
                                    float(np.nanmax(np.abs(hi[ok] - saved[f"{m} [{exp}]|rebuilt96|hi"][ok]))))
                    inside = (ytl >= lo) & (ytl <= hi)
                    for k0 in ((0, 2, 5, 10) if k == 0 else (k,)):
                        rows.append(dict(input=inp, model=m, issued_min=k, lead_from_min=k0, n_members=len(Lt), n_val=Lv.shape[1], n_test=int(ok.sum()),
                                         cov_4band=inside[ok][:, k0:].mean(), width_4band_dex=(hi - lo)[ok][:, k0:].mean(),
                                         cov_w0_Xclass=inside[ok & isx][:, k0:, 0].mean(), cov_w0_Mclass=inside[ok & ~isx][:, k0:, 0].mean(),
                                         **{f"cov_{b}": inside[ok][:, k0:, j].mean() for j, b in enumerate(B)},
                                         **{f"width_{b}_dex": (hi - lo)[ok][:, k0:, j].mean() for j, b in enumerate(B)}))
        pd.DataFrame(rows).to_csv(OUT / "paper1_p7_conformal.csv", index=False)
        print(f"paper1_p7_conformal.csv: {len(rows)} rows" + (f"; peak-issued intervals vs conformal_intervals_none.npz: max |difference| {worst:.1e} dex" if saved is not None else ""))

    # ------------------------------------------------------------------------------------------------ P8
    if have_p8:
        d = np.load(a.events, allow_pickle=False)
        cat = pd.read_csv(ROOT / "data/processed/flare_catalogue.csv", parse_dates=["peak"])
        ev_tab, grp_rows, blk_rows, rc_rows, rcb_rows = None, [], [], [], []
        rng_rc = np.random.default_rng(31)
        ri_rows, rit_rows, rib_rows, rie_rows, ria_rows = [], [], [], [], []
        rng_ri = np.random.default_rng(41)
        fst = Path(a.streams) if a.streams else R / "full_server/peak_detector_streams.npz"
        Tn = None                                          # {rise factor: {dataset row: minute of the next declaration}}
        if fst.exists():
            st = np.load(fst, allow_pickle=False)
            assert int(st["pre"]) == PRE and (st["event_id"] == d["event_id"][st["row"]]).all(), "the streams belong to another dataset"
            Tn = {rise: {int(r): next_declaration(st["xrsb"][j], 2, rise) for j, r in enumerate(st["row"])} for rise in (2.0, 1.5, 1.2)}
        rowL = {int(d["test_k"][i]): int(i) for i in np.where(d["is_test"])[0]}
        for inp in ("fism", "goes_convnn"):
            exp = p8_exp(inp)
            P = get_pool(inp, 0, 1)
            ex_idx, ex_rows = P["ex"]
            E = pd.read_csv(R / exp / "extra_rows.csv")
            assert np.array_equal(E.pool_index.values, ex_idx) and np.array_equal(E.dataset_row.values, ex_rows), f"{exp}: extra_rows.csv does not match the pool"
            c = cat.iloc[d["row"][ex_rows]]
            assert (c.event_id.values == d["event_id"][ex_rows].astype(str)).all() and not d["selected"][ex_rows].any()
            yx = d["Y"][ex_rows].astype(np.float64)
            dpk = pd.to_datetime(d["peak"][ex_rows].astype(str)).values               # aligned peak of the dataset (catalogue peak + shift)
            pk = dpk.astype("datetime64[m]").astype(np.int64)
            tv = P["meta"].peak_time.values[np.concatenate([P["tr"], P["va"]])].astype("datetime64[m]").astype(np.int64)
            gap = np.array([np.abs(tv - p).min() for p in pk])                       # minutes to the nearest train/val flare
            reason = np.where((c.n_pre60.values > 0) & (c.n_post180.values > 0), "both",
                              np.where(c.n_pre60.values > 0, "earlier flare within 60 min", "later flare within 180 min"))
            is_x = np.array([str(s).startswith("X") for s in d["cls"][ex_rows]])
            groups = {"all": np.ones(len(ex_rows), bool), **{r: reason == r for r in ("later flare within 180 min", "earlier flare within 60 min", "both")},
                      "M class": ~is_x, "X class": is_x, "no train/val flare within 240 min": gap >= 240, "train/val flare within 240 min": gap < 240}
            tab = pd.DataFrame(dict(dataset_row=ex_rows, event=d["event_id"][ex_rows].astype(str), peak=dpk, cls=d["cls"][ex_rows].astype(str),
                                    n_pre60=c.n_pre60.values, n_post180=c.n_post180.values, reason=reason, min_to_nearest_trainval=gap))
            for m in MODELS:
                Lx, Lt = preds(exp, m, "oofN"), preds(exp, m, "oofL")
                okt = ~np.isnan(Lt).any(axis=(0, 2, 3))
                Ax, At = ev_mape(Lx, yx), ev_mape(Lt, YT["rebuilt96"])[:, okt]
                Lv, yv = val(exp, m, inp, 0)
                _, lo, hi = conf_norm(Lv, yv, Lx)
                ins_x = (np.log10(yx) >= lo) & (np.log10(yx) <= hi)
                _, lot, hit = conf_norm(Lv, yv, Lt)
                ins_t = ((np.log10(YT["rebuilt96"]) >= lot) & (np.log10(YT["rebuilt96"]) <= hit))[okt]
                ens_x = ev_mape(Lx.mean(0, keepdims=True), yx)[0]
                tab[f"MAPE4_{inp}_{m}"] = Ax.mean(0)
                tab[f"ensMAPE4_{inp}_{m}"] = ens_x
                tab[f"cov_w0_{inp}_{m}"] = ins_x[:, :, 0].mean(1)
                lo_t, hi_t = mboot(At, a.n_boot, 21)
                base = dict(input=inp, model=m, n_test=int(okt.sum()), MAPE4_test=At.mean(), MAPE4_test_lo=lo_t, MAPE4_test_hi=hi_t,
                            cov_w0_test=ins_t[:, :, 0].mean(), cov_4band_test=ins_t.mean(), width_w0_test_dex=(hit - lot)[okt][:, :, 0].mean())
                for g, mask in groups.items():
                    if mask.sum() < 5:
                        continue
                    dlt, lo_d, hi_d, p = uboot(Ax[:, mask], At, a.n_boot, 22)
                    lo_m, hi_m = mboot(Ax[:, mask], a.n_boot, 23)
                    grp_rows.append(dict(**base, group=g, n=int(mask.sum()), MAPE4=Ax[:, mask].mean(), MAPE4_lo=lo_m, MAPE4_hi=hi_m, median_event_MAPE4=float(np.median(Ax[:, mask].mean(0))),
                                         d_vs_test=dlt, lo=lo_d, hi=hi_d, p_boot=p, cov_w0=ins_x[mask][:, :, 0].mean(), cov_4band=ins_x[mask].mean(),
                                         width_w0_dex=(hi - lo)[mask][:, :, 0].mean(),
                                         **{f"MAPE_{b}": float(ev_mape(Lx[:, mask], yx[mask], 0, (j,)).mean()) for j, b in enumerate(B)}))
                    for i, bl in enumerate(BLOCKS):
                        rel = lambda L, y: float((100 * np.abs(10 ** L[:, :, bl] - y[None, :, bl]) / y[None, :, bl]).mean())        # noqa: E731
                        blk_rows.append(dict(input=inp, model=m, group=g, n=int(mask.sum()), block_min=f"{bl.start}-{bl.stop - 1}", MAPE4=rel(Lx[:, mask], yx[mask]),
                                             MAPE4_test=rel(Lt[:, okt], YT["rebuilt96"][okt]), cov_w0=ins_x[mask][:, bl, 0].mean(), cov_w0_test=ins_t[:, bl, 0].mean()))
                # ---- intervals recalibrated for all flares: validation flares + flares outside the rule (out-of-fold), weighted
                ytl, yxl = np.log10(YT["rebuilt96"]), np.log10(yx)
                nv, nx = Lv.shape[1], Lx.shape[1]
                assert np.array_equal(conf_norm_w(Lv, yv, np.ones(nv), Lt)[1][okt], lot[okt]), "equal weights must give the existing interval"
                lab = time_clusters(pk)
                folds = {"alt": lab % 2, "half": (lab >= (lab.max() + 1) / 2).astype(int)}

                def cal_w(n_out, weighted):
                    return np.concatenate([np.full(nv, (1 - P_OUTSIDE) / nv), np.full(n_out, P_OUTSIDE / n_out)]) if weighted else np.ones(nv + n_out)
                schemes = {"isolated only": None, "all flares (weighted)": ("alt", True), "all flares (weighted, folds = time halves)": ("half", True),
                           "all flares (unweighted)": ("alt", False)}
                rgroups = {"outside the rule: all": groups["all"], **{f"outside the rule: {r}": groups[r] for r in ("later flare within 180 min", "earlier flare within 60 min", "both")
                                                                      if groups[r].sum() >= 5}}
                for cname, sch in schemes.items():
                    if sch is None:
                        lo_x, hi_x, lo_tt, hi_tt = lo, hi, lot, hit
                    else:
                        fold, weighted = folds[sch[0]], sch[1]
                        lo_x, hi_x = np.zeros_like(lo), np.zeros_like(hi)
                        for f in (0, 1):
                            cal, ev_ = fold != f, fold == f
                            _, lo_x[ev_], hi_x[ev_] = conf_norm_w(np.concatenate([Lv, Lx[:, cal]], axis=1), np.concatenate([yv, yxl[cal]]),
                                                                  cal_w(int(cal.sum()), weighted), Lx[:, ev_])
                        _, lo_tt, hi_tt = conf_norm_w(np.concatenate([Lv, Lx], axis=1), np.concatenate([yv, yxl]), cal_w(nx, weighted), Lt)
                    in_x, wd_x = (yxl >= lo_x) & (yxl <= hi_x), hi_x - lo_x
                    in_t, wd_t = ((ytl >= lo_tt) & (ytl <= hi_tt))[okt], (hi_tt - lo_tt)[okt]
                    ct, cx = in_t[:, :, 0].mean(1), in_x[:, :, 0].mean(1)                    # per-event coverage of the 0.1-0.8 nm minutes
                    bt = np.array([ct[rng_rc.integers(0, len(ct), len(ct))].mean() for _ in range(2000)])
                    bx = np.array([cx[rng_rc.integers(0, len(cx), len(cx))].mean() for _ in range(2000)])
                    rec = dict(input=inp, model=m, calibration=cname, n_cal_validation=nv, n_cal_outside=0 if sch is None else nx, n_folds=0 if sch is None else 2,
                               n_clusters=int(lab.max() + 1))
                    rc_rows.append(dict(**rec, group="test flares (isolated)", n=int(okt.sum()), cov_w0=ct.mean(), cov_w0_lo=np.percentile(bt, 2.5), cov_w0_hi=np.percentile(bt, 97.5),
                                        cov_4band=in_t.mean(), width_w0_dex=wd_t[:, :, 0].mean(), width_4band_dex=wd_t.mean()))
                    for g, mask in rgroups.items():
                        bg = np.array([cx[mask][rng_rc.integers(0, mask.sum(), mask.sum())].mean() for _ in range(2000)])
                        rc_rows.append(dict(**rec, group=g, n=int(mask.sum()), cov_w0=cx[mask].mean(), cov_w0_lo=np.percentile(bg, 2.5), cov_w0_hi=np.percentile(bg, 97.5),
                                            cov_4band=in_x[mask].mean(), width_w0_dex=wd_x[mask][:, :, 0].mean(), width_4band_dex=wd_x[mask].mean()))
                    mix = lambda t_, x_: (1 - P_OUTSIDE) * t_ + P_OUTSIDE * x_          # noqa: E731
                    bm = mix(bt, bx)
                    rc_rows.append(dict(**rec, group="all flares (population-weighted)", n=int(okt.sum()) + nx, cov_w0=mix(ct.mean(), cx.mean()), cov_w0_lo=np.percentile(bm, 2.5),
                                        cov_w0_hi=np.percentile(bm, 97.5), cov_4band=mix(in_t.mean(), in_x.mean()), width_w0_dex=mix(wd_t[:, :, 0].mean(), wd_x[:, :, 0].mean()),
                                        width_4band_dex=mix(wd_t.mean(), wd_x.mean())))
                    for bl in BLOCKS:
                        rcb_rows.append(dict(input=inp, model=m, calibration=cname, block_min=f"{bl.start}-{bl.stop - 1}", cov_w0_test=in_t[:, bl, 0].mean(),
                                             width_w0_test_dex=wd_t[:, bl, 0].mean(), cov_w0_outside=in_x[:, bl, 0].mean(), width_w0_outside_dex=wd_x[:, bl, 0].mean(),
                                             cov_w0_all=mix(in_t[:, bl, 0].mean(), in_x[:, bl, 0].mean()),
                                             **{f"cov_w0_{k.split(': ')[1].split(' ')[0]}": in_x[v][:, bl, 0].mean() for k, v in rgroups.items() if not k.endswith("all")}))
                # ---- re-issue at the next declaration: every forecast is scored on the minutes for which it is in force
                if Tn is not None:
                    tidx = np.where(okt)[0]
                    trow = np.array([rowL[int(k)] for k in tidx])
                    relx = 100 * np.abs(10 ** Lx - yx[None]) / yx[None]
                    relt = 100 * np.abs(10 ** Lt[:, okt] - YT["rebuilt96"][okt][None]) / YT["rebuilt96"][okt][None]
                    for rise in (2.0, 1.5, 1.2):
                        Tx = np.array([Tn[rise].get(int(r), -1) for r in ex_rows])
                        Tt = np.array([Tn[rise].get(int(r), -1) for r in trow])
                        hx, ht = Tx >= 0, Tt >= 0                    # flares with a GOES stream
                        sets = {"test flares (isolated)": (relt[:, ht], ins_t[ht], Tt[ht]),
                                **{g: (relx[:, mk & hx], ins_x[mk & hx], Tx[mk & hx]) for g, mk in rgroups.items()}}
                        for censor, sname in ((False, "all 180 min"), (True, "minutes in force")):
                            res = {g: in_force(r_, i_, T_, censor, rng_ri) for g, (r_, i_, T_) in sets.items()}
                            pt, px = res["test flares (isolated)"], res["outside the rule: all"]
                            comb = lambda a_, b_: (1 - P_OUTSIDE) * a_ + P_OUTSIDE * b_          # noqa: E731
                            res["all flares (population-weighted)"] = (
                                {k: comb(pt[0][k], px[0][k]) for k in pt[0]}, {k: comb(pt[1][k], px[1][k]) for k in pt[1]})
                            for g, (pv, bv) in res.items():
                                T_ = np.concatenate([Tt[ht], Tx[hx]]) if g.startswith("all flares") else sets[g][2]
                                nxt = T_ < 180
                                rec = dict(input=inp, model=m, rise_factor=rise, scoring=sname, group=g, n=len(T_), share_with_next_declaration=float(nxt.mean()),
                                           median_T_if_next=float(np.median(T_[nxt])) if nxt.any() else np.nan,
                                           share_minutes_in_force=pv["den"] / 180, MAPE4=pv["MAPE4"] / pv["den"], MAPE_w0=pv["MAPE_w0"] / pv["den"],
                                           cov_w0=pv["cov_w0"] / pv["den"], cov_4band=pv["cov_4band"] / pv["den"])
                                for k in ("MAPE4", "cov_w0"):
                                    rec[f"{k}_lo"], rec[f"{k}_hi"] = (float(v) for v in np.percentile(bv[k] / bv["den"], [2.5, 97.5]))
                                if g.startswith("all flares"):       # flare shares; the minute shares follow from the minutes in force
                                    rec["share_with_next_declaration"] = comb(float((Tt[ht] < 180).mean()), float((Tx[hx] < 180).mean()))
                                    rec["median_T_if_next"] = np.nan
                                ri_rows.append(rec)
                        if rise == 2.0:
                            for g in ("test flares (isolated)", "outside the rule: all"):
                                r_, i_, T_ = sets[g]
                                c = (T_ < 180) & (T_ >= 20)          # flares with a next declaration and at least 20 minutes in force
                                for part, lo_, hi_ in (("last 10 minutes before the next declaration", T_[c] - 10, T_[c]), ("earlier minutes in force", np.zeros(c.sum(), int), T_[c] - 10)):
                                    pv, _ = in_force(r_[:, c], i_[c], T_[c], True, rng_ri, 10, lo_, hi_)
                                    rit_rows.append(dict(input=inp, model=m, group=g, n=int(c.sum()), minutes=part, MAPE4=pv["MAPE4"] / pv["den"], MAPE_w0=pv["MAPE_w0"] / pv["den"],
                                                         cov_w0=pv["cov_w0"] / pv["den"], cov_4band=pv["cov_4band"] / pv["den"]))
                                cn = np.where(T_ < 180)[0]           # minute by minute relative to the next declaration (x = t - T)
                                for x in range(-120, 60):
                                    use = cn[(T_[cn] + x >= 0) & (T_[cn] + x < 180)]
                                    if len(use) >= 20:
                                        tt_ = T_[use] + x
                                        ria_rows.append(dict(input=inp, model=m, group=g, minutes_from_next_declaration=x, n=len(use),
                                                             cov_w0=float(i_[use, tt_, 0].mean()), cov_4band=float(i_[use, tt_].mean()),
                                                             MAPE_w0=float(r_[:, use, tt_, 0].mean())))
                                for bl in BLOCKS:
                                    n_ = len(T_)
                                    pv, _ = in_force(r_, i_, T_, True, rng_ri, 10, np.full(n_, bl.start), np.full(n_, bl.stop))
                                    pf, _ = in_force(r_, i_, T_, False, rng_ri, 10, np.full(n_, bl.start), np.full(n_, bl.stop))
                                    rib_rows.append(dict(input=inp, model=m, group=g, block_min=f"{bl.start}-{bl.stop - 1}", share_minutes_in_force=pv["den"] / 30,
                                                         MAPE4_in_force=pv["MAPE4"] / pv["den"], cov_w0_in_force=pv["cov_w0"] / pv["den"],
                                                         MAPE4_all=pf["MAPE4"] / pf["den"], cov_w0_all=pf["cov_w0"] / pf["den"]))
                    if inp == "goes_convnn" and m == MODELS[0]:      # next declaration per flare, once
                        for grp_, rows_, lab_ in (("test flares (isolated)", trow, np.full(len(trow), "isolated")), ("outside the rule", ex_rows, reason)):
                            for r_, l_ in zip(rows_, lab_):
                                if int(r_) in Tn[2.0]:
                                    rie_rows.append(dict(group=grp_, dataset_row=int(r_), event=str(d["event_id"][int(r_)]), cls=str(d["cls"][int(r_)]), reason=l_,
                                                         **{f"T_next_r{rs:g}": Tn[rs][int(r_)] for rs in (2.0, 1.5, 1.2)}))
            ev_tab = tab if ev_tab is None else ev_tab.merge(tab, on=["dataset_row", "event", "peak", "cls", "n_pre60", "n_post180", "reason"], how="outer",
                                                             suffixes=("", f"_{inp}"))
        G = pd.DataFrame(grp_rows)
        G.to_csv(OUT / "paper1_p8_groups.csv", index=False)
        pd.DataFrame(blk_rows).to_csv(OUT / "paper1_p8_blocks.csv", index=False)
        ev_tab.to_csv(OUT / "paper1_p8_events.csv", index=False)
        print(f"paper1_p8_groups.csv: {len(G)} rows; paper1_p8_blocks.csv: {len(blk_rows)} rows; paper1_p8_events.csv: {len(ev_tab)} flares")
        pd.DataFrame(rc_rows).to_csv(OUT / "paper1_p8_recal.csv", index=False)
        pd.DataFrame(rcb_rows).to_csv(OUT / "paper1_p8_recal_blocks.csv", index=False)
        print(f"paper1_p8_recal.csv: {len(rc_rows)} rows; paper1_p8_recal_blocks.csv: {len(rcb_rows)} rows")
        if Tn is not None:
            pd.DataFrame(ri_rows).to_csv(OUT / "paper1_p8_reissue.csv", index=False)
            pd.DataFrame(rit_rows).to_csv(OUT / "paper1_p8_reissue_tail.csv", index=False)
            pd.DataFrame(rib_rows).to_csv(OUT / "paper1_p8_reissue_blocks.csv", index=False)
            pd.DataFrame(rie_rows).to_csv(OUT / "paper1_p8_reissue_events.csv", index=False)
            pd.DataFrame(ria_rows).to_csv(OUT / "paper1_p8_reissue_aligned.csv", index=False)
            print(f"paper1_p8_reissue.csv: {len(ri_rows)} rows; _tail {len(rit_rows)}, _blocks {len(rib_rows)}, _events {len(rie_rows)} flares")
        else:
            print(f"[re-issue] skipped: {fst} not found")
    os.environ.update(FISM_INPUT="default", FISM_ISSUE_DELAY="0", FISM_PREDICT_EXTRA="0")
    pool.load_full.cache_clear()
    print("written to", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
