"""Replicate-aware statistics for the full-protocol results (server bundle v6 and later).

Why: the paired sign-flip test on event MAPE4 (tests_phase3.csv) conditions on the trained models. Retraining the same
configuration moves its 3-seed mean MAPE4 by about +-0.3 pt (1 sigma; measured by comparing the CPU-era run with the
all-GPU v6 run: SD 0.26 pt for the 54 configurations with fixed hyperparameters, 0.30 pt including the two Optuna
configurations whose search was repeated), so a third of the "significant" event-level differences do not survive once
the training variability is included. Here every difference is tested with a hierarchical bootstrap that resamples the
test events (paired) AND the training replicates of both configurations.

Training replicates of a configuration = its seeds in --results, plus its seeds in --prev when those are a different
training (arrays not identical; e.g. the CPU-era run of the same experiment). Random-exclusion draws are pooled.

    python notebooks/replicate_tests.py --results results --prev <older results root> --rebuilt data/processed/events_test96_rebuilt.npz

Outputs (results/full_server/):
  seed_variability.csv        between-seed SD of the 96-event MAPE4 per configuration and run
  retraining_differences.csv  seed-mean MAPE4 of each configuration in --results and in --prev (with --prev)
  ledger_hboot.csv            every comparison of tests_phase3.csv re-tested (difference, 95 % CI, p)
  exclusion_hboot.csv/_runs   +-24 h / same-AR exclusion vs baseline and vs the size-matched random draws
  paper1_inputs_hboot.csv     FISM2 / GOES direct / linear converter / FCN converter, per model and for all five
  paper1_inputs_ens_all.csv   5-seed ensemble metrics per model x input (incl. per-band MAPE)
  paper1_delay_hboot.csv      delayed issuance (k = 2, 5, 10 min) vs the peak-issued model on the same lead times
  paper1_drap_delay_hboot.csv the same comparison for the 10 MHz absorption error over Korea (daytime events)
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.cv96 import load_test96  # noqa: E402

_spec = importlib.util.spec_from_file_location("afs", ROOT / "notebooks/analyze_full_server.py")
afs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(afs)
B = afs.B


def hboot(A: np.ndarray, Bm: np.ndarray, n: int = 10000, seed: int = 0):
    """A, Bm: (n_replicates, n_events) event MAPE4. Returns (difference of means, lo95, hi95, two-sided p) from a
    bootstrap over events (paired between A and Bm) and over the replicates of A and of Bm."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(A).all(0) & np.isfinite(Bm).all(0)
    A, Bm = A[:, ok], Bm[:, ok]
    ne, d0, out = A.shape[1], A.mean() - Bm.mean(), np.empty(n)
    for i in range(n):
        ev = rng.integers(0, ne, ne)
        out[i] = A[rng.integers(0, len(A), len(A))][:, ev].mean() - Bm[rng.integers(0, len(Bm), len(Bm))][:, ev].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    p = 2 * min((out <= 0).mean(), (out >= 0).mean())
    return float(d0), float(lo), float(hi), float(min(1.0, max(p, 1.0 / n)))


def perm(d, n=20000, seed=0):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    s = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n, len(d)))
    return float(((np.abs((s * d).mean(1)) >= abs(d.mean()) - 1e-12).sum() + 1) / (n + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--prev", default=None, help="older results root with independent trainings of the same experiments")
    ap.add_argument("--rebuilt", default=str(ROOT / "data/processed/events_test96_rebuilt.npz"))
    ap.add_argument("--out", default=None, help="default: <results>/full_server")
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    NEW, OLD = a.results, a.prev
    OUT = Path(a.out or f"{NEW}/full_server")
    OUT.mkdir(parents=True, exist_ok=True)
    X96, Y96, ev = load_test96()
    YT = {"zenodo96": Y96}
    if Path(a.rebuilt).exists():
        d = np.load(a.rebuilt, allow_pickle=False)
        o = np.argsort(d["test_k"])
        if len(o) == 96 and (d["test_k"][o] == np.arange(96)).all():
            YT["rebuilt96"] = d["Y"][o].astype(np.float64)

    def preds(root, exp, m, test):
        pre = "oof" if test == "zenodo96" else "oofL"
        return [10 ** np.load(f).astype(np.float64) for f in sorted(glob.glob(f"{root}/{exp}/{pre}_{m}_s*.npy"))]

    def ev_mape4(root, exp, m, test, k0=0):
        yt, out = YT[test], []
        for P in preds(root, exp, m, test):
            ok = ~np.isnan(P).any(axis=(1, 2))
            e = np.full(96, np.nan)
            e[ok] = afs.ev_metrics(yt[ok][:, k0:], P[ok][:, k0:])[0].mean(1)
            out.append(e)
        return np.array(out)

    cache = {}

    def reps(exp, m, test):
        k = (exp, m, test)
        if k not in cache:
            A = ev_mape4(NEW, exp, m, test)
            if OLD:
                P = ev_mape4(OLD, exp, m, test)
                if len(P) and not (A.shape == P.shape and np.allclose(A, P, equal_nan=True)):
                    A = np.concatenate([A, P]) if len(A) else P
            cache[k] = A
        return cache[k]

    def models_of(root, exp):
        return sorted({os.path.basename(f)[4:].rsplit("_s", 1)[0] for f in glob.glob(f"{root}/{exp}/oof_*_s*.npy")})

    test0 = "rebuilt96" if "rebuilt96" in YT else "zenodo96"
    # ---------------------------------------------------------------- 1. seed variability, retraining differences
    rows = []
    for root, tag in ((NEW, "new"), (OLD, "prev")):
        if not root:
            continue
        for dd in sorted(glob.glob(f"{root}/*_full*")):
            exp = os.path.basename(dd)
            for m in models_of(root, exp):
                E = ev_mape4(root, exp, m, test0)
                if len(E) >= 3:
                    s = np.nanmean(E, axis=1)
                    rows.append(dict(run=tag, exp=exp, model=m, n=len(s), mean=s.mean(), sd=s.std(ddof=1), lo=s.min(), hi=s.max()))
    SV = pd.DataFrame(rows)
    SV.to_csv(OUT / "seed_variability.csv", index=False)
    G = SV[SV.run == "new"]
    print(f"between-seed SD of the 96-event MAPE4 ({test0}): median {G.sd.median():.3f}, mean {G.sd.mean():.3f} ({len(G)} configurations)")
    if OLD:
        P = SV[SV.run == "prev"].merge(G, on=["exp", "model"], suffixes=("_prev", "_new"))
        P["d"] = P.mean_new - P.mean_prev
        ch = P[P.d.abs() > 1e-9]
        print(f"retrained configurations: {len(ch)}; seed-mean new - prev: mean {ch.d.mean():+.3f}, SD {ch.d.std(ddof=1):.3f}, max |d| {ch.d.abs().max():.3f} "
              f"(identical: {int((P.d.abs() <= 1e-9).sum())})")
        fx = ch[~ch.exp.str.contains("optuna")]          # Optuna experiments repeat the hyperparameter search as well
        if 0 < len(fx) < len(ch):
            exp_sd = np.sqrt((fx.sd_new ** 2 / fx.n_new + fx.sd_prev ** 2 / fx.n_prev).mean())
            print(f"  fixed hyperparameters only: {len(fx)}; mean {fx.d.mean():+.3f}, SD {fx.d.std(ddof=1):.3f}, max |d| {fx.d.abs().max():.3f} "
                  f"(independent seeds would give SD {exp_sd:.3f}); re-tuned: "
                  + ", ".join(f"{q.model} {q.d:+.2f}" for q in ch[ch.exp.str.contains('optuna')].itertuples()))
        P.rename(columns={"d": "new_minus_prev"}).to_csv(OUT / "retraining_differences.csv", index=False)

    # ---------------------------------------------------------------- 2. ledger: tests_phase3 re-tested
    f = Path(NEW) / "full_server/tests_phase3.csv"
    if f.exists():
        rows = []
        for r in pd.read_csv(f).itertuples():
            if r.test not in YT:
                continue
            Bm = np.concatenate([reps(e, r.A, r.test) for e in str(r.B_exp).split(",")]) if "," in str(r.B_exp) else reps(r.B_exp, r.B, r.test)
            A = reps(r.A_exp, r.A, r.test)
            if not len(A) or not len(Bm):
                continue
            dlt, lo, hi, p = hboot(A, Bm, n=max(2000, a.n_boot // 2), seed=1)
            rows.append(dict(tag=r.tag, A=r.A, B=r.B, test=r.test, nA=len(A), nB=len(Bm), MAPE4_A=np.nanmean(A), MAPE4_B=np.nanmean(Bm),
                             d=dlt, lo=lo, hi=hi, p_hboot=p, d_event_test=r.dMAPE4, p_event_test=r.p))
        L = pd.DataFrame(rows)
        L["sig_hboot"] = (L.lo > 0) | (L.hi < 0)
        L["sig_event"] = L.p_event_test < 0.05
        L.to_csv(OUT / "ledger_hboot.csv", index=False)
        print(f"ledger: {len(L)} comparisons; significant by the event-only test {int(L.sig_event.sum())}, by the hierarchical bootstrap "
              f"{int(L.sig_hboot.sum())}, event-only significant but not robust {int((L.sig_event & ~L.sig_hboot).sum())}")

    # ---------------------------------------------------------------- 3. exclusion experiments
    rows, runs = [], []
    for test in YT:
        for m in ("GRU-peaknorm", "FCN-peaknorm"):
            base = reps("p2_norm_full", m, test)
            if not len(base):
                continue
            conds = {"ex24h": [reps("p2_norm_full_ex24h", m, test)], "rnd163": [reps(f"p2_norm_full_rnd163s{s}", m, test) for s in (0, 1, 2)],
                     "exAR": [reps("p2_norm_full_exAR", m, test)], "rnd516": [reps(f"p2_norm_full_rnd516s{s}", m, test) for s in (10, 11, 12)]}
            conds = {c: [p for p in parts if len(p)] for c, parts in conds.items()}
            for v in base.mean(1):
                runs.append(dict(test=test, model=m, cond="baseline", draw=0, MAPE4=v, d=v - base.mean()))
            for c, parts in conds.items():
                if not parts:
                    continue
                A = np.concatenate(parts)
                dlt, lo, hi, p = hboot(A, base, n=2 * a.n_boot, seed=5)
                rows.append(dict(test=test, model=m, cond=c, vs="baseline", nA=len(A), nB=len(base), MAPE4=A.mean(), ref=base.mean(), d=dlt, lo=lo, hi=hi, p=p))
                for j, part in enumerate(parts):
                    for v in part.mean(1):
                        runs.append(dict(test=test, model=m, cond=c, draw=j, MAPE4=v, d=v - base.mean()))
            for x, y in (("ex24h", "rnd163"), ("exAR", "rnd516")):
                if conds[x] and conds[y]:
                    A, C = np.concatenate(conds[x]), np.concatenate(conds[y])
                    dlt, lo, hi, p = hboot(A, C, n=2 * a.n_boot, seed=6)
                    rows.append(dict(test=test, model=m, cond=x, vs=y, nA=len(A), nB=len(C), MAPE4=A.mean(), ref=C.mean(), d=dlt, lo=lo, hi=hi, p=p))
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "exclusion_hboot.csv", index=False)
        pd.DataFrame(runs).to_csv(OUT / "exclusion_runs.csv", index=False)
        print("exclusion_hboot.csv:", len(rows), "rows")

    # ---------------------------------------------------------------- 4. paper-1 inputs
    EXP = {"FISM2": "p2_zscore_full_gpu5", "GOES": "p2_zscore_full_in-goes_b_p1", "linear": "p2_zscore_full_in-goes_conv_p2",
           "FCNconv": "p2_zscore_full_in-goes_convnn_p2nn"}
    EXP = {k: e for k, e in EXP.items() if glob.glob(f"{NEW}/{e}/oof_*_s*.npy")}
    MODELS = [m for m in ("FCN-z", "GRU-z", "LSTM-z", "RNN-z", "Transformer-z") if all(glob.glob(f"{NEW}/{e}/oof_{m}_s*.npy") for e in EXP.values())]
    pairs = [(x, y) for x, y in (("GOES", "FISM2"), ("linear", "FISM2"), ("FCNconv", "FISM2"), ("linear", "GOES"), ("FCNconv", "GOES"), ("FCNconv", "linear"))
             if x in EXP and y in EXP]
    if pairs and MODELS:
        E = {(k, m, t): ev_mape4(NEW, e, m, t) for k, e in EXP.items() for m in MODELS for t in YT}
        rows, allr, er = [], [], []
        for t in YT:
            for x, y in pairs:
                for m in MODELS:
                    A, C = E[(x, m, t)], E[(y, m, t)]
                    ok = np.isfinite(A).all(0) & np.isfinite(C).all(0)
                    dlt, lo, hi, p = hboot(A, C, n=a.n_boot, seed=2)
                    rows.append(dict(test=t, model=m, A=x, B=y, MAPE4_A=A[:, ok].mean(), MAPE4_B=C[:, ok].mean(), d=dlt, lo=lo, hi=hi, p_hboot=p,
                                     p_event=perm((A.mean(0) - C.mean(0))[ok]), sd_seed_A=A[:, ok].mean(1).std(ddof=1), sd_seed_B=C[:, ok].mean(1).std(ddof=1)))
                # all architectures: mean of the per-model differences; seeds resampled within each model, events paired
                rng = np.random.default_rng(3)
                ok = np.all([np.isfinite(E[(k, m, t)]).all(0) for k in (x, y) for m in MODELS], axis=0)
                As, Cs = [E[(x, m, t)][:, ok] for m in MODELS], [E[(y, m, t)][:, ok] for m in MODELS]
                ne, out = int(ok.sum()), np.empty(a.n_boot)
                for i in range(a.n_boot):
                    evi = rng.integers(0, ne, ne)
                    out[i] = np.mean([u[rng.integers(0, len(u), len(u))][:, evi].mean() - v[rng.integers(0, len(v), len(v))][:, evi].mean()
                                      for u, v in zip(As, Cs)])
                lo, hi = np.percentile(out, [2.5, 97.5])
                allr.append(dict(test=t, model="all", A=x, B=y, MAPE4_A=np.mean([u.mean() for u in As]), MAPE4_B=np.mean([v.mean() for v in Cs]),
                                 d=np.mean([u.mean() - v.mean() for u, v in zip(As, Cs)]), lo=lo, hi=hi,
                                 p_hboot=min(1.0, max(2 * min((out <= 0).mean(), (out >= 0).mean()), 1.0 / a.n_boot))))
            for m in MODELS:
                for k, e in EXP.items():
                    Ps = preds(NEW, e, m, t)
                    ok = ~np.isnan(np.stack(Ps)).any(axis=(0, 2, 3))
                    q = afs.pooled(YT[t][ok], (10 ** np.mean([np.log10(P) for P in Ps], axis=0))[ok])
                    er.append(dict(test=t, model=m, input=k, n_seeds=len(Ps), n_ev=int(ok.sum()), ensMAPE4=q["MAPE4"], ensAPE4=q["APE4"], ensCC4=q["CC4"],
                                   **{f"MAPE_{b}": q[f"MAPE_{b}"] for b in B}, **{f"APE_{b}": q[f"APE_{b}"] for b in B}))
        pd.concat([pd.DataFrame(rows), pd.DataFrame(allr)], ignore_index=True).to_csv(OUT / "paper1_inputs_hboot.csv", index=False)
        pd.DataFrame(er).to_csv(OUT / "paper1_inputs_ens_all.csv", index=False)
        print("paper1_inputs_hboot.csv:", len(rows) + len(allr), "rows")

    # ---------------------------------------------------------------- 5. issue delay
    BASE = {"FCN-z": ("p2_zscore_full_gpu5", "FCN-z"), "GRU-z": ("p2_zscore_full_gpu5", "GRU-z"), "GRU-pn": ("p3_ensemble_full_gpu", "GRU-peaknorm")}
    rows = []
    for test in YT:
        for fam in BASE:
            for k in (2, 5, 10):
                A = ev_mape4(NEW, "paper1_delay_full", f"{fam}-d{k}", test, k0=k)
                C = ev_mape4(NEW, *BASE[fam], test, k0=k)
                if not len(A) or not len(C):
                    continue
                dlt, lo, hi, p = hboot(A, C, n=a.n_boot, seed=4)
                yt = YT[test]
                ape_k = np.mean([(100 * np.abs(P[:, k] - yt[:, k]) / yt[:, k]).mean() for P in preds(NEW, "paper1_delay_full", f"{fam}-d{k}", test)])
                ape_0 = np.mean([(100 * np.abs(P[:, 0] - yt[:, 0]) / yt[:, 0]).mean() for P in preds(NEW, *BASE[fam], test)])
                rows.append(dict(test=test, family=fam, k=k, nA=len(A), nB=len(C), MAPE4_delayed=A.mean(), MAPE4_peak_issued_same_leads=C.mean(), d=dlt, lo=lo, hi=hi,
                                 p_hboot=p, p_event=perm(A.mean(0) - C.mean(0)), APE4_at_issue=ape_k, APE4_peak_issued_at_peak=ape_0))
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "paper1_delay_hboot.csv", index=False)
        print("paper1_delay_hboot.csv:", len(rows), "rows")

    # ---------------------------------------------------------------- 6. D-RAP error of the delayed models on lead >= k
    # paper1_drap.csv scores the delayed models on minutes 0..179, but their minutes 0..k are (partly) inputs. Here the
    # 10 MHz absorption error is taken on the minutes that are really forecast (lead >= k, daytime) and compared with
    # the peak-issued ensemble on the same minutes; CI from a bootstrap over events and over the seeds of both ensembles.
    from src.data.cv96 import load_meta
    from src.eval.drap import absorption_db, solar_zenith_deg, SITE_KOREA
    meta = load_meta()
    tt = np.array([np.datetime64(t, "s") for t in meta.peak_time.values[:96]])[:, None] + np.arange(180).astype("timedelta64[m]")[None, :]
    ZEN = solar_zenith_deg(tt, *SITE_KOREA)
    day = ZEN[:, 0] < 90
    rows, rng = [], np.random.default_rng(6)
    for test in YT:
        AT = absorption_db(YT[test][:, :, 0], ZEN)
        for fam in BASE:
            Pb = preds(NEW, *BASE[fam], test)
            for k in (2, 5, 10):
                Pd = preds(NEW, "paper1_delay_full", f"{fam}-d{k}", test)
                if not Pd or not Pb:
                    continue
                Ld, Lb = np.log10(np.stack(Pd))[..., 0], np.log10(np.stack(Pb))[..., 0]       # (S, 96, 180)
                ok = day & np.isfinite(Ld).all((0, 2)) & np.isfinite(Lb).all((0, 2))
                idx = np.where(ok)[0]
                msk = (ZEN < 90) & (np.arange(180)[None, :] >= k)

                def mae(L, ee, ss):
                    err = np.abs(absorption_db(10 ** L[ss][:, ee].mean(0), ZEN[ee]) - AT[ee])
                    return np.nanmean(np.where(msk[ee], err, np.nan), axis=1).mean()
                full = lambda L: mae(L, idx, np.arange(len(L)))                                 # noqa: E731
                bt = np.empty(a.n_boot)
                for i in range(a.n_boot):
                    ee = rng.choice(idx, len(idx))
                    bt[i] = mae(Ld, ee, rng.integers(0, len(Ld), len(Ld))) - mae(Lb, ee, rng.integers(0, len(Lb), len(Lb)))
                lo, hi = np.percentile(bt, [2.5, 97.5])
                rows.append(dict(test=test, family=fam, k=k, n_day=len(idx), MAE_dB_delayed=full(Ld), MAE_dB_peak_issued_same_leads=full(Lb),
                                 d=full(Ld) - full(Lb), lo=lo, hi=hi, p_hboot=min(1.0, max(2 * min((bt <= 0).mean(), (bt >= 0).mean()), 1.0 / a.n_boot))))
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "paper1_drap_delay_hboot.csv", index=False)
        print("paper1_drap_delay_hboot.csv:", len(rows), "rows")
    print("written to", OUT)


if __name__ == "__main__":
    main()
