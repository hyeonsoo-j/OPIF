"""Analysis of the paper-setting ("full" protocol) server results (sections ⑪/⑫ of REPORT.md).

Inputs
  --results   folder holding results/<exp>_full*/ (an extracted results_full_*.tar.gz, or this repo's results/)
  --rebuilt   npz of the 96 test events rebuilt from the downloads (scripts/download/build_events.py --stage test
              --out ...), used as the second truth ("rebuilt96"); omit to score only on the Zenodo arrays
  --out       output folder (default results/full_server/)
Outputs (CSV): summary_all, tests_phase3 (paired sign-flip permutation tests on per-event MAPE4 + Holm),
  tests_final_candidates, drap_full (10 MHz Korea, daytime events), conformal_full (cross-conformal on the test
  events using the seed ensembles), leadtime_rmse_log10, physics_early_full, scale_shift_full, perm_importance_full.
Numbers in REPORT ⑪/⑫ come from these files. Needs only numpy/pandas (runs on the server VM too).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.cv96 import load_meta, load_test96, outer_folds  # noqa: E402
from src.eval.drap import event_absorption  # noqa: E402

B = ["w0", "w1", "w2", "w3"]
rng = np.random.default_rng(0)


def pearson(x, y):
    x, y = x.ravel(), y.ravel()
    sx, sy = x.std(), y.std()
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy)) if sx > 0 and sy > 0 else np.nan


def ev_metrics(yt, yp):
    rel = np.abs(yp - yt) / yt
    mape = 100 * rel.mean(axis=1)
    ape = 100 * np.abs(yp[:, 0] - yt[:, 0]) / yt[:, 0]
    cc = np.array([[pearson(yt[i, :, b], yp[i, :, b]) for b in range(4)] for i in range(len(yt))])
    return mape, ape, cc


def pooled(yt, yp):
    mape, ape, cc = ev_metrics(yt, yp)
    return dict(CC4=np.mean([pearson(yt[:, :, b], yp[:, :, b]) for b in range(4)]), MAPE4=mape.mean(), APE4=ape.mean(),
                evCC4=np.nanmean(cc), **{f"CC_{b}": pearson(yt[:, :, j], yp[:, :, j]) for j, b in enumerate(B)},
                **{f"MAPE_{b}": mape[:, j].mean() for j, b in enumerate(B)}, **{f"APE_{b}": ape[:, j].mean() for j, b in enumerate(B)})


def perm(a, b, n=20000):
    d = a - b
    d = d[np.isfinite(d)]
    obs = d.mean()
    null = (rng.choice([-1, 1], size=(n, len(d))) * d).mean(axis=1)
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def cq(r, level=0.9):
    n = len(r)
    k = min(n, int(math.ceil((n + 1) * level)))
    return np.sort(r)[k - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--rebuilt", default=None)
    ap.add_argument("--out", default=str(ROOT / "results/full_server"))
    a = ap.parse_args()
    R, OUT = Path(a.results), Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    X96, Y96, ev = load_test96()
    Ytest = {"zenodo96": Y96}
    if a.rebuilt:
        d = np.load(a.rebuilt, allow_pickle=False)
        o = np.argsort(d["test_k"])
        if len(o) == 96 and (d["test_k"][o] == np.arange(96)).all():
            Ytest["rebuilt96"] = d["Y"][o].astype(np.float64)
    META = load_meta()
    fold = outer_folds(10, "interleaved")

    def load(exp, name, test):
        pre = "oof" if test == "zenodo96" else "oofL"
        return [10 ** np.load(f).astype(np.float64) for f in sorted(glob.glob(f"{R}/{exp}/{pre}_{name}_s*.npy"))]

    # 1. summary of every exp/model/test (seed mean + log-mean ensemble)
    rows, PE, ENS = [], {}, {}
    for dd in sorted(glob.glob(f"{R}/*_full*")):
        exp = os.path.basename(dd)
        names = sorted({os.path.basename(f)[4:].rsplit("_s", 1)[0] for f in glob.glob(f"{dd}/oof_*_s*.npy")})
        for m in names:
            for test, yt in Ytest.items():
                Ps = load(exp, m, test)
                if not Ps:
                    continue
                ok = ~np.isnan(np.stack(Ps)).any(axis=(0, 2, 3))
                sm = pd.DataFrame([pooled(yt[ok], P[ok]) for P in Ps]).mean()
                ens = 10 ** np.mean([np.log10(P) for P in Ps], axis=0)
                en = pooled(yt[ok], ens[ok])
                rows.append(dict(exp=exp, model=m, test=test, n_seeds=len(Ps), n_ev=int(ok.sum()), **sm.to_dict(),
                                 ensCC4=en["CC4"], ensMAPE4=en["MAPE4"], ensAPE4=en["APE4"]))
                PE[(exp, m, test)] = np.mean([ev_metrics(yt, P)[0].mean(1) for P in Ps], axis=0)
                ENS[(exp, m, test)] = ens
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "summary_all.csv", index=False)
    print(f"summary_all.csv: {len(S)} rows")

    # 2. paired tests (Phase-3 ledger at full scale)
    bg, bf, bfp = ("p2_norm_full", "GRU-peaknorm"), ("p2_zscore_full", "FCN-z"), ("p2_norm_full", "FCN-peaknorm")
    pairs = []
    for s in (0, 1, 2):
        pairs += [(f"p2_norm_full_rnd163s{s}", "GRU-peaknorm", *bg, f"C1 random-{s}"), (f"p2_norm_full_rnd163s{s}", "FCN-peaknorm", *bfp, f"C1 random-{s}")]
    for w in (15, 30, 45):
        pairs += [("p2_window_full", f"GRU-pn-w{w}", *bg, "C2 window"), ("p2_window_full", f"FCN-pn-w{w}", *bfp, "C2 window")]
    for arch in ("PatchTST", "TSMixer", "N-BEATS", "ModernTCN"):
        pairs += [("p3_arch_full", arch, *bg, "C3 arch vs GRU-pn"), ("p3_arch_full", arch, *bf, "C3 arch vs FCN-z")]
    pairs += [("p3_resid_full", "GRU-pn-resid", *bg, "C4 resid"), ("p3_resid_full", "FCN-z-resid", *bf, "C4 resid"),
              ("p3_resid_full", "Prior-only", "repro_cv96_full", "Power-law", "C4 prior vs powerlaw"),
              ("p3_extreme_full", "GRU-pn-wtop3", *bg, "C5 extreme"), ("p3_extreme_full", "GRU-pn-noise", *bg, "C5 noise"),
              ("p3_extreme_full", "FCN-z-noise", *bf, "C5 noise"), ("p3_type_full", "GRU-pn-experts", *bg, "C6 experts"),
              ("p3_type_full", "GRU-pn-typefeat", *bg, "C6 typefeat"), ("p3_joint_full", "GRU-pn-perband", *bg, "C7 perband"),
              ("p3_joint_full", "FCN-z-joint", *bf, "C7 joint"), ("p3_features_full", "GRU-pn-feat", *bg, "C8 features"),
              ("p3_features_full", "FCN-z-feat", *bf, "C8 features")]
    for lead in (10, 20, 30):
        pairs += [("p3_early_full", f"GRU-pn-lead{lead}", *bg, "C9 early"), ("p3_early_full", f"FCN-z-lead{lead}", *bf, "C9 early")]
    pairs += [("p3_final_full", "GRU-final", "p3_ensemble_full", "GRU-peaknorm", "C10 combined (5 seeds)"),
              ("p3_optuna_full", "FCN-z-optuna", *bf, "C12 optuna"), ("p3_optuna_full", "GRU-z-optuna", "p2_zscore_full", "GRU-z", "C12 optuna"),
              ("p3_optuna_full", "FCN-z-optuna", "repro_cv96_full", "RandomForest", "C12 optuna vs RF"),
              ("p2_norm_full_exAR", "GRU-peaknorm", *bg, "G2 AR exclusion"), ("p2_norm_full_exAR", "FCN-peaknorm", *bfp, "G2 AR exclusion"),
              ("p2_zscore_full", "GRU-z", "repro_cv96_full", "GRU", "E2 z-score"), ("p2_zscore_full", "FCN-z", "repro_cv96_full", "FCN", "E2 z-score"),
              ("p2_norm_full", "GRU-peaknorm", "repro_cv96_full", "GRU", "E2 peaknorm"), ("p2_norm_full", "FCN-peaknorm", "repro_cv96_full", "FCN", "E2 peaknorm"),
              ("p2_zscore_full", "FCN-z", "repro_cv96_full", "RandomForest", "E2 FCN-z vs RF"), ("repro_cv96_full", "GRU", "repro_cv96_full", "RandomForest", "E1 GRU vs RF"),
              ("repro_cv96_full", "RNN", "repro_cv96_full", "GRU", "E1 RNN vs GRU"),
              ("p2_norm_full_in-goes_b", "GRU-peaknorm", *bg, "E4 GOES input"), ("p2_norm_full_in-goes_b", "FCN-peaknorm", *bfp, "E4 GOES input"),
              ("p2_norm_full_ex24h", "GRU-peaknorm", *bg, "E5 ex24h"), ("p2_norm_full_ex24h", "FCN-peaknorm", *bfp, "E5 ex24h")]
    out = []
    for ea, ma, eb, mb, tag in pairs:
        for test in Ytest:
            x, y = PE.get((ea, ma, test)), PE.get((eb, mb, test))
            if x is None or y is None:
                continue
            dm, p = perm(x, y)
            sa = S[(S.exp == ea) & (S.model == ma) & (S.test == test)].iloc[0]
            sb = S[(S.exp == eb) & (S.model == mb) & (S.test == test)].iloc[0]
            out.append(dict(tag=tag, A=ma, A_exp=ea, B=mb, B_exp=eb, test=test, MAPE4_A=sa.MAPE4, MAPE4_B=sb.MAPE4, dMAPE4=dm, p=p,
                            dCC4=sa.CC4 - sb.CC4, dAPE4=sa.APE4 - sb.APE4))
    for m, base in (("GRU-peaknorm", bg), ("FCN-peaknorm", bfp)):      # exclusion vs size-matched random controls
        for excl, rnd_glob in (("p2_norm_full_ex24h", "p2_norm_full_rnd163s[0-9]"), ("p2_norm_full_exAR", "p2_norm_full_rnd*s1[0-9]")):
            rnd_exps = sorted(os.path.basename(x) for x in glob.glob(f"{R}/{rnd_glob}"))
            for test in Ytest:
                if (excl, m, test) not in PE or not all((e, m, test) in PE for e in rnd_exps) or not rnd_exps:
                    continue
                r = np.mean([PE[(e, m, test)] for e in rnd_exps], axis=0)
                dm, p = perm(PE[(excl, m, test)], r)
                out.append(dict(tag=f"{excl.split('_')[-1]} vs mean(random x{len(rnd_exps)})", A=m, A_exp=excl, B=f"{m} rnd-mean", B_exp=",".join(rnd_exps),
                                test=test, MAPE4_A=PE[(excl, m, test)].mean(), MAPE4_B=r.mean(), dMAPE4=dm, p=p, dCC4=np.nan, dAPE4=np.nan))
    T = pd.DataFrame(out)
    if len(T):
        T["p_holm"] = holm(T.p.values)
        T.to_csv(OUT / "tests_phase3.csv", index=False)
    print(f"tests_phase3.csv: {len(T)} comparisons")

    # 3. final candidates (ensembles) on MAPE4 / APE4 / event CC
    cands = {"FCN-z-ens5": ("p3_ensemble_full", "FCN-z"), "GRU-pn-ens5": ("p3_ensemble_full", "GRU-peaknorm"),
             "GRU-final-ens5": ("p3_final_full", "GRU-final"), "GRU-z-optuna-ens3": ("p3_optuna_full", "GRU-z-optuna"),
             "GRU-z-ens3": ("p2_zscore_full", "GRU-z"), "RF-ens3": ("repro_cv96_full", "RandomForest"), "RNN-ens3": ("repro_cv96_full", "RNN")}
    rows = []
    for test, yt in Ytest.items():
        M = {k: ev_metrics(yt, ENS[(e, m, test)]) for k, (e, m) in cands.items() if (e, m, test) in ENS}
        if "FCN-z-ens5" in M and "GRU-z-optuna-ens3" in M:       # the average of the two complementary ensembles
            avg = 10 ** ((np.log10(ENS[cands["FCN-z-ens5"] + (test,)]) + np.log10(ENS[cands["GRU-z-optuna-ens3"] + (test,)])) / 2)
            M["FCNz+GRUz-optuna avg"] = ev_metrics(yt, avg)
        for aa, bb in [("GRU-z-optuna-ens3", "FCN-z-ens5"), ("GRU-z-optuna-ens3", "GRU-z-ens3"), ("GRU-z-optuna-ens3", "RF-ens3"),
                       ("FCN-z-ens5", "RF-ens3"), ("GRU-final-ens5", "GRU-pn-ens5"), ("FCN-z-ens5", "RNN-ens3"),
                       ("FCNz+GRUz-optuna avg", "FCN-z-ens5"), ("FCNz+GRUz-optuna avg", "GRU-z-optuna-ens3")]:
            if aa not in M or bb not in M:
                continue
            for metric, idx in (("MAPE4", 0), ("APE4", 1), ("evCC4", 2)):
                x, y = M[aa][idx], M[bb][idx]
                x = x.mean(1) if x.ndim == 2 else x
                y = y.mean(1) if y.ndim == 2 else y
                dm, p = perm(x, y)
                rows.append(dict(test=test, A=aa, B=bb, metric=metric, A_val=np.nanmean(x), B_val=np.nanmean(y), diff=dm, p=p))
    pd.DataFrame(rows).to_csv(OUT / "tests_final_candidates.csv", index=False)

    # 4. D-RAP, lead time, conformal, physics, scale, permutation importance
    key = [("repro_cv96_full", "Power-law"), ("repro_cv96_full", "Climatology"), ("repro_cv96_full", "RandomForest"),
           ("repro_cv96_full", "GRU"), ("repro_cv96_full", "FCN"), ("p2_norm_full", "GRU-peaknorm"), ("p2_zscore_full", "GRU-z"),
           ("p3_ensemble_full", "FCN-z"), ("p3_ensemble_full", "GRU-peaknorm"), ("p3_final_full", "GRU-final"), ("p3_optuna_full", "GRU-z-optuna")]
    dr, curves, ph = [], {}, []
    for test, yt in Ytest.items():
        for exp, m in key:
            P = ENS.get((exp, m, test))
            if P is None:
                continue
            errs = []
            for e in range(96):
                a_t, zen = event_absorption(yt[e, :, 0], META.peak_time.iloc[e])
                if zen[0] >= 90:
                    continue
                a_p, _ = event_absorption(P[e, :, 0], META.peak_time.iloc[e])
                dd = zen < 90
                err = a_p[dd] - a_t[dd]
                errs.append(dict(MAE=np.abs(err).mean(), RMSE=np.sqrt((err ** 2).mean()), bias=err.mean(), peak_abs=abs(a_p[0] - a_t[0]), tp=a_t[0]))
            E = pd.DataFrame(errs)
            dr.append(dict(test=test, exp=exp, model=m, n_day=len(E), MAE_dB=E.MAE.mean(), RMSE_dB=E.RMSE.mean(), bias_dB=E.bias.mean(),
                           abs_peak_err_dB=E.peak_abs.mean(), mean_true_peak_dB=E.tp.mean()))
            r = np.sqrt(np.mean((np.log10(P) - np.log10(yt)) ** 2, axis=0))
            for b in range(4):
                curves[f"{test}|{m}|{B[b]}"] = r[:, b]
    pd.DataFrame(dr).to_csv(OUT / "drap_full.csv", index=False)
    LT = pd.DataFrame(curves)
    LT.index.name = "minute"
    LT.to_csv(OUT / "leadtime_rmse_log10.csv")
    for exp, m in key + [("p3_early_full", f"{n}-lead{lead}") for n in ("GRU-pn", "FCN-z") for lead in (10, 20, 30)]:
        P = ENS.get((exp, m, "zenodo96"))
        if P is None:
            continue
        L = np.log10(P)
        Dd = L[:, 0, :] - L[:, -1, :]
        ph.append(dict(model=m, frac_rerise10_w0=float(((L[:, 10, 0] - L[:, 0, 0]) > 0).mean()),
                       frac_above_peak_w0=float((P[:, 1:, 0].max(1) > 1.01 * P[:, 0, 0]).mean()),
                       frac_order_full=float(((Dd[:, 0] > Dd[:, 2]) & (Dd[:, 2] > Dd[:, 1]) & (Dd[:, 1] > Dd[:, 3])).mean()),
                       rough_w0=float(np.mean(np.abs(np.diff(L[:, :, 0], 2, axis=1)))),
                       w0_APE=float((100 * np.abs(P[:, 0, 0] - Y96[:, 0, 0]) / Y96[:, 0, 0]).mean()),
                       peak_err_dex=float(np.mean(np.abs(L[:, 0, 0] - np.log10(Y96[:, 0, 0]))))))
    pd.DataFrame(ph).to_csv(OUT / "physics_early_full.csv", index=False)
    cf = []
    for exp, m in (("p3_ensemble_full", "FCN-z"), ("p3_ensemble_full", "GRU-peaknorm"), ("p3_final_full", "GRU-final")):
        for test, ytp in Ytest.items():
            Ps = load(exp, m, test)
            if len(Ps) < 3:
                continue
            L = np.log10(np.stack(Ps))
            mu, sd, yt = L.mean(0), L.std(0), np.log10(ytp)
            res = yt - mu
            blocks = [slice(i * 30, (i + 1) * 30) for i in range(6)]
            cov = {k: np.zeros((96, 4)) for k in ("raw", "abs", "norm")}
            wid = {k: np.zeros((96, 4)) for k in ("raw", "abs", "norm")}
            lo_n, hi_n = np.zeros_like(mu), np.zeros_like(mu)
            for f in range(10):
                te, ca = fold == f, fold != f
                for b in range(4):
                    for bl in blocks:
                        r = np.abs(res[ca, bl, b]).ravel()
                        s0 = np.median(sd[ca, bl, b])
                        qa, qn = cq(r), cq((np.abs(res[ca, bl, b]) / (sd[ca, bl, b] + s0)).ravel())
                        for k, half in (("raw", 1.645 * sd[te, bl, b]), ("abs", np.full(sd[te, bl, b].shape, qa)), ("norm", qn * (sd[te, bl, b] + s0))):
                            cov[k][te, b] += (np.abs(res[te, bl, b]) <= half).mean(1) * (bl.stop - bl.start) / 180
                            wid[k][te, b] += (2 * half).mean(1) * (bl.stop - bl.start) / 180
                        lo_n[te, bl, b] = mu[te, bl, b] - qn * (sd[te, bl, b] + s0)
                        hi_n[te, bl, b] = mu[te, bl, b] + qn * (sd[te, bl, b] + s0)
            rec = dict(exp=exp, model=m, test=test, n_seeds=len(Ps))
            for k in ("raw", "abs", "norm"):
                for b in range(4):
                    rec[f"cov_{k}_{B[b]}"] = cov[k][:, b].mean()
                    rec[f"width_{k}_{B[b]}_dex"] = wid[k][:, b].mean()
            isx = META.is_x.values.astype(bool)
            rec["cov_norm_w0_Xclass"], rec["cov_raw_w0_Xclass"] = cov["norm"][isx, 0].mean(), cov["raw"][isx, 0].mean()
            c, w = [], []
            for e in range(96):
                a_t, zen = event_absorption(ytp[e, :, 0], META.peak_time.iloc[e])
                if zen[0] >= 90:
                    continue
                a_lo, _ = event_absorption(10 ** lo_n[e, :, 0], META.peak_time.iloc[e])
                a_hi, _ = event_absorption(10 ** hi_n[e, :, 0], META.peak_time.iloc[e])
                dd = zen < 90
                c.append(((a_t[dd] >= a_lo[dd]) & (a_t[dd] <= a_hi[dd])).mean())
                w.append((a_hi[dd] - a_lo[dd]).mean())
            rec["drap_cov_norm"], rec["drap_width_dB"], rec["drap_n_day"] = np.mean(c), np.mean(w), len(c)
            cf.append(rec)
    pd.DataFrame(cf).to_csv(OUT / "conformal_full.csv", index=False)
    sc = []
    for f in sorted(glob.glob(f"{R}/p2_scale_full/info_*_s0.json")):
        d0 = json.load(open(f))[0]
        for fct, v in d0.get("scale_shift", {}).items():
            for bands, arr in v.items():
                sc.append(dict(model=os.path.basename(f)[5:-8], factor=float(fct), bands=bands, mean_shift_dex=float(np.mean(arr)), expected=float(np.log10(float(fct)))))
    pd.DataFrame(sc).to_csv(OUT / "scale_shift_full.csv", index=False)
    pi = []
    for f in sorted(glob.glob(f"{R}/p3_features_full/info_*_s*.json")):
        for d0 in json.load(open(f)):
            for r in d0.get("perm_importance", []):
                pi.append(dict(model=os.path.basename(f)[5:].rsplit("_s", 1)[0], **r))
    if pi:
        pd.DataFrame(pi).groupby(["model", "feature"]).delta_mse_log.mean().reset_index().to_csv(OUT / "perm_importance_full.csv", index=False)
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
