"""Split conformal 90 % intervals for the paper-setting ("full" protocol) models, calibrated on the VALIDATION
events (predictions saved by run.py --protocol full as results/<exp>/val_<name>_s<seed>.npz) and evaluated on
the test events (Zenodo arrays and the rebuilt test set). Runs where the event pool is available (the server).

  python notebooks/split_conformal_full.py --events data/processed/events_latis.npz \
      --models p3_ensemble_full_gpu:FCN-z p3_ensemble_full_gpu:GRU-peaknorm p2_zscore_full_gpu5:GRU-z

Per model: mu/sd = mean/std over seeds of the log10 predictions. Intervals:
  raw       mu +- 1.645 sd                                   (ensemble spread only)
  conf-abs  mu +- q_abs,       q_abs  = 90 % quantile of |r| on validation          (per band x 30-min block)
  conf-norm mu +- q_n (sd+s0), q_n    = 90 % quantile of |r|/(sd+s0), s0 = median sd (adaptive)
  avg       the FCN-z / GRU-z ensemble average, when both are given (sd = std over all member seeds)
Also the D-RAP 10 MHz (Korea) absorption interval from the w0 conf-norm interval (daytime events).
Output: results/full_server/conformal_split.csv (+ _events.csv with per-event coverage).
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
B = ["w0", "w1", "w2", "w3"]


def cq(r, level=0.9):
    n = len(r)
    k = min(n, int(math.ceil((n + 1) * level)))
    return np.sort(r)[k - 1]


def _exp_input(exp_name: str) -> str:
    """Detect which FISM_INPUT variant was used when results/<exp_name>/val_*.npz was created."""
    for k in ("goes_convnn", "goes_conv", "goes_b", "fism"):
        if f"_in-{k}" in exp_name:
            return k
    return "default"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--events", default=str(ROOT / "data/processed/events_latis.npz"))
    ap.add_argument("--models", nargs="+", default=["p3_ensemble_full_gpu:FCN-z", "p3_ensemble_full_gpu:GRU-peaknorm",
                                                    "p2_zscore_full_gpu5:GRU-z", "p2_zscore_full_gpu5:FCN-z"])
    ap.add_argument("--avg", default="p2_zscore_full_gpu5:FCN-z+p2_zscore_full_gpu5:GRU-z",
                    help="two exp:model entries joined by '+' to evaluate their ensemble average ('' to skip)")
    ap.add_argument("--level", type=float, default=0.9)
    ap.add_argument("--cond", choices=["none", "peak", "class"], default="none",
                    help="conditional quantiles: 'peak' = 2 groups by the input peak level (split at the validation "
                         "median), 'class' = M vs X (X-class validation events are few; falls back to 'none' if < 8)")
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    ap.add_argument("--out", default=str(ROOT / "results/full_server"))
    ap.add_argument("--save-intervals", action="store_true",
                    help="also write conformal_intervals<tag>.npz: ensemble mean and conf-norm lower/upper bound (log10) "
                         "of every model on both test sets, shape (96, 180, 4) - for figures of individual events")
    a = ap.parse_args()
    os.environ["FISM_PROTOCOL"] = "full"
    os.environ["FISM_EVENTS"] = str(Path(a.events).resolve())
    os.environ.setdefault("FISM_DEVICE_QUIET", "1")
    from src.data import pool
    from src.eval.drap import event_absorption

    pool_cache = {}

    def get_pool(inp: str):
        if inp not in pool_cache:
            os.environ["FISM_INPUT"] = inp
            pool.load_full.cache_clear()
            pool_cache[inp] = pool.load_full()
            print(f"[conformal] pool for input {inp!r}: {pool.describe()}", flush=True)
        return pool_cache[inp]

    X, Y, names, meta, tr, va, te, te_latis = get_pool("default")
    Ytest = {"zenodo96": Y[:96], "rebuilt96": Y[96:192]}
    R, OUT = Path(a.results), Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    isx = meta.is_x.values[:96].astype(bool)
    peak = meta.peak_time.iloc[:96]

    def load_val(exp, m):
        fs = sorted(glob.glob(f"{R}/{exp}/val_{m}_s*.npz"))
        if not fs:
            return None, None
        ds = [np.load(f, allow_pickle=False) for f in fs]
        idx = ds[0]["idx"]
        assert all((d["idx"] == idx).all() for d in ds), "validation sets differ between seeds"
        return idx, np.stack([d["pred_log"].astype(np.float64) for d in ds])

    def load_test(exp, m, test):
        pre = "oof" if test == "zenodo96" else "oofL"
        fs = sorted(glob.glob(f"{R}/{exp}/{pre}_{m}_s*.npy"))
        return np.stack([np.load(f).astype(np.float64) for f in fs]) if fs else None

    entries = [tuple(s.split(":")) for s in a.models]
    if a.avg:
        entries.append(tuple(tuple(s.split(":")) for s in a.avg.split("+")))
    rows, ev_rows, bands = [], [], {}
    blocks = [slice(i * 30, (i + 1) * 30) for i in range(6)]
    for ent in entries:
        if isinstance(ent[0], tuple):                       # average of two ensembles
            label = "avg(" + ",".join(m for _, m in ent) + ")"
            parts = [load_val(e, m) for e, m in ent]
            if any(p[0] is None for p in parts):
                print(f"[skip] {label}: no validation predictions")
                continue
            idx = parts[0][0]
            Lv = np.concatenate([p[1] for p in parts])      # members of both ensembles
            tests = {t: np.concatenate([load_test(e, m, t) for e, m in ent]) for t in Ytest}
            inps = {_exp_input(e) for e, _ in ent}
            if len(inps) > 1 or not all((p[0] == idx).all() for p in parts):
                print(f"[skip] {label}: the averaged ensembles use different inputs or validation sets")
                continue
            Xp, Yp, _, metap, _, vap, *_ = get_pool(_exp_input(ent[0][0]))
        else:
            e, m = ent
            label = f"{m} [{e}]"
            idx, Lv = load_val(e, m)
            if idx is None:
                print(f"[skip] {label}: no validation predictions (rerun with the current code)")
                continue
            tests = {t: load_test(e, m, t) for t in Ytest}
            Xp, Yp, _, metap, _, vap, *_ = get_pool(_exp_input(e))
        if not np.isin(idx, vap).all():      # e.g. an exclusion variant: its rows belong to another pool
            print(f"[skip] {label}: its validation rows are not the validation split of the pool that was loaded")
            continue
        xpk = np.log10(Xp[:, -1])                           # input peak level of every row in this experiment's pool
        n_seeds = Lv.shape[0]
        if n_seeds < 2:
            print(f"[skip] {label}: {n_seeds} member(s) - the ensemble spread needs >= 2 seeds")
            continue
        mu_v, sd_v = Lv.mean(0), Lv.std(0)
        res_v = np.log10(Yp[idx]) - mu_v
        # conditional groups: validation group labels g_v, test group labels g_t (same rule)
        cond = a.cond
        if cond == "class" and int(metap.is_x.values[idx].sum()) < 8:
            print(f"[cond] only {int(metap.is_x.values[idx].sum())} X-class validation events - using 'none'")
            cond = "none"
        if cond == "peak":
            thr = float(np.median(xpk[idx]))
            g_v, g_t = (xpk[idx] >= thr).astype(int), (xpk[:96] >= thr).astype(int)
        elif cond == "class":
            g_v, g_t = metap.is_x.values[idx].astype(int), isx.astype(int)
        else:
            g_v, g_t = np.zeros(len(idx), int), np.zeros(96, int)
        for test, Lt in tests.items():
            if Lt is None or np.isnan(Lt).all():
                continue
            ok = ~np.isnan(Lt).any(axis=(0, 2, 3))
            mu_t, sd_t = Lt.mean(0), Lt.std(0)
            yt = np.log10(Ytest[test])
            res_t = yt - mu_t
            cov = {k: np.full((96, 4), np.nan) for k in ("raw", "abs", "norm")}
            wid = {k: np.full((96, 4), np.nan) for k in ("raw", "abs", "norm")}
            lo_n, hi_n = np.zeros_like(mu_t), np.zeros_like(mu_t)
            z = 1.645
            for b in range(4):
                for bl in blocks:
                    for g in np.unique(g_t):
                        mv, mt = g_v == g, g_t == g
                        if mv.sum() < 8:                     # too few calibration events in the group: pool all
                            mv = np.ones(len(idx), bool)
                        s0 = np.median(sd_v[mv][:, bl, b])
                        qa = cq(np.abs(res_v[mv][:, bl, b]).ravel(), a.level)
                        qn = cq((np.abs(res_v[mv][:, bl, b]) / (sd_v[mv][:, bl, b] + s0)).ravel(), a.level)
                        sdt = sd_t[mt][:, bl, b]
                        halves = {"raw": z * sdt, "abs": np.full(sdt.shape, qa), "norm": qn * (sdt + s0)}
                        for k, half in halves.items():
                            inside = (np.abs(res_t[mt][:, bl, b]) <= half).mean(1)
                            w = (2 * half).mean(1)
                            f = (bl.stop - bl.start) / 180
                            cov[k][mt, b] = np.nan_to_num(cov[k][mt, b]) + inside * f
                            wid[k][mt, b] = np.nan_to_num(wid[k][mt, b]) + w * f
                        lo_n[mt, bl, b] = mu_t[mt][:, bl, b] - halves["norm"]
                        hi_n[mt, bl, b] = mu_t[mt][:, bl, b] + halves["norm"]
            rec = dict(model=label, test=test, cond=cond, n_members=n_seeds, n_val=len(idx), n_test=int(ok.sum()), level=a.level)
            for k in ("raw", "abs", "norm"):
                for b in range(4):
                    rec[f"cov_{k}_{B[b]}"] = float(np.nanmean(cov[k][ok, b]))
                    rec[f"width_{k}_{B[b]}_dex"] = float(np.nanmean(wid[k][ok, b]))
            rec["cov_norm_w0_Xclass"] = float(np.nanmean(cov["norm"][ok & isx, 0])) if (ok & isx).any() else np.nan
            rec["cov_norm_w0_Mclass"] = float(np.nanmean(cov["norm"][ok & ~isx, 0]))
            c, w = [], []
            for i in range(96):
                if not ok[i]:
                    continue
                a_t, zen = event_absorption(Ytest[test][i, :, 0], peak.iloc[i])
                if zen[0] >= 90:
                    continue
                a_lo, _ = event_absorption(10 ** lo_n[i, :, 0], peak.iloc[i])
                a_hi, _ = event_absorption(10 ** hi_n[i, :, 0], peak.iloc[i])
                d = zen < 90
                c.append(((a_t[d] >= a_lo[d]) & (a_t[d] <= a_hi[d])).mean())
                w.append((a_hi[d] - a_lo[d]).mean())
            rec["drap_cov_norm"], rec["drap_width_dB"], rec["drap_n_day"] = float(np.mean(c)), float(np.mean(w)), len(c)
            rows.append(rec)
            if a.save_intervals:
                key = f"{label}|{test}"
                bands[key + "|mu"], bands[key + "|lo"], bands[key + "|hi"] = (x.astype(np.float32) for x in (mu_t, lo_n, hi_n))
            for i in np.where(ok)[0]:
                ev_rows.append(dict(model=label, test=test, event=names[i], is_x=bool(isx[i]),
                                    **{f"cov_norm_{B[b]}": cov["norm"][i, b] for b in range(4)},
                                    **{f"cov_raw_{B[b]}": cov["raw"][i, b] for b in range(4)}))
    os.environ["FISM_INPUT"] = "default"
    pool.load_full.cache_clear()
    S = pd.DataFrame(rows)
    sfx = f"_{a.tag}" if a.tag else ""
    S.to_csv(OUT / f"conformal_split{sfx}.csv", index=False)
    pd.DataFrame(ev_rows).to_csv(OUT / f"conformal_split_events{sfx}.csv", index=False)
    pd.set_option("display.width", 250)
    if len(S):
        print(S[["model", "test", "n_members"] + [f"cov_raw_{b}" for b in B] + [f"cov_norm_{b}" for b in B]
                + ["width_norm_w0_dex", "cov_norm_w0_Xclass", "drap_cov_norm", "drap_width_dB"]].round(3).to_string(index=False))
    print(f"written: {OUT / f'conformal_split{sfx}.csv'}")
    if a.save_intervals and bands:
        np.savez_compressed(OUT / f"conformal_intervals{sfx}.npz", **bands)
        print(f"written: {OUT / f'conformal_intervals{sfx}.npz'} ({len(bands) // 3} model x test entries)")


if __name__ == "__main__":
    main()
