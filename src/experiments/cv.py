"""CV96 experiment runner: fit a model spec on each outer-training fold, predict the held-out fold,
assemble out-of-fold (OOF) predictions for all 96 Zenodo test events, and score them with
src/eval/metrics.py.

A *spec* is a dict (usually from configs/*.yaml):
  name      : label used in tables
  type      : "baseline" | "dl"
  baseline  : key of src.baselines.stats.BASELINES                      (type=baseline)
  configs   : list of PAPER_CONFIGS keys or inline config dicts           (type=dl)
              each config predicts cfg["bands"]; band-specific FCNs are stacked
  overrides : dict merged into every config (e.g. {"bands": [0,1,2,3]})
  target    : "log" (default) | "peaknorm" | "resid_prior"
  in_window : number of most recent input minutes used (default 60)
  issue_delay: k >= 0 - the input window ends k minutes after the peak (w0 minutes 1..k appended); targets stay
              minutes 0..179 from the peak, so scores should be read for lead >= k (notebooks/analyze_paper1.py)
  extra     : list of static features (I-7), standardised on the training fold
  weight_top: loss weight for the top-decile (by peak flux) training events (I-3)
  augment   : {"amp": dex, "noise": dex} amplitude / noise augmentation (I-3)
  train     : {"epochs", "patience", "sched_patience"} (paper: 300 / 30 / 15)
"""
from __future__ import annotations

import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import pool
from src.data.cv96 import BANDS, N_OUT, cv_splits, load_meta, load_test96
from src.eval.metrics import per_event_metrics, pooled_metrics

ROOT = Path(__file__).resolve().parents[2]


def _data():
    """(X, Y, names, meta) of the event pool: the 96 test events (CV96) or test + downloaded
    train/val events (protocol 'full', src/data/pool.py). Test events are always indices 0..95."""
    if pool.protocol() == "full":
        X, Y, names, meta, *_ = pool.load_full()
        return X, Y, names, meta
    X, Y, ev = load_test96()
    return X, Y, ev, load_meta()


TRAIN_DEFAULT = dict(epochs=300, patience=30, sched_patience=15)


# ----------------------------------------------------------------------------------------
# data transforms
# ----------------------------------------------------------------------------------------
def _alpha_fits(Y_raw: np.ndarray) -> np.ndarray:
    """Median per-event curve_fit alpha per band using (t+1)^-alpha, t = 0..179 (exact at peak)."""
    from src.baselines.stats import fit_alpha_curvefit
    t = np.arange(0, N_OUT, dtype=float)
    al = np.zeros(4)
    for b in range(4):
        a = []
        for i in range(len(Y_raw)):
            try:
                a.append(fit_alpha_curvefit(Y_raw[i, :, b], t))
            except Exception:
                pass
        al[b] = np.median(a)
    return al


def prior_curve(xl_peak: np.ndarray, c: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """log10 prior: x_peak + c_b - alpha_b log10(t+1), t = 0..179 -> (N,180,4)."""
    t = np.arange(0, N_OUT, dtype=float)
    return xl_peak[:, None, None] + c[None, None, :] - alpha[None, None, :] * np.log10(t + 1.0)[None, :, None]


def fit_prior(X_raw, Y_raw):
    xl = np.log10(X_raw[:, -1])
    yl0 = np.log10(Y_raw[:, 0, :])
    c = np.median(yl0 - xl[:, None], axis=0)
    c[0] = 0.0  # w0 at the peak equals the last input sample (shared minute)
    return c, _alpha_fits(Y_raw)


FEATURES = ("rise_min_10pct", "rise_slope_10", "x_ratio_60", "hgs_lon", "has_lon", "is_x")


def static_features(names, meta: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    cols = []
    for n in names:
        if n == "has_lon":
            cols.append(meta.hgs_lon.notna().values.astype(float))
        elif n == "hgs_lon":
            cols.append(np.nan_to_num(meta.hgs_lon.values.astype(float) / 90.0, nan=0.0))
        else:
            v = meta[n].values.astype(float)
            if n == "bg_goes_b_24h":            # flux -> log10 (full protocol only)
                v = np.log10(v)
            if np.isnan(v).any():               # optional context features may be missing
                v = np.nan_to_num(v, nan=float(np.nanmedian(v)) if np.isfinite(v).any() else 0.0)
            cols.append(v)
    F = np.stack(cols, axis=1)
    return F[idx]


def make_augment(spec_aug: dict, bands: list[int], beta: np.ndarray):
    import torch
    amp = float(spec_aug.get("amp", 0.0))
    noise = float(spec_aug.get("noise", 0.0))
    b = torch.tensor(beta[bands], dtype=torch.float32)

    def fn(xb, yb, g):
        n = xb.shape[0]
        if amp > 0:
            d = ((torch.rand(n, generator=g) * 2 - 1) * amp).to(xb.device)   # dex shift of the X-ray level
            xb = xb + d[:, None, None]
            yb = yb + d[:, None, None] * b.to(yb.device)[None, None, :]      # band response (train slope)
        if noise > 0:
            xb = xb + torch.randn(xb.shape, generator=g).to(xb.device) * noise
        return xb, yb
    return fn


# ----------------------------------------------------------------------------------------
# one fold / one seed
# ----------------------------------------------------------------------------------------
def _resolve_configs(spec):
    from src.models.nets import PAPER_CONFIGS
    out = []
    for c in spec["configs"]:
        cfg = copy.deepcopy(PAPER_CONFIGS[c]) if isinstance(c, str) else copy.deepcopy(c)
        cfg.update(spec.get("overrides", {}))
        out.append(cfg)
    return out


def run_fold_dl(spec, fold, tr, va, te, seed, threads=1):
    import torch
    torch.set_num_threads(threads)
    from src.models.nets import count_params
    from src.train.trainer import predict_log, train_model

    X, Y, _, meta = _data()
    w = int(spec.get("in_window", 60))
    off = int(spec.get("input_offset", 0))          # I-8: forecast issued `off` minutes before the peak
    k_delay = int(spec.get("issue_delay", 0))       # paper-1 P3: issued `k` minutes AFTER the peak (minutes 1..k observed)
    if k_delay and pool.protocol() == "full" and pool.issue_delay():
        raise ValueError("issue_delay is set both in the spec and by --issue-delay (FISM_ISSUE_DELAY): the window would be shifted twice")
    xl_full, yl = np.log10(X), np.log10(Y)
    if k_delay > 0:
        xl_full = np.concatenate([xl_full, yl[:, 1:k_delay + 1, 0]], axis=1)
    xl = xl_full[:, : xl_full.shape[1] - off][:, -w:]
    target = spec.get("target", "log")
    base = np.zeros_like(yl)
    xlast = xl[:, -1]                               # last *available* input sample (= peak if off == 0)
    if target == "peaknorm":
        base = np.repeat(xlast[:, None, None], 4, axis=2) * np.ones((1, N_OUT, 1))
        for b in set(range(4)) - set(spec.get("pn_bands", [0, 1, 2, 3])):
            base[:, :, b] = 0.0      # hybrid: these bands keep absolute log10 targets
        xin = xl - xlast[:, None]
    elif target == "resid_prior":
        c, alpha = fit_prior(X[tr], Y[tr])
        base = prior_curve(xlast, c, alpha)
        xin = xl
    elif target == "pn_resid":      # peak-normalised input + residual on the band power-law prior
        c, alpha = fit_prior(X[tr], Y[tr])
        base = prior_curve(xlast, c, alpha)
        xin = xl - xlast[:, None]
    else:
        xin = xl
    scale = np.ones((1, 1, 4))
    if spec.get("standardize"):     # global z-score (training fold): x scalar, y per band
        mx, sx = xin[tr].mean(), xin[tr].std() + 1e-8
        xin = (xin - mx) / sx
        r = yl[tr] - base[tr]
        mu_b = r.mean(axis=(0, 1))[None, None, :]
        scale = (r.std(axis=(0, 1)) + 1e-8)[None, None, :]
        base = base + mu_b
    ytgt = (yl - base) / scale

    extra = None
    names = list(spec.get("extra", []))
    if target in ("peaknorm", "pn_resid"):
        names = names + ["__xpeak"]
    if names:
        cols = []
        for n in names:
            if n == "__xpeak":
                cols.append(xlast[:, None])
            else:
                cols.append(static_features([n], meta, np.arange(len(X))))
        F = np.concatenate(cols, axis=1)
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-8
        extra = (F - mu) / sd
        feat_sd = sd

    lw = None
    if spec.get("weight_top"):
        xp = X[tr, -1]
        thr = np.quantile(xp, 0.9)
        lw = np.where(xp >= thr, float(spec["weight_top"]), 1.0)

    beta = np.ones(4)
    if spec.get("augment"):
        xp = np.log10(X[tr, -1])
        for b in range(4):
            beta[b] = np.polyfit(xp, yl[tr, :, b].mean(1), 1)[0]
        beta[0] = 1.0

    tk = dict(TRAIN_DEFAULT)
    tk.update(spec.get("train", {}))
    pred = np.zeros((len(te), N_OUT, 4))
    info = {"fold": fold, "seed": seed, "params": 0, "seconds": 0.0, "epochs": [], "best_val": []}
    if spec.get("experts"):   # I-4: one model per flare type, gate = training-fold median of a feature
        feat = meta[spec["experts"]].values.astype(float)
        thr = float(np.median(feat[tr]))
        info["expert_threshold"] = thr
        sub = dict(spec)
        sub.pop("experts")
        for grp in (feat <= thr, feat > thr):
            g_tr, g_va, g_te = tr[grp[tr]], va[grp[va]], te[grp[te]]
            if len(g_va) < 2:   # borrow validation events from the other group if needed
                g_va = va
            if len(g_te) == 0:
                continue
            _, p, inf = run_fold_dl(sub, fold, g_tr, g_va, g_te, seed, threads)
            pos = np.searchsorted(te, g_te)
            pred[pos] = p
            info["params"] += inf["params"]
            info["seconds"] += inf["seconds"]
            info["epochs"] += inf["epochs"]
        return te, pred, info
    perm_rows = []
    for cfg in _resolve_configs(spec):
        bands = cfg.get("bands", [0, 1, 2, 3])
        aug = make_augment(spec["augment"], bands, beta) if spec.get("augment") else None
        model, h = train_model(cfg, xin[tr], ytgt[tr], xin[va], ytgt[va], seed=seed,
                               extra_tr=None if extra is None else extra[tr],
                               extra_va=None if extra is None else extra[va],
                               loss_weights=lw, augment_fn=aug, **tk)
        p = predict_log(model, xin[te], None if extra is None else extra[te])
        pred[:, :, bands] = p
        if pool.protocol() == "full":          # validation predictions (log10) for split conformal
            pv = info.setdefault("_val_pred", np.zeros((len(va), N_OUT, 4)))
            pv[:, :, bands] = predict_log(model, xin[va], None if extra is None else extra[va])
        if spec.get("scale_test") and not spec.get("standardize") and target in ("log", "peaknorm"):
            # Phase-2 item 5: input x f -> shift of the final log10 output (pure scaling => +log10 f)
            sh = info.setdefault("scale_shift", {})
            for fct in spec["scale_test"]:
                lf = np.log10(fct)
                if target == "log":
                    ps = predict_log(model, xin[te] + lf, None if extra is None else extra[te])
                    shift = ps - p
                else:  # peaknorm: shape input unchanged, peak feature (last extra column) and base shift
                    e2 = extra[te].copy()
                    e2[:, -1] += lf / feat_sd[-1]
                    shift = predict_log(model, xin[te], e2) - p + lf
                sh.setdefault(str(fct), {})[",".join(map(str, bands))] = shift.mean(axis=1).tolist()
        if spec.get("perm_importance") and extra is not None:   # I-7 permutation importance (test fold)
            rng = np.random.default_rng(1000 + seed)
            base_mse = float(np.mean((p - ytgt[te][:, :, bands]) ** 2))
            for j, n in enumerate(names):
                vals = []
                for _ in range(10):
                    e2 = extra[te].copy()
                    e2[:, j] = rng.permutation(e2[:, j])
                    p2 = predict_log(model, xin[te], e2)
                    vals.append(float(np.mean((p2 - ytgt[te][:, :, bands]) ** 2)) - base_mse)
                perm_rows.append({"feature": n, "delta_mse_log": float(np.mean(vals))})
            info["perm_importance"] = perm_rows
        info["params"] += count_params(model)
        info["seconds"] += h["seconds"]
        info["epochs"].append(h["epochs_run"])
        info["best_val"].append(h["best_val"])
        # CPU latency for one event (batch 1), median of 50 calls
        import torch as _t
        _dev = next(model.parameters()).device
        xi = _t.tensor(xin[te[:1]], dtype=_t.float32, device=_dev).unsqueeze(-1)
        ei = None if extra is None else _t.tensor(extra[te[:1]], dtype=_t.float32, device=_dev)
        ts = []
        with _t.no_grad():
            for _ in range(50):
                t0 = time.perf_counter()
                model(xi, ei)
                ts.append(time.perf_counter() - t0)
        info.setdefault("latency_ms", 0.0)
        info["latency_ms"] += 1000 * float(np.median(ts))
    pred = pred * scale + base[te]
    if "_val_pred" in info:
        info["_val_pred"] = info["_val_pred"] * scale + base[va]
        info["_val_idx"] = va
    return te, pred, info


def run_fold_baseline(spec, fold, tr, va, te, seed):
    from src.baselines.stats import BASELINES
    X, Y, _, _ = _data()
    cls = BASELINES[spec["baseline"]]
    kw = dict(spec.get("kwargs", {}))
    if spec["baseline"] == "rf":
        kw.setdefault("random_state", 42 + seed)
        kw.setdefault("n_jobs", 1)
    m = cls(**kw)
    trv = np.concatenate([tr, va])  # baselines have no early stopping: use train+val
    t0 = time.time()
    m.fit(X[trv], Y[trv])
    p = m.predict(X[te])
    return te, np.log10(np.maximum(p, 1e-30)), {"fold": fold, "seed": seed, "seconds": time.time() - t0,
                                                "params": 0}


def _job(args):
    spec, fold, tr, va, te, seed = args
    if spec["type"] == "baseline":
        return spec["name"], seed, run_fold_baseline(spec, fold, tr, va, te, seed)
    from src.train.trainer import OOM, force_cpu, reselect_device

    def is_oom(err):
        return isinstance(err, OOM) or "out of memory" in str(err).lower()
    try:
        return spec["name"], seed, run_fold_dl(spec, fold, tr, va, te, seed)
    except Exception as err:  # noqa: BLE001
        if not is_oom(err):
            raise
        # CUDA out of memory: once more on a (re-selected) GPU, then on the CPU
        print(f"[cv] CUDA out of memory for {spec['name']} s{seed} fold {fold} - re-selecting the GPU and retrying", flush=True)
        try:
            reselect_device()
            return spec["name"], seed, run_fold_dl(spec, fold, tr, va, te, seed)
        except Exception as err2:  # noqa: BLE001
            if not is_oom(err2):
                raise
            print(f"[cv] CUDA out of memory again for {spec['name']} s{seed} fold {fold} - retrying on the CPU", flush=True)
            force_cpu()
            return spec["name"], seed, run_fold_dl(spec, fold, tr, va, te, seed)


# ----------------------------------------------------------------------------------------
# experiment driver
# ----------------------------------------------------------------------------------------
def run_specs(exp_id: str, specs: list[dict], seeds=(0, 1, 2), workers: int = 2, n_folds: int = 10,
              skip_existing: bool = True, folds: list[int] | None = None, fold_scheme: str = "interleaved"):
    out = ROOT / "results" / exp_id
    out.mkdir(parents=True, exist_ok=True)
    full = pool.protocol() == "full"
    if full:      # one fit per seed: paper-style train/val; test = Zenodo 96 + the same 96 rebuilt (latis)
        _, _, names_all, _, tr, va, te, te_latis = pool.load_full()
        ex_idx, ex_rows = pool.extra_index()          # flares outside the isolation rule (FISM_PREDICT_EXTRA=1), else empty
        ops = pool.ops_index()                        # twin rows with the operational GOES input (FISM_PREDICT_OPS=1), else empty
        ops_idx = ops["idx"]
        splits = [(0, tr, va, np.concatenate([te, te_latis, ex_idx, ops_idx]).astype(int))]
        print(f"[{exp_id}] {pool.describe()}", flush=True)
        if len(ex_idx):
            fold, txf = pool.extra_fold(), pool.train_extra_fold()
            pd.DataFrame(dict(pool_index=ex_idx, dataset_row=ex_rows, event=[names_all[i].split("|")[0] for i in ex_idx], fold=fold,
                              trained=(fold == int(txf)) if txf != "" else np.zeros(len(fold), bool))).to_csv(out / "extra_rows.csv", index=False)
        if len(ops_idx):
            pd.DataFrame(dict(pool_index=ops_idx, dataset_row=ops["rows"], event=[names_all[i].split("|")[0] for i in ops_idx], kind=ops["kind"],
                              variant=ops["variant"], twin_pool_index=ops["twin"])).to_csv(out / "ops_rows.csv", index=False)
    else:
        splits = list(cv_splits(n_folds, fold_scheme))
    if folds is not None:
        splits = [s for s in splits if s[0] in folds]
    jobs = []
    for spec in specs:
        s_seeds = tuple(spec.get("seeds", seeds)) if spec["type"] == "dl" or spec.get("baseline") == "rf" else (0,)
        for seed in s_seeds:
            f = out / f"oof_{spec['name']}_s{seed}.npy"
            if skip_existing and f.exists():
                continue
            for fold, tr, va, te in splits:
                jobs.append((spec, fold, tr, va, te, seed))
    print(f"[{exp_id}] {len(jobs)} fold-jobs", flush=True)
    res: dict[tuple, list] = {}
    t0 = time.time()
    if workers > 1 and len(jobs) > 1:
        # workers are forked (they inherit the loaded data); FISM_MP_CONTEXT=spawn starts fresh interpreters
        # instead (each reloads the data) - the escape hatch if something in this process has touched CUDA
        # or autograd before this point, which makes forked workers unable to use either.
        import multiprocessing as mp
        import os
        ctx_name = os.environ.get("FISM_MP_CONTEXT", "fork" if hasattr(os, "fork") else "spawn")
        if ctx_name != "fork":
            print(f"[{exp_id}] worker start method: {ctx_name}", flush=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context(ctx_name)) as ex:
            for i, (name, seed, r) in enumerate(ex.map(_job, jobs)):
                res.setdefault((name, seed), []).append(r)
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)
    else:
        for i, j in enumerate(jobs):
            name, seed, r = _job(j)
            res.setdefault((name, seed), []).append(r)
    X, Y, ev = load_test96()
    for (name, seed), parts in res.items():
        oof = np.full((len(ev), N_OUT, 4), np.nan)
        oofL = np.full((len(ev), N_OUT, 4), np.nan)
        infos = []
        for te, p, info in parts:
            m = te < len(ev)
            oof[te[m]] = p[m]
            if full:
                ml = (te >= len(ev)) & (te < 2 * len(ev))
                oofL[te[ml] - len(ev)] = p[ml]
                if len(ex_idx):                   # predictions for the flares outside the isolation rule -> oofN_*
                    mx = np.isin(te, ex_idx)
                    oofN = np.full((len(ex_idx), N_OUT, 4), np.nan)
                    oofN[np.searchsorted(ex_idx, te[mx])] = p[mx]
                    np.save(out / f"oofN_{name}_s{seed}.npy", oofN.astype(np.float32))
                if len(ops_idx):                  # the same flares with the operational GOES input -> oofO_*
                    mo = np.isin(te, ops_idx)
                    oofO = np.full((len(ops_idx), N_OUT, 4), np.nan)
                    oofO[np.searchsorted(ops_idx, te[mo])] = p[mo]
                    np.save(out / f"oofO_{name}_s{seed}.npy", oofO.astype(np.float32))
            if "_val_pred" in info:          # full protocol: validation predictions -> val_<name>_s<seed>.npz
                np.savez_compressed(out / f"val_{name}_s{seed}.npz", idx=info.pop("_val_idx"),
                                    pred_log=info.pop("_val_pred").astype(np.float32))
            infos.append(info)
        if full and not np.isnan(oofL).all():   # rebuilt test set (rows of missing events stay NaN)
            np.save(out / f"oofL_{name}_s{seed}.npy", oofL.astype(np.float32))
        if np.isnan(oof).any():
            print(f"  WARNING {name} s{seed}: incomplete OOF ({np.isnan(oof).any(axis=(1, 2)).sum()} events)")
            continue
        np.save(out / f"oof_{name}_s{seed}.npy", oof.astype(np.float32))
        with open(out / f"info_{name}_s{seed}.json", "w") as fh:
            json.dump(infos, fh, default=float)
    print(f"[{exp_id}] finished in {time.time() - t0:.0f}s", flush=True)
    return out


def load_oof(exp_id: str, name: str, seed: int) -> np.ndarray:
    return 10 ** np.load(ROOT / "results" / exp_id / f"oof_{name}_s{seed}.npy").astype(np.float64)


def available(exp_id: str):
    d = ROOT / "results" / exp_id
    out = {}
    for f in sorted(d.glob("oof_*_s*.npy")):
        stem = f.stem[4:]
        name, s = stem.rsplit("_s", 1)
        out.setdefault(name, []).append(int(s))
    return out


def _score(out: Path, exp_id: str, names, av, Y, ev, prefix: str, suffix: str):
    ev_rows, pool_rows = [], []
    for name in names:
        for seed in sorted(av.get(name, [])):
            f = out / f"{prefix}_{name}_s{seed}.npy"
            if not f.exists():
                continue
            P = 10 ** np.load(f).astype(np.float64)
            ok = ~np.isnan(P).any(axis=(1, 2))                  # rebuilt set: score available events only
            pe = per_event_metrics(Y[ok], P[ok])
            pm = pooled_metrics(Y[ok], P[ok])
            evs = [e for e, o in zip(ev, ok) if o]
            for b, bn in enumerate(BANDS):
                for i, e in enumerate(evs):
                    ev_rows.append(dict(model=name, seed=seed, band=bn, event=e,
                                        **{k: float(pe[k][i, b]) for k in pe}))
                pool_rows.append(dict(model=name, seed=seed, band=bn, n_events=int(ok.sum()),
                                      **{k: float(v[b]) for k, v in pm.items()}))
    E = pd.DataFrame(ev_rows)
    Pm = pd.DataFrame(pool_rows)
    if Pm.empty:
        return E, Pm, Pm
    E.to_csv(out / f"metrics_event{suffix}.csv", index=False)
    Pm.to_csv(out / f"metrics_pooled{suffix}.csv", index=False)
    metrics = ["CC", "RMSE", "MRE", "MAPE", "APE", "Mean_event_CC", "Median_event_RMSE"]
    S = Pm.groupby(["model", "band"])[metrics].agg(["mean", "std", "count"])
    S.columns = [f"{a}_{b}" for a, b in S.columns]
    S = S.reset_index()
    S.to_csv(out / f"summary{suffix}.csv", index=False)
    return E, Pm, S


def score_experiment(exp_id: str, names: list[str] | None = None):
    """Writes metrics_event.csv, metrics_pooled.csv and summary.csv in results/<exp_id>/ (Zenodo test arrays);
    protocol 'full' additionally writes *_latis.csv for the rebuilt test events."""
    X, Y, ev = load_test96()
    out = ROOT / "results" / exp_id
    av = available(exp_id)
    names = names or list(av)
    res = _score(out, exp_id, names, av, Y, ev, "oof", "")
    if pool.protocol() == "full":
        YL = pool.load_full()[1][len(ev): 2 * len(ev)]
        _score(out, exp_id, names, av, YL, ev, "oofL", "_latis")
    return res
