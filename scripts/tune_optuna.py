"""I-10 — nested Optuna re-tuning of an adopted CV96 pipeline (validation events only; the outer test
fold is never used for tuning).

The pipeline (preprocessing, augmentation, features, window) is taken from a spec in a config file;
only the network hyper-parameters are searched (paper search space). For every outer fold: TPE search
(n trials; tuning budget 100 epochs / early-stop 15 / plateau 8 as in the Zenodo tuning scripts) on
inner-train -> inner-val loss; the best configuration is then retrained with the final protocol
(300 / 30 / 15) for each seed and the outer test fold is predicted. OOF -> results/p3_optuna/.

usage: python scripts/tune_optuna.py --config p3_final --spec GRU-final --family gru --trials 50
       python scripts/tune_optuna.py --protocol full --config p2_zscore --spec FCN-z --family fcn --trials 40
       (full: one TPE study on the train/val split; the test events - Zenodo 96 + rebuilt 96 - are predicted
        with the best configuration per seed; OOF -> results/p3_optuna_full/)
"""
import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def suggest(trial, family):
    if family in ("gru", "lstm", "rnn"):
        return dict(kind=family, hidden_dim=trial.suggest_int("hidden_dim", 32, 256),
                    num_layers=trial.suggest_int("num_layers", 1, 3),
                    dropout=trial.suggest_float("dropout", 0.0, 0.3),
                    lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                    batch_size=trial.suggest_categorical("batch_size", [8, 16, 32, 64]), bands=[0, 1, 2, 3])
    if family == "transformer":
        return dict(kind="transformer", d_model=trial.suggest_categorical("d_model", [32, 64, 128]),
                    nhead=trial.suggest_categorical("nhead", [2, 4, 8]),
                    num_layers=trial.suggest_int("num_layers", 1, 4),
                    dropout=trial.suggest_float("dropout", 0.0, 0.3),
                    lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                    batch_size=trial.suggest_categorical("batch_size", [8, 16, 32, 64]), bands=[0, 1, 2, 3])
    if family == "fcn":
        return dict(kind="fcn", num_expansion=trial.suggest_int("num_expansion", 1, 3),
                    scale_factor=trial.suggest_float("scale_factor", 0.3, 1.5),
                    activation=trial.suggest_categorical("activation", ["selu", "relu", "gelu"]),
                    use_bn=trial.suggest_categorical("use_bn", [True, False]),
                    dropout=trial.suggest_float("dropout", 0.0, 0.3),
                    lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                    batch_size=trial.suggest_categorical("batch_size", [8, 16, 32, 64]))
    raise ValueError(family)


def fold_job_fcn(args):
    """Band-specific FCN pipeline: one TPE study per band configuration (anchor flag kept from the base)."""
    spec, family, fold, tr, va, te, trials, seeds = args
    import optuna
    from src.experiments.cv import _resolve_configs, _job
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tune_train = dict(epochs=100, patience=15, sched_patience=8)
    t0 = time.time()
    base_cfgs = _resolve_configs(spec)
    best_cfgs, best_params, best_vals = [], [], []
    for j, bc in enumerate(base_cfgs):
        def objective(trial, bc=bc):
            c = suggest(trial, "fcn")
            c.update(bands=bc["bands"], anchor=bc.get("anchor", False))
            s = copy.deepcopy(spec)
            s["configs"] = [c]
            s.pop("overrides", None)
            s["train"] = tune_train
            _, _, info = _job((s, fold, tr, va, va, 0))[2]
            return float(np.mean(info["best_val"]))
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=100 * fold + j))
        study.optimize(objective, n_trials=trials)
        c = suggest(optuna.trial.FixedTrial(study.best_params), "fcn")
        c.update(bands=bc["bands"], anchor=bc.get("anchor", False))
        best_cfgs.append(c)
        best_params.append(study.best_params)
        best_vals.append(study.best_value)
    preds = {}
    for sd in seeds:
        s = copy.deepcopy(spec)
        s["configs"] = best_cfgs
        s.pop("overrides", None)
        s.pop("train", None)
        _, p, _ = _job((s, fold, tr, va, te, sd))[2]
        preds[sd] = p
    return fold, te, preds, dict(best_params=best_params, best_val=best_vals, seconds=time.time() - t0)


def fold_job(args):
    spec, family, fold, tr, va, te, trials, seeds = args
    import optuna
    from src.experiments.cv import _job
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tune_train = dict(epochs=100, patience=15, sched_patience=8)

    def objective(trial):
        s = copy.deepcopy(spec)
        s["configs"] = [suggest(trial, family)]
        s["train"] = tune_train
        _, _, info = _job((s, fold, tr, va, va, 0))[2]   # predicts only validation events
        return float(np.mean(info["best_val"]))

    t0 = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=fold))
    study.optimize(objective, n_trials=trials)
    best = suggest(optuna.trial.FixedTrial(study.best_params), family)
    preds = {}
    for sd in seeds:
        s = copy.deepcopy(spec)
        s["configs"] = [best]
        s.pop("train", None)
        _, p, _ = _job((s, fold, tr, va, te, sd))[2]
        preds[sd] = p
    return fold, te, preds, dict(best_params=study.best_params, best_val=study.best_value, seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="p3_final")
    ap.add_argument("--spec", default="GRU-final")
    ap.add_argument("--family", default="gru")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--protocol", choices=["cv96", "full"], default="cv96")
    ap.add_argument("--events", default=None, help="full: built npz (default data/processed/events_latis.npz)")
    ap.add_argument("--tag", default="", help="suffix for the result name, e.g. t100 -> <spec>-optuna-t100")
    ap.add_argument("--device", default=None, help="auto | cpu | cuda | cuda:N (env FISM_DEVICE)")
    a = ap.parse_args()
    if a.device:
        os.environ["FISM_DEVICE"] = a.device
    os.environ["FISM_PROTOCOL"] = a.protocol
    if a.protocol == "full" and a.events:
        os.environ["FISM_EVENTS"] = str(Path(a.events).resolve())
    from src.experiments.cv import score_experiment
    cfg = yaml.safe_load(open(ROOT / "configs" / f"{a.config}.yaml"))
    spec = [s for s in cfg["specs"] if s["name"] == a.spec][0]
    name = f"{a.spec}-optuna" + (f"-{a.tag}" if a.tag else "")
    exp_id = "p3_optuna" + ("_full" if a.protocol == "full" else "")
    out = ROOT / "results" / exp_id
    out.mkdir(parents=True, exist_ok=True)
    if a.protocol == "full":
        from src.data import pool
        _, _, _, _, tr, va, te, te_latis = pool.load_full()
        splits = [(0, tr, va, np.concatenate([te, te_latis]))]
        print(pool.describe(), flush=True)
    else:
        from src.data.cv96 import cv_splits
        splits = list(cv_splits())
    jobs = [(spec, a.family, f, tr, va, te, a.trials, a.seeds) for f, tr, va, te in splits]
    oof = {s: np.full((96, 180, 4), np.nan) for s in a.seeds}
    oofL = {s: np.full((96, 180, 4), np.nan) for s in a.seeds}
    infos = []
    fn = fold_job_fcn if a.family == "fcn" else fold_job
    with ProcessPoolExecutor(max_workers=min(a.workers, len(jobs))) as ex:
        for fold, te, preds, info in ex.map(fn, jobs):
            for s, p in preds.items():
                m = te < 96
                oof[s][te[m]] = p[m]
                ml = (te >= 96) & (te < 192)
                oofL[s][te[ml] - 96] = p[ml]
            infos.append(dict(fold=fold, **info))
            print(f"fold {fold}: {info['best_params']} val {info['best_val']} ({info['seconds']:.0f}s)", flush=True)
    for s in a.seeds:
        np.save(out / f"oof_{name}_s{s}.npy", oof[s].astype(np.float32))
        if a.protocol == "full":
            np.save(out / f"oofL_{name}_s{s}.npy", oofL[s].astype(np.float32))
    json.dump(infos, open(out / f"optuna_{name}.json", "w"), indent=1, default=float)
    score_experiment(exp_id)


if __name__ == "__main__":
    main()
