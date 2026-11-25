"""Single entry point:  python run.py --exp <config name>   (configs/<name>.yaml)

Runs every spec of the experiment on the CV96 protocol (src/experiments/cv.py), writes OOF
predictions, per-event / pooled metrics and summary.csv to results/<exp_id>/.

  --protocol full   paper setting: train/val on the downloaded FISM2 events (scripts/download,
                    data/processed/events_latis.npz), test on the 96 Zenodo events; results go to
                    results/<exp_id>_full[...]/ so CV96 results are never mixed or overwritten.
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.experiments.cv import run_specs, score_experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="config name in configs/ (without .yaml)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--protocol", choices=["cv96", "full"], default="cv96")
    ap.add_argument("--events", default=None, help="full: built npz (default data/processed/events_latis.npz)")
    ap.add_argument("--input", choices=["default", "fism", "goes_b", "goes_conv", "goes_convnn"], default="default",
                    help="full: input series (default = as calibrated; fism / goes_b = variants; goes_conv[nn] = "
                         "GOES converted to the FISM2 band by a linear / FCN converter fitted on the training events)")
    ap.add_argument("--exclude-near-test-hours", type=float, default=0.0,
                    help="full: drop train/val flares within H hours of a test flare (leakage check)")
    ap.add_argument("--exclude-random", type=int, default=0,
                    help="full: drop N random train/val candidates (size-matched control for --exclude-near-test-hours)")
    ap.add_argument("--exclude-seed", type=int, default=0)
    ap.add_argument("--split-like-goes", action="store_true",
                    help="full: train/val assignment of the GOES-input pools (only flares with a GOES XRS-B window; control experiment 1)")
    ap.add_argument("--exclude-same-ar", action="store_true",
                    help="full: drop train/val flares from the same active region as a test flare (AR-level leakage check)")
    ap.add_argument("--device", default=None, help="auto (default) | cpu | cuda | cuda:N  (env FISM_DEVICE)")
    ap.add_argument("--tag", default=None, help="suffix added to the results folder (rerun without overwriting)")
    ap.add_argument("--seeds", default=None, help="comma list overriding the config, e.g. 0,1,2,3,4")
    ap.add_argument("--models", default=None, help="comma list: run only these specs of the config, e.g. GRU-z,FCN-z")
    ap.add_argument("--issue-delay", type=int, default=0,
                    help="full: forecast issued K minutes after the peak; every input window covers minutes -59+K..K "
                         "(GOES inputs need data/processed/events_goes_post.npz). Read the scores for lead >= K")
    ap.add_argument("--predict-extra", action="store_true",
                    help="full: also predict the flares that fail the isolation rule (oofN_*.npy, extra_rows.csv)")
    ap.add_argument("--train-extra-fold", choices=["0", "1"], default=None,
                    help="full, with --predict-extra: add the flares outside the isolation rule of this fold (time clusters) to the "
                         "training rows; score the flares of the other fold only (extra_rows.csv: fold, trained)")
    ap.add_argument("--predict-ops", action="store_true",
                    help="full, GOES inputs: also predict the rebuilt test flares (and, with --predict-extra, the flares outside the rule) "
                         "from the OPERATIONAL GOES windows of scripts/build_goes_ops.py (oofO_*.npy, ops_rows.csv); never used for fitting")
    ap.add_argument("--goes-ops", default=None, help="npz of scripts/build_goes_ops.py (default data/processed/events_goes_ops.npz)")
    a = ap.parse_args()
    if a.train_extra_fold is not None and not a.predict_extra:
        ap.error("--train-extra-fold needs --predict-extra")
    if a.predict_ops and (not a.input.startswith("goes_") or a.issue_delay > 0 or a.protocol != "full"):
        ap.error("--predict-ops needs --protocol full, a GOES input (goes_b, goes_conv, goes_convnn) and no --issue-delay")
    cfg = yaml.safe_load(open(ROOT / "configs" / f"{a.exp}.yaml"))
    exp_id = cfg.get("exp_id", a.exp)
    if a.seeds:
        cfg["seeds"] = [int(x) for x in a.seeds.split(",")]
    if a.models:
        keep = [m.strip() for m in a.models.split(",")]
        unknown = sorted(set(keep) - {s["name"] for s in cfg["specs"]})
        if unknown:
            ap.error(f"--models: not in configs/{a.exp}.yaml: {unknown}")
        cfg["specs"] = [s for s in cfg["specs"] if s["name"] in keep]
    os.environ["FISM_PROTOCOL"] = a.protocol
    if a.device:
        os.environ["FISM_DEVICE"] = a.device
    if a.protocol == "full":
        os.environ["FISM_EXCLUDE_SAME_AR"] = "1" if a.exclude_same_ar else "0"
        os.environ["FISM_SPLIT_LIKE_GOES"] = "1" if a.split_like_goes else "0"
        if a.events:
            os.environ["FISM_EVENTS"] = str(Path(a.events).resolve())
        os.environ["FISM_INPUT"] = a.input
        os.environ["FISM_EXCLUDE_NEAR_TEST_H"] = str(a.exclude_near_test_hours)
        os.environ["FISM_EXCLUDE_RANDOM_N"] = str(a.exclude_random)
        os.environ["FISM_EXCLUDE_SEED"] = str(a.exclude_seed)
        os.environ["FISM_ISSUE_DELAY"] = str(a.issue_delay)
        os.environ["FISM_PREDICT_EXTRA"] = "1" if a.predict_extra else "0"
        os.environ["FISM_TRAIN_EXTRA_FOLD"] = a.train_extra_fold or ""
        os.environ["FISM_PREDICT_OPS"] = "1" if a.predict_ops else "0"
        if a.goes_ops:
            os.environ["FISM_GOES_OPS"] = str(Path(a.goes_ops).resolve())
        exp_id += "_full" + ("" if a.input == "default" else f"_in-{a.input}") + \
            (f"_ex{a.exclude_near_test_hours:g}h" if a.exclude_near_test_hours > 0 else "") + \
            (f"_rnd{a.exclude_random}s{a.exclude_seed}" if a.exclude_random > 0 else "") + \
            ("_exAR" if a.exclude_same_ar else "") + \
            ("_splitgoes" if a.split_like_goes else "") + \
            (f"_d{a.issue_delay}" if a.issue_delay > 0 else "") + \
            (f"_tx{a.train_extra_fold}" if a.train_extra_fold is not None else "") + \
            ("_ops" if a.predict_ops else "")
    if a.tag:
        exp_id += f"_{a.tag}"
    if not a.score_only:
        run_specs(exp_id, cfg["specs"], seeds=tuple(cfg.get("seeds", [0, 1, 2])),
                  workers=a.workers or cfg.get("workers", 2), fold_scheme=cfg.get("fold_scheme", "interleaved"))
    names = [s["name"] for s in cfg["specs"]]
    _, _, S = score_experiment(exp_id, names)
    cols = ["model", "band", "CC_mean", "CC_std", "RMSE_mean", "MRE_mean", "MAPE_mean", "APE_mean"]
    print(S[cols].to_string(index=False))


if __name__ == "__main__":
    main()
