"""Phase 3 improvement ledger (results/improve/ledger.csv).

Each row compares a variant with its base model on the same 96 events (CV96 OOF, 3-seed average of
per-event metrics). Pre-specified decision rule (docs/decisions.md):
  primary   : per-event MAPE averaged over the 4 bands (lower is better)
  secondary : per-event CC averaged over the 4 bands, pooled CC (paper metric), log10-RMSE
  test      : two-sided Wilcoxon signed-rank on the 96 paired per-event primary values + paired
              bootstrap 95 % CI of the mean difference (10 000 resamples)
  ADOPT     : primary improves with p < 0.05 AND band-mean per-event CC does not degrade
              significantly (p < 0.05). Otherwise REJECT (or 'neutral' if p >= 0.05 both ways).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.cv96 import load_test96  # noqa: E402
from src.eval.stats_tests import paired_bootstrap_ci  # noqa: E402
from src.experiments.cv import available, load_oof  # noqa: E402

X, Y, ev = load_test96()


def per_event_table(exp, name):
    E = pd.read_csv(ROOT / f"results/{exp}/metrics_event.csv")
    E = E[E.model == name]
    if not len(E):
        raise KeyError(f"{exp}:{name}")
    g = E.groupby(["event", "band"])[["CC", "MAPE", "APE", "RMSE"]].mean().reset_index()
    w = g.pivot(index="event", columns="band", values=["CC", "MAPE", "APE"])
    out = pd.DataFrame({"CC4": w["CC"].mean(axis=1), "MAPE4": w["MAPE"].mean(axis=1), "APE4": w["APE"].mean(axis=1)})
    for b in ["w0", "w1", "w2", "w3"]:
        out[f"MAPE_{b}"] = w["MAPE"][b]
        out[f"CC_{b}"] = w["CC"][b]
    return out.loc[ev]


def pooled(exp, name):
    P = pd.read_csv(ROOT / f"results/{exp}/metrics_pooled.csv")
    P = P[P.model == name]
    return P.groupby("band")[["CC", "RMSE", "MAPE"]].mean()


def log_rmse(exp, name):
    seeds = available(exp)[name]
    v = [np.sqrt(np.mean((np.log10(load_oof(exp, name, s)) - np.log10(Y)) ** 2)) for s in seeds]
    return float(np.mean(v))


def compare(idea, hypothesis, change, exp, name, base_exp, base_name, extra=None):
    a = per_event_table(exp, name)
    b = per_event_table(base_exp, base_name)
    d_m = (a.MAPE4 - b.MAPE4).values
    d_c = (a.CC4 - b.CC4).values
    p_m = wilcoxon(a.MAPE4, b.MAPE4).pvalue
    p_c = wilcoxon(a.CC4, b.CC4).pvalue
    lo, hi, _ = paired_bootstrap_ci(d_m)
    pa, pb = pooled(exp, name), pooled(base_exp, base_name)
    improve = d_m.mean() < 0 and p_m < 0.05
    cc_hurt = d_c.mean() < 0 and p_c < 0.05
    worse = d_m.mean() > 0 and p_m < 0.05
    decision = "ADOPT" if (improve and not cc_hurt) else ("REJECT (worse)" if worse or cc_hurt else "neutral (n.s.)")
    rec = dict(idea=idea, hypothesis=hypothesis, change=change, variant=f"{exp}:{name}", base=f"{base_exp}:{base_name}",
               MAPE4_base=b.MAPE4.mean(), MAPE4_var=a.MAPE4.mean(), dMAPE4=d_m.mean(), dMAPE4_ci_lo=lo, dMAPE4_ci_hi=hi,
               p_MAPE4=p_m, CC4_base=b.CC4.mean(), CC4_var=a.CC4.mean(), dCC4=d_c.mean(), p_CC4=p_c,
               pooledCC4_base=pb.CC.mean(), pooledCC4_var=pa.CC.mean(),
               logRMSE_base=log_rmse(base_exp, base_name), logRMSE_var=log_rmse(exp, name),
               win_frac=float(np.mean(d_m < 0)), decision=decision)
    for bnd in ["w0", "w1", "w2", "w3"]:
        rec[f"dMAPE_{bnd}"] = (a[f"MAPE_{bnd}"] - b[f"MAPE_{bnd}"]).mean()
    if extra:
        rec.update(extra)
    return rec


# (idea, hypothesis, change, exp, variant, base_exp, base)
G = ("p2_norm", "GRU-peaknorm")
F = ("p2_zscore", "FCN-z")   # FCN family base: z-score beat peak normalisation (P2-5)
PLAN = [
    ("P2-5", "event-wise peak normalisation fixes the small-data collapse", "input/target relative to the peak (+peak feature)", "p2_norm", "GRU-peaknorm", "repro_cv96", "GRU"),
    ("P2-5", "event-wise peak normalisation fixes the small-data collapse", "input/target relative to the peak (+peak feature)", "p2_norm", "FCN-peaknorm", "repro_cv96", "FCN"),
    ("P2-5", "event-wise peak normalisation helps", "peak normalisation", "p2_norm2", "LSTM-peaknorm", "repro_cv96", "LSTM"),
    ("P2-5", "event-wise peak normalisation helps", "peak normalisation", "p2_norm2", "RNN-peaknorm", "repro_cv96", "RNN"),
    ("P2-5", "event-wise peak normalisation helps", "peak normalisation", "p2_norm2", "Transformer-peaknorm", "repro_cv96", "Transformer"),
    ("P2-5", "global z-score standardisation helps", "z-score of log10 input / per-band targets", "p2_zscore", "GRU-z", "repro_cv96", "GRU"),
    ("P2-5", "global z-score standardisation helps", "z-score of log10 input / per-band targets", "p2_zscore", "FCN-z", "repro_cv96", "FCN"),
    ("P2-5", "global z-score standardisation helps", "z-score", "p2_zscore", "LSTM-z", "repro_cv96", "LSTM"),
    ("P2-5", "global z-score standardisation helps", "z-score", "p2_zscore", "RNN-z", "repro_cv96", "RNN"),
    ("P2-5", "global z-score standardisation helps", "z-score", "p2_zscore", "Transformer-z", "repro_cv96", "Transformer"),
    ("P2-5", "peak normalisation vs z-score", "z-score instead of peak normalisation", "p2_zscore", "GRU-z", *G),
    ("P2-5", "peak normalisation vs z-score", "z-score instead of peak normalisation", "p2_zscore", "FCN-z", "p2_norm", "FCN-peaknorm"),
    ("P2-5", "hybrid normalisation beats peak normalisation", "w0 peak-relative, EUV absolute, z-scored", "p2_hybrid", "GRU-hyb", *G),
    ("P2-5", "hybrid normalisation beats peak normalisation", "w0 peak-relative, EUV absolute, z-scored", "p2_hybrid", "FCN-hyb", "p2_norm", "FCN-peaknorm"),
    ("P2-5", "hybrid normalisation beats z-score (FCN)", "hybrid instead of z-score", "p2_hybrid", "FCN-hyb", *F),
    ("P2-5", "hybrid normalisation helps the Transformer", "hybrid vs paper preprocessing", "p2_hybrid", "Transformer-hyb", "repro_cv96", "Transformer"),
    ("P2-8", "a shorter input window is enough", "input window 45 min", "p2_window", "GRU-pn-w45", *G),
    ("P2-8", "a shorter input window is enough", "input window 30 min", "p2_window", "GRU-pn-w30", *G),
    ("P2-8", "a shorter input window is enough", "input window 15 min", "p2_window", "GRU-pn-w15", *G),
    ("P2-8", "a shorter input window is enough", "input window 30 min", "p2_window", "FCN-pn-w30", "p2_norm", "FCN-peaknorm"),
    ("P2-8", "a shorter input window is enough", "input window 15 min", "p2_window", "FCN-pn-w15", "p2_norm", "FCN-peaknorm"),
    ("I-1", "modern TS architectures beat GRU", "PatchTST (joint, untuned)", "p3_arch", "PatchTST", *G),
    ("I-1", "modern TS architectures beat GRU", "TSMixer (joint, untuned)", "p3_arch", "TSMixer", *G),
    ("I-1", "modern TS architectures beat GRU", "N-BEATS (joint, untuned)", "p3_arch", "N-BEATS", *G),
    ("I-1", "modern TS architectures beat GRU", "ModernTCN-lite (joint, untuned)", "p3_arch", "ModernTCN", *G),
    ("I-1", "paper Transformer vs GRU (both peak-normalised)", "Transformer instead of GRU", "p2_norm2", "Transformer-peaknorm", *G),
    ("I-2", "a residual on a band power-law prior reduces tail error", "target = log y - prior", "p3_resid", "GRU-pn-resid", *G),
    ("I-2", "a residual on a band power-law prior reduces tail error", "target = log y - prior", "p3_resid", "FCN-z-resid", *F),
    ("I-2", "the real-time band prior alone beats the authors' power law", "A_b = x_peak 10^c_b, alpha_b band-wise", "p3_resid", "Prior-only", "repro_cv96", "Power-law"),
    ("I-3", "up-weighting the top-decile events helps", "loss weight x3 for top-10 % peak flux", "p3_extreme", "GRU-pn-wtop3", *G),
    ("I-3", "input-noise augmentation regularises small data", "0.01 dex Gaussian input noise", "p3_extreme", "GRU-pn-noise", *G),
    ("I-3", "input-noise augmentation regularises small data", "0.01 dex Gaussian input noise", "p3_extreme", "FCN-z-noise", *F),
    ("I-4", "type-specific experts (impulsive/gradual) help", "2 GRU experts split at median rise time", "p3_type", "GRU-pn-experts", *G),
    ("I-4", "type information helps", "rise-time feature", "p3_type", "GRU-pn-typefeat", *G),
    ("I-6", "band-specific models beat joint models", "GRU: 4 band-specific models", "p3_joint", "GRU-pn-perband", *G),
    ("I-6", "joint models beat band-specific models", "FCN: one joint 4-band model (w0 hyper-params)", "p3_joint", "FCN-z-joint", *F),
    ("I-7", "extra flare descriptors help", "rise time/slope, 60-min rise, longitude, X flag", "p3_features", "GRU-pn-feat", *G),
    ("I-7", "extra flare descriptors help", "rise time/slope, 60-min rise, longitude, X flag", "p3_features", "FCN-z-feat", *F),
    ("I-5", "a 5-seed ensemble mean is more accurate than single seeds", "5-seed ensemble mean (log space)", "p3_ensemble", "GRU-peaknorm-ens5", *G),
    ("I-5", "a 5-seed ensemble mean is more accurate than single seeds", "5-seed ensemble mean (log space)", "p3_ensemble", "FCN-z-ens5", *F),
    ("COMB", "adopted changes combine", "peaknorm + 45-min window + input noise + rise-time feature", "p3_final", "GRU-final", *G),
    ("COMB", "adopted changes combine (ensembles)", "5-seed ensemble of the combination", "p3_final", "GRU-final-ens5", "p3_ensemble", "GRU-peaknorm-ens5"),
    ("COMB", "GRU line vs FCN line (ensembles)", "GRU-final ensemble instead of FCN-z ensemble", "p3_final", "GRU-final-ens5", "p3_ensemble", "FCN-z-ens5"),
    ("COMB", "final vs paper configuration", "all adopted changes", "p3_final", "GRU-final-ens5", "repro_cv96", "GRU"),
    ("I-10", "nested Optuna re-tuning of the best pipeline helps", "per-fold, per-band TPE search, 50 trials (FCN-z)", "p3_optuna", "FCN-z-optuna", *F),
    ("COMB", "final (FCN-z ens5) vs paper-configured GRU", "z-score + band FCN + 5-seed ensemble", "p3_ensemble", "FCN-z-ens5", "repro_cv96", "GRU"),
    ("COMB", "final (FCN-z ens5) vs paper-configured FCN", "z-score + 5-seed ensemble", "p3_ensemble", "FCN-z-ens5", "repro_cv96", "FCN"),
]

if __name__ == "__main__":
    rows = []
    for p in PLAN:
        try:
            rows.append(compare(*p))
        except (KeyError, FileNotFoundError) as e:
            print("skip", p[3], p[4], e)
    L = pd.DataFrame(rows)
    out = ROOT / "results/improve"
    out.mkdir(parents=True, exist_ok=True)
    L.to_csv(out / "ledger.csv", index=False)
    pd.set_option("display.width", 250)
    print(L[["idea", "variant", "MAPE4_base", "MAPE4_var", "dMAPE4", "p_MAPE4", "dCC4", "p_CC4",
             "pooledCC4_base", "pooledCC4_var", "logRMSE_base", "logRMSE_var", "decision"]].round(4).to_string(index=False))
