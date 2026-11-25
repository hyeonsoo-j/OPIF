"""Paired significance tests on per-event metrics (Phase 2, item 4).

For a reference model R and a competitor C evaluated on the same N events:
  * paired bootstrap (B resamples of events) of mean(C) - mean(R): 95 % percentile CI
  * Wilcoxon signed-rank test (two-sided, zero differences dropped: 'wilcox')
  * Cliff's delta between the two per-event samples (unpaired dominance statistic,
    delta = P(C > R) - P(C < R)), and the paired variant P(C_i > R_i) - P(C_i < R_i)
  * Holm-Bonferroni adjustment within each (metric, band) family
"Better" direction: higher CC, lower RMSE/|MRE|/MAPE/APE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HIGHER_BETTER = {"CC": True, "RMSE": False, "MRE": False, "MAPE": False, "APE": False, "absMRE": False}


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float(gt - lt) / (len(a) * len(b))


def paired_bootstrap_ci(d: np.ndarray, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    p_boot = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(lo), float(hi), float(min(1.0, p_boot))


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def compare_to_reference(E: pd.DataFrame, ref: str, metrics=("CC", "RMSE", "absMRE", "MAPE", "APE"),
                         n_boot: int = 10000, key: str = "event") -> pd.DataFrame:
    """E: long table with columns model, band, event, <metrics> (one row per model x band x event;
    average seeds beforehand). Returns one row per (metric, band, competitor)."""
    E = E.copy()
    if "absMRE" in metrics and "absMRE" not in E:
        E["absMRE"] = E["MRE"].abs()
    rows = []
    for met in metrics:
        for band, g in E.groupby("band"):
            piv = g.pivot_table(index=key, columns="model", values=met)
            if ref not in piv:
                continue
            for c in piv.columns:
                if c == ref:
                    continue
                ok = piv[[ref, c]].dropna()
                r, x = ok[ref].values, ok[c].values
                d = x - r
                lo, hi, pb = paired_bootstrap_ci(d, n_boot=n_boot)
                try:
                    pw = wilcoxon(x, r, zero_method="wilcox").pvalue if np.any(d != 0) else 1.0
                except ValueError:
                    pw = 1.0
                sgn = 1 if HIGHER_BETTER[met] else -1
                rows.append(dict(metric=met, band=band, reference=ref, competitor=c, n=len(d),
                                 mean_ref=r.mean(), mean_comp=x.mean(), mean_diff=d.mean(),
                                 ci_lo=lo, ci_hi=hi, p_boot=pb, p_wilcoxon=pw,
                                 cliffs_delta=cliffs_delta(x, r),
                                 paired_dominance=float(np.mean(np.sign(d))),
                                 ref_better_frac=float(np.mean(sgn * d < 0)),
                                 ref_better=bool(sgn * d.mean() < 0)))
    R = pd.DataFrame(rows)
    if len(R):
        R["p_wilcoxon_holm"] = R.groupby(["metric", "band"])["p_wilcoxon"].transform(lambda p: holm(p.values))
        R["significant_0.05"] = R["p_wilcoxon_holm"] < 0.05
    return R


def band_mean_table(E: pd.DataFrame, metrics=("CC", "RMSE", "absMRE", "MAPE", "APE")) -> pd.DataFrame:
    """Per-event band-mean of each metric (RMSE is normalised per band by the band's median RMSE
    over all models, so that the four bands contribute comparably)."""
    E = E.copy()
    if "absMRE" not in E:
        E["absMRE"] = E["MRE"].abs()
    med = E.groupby("band")["RMSE"].transform("median")
    E["RMSE"] = E["RMSE"] / med
    G = E.groupby(["model", "event"])[list(metrics)].mean().reset_index()
    G["band"] = "mean4"
    return G
