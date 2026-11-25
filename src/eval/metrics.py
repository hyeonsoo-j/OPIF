"""Evaluation metrics replicating Kim et al. (2026) / Zenodo README_metrics.md.

All metrics operate on *physical* irradiance (W m^-2), shapes (N_events, N_t[, N_bands]).

Definitions (per band):
  CC_pooled, RMSE_pooled : over all N_events x N_t pairs (paper Table / summary csv)
  CC_i, RMSE_i           : per event (Zenodo event csv)
  MRE_i  = 100/N_t * sum_t (yhat - y)/y          (signed)
  MAPE_i = 100/N_t * sum_t |yhat - y|/y
  APE_i  = 100 * |yhat_1 - y_1| / y_1            (first forecast step = flare peak)
  Event-averaged MRE/MAPE/APE = mean_i.
  Skill_E = 1 - E_model / E_reference  (RMSE, MAPE, APE).

Epsilon policy for relative errors: FISM2 irradiance is strictly positive, so no epsilon is
needed; for safety, denominators |y| < EPS are excluded from the mean (EPS = 1e-30), and
the number of excluded samples is reported by `relative_error_mask`.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-30


def _as3d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError(f"expected (N,T) or (N,T,B), got {a.shape}")
    return a


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0 or x.size < 2:
        return np.nan
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def per_event_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    """Per-event metrics. Returns dict of arrays with shape (N, B)."""
    yt, yp = _as3d(y_true), _as3d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch {yt.shape} vs {yp.shape}")
    N, T, B = yt.shape
    cc = np.full((N, B), np.nan)
    rmse = np.zeros((N, B))
    mre = np.zeros((N, B))
    mape = np.zeros((N, B))
    ape = np.zeros((N, B))
    for b in range(B):
        for i in range(N):
            t, p = yt[i, :, b], yp[i, :, b]
            cc[i, b] = pearson(t, p)
            rmse[i, b] = np.sqrt(np.mean((p - t) ** 2))
            ok = np.abs(t) > EPS
            rel = (p[ok] - t[ok]) / t[ok]
            mre[i, b] = 100.0 * rel.mean() if ok.any() else np.nan
            mape[i, b] = 100.0 * np.abs(rel).mean() if ok.any() else np.nan
            ape[i, b] = 100.0 * abs(p[0] - t[0]) / abs(t[0]) if abs(t[0]) > EPS else np.nan
    return {"CC": cc, "RMSE": rmse, "MRE": mre, "MAPE": mape, "APE": ape}


def pooled_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    """Band-level summary exactly as the authors' summary csv: pooled CC/RMSE, event-mean
    MRE/MAPE/APE, median/mean of per-event RMSE. Returns dict of arrays (B,)."""
    yt, yp = _as3d(y_true), _as3d(y_pred)
    N, T, B = yt.shape
    pe = per_event_metrics(yt, yp)
    out = {
        "CC": np.array([pearson(yt[:, :, b], yp[:, :, b]) for b in range(B)]),
        "RMSE": np.array([np.sqrt(np.mean((yp[:, :, b] - yt[:, :, b]) ** 2)) for b in range(B)]),
        "MRE": np.nanmean(pe["MRE"], axis=0),
        "MAPE": np.nanmean(pe["MAPE"], axis=0),
        "APE": np.nanmean(pe["APE"], axis=0),
        "Median_event_RMSE": np.median(pe["RMSE"], axis=0),
        "Mean_event_RMSE": pe["RMSE"].mean(axis=0),
        "Mean_event_CC": np.nanmean(pe["CC"], axis=0),
    }
    return out


def skill_score(e_model: np.ndarray, e_ref: np.ndarray) -> np.ndarray:
    e_model, e_ref = np.asarray(e_model, float), np.asarray(e_ref, float)
    return 1.0 - e_model / e_ref


def relative_error_mask(y_true: np.ndarray) -> int:
    """Number of samples excluded from relative-error means because |y| < EPS."""
    return int((np.abs(np.asarray(y_true)) <= EPS).sum())


def pooled_mse_skill(y_true, y_model, y_ref) -> np.ndarray:
    """SS = 1 - MSE_model / MSE_ref per band (Phase-2 item 3)."""
    yt, ym, yr = _as3d(y_true), _as3d(y_model), _as3d(y_ref)
    mse_m = np.mean((ym - yt) ** 2, axis=(0, 1))
    mse_r = np.mean((yr - yt) ** 2, axis=(0, 1))
    return 1.0 - mse_m / mse_r
