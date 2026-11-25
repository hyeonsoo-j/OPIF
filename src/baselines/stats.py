"""Statistical baselines: power-law decay (authors' and band variants), persistence,
climatology, linear regression and random forest (Kim et al. 2026, Sect. 3.3 and the
Zenodo train_baseline_log10.py script).

Conventions: X_raw (N, 60) physical input flux; Y_raw (N, 180, 4) physical targets.
All models expose fit(X_raw, Y_raw) and predict(X_raw) -> (N, 180, 4) physical units.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

N_OUT = 180
N_BANDS = 4


def _power_law(t, alpha):
    return (t + 1.0) ** (-alpha)


def fit_alpha_curvefit(y: np.ndarray, t: np.ndarray | None = None) -> float:
    """Authors' per-event fit: y_norm = y / y[0], curve_fit((t+1)^-a), t = 1..180, bounds (0.01, 5)."""
    if t is None:
        t = np.arange(1, len(y) + 1, dtype=float)
    y_norm = np.clip(y / y[0], 1e-10, None)
    popt, _ = curve_fit(_power_law, t, y_norm, p0=[0.5], bounds=(0.01, 5.0), maxfev=1000)
    return float(popt[0])


def fit_alpha_loglog(y: np.ndarray, t: np.ndarray | None = None) -> float:
    """Closed-form least squares in log-log space (Phase-2 item 2):
    log(y/y0) = -alpha * log(t+1)  =>  alpha = -sum(u v)/sum(u^2), u = log(t+1), v = log(y/y0)."""
    if t is None:
        t = np.arange(1, len(y) + 1, dtype=float)
    u = np.log(t + 1.0)
    v = np.log(np.clip(y / y[0], 1e-10, None))
    return float(-np.sum(u * v) / np.sum(u * u))


class PowerLawAuthors:
    """f_b(t) = A (t+1)^-alpha_b with A = last *input* (X-ray) value for ALL bands (authors' script)."""

    name = "Power-law"

    def __init__(self, alphas: np.ndarray | None = None):
        self.alphas = np.ones(N_BANDS) * 0.5 if alphas is None else np.asarray(alphas, float)
        self.alpha_samples: list[list[float]] = [[] for _ in range(N_BANDS)]

    def fit(self, X_raw, Y_raw, method: str = "curve_fit"):
        t = np.arange(1, N_OUT + 1, dtype=float)
        self.alpha_samples = [[] for _ in range(N_BANDS)]
        for i in range(len(X_raw)):
            if X_raw[i, -1] <= 0:
                continue
            for b in range(N_BANDS):
                y = Y_raw[i, :, b]
                if y[0] <= 0:
                    continue
                try:
                    a = fit_alpha_curvefit(y, t) if method == "curve_fit" else fit_alpha_loglog(y, t)
                    self.alpha_samples[b].append(a)
                except Exception:
                    pass
        for b in range(N_BANDS):
            if self.alpha_samples[b]:
                self.alphas[b] = float(np.median(self.alpha_samples[b]))
        return self

    def predict(self, X_raw):
        N = X_raw.shape[0]
        t = np.arange(1, N_OUT + 1, dtype=float)
        A = np.maximum(X_raw[:, -1], 1e-10)[:, None]
        preds = np.zeros((N, N_OUT, N_BANDS))
        for b in range(N_BANDS):
            preds[:, :, b] = A * (t + 1.0) ** (-self.alphas[b])
        return preds


class PowerLawBand(PowerLawAuthors):
    """Oracle variant: A = true band value at the peak (y[0]) — not available in real time for EUV."""

    name = "Power-law(band-oracle)"

    def predict_with_peak(self, Y_peak):
        N = Y_peak.shape[0]
        t = np.arange(1, N_OUT + 1, dtype=float)
        preds = np.zeros((N, N_OUT, N_BANDS))
        for b in range(N_BANDS):
            preds[:, :, b] = np.maximum(Y_peak[:, b], 1e-10)[:, None] * (t + 1.0) ** (-self.alphas[b])
        return preds


class PersistenceAuthors:
    """Authors' 'Persistence': linear extrapolation of the last 10 input minutes (raw flux)."""

    name = "Persistence(lin-extrap)"

    def fit(self, X_raw, Y_raw):
        return self

    def predict(self, X_raw):
        N = X_raw.shape[0]
        preds = np.zeros((N, N_OUT, N_BANDS))
        for i in range(N):
            x = X_raw[i]
            w = 10
            t_in = np.arange(60 - w, 60, dtype=float)
            p = np.polyfit(t_in, x[-w:], 1)
            t_out = np.arange(60, 60 + N_OUT, dtype=float)
            x_ext = np.clip(p[0] * t_out + p[1], 1e-12, None)
            last = max(x[-1], 1e-12)
            ratio = x_ext / last
            for b in range(N_BANDS):
                preds[i, :, b] = x_ext if b == 0 else last * ratio
        return preds


class PersistencePeak:
    """Hold the peak value: w0 = last input value; EUV bands need the band peak (oracle)
    or, in the real-time version, use the train-set mean ratio band_peak/xray_peak."""

    name = "Persistence(peak-hold)"

    def fit(self, X_raw, Y_raw):
        xp = np.maximum(X_raw[:, -1], 1e-12)
        self.log_ratio = np.array([np.median(np.log10(np.maximum(Y_raw[:, 0, b], 1e-30)) - np.log10(xp)) for b in range(N_BANDS)])
        return self

    def predict(self, X_raw):
        N = X_raw.shape[0]
        xp = np.maximum(X_raw[:, -1], 1e-12)
        preds = np.zeros((N, N_OUT, N_BANDS))
        for b in range(N_BANDS):
            preds[:, :, b] = (xp * 10 ** self.log_ratio[b])[:, None]
        return preds


class Climatology:
    """Mean (log-space) decay curve of the training set, scaled to the input peak:
    log y_b(t) = log x_peak + mean_i[ log y_ib(t) - log x_peak_i ]."""

    name = "Climatology(mean-decay)"

    def fit(self, X_raw, Y_raw):
        lx = np.log10(np.maximum(X_raw[:, -1], 1e-30))[:, None, None]
        ly = np.log10(np.maximum(Y_raw, 1e-30))
        self.curve = np.mean(ly - lx, axis=0)  # (180, 4)
        return self

    def predict(self, X_raw):
        lx = np.log10(np.maximum(X_raw[:, -1], 1e-30))[:, None, None]
        return 10 ** (lx + self.curve[None])


class LinRegLog10:
    name = "LinearReg"

    def fit(self, X_raw, Y_raw):
        self.m = LinearRegression().fit(np.log10(X_raw), np.log10(Y_raw).reshape(len(Y_raw), -1))
        return self

    def predict(self, X_raw):
        return 10 ** self.m.predict(np.log10(X_raw)).reshape(len(X_raw), N_OUT, N_BANDS)


class RFLog10:
    name = "RandomForest"

    def __init__(self, n_estimators=200, max_depth=10, random_state=42, n_jobs=-1):
        self.kw = dict(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=n_jobs)

    def fit(self, X_raw, Y_raw):
        self.m = RandomForestRegressor(**self.kw).fit(np.log10(X_raw), np.log10(Y_raw).reshape(len(Y_raw), -1))
        return self

    def predict(self, X_raw):
        return 10 ** self.m.predict(np.log10(X_raw)).reshape(len(X_raw), N_OUT, N_BANDS)


class PowerLawRT:
    """Real-time band-wise power law (used as the I-2 prior):
    log10 y_b(t) = log10 x_peak + c_b - alpha_b log10(t+1), t = 0..179 (exact at the peak for w0);
    c_b = median training offset of the band peak from the X-ray peak (c_0 = 0), alpha_b = median of
    per-event curve_fit exponents on y/y_peak."""

    name = "Power-law(RT band)"

    def fit(self, X_raw, Y_raw):
        t = np.arange(0, N_OUT, dtype=float)
        xl = np.log10(X_raw[:, -1])
        self.c = np.median(np.log10(Y_raw[:, 0, :]) - xl[:, None], axis=0)
        self.c[0] = 0.0
        self.alphas = np.zeros(N_BANDS)
        for b in range(N_BANDS):
            a = []
            for i in range(len(Y_raw)):
                try:
                    a.append(fit_alpha_curvefit(Y_raw[i, :, b], t))
                except Exception:
                    pass
            self.alphas[b] = float(np.median(a))
        return self

    def predict(self, X_raw):
        t = np.arange(0, N_OUT, dtype=float)
        xl = np.log10(np.maximum(X_raw[:, -1], 1e-30))
        L = xl[:, None, None] + self.c[None, None, :] - self.alphas[None, None, :] * np.log10(t + 1.0)[None, :, None]
        return 10 ** L


BASELINES = {
    "powerlaw": PowerLawAuthors,
    "persistence_linextrap": PersistenceAuthors,
    "persistence_peak": PersistencePeak,
    "climatology": Climatology,
    "linreg": LinRegLog10,
    "rf": RFLog10,
    "powerlaw_rt": PowerLawRT,
}
