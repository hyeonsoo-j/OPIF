"""Metric implementation vs closed-form hand computation on synthetic power-law series
(Phase 2, item 1) and the closed-form log-log power-law fit (item 2)."""
import numpy as np
import pytest

from src.baselines.stats import fit_alpha_curvefit, fit_alpha_loglog
from src.eval.metrics import EPS, pearson, per_event_metrics, pooled_metrics, skill_score

rng = np.random.default_rng(0)
T = 180
t = np.arange(1, T + 1, dtype=float)


def synth(n=20, noise=0.05):
    a = 10 ** rng.uniform(-5, -4, size=n)
    al = rng.uniform(0.1, 1.0, size=n)
    y = a[:, None] * (t[None] + 1) ** -al[:, None]
    yhat = y * (1 + noise * rng.standard_normal(y.shape))
    return y, yhat


def test_per_event_against_formulas():
    y, yh = synth()
    pe = per_event_metrics(y, yh)
    for i in range(len(y)):
        e = yh[i] - y[i]
        assert np.isclose(pe["RMSE"][i, 0], np.sqrt((e ** 2).sum() / T))
        assert np.isclose(pe["MRE"][i, 0], 100 * (e / y[i]).sum() / T)
        assert np.isclose(pe["MAPE"][i, 0], 100 * np.abs(e / y[i]).sum() / T)
        assert np.isclose(pe["APE"][i, 0], 100 * abs(e[0]) / y[i, 0])
        num = ((y[i] - y[i].mean()) * (yh[i] - yh[i].mean())).sum()
        den = np.sqrt(((y[i] - y[i].mean()) ** 2).sum() * ((yh[i] - yh[i].mean()) ** 2).sum())
        assert np.isclose(pe["CC"][i, 0], num / den)
        assert np.isclose(pe["CC"][i, 0], np.corrcoef(y[i], yh[i])[0, 1])


def test_pooled_definitions():
    y, yh = synth()
    pm = pooled_metrics(y, yh)
    assert np.isclose(pm["CC"][0], np.corrcoef(y.ravel(), yh.ravel())[0, 1])
    assert np.isclose(pm["RMSE"][0], np.sqrt(np.mean((yh - y) ** 2)))
    pe = per_event_metrics(y, yh)
    assert np.isclose(pm["MAPE"][0], pe["MAPE"][:, 0].mean())


def test_perfect_prediction():
    y, _ = synth()
    pe = per_event_metrics(y, y.copy())
    assert np.allclose(pe["RMSE"], 0) and np.allclose(pe["MAPE"], 0) and np.allclose(pe["CC"], 1)


def test_relative_error_epsilon_rule():
    y = np.ones((1, T)) * 1e-5
    y[0, 10] = 0.0                                   # would diverge
    yh = np.ones((1, T)) * 1.1e-5
    pe = per_event_metrics(y, yh)
    assert np.isfinite(pe["MRE"][0, 0]) and np.isclose(pe["MRE"][0, 0], 10.0)
    assert EPS == 1e-30


def test_constant_series_cc_nan():
    assert np.isnan(pearson(np.ones(10), np.arange(10)))


def test_skill():
    assert np.isclose(skill_score(1.0, 4.0), 0.75)


@pytest.mark.parametrize("alpha", [0.01, 0.073, 0.4, 1.5])
def test_powerlaw_closed_form(alpha):
    """log y/A = -alpha log(t+1): closed-form LS == scipy nonlinear LS on the same (log) objective,
    and both recover alpha exactly on noise-free data; curve_fit (linear objective) too."""
    from scipy.optimize import least_squares
    A = 3e-5
    y = A * (t + 1) ** -alpha
    u, v = np.log(t + 1), np.log(y / A)
    a_closed = -np.sum(u * v) / np.sum(u * u)
    a_scipy = least_squares(lambda a: v + a[0] * u, x0=[0.5]).x[0]
    assert abs(a_closed - alpha) / alpha < 1e-10
    assert abs(a_scipy - a_closed) / a_closed < 1e-6
    # src implementation normalises by y[0] and evaluates at t' = t-1 so that y/y0 = (t'+1)^-alpha
    tt = np.arange(0, T, dtype=float)
    yy = A * (tt + 1) ** -alpha
    assert abs(fit_alpha_loglog(yy, tt) - alpha) / alpha < 1e-10
    assert abs(fit_alpha_curvefit(yy, tt) - alpha) / alpha < 1e-3


def test_authors_powerlaw_first_step_offset():
    """Authors' convention (t = 1..180, y normalised by its first value) puts the first forecast at
    A * 2^-alpha: APE(peak) = 1 - 2^-alpha (24.24 % for alpha = 0.4005, as in the Zenodo csv)."""
    from src.baselines.stats import PowerLawAuthors
    X = np.full((1, 60), 1e-5)
    m = PowerLawAuthors(alphas=np.array([0.4005, 0.02, 0.07, 0.01]))
    p = m.predict(X)
    assert np.isclose(100 * (1 - p[0, 0, 0] / 1e-5), 100 * (1 - 2 ** -0.4005))
    assert np.isclose(100 * (1 - 2 ** -0.4005), 24.24, atol=0.01)
