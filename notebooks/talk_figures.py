"""Figures for the lab-seminar deck (v4: colour concept v1 + Times New Roman, Solar Flare AI_v5.pptx): the authors' benchmark drawn from their own arrays and the
operational re-design, at slide size (fonts are in points at the size the picture has on the slide).

    python notebooks/talk_figures.py --results results_v6 --out figures/talk_v2

Pictures (PNG, 300 dpi) and talk_numbers.json (every number that the slides quote, computed here from the arrays / tables):
  talk_fig1_task.png        one flare of the authors' test set: 60-min input, FISM2 in 4 bands, the authors' GRU forecast
  talk_lim1..4_*.png        four small panels: input source, peak declaration, spread of the error, isolation rule
  talk_fig5_events.png      the 96 test flares one by one: authors' GRU vs our GRU-z ensemble (4-band MAPE)
  talk_fig7_reissue.png     forecast issued at the peak and re-issued 10 min later (GOES-driven), one flare + change of the error
  talk_fig8_example.png     a flare outside the isolation rule: forecast, 90 % interval, declaration of the next flare
  talk_fig9_coverage.png    interval coverage over all 180 min and over the minutes in force; coverage around the next declaration
The bar charts of the deck (authors' models, steps of the improvement, inputs) are native PowerPoint charts filled from
talk_numbers.json by the deck builder.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BAND = ["0.1–0.8 nm", "9.0–9.9 nm", "13.0–13.9 nm", "30.0–30.9 nm"]
# Colour concept v1 (docs/figure_style.md, sf_colors.py): grey/black = data and reference, including the previous
# paper (Kim et al.: grey, dashed or hollow); green = WHAT we forecast; orange = WHEN we issue and stop (detector, re-issue, validity).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sf_colors import SF, use_times  # noqa: E402
C = dict(prior=SF["prior"], ours=SF["ours"], goes=SF["input"], truth=SF["truth"], grey=SF["axis"], light="#e2e2e2", soft="#f0f0f0",
         text="#0b0b0b", text2="#505050", wait=SF["ours_mid"], bar="#c8c8c8", rule=SF["rule"], rule_light=SF["rule_light"])
# Times New Roman only (use_times below stops if it is missing)
plt.rcParams.update({"font.family": ["Times New Roman"], "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
                     "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": C["light"], "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "lines.linewidth": 2.0, "savefig.dpi": 300, "figure.dpi": 100,
                     "legend.frameon": False, "axes.edgecolor": "#b8b8b8", "axes.linewidth": 0.8, "xtick.color": C["text2"],
                     "ytick.color": C["text2"], "axes.labelcolor": C["text2"], "text.color": C["text"],
                     "xtick.major.size": 3, "ytick.major.size": 3})
use_times()                                    # Times New Roman for text and math; stops if it is not installed
NUM: dict = {}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"notebooks/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Data:
    """Arrays shared by the figures. Predictions are log10; truth is physical (W m^-2)."""

    def __init__(self, results: Path, events: Path, goes_post: Path):
        self.R = Path(results)
        os.environ.update(FISM_PROTOCOL="full", FISM_EVENTS=str(events.resolve()), FISM_GOES_POST=str(goes_post.resolve()),
                          FISM_DEVICE_QUIET="1", FISM_EXCLUDE_NEAR_TEST_H="0", FISM_EXCLUDE_RANDOM_N="0", FISM_EXCLUDE_SAME_AR="0")
        from src.data import pool
        from src.data.cv96 import load_meta
        self.pool = pool
        self.ap = _load("analyze_p7_p8")
        z = np.load(ROOT / "data/processed/zenodo_full.npz", allow_pickle=False)
        self.z = z
        self.names = [str(e) for e in z["event_names"]]
        self.Xa, self.Ya = z["X_real"].astype(float), z["Y_true"].astype(float)          # authors' arrays (96 test flares)
        self.meta = load_meta()
        self.d = np.load(events, allow_pickle=False)                                       # rebuilt data set (1,606 flares)
        self.st = np.load(self.R / "full_server/peak_detector_streams.npz", allow_pickle=False)
        self._pools = {}

    def get_pool(self, inp="fism", k=0, extra=0):
        key = ("default" if inp == "fism" else "goes_b", 0 if inp == "fism" else k, extra)
        if key not in self._pools:
            os.environ.update(FISM_INPUT=key[0], FISM_ISSUE_DELAY=str(key[1]), FISM_PREDICT_EXTRA=str(key[2]))
            self.pool.load_full.cache_clear()
            X, Y, names, meta, tr, va, te, teL = self.pool.load_full()
            self._pools[key] = dict(X=X, Y=Y, names=names, meta=meta, tr=tr, va=va, teL=teL, ex=self.pool.extra_index())
        return self._pools[key]

    def preds(self, exp, m, pre):
        fs = sorted(glob.glob(f"{self.R}/{exp}/{pre}_{m}_s*.npy"))
        assert fs, f"no predictions {exp}/{pre}_{m}"
        return np.stack([np.load(f, allow_pickle=False).astype(np.float64) for f in fs])

    def val(self, exp, m, inp, k):
        fs = sorted(glob.glob(f"{self.R}/{exp}/val_{m}_s*.npz"))
        ds = [np.load(f, allow_pickle=False) for f in fs]
        idx = ds[0]["idx"]
        assert all(np.array_equal(q["idx"], idx) for q in ds)
        P = self.get_pool(inp, k)
        assert np.array_equal(np.sort(idx), np.sort(P["va"]))
        return np.stack([q["pred_log"].astype(np.float64) for q in ds]), np.log10(P["Y"][idx])


def ev_mape(yt, yp, bands=(0, 1, 2, 3), k0=0):
    """Event MAPE (percent) over the bands and the leads >= k0. yt, yp physical (n, 180, 4)."""
    b = list(bands)
    return 100 * (np.abs(yp[:, k0:][..., b] - yt[:, k0:][..., b]) / yt[:, k0:][..., b]).mean(axis=(1, 2))


def save(fig, out, name):
    fig.savefig(out / name, facecolor="white")
    plt.close(fig)
    print(f"[talk] {name}")


# --------------------------------------------------------------------------------------------------------------
def pick_example(D):
    """Flare of the authors' test set used on the 'previous work' slides: the one with the median change of the 4-band event
    MAPE between the authors' GRU and our GRU-z ensemble (so it is neither a best nor a worst case)."""
    G = D.preds("p2_zscore_full_gpu5", "GRU-z", "oof").mean(0)
    pa, po = ev_mape(D.Ya, D.z["Y_GRU"].astype(float)), ev_mape(D.Ya, 10 ** G)
    k = int(np.argsort(po - pa)[len(pa) // 2])
    NUM["example_event"] = dict(k=k, name=D.names[k], MAPE4_authors_GRU=float(pa[k]), MAPE4_ours_GRUz_ens=float(po[k]))
    return k, pa, po


def _logticks(ax, lo, hi):
    """Plain-number ticks on a log axis whose data are already scaled to a power of ten."""
    from matplotlib.ticker import NullFormatter
    cand = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    tk = [v for v in cand if lo <= v <= hi]
    ax.set_yticks(tk); ax.set_yticklabels([f"{v:g}" for v in tk])
    ax.yaxis.set_minor_formatter(NullFormatter())


def fig1_task(D, out, k):
    """One flare from the authors' arrays: input window, FISM2 in the four bands, the authors' GRU forecast."""
    X, Y, P = D.Xa[k], D.Ya[k], D.z["Y_GRU"][k].astype(float)
    fig = plt.figure(figsize=(8.35, 5.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.52, wspace=0.34, left=0.085, right=0.985, top=0.925, bottom=0.105)
    ax = fig.add_subplot(gs[0, :])
    ti, t = np.arange(-59, 1), np.arange(180)
    e = int(np.floor(np.log10(Y[:, 0].max())))
    s = 10.0 ** -e
    ax.axvspan(-60, 0, color=C["soft"], lw=0, zorder=0)
    ax.plot(ti, X * s, color=C["goes"], lw=2.2, label="input")
    ax.plot(t, Y[:, 0] * s, color=C["truth"], lw=2.2, label="observed (FISM2)")
    ax.plot(t, P[:, 0] * s, color=C["prior"], lw=2.2, ls=(0, (4, 2)), label="forecast (authors' GRU)")
    ax.set_yscale("log")
    lo, hi = min(X.min(), Y[:, 0].min(), P[:, 0].min()) * s, max(X.max(), Y[:, 0].max(), P[:, 0].max()) * s
    ax.set_ylim(lo / 1.25, hi * 2.3)
    _logticks(ax, lo / 1.25, hi * 1.2)
    ax.set(xlim=(-60, 180), xlabel="minutes from the flare peak", ylabel=f"irradiance (10$^{{{e}}}$ W m$^{{-2}}$)")
    ax.set_xticks(range(-60, 181, 30))
    ax.axvline(0, color=C["truth"], lw=0.9)
    ax.text(-30, hi * 1.62, "INPUT  60 min", ha="center", va="center", fontsize=12, fontweight="bold", color=C["text2"])
    ax.text(90, hi * 1.62, "FORECAST  180 min × 4 bands", ha="center", va="center", fontsize=12, fontweight="bold", color=C["text2"])
    ax.set_title(BAND[0] + "  (soft X-ray)", loc="left", fontweight="bold")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.99), ncol=3, columnspacing=1.5, handlelength=1.8, borderaxespad=0.0)
    for b in (1, 2, 3):
        a2 = fig.add_subplot(gs[1, b - 1])
        eb = int(np.floor(np.log10(np.median(Y[:, b]))))
        a2.plot(t, Y[:, b] / 10 ** eb, color=C["truth"], lw=2.0)
        a2.plot(t, P[:, b] / 10 ** eb, color=C["prior"], lw=2.0, ls=(0, (4, 2)))
        a2.set(xlim=(0, 180), xlabel="minutes after the peak")
        a2.set_xticks(range(0, 181, 60))
        a2.set_title(BAND[b], loc="left", fontweight="bold")
        a2.text(1.0, 1.03, f"10$^{{{eb}}}$ W m$^{{-2}}$", transform=a2.transAxes, ha="right", va="bottom", fontsize=10, color=C["text2"])
        l2, h2 = min(Y[:, b].min(), P[:, b].min()) / 10 ** eb, max(Y[:, b].max(), P[:, b].max()) / 10 ** eb
        a2.set_ylim(l2 - 0.12 * (h2 - l2), h2 + 0.12 * (h2 - l2))
    NUM["fig1"] = dict(event=D.names[k], band_MAPE_authors_GRU=[float(v) for v in 100 * (np.abs(P - Y) / Y).mean(0)])
    save(fig, out, "talk_fig1_task.png")


def lim_panels(D, out, k, pa):
    """Four small panels for the 'what the benchmark leaves out' slide (their titles are native text on the slide)."""
    d, st = D.d, D.st
    rowL = {int(d["test_k"][i]): int(i) for i in np.where(d["is_test"])[0]}
    srow = {int(r): j for j, r in enumerate(st["row"])}
    rows = np.array([rowL[j] for j in range(96)])
    assert [str(x) for x in d["event_id"][rows]] == D.names
    size = (3.1, 1.93)
    kw = dict(left=0.2, right=0.96, top=0.93, bottom=0.27)
    # 1 input source: GOES XRS-B (what arrives in real time) against the authors' input (FISM2 0.1-0.8 nm), 95 test flares
    xg = d["X_goes_b"][rows].astype(float)
    ok = np.all(np.isfinite(xg) & (xg > 0), axis=1)
    ratio = xg[ok] / D.Xa[ok]
    q = np.percentile(ratio, [25, 50, 75], axis=0)
    fig, ax = plt.subplots(figsize=size); fig.subplots_adjust(**kw)
    ti = np.arange(-59, 1)
    ax.fill_between(ti, q[0], q[2], color=C["goes"], alpha=0.2, lw=0)
    ax.plot(ti, q[1], color=C["goes"], lw=2.0)
    ax.axhline(1, color=C["truth"], lw=1.0)
    ax.set_yscale("log"); ax.set_ylim(0.5, 1.6)
    _logticks(ax, 0.5, 1.6)
    ax.set_yticks([0.5, 0.7, 1.0, 1.4]); ax.set_yticklabels(["0.5", "0.7", "1", "1.4"])
    ax.set(xlim=(-60, 0), xlabel="minutes before the peak", ylabel="GOES / FISM2")
    ax.set_xticks([-60, -40, -20, 0])
    ax.text(-58, 1.06, "equal", fontsize=10, color=C["text2"], va="bottom")
    NUM["lim1"] = dict(n=int(ok.sum()), median_ratio_t_minus59=float(q[1, 0]), median_ratio_peak=float(q[1, -1]),
                       iqr_t_minus59=[float(q[0, 0]), float(q[2, 0])], iqr_peak=[float(q[0, -1]), float(q[2, -1])],
                       rms_log10=float(np.sqrt(np.mean(np.log10(ratio) ** 2))))
    save(fig, out, "talk_lim1_input.png")
    # 2 peak declaration: the GOES stream of the example flare; the detector declares a maximum after 2 falling minutes
    r = rowL[k]
    fig, ax = plt.subplots(figsize=size); fig.subplots_adjust(**kw)
    b = st["xrsb"][srow[r]].astype(float)
    pre = int(st["pre"])
    tt = np.arange(-pre, len(b) - pre)
    sel = (tt >= -12) & (tt <= 12)
    ax.plot(tt[sel], b[sel] * 1e5, color=C["goes"], lw=2.0, marker="o", ms=3.5)
    dec = [x for x in D.ap.pdet.detect_all(b, 2, D.ap.THR, rise_factor=2.0) if abs(x["t_max"] - pre) <= D.ap.TOL]
    assert dec, "the detector does not declare the peak of the example flare"
    t_dec, tm = int(dec[0]["t"] - pre), int(dec[0]["t_max"] - pre)
    ax.axvline(tm, color=C["truth"], lw=1.0)
    ax.axvline(t_dec, color=C["rule"], lw=1.4, ls=(0, (4, 2)))
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.3 * (y1 - y0))
    ax.text(tm - 0.5, y1 + 0.24 * (y1 - y0), "peak", ha="right", va="top", fontsize=10)
    ax.text(t_dec + 0.5, y1 + 0.24 * (y1 - y0), f"declared\n+{t_dec - tm} min", ha="left", va="top", fontsize=10, linespacing=1.0, color=C["rule"])
    ax.set(xlim=(-12, 12), xlabel="minutes from the peak", ylabel="GOES (10$^{-5}$ W m$^{-2}$)")
    NUM["lim2"] = dict(event=D.names[k], declared_min_after_peak=t_dec - tm)
    save(fig, out, "talk_lim2_peak.png")
    # 3 spread of the error of the authors' GRU over the 96 test flares
    fig, ax = plt.subplots(figsize=size); fig.subplots_adjust(**kw)
    assert pa.max() < 60
    ax.hist(pa, bins=np.arange(0, 65, 5), color=C["prior"], rwidth=0.86)
    ax.set(xlim=(0, 60), xlabel="error of one flare, 4-band MAPE (%)", ylabel="flares")
    ax.set_xticks(range(0, 61, 10))
    ax.grid(axis="x", visible=False)
    q10, q50, q90 = np.percentile(pa, [10, 50, 90])
    ax.text(0.97, 0.95, f"median {q50:.0f} %\n10–90 %: {q10:.0f}–{q90:.0f} %", transform=ax.transAxes, ha="right", va="top", fontsize=10, linespacing=1.15)
    NUM["lim3"] = dict(median=float(q50), p10=float(q10), p90=float(q90), n=int(len(pa)))
    save(fig, out, "talk_lim3_spread.png")
    # 4 share of the M/X flares outside the isolation rule
    n_all, n_out = 1604, 398
    assert abs(n_out / n_all - D.ap.P_OUTSIDE) < 1e-12
    fig, ax = plt.subplots(figsize=size); fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.27)
    ax.barh([0], [n_all - n_out], color=C["bar"], height=0.5)
    ax.barh([0], [n_out], left=[n_all - n_out + 12], color=C["truth"], height=0.5)
    ax.text((n_all - n_out) / 2, 0, f"isolated\n{n_all - n_out:,}", ha="center", va="center", fontsize=10.5, color=C["text"], linespacing=1.05)
    ax.text(n_all - n_out / 2 + 12, 0, f"not isolated\n{n_out}", ha="center", va="center", fontsize=10.5, color="white", linespacing=1.05)
    ax.text(n_all - n_out / 2 + 12, 0.36, f"{100 * n_out / n_all:.0f} % left out", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.set(xlim=(0, n_all + 12), ylim=(-0.45, 0.75), xlabel=f"M- and X-class flares 2003–2023 (n = {n_all:,})")
    ax.set_yticks([]); ax.set_xticks([]); ax.grid(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    NUM["lim4"] = dict(n_all=n_all, n_outside=n_out, share_outside=n_out / n_all)
    save(fig, out, "talk_lim4_isolation.png")


def fig5_events(D, out, k, pa, po):
    """The 96 flares of the authors' test set one by one: authors' GRU (their prediction array) vs our GRU-z ensemble."""
    better = po < pa
    fig, ax = plt.subplots(figsize=(5.2, 4.95))
    fig.subplots_adjust(left=0.15, right=0.975, top=0.975, bottom=0.135)
    lo, hi = 4.0, 60.0
    assert pa.min() > lo and po.min() > lo and pa.max() < hi and po.max() < hi
    ax.fill_between([lo, hi], [lo, hi], [lo, lo], color=C["ours"], alpha=0.07, lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color=C["grey"], lw=1.0, zorder=1)
    ax.scatter(pa[better], po[better], s=58, color=C["ours"], edgecolor="white", lw=1.3, zorder=3, label=f"ours lower: {int(better.sum())} flares")
    ax.scatter(pa[~better], po[~better], s=52, marker="s", color=C["prior"], edgecolor="white", lw=1.3, zorder=3, label=f"authors' lower: {int((~better).sum())} flares")
    ax.scatter([pa[k]], [po[k]], s=150, facecolor="none", edgecolor=C["truth"], lw=1.3, zorder=4)
    ax.annotate("flare of Fig. 1", (pa[k], po[k]), xytext=(pa[k] * 1.12, po[k] / 1.9), fontsize=10.5, color=C["text2"], ha="left",
                arrowprops=dict(arrowstyle="-", color=C["grey"], lw=0.8, shrinkA=0, shrinkB=7))
    ax.set_xscale("log"); ax.set_yscale("log")
    from matplotlib.ticker import NullFormatter
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks([5, 10, 20, 50]); axis.set_ticklabels(["5", "10", "20", "50"]); axis.set_minor_formatter(NullFormatter())
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="authors' GRU: error of one flare, 4-band MAPE (%)", ylabel="ours (GRU-z, 5-seed ensemble) (%)")
    ax.set_aspect("equal")
    ax.text(45, 49, "equal", rotation=45, fontsize=10, color=C["text2"], ha="center", va="bottom", rotation_mode="anchor")
    ax.legend(loc="upper left", handletextpad=0.2, borderaxespad=0.3)
    NUM["fig5"] = dict(n=int(len(pa)), n_ours_lower=int(better.sum()), n_authors_lower=int((~better).sum()), median_authors=float(np.median(pa)),
                       median_ours=float(np.median(po)), mean_authors=float(pa.mean()), mean_ours=float(po.mean()))
    save(fig, out, "talk_fig5_events.png")


OP, OP10, OP8 = "p2_zscore_full_in-goes_convnn_p2nn", "p2_zscore_full_in-goes_convnn_d10_p7", "p2_zscore_full_in-goes_convnn_p8"


def fig7_reissue(D, out):
    """GOES-driven forecast issued with the data up to the peak and re-issued 10 min later: (a) the test flare with the median
    change of the 0.1-0.8 nm error, (b) change of that error over the 95 test flares (paper1_p7_delay_hboot.csv)."""
    YL = D.get_pool("fism")["Y"][96:192]
    m = "GRU-z"
    Lb, Ld = D.preds(OP, m, "oofL").mean(0), D.preds(OP10, m, "oofL").mean(0)
    ok = np.isfinite(Lb).all((1, 2)) & np.isfinite(Ld).all((1, 2))
    a, b = np.full(96, np.nan), np.full(96, np.nan)
    a[ok], b[ok] = ev_mape(YL[ok], 10 ** Lb[ok], (0,), 10), ev_mape(YL[ok], 10 ** Ld[ok], (0,), 10)
    order = np.argsort(np.where(ok, b - a, np.inf))[: int(ok.sum())]
    k = int(order[len(order) // 2])
    fig = plt.figure(figsize=(7.85, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.3, left=0.085, right=0.985, top=0.92, bottom=0.125)
    ax = fig.add_subplot(gs[0])
    t = np.arange(180)
    y = YL[k, :, 0]
    e = int(np.floor(np.log10(y.max())))
    s = 10.0 ** -e
    ax.plot(t, y * s, color=C["truth"], lw=2.2, label="observed (FISM2)")
    ax.plot(t, 10 ** Lb[k, :, 0] * s, color=C["wait"], lw=2.0, ls=(0, (5, 2)), label="first forecast (data up to the peak)")
    ax.plot(t[10:], 10 ** Ld[k, 10:, 0] * s, color=C["ours"], lw=2.4, label="re-issued at +10 min")
    ax.axvline(10, color=C["rule"], lw=1.2, ls=(0, (3, 2)), zorder=1)
    ax.text(11.5, 0.03, "re-issue\n(+10 min)", transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=10, color=C["rule"], linespacing=1.0)
    ax.set_yscale("log")
    lo, hi = min(y.min(), 10 ** Lb[k, :, 0].min(), 10 ** Ld[k, 10:, 0].min()) * s, max(y.max(), 10 ** Lb[k, :, 0].max()) * s
    ax.set_ylim(lo / 1.3, hi * 1.25)
    _logticks(ax, lo / 1.3, hi * 1.25)
    ax.set(xlim=(-5, 180), xlabel="minutes from the flare peak", ylabel=f"0.1–0.8 nm irradiance (10$^{{{e}}}$ W m$^{{-2}}$)")
    ax.set_xticks(range(0, 181, 30))
    ax.legend(loc="upper right", handlelength=2.2)
    ax.set_title("a   one test flare", loc="left", fontweight="bold")
    NUM["fig7a"] = dict(event=D.names[k], k=k, rule="median change of the 0.1-0.8 nm MAPE (lead >= 10 min) among the 95 test flares, GRU-z ensemble",
                        w0_MAPE_first=float(a[k]), w0_MAPE_reissued=float(b[k]), ens_w0_first_mean=float(np.nanmean(a)), ens_w0_reissued_mean=float(np.nanmean(b)),
                        n_improved=int((b[ok] < a[ok]).sum()), n=int(ok.sum()))
    # (b) table values
    T = pd.read_csv(D.R / "full_server/paper1_p7_delay_hboot.csv")
    T = T[(T.family == "delayed_vs_peak") & (T.test == "rebuilt96") & (T.input == "goes_convnn")]
    ax = fig.add_subplot(gs[1])
    for mi, (mm, fc) in enumerate((("GRU-z", C["rule"]), ("FCN-z", "white"))):
        for ki, kk in enumerate((2, 5, 10)):
            r = T[(T.model == mm) & (T.k == kk)].iloc[0]
            x = ki + (mi - 0.5) * 0.3
            ax.plot([x, x], [r.lo_w0, r.hi_w0], color=C["rule"], lw=1.6, solid_capstyle="round", zorder=2)
            ax.scatter([x], [r.d_w0], s=64, marker="o", facecolor=fc, edgecolor=C["rule"], lw=1.6, zorder=3, label=mm if ki == 0 else None)
            if r.p_holm_w0 < 0.05:
                ax.text(x, r.lo_w0 - 0.5, "*", ha="center", va="top", fontsize=14)
            NUM[f"fig7b_{mm}_k{kk}"] = dict(d_w0=float(r.d_w0), lo=float(r.lo_w0), hi=float(r.hi_w0), p_holm=float(r.p_holm_w0), d4=float(r.d), lo4=float(r.lo),
                                           hi4=float(r.hi), p_holm4=float(r.p_holm), w0_delayed=float(r.MAPE_w0_delayed),
                                           w0_peak=float(r.MAPE_w0_peak_issued_same_leads), n=int(r.n_ev))
    ax.axhline(0, color=C["truth"], lw=1.0)
    ax.set_xticks(range(3)); ax.set_xticklabels(["+2", "+5", "+10 min"])
    ax.set(xlim=(-0.55, 2.55), ylim=(-13, 4.5), xlabel="issue time after the peak", ylabel="change of the 0.1–0.8 nm error (pt)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", handletextpad=0.2, borderaxespad=0.2)
    ax.text(0.98, 0.985, "95 % interval;  * adjusted p < 0.05", transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=C["text2"])
    ax.set_title("b   all 95 test flares", loc="left", fontweight="bold")
    save(fig, out, "talk_fig7_reissue.png")


EXAMPLE_OUTSIDE = "20110308_M1.3"      # illustrative flare with a later flare inside the 180 minutes (its own coverage is quoted on the slide)


def fig8_example(D, out):
    """A flare outside the isolation rule: operational forecast (GRU-z from GOES through the converter), its 90 % interval
    calibrated on the isolated validation flares, and the minute at which the detector declares the next flare."""
    d = D.d
    P = D.get_pool("goes_convnn", 0, 1)
    _, ex_rows = P["ex"]
    E = pd.read_csv(D.R / OP8 / "extra_rows.csv")
    assert np.array_equal(E.dataset_row.values, ex_rows)
    m = "GRU-z"
    Lx = D.preds(OP8, m, "oofN")
    Lv, yv = D.val(OP8, m, "goes_convnn", 0)
    mu, lo, hi = D.ap.conf_norm(Lv, yv, Lx)
    i = int(np.where(d["event_id"][ex_rows].astype(str) == EXAMPLE_OUTSIDE)[0][0])
    r = int(ex_rows[i])
    ev = pd.read_csv(D.R / "full_server/paper1_p8_reissue_events.csv").set_index("dataset_row")
    T = int(ev.T_next_r2[r])
    y, x_in = d["Y"][r, :, 0].astype(float), d["X"][r].astype(float)
    ins = (np.log10(y) >= lo[i, :, 0]) & (np.log10(y) <= hi[i, :, 0])
    e = int(np.floor(np.log10(y.max())))
    s = 10.0 ** -e
    t, ti = np.arange(180), np.arange(-59, 1)
    fig, ax = plt.subplots(figsize=(6.3, 4.9))
    fig.subplots_adjust(left=0.115, right=0.975, top=0.92, bottom=0.125)
    ax.axvspan(-60, 0, color="#f5f5f5", lw=0, zorder=0)
    ax.axvspan(T, 180, color=C["rule_light"], lw=0, zorder=0)
    ax.fill_between(t[:T + 1], 10 ** lo[i, :T + 1, 0] * s, 10 ** hi[i, :T + 1, 0] * s, color=C["ours"], alpha=0.3, lw=0, label="90 % interval")
    ax.plot(t[:T + 1], 10 ** mu[i, :T + 1, 0] * s, color=C["ours"], lw=2.4, label="forecast (ours, from GOES)")
    ax.fill_between(t[T:], 10 ** lo[i, T:, 0] * s, 10 ** hi[i, T:, 0] * s, color=C["ours"], alpha=0.07, lw=0)
    ax.plot(t[T:], 10 ** mu[i, T:, 0] * s, color=C["ours"], lw=1.6, ls=(0, (4, 2)), alpha=0.55)
    ax.plot(np.concatenate([ti, t]), np.concatenate([x_in, y]) * s, color=C["truth"], lw=2.2, label="observed (FISM2)")
    ax.axvline(0, color=C["truth"], lw=0.9)
    ax.axvline(T, color=C["rule"], lw=1.6, ls=(0, (4, 2)))
    ax.set_yscale("log")
    ylo, yhi = min(y.min(), x_in.min(), 10 ** lo[i, :, 0].min()) * s, max(y.max(), 10 ** hi[i, :, 0].max()) * s
    ax.set_ylim(ylo / 1.15, yhi * 2.1)
    _logticks(ax, ylo / 1.15, yhi * 1.1)
    ax.text(-30, yhi * 1.55, "input", ha="center", va="center", fontsize=11, fontweight="bold", color=C["text2"])
    ax.text(T / 2, yhi * 1.55, "forecast in force", ha="center", va="center", fontsize=11, fontweight="bold", color=C["ours"])
    ax.text((T + 180) / 2, yhi * 1.55, "new forecast", ha="center", va="center", fontsize=11, fontweight="bold", color=C["rule"])
    ax.annotate(f"next flare declared\n(+{T} min)", (T, ylo * 1.02), xytext=(T + 4, ylo * 0.98), fontsize=10, va="bottom", ha="left", linespacing=1.05, color=C["rule"])
    ax.set(xlim=(-60, 180), xlabel="minutes from the flare peak", ylabel=f"0.1–0.8 nm irradiance (10$^{{{e}}}$ W m$^{{-2}}$)")
    ax.set_xticks(range(-60, 181, 30))
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, columnspacing=1.3, handlelength=1.5, borderaxespad=0.1)
    NUM["fig8"] = dict(event=EXAMPLE_OUTSIDE, T_next=T, cov_w0_all180=float(ins.mean()), cov_w0_in_force=float(ins[:T].mean()),
                       cov_w0_last10_before=float(ins[T - 10:T].mean()), table_cov_all180=float(pd.read_csv(D.R / "full_server/paper1_p8_events.csv")
                                                                                              .set_index("event").loc[EXAMPLE_OUTSIDE, "cov_w0_goes_convnn_GRU-z"]))
    save(fig, out, "talk_fig8_example.png")


def fig9_coverage(D, out):
    """0.1-0.8 nm coverage of the 90 % interval over all 180 minutes and over the minutes in force (paper1_p8_reissue.csv),
    operational models."""
    T = pd.read_csv(D.R / "full_server/paper1_p8_reissue.csv")
    T = T[(T.input == "goes_convnn") & (T.rise_factor == 2.0)]
    groups = [("test flares (isolated)", "isolated test flares"), ("outside the rule: all", "flares outside the isolation rule"),
              ("all flares (population-weighted)", "all M/X flares (weighted)")]
    fig, ax = plt.subplots(figsize=(5.9, 4.9))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.91, bottom=0.125)
    y = 0.0
    yt, yl = [], []
    for gi, (g, lab) in enumerate(groups):
        n = int(T[(T.group == g)].n.iloc[0])
        ax.text(0.665, y - 0.62, f"{lab}  (n = {n})", fontsize=11, fontweight="bold", va="center", ha="left")
        for m in ("GRU-z", "FCN-z"):
            a = T[(T.model == m) & (T.group == g) & (T.scoring == "all 180 min")].iloc[0]
            b = T[(T.model == m) & (T.group == g) & (T.scoring == "minutes in force")].iloc[0]
            ax.plot([b.cov_w0_lo, b.cov_w0_hi], [y, y], color=C["rule"], lw=5, alpha=0.22, solid_capstyle="round", zorder=1)
            ax.annotate("", (b.cov_w0, y), (a.cov_w0, y), arrowprops=dict(arrowstyle="-", color=C["grey"], lw=1.6, shrinkA=4, shrinkB=4), zorder=2)
            ax.scatter([a.cov_w0], [y], s=64, facecolor="white", edgecolor=C["goes"], lw=1.6, zorder=3, label="all 180 min" if (gi == 0 and m == "GRU-z") else None)
            ax.scatter([b.cov_w0], [y], s=64, facecolor=C["rule"], edgecolor=C["rule"], lw=1.6, zorder=3, label="minutes in force" if (gi == 0 and m == "GRU-z") else None)
            wb = dict(facecolor="white", edgecolor="none", pad=0.8)
            ax.text(a.cov_w0 - 0.009, y, f"{a.cov_w0:.2f}", ha="right", va="center", fontsize=10.5, bbox=wb, zorder=2.5)
            ax.text(b.cov_w0 + 0.009, y, f"{b.cov_w0:.2f}", ha="left", va="center", fontsize=10.5, bbox=wb, zorder=2.5)
            yt.append(y); yl.append(m)
            NUM[f"fig9_{m}_{g}"] = dict(all180=float(a.cov_w0), in_force=float(b.cov_w0), lo=float(b.cov_w0_lo), hi=float(b.cov_w0_hi), n=n,
                                        MAPE4_all180=float(a.MAPE4), MAPE4_in_force=float(b.MAPE4), share_next=float(a.share_with_next_declaration),
                                        share_minutes=float(b.share_minutes_in_force))
            y += 0.62
        y += 0.95
    ax.axvline(0.9, color=C["truth"], lw=1.0, ls=(0, (4, 2)))
    ax.text(0.905, -1.02, "nominal 0.90", fontsize=10, color=C["text2"], ha="left", va="center")
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=10.5)
    ax.set(xlim=(0.66, 1.0), ylim=(y - 0.75, -1.3), xlabel="coverage of the 90 % interval (0.1–0.8 nm)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, columnspacing=1.4, handletextpad=0.2, borderaxespad=0.1)
    save(fig, out, "talk_fig9_coverage.png")


def numbers(D):
    """Numbers for the native charts and the text of the deck."""
    sys.path.insert(0, str(ROOT))
    from src.eval.metrics import pooled_metrics
    A = {}
    for m in ["Power-law", "LinearReg", "RandomForest", "FCN", "RNN", "LSTM", "GRU", "Transformer"]:
        pm = pooled_metrics(D.Ya, D.z[f"Y_{m}"].astype(float))
        A[m] = dict(CC4=float(pm["CC"].mean()), MAPE4=float(pm["MAPE"].mean()), APE4=float(pm["APE"].mean()))
    NUM["authors_models"] = A
    S = pd.read_csv(D.R / "full_server/summary_all.csv")
    g = lambda exp, m, col: float(S[(S.exp == exp) & (S.model == m) & (S.test == "zenodo96")][col].iloc[0])      # noqa: E731
    NUM["steps"] = {arch: dict(authors=A[arch]["MAPE4"], retrained=g("repro_cv96_full", arch, "MAPE4"), zscore=g("p2_zscore_full_gpu5", arch + "-z", "MAPE4"),
                               ensemble=g("p2_zscore_full_gpu5", arch + "-z", "ensMAPE4"), ens_CC4=g("p2_zscore_full_gpu5", arch + "-z", "ensCC4"),
                               ens_APE4=g("p2_zscore_full_gpu5", arch + "-z", "ensAPE4"), n_seeds_retrained=int(g("repro_cv96_full", arch, "n_seeds")))
                    for arch in ("GRU", "FCN")}
    H = pd.read_csv(D.R / "full_server/paper1_inputs_hboot.csv")
    H = H[(H.test == "rebuilt96") & (H.model == "all")]          # the five architectures together (seed means, the same 95 flares)
    q = lambda Aname, Bname: H[(H.A == Aname) & (H.B == Bname)].iloc[0]      # noqa: E731
    gf, cg = q("GOES", "FISM2"), q("FCNconv", "GOES")
    assert abs(q("FCNconv", "FISM2").MAPE4_A - cg.MAPE4_A) < 1e-9 and abs(cg.MAPE4_B - gf.MAPE4_A) < 1e-9
    NUM["inputs_5arch_mean"] = dict(FISM2=float(gf.MAPE4_B), GOES_direct=float(gf.MAPE4_A), GOES_converter=float(cg.MAPE4_A),
                                    d_goes=float(gf.d), d_goes_ci=[float(gf.lo), float(gf.hi)], d_conv_vs_goes=float(cg.d),
                                    d_conv_vs_goes_ci=[float(cg.lo), float(cg.hi)], recovered=float(-cg.d / gf.d))
    I = pd.read_csv(D.R / "full_server/paper1_inputs.csv")
    NUM["operational_ens"] = {f"{r.model}|{r.test}": float(r.ensMAPE4) for r in I[I.input == "GOES->FISM2 FCN"].itertuples()}
    NUM["benchmark_ens"] = {f"{r.model}|{r.test}": float(r.ensMAPE4) for r in I[I.input == "FISM2 (benchmark input)"].itertuples()}
    G = pd.read_csv(D.R / "full_server/paper1_p8_groups.csv")
    G = G[(G.group == "all") & (G.input == "goes_convnn")]
    NUM["outside_rule"] = {r.model: dict(MAPE4=float(r.MAPE4), MAPE4_test=float(r.MAPE4_test), d=float(r.d_vs_test), lo=float(r.lo), hi=float(r.hi), n=int(r.n),
                                         n_test=int(r.n_test), cov_w0=float(r.cov_w0), cov_w0_test=float(r.cov_w0_test)) for r in G.itertuples()}
    Dr = pd.read_csv(D.R / "full_server/paper1_p7_drap_hboot.csv")
    Dr = Dr[(Dr.input == "goes_convnn") & (Dr.k == 10)]
    NUM["drap_k10"] = {r.model: dict(delayed=float(r.MAE_dB_delayed), peak=float(r.MAE_dB_peak_issued_same_leads), p_holm=float(r.p_holm), n_day=int(r.n_day)) for r in Dr.itertuples()}
    Sn = pd.read_csv(D.R / "full_server/peak_detector_sensitivity.csv")          # declaration after k falling minutes (rise factor 2, all 1,604 flares)
    Sn = Sn[Sn.rise_factor == 2.0].set_index("k")
    NUM["detector"] = {f"k{k}": dict(own_declared=float(Sn.own_declared[k]), median_latency=float(Sn.median_latency[k]), with_earlier=float(Sn.with_earlier[k]),
                                     declarations_per_flare=float(Sn.declarations_per_flare[k])) for k in (1, 2)}
    fo = D.R / "full_server/paper1_onset.csv"
    if fo.exists():          # notebooks/analyze_rise_xclass.py (2026-11-19)
        O = pd.read_csv(fo)
        O = O[O.input == "goes_convnn"]
        q = lambda m, rule, grp, col: float(O[(O.model == m) & (O.rule == rule) & (O.group == grp)][col].iloc[0])      # noqa: E731
        NUM["onset"] = {m: dict(declaration=q(m, "declaration", "outside the rule: all", "cov_w0"), onset_M1=q(m, "onset r=2, flux >= M1", "outside the rule: all", "cov_w0"),
                                test_minutes_declaration=q(m, "declaration", "test flares (isolated)", "share_minutes_in_force"),
                                test_minutes_onset_M1=q(m, "onset r=2, flux >= M1", "test flares (isolated)", "share_minutes_in_force")) for m in ("GRU-z", "FCN-z")}
    P7 = pd.read_csv(D.R / "full_server/paper1_p7_delay_hboot.csv")
    P7 = P7[(P7.family == "delayed_vs_peak") & (P7.test == "rebuilt96") & (P7.input == "goes_convnn") & (P7.k == 2)]
    NUM["issue_plus2"] = {r.model: dict(d=float(r.d), lo=float(r.lo), hi=float(r.hi), MAPE4_lead_ge2_peak_model=float(r.MAPE4_peak_issued_same_leads)) for r in P7.itertuples()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--events", default=str(ROOT / "data/processed/events_latis.npz"))
    ap.add_argument("--goes-post", default=str(ROOT / "data/processed/events_goes_post.npz"))
    ap.add_argument("--out", default=str(ROOT / "figures/talk_v2"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = Data(Path(a.results), Path(a.events), Path(a.goes_post))
    k, pa, po = pick_example(D)
    fig1_task(D, out, k)
    lim_panels(D, out, k, pa)
    fig5_events(D, out, k, pa, po)
    fig7_reissue(D, out)
    fig8_example(D, out)
    fig9_coverage(D, out)
    try:
        numbers(D)
    except Exception as e:                      # a table of the deck charts is missing: keep the figure numbers
        print(f"[talk] numbers() skipped: {type(e).__name__}: {e}")
    (out / "talk_numbers.json").write_text(json.dumps(NUM, indent=1, sort_keys=True))
    print(f"[talk] talk_numbers.json: {len(NUM)} entries")


if __name__ == "__main__":
    main()
