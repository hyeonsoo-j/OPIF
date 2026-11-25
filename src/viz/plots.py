"""Plotting helpers (matplotlib, Agg)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.cv96 import BAND_LABELS, BANDS  # noqa: E402

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#7f7f7f",
           "#bcbd22", "#17becf", "#393b79", "#637939"]


def event_panels(ev_idx, ev_names, X, Y, preds: dict, path, title=""):
    """Rows = events, cols = 4 bands. X input shown in the w0 column (t <= 0)."""
    n = len(ev_idx)
    fig, axs = plt.subplots(n, 4, figsize=(16, 2.3 * n), squeeze=False)
    t_in, t_out = np.arange(-59, 1), np.arange(0, 180)
    for r, k in enumerate(ev_idx):
        for b in range(4):
            ax = axs[r, b]
            if b == 0:
                ax.plot(t_in, X[k], color="0.5", lw=1, label="GOES input")
            ax.plot(t_out, Y[k, :, b], color="k", lw=1.8, label="FISM2 (truth)")
            for j, (name, P) in enumerate(preds.items()):
                ax.plot(t_out, P[k, :, b], lw=1, color=PALETTE[j % len(PALETTE)], label=name)
            ax.set_yscale("log")
            if r == 0:
                ax.set_title(BAND_LABELS[b])
            if b == 0:
                ax.set_ylabel(ev_names[k], fontsize=8)
            if r == n - 1:
                ax.set_xlabel("minutes from peak")
            ax.tick_params(labelsize=7)
    axs[0, 0].legend(fontsize=6, loc="lower left")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def metric_boxplot(E: pd.DataFrame, metric: str, path, order=None, title="", log=False, hue_col="source"):
    """E: long table model, band, event, metric, [source]. One panel per band."""
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8), sharey=False)
    order = order or sorted(E.model.unique())
    sources = sorted(E[hue_col].unique()) if hue_col in E else [None]
    w = 0.8 / len(sources)
    for b, bn in enumerate(BANDS):
        ax = axs[b]
        g = E[E.band == bn]
        for s_i, s in enumerate(sources):
            data, pos = [], []
            for m_i, m in enumerate(order):
                sel = g[(g.model == m) & ((g[hue_col] == s) if s is not None else True)][metric].dropna()
                if len(sel):
                    data.append(sel.values)
                    pos.append(m_i + (s_i - (len(sources) - 1) / 2) * w)
            if data:
                bp = ax.boxplot(data, positions=pos, widths=w * 0.9, patch_artist=True, showfliers=False)
                for patch in bp["boxes"]:
                    patch.set_facecolor(PALETTE[s_i])
                    patch.set_alpha(0.55)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=60, fontsize=7)
        ax.set_title(BAND_LABELS[b])
        if log:
            ax.set_yscale("log")
    axs[0].set_ylabel(metric)
    if sources[0] is not None:
        from matplotlib.patches import Patch
        axs[0].legend([Patch(color=PALETTE[i], alpha=0.55) for i in range(len(sources))], sources, fontsize=7)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def lead_time_curves(curves: dict, path, ylabel="RMSE", title="", log=True):
    """curves: name -> (180, 4) array."""
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.5))
    t = np.arange(0, 180)
    for b in range(4):
        for j, (name, c) in enumerate(curves.items()):
            axs[b].plot(t, c[:, b], color=PALETTE[j % len(PALETTE)], lw=1.2, label=name)
        axs[b].set_title(BAND_LABELS[b])
        axs[b].set_xlabel("lead time after peak (min)")
        if log:
            axs[b].set_yscale("log")
    axs[0].set_ylabel(ylabel)
    axs[0].legend(fontsize=7)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
