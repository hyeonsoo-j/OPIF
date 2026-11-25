"""Figures of the paper (manuscript v6, 2026-11-25): layout of REVIEW_v1 option A (7 main-text figures + 6 Supporting-Information figures),
colour concept of docs/figure_style.md (sf_colors.py), Times New Roman only, vector PDF.

Colour = one question per element:  grey/black = data and reference (targets, inputs, flare groups, the Kim et al. setting and
benchmark-input runs of our models, GOES direct);  green = WHAT we forecast (operational path: GOES -> FCN converter -> GRU-z /
FCN-z, its interval, the HF product);  orange = WHEN we issue / stop (detector, +2 / +10 min issue marks, minutes in force, next
declaration). Orange never fills a forecast value of its own; prior work is never coloured.

    python notebooks/paper1_figures.py --results results --out figures/paper     [--only fig03_issue ...]
Inputs: results/full_server/*.csv (server tables), results/<exp>/oofL_*, val_*, oofN_* (5-seed arrays),
data/processed/{events_latis, events_test96_rebuilt, zenodo_full}.npz. Writes <out>/figNN_*.pdf, <out>/preview/*.png (300 dpi,
for checking only) and <out>/fig_numbers.json. No pickled data are loaded.

Main text: fig01 (TikZ, separate), fig02_examples, fig03_issue, fig04_validity, fig05_example, fig06_intervals, fig07_drap.
SI:        figS1_task, figS2_data, figS3_exclusion, figS4_inputs, figS5_issue_time, figS6_conformal_x.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from sf_colors import SF, use_times  # noqa: E402

B = ["w0", "w1", "w2", "w3"]
BAND = {"w0": "0.1–0.8 nm", "w1": "9.0–9.9 nm", "w2": "13.0–13.9 nm", "w3": "30.0–30.9 nm"}
C = {"prior": SF["prior"], "ref": SF["ref"], "ours": SF["ours"], "ours_mid": SF["ours_mid"], "ours_light": SF["ours_light"],
     "axis": SF["axis"], "truth": SF["truth"], "input": SF["input"], "soft": SF["soft"], "rule": SF["rule"],
     "rule_mid": SF["rule_mid"], "rule_pale": SF["rule_pale"], "rule_light": SF["rule_light"], "light": "#c8c8c8"}
INPUTS = [("FISM2 (benchmark input)", "FISM2 input (benchmark)", C["prior"]), ("GOES XRS-B direct", "GOES XRS-B direct", C["ref"]),
          ("GOES->FISM2 linear", "GOES→FISM2 linear", C["ours_mid"]), ("GOES->FISM2 FCN", "GOES→FISM2 FCN (operational)", C["ours"])]
EXP = {"fism": "p2_zscore_full_gpu5", "goes_b": "p2_zscore_full_in-goes_b_p1", "goes_convnn": "p2_zscore_full_in-goes_convnn_p2nn"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e2e2e2", "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4,
                     "savefig.dpi": 300, "figure.dpi": 110, "legend.frameon": False, "axes.edgecolor": "#b8b8b8",
                     "axes.linewidth": 0.8, "xtick.color": "#505050", "ytick.color": "#505050", "axes.labelcolor": "#303030",
                     "text.color": "#0b0b0b", "pdf.fonttype": 42, "ps.fonttype": 42})
use_times()                                    # Times New Roman for text and math; stops if it is not installed
NUM: dict = {}
T_LABEL = "minutes from the peak"
LEAD_LABEL = "lead time after the peak (min)"
BLOCKS = [slice(i * 30, (i + 1) * 30) for i in range(6)]


def panel_label(ax, s, x=None):
    kw = {} if x is None else {"x": x}
    ax.set_title(s, loc="left", fontweight="bold", fontsize=9, **kw)


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / "preview" / f"{name}.png", dpi=200, facecolor="white")
    plt.close(fig); print(f"[{name}] written")


def stack(R, exp, m, pre="oofL"):
    fs = sorted(glob.glob(f"{R}/{exp}/{pre}_{m}_s*.npy"))
    return np.stack([np.load(f, allow_pickle=False).astype(np.float64) for f in fs]) if fs else None


def val(R, exp, m):
    vs = [np.load(f, allow_pickle=False) for f in sorted(glob.glob(f"{R}/{exp}/val_{m}_s*.npz"))]
    assert vs and all(np.array_equal(v["idx"], vs[0]["idx"]) for v in vs)
    return np.stack([v["pred_log"].astype(np.float64) for v in vs]), vs[0]["idx"]


def cq(x, level):
    x = np.sort(np.asarray(x).ravel()); n = len(x)
    return x[min(int(np.ceil((n + 1) * level)) - 1, n - 1)]


def conf_norm(Lv, yv, Lt, level=0.9):
    """'conf-norm' split-conformal interval (identical to notebooks/analyze_p7_p8.conf_norm)."""
    mu_v, sd_v = Lv.mean(0), Lv.std(0); res_v = yv - mu_v
    mu_t, sd_t = Lt.mean(0), Lt.std(0)
    lo, hi = np.zeros_like(mu_t), np.zeros_like(mu_t)
    for b in range(4):
        for bl in BLOCKS:
            s0 = np.median(sd_v[:, bl, b])
            qn = cq((np.abs(res_v[:, bl, b]) / (sd_v[:, bl, b] + s0)).ravel(), level)
            half = qn * (sd_t[:, bl, b] + s0)
            lo[:, bl, b], hi[:, bl, b] = mu_t[:, bl, b] - half, mu_t[:, bl, b] + half
    return mu_t, lo, hi


class Data:
    """Data set arrays and pool rows (as src/data/pool.py; validation indices are pool indices 192 + position)."""
    def __init__(self, root: Path):
        self.D = np.load(root / "data/processed/events_latis.npz", allow_pickle=False)
        rb = np.load(root / "data/processed/events_test96_rebuilt.npz", allow_pickle=False)
        o = np.argsort(rb["test_k"]); assert (rb["test_k"][o] == np.arange(96)).all()
        self.rb = {k: rb[k][o] for k in ("X", "Y", "X_goes_b", "X_goes_a", "event_id", "cls", "goes_sat")}
        self.z = np.load(root / "data/processed/zenodo_full.npz", allow_pickle=False)
        D = self.D; eid = D["event_id"].astype(str); ev96 = [str(e) for e in self.z["event_names"]]
        y_ok = np.all(np.isfinite(D["Y"]) & (D["Y"] > 0), axis=(1, 2))
        pk = pd.to_datetime(D["peak"].astype(str)).values
        self.rest = {}
        for inp, key in (("goes_b", "X_goes_b"), ("default", "X")):
            x0 = D[key].astype(np.float64); valid = np.all(np.isfinite(x0) & (x0 > 0), axis=1) & y_ok
            r = np.where(D["selected"].astype(bool) & ~D["is_test"].astype(bool) & ~np.isin(eid, ev96) & valid)[0]
            self.rest[inp] = r[np.argsort(pk[r], kind="stable")]

    def yval(self, inp, idx):
        return np.log10(self.D["Y"][self.rest[inp][idx - 192]].astype(np.float64))


def evname(e):
    s, c = str(e).split("_")
    return f"{s[:4]}-{s[4:6]}-{s[6:]} {c}"


# ============================================================ main text
def fig02_examples(R, out, dat):
    """Two test flares of the rebuilt set (median M, largest X): operational forecast and interval, benchmark-input forecast."""
    I = np.load(Path(R) / "full_server/conformal_intervals_none.npz", allow_pickle=False)
    op, bm = "GRU-z [p2_zscore_full_in-goes_convnn_p2nn]", "GRU-z [p2_zscore_full_gpu5]"
    Y, Xf, Xg = np.log10(dat.rb["Y"]), np.log10(dat.rb["X"]), np.log10(dat.rb["X_goes_b"])
    names, cls = [str(e) for e in dat.rb["event_id"]], dat.rb["cls"].astype(str)
    mu, lo, hi = (I[f"{op}|rebuilt96|{k}"].astype(float) for k in ("mu", "lo", "hi"))
    mub = I[f"{bm}|rebuilt96|mu"].astype(float)
    ok = np.isfinite(mu).all((1, 2)) & np.isfinite(Xg).all(1)
    isx = np.char.startswith(cls, "X"); pk = Xf[:, -1]
    ix = int(np.argmax(np.where(isx & ok, pk, -np.inf)))
    mm = np.where(~isx & ok)[0]; im = int(mm[np.argsort(pk[mm])[len(mm) // 2]])
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0))
    t, ti = np.arange(180), np.arange(-59, 1)
    for row, i in enumerate((im, ix)):
        for b in range(4):
            ax = axes[row][b]
            ax.fill_between(t, lo[i, :, b], hi[i, :, b], color=C["ours"], alpha=0.22, lw=0, label="90 % interval (operational)")
            ax.plot(t, mu[i, :, b], color=C["ours"], label="operational forecast (GOES → FCN converter → GRU-z)")
            ax.plot(t, mub[i, :, b], color=C["prior"], lw=1.0, ls="--", label="benchmark-input forecast (FISM2 0.1–0.8 nm → GRU-z)")
            ax.plot(t, Y[i, :, b], color=C["truth"], lw=1.2, label="FISM2 target")
            if b == 0:
                ax.plot(ti, Xg[i], color=C["input"], lw=1.0, label="GOES XRS-B input")
                ax.plot(ti, Xf[i], color=C["prior"], lw=1.0, ls=(0, (3, 1.5)), label="FISM2 0.1–0.8 nm input (benchmark)")
                ax.set_ylabel("log$_{10}$ irradiance (W m$^{-2}$)")
            ax.set_title((f"{evname(names[i])}   " if b == 0 else "") + BAND[B[b]], fontsize=7, loc="left")
            if row == 1:
                ax.set_xlabel(T_LABEL)
            cov = float(((Y[i, :, b] >= lo[i, :, b]) & (Y[i, :, b] <= hi[i, :, b])).mean())
            NUM[f"fig02_{names[i]}_{B[b]}_coverage"] = cov
            NUM[f"fig02_{names[i]}_{B[b]}_mean_width_dex"] = float((hi[i, :, b] - lo[i, :, b]).mean())
    hh, ll = axes[0][0].get_legend_handles_labels()
    order = [ll.index(x) for x in ("FISM2 target", "operational forecast (GOES → FCN converter → GRU-z)", "90 % interval (operational)",
                                   "benchmark-input forecast (FISM2 0.1–0.8 nm → GRU-z)", "GOES XRS-B input", "FISM2 0.1–0.8 nm input (benchmark)")]
    fig.legend([hh[j] for j in order], [ll[j] for j in order], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=6.5,
               columnspacing=1.4, handlelength=1.8)
    fig.tight_layout(w_pad=0.6, h_pad=0.8, rect=(0, 0, 1, 0.9))
    NUM["fig02_events"] = [names[im], names[ix]]
    save(fig, out, "fig02_examples")


def _detector_bars(ax, R, compact=False):
    T = pd.read_csv(Path(R) / "full_server/peak_detector_rules.csv")
    rules = [("first", "stops at the first\nmaximum"), ("first_rise", "stops, needs a\nrise by 2"), ("armed2", "stays armed,\nrise by 2 (used)"),
             ("armed1.5", "stays armed,\nrise by 1.5"), ("armed1.2", "stays armed,\nrise by 1.2")]
    cats = [("own_first", "own peak, first declaration", C["rule"]), ("own_after", "own peak, after earlier declarations", C["rule_mid"]),
            ("other_only", "another maximum only", C["light"]), ("none", "nothing declared", C["input"])]
    A = T[T.group == "all"].set_index("rule"); Iso = T[T.group == "isolated"].set_index("rule")
    y = np.arange(len(rules)); left = np.zeros(len(rules))
    for key, lab, col in cats:
        v = np.array([100 * A.loc[r, key] for r, _ in rules])
        ax.barh(y, v, left=left, color=col, lw=0, height=0.62, label=lab); left += v
    for j, (r, _) in enumerate(rules):
        own = 100 * A.loc[r, "own_declared"]; on_light = 100 * A.loc[r, "own_after"] > 5
        ax.text(own - 1.5, j, f"{own:.1f}", ha="right", va="center", fontsize=6.5, fontweight="bold", color=C["truth"] if on_light else "white")
        NUM[f"detector_{r}_all"] = [float(A.loc[r, c]) for c in ("own_first", "own_after", "other_only", "none")]
        NUM[f"detector_{r}_isolated_own_declared"] = float(Iso.loc[r, "own_declared"])
    ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in rules], fontsize=6); ax.invert_yaxis()
    ax.set(xlabel=f"share of the {int(A.loc['first', 'n']):,} M/X flares (%)", xlim=(0, 100)); ax.grid(axis="y", visible=False)


def _delay_panel(ax, D, dc, lc, hc, pc, ylab, key_prefix):
    inputs = [("fism", "FISM2 input (benchmark)", C["prior"], "o"), ("goes_b", "GOES XRS-B direct", C["ref"], "s"),
              ("goes_convnn", "GOES→FISM2 FCN (operational)", C["ours"], "D")]
    ks = [2, 5, 10]
    for ii, (key, lab, col, mk) in enumerate(inputs):
        for mi, m in enumerate(("FCN-z", "GRU-z")):
            for ki, k in enumerate(ks):
                r = D[(D.input == key) & (D.model == m) & (D.k == k)]
                if not len(r):
                    continue
                r = r.iloc[0]; x = ki + (ii - 1) * 0.27 + (mi - 0.5) * 0.11
                ax.plot([x, x], [r[lc], r[hc]], color=col, lw=1.1, solid_capstyle="round", zorder=2)
                ax.scatter([x], [r[dc]], s=20, marker=mk, zorder=3, lw=1.0, edgecolor=col, facecolor=col if m == "GRU-z" else "white",
                           label=lab if (ki == 0 and m == "GRU-z") else None)
                if r[pc] < 0.05:
                    ax.annotate("*", (x, r[lc]), xytext=(0, -9), textcoords="offset points", ha="center", fontsize=8, color=C["truth"])
                NUM[f"{key_prefix}_{key}_{m}_k{k}"] = [float(r[dc]), float(r[lc]), float(r[hc]), float(r[pc])]
    ax.axhline(0, color=C["axis"], lw=0.8)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(["+2 min\nfirst forecast", "+5 min", "+10 min\nreissue"])
    for tl, col in zip(ax.get_xticklabels(), (C["rule"], C["axis"], C["rule"])):
        tl.set_color(col)
    y0, y1 = float(D[lc].min()), float(D[hc].max())
    ax.set(ylabel=ylab, xlim=(-0.55, len(ks) - 0.45), ylim=(y0 - 0.2 * (y1 - y0), y1 + 0.05 * (y1 - y0)))
    ax.grid(axis="x", visible=False)


def fig03_issue(R, out, dat):
    """When to issue: (a) detector outcome per rule, (b, c) change of the error when the forecast is issued k minutes after the peak."""
    D = pd.read_csv(Path(R) / "full_server/paper1_p7_delay_hboot.csv"); D = D[(D.family == "delayed_vs_peak") & (D.test == "rebuilt96")]
    fig = plt.figure(figsize=(7.2, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], left=0.115, right=0.99, bottom=0.2, top=0.8, wspace=0.42)
    ax = fig.add_subplot(gs[0]); _detector_bars(ax, R); panel_label(ax, "a", x=-0.42)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.42, 1.05), ncol=2, fontsize=5.8, columnspacing=0.8, handlelength=1.1, handletextpad=0.4)
    ax1 = fig.add_subplot(gs[1]); _delay_panel(ax1, D, "d", "lo", "hi", "p_holm", "ΔMAPE$_4$ (pt), lead ≥ issue time", "fig03b"); panel_label(ax1, "b")
    ax2 = fig.add_subplot(gs[2]); _delay_panel(ax2, D, "d_w0", "lo_w0", "hi_w0", "p_holm_w0", "ΔMAPE, 0.1–0.8 nm band (pt)", "fig03c"); panel_label(ax2, "c")
    hh, ll = ax1.get_legend_handles_labels()
    hh += [Line2D([], [], ls="none", marker="o", mfc="white", mec=C["axis"], ms=4), Line2D([], [], ls="none", marker="o", mfc=C["axis"], mec=C["axis"], ms=4)]
    ll += ["FCN-z (open)", "GRU-z (filled)"]
    fig.legend(hh, ll, loc="upper left", bbox_to_anchor=(0.43, 1.0), ncol=3, fontsize=6.2, columnspacing=1.0, handletextpad=0.3)
    save(fig, out, "fig03_issue")


def fig04_validity(R, out, dat):
    """Flares outside the isolation rule: error and coverage by lead-time block (operational GRU-z); coverage over the minutes in force."""
    Bk = pd.read_csv(Path(R) / "full_server/paper1_p8_blocks.csv"); Bk = Bk[(Bk.input == "goes_convnn") & (Bk.model == "GRU-z")]
    T = pd.read_csv(Path(R) / "full_server/paper1_p8_reissue.csv"); T = T[(T.input == "goes_convnn") & (T.rise_factor == 2.0)]
    A = pd.read_csv(Path(R) / "full_server/paper1_p8_reissue_aligned.csv"); A = A[(A.input == "goes_convnn") & (A.group == "outside the rule: all")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    groups = [("later flare within 180 min", "later flare within 180 min", C["input"], "s", "-"),
              ("both", "flares before and after", C["prior"], "D", (0, (4, 1.5))),
              ("earlier flare within 60 min", "earlier flare within 60 min only", C["ref"], "^", (0, (1.2, 1.2)))]
    xc = np.arange(15, 180, 30); key_sort = lambda v: v.str.split("-").str[0].astype(int)  # noqa: E731
    for ax, col_x, col_t, ylab, pl in zip(axes[0], ["MAPE4", "cov_w0"], ["MAPE4_test", "cov_w0_test"],
                                          ["MAPE$_4$ (%)", "coverage of the 90 % interval\n(0.1–0.8 nm)"], "ab"):
        t = Bk[Bk.group == "all"].sort_values("block_min", key=key_sort)
        ax.plot(xc, t[col_t], color=C["truth"], marker="o", ms=3.5, label="isolated test flares (n = 95)")
        NUM[f"fig04{pl}_test"] = [float(v) for v in t[col_t]]
        for key, lab, col, mk, ls in groups:
            g = Bk[Bk.group == key].sort_values("block_min", key=key_sort)
            ax.plot(xc, g[col_x], color=col, marker=mk, ms=3.5, ls=ls, label=f"{lab} (n = {int(g.n.iloc[0])})")
            NUM[f"fig04{pl}_{key}"] = [float(v) for v in g[col_x]]
        if col_x == "cov_w0":
            ax.axhline(0.9, color=C["axis"], lw=0.8, ls="--"); ax.set_ylim(0.45, 1.0)
        ax.set(xlabel=LEAD_LABEL, ylabel=ylab, xlim=(0, 180)); ax.set_xticks(range(0, 181, 30)); panel_label(ax, pl)
    hh, ll = axes[0][0].get_legend_handles_labels()
    fig.legend(hh, ll, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=6.5, columnspacing=1.6, handlelength=2.2)
    ax = axes[1][0]
    models = [("FCN-z", "o"), ("GRU-z", "s")]
    grp = [("test flares (isolated)", "isolated\ntest flares"), ("outside the rule: all", "flares outside\nthe isolation rule"),
           ("all flares (population-weighted)", "all flares\n(weighted)")]
    for gi, (g, _) in enumerate(grp):
        for mi, (m, mk) in enumerate(models):
            for si, sc in enumerate(("all 180 min", "minutes in force")):
                r = T[(T.model == m) & (T.group == g) & (T.scoring == sc)]
                if not len(r):
                    continue
                r = r.iloc[0]; x = gi + (mi - 0.5) * 0.36 + (si - 0.5) * 0.14
                c2 = C["rule"] if si == 1 else C["input"]
                ax.plot([x, x], [r.cov_w0_lo, r.cov_w0_hi], color=c2, lw=1.1, solid_capstyle="round", zorder=2)
                ax.scatter([x], [r.cov_w0], s=22, marker=mk, zorder=3, lw=1.0, edgecolor=c2, facecolor=c2 if si == 1 else "white")
                NUM[f"fig04c_{m}_{g}_{sc}"] = [float(r.cov_w0), float(r.cov_w0_lo), float(r.cov_w0_hi), int(r.n)]
    ax.axhline(0.9, color=C["axis"], lw=0.8, ls="--")
    ax.set_xticks(range(len(grp))); ax.set_xticklabels([lab for _, lab in grp], fontsize=6.5)
    ax.set(ylabel="coverage of the 90 % interval\n(0.1–0.8 nm)", ylim=(0.68, 1.0), xlim=(-0.5, len(grp) - 0.5))
    ax.grid(axis="x", visible=False); panel_label(ax, "c")
    ax.legend([Line2D([], [], ls="none", marker="o", mfc="white", mec=C["input"], ms=4.5), Line2D([], [], ls="none", marker="o", mfc=C["rule"], mec=C["rule"], ms=4.5),
               Line2D([], [], ls="none", marker="o", mfc=C["axis"], mec=C["axis"], ms=4), Line2D([], [], ls="none", marker="s", mfc=C["axis"], mec=C["axis"], ms=4)],
              ["all 180 min", "minutes in force", "FCN-z", "GRU-z"], loc="lower left", ncol=2, fontsize=6, columnspacing=0.8, handletextpad=0.2)
    ax = axes[1][1]
    ax.axvspan(0, 30, color=C["rule_light"], lw=0, zorder=0)
    for m, col in (("FCN-z", C["ours_mid"]), ("GRU-z", C["ours"])):
        q = A[(A.model == m) & A.minutes_from_next_declaration.between(-90, 30)].sort_values("minutes_from_next_declaration")
        ax.plot(q.minutes_from_next_declaration, q.cov_w0, color=col, label=f"operational {m}")
        for xx in (-60, -30, -20, -10, -4, 0):
            v = q[q.minutes_from_next_declaration == xx]
            if len(v):
                NUM[f"fig04d_{m}_x{xx}"] = [float(v.cov_w0.iloc[0]), int(v.n.iloc[0])]
    ax.axhline(0.9, color=C["axis"], lw=0.8, ls="--"); ax.axvline(0, color=C["rule"], lw=1.0)
    ax.text(1.5, 0.85, "next flare\ndeclared:\nnew forecast", fontsize=6, color=C["rule"], va="top", ha="left", linespacing=1.25)
    ax.set(xlabel="minutes from the declaration of the next flare", ylabel="coverage of the 90 % interval\n(0.1–0.8 nm)", xlim=(-90, 30), ylim=(0, 1.0))
    ax.set_xticks(range(-90, 31, 15)); ax.legend(loc="lower left", fontsize=6.5); panel_label(ax, "d")
    fig.tight_layout(w_pad=1.2, h_pad=1.6, rect=(0, 0, 1, 0.93))
    save(fig, out, "fig04_validity")


def fig05_example(R, out, dat, event="20110308_M1.3", m="GRU-z"):
    """A flare outside the isolation rule: operational 0.1-0.8 nm forecast in force until the detector declares the next flare."""
    D = dat.D; exp = Path(R) / "p2_zscore_full_in-goes_convnn_p8"
    ex_rows = pd.read_csv(exp / "extra_rows.csv").dataset_row.values
    Lx = stack(R, exp.name, m, "oofN"); Lv, idx = val(R, exp.name, m); yv = dat.yval("goes_b", idx)
    mu, lo, hi = conf_norm(Lv, yv, Lx)
    i = int(np.where(D["event_id"][ex_rows].astype(str) == event)[0][0]); r = int(ex_rows[i])
    T = int(pd.read_csv(Path(R) / "full_server/paper1_p8_reissue_events.csv").set_index("dataset_row").T_next_r2[r])
    y, x_in = D["Y"][r, :, 0].astype(float), D["X"][r].astype(float)
    ins = (np.log10(y) >= lo[i, :, 0]) & (np.log10(y) <= hi[i, :, 0])
    e = int(np.floor(np.log10(y.max()))); s = 10.0 ** -e
    t, ti = np.arange(180), np.arange(-59, 1)
    fig, ax = plt.subplots(figsize=(5.0, 3.3)); fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.14)
    ax.axvspan(-60, 0, color=C["soft"], lw=0, zorder=0); ax.axvspan(T, 180, color=C["rule_light"], lw=0, zorder=0)
    ax.fill_between(t[:T + 1], 10 ** lo[i, :T + 1, 0] * s, 10 ** hi[i, :T + 1, 0] * s, color=C["ours"], alpha=0.28, lw=0, label="90 % interval")
    ax.plot(t[:T + 1], 10 ** mu[i, :T + 1, 0] * s, color=C["ours"], lw=1.8, label="operational forecast (in force)")
    ax.fill_between(t[T:], 10 ** lo[i, T:, 0] * s, 10 ** hi[i, T:, 0] * s, color=C["ours"], alpha=0.07, lw=0)
    ax.plot(t[T:], 10 ** mu[i, T:, 0] * s, color=C["ours"], lw=1.1, ls=(0, (4, 2)), alpha=0.5, label="same forecast after it ends")
    ax.plot(np.concatenate([ti, t]), np.concatenate([x_in, y]) * s, color=C["truth"], lw=1.5, label="FISM2 0.1–0.8 nm (input and target)")
    ax.axvline(0, color=C["truth"], lw=0.7); ax.axvline(T, color=C["rule"], lw=1.3, ls=(0, (4, 2)))
    ax.set_yscale("log")
    ylo, yhi = min(y.min(), x_in.min(), 10 ** lo[i, :, 0].min()) * s, max(y.max(), 10 ** hi[i, :, 0].max()) * s
    ax.set_ylim(ylo / 1.15, yhi * 2.0)
    tk = [v for v in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50] if ylo / 1.15 <= v <= yhi * 1.1]
    from matplotlib.ticker import NullFormatter
    ax.set_yticks(tk); ax.set_yticklabels([f"{v:g}" for v in tk]); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.text(-30, yhi * 1.5, "input", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#505050")
    ax.text(T / 2, yhi * 1.5, "forecast in force", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C["ours"])
    ax.text((T + 180) / 2, yhi * 1.5, "new forecast", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C["rule"])
    ax.annotate(f"next flare declared (+{T} min)", (T, ylo * 1.02), xytext=(T + 3, ylo * 0.99), fontsize=6.5, va="bottom", ha="left", color=C["rule"])
    ax.set(xlim=(-60, 180), xlabel=T_LABEL, ylabel=f"0.1–0.8 nm irradiance (10$^{{{e}}}$ W m$^{{-2}}$)"); ax.set_xticks(range(-60, 181, 30))
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=6.3, columnspacing=1.2, handlelength=1.6, borderaxespad=0.1)
    NUM["fig05"] = dict(event=event, T_next=T, cov_w0_all180=float(ins.mean()), cov_w0_in_force=float(ins[:T].mean()),
                        cov_w0_last10_before=float(ins[T - 10:T].mean()))
    save(fig, out, "fig05_example")


CONF_ROWS = [("FCN-z [p2_zscore_full_gpu5]", "FCN-z", 0), ("GRU-z [p2_zscore_full_gpu5]", "GRU-z", 0), ("avg(FCN-z,GRU-z)", "FCN-z + GRU-z", 0),
             ("GRU-peaknorm [p3_ensemble_full_gpu]", "GRU-pn", 0),
             ("FCN-z [p2_zscore_full_in-goes_b_p1]", "FCN-z", 1), ("GRU-z [p2_zscore_full_in-goes_b_p1]", "GRU-z", 1),
             ("FCN-z [p2_zscore_full_in-goes_convnn_p2nn]", "FCN-z", 2), ("GRU-z [p2_zscore_full_in-goes_convnn_p2nn]", "GRU-z", 2)]
CONF_GRP = {0: ("FISM2 input (benchmark)", C["prior"]), 1: ("GOES XRS-B direct", C["ref"]), 2: ("GOES→FISM2 FCN (operational)", C["ours"])}


def _conf_rows(T):
    rows = [r for r in CONF_ROWS if (T.model == r[0]).any()]
    ypos, y = [], 0.0
    for i, r in enumerate(rows):
        if i and r[2] != rows[i - 1][2]:
            y += 0.6
        ypos.append(y); y += 1.0
    return rows, ypos


def fig06_intervals(R, out, dat):
    """(a) coverage of the validation-calibrated 90 % intervals per band; (b) reliability at four nominal levels (0.1-0.8 nm)."""
    T = pd.read_csv(Path(R) / "full_server/conformal_split_none.csv"); T = T[T.test == "rebuilt96"]
    rows, ypos = _conf_rows(T)
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], left=0.11, right=0.985, bottom=0.15, top=0.84, wspace=0.62)
    ax = fig.add_subplot(gs[0])
    bmk = {"w0": "o", "w1": "s", "w2": "^", "w3": "D"}
    for (key, lab, g), yy in zip(rows, ypos):
        r = T[T.model == key].iloc[0]
        for b in B:
            ax.scatter([r[f"cov_norm_{b}"]], [yy], s=22, color=CONF_GRP[g][1], marker=bmk[b], edgecolor="white", lw=0.5, zorder=3)
        NUM[f"fig06a_cov_{CONF_GRP[g][0]}_{lab}"] = [float(r[f"cov_norm_{b}"]) for b in B]
        NUM[f"fig06a_width_w0_{CONF_GRP[g][0]}_{lab}"] = float(r.width_norm_w0_dex)
    ax.axvline(0.9, color=C["axis"], lw=0.8, ls="--")
    ax.set_yticks(ypos); ax.set_yticklabels([r[1] for r in rows]); ax.invert_yaxis()
    for g, (name, col) in CONF_GRP.items():
        ys = [yy for (k, l, gg), yy in zip(rows, ypos) if gg == g]
        if ys:
            ax.text(1.005, np.mean(ys), {0: "FISM2 input\n(benchmark)", 1: "GOES XRS-B\ndirect", 2: "operational\n(GOES→FISM2\nFCN)"}[g], va="center", ha="left", fontsize=6, color=C["axis"] if g < 2 else col,
                    transform=ax.get_yaxis_transform())
    ax.set(xlabel="coverage of the 90 % interval, rebuilt test flares", xlim=(0.84, 1.0)); panel_label(ax, "a")
    ax.legend([Line2D([], [], ls="none", marker=bmk[b], mfc=C["axis"], mec="white", ms=5) for b in B], [BAND[b] for b in B],
              loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=4, fontsize=6.3, columnspacing=0.8, handletextpad=0.2)
    # (b) reliability: nominal vs empirical coverage, 0.1-0.8 nm, 95 test flares with a GOES window
    ax = fig.add_subplot(gs[1])
    ly = np.log10(dat.rb["Y"]); levels = (0.5, 0.8, 0.9, 0.95); rng = np.random.default_rng(5)
    sets = [("goes_convnn", "goes_b", "GRU-z", C["ours"], "s", True, -0.006), ("goes_convnn", "goes_b", "FCN-z", C["ours_mid"], "o", True, -0.002),
            ("fism", "default", "GRU-z", C["prior"], "s", False, 0.002), ("fism", "default", "FCN-z", C["prior"], "o", False, 0.006)]
    Lt0 = stack(R, EXP["goes_convnn"], "GRU-z"); ok = np.isfinite(Lt0).all((0, 2, 3))
    for key, inp, m, col, mk, filled, dx in sets:
        Lv, idx = val(R, EXP[key], m); yv = dat.yval(inp, idx); Lt = stack(R, EXP[key], m)[:, ok]
        cv, lo_, hi_ = [], [], []
        for lev in levels:
            _, lo, hi = conf_norm(Lv, yv, Lt, lev); ins = ((ly[ok] >= lo) & (ly[ok] <= hi))[:, :, 0]
            e = ins.mean(1); bt = e[rng.integers(0, len(e), (4000, len(e)))].mean(1)
            cv.append(ins.mean()); lo_.append(np.percentile(bt, 2.5)); hi_.append(np.percentile(bt, 97.5))
        x = np.array(levels) + dx
        ax.vlines(x, lo_, hi_, color=col, lw=1.0); ax.plot(x, cv, ls="none", marker=mk, ms=4, mec=col, mfc=col if filled else "white", mew=1.0,
                                                          label=f"{'operational' if filled else 'benchmark input'} {m}")
        NUM[f"fig06b_{key}_{m}"] = {f"{lev}": [float(c), float(a), float(b)] for lev, c, a, b in zip(levels, cv, lo_, hi_)}
    ax.plot([0.45, 1.0], [0.45, 1.0], color=C["axis"], lw=0.8, ls="--")
    ax.set(xlabel="nominal level", ylabel="empirical coverage, 0.1–0.8 nm", xlim=(0.45, 1.0), ylim=(0.45, 1.0), xticks=levels)
    ax.legend(loc="upper left", fontsize=6.3, handletextpad=0.3); panel_label(ax, "b")
    NUM["fig06b_n_flares"] = int(ok.sum())
    save(fig, out, "fig06_intervals")


def fig07_drap(R, out, dat):
    """10 MHz absorption over Korea, 31 daytime test flares: error propagated from the 0.1-0.8 nm forecast (a), dB intervals (b)."""
    D = pd.read_csv(Path(R) / "full_server/paper1_drap.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1.45, 1]))
    ax = axes[0]
    r = D[D.test == "rebuilt96"].copy()
    order = [f"{m} | {k}" for m in ("FCN-z", "GRU-z") for k, _, _ in INPUTS]
    r = r.set_index("model").reindex([o for o in order if o in set(r.model)]).reset_index()
    lab = {"FISM2 (benchmark input)": "FISM2 input", "GOES XRS-B direct": "GOES direct", "GOES->FISM2 linear": "GOES→FISM2 linear", "GOES->FISM2 FCN": "GOES→FISM2 FCN"}
    names = [f"{m.split(' | ')[0]}, {lab[m.split(' | ')[1]]}" for m in r.model]
    mae, pk, edge = list(r.MAE_dB), list(r.abs_peak_err_dB), [None] * len(r)
    for q in r.itertuples():
        NUM[f"fig07_drap_{q.model}"] = [float(q.MAE_dB), float(q.abs_peak_err_dB)]
    ref = []
    K = pd.read_csv(Path(R) / "full_server/paper1_p7_drap_hboot.csv"); K = K[(K.test == "rebuilt96") & (K.input == "goes_convnn")]
    for fam in ("FCN-z", "GRU-z"):
        for k in (5, 10):
            q = K[(K.model == fam) & (K.k == k)]
            if len(q):
                q = q.iloc[0]; ref.append((len(mae), float(q.MAE_dB_peak_issued_same_leads)))
                names.append(f"{fam}, GOES→FISM2 FCN, issued +{k} min"); mae.append(float(q.MAE_dB_delayed)); pk.append(np.nan)
                edge.append(C["rule"] if k == 10 else C["rule_mid"])
                NUM[f"fig07_drap_delay_goes_convnn_{fam}_k{k}"] = [float(q.MAE_dB_delayed), float(q.MAE_dB_peak_issued_same_leads), float(q.d), float(q.lo), float(q.hi), float(q.p_holm)]
    def _rc(nm):
        return C["prior"] if "FISM2 input" in nm else C["ref"] if "GOES direct" in nm else C["ours_mid"] if "linear" in nm else C["ours"]
    y = np.arange(len(mae))
    for yy, v, nm, ec in zip(y, mae, names, edge):
        ax.barh(yy, v, color=_rc(nm), lw=1.6 if ec else 0, edgecolor=ec if ec else "none", height=0.7)
    ax.barh(y, pk, color=C["truth"], edgecolor="white", lw=0.6, height=0.28)
    hh = [Patch(color=c_, lw=0) for c_ in (C["prior"], C["ref"], C["ours_mid"], C["ours"])]
    ll = ["FISM2 input (benchmark)", "GOES XRS-B direct", "GOES→FISM2 linear", "GOES→FISM2 FCN (operational)"]
    hh += [Patch(facecolor=C["ours"], edgecolor=C["rule_mid"], lw=1.6), Patch(facecolor=C["ours"], edgecolor=C["rule"], lw=1.6),
           Patch(color=C["truth"], lw=0), Line2D([], [], ls="none", marker="|", ms=8, mew=1.4, color=C["truth"])]
    ll += ["issued +5 min (orange edge)", "reissued +10 min (orange edge)", "|error| at the peak minute", "peak-issued model, same minutes"]
    if ref:
        ax.plot([v for _, v in ref], [i for i, _ in ref], ls="none", marker="|", ms=9, mew=1.4, color=C["truth"])
        ax.axhline(ref[0][0] - 0.5, color=C["axis"], lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.3); ax.invert_yaxis()
    ax.set(xlabel="10 MHz absorption error (dB), 31 daytime test flares", xlim=(0, 0.13)); panel_label(ax, "a")
    fig.legend(hh, ll, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=6.0, columnspacing=0.9, handlelength=1.3)
    ax = axes[1]
    Cc = pd.read_csv(Path(R) / "full_server/conformal_split_none.csv"); Cc = Cc[Cc.test == "rebuilt96"]
    def grp(mdl):
        return 1 if "in-goes_b" in mdl else 2 if "in-goes_convnn" in mdl else 3 if "in-goes_conv_" in mdl else 0
    gl = {0: ("FISM2 input", C["prior"], "o"), 1: ("GOES direct", C["ref"], "s"), 3: ("GOES→FISM2 linear", C["ours_mid"], "^"), 2: ("GOES→FISM2 FCN", C["ours"], "D")}
    off = {(0, "FCN"): ((4, 3), "left"), (0, "FCN+GRU"): ((4, -9), "left"), (0, "GRU"): ((4, 4), "left"), (0, "GRU-pn"): ((4, -9), "left"),
           (3, "FCN"): ((4, 4), "left"), (3, "GRU"): ((4, -9), "left"), (2, "FCN"): ((-4, 3), "right"), (2, "GRU"): ((-4, 4), "right"),
           (1, "GRU"): ((5, -3), "left"), (1, "FCN"): ((4, 4), "left")}
    for q in Cc.sort_values("drap_width_dB").itertuples():
        g = grp(q.model); name, col, mk = gl[g]
        ax.scatter([q.drap_width_dB], [q.drap_cov_norm], s=24, color=col, marker=mk, lw=0, zorder=3)
        short = q.model.split(" [")[0].replace("avg(FCN-z,GRU-z)", "FCN+GRU").replace("GRU-peaknorm", "GRU-pn").replace("-z", "")
        xy, ha = off.get((g, short), ((4, 3), "left"))
        ax.annotate(short, (q.drap_width_dB, q.drap_cov_norm), fontsize=5.5, xytext=xy, textcoords="offset points", color=C["axis"], ha=ha)
        NUM[f"fig07_dBinterval_{q.model}"] = [float(q.drap_cov_norm), float(q.drap_width_dB)]
    ax.axhline(0.9, color=C["axis"], lw=0.8, ls="--")
    x0, x1 = Cc.drap_width_dB.min(), Cc.drap_width_dB.max()
    ax.set(xlabel="mean width of the dB interval (dB)", ylabel="coverage of the dB interval", ylim=(0.86, 0.97), xlim=(x0 - 0.008, x1 + 0.016))
    panel_label(ax, "b")
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.85))
    save(fig, out, "fig07_drap")


# ============================================================ Supporting Information
def figS1_task(R, out, dat, event="20150114_M2.2"):
    """The prediction task on one isolated test flare; the authors' GRU forecast (prior work) in grey dashes. Event: among the 24 test
    flares with no other flare of our list within 6 h and no detector declaration or 1.5-fold rise in the window, the one whose MAPE4
    of the authors' GRU is closest to the median over the 96 test flares."""
    z = dat.z; k = [str(e) for e in z["event_names"]].index(event)
    X, Y, P = z["X_real"][k].astype(float), z["Y_true"][k].astype(float), z["Y_GRU"][k].astype(float)
    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.55, wspace=0.34, left=0.085, right=0.985, top=0.9, bottom=0.11)
    ax = fig.add_subplot(gs[0, :]); ti, t = np.arange(-59, 1), np.arange(180)
    e = int(np.floor(np.log10(Y[:, 0].max()))); s = 10.0 ** -e
    ax.axvspan(-60, 0, color=C["soft"], lw=0, zorder=0)
    ax.plot(ti, X * s, color=C["input"], lw=1.6, label="input: FISM2 0.1–0.8 nm (benchmark)")
    ax.plot(t, Y[:, 0] * s, color=C["truth"], lw=1.6, label="FISM2 target")
    ax.plot(t, P[:, 0] * s, color=C["prior"], lw=1.6, ls=(0, (4, 2)), label="forecast of Kim et al. (2026), GRU")
    ax.set_yscale("log")
    lo, hi = min(X.min(), Y[:, 0].min(), P[:, 0].min()) * s, max(X.max(), Y[:, 0].max(), P[:, 0].max()) * s
    ax.set_ylim(lo / 1.25, hi * 2.3)
    from matplotlib.ticker import NullFormatter
    tk = [v for v in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100] if lo / 1.25 <= v <= hi * 1.2]
    ax.set_yticks(tk); ax.set_yticklabels([f"{v:g}" for v in tk]); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set(xlim=(-60, 180), xlabel=T_LABEL, ylabel=f"irradiance (10$^{{{e}}}$ W m$^{{-2}}$)"); ax.set_xticks(range(-60, 181, 30))
    ax.axvline(0, color=C["truth"], lw=0.7)
    ax.text(-30, hi * 1.62, "input  60 min", ha="center", va="center", fontsize=8, fontweight="bold", color="#505050")
    ax.text(90, hi * 1.62, "forecast  180 min × 4 bands", ha="center", va="center", fontsize=8, fontweight="bold", color="#505050")
    ax.set_title(f"{evname(event)}   {BAND['w0']} (soft X-ray)", loc="left", fontweight="bold")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.99), ncol=3, columnspacing=1.5, handlelength=1.8, borderaxespad=0.0, fontsize=6.5)
    for b in (1, 2, 3):
        a2 = fig.add_subplot(gs[1, b - 1]); eb = int(np.floor(np.log10(np.median(Y[:, b]))))
        a2.plot(t, Y[:, b] / 10 ** eb, color=C["truth"], lw=1.4); a2.plot(t, P[:, b] / 10 ** eb, color=C["prior"], lw=1.4, ls=(0, (4, 2)))
        a2.set(xlim=(0, 180), xlabel=LEAD_LABEL); a2.set_xticks(range(0, 181, 60))
        a2.set_title(BAND[B[b]], loc="left", fontweight="bold")
        a2.text(1.0, 1.03, f"10$^{{{eb}}}$ W m$^{{-2}}$", transform=a2.transAxes, ha="right", va="bottom", fontsize=6.5, color="#505050")
        l2, h2 = min(Y[:, b].min(), P[:, b].min()) / 10 ** eb, max(Y[:, b].max(), P[:, b].max()) / 10 ** eb
        a2.set_ylim(l2 - 0.12 * (h2 - l2), h2 + 0.12 * (h2 - l2))
    NUM["figS1"] = dict(event=event, band_MAPE_authors_GRU=[float(v) for v in 100 * (np.abs(P - Y) / Y).mean(0)])
    save(fig, out, "figS1_task")


def figS2_data(R, out, dat):
    """Public-data series vs the authors' test arrays (log10 ratio profiles)."""
    X96, Y96 = dat.z["X_real"].astype(float), dat.z["Y_true"].astype(float)
    Xr, Yr, Xg = dat.rb["X"].astype(float), dat.rb["Y"].astype(float), dat.rb["X_goes_b"].astype(float)
    okg = np.all(np.isfinite(Xg) & (Xg > 0), axis=1)
    dx = np.log10(Xr) - np.log10(X96); dg = np.log10(Xg[okg]) - np.log10(X96[okg]); dy = np.log10(Yr) - np.log10(Y96)
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 2.4), gridspec_kw=dict(width_ratios=[1.25, 1, 1, 1, 1]))
    t_in, t_out = np.arange(-59, 1), np.arange(180); ax = axes[0]
    for arr, lab, col, ls in ((dx, "public FISM2 0.1–0.8 nm", C["input"], "-"), (dg, f"GOES XRS-B (n = {okg.sum()})", C["prior"], "--")):
        q = np.nanpercentile(arr, [25, 50, 75], axis=0)
        ax.fill_between(t_in, q[0], q[2], color=col, alpha=0.18, lw=0); ax.plot(t_in, q[1], color=col, label=lab, ls=ls)
    ax.axhline(0, color=C["axis"], lw=0.6); ax.set(xlabel=T_LABEL, ylabel="log$_{10}$(public / authors' array)")
    ax.legend(loc="upper left", fontsize=6); panel_label(ax, "a   input window")
    for b in range(4):
        ax = axes[b + 1]; q = np.nanpercentile(dy[:, :, b], [25, 50, 75], axis=0)
        ax.fill_between(t_out, q[0], q[2], color=C["input"], alpha=0.18, lw=0); ax.plot(t_out, q[1], color=C["input"])
        ax.axhline(0, color=C["axis"], lw=0.6); ax.set(xlabel=LEAD_LABEL, ylim=(-0.12, 0.12))
        if b:
            ax.set_yticklabels([])
        ax.set_title(f"{'bcde'[b]}   {BAND[B[b]]}", loc="left", fontweight="bold", fontsize=7.5)
        NUM[f"figS2_dy_{B[b]}_rms_dex"] = float(np.sqrt(np.nanmean(dy[:, :, b] ** 2)))
    axes[0].set_ylim(-0.4, 0.42)
    NUM["figS2_dx_fism2_rms_dex"] = float(np.sqrt(np.nanmean(dx ** 2))); NUM["figS2_dx_goes_rms_dex"] = float(np.sqrt(np.nanmean(dg ** 2)))
    NUM["figS2_dx_goes_median_dex"] = float(np.nanmedian(dg))
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "figS2_data")


def figS3_exclusion(R, out, dat):
    """Training-set exclusion experiments (benchmark input, peak-normalised models): change vs the full training set."""
    H, RU = pd.read_csv(Path(R) / "full_server/exclusion_hboot.csv"), pd.read_csv(Path(R) / "full_server/exclusion_runs.csv")
    conds = [("ex24h", "±24 h of a test flare (−163)"), ("rnd163", "random (−163), 3 draws"), ("exAR", "same active region (−516)"),
             ("rnd516", "random (−516), 3 draws")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True, sharex=True)
    for ax, (m, lab), pl in zip(axes, [("GRU-peaknorm", "GRU (peak-normalised)"), ("FCN-peaknorm", "FCN (peak-normalised)")], "ab"):
        for test, col, off, filled in (("rebuilt96", C["input"], -0.14, True), ("zenodo96", C["prior"], 0.14, False)):
            for i, (c, _) in enumerate(conds):
                r = H[(H.test == test) & (H.model == m) & (H.cond == c) & (H.vs == "baseline")]
                if not len(r):
                    continue
                r = r.iloc[0]; y = i + off
                ru = RU[(RU.test == test) & (RU.model == m) & (RU.cond == c)]
                ax.scatter(ru.d, np.full(len(ru), y), s=7, color=col, alpha=0.35, lw=0, zorder=2)
                ax.plot([r.lo, r.hi], [y, y], color=col, lw=1.6, zorder=3)
                ax.scatter([r.d], [y], s=26, facecolor=col if filled else "white", edgecolor=col, zorder=4, lw=1.1,
                           label=("rebuilt test set" if test == "rebuilt96" else "authors' test arrays") if i == 0 else None)
        ax.axvline(0, color=C["axis"], lw=0.8)
        ax.set_yticks(range(len(conds))); ax.set_yticklabels([l for _, l in conds])
        ax.set_xlabel("ΔMAPE$_4$ vs the full training set (pt)"); ax.set_title(lab, fontsize=8.5); panel_label(ax, pl)
    axes[0].invert_yaxis()
    hh, ll = axes[0].get_legend_handles_labels()
    fig.legend(hh, ll, loc="upper center", ncol=2, bbox_to_anchor=(0.6, 1.0), fontsize=7)
    fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.93))
    save(fig, out, "figS3_exclusion")


def figS4_inputs(R, out, dat):
    """Input paths: ensemble MAPE4 and APE4, converter effect per architecture."""
    T = pd.read_csv(Path(R) / "full_server/paper1_inputs.csv")
    inputs = [(k, lab, col) for k, lab, col in INPUTS if (T.input == k).any()]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    models = ["FCN-z", "GRU-z"]
    for ax, met, ylab, pl in zip(axes[:2], ["ensMAPE4", "ensAPE4"], ["MAPE$_4$ (%), 5-seed ensemble", "APE$_4$ at the peak (%)"], "ab"):
        for mi, m in enumerate(models):
            for ii, (k, lab, col) in enumerate(inputs):
                for j, (test, alpha) in enumerate((("rebuilt96", 1.0), ("zenodo96", 0.45))):
                    r = T[(T.model == m) & (T.input == k) & (T.test == test)]
                    if not len(r):
                        continue
                    w = 0.84 / len(inputs); x = mi + (ii - (len(inputs) - 1) / 2) * w + (-0.23 if j == 0 else 0.23) * w
                    ax.bar(x, float(r[met].iloc[0]), width=0.44 * w, color=col, alpha=alpha, lw=0, label=lab if (mi == 0 and j == 0) else None)
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models); ax.set_ylabel(ylab); panel_label(ax, pl)
        ax.set_ylim(0 if met == "ensAPE4" else 10, None if met == "ensAPE4" else 17.4)
    hh, ll = axes[0].get_legend_handles_labels()
    fig.legend(hh, ll, loc="upper center", ncol=len(inputs), bbox_to_anchor=(0.5, 1.0))
    axes[0].text(0.02, 0.97, "dark: rebuilt test set\nlight: authors' arrays", transform=axes[0].transAxes, va="top", fontsize=6, color=C["axis"])
    ax = axes[2]
    H = pd.read_csv(Path(R) / "full_server/paper1_inputs_hboot.csv"); H = H[(H.test == "rebuilt96") & (H.B == "GOES") & H.A.isin(["linear", "FCNconv"])]
    order = ["FCN-z", "GRU-z", "LSTM-z", "RNN-z", "Transformer-z", "all"]
    for i, m in enumerate(order):
        for a, col, off in (("linear", C["ours_mid"], -0.16), ("FCNconv", C["ours"], 0.16)):
            r = H[(H.model == m) & (H.A == a)]
            if not len(r):
                continue
            r = r.iloc[0]; y = i + off
            ax.plot([r.lo, r.hi], [y, y], color=col, lw=1.6)
            ax.scatter([r.d], [y], s=24 if m != "all" else 40, color=col, zorder=3, edgecolor="white", lw=0.8, marker="o" if m != "all" else "D")
    ax.axvline(0, color=C["axis"], lw=0.8); ax.axhline(4.5, color=C["axis"], lw=0.5, ls=":")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order[:-1] + ["all five"]); ax.invert_yaxis()
    ax.set_xlabel("ΔMAPE$_4$ vs GOES direct (pt), test flares"); panel_label(ax, "c")
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.91))
    save(fig, out, "figS4_inputs")


def figS5_issue_time(R, out, dat):
    """Error vs issue time with the benchmark input (-30 .. +10 min) and lead-time RMSE by issue time."""
    D = pd.read_csv(Path(R) / "full_server/paper1_delay.csv")
    Y96 = dat.z["Y_true"].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    fam = [("FCN-z", C["truth"], "o"), ("GRU-z", C["input"], "s"), ("GRU-pn", C["ref"], "^")]
    for ax, met, ylab, pl in zip(axes[:2], ["MAPE4_lead_ge_k", "APE4_at_issue"], ["MAPE$_4$ (%), lead ≥ issue time", "APE$_4$ at the issue time (%)"], "ab"):
        for name, col, mk in fam:
            for test, ls, mf in (("rebuilt96", "-", col), ("zenodo96", "--", "white")):
                r = D[(D.family == name) & (D.test == test)].sort_values("issue_delay_min")
                if len(r):
                    ax.plot(r.issue_delay_min, r[met], color=col, ls=ls, marker=mk, ms=3.2, mfc=mf, label=name if test == "rebuilt96" else None)
        ax.axvline(0, color=C["axis"], lw=0.7)
        for kk, tx in ((2, "+2"), (10, "+10")):
            ax.axvline(kk, color=C["rule"], lw=0.9, ls=(0, (3, 2)), zorder=1)
            ax.text(kk, 1.01, tx, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6, color=C["rule"])
        ax.set(xlabel="issue time relative to the peak (min)", ylabel=ylab); panel_label(ax, pl)
    axes[0].legend(loc="lower left", fontsize=6.3)
    axes[1].text(0.03, 0.04, "solid: rebuilt test set\ndashed: authors' arrays", transform=axes[1].transAxes, ha="left", va="bottom", fontsize=6, color=C["axis"])
    ax = axes[2]; cols = {0: C["input"], 2: C["rule_pale"], 5: C["rule_mid"], 10: C["rule"]}; yt = np.log10(Y96)
    for k in (0, 2, 5, 10):
        exp, m = ("p2_zscore_full_gpu5", "FCN-z") if k == 0 else ("paper1_delay_full", f"FCN-z-d{k}")
        L = stack(R, exp, m, "oof")
        if L is None:
            continue
        r = np.sqrt(np.nanmean((L.mean(0) - yt) ** 2, axis=0))[:, 0]; t = np.arange(180)
        ax.plot(t[k:], r[k:], color=cols[k], label=f"issued +{k} min" if k else "issued at the peak")
        NUM[f"figS5_rmse_w0_k{k}_at_30"] = float(r[30])
    ax.set(xlabel=LEAD_LABEL, ylabel="RMSE of log$_{10}$ flux, 0.1–0.8 nm (dex)", xlim=(0, 180)); ax.legend(loc="lower right", fontsize=6.3); panel_label(ax, "c")
    fig.tight_layout(w_pad=1.0)
    save(fig, out, "figS5_issue_time")


def figS6_conformal_x(R, out, dat):
    """X-class coverage (unconditional vs class-conditional calibration) and interval width, 0.1-0.8 nm."""
    T = {c: pd.read_csv(Path(R) / f"full_server/conformal_split_{c}.csv") for c in ("none", "class")}
    T = {c: v[v.test == "rebuilt96"] for c, v in T.items()}
    rows, ypos = _conf_rows(T["none"])
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.9), sharey=True, gridspec_kw=dict(width_ratios=[1.1, 1]))
    for (key, lab, g), yy in zip(rows, ypos):
        r = T["none"][T["none"].model == key].iloc[0]; col = CONF_GRP[g][1]; xu = float(r.cov_norm_w0_Xclass)
        axes[0].scatter([xu], [yy], s=22, color=col, lw=0, zorder=3, label="one quantile for all flares" if yy == ypos[0] else None)
        q = T["class"][T["class"].model == key]
        if len(q):
            xc = float(q.cov_norm_w0_Xclass.iloc[0])
            axes[0].plot([xu, xc], [yy, yy], color=C["axis"], lw=0.8, zorder=2)
            axes[0].scatter([xc], [yy], s=24, facecolor="white", edgecolor=col, marker="D", lw=1.1, zorder=4, label="class-conditional" if yy == ypos[0] else None)
        axes[1].barh(yy, float(r.width_norm_w0_dex), height=0.62, color=col, lw=0)
    axes[0].axvline(0.9, color=C["axis"], lw=0.8, ls="--")
    axes[0].set_yticks(ypos); axes[0].set_yticklabels([r[1] for r in rows]); axes[0].invert_yaxis()
    for g, (name, col) in CONF_GRP.items():
        ys = [yy for (k, l, gg), yy in zip(rows, ypos) if gg == g]
        axes[1].text(0.72, np.mean(ys), name.replace(" (", "\n("), va="center", ha="left", fontsize=6, color=C["axis"] if g < 2 else col)
    axes[0].set(xlabel="X-class coverage, 0.1–0.8 nm (8 flares)", xlim=(0.72, 0.96)); axes[1].set(xlabel="mean interval width, 0.1–0.8 nm (dex)", xlim=(0, 1.25))
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=2, fontsize=6.3)
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "figS6_conformal_x")


ALL = ["fig02_examples", "fig03_issue", "fig04_validity", "fig05_example", "fig06_intervals", "fig07_drap",
       "figS1_task", "figS2_data", "figS3_exclusion", "figS4_inputs", "figS5_issue_time", "figS6_conformal_x"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results")); ap.add_argument("--out", default=str(ROOT / "figures/paper"))
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(); out = Path(a.out); (out / "preview").mkdir(parents=True, exist_ok=True)
    dat = Data(ROOT); fails = 0
    for name in a.only or ALL:
        try:
            globals()[name](a.results, out, dat)
        except Exception as e:  # noqa: BLE001
            fails += 1; import traceback; traceback.print_exc(); print(f"[{name}] FAILED: {type(e).__name__}: {e}")
    fj = out / "fig_numbers.json"
    old = json.load(open(fj)) if (a.only and fj.exists()) else {}
    old.update(NUM); json.dump(old, open(fj, "w"), indent=1)
    print("numbers ->", fj); return fails


if __name__ == "__main__":
    sys.exit(main())
