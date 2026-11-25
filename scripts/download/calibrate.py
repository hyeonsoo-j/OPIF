"""Calibrate the reconstruction rules on the 96 Zenodo test events (run after fetch_fism2/fetch_goes --stage test).

The Zenodo arrays (X_real 96x60, Y_true 96x180x4; data/processed/zenodo_test_xy.npz) are the authors'
exact inputs/targets, so they tell us how the authors built them. For every test event we compare
our LaTiS/GOES download against them and choose, by minimum log10 error:

  1. time alignment : shift s (minutes) of the authors' peak minute relative to the catalogue peak
                      (searched in +-20 min with the FISM2 0.1-0.8 nm curve, scale-free);
  2. peak rule      : which real-time rule reproduces s (catalogue peak / argmax of FISM2 0.1-0.8 nm
                      or GOES XRS-B within a window around the catalogue peak);
  3. bin rule       : which 0.1-nm bins were summed per band (paper "9.0-9.9 nm" is ambiguous:
                      centres 9.05..9.85 = 'inner', 9.05..9.95 = 'plus', 8.95..9.85 = 'minus',
                      8.95..9.95 = 'wide') and how (sum x 0.1 nm = W m^-2, plain sum, or mean);
  4. input source   : X_real = FISM2 0.1-0.8 nm, or GOES XRS-B science (with / without quality
                      flags, or x0.7 = SWPC operational scaling for GOES 8-15).

Outputs (in data/download/): calibration.json (used by build_events.py), calibration_report.md,
calibration_events.csv, calibration_examples.png.

    python scripts/download/calibrate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import fetch_goes_xrs as G  # noqa: E402

BANDS = ["w0", "w1", "w2", "w3"]
PAPER_BANDS = {"w0": (0.1, 0.8), "w1": (9.0, 9.9), "w2": (13.0, 13.9), "w3": (30.0, 30.9)}
BIN_RULES = {"inner": (0.0, 0.0), "plus": (0.0, 0.1), "minus": (-0.1, 0.0), "wide": (-0.1, 0.1)}
AGGS = ("sum_x_dlambda", "sum", "mean")
PEAK_RULES = ("catalogue", "argmax_w0_pm15_clip", "argmax_w0_pm15", "argmax_w0_pm5", "argmax_goesb_pm15")
GOES_OPS_SCALE = 0.7          # SWPC operational XRS-B = 0.7 x science (GOES 8-15 only)


# ------------------------------------------------------------------------------------------
# reconstruction primitives (shared with build_events.py)
# ------------------------------------------------------------------------------------------
def bin_mask(wl: np.ndarray, band: str, rule: str) -> np.ndarray:
    a, b = PAPER_BANDS[band]
    if rule == "closed":
        return (wl >= a - 1e-6) & (wl <= b + 1e-6)
    lo, hi = BIN_RULES[rule]
    return (wl > a + lo + 1e-6) & (wl < b + hi - 1e-6)


def band_series(fz, band: str, rule: str, agg: str = "sum_x_dlambda") -> np.ndarray:
    """(n_min,) band value; NaN where any selected bin is missing."""
    wl, irr = np.asarray(fz[f"wl_{band}"]), np.asarray(fz[f"irr_{band}"])
    m = bin_mask(wl, band, rule)
    if m.sum() == 0:
        return np.full(irr.shape[0], np.nan)
    s = irr[:, m].sum(axis=1)
    if agg == "sum_x_dlambda":
        return s * 0.1
    if agg == "sum":
        return s
    if agg == "mean":
        return s / m.sum()
    raise ValueError(agg)


def unique_rules(wl: np.ndarray, band: str) -> list[str]:
    seen, out = set(), []
    for r in list(BIN_RULES) + ["closed"]:
        k = tuple(np.where(bin_mask(wl, band, r))[0])
        if k and k not in seen:
            seen.add(k)
            out.append(r)
    return out


def peak_index(rule: str, w0: np.ndarray, pre: int, start_off: int, end_off: int, goes_b: np.ndarray | None) -> int | None:
    """Index of the authors' peak minute in the downloaded grid (catalogue peak = index `pre`)."""
    if rule == "catalogue":
        return pre
    if rule.startswith("argmax_goesb"):
        if goes_b is None or not np.isfinite(goes_b[pre - 15: pre + 16]).any():
            return None
        seg = goes_b[pre - 15: pre + 16]
        return pre - 15 + int(np.nanargmax(seg))
    w = 5 if rule.endswith("pm5") else 15
    lo, hi = pre - w, pre + w
    if rule.endswith("_clip"):                      # also clipped to the catalogue start/end times
        lo, hi = max(lo, pre - start_off), min(hi, pre + end_off)
    seg = w0[lo: hi + 1]
    if not np.isfinite(seg).any():
        return None
    return lo + int(np.nanargmax(seg))


def choose_goes(root: Path, t0: int, n: int, ref_log: np.ndarray, sel: slice, use_flags: bool, sats: list[int]):
    """Pick the satellite with the best coverage (input window `sel`) then the highest log-correlation
    with the FISM2 0.1-0.8 nm curve. Returns (sat, {'xrsa','xrsb'}) or (None, None)."""
    best, best_key = None, None
    for sat in sats:
        s = G.load_series(root, sat, t0, n, use_flags=use_flags)
        if s is None:
            continue
        b = s["xrsb"]
        cov = float(np.isfinite(b[sel]).mean())
        ok = np.isfinite(b) & np.isfinite(ref_log) & (b > 0)
        cc = -1.0
        if ok.sum() > 30:
            with np.errstate(invalid="ignore", divide="ignore"):
                cc = float(np.corrcoef(np.log10(b[ok]), ref_log[ok])[0, 1])
            cc = cc if np.isfinite(cc) else -1.0
        key = (round(cov, 3), cc)
        if best_key is None or key > best_key:
            best, best_key = (sat, s), key
    return best if best else (None, None)


def sats_for(peak: pd.Timestamp) -> list[int]:
    return G.candidate_sats(pd.Timestamp(peak).normalize())


# ------------------------------------------------------------------------------------------
def _rms(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d ** 2))) if len(d) else np.nan


def _agg_offset(agg: str, nbins: int) -> float:
    return {"sum_x_dlambda": 0.0, "sum": 1.0, "mean": float(np.log10(10.0 / nbins))}[agg]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--pre", type=int, default=120, help="must match fetch_fism2 --pre")
    ap.add_argument("--max-shift", type=int, default=20)
    a = ap.parse_args(argv)
    root = C.dl_root(a.root)
    z = np.load(C.ROOT / "data/processed/zenodo_test_xy.npz")
    Xz, Yz = z["X_real"].astype(float), z["Y_true"].astype(float)
    ev = C.load_events("test")
    pre, S = a.pre, a.max_shift

    recs, cache = [], {}
    for e in ev.itertuples(index=False):
        f = root / "fism2" / f"{e.key}.npz"
        if not f.exists():
            continue
        fz = dict(np.load(f, allow_pickle=False))
        if int(fz["n"]) < pre + S + 180:
            continue
        cache[e.key] = fz
    if not cache:
        print("[calibrate] no FISM2 windows found - run fetch_fism2.py --stage test first")
        return 1
    print(f"[calibrate] {len(cache)}/{len(ev)} test events have FISM2 windows")

    # ---- 1. shift (scale-free, FISM2 0.1-0.8 nm 'inner' bins vs Y_true w0)
    shifts = {}
    for e in ev.itertuples(index=False):
        if e.key not in cache:
            continue
        L = np.log10(band_series(cache[e.key], "w0", "inner"))
        y0 = np.log10(Yz[e.test_k, :, 0])
        best = (np.inf, None)
        for s in range(-S, S + 1):
            seg = L[pre + s: pre + s + 180]
            d = y0 - seg
            if np.isfinite(d).sum() < 150:
                continue
            d = d[np.isfinite(d)]
            sc = float(np.median(np.abs(d - np.median(d))))
            if sc < best[0]:
                best = (sc, s)
        shifts[e.key] = best[1]

    # ---- 2. bin rule x aggregation per band (at the best shift)
    rule_rows = []
    for b in BANDS:
        wl_any = next(iter(cache.values()))[f"wl_{b}"]
        for r in unique_rules(np.asarray(wl_any), b):
            nb = int(bin_mask(np.asarray(wl_any), b, r).sum())
            d_all = []
            for e in ev.itertuples(index=False):
                s = shifts.get(e.key)
                if s is None:
                    continue
                V = band_series(cache[e.key], b, r)[pre + s: pre + s + 180]
                d_all.append(np.log10(Yz[e.test_k, :, BANDS.index(b)]) - np.log10(V))
            d = np.concatenate(d_all)
            d = d[np.isfinite(d)]
            med = float(np.median(d))
            row = dict(band=b, rule=r, n_bins=nb, n_values=len(d), free_offset_dex=med,
                       spread_mad_dex=float(np.median(np.abs(d - med))), rms_free_dex=_rms(d - med))
            for g in AGGS:
                row[f"rms_{g}"] = _rms(d - _agg_offset(g, nb))
            rule_rows.append(row)
    R = pd.DataFrame(rule_rows)
    choice = {}
    for b in BANDS:
        sub = R[R.band == b]
        # preference order on ties (e.g. 10 bins: mean == sum x 0.1 nm): AGGS order, then BIN_RULES order
        rms, g, r = np.inf, None, None
        for g_ in AGGS:
            j = sub[f"rms_{g_}"].idxmin()
            if sub.loc[j, f"rms_{g_}"] < rms - 1e-6:
                rms, g, r = float(sub.loc[j, f"rms_{g_}"]), g_, sub.loc[j, "rule"]
        choice[b] = dict(rule=r, agg=g, rms_dex=float(rms))

    # ---- 3/4. peak rule and input source
    use_flag_opts = (True, False)
    ev_rows = []
    for e in ev.itertuples(index=False):
        if e.key not in cache:
            continue
        fz, k, s = cache[e.key], e.test_k, shifts[e.key]
        w0 = band_series(fz, "w0", choice["w0"]["rule"], choice["w0"]["agg"])
        n, t0 = int(fz["n"]), int(fz["t0"])
        i0 = pre + s
        sel = slice(i0 - 59, i0 + 1)
        rec = dict(key=e.key, event=e.event_id, test_k=k, shift_best=s)
        so = int((pd.Timestamp(e.peak) - pd.Timestamp(e.start)).total_seconds() // 60)
        eo = int((pd.Timestamp(e.end) - pd.Timestamp(e.peak)).total_seconds() // 60)
        xz = np.log10(Xz[k])
        rec["rms_X_fism2_w0"] = _rms(xz - np.log10(w0[sel]))
        goes_b = None
        for uf in use_flag_opts:
            sat, gs = choose_goes(root, t0, n, np.log10(w0), sel, uf, sats_for(e.peak))
            tag = "flags" if uf else "noflags"
            rec[f"goes_sat_{tag}"] = sat
            if sat is None:
                continue
            gb = gs["xrsb"]
            if uf:
                goes_b = gb
            rec[f"rms_X_goes_xrsb_{tag}"] = _rms(xz - np.log10(gb[sel]))
            rec[f"cov_X_goes_xrsb_{tag}"] = float(np.isfinite(gb[sel]).mean())
            if sat <= 15:
                rec[f"rms_X_goes_xrsb_ops_{tag}"] = _rms(xz - np.log10(GOES_OPS_SCALE * gb[sel]))
            else:
                rec[f"rms_X_goes_xrsb_ops_{tag}"] = rec[f"rms_X_goes_xrsb_{tag}"]
        for pr in PEAK_RULES:
            pi = peak_index(pr, w0, pre, so, eo, goes_b)
            rec[f"shift_{pr}"] = None if pi is None else pi - pre
        for b in BANDS:
            V = band_series(fz, b, choice[b]["rule"], choice[b]["agg"])[i0: i0 + 180]
            rec[f"rms_Y_{b}"] = _rms(np.log10(Yz[k, :, BANDS.index(b)]) - np.log10(V))
        ev_rows.append(rec)
    E = pd.DataFrame(ev_rows)

    # peak rule: fraction reproducing the best shift
    pr_rows = []
    for pr in PEAK_RULES:
        v = E[f"shift_{pr}"]
        okm = v.notna()
        diff = (v[okm].astype(float) - E.shift_best[okm])
        pr_rows.append(dict(peak_rule=pr, n=int(okm.sum()), frac_exact=float((diff == 0).mean()) if okm.any() else 0.0,
                            frac_within1=float((diff.abs() <= 1).mean()) if okm.any() else 0.0))
    PR = pd.DataFrame(pr_rows)
    PRv = PR[PR.n >= 0.8 * len(E)]
    peak_rule = PRv.sort_values(["frac_exact", "frac_within1"], ascending=False, kind="stable").iloc[0].peak_rule

    src_rows = []
    for col, name in [("rms_X_fism2_w0", "fism2_w0"),
                      ("rms_X_goes_xrsb_flags", "goes_xrsb|flags"), ("rms_X_goes_xrsb_noflags", "goes_xrsb|noflags"),
                      ("rms_X_goes_xrsb_ops_flags", "goes_xrsb_ops|flags"),
                      ("rms_X_goes_xrsb_ops_noflags", "goes_xrsb_ops|noflags")]:
        if col in E:
            v = E[col].dropna()
            src_rows.append(dict(source=name, n=len(v), median_rms_dex=float(v.median()) if len(v) else np.nan,
                                 p90_rms_dex=float(v.quantile(0.9)) if len(v) else np.nan))
    SR = pd.DataFrame(src_rows)
    SRv = SR[SR.n >= 0.8 * len(E)]
    src = SRv.sort_values("median_rms_dex", kind="stable").iloc[0].source
    input_source, flag_mode = (src.split("|") + ["flags"])[:2]

    yq = {b: float(E[f"rms_Y_{b}"].median()) for b in BANDS}
    xq = float(SRv.sort_values("median_rms_dex", kind="stable").iloc[0].median_rms_dex)
    pr_exact = float(PR.set_index("peak_rule").loc[peak_rule, "frac_exact"])
    full = np.concatenate([Xz, Yz[:, 1:, 0]], axis=1)
    n_max_at_peak = int((full[:, 44:75].argmax(1) == 15).sum())
    ok = all(v < 0.01 for v in yq.values()) and xq < 0.01 and pr_exact >= 0.85
    cal = dict(
        created_utc=pd.Timestamp.now('UTC').strftime("%Y-%m-%dT%H:%M:%SZ"),
        n_events=int(len(E)), pre=pre,
        bands={b: dict(edges_nm=PAPER_BANDS[b], **choice[b]) for b in BANDS},
        peak_rule=str(peak_rule), input_source=input_source, goes_flags=flag_mode,
        median_rms_dex=dict(X=xq, **yq), peak_rule_frac_exact=pr_exact,
        zenodo_peak_is_w0_max_pm15=f"{n_max_at_peak}/96",
        shift_distribution=E.shift_best.value_counts().sort_index().to_dict(),
        status="PASS" if ok else "CHECK",
        note=("median per-event RMS of log10 differences vs the Zenodo arrays (at each event's best shift); "
              "PASS if all < 0.01 dex (2.3 %) and the peak rule reproduces the best shift for >= 85 % of "
              "events. CHECK means the rules found are the best available but differences remain "
              "(e.g. a different FISM2 version) - read calibration_report.md."),
    )
    (root / "calibration.json").write_text(json.dumps(cal, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    E.to_csv(root / "calibration_events.csv", index=False)
    with open(root / "calibration_report.md", "w", encoding="utf-8") as fh:
        fh.write("# Calibration against the Zenodo test arrays\n\n")
        fh.write(f"events: {len(E)} (of 96), status: **{cal['status']}**\n\n")
        fh.write("## Chosen rules\n\n```json\n" + json.dumps({k: cal[k] for k in ("bands", "peak_rule", "peak_rule_frac_exact", "zenodo_peak_is_w0_max_pm15", "input_source", "goes_flags", "median_rms_dex")}, indent=2) + "\n```\n\n")
        fh.write("## Bin rule x aggregation (log10 differences Zenodo - ours, all minutes)\n\n")
        fh.write(R.round(5).to_markdown(index=False) + "\n\n")
        fh.write("## Peak rule (fraction of events reproducing the best shift)\n\n" + PR.round(3).to_markdown(index=False) + "\n\n")
        fh.write("## Input source (per-event RMS of log10 X differences)\n\n" + SR.round(5).to_markdown(index=False) + "\n\n")
        fh.write("## Best shift distribution (minutes, authors' peak - catalogue peak)\n\n")
        fh.write(E.shift_best.value_counts().sort_index().to_frame("n").to_markdown() + "\n\n")
        worst = E.assign(worst=E[[f"rms_Y_{b}" for b in BANDS]].max(axis=1)).sort_values("worst", ascending=False).head(10)
        fh.write("## Worst 10 events (max band RMS, dex)\n\n" + worst[["event", "shift_best", "rms_X_fism2_w0"] + [f"rms_Y_{b}" for b in BANDS]].round(4).to_markdown(index=False) + "\n")
    _plot_examples(root, ev, cache, shifts, choice, Xz, Yz, pre)
    print(json.dumps({k: cal[k] for k in ("bands", "peak_rule", "peak_rule_frac_exact", "input_source", "goes_flags", "median_rms_dex", "status")}, indent=1))
    print(f"[calibrate] report: {root / 'calibration_report.md'}")
    return 0


def _plot_examples(root, ev, cache, shifts, choice, Xz, Yz, pre):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    keys = [e for e in ev.itertuples(index=False) if e.key in cache][:4]
    if not keys:
        return
    fig, axs = plt.subplots(len(keys), 4, figsize=(14, 2.6 * len(keys)), squeeze=False)
    for i, e in enumerate(keys):
        s = shifts[e.key]
        i0 = pre + s
        for j, b in enumerate(BANDS):
            V = band_series(cache[e.key], b, choice[b]["rule"], choice[b]["agg"])
            ax = axs[i, j]
            t = np.arange(-59, 180)
            if b == "w0":
                ax.semilogy(np.arange(-59, 1), Xz[e.test_k], "k-", lw=2.5, alpha=0.3, label="Zenodo X")
            ax.semilogy(np.arange(180), Yz[e.test_k, :, j], "k-", lw=2.5, alpha=0.5, label="Zenodo Y")
            ax.semilogy(t, V[i0 - 59: i0 + 180], "C1-", lw=1, label="ours")
            ax.set_title(f"{e.event_id} {b} (shift {s:+d})", fontsize=8)
            ax.tick_params(labelsize=7)
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "calibration_examples.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
