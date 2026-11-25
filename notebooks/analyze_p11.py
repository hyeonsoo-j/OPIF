"""P11: forecasts from the OPERATIONAL (archived real-time) GOES input. Design fixed in docs/decisions.md (2026-11-20, patch_v16)
before the run; this script follows that entry. No training.

    python notebooks/analyze_p11.py --results results [--n-boot 10000]

The run trained GRU-z / FCN-z (GOES + FCN converter, seeds 0-4) on science-quality inputs exactly as P8; the same trained model
predicted the rebuilt test flares and the flares outside the isolation rule from the science-quality input (oofL, oofN) and from the
operational input (twin rows, oofO; ops_rows.csv). Only flares with a complete operational window have a twin.

  paper1_p11_inputs.csv    (a) availability of the operational window and log10(operational / science) over the 60 input minutes,
                           per group (isolated test flares, flares outside the rule) and satellite series: XRS-B (the input channel
                           every model needs) and XRS-A (the second channel of the converter)
  paper1_p11_hboot.csv     (b) MAPE4 (all 180 min) operational - science, same model, same seed, same flares; family of 4
                           (2 models x {isolated test, outside the rule}); bootstrap over events and seeds with the seeds PAIRED
                           (the two forecasts of a seed come from the same weights), Holm. Next to it: the same with unpaired seeds
                           (rt.hboot), the 0.1-0.8 nm band, the 5-seed ensemble, and by satellite series
  paper1_p11_rule.csv      the text rule of the design entry, evaluated
  paper1_p11_inforce.csv   (c) minutes in force (declaration times of the science stream): MAPE4, 0.1-0.8 nm MAPE and coverage of the
                           90 % intervals (calibrated on the validation flares with science inputs), science and operational input
  paper1_p11_detector.csv  (c) the armed detector (k = 2, rise factor 2) on the operational stream next to the science stream, flares
                           with both streams: own peak declared, latency, earlier declarations, same declaration minute
  paper1_p11_events.csv    per twin: satellite, input agreement, event MAPE4 with either input (seed mean), declaration minutes
  paper1_p11_strata.csv    descriptive: change of the event MAPE4 by how much the inputs differ (XRS-B identical or not, XRS-A offset)
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("ap7", ROOT / "notebooks/analyze_p7_p8.py")
ap7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ap7)
rt, afs, pdet = ap7.rt, ap7.afs, ap7.pdet

MODELS, PRE, THR, TOL = ap7.MODELS, ap7.PRE, ap7.THR, ap7.TOL
EXP = "p2_zscore_full_in-goes_convnn_ops_p11"
SERIES = {8: "GOES 8-12", 9: "GOES 8-12", 10: "GOES 8-12", 11: "GOES 8-12", 12: "GOES 8-12", 13: "GOES 13-15", 14: "GOES 13-15",
          15: "GOES 13-15", 16: "GOES 16-18", 17: "GOES 16-18", 18: "GOES 16-18", 19: "GOES 16-18"}
MARGIN = 0.5          # points of MAPE4: the text rule of the design entry


def ev_mape(L, y, bands=(0, 1, 2, 3)):
    bands = list(bands)
    return 100 * (np.abs(10 ** L[..., bands] - y[None][..., bands]) / y[None][..., bands]).mean(axis=(2, 3))


def paired_boot(D, n=10000, seed=0):
    """D (S, n_events): operational - science per seed and event. Mean, 95 % interval and two-sided p from a bootstrap over
    events and over seeds (the same seeds for both inputs)."""
    rng = np.random.default_rng(seed)
    S, ne = D.shape
    out = np.empty(n)
    for i in range(n):
        out[i] = D[rng.integers(0, S, S)][:, rng.integers(0, ne, ne)].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    p = 2 * min((out <= 0).mean(), (out >= 0).mean())
    return float(D.mean()), float(lo), float(hi), float(min(1.0, max(p, 1.0 / n)))


def dlog_stats(o, s):
    ok = np.isfinite(o) & (o > 0) & np.isfinite(s) & (s > 0)
    if not ok.any():
        return dict(median_dlog=np.nan, median_abs_dlog=np.nan, rms_dlog=np.nan, share_abs_gt_0p05=np.nan, max_abs_dlog=np.nan)
    dl = np.log10(o[ok] / s[ok])
    return dict(median_dlog=float(np.median(dl)), median_abs_dlog=float(np.median(np.abs(dl))), rms_dlog=float(np.sqrt(np.mean(dl ** 2))),
                share_abs_gt_0p05=float((np.abs(dl) > 0.05).mean()), max_abs_dlog=float(np.abs(dl).max()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--events", default=str(ROOT / "data/processed/events_latis.npz"))
    ap.add_argument("--goes-post", default=str(ROOT / "data/processed/events_goes_post.npz"))
    ap.add_argument("--goes-ops", default=str(ROOT / "data/processed/events_goes_ops.npz"))
    ap.add_argument("--out", default=None, help="default: <results>/full_server")
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args(argv)
    R = Path(a.results)
    OUT = Path(a.out or R / "full_server")
    os.environ.update(FISM_PROTOCOL="full", FISM_EVENTS=str(Path(a.events).resolve()), FISM_GOES_POST=str(Path(a.goes_post).resolve()),
                      FISM_DEVICE_QUIET="1", FISM_EXCLUDE_NEAR_TEST_H="0", FISM_EXCLUDE_RANDOM_N="0", FISM_EXCLUDE_SAME_AR="0",
                      FISM_TRAIN_EXTRA_FOLD="", FISM_PREDICT_OPS="0", FISM_INPUT="goes_b", FISM_ISSUE_DELAY="0", FISM_PREDICT_EXTRA="1")
    from src.data import pool
    pool.load_full.cache_clear()
    X, Y, names, meta, tr, va, te, teL = pool.load_full()        # XRS-B pool: the same rows as the converter pool of the run
    ex_idx, ex_rows = pool.extra_index()
    d = np.load(a.events, allow_pickle=False)
    g = np.load(a.goes_ops, allow_pickle=False)
    st = np.load(R / "full_server/peak_detector_streams.npz", allow_pickle=False)
    assert np.array_equal(g["event_id"], d["event_id"]) and np.array_equal(g["row"], st["row"]) and not bool(g["has_aligned"])
    L = int(g["max_lag"])
    ops_stream = g["wide_b"][:, L:L + int(g["pre"]) + int(g["post"])]            # time stamps as archived
    srow = {int(r): j for j, r in enumerate(st["row"])}
    T = pd.read_csv(R / EXP / "ops_rows.csv")
    E = pd.read_csv(R / EXP / "extra_rows.csv")
    assert set(T.variant) == {"asis"} and np.array_equal(E.pool_index.values, ex_idx) and np.array_equal(E.dataset_row.values, ex_rows)
    pos = {int(p): j for j, p in enumerate(E.pool_index)}
    rows = T.dataset_row.values
    kind = T.kind.values
    y = d["Y"][rows].astype(np.float64)
    assert np.allclose(y, Y[T.twin_pool_index.values], rtol=0, atol=0), "targets of the twins differ from the pool"
    sat = d["goes_sat"][rows].astype(int)
    series = np.array([SERIES[s] for s in sat])
    is_test, is_out = kind == "test", kind == "outside"
    # ------------------------------------------------------------------------------------------------ (a) inputs
    xs, xo = d["X_goes_b"][rows].astype(float), g["xrsb_ops"][rows].astype(float)
    assert np.isfinite(xo).all() and (xo > 0).all()
    a_s, a_o = d["X_goes_a"][rows].astype(float), g["xrsa_ops"][rows].astype(float)      # XRS-A: used by the converter when the window is complete
    a_ok_s, a_ok_o = np.all(np.isfinite(a_s) & (a_s > 0), axis=1), np.all(np.isfinite(a_o) & (a_o > 0), axis=1)
    a_both = a_ok_s & a_ok_o
    n_all = {"isolated test flares": int(len(teL)), "flares outside the rule": int(len(ex_rows))}
    all_rows = {"isolated test flares": np.array([int(np.where(d["is_test"] & (d["test_k"] == k - 96))[0][0]) for k in teL]), "flares outside the rule": ex_rows}
    inp_rows = []
    for gname, mk in (("isolated test flares", is_test), ("flares outside the rule", is_out)):
        ar = all_rows[gname]
        has_any = np.array([np.isfinite(ops_stream[srow[int(r)], PRE - 59:PRE + 1]).any() if int(r) in srow else False for r in ar])
        for ser in ("all", "GOES 8-12", "GOES 13-15", "GOES 16-18"):
            m2 = mk if ser == "all" else mk & (series == ser)
            sel_all = np.ones(len(ar), bool) if ser == "all" else np.array([SERIES[int(s)] == ser for s in d["goes_sat"][ar]])
            inp_rows.append(dict(group=gname, series=ser, n_flares=int(sel_all.sum()), n_with_operational_data=int((has_any & sel_all).sum()),
                                 n_complete_window=int(m2.sum()), share_complete=float(m2.sum() / max(sel_all.sum(), 1)),
                                 **dlog_stats(xo[m2], xs[m2]),
                                 **{f"xrsa_{k_}": v for k_, v in dlog_stats(a_o[m2 & a_both], a_s[m2 & a_both]).items()},
                                 xrsa_n_both_complete=int((m2 & a_both).sum()), xrsa_n_operational_only=int((m2 & a_ok_o & ~a_ok_s).sum()),
                                 xrsa_n_science_only=int((m2 & ~a_ok_o & a_ok_s).sum()),
                                 share_flares_identical=float(np.mean(np.abs(np.log10(xo[m2] / xs[m2])).max(1) < 1e-4)) if m2.any() else np.nan,
                                 share_flares_rms_gt_0p02=float(np.mean(np.sqrt((np.log10(xo[m2] / xs[m2]) ** 2).mean(1)) > 0.02)) if m2.any() else np.nan))
    pd.DataFrame(inp_rows).to_csv(OUT / "paper1_p11_inputs.csv", index=False)
    ev_rms = np.sqrt((np.log10(xo / xs) ** 2).mean(1))
    ev_peak = np.log10(xo[:, -1] / xs[:, -1])
    # declaration minutes on either stream (armed detector, k = 2, rise factor 2)
    Ts = np.array([ap7.next_declaration(st["xrsb"][srow[int(r)]], 2, 2.0) for r in rows])
    To = np.array([ap7.next_declaration(ops_stream[srow[int(r)]], 2, 2.0) for r in rows])
    a_dl = np.full(a_o.shape, np.nan)
    a_dl[a_both] = np.log10(a_o[a_both] / a_s[a_both])
    a_rms, a_med = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    a_rms[a_both], a_med[a_both] = np.sqrt((a_dl[a_both] ** 2).mean(1)), np.median(a_dl[a_both], axis=1)
    tab = pd.DataFrame(dict(dataset_row=rows, event=d["event_id"][rows].astype(str), cls=d["cls"][rows].astype(str), kind=kind, sat=sat, series=series,
                            input_rms_dlog=ev_rms, input_dlog_peak_minute=ev_peak, xrsa_science_complete=a_ok_s, xrsa_operational_complete=a_ok_o,
                            xrsa_rms_dlog=a_rms, xrsa_median_dlog=a_med,
                            T_next_science=Ts, T_next_operational=To))
    # ------------------------------------------------------------------------------------------------ (b), (c)
    preds = lambda m, pre: np.stack([np.load(f, allow_pickle=False).astype(np.float64) for f in sorted(glob.glob(f"{R}/{EXP}/{pre}_{m}_s*.npy"))])   # noqa: E731
    hb, inf = [], []
    rng = np.random.default_rng(71)
    t180 = np.arange(180)[None, :]
    for m in MODELS:
        LO, Lt, Lx = preds(m, "oofO"), preds(m, "oofL"), preds(m, "oofN")
        assert LO.shape[0] == Lt.shape[0] == Lx.shape[0] == 5 and LO.shape[1] == len(T) and not np.isnan(LO).any()
        LS = np.stack([Lt[:, int(t) - 96] if k == "test" else Lx[:, pos[int(t)]] for t, k in zip(T.twin_pool_index, kind)], axis=1)   # science twin
        assert not np.isnan(LS).any()
        A, C = ev_mape(LO, y), ev_mape(LS, y)
        A0, C0 = ev_mape(LO, y, (0,)), ev_mape(LS, y, (0,))
        Ae, Ce = ev_mape(LO.mean(0, keepdims=True), y)[0], ev_mape(LS.mean(0, keepdims=True), y)[0]
        tab[f"MAPE4_science_{m}"], tab[f"MAPE4_operational_{m}"] = C.mean(0), A.mean(0)
        for gname, mk in (("isolated test flares", is_test), ("flares outside the rule", is_out)):
            for ser in ("all", "GOES 8-12", "GOES 13-15", "GOES 16-18"):
                m2 = mk if ser == "all" else mk & (series == ser)
                if m2.sum() < 5:
                    continue
                D = (A - C)[:, m2]
                dd, lo, hi, p = paired_boot(D, a.n_boot, 21)
                du, lou, hiu, pu = rt.hboot(A[:, m2], C[:, m2], n=a.n_boot, seed=22)
                d0, lo0, hi0, p0 = paired_boot((A0 - C0)[:, m2], a.n_boot, 23)
                de, loe, hie, pe = paired_boot((Ae - Ce)[None, m2], a.n_boot, 24)
                evd = D.mean(0)
                hb.append(dict(family=ser == "all", model=m, set=gname, series=ser, n=int(m2.sum()), MAPE4_science=C[:, m2].mean(), MAPE4_operational=A[:, m2].mean(),
                               d=dd, lo=lo, hi=hi, p_boot=p, d_unpaired_seeds=du, lo_unpaired_seeds=lou, hi_unpaired_seeds=hiu, p_unpaired_seeds=pu,
                               median_event_d=float(np.median(evd)), share_events_abs_d_gt_1pt=float((np.abs(evd) > 1).mean()), max_event_d=float(evd.max()),
                               min_event_d=float(evd.min()), MAPE_w0_science=C0[:, m2].mean(), MAPE_w0_operational=A0[:, m2].mean(), d_w0=d0, lo_w0=lo0, hi_w0=hi0,
                               p_boot_w0=p0, ensMAPE4_science=Ce[m2].mean(), ensMAPE4_operational=Ae[m2].mean(), d_ens=de, lo_ens=loe, hi_ens=hie, p_boot_ens=pe))
        # minutes in force; intervals calibrated on the validation flares (science inputs)
        vs = [np.load(f, allow_pickle=False) for f in sorted(glob.glob(f"{R}/{EXP}/val_{m}_s*.npz"))]
        vidx = vs[0]["idx"]
        assert len(vs) == 5 and all(np.array_equal(v["idx"], vidx) for v in vs) and np.array_equal(np.sort(vidx), np.sort(va))
        Lv, yv = np.stack([v["pred_log"].astype(np.float64) for v in vs]), np.log10(Y[vidx])
        ly = np.log10(y)
        res = {}
        for iname, Lq, Tq in (("science-quality input", LS, Ts), ("operational input", LO, Ts), ("operational input, declarations of the operational stream", LO, To)):
            _, lo_, hi_ = ap7.conf_norm(Lv, yv, Lq)
            ins = (ly >= lo_) & (ly <= hi_)
            rel = 100 * np.abs(10 ** Lq - y[None]) / y[None]
            for gname, mk in (("isolated test flares", is_test), ("flares outside the rule", is_out)):
                pv, bv = ap7.in_force(rel[:, mk], ins[mk], Tq[mk], True, rng, 2000)
                V = t180 < Tq[mk][:, None]
                res[(iname, gname)] = (pv, bv)
                rec = dict(model=m, input=iname, group=gname, n=int(mk.sum()), share_minutes_in_force=pv["den"] / 180, MAPE4=pv["MAPE4"] / pv["den"],
                           MAPE_w0=pv["MAPE_w0"] / pv["den"], cov_w0=pv["cov_w0"] / pv["den"], cov_4band=pv["cov_4band"] / pv["den"],
                           width_w0_dex=float(((hi_ - lo_)[mk][:, :, 0] * V).sum() / V.sum()), cov_w0_all180=float(ins[mk][:, :, 0].mean()))
                for k_ in ("MAPE4", "cov_w0"):
                    rec[f"{k_}_lo"], rec[f"{k_}_hi"] = (float(v) for v in np.percentile(bv[k_] / bv["den"], [2.5, 97.5]))
                inf.append(rec)
            pt, px = res[(iname, "isolated test flares")], res[(iname, "flares outside the rule")]
            comb = lambda u, v: (1 - ap7.P_OUTSIDE) * u + ap7.P_OUTSIDE * v              # noqa: E731
            pv, bv = {k_: comb(pt[0][k_], px[0][k_]) for k_ in pt[0]}, {k_: comb(pt[1][k_], px[1][k_]) for k_ in pt[1]}
            rec = dict(model=m, input=iname, group="all flares (population-weighted)", n=int(len(T)), share_minutes_in_force=pv["den"] / 180, MAPE4=pv["MAPE4"] / pv["den"],
                       MAPE_w0=pv["MAPE_w0"] / pv["den"], cov_w0=pv["cov_w0"] / pv["den"], cov_4band=pv["cov_4band"] / pv["den"], width_w0_dex=np.nan, cov_w0_all180=np.nan)
            for k_ in ("MAPE4", "cov_w0"):
                rec[f"{k_}_lo"], rec[f"{k_}_hi"] = (float(v) for v in np.percentile(bv[k_] / bv["den"], [2.5, 97.5]))
            inf.append(rec)
    H = pd.DataFrame(hb)
    fam = H.family.values
    assert fam.sum() == 4
    H["p_holm"] = np.nan
    H.loc[fam, "p_holm"] = afs.holm(H.loc[fam, "p_boot"].values)
    H.to_csv(OUT / "paper1_p11_hboot.csv", index=False)
    pd.DataFrame(inf).to_csv(OUT / "paper1_p11_inforce.csv", index=False)
    tab.to_csv(OUT / "paper1_p11_events.csv", index=False)
    # descriptive strata: where does the forecast change come from
    b_same = ev_rms < 1e-4
    strata = {"XRS-B identical, XRS-A within 0.02 dex rms": b_same & a_both & (tab.xrsa_rms_dlog.values < 0.02),
              "XRS-B identical, XRS-A differs (>= 0.02 dex rms)": b_same & a_both & (tab.xrsa_rms_dlog.values >= 0.02),
              "XRS-B differs by < 0.005 dex rms": ~b_same & (ev_rms < 0.005), "XRS-B differs by >= 0.005 dex rms": ev_rms >= 0.005,
              "XRS-A complete in one archive only": a_ok_s != a_ok_o}
    srows = []
    for gname, mk in (("isolated test flares", is_test), ("flares outside the rule", is_out), ("both groups", np.ones(len(T), bool))):
        for sname, sm in strata.items():
            q = mk & sm
            if q.sum() == 0:
                continue
            rec = dict(group=gname, stratum=sname, n=int(q.sum()))
            for m in MODELS:
                dv = (tab[f"MAPE4_operational_{m}"] - tab[f"MAPE4_science_{m}"]).values[q]
                rec[f"mean_d_{m}"], rec[f"mean_abs_d_{m}"], rec[f"max_abs_d_{m}"] = float(dv.mean()), float(np.abs(dv).mean()), float(np.abs(dv).max())
            srows.append(rec)
    pd.DataFrame(srows).to_csv(OUT / "paper1_p11_strata.csv", index=False)
    # ------------------------------------------------------------------------------------------------ the text rule
    rr = []
    for m in MODELS:
        t = H[fam & (H.model == m) & (H.set == "isolated test flares")].iloc[0]
        o = H[fam & (H.model == m) & (H.set == "flares outside the rule")].iloc[0]
        rr.append(dict(model=f"operational {m}", n_test=int(t.n), d_test=t.d, lo_test=t.lo, hi_test=t.hi, margin=MARGIN, upper_end_below_margin=bool(t.hi < MARGIN),
                       upper_end_below_margin_unpaired_seeds=bool(t.hi_unpaired_seeds < MARGIN), hi_test_unpaired_seeds=t.hi_unpaired_seeds,
                       n_outside=int(o.n), d_outside=o.d, lo_outside=o.lo, hi_outside=o.hi))
    Rl = pd.DataFrame(rr)
    Rl["rule_met_for_both"] = bool(Rl.upper_end_below_margin.all())
    Rl.to_csv(OUT / "paper1_p11_rule.csv", index=False)
    # ------------------------------------------------------------------------------------------------ detector on either stream
    det = []
    both = np.array([np.isfinite(ops_stream[j, PRE - 10:PRE + 10]).mean() >= 0.5 for j in range(len(st["row"]))])
    rec_s, rec_o = [], []
    for j in np.where(both)[0]:
        for store, b in ((rec_s, st["xrsb"][j]), (rec_o, ops_stream[j])):
            dec = pdet.detect_all(b, 2, THR, rise_factor=2.0)
            q = pdet.own_peak(dec, PRE, TOL)
            q["T_next"] = ap7.next_declaration(b, 2, 2.0)
            store.append(q)
    Ds, Do = pd.DataFrame(rec_s), pd.DataFrame(rec_o)
    jrow = st["row"][both]
    groups = {"all flares with both streams": np.ones(len(jrow), bool), "isolated test flares": d["is_test"][jrow].astype(bool),
              "flares outside the rule": ~d["selected"][jrow].astype(bool) & ~d["is_test"][jrow].astype(bool)}
    for gname, mk in groups.items():
        for sname, Dq in (("science-quality stream", Ds), ("operational stream", Do)):
            q = Dq[mk]
            fin = q.own.dropna()
            det.append(dict(group=gname, stream=sname, n=int(mk.sum()), frac_own_declared=float(q.own.notna().mean()), median_own_latency=float(fin.median()),
                            p90_own_latency=float(fin.quantile(0.9)), frac_own_on_time=float(((q.own >= 0) & (q.own <= 4)).mean()),
                            frac_with_earlier=float((q.n_before > 0).mean()), frac_no_declaration=float(q["first"].isna().mean()),
                            frac_next_declared_within_180=float((q.T_next < 180).mean())))
        same_own = (Ds.own[mk].fillna(-999).values == Do.own[mk].fillna(-999).values)
        dT = Do.T_next[mk].values - Ds.T_next[mk].values
        det.append(dict(group=gname, stream="operational vs science", n=int(mk.sum()), frac_same_own_declaration_minute=float(same_own.mean()),
                        frac_own_within_1min=float((np.abs(Ds.own[mk].fillna(-999).values - Do.own[mk].fillna(-999).values) <= 1).mean()),
                        frac_same_next_declaration=float((dT == 0).mean()), frac_next_within_2min=float((np.abs(dT) <= 2).mean()),
                        mean_abs_dT_next=float(np.abs(dT).mean())))
    pd.DataFrame(det).to_csv(OUT / "paper1_p11_detector.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(pd.DataFrame(inp_rows).round(4).to_string(index=False))
    print(pd.DataFrame(srows).round(3).to_string(index=False))
    print(H[["family", "model", "set", "series", "n", "MAPE4_science", "MAPE4_operational", "d", "lo", "hi", "p_boot", "p_holm", "hi_unpaired_seeds", "share_events_abs_d_gt_1pt"]].round(4).to_string(index=False))
    print(Rl.round(4).to_string(index=False))
    print("text rule (forecasts hold with operational data): " + ("MET for both models" if Rl.rule_met_for_both.all() else "NOT met"))


if __name__ == "__main__":
    main()
