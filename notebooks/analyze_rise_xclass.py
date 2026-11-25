"""Two analyses on the arrays of the patch_v14 run, fixed in docs/decisions.md (2026-11-19) before computing. No training.

    python notebooks/analyze_rise_xclass.py --results results [--n-boot 2000]

(2) Forecast ended at the onset of the next rise. Onset O_r = first minute t >= 1 after the peak at which the raw XRS-B stream
    is >= r times its running minimum since the peak (causal). The forecast is in force on minutes 0 .. min(O_r, T) - 1, T = minute of
    the next declaration (armed detector, the rule of 2026-11-12).
      paper1_onset.csv            per input, model, rule (declaration / onset r) and group: flares ended by an onset, minutes in force,
                                  MAPE and 0.1-0.8 nm coverage in force (95 % interval)
      paper1_onset_discarded.csv  the minutes that the onset rule gives up (between the onset and T): how many, and how well the
                                  old forecast covered them, for flares with and without a declared next flare
      paper1_onset_events.csv     per flare: T and the onset minutes
(3) X-class intervals, minutes in force under the declaration rule.
      paper1_xclass_where.csv     current intervals (M0): coverage by class and 30-min block / peak-class bin, misses above and below,
                                  half-width against absolute error
      paper1_xclass_methods.csv   calibrations M0 (current), M1 (class-conditional, validation flares), M2 (peak-localized, h = 0.3;
                                  0.2 and 0.5), M3 (class-conditional, validation flares + flares outside the rule, out-of-fold): coverage
                                  by class and group, class deficit (M - X) with event-bootstrap intervals, interval width
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

PRE, BLOCKS, MODELS = ap7.PRE, ap7.BLOCKS, ap7.MODELS
RISES = (2.0, 1.5, 1.2)
# (name, rise factor, flux floor): the three pre-specified onset rules, then the floor variants added after the first result (descriptive)
ONSETS = [(f"onset r={r:g}", r, 0.0) for r in RISES] + [(f"onset r=2, flux >= {lab}", 2.0, fl) for lab, fl in (("C3", 3e-6), ("C5", 5e-6), ("M1", 1e-5))]
LABEL = {"fism": "FISM2", "goes_convnn": "GOES + FCN converter"}


def onset(b, r, floor=0.0, pre=PRE):
    """First minute t >= 1 after the peak (stream index pre) with b_t >= r * min(b_0 .. b_t) and b_t >= floor; 180 if there is none."""
    x = np.asarray(b[pre:pre + 180], float)
    fin = np.isfinite(x)
    m = np.minimum.accumulate(np.where(fin, x, np.inf))
    hit = np.where(fin & (x >= r * m) & (x >= floor) & (np.arange(len(x)) >= 1))[0]
    return int(hit[0]) if len(hit) else 180


def cal_intervals(mu_c, sd_c, y_c, mask_c, w_c, mu_t, sd_t, level=0.9):
    """conf-norm interval with a minute mask and event weights on the calibration flares.
    mu_c, sd_c, y_c (n, 180, 4) log10; mask_c (n, 180) usable minutes; w_c (n,); mu_t, sd_t (m, 180, 4). Returns lo, hi (m, 180, 4).
    With an all-true mask and equal weights this is analyze_p7_p8.conf_norm()."""
    lo, hi = np.zeros_like(mu_t), np.zeros_like(mu_t)
    for b in range(4):
        for bl in BLOCKS:
            mk = mask_c[:, bl]
            s0 = np.median(sd_c[:, bl, b][mk])
            sc = (np.abs(y_c[:, bl, b] - mu_c[:, bl, b]) / (sd_c[:, bl, b] + s0))[mk]
            w = np.broadcast_to(w_c[:, None], mk.shape)[mk]
            half = ap7.wq(sc, w, level) * (sd_t[:, bl, b] + s0)
            lo[:, bl, b], hi[:, bl, b] = mu_t[:, bl, b] - half, mu_t[:, bl, b] + half
    return lo, hi


def pooled_cov(ins, V):
    """Coverage pooled over the event-minutes of the mask. ins (n, 180), V (n, 180)."""
    return float((ins & V).sum() / max(V.sum(), 1))


def boot_cov(ins, V, rng, n_boot):
    num, den = (ins & V).sum(1).astype(float), V.sum(1).astype(float)
    idx = rng.integers(0, len(den), (n_boot, len(den)))
    return num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--events", default=str(ROOT / "data/processed/events_latis.npz"))
    ap.add_argument("--goes-post", default=str(ROOT / "data/processed/events_goes_post.npz"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    R = Path(a.results)
    OUT = Path(a.out or R / "full_server")
    os.environ.update(FISM_PROTOCOL="full", FISM_EVENTS=str(Path(a.events).resolve()), FISM_GOES_POST=str(Path(a.goes_post).resolve()),
                      FISM_DEVICE_QUIET="1", FISM_EXCLUDE_NEAR_TEST_H="0", FISM_EXCLUDE_RANDOM_N="0", FISM_EXCLUDE_SAME_AR="0")
    from src.data import pool

    pools = {}

    def get_pool(inp, extra=0):
        key = ("default" if inp == "fism" else "goes_b", 0, extra)
        if key not in pools:
            os.environ.update(FISM_INPUT=key[0], FISM_ISSUE_DELAY="0", FISM_PREDICT_EXTRA=str(extra))
            pool.load_full.cache_clear()
            X, Y, names, meta, tr, va, te, teL = pool.load_full()
            pools[key] = dict(X=X, Y=Y, meta=meta, va=va, ex=pool.extra_index())
        return pools[key]

    def preds(exp, m, pre):
        fs = sorted(glob.glob(f"{R}/{exp}/{pre}_{m}_s*.npy"))
        return np.stack([np.load(f, allow_pickle=False).astype(np.float64) for f in fs])

    d = np.load(a.events, allow_pickle=False)
    st = np.load(R / "full_server/peak_detector_streams.npz", allow_pickle=False)
    assert int(st["pre"]) == PRE and (st["event_id"] == d["event_id"][st["row"]]).all()
    srow = {int(r): j for j, r in enumerate(st["row"])}
    rowL = {int(d["test_k"][i]): int(i) for i in np.where(d["is_test"])[0]}
    EV = pd.read_csv(R / "full_server/paper1_p8_reissue_events.csv").set_index("dataset_row")
    CAT = pd.read_csv(ROOT / "data/processed/flare_catalogue.csv")
    YL = get_pool("fism")["Y"][96:192]
    isx_d = np.char.startswith(d["cls"].astype(str), "X")
    peak_min = pd.to_datetime(d["peak"].astype(str)).values.astype("datetime64[m]").astype(np.int64)

    on_rows, dis_rows, ev_rows, wh_rows, me_rows = [], [], [], [], []
    rng = np.random.default_rng(51)
    for inp in ("goes_convnn", "fism"):
        exp = ap7.p8_exp(inp)
        Pv, Px = get_pool(inp), get_pool(inp, 1)
        ex_idx, ex_rows = Px["ex"]
        E = pd.read_csv(R / exp / "extra_rows.csv")
        assert np.array_equal(E.dataset_row.values, ex_rows)
        yx = d["Y"][ex_rows].astype(np.float64)
        for m in MODELS:
            Lx, Lt = preds(exp, m, "oofN"), preds(exp, m, "oofL")
            okt = ~np.isnan(Lt).any(axis=(0, 2, 3))
            tk = np.where(okt)[0]
            trow = np.array([rowL[int(k)] for k in tk])
            Lt, yt = Lt[:, okt], YL[okt]
            fs = sorted(glob.glob(f"{R}/{exp}/val_{m}_s*.npz"))
            vs = [np.load(f, allow_pickle=False) for f in fs]
            vidx = vs[0]["idx"]
            assert all(np.array_equal(v["idx"], vidx) for v in vs) and np.array_equal(np.sort(vidx), np.sort(Pv["va"]))
            Lv, yv = np.stack([v["pred_log"].astype(np.float64) for v in vs]), np.log10(Pv["Y"][vidx])
            mu_v, sd_v = Lv.mean(0), Lv.std(0)
            mu_x, sd_x, mu_t, sd_t = Lx.mean(0), Lx.std(0), Lt.mean(0), Lt.std(0)
            lyx, lyt = np.log10(yx), np.log10(yt)
            allv = np.ones((len(vidx), 180), bool)
            lo_x, hi_x = cal_intervals(mu_v, sd_v, yv, allv, np.ones(len(vidx)), mu_x, sd_x)
            lo_t, hi_t = cal_intervals(mu_v, sd_v, yv, allv, np.ones(len(vidx)), mu_t, sd_t)
            _, lo_ref, hi_ref = ap7.conf_norm(Lv, yv, Lx)
            assert np.abs(lo_x - lo_ref).max() < 1e-12 and np.abs(hi_x - hi_ref).max() < 1e-12, "cal_intervals() differs from conf_norm()"
            ins_x, ins_t = (lyx >= lo_x) & (lyx <= hi_x), (lyt >= lo_t) & (lyt <= hi_t)
            relx = 100 * np.abs(10 ** Lx - yx[None]) / yx[None]
            relt = 100 * np.abs(10 ** Lt - yt[None]) / yt[None]
            # streams: next declaration (as in the 2026-11-12 table) and onset minutes
            hx = np.array([int(r) in srow for r in ex_rows])
            ht = np.array([int(r) in srow for r in trow])
            assert ht.all(), "a test flare has no stream"
            Tx = np.array([ap7.next_declaration(st["xrsb"][srow[int(r)]], 2, 2.0) if int(r) in srow else -1 for r in ex_rows])
            Tt = np.array([ap7.next_declaration(st["xrsb"][srow[int(r)]], 2, 2.0) for r in trow])
            assert all(Tx[i] == EV.T_next_r2[int(r)] for i, r in enumerate(ex_rows) if hx[i] and int(r) in EV.index) and all(Tt[i] == EV.T_next_r2[int(r)] for i, r in enumerate(trow) if int(r) in EV.index)
            Ox = {nm: np.array([onset(st["xrsb"][srow[int(q)]], r, fl) if int(q) in srow else -1 for q in ex_rows]) for nm, r, fl in ONSETS}
            Ot = {nm: np.array([onset(st["xrsb"][srow[int(q)]], r, fl) for q in trow]) for nm, r, fl in ONSETS}
            c = CAT.iloc[d["row"][ex_rows]]
            assert (c.event_id.values == d["event_id"][ex_rows].astype(str)).all()
            reason = np.where((c.n_pre60.values > 0) & (c.n_post180.values > 0), "both",
                              np.where(c.n_pre60.values > 0, "earlier flare within 60 min", "later flare within 180 min"))
            assert all(reason[i] == EV.reason[int(r)] for i, r in enumerate(ex_rows) if int(r) in EV.index)
            groups = {"outside the rule: all": hx, **{f"outside the rule: {g}": hx & (reason == g) for g in ("later flare within 180 min", "earlier flare within 60 min", "both")}}
            if inp == "goes_convnn" and m == MODELS[0]:
                for grp, rows_, T_, O_ in (("test flares (isolated)", trow, Tt, Ot), ("outside the rule", ex_rows[hx], Tx[hx], {nm: Ox[nm][hx] for nm, _, _ in ONSETS})):
                    for i, r_ in enumerate(rows_):
                        ev_rows.append(dict(group=grp, dataset_row=int(r_), event=str(d["event_id"][r_]), cls=str(d["cls"][r_]), T_next=int(T_[i]),
                                            **{nm.replace("onset ", "onset_").replace(", flux >= ", "_floor").replace("r=", "r"): int(O_[nm][i]) for nm, _, _ in ONSETS}))
            # ---------------------------------------------------------------- (2) onset rule
            rules = [("declaration", None)] + [(nm, nm) for nm, _, _ in ONSETS]
            for rname, r in rules:
                Et = Tt if r is None else np.minimum(Ot[r], Tt)
                Ex = Tx if r is None else np.minimum(Ox[r], Tx)
                sets = {"test flares (isolated)": (relt, ins_t, Et, Tt), **{g: (relx[:, mk], ins_x[mk], Ex[mk], Tx[mk]) for g, mk in groups.items()}}
                res = {g: ap7.in_force(r_, i_, e_, True, rng, a.n_boot) for g, (r_, i_, e_, _) in sets.items()}
                pt, px = res["test flares (isolated)"], res["outside the rule: all"]
                comb = lambda u, v: (1 - ap7.P_OUTSIDE) * u + ap7.P_OUTSIDE * v          # noqa: E731
                res["all flares (population-weighted)"] = ({k: comb(pt[0][k], px[0][k]) for k in pt[0]}, {k: comb(pt[1][k], px[1][k]) for k in pt[1]})
                for g, (pv, bv) in res.items():
                    if g.startswith("all flares"):
                        n, ended = len(Et) + int(hx.sum()), comb(float((Et < Tt).mean()), float((Ex[hx] < Tx[hx]).mean()))
                    else:
                        e_, T_ = sets[g][2], sets[g][3]
                        n, ended = len(e_), float((e_ < T_).mean())
                    rec = dict(input=inp, model=m, rule=rname, group=g, n=n, share_ended_by_onset=ended, share_minutes_in_force=pv["den"] / 180,
                               MAPE4=pv["MAPE4"] / pv["den"], MAPE_w0=pv["MAPE_w0"] / pv["den"], cov_w0=pv["cov_w0"] / pv["den"], cov_4band=pv["cov_4band"] / pv["den"])
                    for k in ("MAPE4", "cov_w0"):
                        rec[f"{k}_lo"], rec[f"{k}_hi"] = (float(v) for v in np.percentile(bv[k] / bv["den"], [2.5, 97.5]))
                    on_rows.append(rec)
                if r is not None:                       # the minutes given up by the onset rule: [E, T)
                    for g in ("test flares (isolated)", "outside the rule: all"):
                        r_, i_, e_, T_ = sets[g]
                        for part, mk in (("next flare declared within 180 min", T_ < 180), ("no declaration within 180 min", T_ == 180), ("all", np.ones(len(T_), bool))):
                            c = mk & (e_ < T_)
                            if c.sum() == 0:
                                continue
                            V = (np.arange(180)[None, :] >= e_[c][:, None]) & (np.arange(180)[None, :] < T_[c][:, None])
                            dis_rows.append(dict(input=inp, model=m, rule=rname, group=g, part=part, n_flares=int(mk.sum()), n_ended=int(c.sum()),
                                                 discarded_min_per_flare=float(V.sum() / mk.sum()), cov_w0_discarded=pooled_cov(i_[c][:, :, 0], V),
                                                 MAPE_w0_discarded=float((r_[:, c][:, :, :, 0] * V[None]).sum() / (V.sum() * r_.shape[0]))))
            # ---------------------------------------------------------------- (3) X class, minutes in force (declaration rule)
            t180 = np.arange(180)[None, :]
            Vx, Vt = (t180 < Tx[:, None]) & hx[:, None], t180 < Tt[:, None]
            xx, xt = isx_d[ex_rows], isx_d[trow]
            pk_v, pk_x, pk_t = (np.log10(Pv["X"][vidx, -1]), np.log10(Px["X"][ex_idx, -1]), np.log10(get_pool(inp)["X"][96:192][okt][:, -1]))
            xv = Pv["meta"].is_x.values[vidx].astype(bool)
            # (A) where the current intervals miss
            Lall = dict(mu=np.concatenate([mu_t, mu_x]), lo=np.concatenate([lo_t, lo_x]), hi=np.concatenate([hi_t, hi_x]), y=np.concatenate([lyt, lyx]),
                        V=np.concatenate([Vt, Vx]), isx=np.concatenate([xt, xx]), grp=np.array(["test"] * len(xt) + ["outside"] * len(xx)),
                        cls=np.concatenate([d["cls"][trow], d["cls"][ex_rows]]).astype(str))
            mag = np.array([{"M": 1.0, "X": 10.0}[c[0]] * float(c[1:]) for c in Lall["cls"]])
            y0, lo0, hi0, mu0, V0 = Lall["y"][:, :, 0], Lall["lo"][:, :, 0], Lall["hi"][:, :, 0], Lall["mu"][:, :, 0], Lall["V"]
            ins0 = (y0 >= lo0) & (y0 <= hi0)
            Vfull = np.concatenate([np.ones_like(Vt), np.broadcast_to(hx[:, None], Vx.shape)])

            def where(table, key, sel, Vsel):
                n_min = int(Vsel.sum())
                if n_min == 0:
                    return
                bt = boot_cov(ins0[sel], Vsel, rng, a.n_boot) if table != "block" else np.array([np.nan, np.nan])
                wh_rows.append(dict(input=inp, model=m, table=table, key=key, n_events=int((Vsel.sum(1) > 0).sum()), n_minutes=n_min,
                                    cov_w0=float((ins0[sel] & Vsel).sum() / n_min), cov_w0_lo=float(np.nanpercentile(bt, 2.5)) if table != "block" else np.nan,
                                    cov_w0_hi=float(np.nanpercentile(bt, 97.5)) if table != "block" else np.nan, miss_above=float(((y0[sel] > hi0[sel]) & Vsel).sum() / n_min),
                                    miss_below=float(((y0[sel] < lo0[sel]) & Vsel).sum() / n_min),
                                    mean_halfwidth_dex=float((((hi0 - lo0)[sel] / 2) * Vsel).sum() / n_min), mean_abs_err_dex=float((np.abs(y0 - mu0)[sel] * Vsel).sum() / n_min)))
            for cname, cm in (("M", ~Lall["isx"]), ("X", Lall["isx"])):
                for gname, gm in (("pooled", np.ones(len(cm), bool)), ("test", Lall["grp"] == "test"), ("outside", Lall["grp"] == "outside")):
                    where("class", f"{cname} | {gname}", cm & gm, V0[cm & gm])
                    where("class, all 180 min", f"{cname} | {gname}", cm & gm, Vfull[cm & gm])
                for bi, bl in enumerate(BLOCKS):
                    Vb = V0[cm].copy(); Vb[:, :bl.start] = False; Vb[:, bl.stop:] = False
                    where("block", f"{cname} | {bl.start}-{bl.stop} min", cm, Vb)
            for lab, a0, a1 in (("M1-M2", 1, 2), ("M2-M5", 2, 5), ("M5-X1", 5, 10), (">=X1", 10, 1e9)):
                sel = (mag >= a0) & (mag < a1)
                where("peak class", lab, sel, V0[sel])
            # (B) calibrations
            def evaluate(method, h, lo_te, hi_te, lo_ex, hi_ex):
                it, ix_ = (lyt >= lo_te) & (lyt <= hi_te), (lyx >= lo_ex) & (lyx <= hi_ex)
                ins_all = np.concatenate([it, ix_])
                wid = np.concatenate([hi_te - lo_te, hi_ex - lo_ex])[:, :, 0]
                out = {}
                for cname, cm in (("M", ~Lall["isx"]), ("X", Lall["isx"])):
                    for gname, gm in (("pooled", np.ones(len(cm), bool)), ("test", Lall["grp"] == "test"), ("outside", Lall["grp"] == "outside")):
                        s = cm & gm
                        Vs = V0[s]
                        bt = boot_cov(ins_all[s][:, :, 0], Vs, rng, a.n_boot)
                        out[(cname, gname)] = bt
                        me_rows.append(dict(input=inp, model=m, method=method, h=h, group=gname, cls=cname, n_events=int((Vs.sum(1) > 0).sum()), n_minutes=int(Vs.sum()),
                                            cov_w0=pooled_cov(ins_all[s][:, :, 0], Vs), cov_w0_lo=float(np.percentile(bt, 2.5)), cov_w0_hi=float(np.percentile(bt, 97.5)),
                                            cov_4band=float((ins_all[s] & Vs[:, :, None]).sum() / (4 * Vs.sum())), width_w0_dex=float((wid[s] * Vs).sum() / Vs.sum())))
                for gname in ("pooled", "test", "outside"):
                    cM, cX = [r for r in me_rows[-6:] if r["group"] == gname and r["cls"] == "M"][0], [r for r in me_rows[-6:] if r["group"] == gname and r["cls"] == "X"][0]
                    dd = out[("M", gname)] - out[("X", gname)]
                    me_rows.append(dict(input=inp, model=m, method=method, h=h, group=gname, cls="deficit M-X", n_events=cX["n_events"], n_minutes=cX["n_minutes"],
                                        cov_w0=cM["cov_w0"] - cX["cov_w0"], cov_w0_lo=float(np.percentile(dd, 2.5)), cov_w0_hi=float(np.percentile(dd, 97.5))))
                for lab, a0, a1 in (("M1-M2", 1, 2), ("M2-M5", 2, 5), ("M5-X1", 5, 10), (">=X1", 10, 1e9)):       # added after the first result (descriptive)
                    s = (mag >= a0) & (mag < a1)
                    bt = boot_cov(ins_all[s][:, :, 0], V0[s], rng, a.n_boot)
                    me_rows.append(dict(input=inp, model=m, method=method, h=h, group="pooled", cls="peak " + lab, n_events=int((V0[s].sum(1) > 0).sum()), n_minutes=int(V0[s].sum()),
                                        cov_w0=pooled_cov(ins_all[s][:, :, 0], V0[s]), cov_w0_lo=float(np.percentile(bt, 2.5)), cov_w0_hi=float(np.percentile(bt, 97.5)),
                                        width_w0_dex=float((wid[s] * V0[s]).sum() / V0[s].sum())))
            evaluate("M0 current", np.nan, lo_t, hi_t, lo_x, hi_x)
            # M1 class-conditional on the validation flares
            lo1t, hi1t, lo1x, hi1x = lo_t.copy(), hi_t.copy(), lo_x.copy(), hi_x.copy()
            for cflag in (False, True):
                cv = xv == cflag
                for (lo_, hi_, mu_, sd_, sel) in ((lo1t, hi1t, mu_t, sd_t, xt == cflag), (lo1x, hi1x, mu_x, sd_x, xx == cflag)):
                    if sel.any():
                        lo_[sel], hi_[sel] = cal_intervals(mu_v[cv], sd_v[cv], yv[cv], allv[cv], np.ones(int(cv.sum())), mu_[sel], sd_[sel])
            evaluate("M1 class-conditional (validation)", np.nan, lo1t, hi1t, lo1x, hi1x)
            # M2 peak-localized
            for h in (0.3, 0.2, 0.5):
                lo2t, hi2t, lo2x, hi2x = np.zeros_like(mu_t), np.zeros_like(mu_t), np.zeros_like(mu_x), np.zeros_like(mu_x)
                for (lo_, hi_, mu_, sd_, pk) in ((lo2t, hi2t, mu_t, sd_t, pk_t), (lo2x, hi2x, mu_x, sd_x, pk_x)):
                    for e in range(len(pk)):
                        w = np.exp(-0.5 * ((pk_v - pk[e]) / h) ** 2)
                        lo_[e:e + 1], hi_[e:e + 1] = cal_intervals(mu_v, sd_v, yv, allv, w, mu_[e:e + 1], sd_[e:e + 1])
                evaluate("M2 peak-localized", h, lo2t, hi2t, lo2x, hi2x)
            # M3 class-conditional, validation flares + flares outside the rule (minutes in force), out-of-fold for the latter.
            # Event weights: the flares outside the rule get, within each class, their share among all flares of that class.
            fold = np.full(len(ex_rows), -1)
            fold[hx] = ap7.time_clusters(peak_min[ex_rows[hx]]) % 2
            lo3t, hi3t, lo3x, hi3x = lo_t.copy(), hi_t.copy(), lo_x.copy(), hi_x.copy()
            for cflag in (False, True):
                cv = xv == cflag
                p_out = float(((~d["selected"]) & (~d["is_test"]) & (isx_d == cflag)).sum() / (isx_d == cflag).sum())

                def fit(use_x, mu_, sd_):
                    n_v, n_o = int(cv.sum()), int(use_x.sum())
                    w = np.concatenate([np.full(n_v, (1 - p_out) / n_v), np.full(n_o, p_out / n_o)])
                    return cal_intervals(np.concatenate([mu_v[cv], mu_x[use_x]]), np.concatenate([sd_v[cv], sd_x[use_x]]), np.concatenate([yv[cv], lyx[use_x]]),
                                         np.concatenate([allv[cv], Vx[use_x]]), w, mu_, sd_)
                cx_ = hx & (xx == cflag)
                if (xt == cflag).any():
                    lo3t[xt == cflag], hi3t[xt == cflag] = fit(cx_, mu_t[xt == cflag], sd_t[xt == cflag])
                for f in (0, 1):
                    te = cx_ & (fold == f)
                    if te.any():
                        lo3x[te], hi3x[te] = fit(cx_ & (fold == 1 - f), mu_x[te], sd_x[te])
            evaluate("M3 class-conditional (validation + outside, out-of-fold)", np.nan, lo3t, hi3t, lo3x, hi3x)
            print(f"[done] {LABEL[inp]} {m}: test {len(trow)} (X {int(xt.sum())}), outside {int(hx.sum())} (X {int((xx & hx).sum())}), validation X {int(xv.sum())}", flush=True)

    On, Di, Wh, Me = pd.DataFrame(on_rows), pd.DataFrame(dis_rows), pd.DataFrame(wh_rows), pd.DataFrame(me_rows)
    base = Me[Me.method == "M0 current"].set_index(["input", "model", "group", "cls"]).width_w0_dex
    Me["width_ratio_vs_M0"] = [r.width_w0_dex / base.get((r.input, r.model, r.group, r.cls), np.nan) if r.cls != "deficit M-X" else np.nan for r in Me.itertuples()]
    On.to_csv(OUT / "paper1_onset.csv", index=False); Di.to_csv(OUT / "paper1_onset_discarded.csv", index=False)
    pd.DataFrame(ev_rows).to_csv(OUT / "paper1_onset_events.csv", index=False)
    Wh.to_csv(OUT / "paper1_xclass_where.csv", index=False); Me.to_csv(OUT / "paper1_xclass_methods.csv", index=False)
    print(f"paper1_onset.csv {len(On)} rows, paper1_onset_discarded.csv {len(Di)}, paper1_onset_events.csv {len(ev_rows)}, paper1_xclass_where.csv {len(Wh)}, paper1_xclass_methods.csv {len(Me)}")
    bad = On[(On.cov_w0 < On.cov_w0_lo - 1e-9) | (On.cov_w0 > On.cov_w0_hi + 1e-9)]
    assert not len(bad), "an interval does not contain its point value"
    return On, Di, Wh, Me


if __name__ == "__main__":
    main()
