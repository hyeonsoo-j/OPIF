"""P9: models trained with the flares outside the isolation rule. Design fixed in docs/decisions.md (2026-11-19, patch_v15) before the
run; this script follows that entry. No training.

    python notebooks/analyze_p9.py --results results [--n-boot 10000]

Run f (0, 1) trained on the isolated training flares + the flares outside the rule of fold f (folds = 24-h time clusters, alternating).
  flares outside the rule   scored out-of-fold only: a flare of fold g is scored with run 1 - g (which never saw it)
  isolated test flares      scored with both runs; the score of a seed is the mean of the two runs' scores (not an ensemble of the two)
  baseline                  the P8 runs: same seeds, same events, trained on the isolated training flares only

  paper1_p9_hboot.csv     MAPE4 (all 180 min) P9 - P8 per input, model and set, hierarchical bootstrap over events (paired) and seeds,
                          Holm over the 8 comparisons (2 inputs x 2 models x {outside, test}); next to it, not in the family: the same
                          for the 0.1-0.8 nm band, for the 5-seed ensemble, and for the zenodo96 test set (FISM2 input)
  paper1_p9_inforce.csv   descriptive, minutes in force (declaration rule): minutes, MAPE4, 0.1-0.8 nm MAPE, 0.1-0.8 nm and 4-band coverage
                          of the 90 % intervals (calibrated on the validation flares of the same run), P8 and P9, per reason for failing the rule
  paper1_p9_groups.csv    descriptive, all 180 min: MAPE4 P8 and P9 and their difference by reason, class and band, and for the flares a
                          run was trained on (in-fold; a check that training on them changes the fit, not a result)
  paper1_p9_events.csv    per flare outside the rule: fold, reason, class, event MAPE4 of P8 and P9 (seed mean)
  paper1_p9_rule.csv      the text rule of the design entry, evaluated
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
rt, afs = ap7.rt, ap7.afs

MODELS, B = ap7.MODELS, ap7.B
LABEL = {"fism": "FISM2", "goes_convnn": "GOES + FCN converter"}
REASONS = ("later flare within 180 min", "earlier flare within 60 min", "both")


def p9_exp(inp: str, f: int) -> str:
    return f"p2_zscore_full{'' if inp == 'fism' else '_in-' + inp}_tx{f}_p9"


def ev_mape(L, y, bands=(0, 1, 2, 3)):
    """(S, n) event MAPE in percent over the bands and all 180 minutes. L (S, n, 180, 4) log10, y (n, 180, 4)."""
    bands = list(bands)
    return 100 * (np.abs(10 ** L[..., bands] - y[None][..., bands]) / y[None][..., bands]).mean(axis=(2, 3))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--events", default=str(ROOT / "data/processed/events_latis.npz"))
    ap.add_argument("--goes-post", default=str(ROOT / "data/processed/events_goes_post.npz"))
    ap.add_argument("--out", default=None, help="default: <results>/full_server")
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args(argv)
    R = Path(a.results)
    OUT = Path(a.out or R / "full_server")
    os.environ.update(FISM_PROTOCOL="full", FISM_EVENTS=str(Path(a.events).resolve()), FISM_GOES_POST=str(Path(a.goes_post).resolve()),
                      FISM_DEVICE_QUIET="1", FISM_EXCLUDE_NEAR_TEST_H="0", FISM_EXCLUDE_RANDOM_N="0", FISM_EXCLUDE_SAME_AR="0",
                      FISM_TRAIN_EXTRA_FOLD="")
    from src.data import pool

    pools = {}

    def get_pool(inp, extra=0):
        key = ("default" if inp == "fism" else "goes_b", extra)
        if key not in pools:
            os.environ.update(FISM_INPUT=key[0], FISM_ISSUE_DELAY="0", FISM_PREDICT_EXTRA=str(extra))
            pool.load_full.cache_clear()
            X, Y, names, meta, tr, va, te, teL = pool.load_full()
            pools[key] = dict(Y=Y, va=va, tr=tr, ex=pool.extra_index(), fold=pool.extra_fold() if extra else None)
        return pools[key]

    def preds(exp, m, pre):
        fs = sorted(glob.glob(f"{R}/{exp}/{pre}_{m}_s*.npy"))
        assert len(fs) == 5, f"{exp}: {len(fs)} seeds of {pre}_{m}"
        return np.stack([np.load(f, allow_pickle=False).astype(np.float64) for f in fs])

    def val(exp, m, P):
        vs = [np.load(f, allow_pickle=False) for f in sorted(glob.glob(f"{R}/{exp}/val_{m}_s*.npz"))]
        idx = vs[0]["idx"]
        assert len(vs) == 5 and all(np.array_equal(v["idx"], idx) for v in vs) and np.array_equal(np.sort(idx), np.sort(P["va"])), f"{exp}: validation rows"
        return np.stack([v["pred_log"].astype(np.float64) for v in vs]), np.log10(P["Y"][idx])

    d = np.load(a.events, allow_pickle=False)
    st = np.load(R / "full_server/peak_detector_streams.npz", allow_pickle=False)
    assert int(st["pre"]) == ap7.PRE and (st["event_id"] == d["event_id"][st["row"]]).all()
    srow = {int(r): j for j, r in enumerate(st["row"])}
    rowL = {int(d["test_k"][i]): int(i) for i in np.where(d["is_test"])[0]}
    CAT = pd.read_csv(ROOT / "data/processed/flare_catalogue.csv")
    P0 = get_pool("fism")
    YZ, YL = P0["Y"][:96], P0["Y"][96:192]
    t180 = np.arange(180)[None, :]

    hb, inf, grp, rule_rows, ev_tab = [], [], [], [], None
    rng = np.random.default_rng(61)
    for inp in ("goes_convnn", "fism"):
        Pv, Px = get_pool(inp), get_pool(inp, 1)
        ex_idx, ex_rows = Px["ex"]
        E8 = pd.read_csv(R / ap7.p8_exp(inp) / "extra_rows.csv")
        E9 = [pd.read_csv(R / p9_exp(inp, f) / "extra_rows.csv") for f in (0, 1)]
        fold = E9[0].fold.values
        assert np.array_equal(E8.dataset_row.values, ex_rows) and np.array_equal(E8.pool_index.values, ex_idx)
        for f in (0, 1):
            assert np.array_equal(E9[f].dataset_row.values, ex_rows) and np.array_equal(E9[f].fold.values, fold) and np.array_equal(E9[f].trained.values, fold == f)
        assert np.array_equal(fold, Px["fold"]), "folds of the result files differ from src/data/pool.py"
        # the two folds share no 24-h time cluster
        pk = pd.to_datetime(d["peak"][ex_rows].astype(str)).values.astype("datetime64[m]").astype(np.int64)
        assert (np.diff(pk) >= 0).all() and all((fold[i] == fold[i + 1]) or (pk[i + 1] - pk[i] >= 1440) for i in range(len(pk) - 1))
        oof = 1 - fold                                                           # the run that never saw the flare
        yx = d["Y"][ex_rows].astype(np.float64)
        c = CAT.iloc[d["row"][ex_rows]]
        assert (c.event_id.values == d["event_id"][ex_rows].astype(str)).all()
        reason = np.where((c.n_pre60.values > 0) & (c.n_post180.values > 0), "both",
                          np.where(c.n_pre60.values > 0, "earlier flare within 60 min", "later flare within 180 min"))
        is_x = np.char.startswith(d["cls"][ex_rows].astype(str), "X")
        hx = np.array([int(r) in srow for r in ex_rows])
        Tx = np.array([ap7.next_declaration(st["xrsb"][srow[int(r)]], 2, 2.0) if int(r) in srow else 180 for r in ex_rows])
        if ev_tab is None:
            ev_tab = {}
        tab = pd.DataFrame(dict(input=inp, dataset_row=ex_rows, event=d["event_id"][ex_rows].astype(str), cls=d["cls"][ex_rows].astype(str),
                                fold=fold, reason=reason, has_stream=hx, T_next=Tx))
        for m in MODELS:
            L8x, L8t, L8z = (preds(ap7.p8_exp(inp), m, p) for p in ("oofN", "oofL", "oof"))
            L9x, L9t, L9z = ([preds(p9_exp(inp, f), m, p) for f in (0, 1)] for p in ("oofN", "oofL", "oof"))
            okt = ~np.isnan(L8t).any(axis=(0, 2, 3))
            assert all(np.array_equal(okt, ~np.isnan(L9t[f]).any(axis=(0, 2, 3))) for f in (0, 1))
            okz = ~np.isnan(L8z).any(axis=(0, 2, 3))
            trow = np.array([rowL[int(k)] for k in np.where(okt)[0]])
            Tt = np.array([ap7.next_declaration(st["xrsb"][srow[int(r)]], 2, 2.0) for r in trow])
            yt, yz = YL[okt], YZ[okz]
            L8t, L9t = L8t[:, okt], [L[:, okt] for L in L9t]
            L8z, L9z = L8z[:, okz], [L[:, okz] for L in L9z]
            sel = lambda Ls: np.where((oof == 0)[None, :, None, None], Ls[0], Ls[1])                    # noqa: E731  out-of-fold
            L9o, L9i = sel(L9x), np.where((fold == 0)[None, :, None, None], L9x[0], L9x[1])             # out-of-fold, in-fold
            # ------------------------------------------------------------------ the 8 comparisons (all 180 min)
            fam = {"outside the rule (out-of-fold)": (ev_mape(L9o, yx), ev_mape(L8x, yx), ev_mape(L9o, yx, (0,)), ev_mape(L8x, yx, (0,)),
                                                      ev_mape(L9o.mean(0, keepdims=True), yx)[0], ev_mape(L8x.mean(0, keepdims=True), yx)[0]),
                   "test flares (isolated)": ((ev_mape(L9t[0], yt) + ev_mape(L9t[1], yt)) / 2, ev_mape(L8t, yt),
                                              (ev_mape(L9t[0], yt, (0,)) + ev_mape(L9t[1], yt, (0,))) / 2, ev_mape(L8t, yt, (0,)),
                                              (ev_mape(L9t[0].mean(0, keepdims=True), yt)[0] + ev_mape(L9t[1].mean(0, keepdims=True), yt)[0]) / 2,
                                              ev_mape(L8t.mean(0, keepdims=True), yt)[0])}
            for s, (A, C, A0, C0, Ae, Ce) in fam.items():
                dd, lo, hi, p = rt.hboot(A, C, n=a.n_boot, seed=11)
                d0, lo0, hi0, p0 = rt.hboot(A0, C0, n=a.n_boot, seed=12)
                de, loe, hie, pe = rt.hboot(Ae[None], Ce[None], n=a.n_boot, seed=13)
                hb.append(dict(family=True, input=inp, model=m, set=s, n=A.shape[1], MAPE4_P8=C.mean(), MAPE4_P9=A.mean(), d=dd, lo=lo, hi=hi, p_hboot=p,
                               share_events_better=float((A.mean(0) < C.mean(0)).mean()), median_event_d=float(np.median(A.mean(0) - C.mean(0))),
                               MAPE_w0_P8=C0.mean(), MAPE_w0_P9=A0.mean(), d_w0=d0, lo_w0=lo0, hi_w0=hi0, p_hboot_w0=p0,
                               ensMAPE4_P8=Ce.mean(), ensMAPE4_P9=Ae.mean(), d_ens=de, lo_ens=loe, hi_ens=hie, p_boot_ens=pe))
            if inp == "fism":                                                    # the authors' test arrays: not in the family
                A, C = (ev_mape(L9z[0], yz) + ev_mape(L9z[1], yz)) / 2, ev_mape(L8z, yz)
                dd, lo, hi, p = rt.hboot(A, C, n=a.n_boot, seed=14)
                Ae = (ev_mape(L9z[0].mean(0, keepdims=True), yz)[0] + ev_mape(L9z[1].mean(0, keepdims=True), yz)[0]) / 2
                Ce = ev_mape(L8z.mean(0, keepdims=True), yz)[0]
                hb.append(dict(family=False, input=inp, model=m, set="test flares, zenodo96 (authors' arrays)", n=A.shape[1], MAPE4_P8=C.mean(), MAPE4_P9=A.mean(),
                               d=dd, lo=lo, hi=hi, p_hboot=p, share_events_better=float((A.mean(0) < C.mean(0)).mean()),
                               median_event_d=float(np.median(A.mean(0) - C.mean(0))), ensMAPE4_P8=Ce.mean(), ensMAPE4_P9=Ae.mean()))
            A9, A8 = fam["outside the rule (out-of-fold)"][:2]
            tab[f"MAPE4_P8_{m}"], tab[f"MAPE4_P9_{m}"] = A8.mean(0), A9.mean(0)
            # ------------------------------------------------------------------ descriptive groups (all 180 min)
            Ai = ev_mape(L9i, yx)
            gm = {"all": np.ones(len(ex_rows), bool), **{g: reason == g for g in REASONS}, "M class": ~is_x, "X class": is_x,
                  "fold 0 (scored with run 1)": fold == 0, "fold 1 (scored with run 0)": fold == 1}
            for g, mk in gm.items():
                grp.append(dict(input=inp, model=m, group=g, n=int(mk.sum()), MAPE4_P8=A8[:, mk].mean(), MAPE4_P9_out_of_fold=A9[:, mk].mean(),
                                d=A9[:, mk].mean() - A8[:, mk].mean(), MAPE4_P9_in_fold=Ai[:, mk].mean(),
                                **{f"MAPE_{b}_P8": ev_mape(L8x[:, mk], yx[mk], (j,)).mean() for j, b in enumerate(B)},
                                **{f"MAPE_{b}_P9": ev_mape(L9o[:, mk], yx[mk], (j,)).mean() for j, b in enumerate(B)}))
            # ------------------------------------------------------------------ minutes in force (declaration rule), intervals from the validation flares
            Lv8, yv = val(ap7.p8_exp(inp), m, Pv)
            Lv9 = [val(p9_exp(inp, f), m, Pv)[0] for f in (0, 1)]
            lyx, lyt = np.log10(yx), np.log10(yt)

            def inside(Lv, Lq, ly):
                _, lo_, hi_ = ap7.conf_norm(Lv, yv, Lq)
                return (ly >= lo_) & (ly <= hi_), hi_ - lo_

            rel = lambda L, y: 100 * np.abs(10 ** L - y[None]) / y[None]                                # noqa: E731
            i8x, w8x = inside(Lv8, L8x, lyx)
            i9f = [inside(Lv9[f], L9x[f], lyx) for f in (0, 1)]
            i9x = np.where((oof == 0)[:, None, None], i9f[0][0], i9f[1][0])
            w9x = np.where((oof == 0)[:, None, None], i9f[0][1], i9f[1][1])
            i8t, w8t = inside(Lv8, L8t, lyt)
            i9tf = [inside(Lv9[f], L9t[f], lyt) for f in (0, 1)]
            i9t, w9t = (i9tf[0][0].astype(float) + i9tf[1][0]) / 2, (i9tf[0][1] + i9tf[1][1]) / 2
            sets = {"P8 (isolated training flares only)": (rel(L8x, yx), i8x, w8x, rel(L8t, yt), i8t, w8t),
                    "P9 (with the flares outside the rule)": (rel(L9o, yx), i9x, w9x, (rel(L9t[0], yt) + rel(L9t[1], yt)) / 2, i9t, w9t)}
            for tr_name, (rx, ix, wx, rtst, itst, wt) in sets.items():
                gx = {"outside the rule: all": hx, **{f"outside the rule: {g}": hx & (reason == g) for g in REASONS}}
                res = {"test flares (isolated)": ap7.in_force(rtst, itst, Tt, True, rng, 2000),
                       **{g: ap7.in_force(rx[:, mk], ix[mk], Tx[mk], True, rng, 2000) for g, mk in gx.items()}}
                wid = {"test flares (isolated)": float((wt[:, :, 0] * (t180 < Tt[:, None])).sum() / (t180 < Tt[:, None]).sum()),
                       **{g: float((wx[mk][:, :, 0] * (t180 < Tx[mk][:, None])).sum() / (t180 < Tx[mk][:, None]).sum()) for g, mk in gx.items()}}
                pt, px = res["test flares (isolated)"], res["outside the rule: all"]
                comb = lambda u, v: (1 - ap7.P_OUTSIDE) * u + ap7.P_OUTSIDE * v                          # noqa: E731
                res["all flares (population-weighted)"] = ({k: comb(pt[0][k], px[0][k]) for k in pt[0]}, {k: comb(pt[1][k], px[1][k]) for k in pt[1]})
                wid["all flares (population-weighted)"] = np.nan
                for g, (pv, bv) in res.items():
                    n = len(Tt) + int(hx.sum()) if g.startswith("all flares") else (len(Tt) if g.startswith("test") else int(gx[g].sum()))
                    rec = dict(input=inp, model=m, training=tr_name, group=g, n=n, share_minutes_in_force=pv["den"] / 180, MAPE4=pv["MAPE4"] / pv["den"],
                               MAPE_w0=pv["MAPE_w0"] / pv["den"], cov_w0=pv["cov_w0"] / pv["den"], cov_4band=pv["cov_4band"] / pv["den"], width_w0_dex=wid[g])
                    for k in ("MAPE4", "cov_w0"):
                        rec[f"{k}_lo"], rec[f"{k}_hi"] = (float(v) for v in np.percentile(bv[k] / bv["den"], [2.5, 97.5]))
                    inf.append(rec)
        ev_tab[inp] = tab

    H = pd.DataFrame(hb)
    fam = H.family.values
    assert fam.sum() == 8
    H["p_holm"] = np.nan
    H.loc[fam, "p_holm"] = afs.holm(H.loc[fam, "p_hboot"].values)
    H.to_csv(OUT / "paper1_p9_hboot.csv", index=False)
    pd.DataFrame(inf).to_csv(OUT / "paper1_p9_inforce.csv", index=False)
    pd.DataFrame(grp).to_csv(OUT / "paper1_p9_groups.csv", index=False)
    pd.concat(ev_tab.values()).to_csv(OUT / "paper1_p9_events.csv", index=False)
    # ---------------------------------------------------------------------- the text rule
    op = H[fam & (H.input == "goes_convnn")]
    for m in MODELS:
        o = op[(op.model == m) & op.set.str.startswith("outside")].iloc[0]
        t = op[(op.model == m) & op.set.str.startswith("test")].iloc[0]
        rule_rows.append(dict(model=f"operational {m}", d_outside=o.d, lo_outside=o.lo, hi_outside=o.hi, p_holm_outside=o.p_holm,
                              outside_falls_adj_p_lt_0p05=bool(o.d < 0 and o.p_holm < 0.05), d_test=t.d, lo_test=t.lo, hi_test=t.hi,
                              test_interval_not_entirely_above_0=bool(t.lo <= 0)))
    Rl = pd.DataFrame(rule_rows)
    Rl["model_passes"] = Rl.outside_falls_adj_p_lt_0p05 & Rl.test_interval_not_entirely_above_0
    Rl["rule_met_for_both"] = bool(Rl.model_passes.all())
    Rl.to_csv(OUT / "paper1_p9_rule.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(H[["family", "input", "model", "set", "n", "MAPE4_P8", "MAPE4_P9", "d", "lo", "hi", "p_hboot", "p_holm", "share_events_better"]].round(4).to_string(index=False))
    print(Rl.round(4).to_string(index=False))
    print("text rule (training on all flares helps): " + ("MET for both operational models" if Rl.rule_met_for_both.all() else "NOT met"))


if __name__ == "__main__":
    main()
