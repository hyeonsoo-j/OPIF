"""Control experiment 1 (pre-specified in docs/decisions.md, 2026-11-25): GOES input vs FISM2 input with the SAME train/val
assignment. Reads the 5-architecture x 5-seed test forecasts (oofL) of
    FISM2 input, GOES assignment  results/p2_zscore_full_splitgoes_c1      (run.py --split-like-goes --tag c1)
    FISM2 input, own assignment   results/p2_zscore_full_gpu5               (reference, as published)
    GOES XRS-B direct             results/p2_zscore_full_in-goes_b_p1
    GOES + FCN converter          results/p2_zscore_full_in-goes_convnn_p2nn
and scores the rebuilt test flares with a GOES window (95). Family (Holm, 2): GOES direct - FISM2(c1), FCN converter - FISM2(c1);
hierarchical bootstrap (flares paired, seeds of every model resampled independently), 10,000 replicates. Decision rule (fixed before
the run): if the new "GOES direct - FISM2" stays within +-0.3 points of +1.96 the text keeps its sentence and only the numbers change;
otherwise both values are reported and "about 2 points" is revised. Also re-calibrates the benchmark-path conformal interval on the
GOES validation flares (now shared) and reports its 0.1-0.8 nm coverage.
    python notebooks/control1_analysis.py --results results [--events data/processed/events_latis.npz]
"""
import argparse, glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
ARCH = ("FCN-z", "GRU-z", "LSTM-z", "RNN-z", "Transformer-z")
EXP = {"FISM2_c1": "p2_zscore_full_splitgoes_c1", "FISM2_own": "p2_zscore_full_gpu5", "GOES": "p2_zscore_full_in-goes_b_p1",
       "FCNconv": "p2_zscore_full_in-goes_convnn_p2nn"}


def oof(R, exp, m):
    fs = sorted(glob.glob(f"{R}/{exp}/oofL_{m}_s*.npy"))
    if len(fs) != 5:
        raise SystemExit(f"{exp}: expected 5 seeds of {m}, found {len(fs)}")
    return np.stack([np.load(f, allow_pickle=False).astype(float) for f in fs])


def mape4(L, y):
    return 100 * np.mean(np.abs(10 ** L - y) / y, axis=(-2, -1))


def hboot(pairs, B=10000, seed=2026):
    rng = np.random.default_rng(seed); n = pairs[0][0].shape[1]
    W = rng.multinomial(n, np.full(n, 1 / n), size=B) / n; out = np.zeros(B)
    for a, b in pairs:
        Ca = rng.multinomial(a.shape[0], np.full(a.shape[0], 1 / a.shape[0]), size=B) / a.shape[0]
        Cb = rng.multinomial(b.shape[0], np.full(b.shape[0], 1 / b.shape[0]), size=B) / b.shape[0]
        out += ((W @ a.T) * Ca).sum(1) - ((W @ b.T) * Cb).sum(1)
    out /= len(pairs); d = float(np.mean([a.mean() - b.mean() for a, b in pairs]))
    return d, *np.percentile(out, [2.5, 97.5]), float(min(1.0, 2 * min((out <= 0).mean(), (out >= 0).mean())))


def conf_norm(Lv, yv, Lt, level=0.9):
    """'conf-norm' split-conformal interval, identical to notebooks/analyze_p7_p8.conf_norm (30-min blocks, s0 = median spread)."""
    mu_v, sd_v = Lv.mean(0), Lv.std(0); res_v = yv - mu_v; mu_t, sd_t = Lt.mean(0), Lt.std(0)
    lo, hi = np.zeros_like(mu_t), np.zeros_like(mu_t)
    for b in range(4):
        for bl in [slice(i, i + 30) for i in range(0, 180, 30)]:
            s0 = np.median(sd_v[:, bl, b]); r = np.sort((np.abs(res_v[:, bl, b]) / (sd_v[:, bl, b] + s0)).ravel()); n = len(r)
            half = r[min(n, int(np.ceil((n + 1) * level))) - 1] * (sd_t[:, bl, b] + s0)
            lo[:, bl, b], hi[:, bl, b] = mu_t[:, bl, b] - half, mu_t[:, bl, b] + half
    return lo, hi


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--events", default=str(ROOT / "data/processed/events_latis.npz")); a = ap.parse_args(); R = a.results
    os.environ.update(FISM_PROTOCOL="full", FISM_EVENTS=str(Path(a.events).resolve()), FISM_DEVICE_QUIET="1", FISM_ISSUE_DELAY="0",
                      FISM_PREDICT_EXTRA="0", FISM_EXCLUDE_NEAR_TEST_H="0", FISM_EXCLUDE_RANDOM_N="0", FISM_EXCLUDE_SAME_AR="0")
    from src.data import pool
    os.environ.update(FISM_INPUT="goes_b", FISM_SPLIT_LIKE_GOES="0"); pool.load_full.cache_clear()
    _, Yg, _, _, trg, vag, _, _ = pool.load_full()
    os.environ.update(FISM_INPUT="default", FISM_SPLIT_LIKE_GOES="1"); pool.load_full.cache_clear()
    _, Yc, _, _, trc, vac, _, _ = pool.load_full()
    assert np.array_equal(trg, trc) and np.array_equal(vag, vac), "assignment of the control run differs from the GOES pools"
    ok = ~np.isnan(oof(R, EXP["GOES"], "GRU-z")).any(axis=(0, 2, 3)); y = Yg[96:192][ok]
    T = {k: {m: mape4(oof(R, e, m)[:, ok], y) for m in ARCH} for k, e in EXP.items()}
    rows = []
    for A, Bk, fam in (("GOES", "FISM2_c1", True), ("FCNconv", "FISM2_c1", True), ("GOES", "FISM2_own", False), ("FCNconv", "FISM2_own", False),
                       ("FISM2_c1", "FISM2_own", False)):
        d, lo, hi, p = hboot([(T[A][m], T[Bk][m]) for m in ARCH])
        rows.append(dict(A=A, B=Bk, family=fam, n=int(ok.sum()), d=d, lo=lo, hi=hi, p=p,
                         per_arch=" ".join(f"{m}:{T[A][m].mean() - T[Bk][m].mean():+.2f}" for m in ARCH)))
    D = pd.DataFrame(rows); f = D.family.values
    ps = D.p.values[f]; o = np.argsort(ps); adj = np.empty_like(ps); run = 0
    for r_, i in enumerate(o):
        run = max(run, (len(ps) - r_) * ps[i]); adj[i] = min(1, run)
    D.loc[f, "p_holm"] = adj
    new = float(D[(D.A == "GOES") & (D.B == "FISM2_c1")].d.iloc[0])
    D["decision"] = ""; D.loc[(D.A == "GOES") & (D.B == "FISM2_c1"), "decision"] = (
        "within +-0.3 of +1.96: keep the sentence, update the numbers" if abs(new - 1.96) <= 0.3 else
        "outside +-0.3 of +1.96: report both values and revise 'about 2 points'")
    out = Path(R) / "full_server"; out.mkdir(parents=True, exist_ok=True); D.to_csv(out / "control1.csv", index=False)
    pd.set_option("display.width", 220); print(D.round(4).to_string(index=False))
    # benchmark-path interval re-calibrated on the (now shared) validation flares
    rows = []
    for m in ("GRU-z", "FCN-z"):
        vs = [np.load(fv, allow_pickle=False) for fv in sorted(glob.glob(f"{R}/{EXP['FISM2_c1']}/val_{m}_s*.npz"))]
        Lv = np.stack([v["pred_log"].astype(float) for v in vs]); idx = vs[0]["idx"]
        Lt = oof(R, EXP["FISM2_c1"], m); ly = np.log10(Yc[96:192])
        lo, hi = conf_norm(Lv, np.log10(Yc[idx]), Lt, level=0.9); ins = (ly >= lo) & (ly <= hi)
        rows.append(dict(model=m, n_test=96, cov_w0=float(ins[:, :, 0].mean()), cov_4band=float(ins.mean()),
                         width_w0_dex=float((hi - lo)[:, :, 0].mean())))
    C = pd.DataFrame(rows); C.to_csv(out / "control1_conformal.csv", index=False); print(C.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
