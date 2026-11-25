"""Paper-1 P4 — online peak detection on the GOES XRS-B 1-min stream and its latency.

For every event of the built dataset (data/processed/events_latis.npz) the GOES XRS-B series of the chosen
satellite is read from the downloaded files (data/download/goes) for minutes [peak-60, peak+180). A causal
detector declares the peak at the first minute t such that
    max(B[..t]) >= threshold  and  t - argmax(B[..t]) >= k          (k minutes without a new maximum)
optionally on a 3-min median-filtered series. Latency = declared minute - aligned peak minute (the dataset's
peak); negative = declared too early (a premature local maximum), i.e. the forecast would be issued on a
wrong peak. For each k the script reports the latency distribution, the fraction of events declared within
[0, k+2] minutes, premature declarations and the fraction never declared within 180 min.

    python notebooks/peak_detector.py --events data/processed/events_latis.npz --root data/download
    python notebooks/peak_detector.py ... --require-rise --tag _v2       (P4b: window-edge declarations removed)
    python notebooks/peak_detector.py ... --rearm --save-streams --tag _v3   (P4c: detector that stays armed)
Output: results/full_server/peak_detector<tag>.csv (summary) and peak_detector_events<tag>.csv.

P4c (--rearm). The first two rules stop at their first declaration, and one flare in six has an earlier maximum
above the threshold within the preceding hour. The armed detector (detect_all) keeps running: a later, larger
maximum of the same flare is declared again, and a flare that rises on the tail of an earlier one is declared
as a new episode. Every declaration is written to peak_detector_declarations<tag>.csv; per event the script
reports whether the flare's own maximum (declared maximum within --own-tol minutes of the dataset peak) is
declared, when, and how many declarations precede it. The median filter of this mode is causal (trailing 3
minutes); the centred filter of the first two rules looks one minute ahead, so their "med3" latencies are one
minute too short for real-time use. --save-streams writes the XRS-A/B streams of all events
(peak_detector_streams.npz), so that other detector settings can be tried without the downloaded files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/download"))


def detect(b: np.ndarray, k: int, thr: float, start: int = 0, require_rise: bool = False,
           rise_factor: float = 2.0) -> int | None:
    """Index of the first minute at which the peak is declared (None if never).
    require_rise (v2 rule, P4b): the running maximum must exceed the running minimum of the stream so far by
    rise_factor, i.e. the flare must have risen by that factor inside the window. A stream that opens on the
    decaying tail of a preceding flare (already above thr) is then not declared on its first minutes, while a
    flare on an elevated background is still detected on its own rise."""
    best, ib, lo = -np.inf, -1, np.inf
    for t in range(start, len(b)):
        v = b[t]
        if not np.isfinite(v):
            continue
        lo = min(lo, v)
        if v > best:
            best, ib = v, t
        if best >= thr and ib >= 0 and t - ib >= k and (not require_rise or best >= rise_factor * lo):
            return t
    return None


def medfilt3(b: np.ndarray) -> np.ndarray:
    out = b.copy()
    for t in range(1, len(b) - 1):
        w = b[t - 1:t + 2]
        if np.isfinite(w).all():
            out[t] = np.median(w)
    return out


def causal_med3(b: np.ndarray) -> np.ndarray:
    """Trailing 3-minute median (uses minutes t-2..t only); minutes without three finite samples keep their value."""
    out = b.copy()
    for t in range(2, len(b)):
        w = b[t - 2:t + 1]
        if np.isfinite(w).all():
            out[t] = np.median(w)
    return out


def detect_all(b: np.ndarray, k: int, thr: float, rise_factor: float = 2.0, start: int = 0) -> list[dict]:
    """Armed detector (P4c): every declaration in the stream, in time order.

    The candidate peak is the running maximum `best` (minute `t_max`) of the current episode; `lo_pre` is the minimum
    of the episode before that maximum. The candidate is declared at the first minute t with
        best >= thr,   t - t_max >= k,   best >= rise_factor * lo_pre      (a rise by that factor, not a decaying tail)
    and only if it exceeds the maximum already declared in this episode: a later, larger maximum of the same flare is
    declared again (kind 'update'). A new episode starts when the flux has risen by rise_factor above the minimum
    reached after the candidate while still below it (a flare on the tail of an earlier, larger one); its first
    declaration has kind 'new'. Non-finite samples are skipped.
    Returned dicts: t (declaration minute), t_max, value, lo_pre, episode, kind."""
    out: list[dict] = []
    best, ib, lo_pre = -np.inf, -1, np.inf
    trough = np.inf                       # minimum after the candidate
    declared = -np.inf                    # largest maximum declared in this episode
    episode = 0
    for t in range(start, len(b)):
        v = b[t]
        if not np.isfinite(v):
            continue
        if v > best:                      # new maximum of the episode
            lo_pre = v if ib < 0 else min(lo_pre, trough)
            best, ib, trough = v, t, np.inf
        else:
            trough = min(trough, v)
            if v >= rise_factor * trough:  # rising again from the post-peak minimum, below the earlier maximum
                episode += 1
                lo_pre, best, ib, trough, declared = trough, v, t, np.inf, -np.inf
        if best >= thr and t - ib >= k and best >= rise_factor * lo_pre and best > declared:
            out.append(dict(t=t, t_max=ib, value=float(best), lo_pre=float(lo_pre), episode=episode,
                            kind="new" if declared == -np.inf else "update"))
            declared = best
    return out


def own_peak(decl: list[dict], pre: int, tol: int) -> dict:
    """Summary of one event's declarations relative to the dataset peak (stream index `pre`).
    own = declarations whose maximum lies within tol minutes of the dataset peak; the last of them is the final
    declaration of the flare's own maximum (latency `own`), the first one `own_first`."""
    own = [q for q in decl if abs(q["t_max"] - pre) <= tol]
    return dict(first=decl[0]["t"] - pre if decl else np.nan,
                own=own[-1]["t"] - pre if own else np.nan,
                own_first=own[0]["t"] - pre if own else np.nan,
                own_max=own[-1]["t_max"] - pre if own else np.nan,
                n_before=sum(q["t_max"] < pre - tol for q in decl),
                n_own=len(own),
                n_after=sum(q["t_max"] > pre + tol for q in decl))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default=str(ROOT / "data/processed/events_latis.npz"))
    ap.add_argument("--root", default=str(ROOT / "data/download"))
    ap.add_argument("--ks", default="1,2,3,5,10")
    ap.add_argument("--threshold", type=float, default=1e-5, help="W m^-2 (M1 = 1e-5)")
    ap.add_argument("--pre", type=int, default=60)
    ap.add_argument("--post", type=int, default=180)
    ap.add_argument("--out", default=str(ROOT / "results/full_server"))
    ap.add_argument("--require-rise", action="store_true",
                    help="v2 rule (P4b): declare only after the flux has risen by --rise-factor inside the stream")
    ap.add_argument("--rise-factor", type=float, default=2.0)
    ap.add_argument("--tag", default="", help="suffix of the output files, e.g. _v2")
    ap.add_argument("--rearm", action="store_true",
                    help="P4c: armed detector (detect_all) - all declarations, own-peak latency, earlier declarations")
    ap.add_argument("--own-tol", type=int, default=5,
                    help="--rearm: a declaration belongs to the flare's own peak if its maximum is within this many minutes of the dataset peak")
    ap.add_argument("--save-streams", action="store_true",
                    help="write the XRS-A/B streams of all events to peak_detector_streams.npz (float64, minutes [peak-pre, peak+post))")
    a = ap.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    import common as C
    import fetch_goes_xrs as G
    d = np.load(a.events, allow_pickle=False)
    root = Path(a.root)
    ks = [int(x) for x in a.ks.split(",")]
    rows = []
    n_nogoes = 0
    decl_rows, st_b, st_a, st_row = [], [], [], []
    for i in range(len(d["event_id"])):
        sat = int(d["goes_sat"][i])
        if sat < 0:
            n_nogoes += 1
            continue
        p = C.to_minute(pd.Timestamp(str(d["peak"][i])))
        s = G.load_series(root, sat, p - a.pre, a.pre + a.post, use_flags=True)
        if s is None:
            n_nogoes += 1
            continue
        b = s["xrsb"]
        if np.isfinite(b[a.pre - 10:a.pre + 10]).mean() < 0.5:
            n_nogoes += 1
            continue
        if a.save_streams:
            st_b.append(b.astype(np.float64)); st_a.append(s["xrsa"].astype(np.float64)); st_row.append(i)
        rec = dict(event=str(d["event_id"][i]), cls=str(d["cls"][i]), is_test=bool(d["is_test"][i]), sat=sat,
                   peak_flux=float(np.nanmax(b[a.pre - 5:a.pre + 6])) if np.isfinite(b[a.pre - 5:a.pre + 6]).any() else np.nan)
        if a.rearm:
            rec["row"] = i
            for k in ks:
                for lab, ser in (("raw", b), ("cmed3", causal_med3(b))):
                    dec = detect_all(ser, k, a.threshold, rise_factor=a.rise_factor)
                    for n, q in enumerate(dec):
                        decl_rows.append(dict(row=i, event=rec["event"], k=k, series=lab, n=n, t=q["t"] - a.pre, t_max=q["t_max"] - a.pre,
                                              value=q["value"], lo_pre=q["lo_pre"], episode=q["episode"], kind=q["kind"]))
                    for key, val in own_peak(dec, a.pre, a.own_tol).items():
                        rec[f"{key}_{lab}_k{k}"] = val
        else:
            bm = medfilt3(b)
            for k in ks:
                for lab, ser in (("raw", b), ("med3", bm)):
                    t = detect(ser, k, a.threshold, require_rise=a.require_rise, rise_factor=a.rise_factor)
                    rec[f"lat_{lab}_k{k}"] = np.nan if t is None else t - a.pre
        rows.append(rec)
    E = pd.DataFrame(rows)
    E.to_csv(Path(a.out) / f"peak_detector_events{a.tag}.csv", index=False)
    if a.save_streams and st_row:
        rr = np.array(st_row, dtype=np.int64)
        np.savez_compressed(Path(a.out) / "peak_detector_streams.npz", xrsb=np.stack(st_b), xrsa=np.stack(st_a), row=rr,
                            event_id=np.array([str(d["event_id"][i]) for i in rr]), cls=np.array([str(d["cls"][i]) for i in rr]),
                            sat=d["goes_sat"][rr].astype(np.int64), is_test=d["is_test"][rr].astype(bool),
                            pre=np.int64(a.pre), post=np.int64(a.post))
        print(f"written: {Path(a.out) / 'peak_detector_streams.npz'} ({len(rr)} events x {a.pre + a.post} minutes, XRS-A and XRS-B)")
    if a.rearm:
        pd.DataFrame(decl_rows).to_csv(Path(a.out) / f"peak_detector_declarations{a.tag}.csv", index=False)
        summ = []
        for k in ks:
            for lab in ("raw", "cmed3"):
                own, first = E[f"own_{lab}_k{k}"], E[f"first_{lab}_k{k}"]
                nb, fin = E[f"n_before_{lab}_k{k}"], E[f"own_{lab}_k{k}"].dropna()
                summ.append(dict(k=k, series=lab, n=len(E), frac_own_declared=float(own.notna().mean()),
                                 median_own_latency=float(fin.median()) if len(fin) else np.nan,
                                 p90_own_latency=float(fin.quantile(0.9)) if len(fin) else np.nan,
                                 frac_own_on_time=float(((own >= 0) & (own <= k + 2)).mean()),
                                 frac_first_is_own=float((first == E[f"own_first_{lab}_k{k}"]).mean()),
                                 frac_with_earlier=float((nb > 0).mean()), mean_n_earlier=float(nb.mean()),
                                 frac_no_declaration=float(first.isna().mean())))
        S = pd.DataFrame(summ)
        S.to_csv(Path(a.out) / f"peak_detector{a.tag}.csv", index=False)
        pd.set_option("display.width", 220)
        print(f"events with GOES series: {len(E)} (no series: {n_nogoes}); armed detector, rise factor {a.rise_factor}, own-peak tolerance {a.own_tol} min")
        print(S.round(3).to_string(index=False))
        return 0
    summ = []
    for k in ks:
        for lab in ("raw", "med3"):
            v = E[f"lat_{lab}_k{k}"]
            fin = v.dropna()
            summ.append(dict(k=k, series=lab, n=len(v), n_detected=int(fin.shape[0]), frac_never=float(v.isna().mean()),
                             median_latency=float(fin.median()) if len(fin) else np.nan,
                             p90_latency=float(fin.quantile(0.9)) if len(fin) else np.nan,
                             frac_premature=float((fin < 0).mean()) if len(fin) else np.nan,
                             frac_window_edge=float((fin <= -50).mean()) if len(fin) else np.nan,
                             frac_on_time=float(((fin >= 0) & (fin <= k + 2)).mean()) if len(fin) else np.nan,
                             frac_late_gt10=float((fin > 10).mean()) if len(fin) else np.nan))
    S = pd.DataFrame(summ)
    S.to_csv(Path(a.out) / f"peak_detector{a.tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"events with GOES series: {len(E)} (no series: {n_nogoes})")
    print(S.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
