"""GOES XRS minutes after the dataset peak, for forecasts issued k minutes after the peak (paper-1 P7).

The built dataset (data/processed/events_latis.npz) holds the 60 input minutes up to the peak. A GOES-driven forecast
issued k minutes later also sees GOES minutes 1..k. They are taken from the streams exported by the peak detector
(results/full_server/peak_detector_streams.npz: XRS-A/B for [peak-60, peak+180) of every event with a GOES series),
with the same short-gap rule as the input windows (log-linear interpolation over <= 5 missing minutes).

    python scripts/build_goes_post.py --events data/processed/events_latis.npz \
        --streams results/full_server/peak_detector_streams.npz --out data/processed/events_goes_post.npz

Output arrays (all events of the dataset, in dataset order; NaN where a minute is not available):
    xrsb_post, xrsa_post  (N, n_post)   minutes 1..n_post after the peak
The script checks that the stream reproduces the input windows of the dataset and stops if it does not.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/download"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default=str(ROOT / "data/processed/events_latis.npz"))
    ap.add_argument("--streams", default=str(ROOT / "results/full_server/peak_detector_streams.npz"))
    ap.add_argument("--out", default=str(ROOT / "data/processed/events_goes_post.npz"))
    ap.add_argument("--n-post", type=int, default=10)
    ap.add_argument("--max-gap", type=int, default=5)
    a = ap.parse_args(argv)
    from build_events import fill_short_gaps
    d = np.load(a.events, allow_pickle=False)
    st = np.load(a.streams, allow_pickle=False)
    row, pre = st["row"].astype(int), int(st["pre"])
    n = len(d["event_id"])
    if row.max() >= n or not (st["event_id"] == d["event_id"][row]).all():
        print("ERROR: the streams were exported from another dataset (event ids differ)")
        return 1
    if int(st["post"]) < a.n_post + a.max_gap + 1:
        print("ERROR: the streams are too short after the peak")
        return 1
    hi = pre + a.n_post + a.max_gap + 2          # a few minutes beyond n_post so that a gap ending there can be closed
    out = {"xrsb_post": np.full((n, a.n_post), np.nan), "xrsa_post": np.full((n, a.n_post), np.nan)}
    worst, n_cmp, n_filled = 0.0, 0, 0
    for j, i in enumerate(row):
        for key, dkey, okey in (("xrsb", "X_goes_b", "xrsb_post"), ("xrsa", "X_goes_a", "xrsa_post")):
            seg = st[key][j, pre - 59:hi].astype(float)
            filled = fill_short_gaps(seg, a.max_gap)
            post = filled[60:60 + a.n_post]
            n_filled += int((np.isfinite(post) & ~(np.isfinite(seg[60:60 + a.n_post]) & (seg[60:60 + a.n_post] > 0))).sum())
            out[okey][i] = np.where(np.isfinite(post) & (post > 0), post, np.nan)
            w = d[dkey][i].astype(float)             # the dataset's input window, minutes -59..0
            both = np.isfinite(w) & (w > 0) & np.isfinite(seg[:60]) & (seg[:60] > 0)
            if both.any():
                worst = max(worst, float(np.abs(np.log10(w[both] / seg[:60][both])).max()))
                n_cmp += int(both.sum())
    print(f"streams vs dataset input windows: {n_cmp} samples compared, max |dlog10| = {worst:.2e}")
    if worst > 1e-6:
        print("ERROR: the exported streams do not reproduce the input windows of the dataset")
        return 1
    okb, oka = np.isfinite(out["xrsb_post"]).all(1), np.isfinite(out["xrsa_post"]).all(1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, xrsb_post=out["xrsb_post"], xrsa_post=out["xrsa_post"], n_post=np.int64(a.n_post),
                        event_id=d["event_id"].astype(str))
    print(f"written: {a.out}  ({n} events; minutes 1..{a.n_post} complete: XRS-B {int(okb.sum())}, XRS-A {int(oka.sum())}; "
          f"{n_filled} samples filled over short gaps; events without a stream: {n - len(row)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
