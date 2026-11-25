"""A1 (partial): what the armed detector declares inside the 4-h windows [-60, +180] min around every M/X flare of the list.
A declaration counts as a catalogue flare if its maximum lies within 5 min of the peak of any M/X flare of the list.
Continuous (quiet-time) false-declaration rates need full-day GOES files, which this environment cannot download."""
import os, sys, importlib.util, numpy as np, pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = f"{REPO}/results/full_server"
spec = importlib.util.spec_from_file_location("pdet", f"{REPO}/notebooks/peak_detector.py"); pdet = importlib.util.module_from_spec(spec); spec.loader.exec_module(pdet)
d = np.load(f"{REPO}/data/processed/events_latis.npz", allow_pickle=False); S = np.load(f"{R}/peak_detector_streams.npz", allow_pickle=False); pre = int(S["pre"])
pk_all = pd.to_datetime(pd.Series(d["peak"].astype(str))).values.astype("datetime64[m]").astype(np.int64)
for thr_old in (1e-5, 1.43e-5):
    rec = []
    for j, row in enumerate(S["row"]):
        thr = thr_old if S["sat"][j] <= 15 else 1e-5; p0 = pk_all[row]
        for q in pdet.detect_all(S["xrsb"][j], 2, thr, rise_factor=2.0):
            ta = p0 + q["t_max"] - pre; m = np.abs(pk_all - ta).min() <= 5
            rec.append(dict(j=j, own=abs(q["t_max"] - pre) <= 5, cat=m, kind=q["kind"], value=q["value"], sat=int(S["sat"][j])))
    E = pd.DataFrame(rec); oth = E[~E.own]
    print(f"threshold for GOES 8-15 {thr_old:.2e}: declarations {len(E)} in {len(S['row'])} windows; own peak {E.own.sum()}; other {len(oth)}:"
          f" catalogue M/X flare {oth.cat.sum()} ({oth.cat.mean():.1%}), not in the M/X list {(~oth.cat).sum()} ({(~oth.cat).sum() / len(S['row']):.3f} per 4-h window),"
          f" of which updates of an earlier declaration {((~oth.cat) & (oth.kind == 'update')).sum()}")
