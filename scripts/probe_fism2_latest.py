"""paper-1 P12: how recent is the public FISM2 flare product (LASP LISIRD, fism_flare_hr)?

    python scripts/probe_fism2_latest.py [--out results/full_server/fism2_latest_probe.csv] [--today YYYY-MM-DD]

The LaTiS service is asked for the 0.1-0.2 nm bins in one 10-minute window (12:00 UT) per day, going back from the run day by
0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365 days (then 548, 730, 1095, 1460, 1825 if all are empty), and
then for every day between the last empty and the first filled one. The whole of the most recent day with data is read
once more to get its last minute. Written: one row per request and a summary row (most recent day with data, its last time
stamp, days behind the run day). Only a few kB are transferred.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "download"))
import common as C  # noqa: E402
import fetch_fism2 as F  # noqa: E402

STEPS = [0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365]
FAR = [548, 730, 1095, 1460, 1825]


def ask(t0_min: int, n_min: int, base: str):
    """-> (number of data rows, last minute with a finite value or None, status text)."""
    try:
        txt = C.fetch_text(F.latis_url(t0_min, t0_min + n_min, 0.1, 0.2, base), timeout=120.0)
    except Exception as e:  # noqa: BLE001  (a probe: every failure is a result)
        return 0, None, f"{type(e).__name__}: {e}"[:160]
    if txt is None:
        return 0, None, "not found"
    try:
        wl, irr = F.parse_latis_csv(txt, t0_min, n_min)
    except Exception as e:  # noqa: BLE001
        return 0, None, f"unreadable answer ({type(e).__name__}: {e})"[:160]
    ok = np.isfinite(irr).any(axis=1) if irr.size else np.zeros(n_min, bool)
    return int(ok.sum()), (int(t0_min + np.where(ok)[0][-1]) if ok.any() else None), "ok"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "results/full_server/fism2_latest_probe.csv"))
    ap.add_argument("--today", default=None, help="run day (UTC), default: now")
    ap.add_argument("--base-url", default=F.LATIS_URL)
    ap.add_argument("--pause", type=float, default=0.6)
    a = ap.parse_args(argv)
    today = (pd.Timestamp(a.today) if a.today else pd.Timestamp.now(tz="UTC").tz_localize(None)).normalize()
    rows, seen = [], {}

    def day(off: int):
        if off not in seen:
            dmin = C.to_minute(today - pd.Timedelta(days=off))
            n, last, status = ask(dmin + 12 * 60, 10, a.base_url)
            seen[off] = n
            rows.append(dict(kind="day", day=str((today - pd.Timedelta(days=off)).date()), days_before_run=off, window="12:00-12:10 UT", n_minutes_with_data=n, status=status))
            print(f"[fism2-latest] {rows[-1]['day']} (-{off:4d} d): {n:2d} of 10 minutes  [{status}]", flush=True)
            time.sleep(a.pause)
        return seen[off]

    first = next((o for o in STEPS if day(o) > 0), None)
    if first is None:
        first = next((o for o in FAR if day(o) > 0), None)
    summary = dict(kind="summary", run_day=str(today.date()))
    if first is None:
        summary.update(latest_day_with_data=None, last_time_stamp=None, days_behind_run_day=None, note=f"no data in any window back to {FAR[-1]} days")
    else:
        grid = STEPS + FAR
        prev = max([o for o in grid if o < first], default=-1)          # the last empty offset before the first filled one
        best = first
        for off in range(prev + 1, first):                              # every day in between, most recent first
            if day(off) > 0:
                best = off
                break
        dmin = C.to_minute(today - pd.Timedelta(days=best))
        n, last, status = ask(dmin, 1440, a.base_url)
        rows.append(dict(kind="full day", day=str((today - pd.Timedelta(days=best)).date()), days_before_run=best, window="00:00-24:00 UT", n_minutes_with_data=n, status=status))
        summary.update(latest_day_with_data=str((today - pd.Timedelta(days=best)).date()), last_time_stamp=C.minute_to_iso(last) if last is not None else None,
                       days_behind_run_day=best, note="a day counts as filled if the 12:00-12:10 UT window returns data")
    rows.append(summary)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"[fism2-latest] run day {summary['run_day']}: latest day with data {summary.get('latest_day_with_data')}, last time stamp "
          f"{summary.get('last_time_stamp')}, {summary.get('days_behind_run_day')} days behind -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
