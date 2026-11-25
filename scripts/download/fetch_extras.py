"""Optional context data (small): daily F10.7 solar radio flux (Penticton, NRCan).

    python scripts/download/fetch_extras.py
-> <root>/extras/f107_fluxtable.txt (as published) and <root>/extras/f107_daily.csv
   (date, time, observed and adjusted flux in sfu; three measurements per day before 2004-ish
   only one). build_events.py attaches the adjusted flux of the last measurement before the
   flare day (no look-ahead) as feature 'f107_adj'.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

F107_URL = os.environ.get("FISM_F107_URL", "https://spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt")


def parse_fluxtable(text: str) -> pd.DataFrame:
    rows = []
    for ln in text.splitlines():
        p = ln.split()
        if len(p) >= 6 and len(p[0]) == 8 and p[0].isdigit():
            try:
                rows.append(dict(date=pd.to_datetime(p[0], format="%Y%m%d"), time=p[1],
                                 obs_sfu=float(p[4]), adj_sfu=float(p[5])))
            except ValueError:
                continue
    df = pd.DataFrame(rows)
    if len(df):
        hh = df["time"].str.zfill(6).str[:2].astype(int)
        mm = df["time"].str.zfill(6).str[2:4].astype(int)
        df["datetime"] = df["date"] + pd.to_timedelta(hh, unit="h") + pd.to_timedelta(mm, unit="m")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--url", default=F107_URL)
    a = ap.parse_args(argv)
    root = C.dl_root(a.root)
    txt = C.fetch_text(a.url)
    if txt is None:
        print("[extras] F10.7 table not found (404) - skipped")
        return 1
    (root / "extras" / "f107_fluxtable.txt").write_text(txt, encoding="utf-8")
    df = parse_fluxtable(txt)
    df.to_csv(root / "extras" / "f107_daily.csv", index=False)
    print(f"[extras] F10.7: {len(df)} rows, {df.date.min().date()} .. {df.date.max().date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
