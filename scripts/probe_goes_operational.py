"""paper-1 P10: what the OPERATIONAL (real-time product) GOES XRS archives hold, next to the science-quality archive used so far.

    python scripts/probe_goes_operational.py [--out results/full_server/goes_operational_probe.csv]

Lists a few directories of the NOAA archives (no data files are downloaded) and writes, per directory: reachable or not, number
of entries and the first names. The next patch builds the operational inputs from what this shows.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "download"))
from common import fetch_text  # noqa: E402

NGDC = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes"
NCEI = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access"
DIRS = [("GOES-16 L2 products", f"{NGDC}/goes16/l2/data/"),
        ("GOES-16 1-min averages, operational, 2022-05", f"{NGDC}/goes16/l2/data/xrsf-l2-avg1m/2022/05/"),
        ("GOES-16 1-min averages, science, 2022-05", f"{NGDC}/goes16/l2/data/xrsf-l2-avg1m_science/2022/05/"),
        ("GOES-17 L2 products", f"{NGDC}/goes17/l2/data/"),
        ("GOES-18 L2 products", f"{NGDC}/goes18/l2/data/"),
        ("GOES-18 1-min averages, operational, 2023-03", f"{NGDC}/goes18/l2/data/xrsf-l2-avg1m/2023/03/"),
        ("GOES 1-15 archive root", f"{NCEI}/"),
        ("GOES 1-15 averages", f"{NCEI}/avg/"),
        ("GOES 1-15 averages 2014-06", f"{NCEI}/avg/2014/06/"),
        ("GOES-15 averages 2014-06", f"{NCEI}/avg/2014/06/goes15/"),
        ("GOES-15 averages 2014-06 netcdf", f"{NCEI}/avg/2014/06/goes15/netcdf/"),
        ("GOES-15 averages 2014-06 csv", f"{NCEI}/avg/2014/06/goes15/csv/"),
        ("GOES-12 averages 2005-09 csv", f"{NCEI}/avg/2005/09/goes12/csv/"),
        ("GOES-10 averages 2003-10 csv", f"{NCEI}/avg/2003/10/goes10/csv/"),
        ("GOES 1-15 science XRS", f"{NCEI}/science/xrs/"),
        ("GOES-15 science 1-min averages", f"{NCEI}/science/xrs/goes15/xrsf-l2-avg1m_science/"),
        ("GOES 1-15 full resolution", f"{NCEI}/full/")]


def listing(url: str):
    try:
        text = fetch_text(url, timeout=60.0)
    except Exception as e:  # noqa: BLE001  (a probe: every failure is a result)
        return False, 0, f"{type(e).__name__}: {e}"[:200]
    if text is None:
        return False, 0, "not found"
    names = [h for h in re.findall(r'href="([^"?#][^"]*)"', text) if not h.startswith(("/", "http", "..", "mailto"))]
    return True, len(names), " ".join(names[:60])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "results/full_server/goes_operational_probe.csv"))
    a = ap.parse_args(argv)
    rows = []
    for label, url in DIRS:
        ok, n, names = listing(url)
        rows.append(dict(label=label, url=url, reachable=ok, n_entries=n, entries=names))
        print(f"[probe] {'ok ' if ok else 'NO '} {n:4d}  {label}: {names[:140]}", flush=True)
        time.sleep(0.5)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"[probe] {sum(r['reachable'] for r in rows)} of {len(rows)} directories listed -> {a.out}")


if __name__ == "__main__":
    main()
