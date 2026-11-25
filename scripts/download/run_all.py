"""One command for the whole data pipeline (run on a machine with internet access).

  Step 1  smoke test (3 events, ~1 min):
              python scripts/download/run_all.py --stage test --limit 3
  Step 2  calibration on the 96 Zenodo test events (~10-20 min):
              python scripts/download/run_all.py --stage test
          -> read data/download/calibration_report.md (status PASS = our reconstruction matches the
             authors' arrays to < 0.01 dex median)
  Step 3  every M/X flare 2003-2023 (~1600 events; a few hours, resumable) + build:
              python scripts/download/run_all.py --stage all
          -> data/processed/events_latis.npz (used by `python run.py --exp ... --protocol full`)

Each step is resumable: rerun the same command after an interruption; finished files are skipped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_events  # noqa: E402
import calibrate  # noqa: E402
import fetch_extras  # noqa: E402
import fetch_fism2  # noqa: E402
import fetch_goes_xrs  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["test", "all"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2, help="parallel downloads per source (default 2)")
    ap.add_argument("--only-selected", action="store_true", help="stage all: skip events failing the isolation rule")
    ap.add_argument("--no-goes", action="store_true", help="skip GOES XRS (input = FISM2 0.1-0.8 nm only)")
    ap.add_argument("--no-extras", action="store_true")
    ap.add_argument("--out", default=None, help="stage all: output npz (default data/processed/events_latis.npz)")
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    common = ["--stage", a.stage] + (["--limit", str(a.limit)] if a.limit else []) + (["--root", a.root] if a.root else [])
    sel = ["--only-selected"] if a.only_selected else []
    rc = fetch_fism2.main(common + sel + ["--workers", str(a.workers)])
    if not a.no_goes:
        rc |= fetch_goes_xrs.main(common + sel + ["--workers", str(a.workers + 1)])
    if not a.no_extras:
        try:
            fetch_extras.main(["--root", a.root] if a.root else [])
        except Exception as err:  # noqa: BLE001 - optional data
            print(f"[extras] skipped: {err}")
    if rc:
        print("\nSome downloads failed (see data/download/logs/*_errors.csv). Rerun the same command to retry;")
        print("the steps below still run on what is available.\n")
    if a.stage == "test":
        return calibrate.main(["--root", a.root] if a.root else [])
    root = fetch_fism2.C.dl_root(a.root)
    if not (root / "calibration.json").exists():
        print("[run_all] no calibration.json yet - calibrating on the test events first")
        calibrate.main(["--root", a.root] if a.root else [])
    args = (["--root", a.root] if a.root else []) + (["--out", a.out] if a.out else [])
    return build_events.main(args)


if __name__ == "__main__":
    sys.exit(main())
