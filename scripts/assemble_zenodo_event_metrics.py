"""Assemble the compact per-event metric transcript (transcribed from the Zenodo
2_evaluation.zip before the full-precision files were used) into a tidy CSV and verify it against the
author-side per-block column checksums that were computed from the original
full-precision CSV.

Rounding in the compact transcript: CC 4 dp, RMSE 4 significant digits,
MRE/MAPE/APE 2 dp.  The checksum tolerance therefore allows for rounding:
|sum_transcribed - sum_original| <= N * (half unit of last place).
"""
import sys, glob, re
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

SRC = sorted(glob.glob(str(ROOT / 'external/zenodo/event_metrics_compact_part*.txt')))
OUT = str(ROOT / 'external/zenodo/event_metrics_same_predictions_compact.csv')

text = ''.join(open(f).read() for f in SRC)
lines = [l.rstrip('\n') for l in text.splitlines() if l.strip()]

events = None
rows = []
checks = {}
block = None
mode = 'data'
for l in lines:
    if l.startswith('EVENTS('):
        continue
    if events is None and re.match(r'^\d{8}_[MX]', l):
        events = l.split(',')
        continue
    if l.startswith('#CHECKSUMS'):
        mode = 'check'
        continue
    if mode == 'check':
        p = l.split(',')
        checks[(p[0], p[1])] = np.array([float(x) for x in p[2:]])
        continue
    if l.startswith('#'):
        model, band = l[1:].split(',')
        block = (model, band)
        idx = 0
        continue
    v = [float(x) for x in l.split(',')]
    rows.append(dict(model=block[0], band=block[1], event=events[idx], CC=v[0], RMSE=v[1], MRE=v[2], MAPE=v[3], APE=v[4]))
    idx += 1

df = pd.DataFrame(rows)
print('events:', len(events), 'rows:', len(df), 'models:', df.model.unique().tolist())
assert len(events) == 96 and len(df) == 8 * 96 * 4, 'row count mismatch'
ok = True
for (m, b), s in checks.items():
    sub = df[(df.model == m) & (df.band == b)]
    assert len(sub) == 96, (m, b, len(sub))
    got = sub[['CC', 'RMSE', 'MRE', 'MAPE', 'APE']].sum().values
    tol = np.array([96 * 0.5e-4, 96 * 0.5e-3 * sub.RMSE.max(), 96 * 0.005, 96 * 0.005, 96 * 0.005])
    diff = np.abs(got - s)
    bad = diff > tol
    if bad.any():
        ok = False
        print('CHECKSUM MISMATCH', m, b, 'diff', diff, 'tol', tol)
print('all checksums OK' if ok else 'CHECKSUM FAILURES PRESENT')
df.to_csv(OUT, index=False)
print('wrote', OUT)
sys.exit(0 if ok else 1)
