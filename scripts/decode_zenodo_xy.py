"""Decode the delta-coded, quantised log10 transcript of the Zenodo test-set arrays
(X_real 96x60, Y_true 96x180x4) into data/processed/zenodo_test_xy.npz and verify every
sequence against its integer checksum (#sum of quantised values)."""
import glob
import sys

import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

files = sorted(glob.glob(str(ROOT / 'external/zenodo/xy_chunk*.txt')))
text = ''.join(open(f).read() for f in files)
lines = [l.strip() for l in text.splitlines() if l.strip()]

seqs = []  # (section, q array)
section = None
bad = 0
for l in lines:
    if l.startswith('X_real'):
        section = 'X'
        continue
    if l.startswith('Y_true'):
        section = 'Y'
        continue
    body, chk = l.split('#')
    d = np.array([int(x) for x in body.split()], dtype=np.int64)
    q = np.cumsum(d)
    if q.sum() != int(chk):
        bad += 1
        print('CHECKSUM FAIL', section, len(seqs), q.sum(), chk)
    seqs.append((section, q))

X = [q for s, q in seqs if s == 'X']
Y = [q for s, q in seqs if s == 'Y']
print('X seqs', len(X), 'Y seqs', len(Y), 'bad', bad)
if len(X) == 96 and len(Y) == 384:
    Xr = 10 ** (np.stack(X).astype(float) / 1e4)
    Yl = np.stack(Y).astype(float).reshape(96, 4, 180).transpose(0, 2, 1) / 1e4
    Yr = 10 ** Yl
    ev = open(ROOT / 'external/zenodo/event_metrics_compact_part1.txt').read().splitlines()[1].split(',')
    np.savez(ROOT / 'data/processed/zenodo_test_xy.npz', X_real=Xr, Y_true=Yr, event_names=np.array(ev))
    print('saved; X[-1]==Y0 (log, quantised) all:', np.allclose(np.log10(Xr[:, -1]), np.log10(Yr[:, 0, 0])))
    print('Y range', Yr.min(), Yr.max())
sys.exit(1 if bad else 0)
