"""I-1 cost table: parameter count, training time per fold (CPU, 1 thread) and inference latency per
event (batch 1, median of 50 calls, CPU 1 thread) from results/<exp>/info_<model>_s*.json.
Output: results/improve/arch_costs.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [("repro_cv96", m) for m in ["FCN", "RNN", "LSTM", "GRU", "Transformer"]] + \
        [("p2_norm", "GRU-peaknorm"), ("p2_norm", "FCN-peaknorm"), ("p2_zscore", "FCN-z"),
         ("p2_norm2", "Transformer-peaknorm")] + \
        [("p3_arch", m) for m in ["PatchTST", "TSMixer", "N-BEATS", "ModernTCN"]] + \
        [("p3_joint", "GRU-pn-perband"), ("p3_joint", "FCN-z-joint"), ("p3_optuna", "GRU-pn-optuna")]
rows = []
for exp, m in PAIRS:
    fs = sorted((ROOT / f"results/{exp}").glob(f"info_{m}_s*.json"))
    if not fs:
        continue
    I = [i for f in fs for i in json.load(open(f))]
    rows.append(dict(exp=exp, model=m, params=int(np.median([i.get("params", 0) for i in I])),
                     train_s_per_fold=float(np.mean([i.get("seconds", np.nan) for i in I])),
                     epochs_mean=float(np.mean([np.mean(i.get("epochs", [np.nan])) for i in I])),
                     latency_ms=float(np.median([i.get("latency_ms", np.nan) for i in I]))))
D = pd.DataFrame(rows)
D.to_csv(ROOT / "results/improve/arch_costs.csv", index=False)
print(D.round(3).to_string(index=False))
