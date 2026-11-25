"""A2: scale of the absorption reference. 10 MHz absorption over Korea (D-RAP X-ray formula, 37N 127E) on the 31 daytime test flares,
reference from (i) the FISM2 0.1-0.8 nm target (as published), (ii) the science-quality GOES XRS-B of the same minutes (GOES-R flux
scale), (iii) for GOES 8-15 the operational scale of that era (science-quality x 0.7). Ensemble forecasts scored against each. No training."""
import os, sys, glob, numpy as np, pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = f"{REPO}/results"; sys.path.insert(0, REPO)
from src.eval.drap import event_absorption
D = np.load(f"{REPO}/data/processed/events_latis.npz", allow_pickle=False)
rb = np.load(f"{REPO}/data/processed/events_test96_rebuilt.npz", allow_pickle=False); o = np.argsort(rb["test_k"])
ev = rb["event_id"][o].astype(str); Y = rb["Y"][o].astype(float)
row = {e: i for i, e in enumerate(D["event_id"].astype(str))}
S = np.load(f"{R}/full_server/peak_detector_streams.npz", allow_pickle=False); srow = {int(r): j for j, r in enumerate(S["row"])}; PRE = int(S["pre"])
def ensmu(exp, m): return np.stack([np.load(f).astype(float) for f in sorted(glob.glob(f"{R}/{exp}/oofL_{m}_s*.npy"))]).mean(0)
MOD = {"FISM2-input GRU-z": ensmu("p2_zscore_full_gpu5", "GRU-z"), "FISM2-input FCN-z": ensmu("p2_zscore_full_gpu5", "FCN-z"),
       "operational GRU-z": ensmu("p2_zscore_full_in-goes_convnn_p2nn", "GRU-z"), "operational FCN-z": ensmu("p2_zscore_full_in-goes_convnn_p2nn", "FCN-z")}
rec = []
for e in range(96):
    r = row[ev[e]]; pk = pd.Timestamp(str(D["peak"][r]))
    a_f, zen = event_absorption(Y[e, :, 0], pk)
    if zen[0] >= 90 or r not in srow or np.isnan(MOD["operational GRU-z"][e]).any():
        continue
    b = S["xrsb"][srow[r], PRE:PRE + 180].copy(); okb = np.isfinite(b) & (b > 0)
    if okb.mean() < 0.95:
        continue
    b[~okb] = np.interp(np.where(~okb)[0], np.where(okb)[0], b[okb])
    sat = int(S["sat"][srow[r]]); a_g, _ = event_absorption(b, pk); a_o, _ = event_absorption(b * (0.7 if sat <= 15 else 1.0), pk)
    dd = zen < 90; x = dict(event=ev[e], sat=sat, peak_fism=a_f[0], peak_goes=a_g[0], dlog_peak=np.log10(b[0] / Y[e, 0, 0]),
                            bias_goes_minus_fism=(a_g - a_f)[dd].mean(), mad_goes_fism=np.abs(a_g - a_f)[dd].mean(), mad_ops_fism=np.abs(a_o - a_f)[dd].mean())
    for k, M in MOD.items():
        a_p, _ = event_absorption(10 ** M[e, :, 0], pk)
        x[f"MAE_vs_fism|{k}"] = np.abs(a_p - a_f)[dd].mean(); x[f"MAE_vs_goes|{k}"] = np.abs(a_p - a_g)[dd].mean(); x[f"MAE_vs_ops|{k}"] = np.abs(a_p - a_o)[dd].mean()
    rec.append(x)
E = pd.DataFrame(rec); print(f"daytime test flares with a complete GOES series: {len(E)} (GOES 8-15: {(E.sat <= 15).sum()}, 16-18: {(E.sat >= 16).sum()})")
print(f"mean peak absorption: FISM2 reference {E.peak_fism.mean():.3f} dB, GOES science-quality reference {E.peak_goes.mean():.3f} dB; median log10(GOES/FISM2) at the peak {E.dlog_peak.median():+.3f} dex")
print(f"reference difference over the daytime minutes: mean (GOES - FISM2) {E.bias_goes_minus_fism.mean():+.3f} dB, mean |GOES - FISM2| {E.mad_goes_fism.mean():.3f} dB; "
      f"|operational-scale GOES - FISM2| {E.mad_ops_fism.mean():.3f} dB")
for k in MOD:
    print(f"  {k:20s} MAE vs FISM2 reference {E[f'MAE_vs_fism|{k}'].mean():.3f} dB | vs GOES science-quality {E[f'MAE_vs_goes|{k}'].mean():.3f} dB | vs era-operational scale {E[f'MAE_vs_ops|{k}'].mean():.3f} dB")
