"""Write data/processed/events.nc (xarray; event x time x band) from the decoded Zenodo test arrays
and the matched flare-catalogue metadata, plus results/verify/data_checks.csv (quality flags
used in docs/data_card.md).

time axis: minutes relative to the flare peak, -59 .. +179 (the peak minute 0 is shared by the
input window and the output window, see docs/decisions.md).
  xray(event, time)          GOES 0.1-0.8 nm input for t <= 0 (as provided by Zenodo X_real),
                             FISM2 0.1-0.8 nm for t >= 0 (Y_true w0); identical at t = 0.
  fism2(event, time, band)   FISM2 bands for t >= 0, NaN before the peak (not in the archive).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.cv96 import BAND_LABELS, BANDS, load_meta, load_test96  # noqa: E402


def main():
    X, Y, ev = load_test96()
    m = load_meta()
    t = np.arange(-59, 180)
    xray = np.full((96, len(t)), np.nan)
    xray[:, :60] = X
    xray[:, 59:] = Y[:, :, 0]
    fism = np.full((96, len(t), 4), np.nan)
    fism[:, 59:, :] = Y
    ds = xr.Dataset(
        {"xray": (("event", "time"), xray, {"units": "W m-2", "long_name": "0.1-0.8 nm flux (GOES input / FISM2 w0)"}),
         "fism2": (("event", "time", "band"), fism, {"units": "W m-2", "long_name": "FISM2 band irradiance"})},
        coords={"event": ev, "time": ("time", t, {"units": "minutes since flare peak"}),
                "band": BANDS, "band_label": ("band", BAND_LABELS),
                "goes_class": ("event", m.cls.values), "peak_time": ("event", m.peak_time.values),
                "match_flag": ("event", m.match_flag.values)},
        attrs={"source": "Zenodo 10.5281/zenodo.21400071 all_model_predictions_same_ytrue.npz (X_real, Y_true), "
                         "transcribed as 1e-4-dex quantised log10 text; flare metadata from NGDC/HEK SWPC lists",
               "quantisation": "values carry +-0.5e-4 dex rounding (<= 0.0115 % relative)"})
    out = ROOT / "data/processed/events.nc"
    ds.to_netcdf(out)
    # --- quality checks ---------------------------------------------------------------
    xl, yl = np.log10(X), np.log10(Y)
    rows = []
    peak_t = m.peak_time.values.astype("datetime64[m]")
    for k, e in enumerate(ev):
        # output window of another test event overlapping this event's input window
        t_in0 = peak_t[k] - np.timedelta64(59, "m")
        ov = [ev[j] for j in range(96) if j != k and (peak_t[j] <= peak_t[k]) and
              (peak_t[j] + np.timedelta64(179, "m") >= t_in0)]
        rows.append(dict(
            event=e, cls=m.cls[k], year=pd.Timestamp(peak_t[k]).year,
            n_nonpos=int((X[k] <= 0).sum() + (Y[k] <= 0).sum()), n_nan=int(np.isnan(X[k]).sum() + np.isnan(Y[k]).sum()),
            shared_minute_diff_dex=float(xl[k, -1] - yl[k, 0, 0]),
            argmax_input=int(np.argmax(X[k])), argmax_w0_out=int(np.argmax(Y[k, :, 0])),
            peak_vs_class_dex=float(xl[k, -1] - np.log10(m.cls_value[k])),
            overlapping_test_events=";".join(ov), match_flag=m.match_flag[k],
            cat_selected=m.cat_selected[k], n_pre60=m.n_pre60[k], n_post180=m.n_post180[k],
        ))
    D = pd.DataFrame(rows)
    (ROOT / "results/verify").mkdir(parents=True, exist_ok=True)
    D.to_csv(ROOT / "results/verify/data_checks.csv", index=False)
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB) and results/verify/data_checks.csv")
    return ds, D


if __name__ == "__main__":
    main()
