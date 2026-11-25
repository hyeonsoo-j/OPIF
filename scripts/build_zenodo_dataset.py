"""Build the data record of the paper for Zenodo from the results of the server runs (no training).

    python scripts/build_zenodo_dataset.py --results results --out <folder>

Writes <folder>/: flare_list.csv, test_flares_rebuilt.npz, validation_flares.npz, outside_rule_flares.npz, predictions_*.npz,
conformal_intervals.npz, goes_streams_detector.npz, tables.zip (README.md is written by hand). All arrays are float32; forecasts are log10 irradiance (W m-2).
The arrays of Kim et al. (2026) (their inputs, targets and predictions) are NOT included; get them from doi:10.5281/zenodo.21400071.
Event ids (YYYYMMDD_class) are not unique; flares are identified by `row` (row of flare_list.csv) or by event_id + peak_utc.
"""
import argparse, glob, io, json, os, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
ARCH5 = ("FCN-z", "GRU-z", "LSTM-z", "RNN-z", "Transformer-z")
MAIN = {"fism2_input": ("default", "p2_zscore_full_gpu5"), "goes_direct": ("goes_b", "p2_zscore_full_in-goes_b_p1"),
        "goes_linear_converter": ("goes_b", "p2_zscore_full_in-goes_conv_p2"), "goes_fcn_converter_operational": ("goes_b", "p2_zscore_full_in-goes_convnn_p2nn"),
        "fism2_input_goes_assignment": ("c1", "p2_zscore_full_splitgoes_c1")}     # control experiment 1 (FISM_SPLIT_LIKE_GOES=1)
POOLS = {"default": ("default", "0"), "goes_b": ("goes_b", "0"), "c1": ("default", "1")}   # pool -> (FISM_INPUT, FISM_SPLIT_LIKE_GOES)


def seeds(R, exp, pre, m):
    fs = sorted(glob.glob(f"{R}/{exp}/{pre}_{m}_s*.npy"))
    return np.stack([np.load(f, allow_pickle=False).astype(np.float32) for f in fs]) if fs else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default=str(ROOT / "results")); ap.add_argument("--out", required=True)
    ap.add_argument("--events", default=str(ROOT / "data/processed/events_latis.npz"))
    ap.add_argument("--extra", nargs="*", help="further files for tables.zip/paper/ (figure numbers, Supporting-Information outputs)"); a = ap.parse_args()
    R, out = Path(a.results), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    os.environ.update(FISM_PROTOCOL="full", FISM_EVENTS=str(Path(a.events).resolve()), FISM_DEVICE_QUIET="1", FISM_ISSUE_DELAY="0",
                      FISM_PREDICT_EXTRA="0", FISM_EXCLUDE_NEAR_TEST_H="0", FISM_EXCLUDE_RANDOM_N="0", FISM_EXCLUDE_SAME_AR="0", FISM_SPLIT_LIKE_GOES="0")
    from src.data import pool
    D = np.load(a.events, allow_pickle=False)
    eid, cls = D["event_id"].astype(str), D["cls"].astype(str)
    peak = pd.to_datetime(pd.Series(D["peak"].astype(str)))
    key = {(p.strftime("%Y-%m-%d %H:%M"), c): i for i, (p, c) in enumerate(zip(peak, cls))}
    split, valrows = {}, {}
    for inp, (fi, like) in POOLS.items():
        if inp == "c1" and not (R / MAIN["fism2_input_goes_assignment"][1]).exists():
            continue
        os.environ["FISM_INPUT"], os.environ["FISM_SPLIT_LIKE_GOES"] = fi, like; pool.load_full.cache_clear()
        _, _, _, meta, tr, va, _, _ = pool.load_full()
        s = np.array(["outside_rule" if not sel else "" for sel in D["selected"].astype(bool)], dtype=object)
        s[D["is_test"].astype(bool)] = "test"
        rowof = lambda ix: np.array([key[(pd.Timestamp(p).strftime("%Y-%m-%d %H:%M"), c)] for p, c in zip(meta.peak_time.values[ix], meta.cls.values[ix])])  # noqa: E731
        s[rowof(tr)] = "train"; s[rowof(va)] = "validation"
        s[(s == "")] = "not_used"
        split[inp] = s
        valrows[inp] = {int(p): int(r) for p, r in zip(va, rowof(va))}       # pool index -> dataset row
    xg = D["X_goes_b"].astype(float); goes_ok = np.all(np.isfinite(xg) & (xg > 0), axis=1)
    F = pd.DataFrame(dict(row=np.arange(len(eid)), event_id=eid, peak_utc=peak.dt.strftime("%Y-%m-%d %H:%M"), goes_class=cls, goes_satellite=D["goes_sat"].astype(int),
                          isolated=D["selected"].astype(bool), kim2026_test=D["is_test"].astype(bool),
                          kim2026_test_index=np.where(D["is_test"].astype(bool), D["test_k"].astype(int), -1),
                          goes_window_complete=goes_ok, split_fism2_input=split["default"], split_goes_input=split["goes_b"]))
    if "c1" in split:                                   # the control run uses the GOES-input assignment exactly
        assert np.array_equal(split["c1"] == "train", split["goes_b"] == "train") and np.array_equal(split["c1"] == "validation", split["goes_b"] == "validation")
    F.loc[~goes_ok, "split_goes_input"] = np.where(F.loc[~goes_ok, "split_goes_input"].isin(["train", "validation", "not_used"]), "no_goes_window", F.loc[~goes_ok, "split_goes_input"])
    F.to_csv(out / "flare_list.csv", index=False)
    print("flare_list:", F.split_fism2_input.value_counts().to_dict(), F.split_goes_input.value_counts().to_dict())

    rb = np.load(ROOT / "data/processed/events_test96_rebuilt.npz", allow_pickle=False); o = np.argsort(rb["test_k"])
    np.savez_compressed(out / "test_flares_rebuilt.npz", event_id=rb["event_id"][o].astype(str), kim2026_test_index=rb["test_k"][o].astype(int),
                        input_fism2_0p1_0p8nm=rb["X"][o].astype(np.float32), input_goes_xrsb=rb["X_goes_b"][o].astype(np.float32),
                        input_goes_xrsa=rb["X_goes_a"][o].astype(np.float32), target_fism2=rb["Y"][o].astype(np.float32),
                        note=np.array("W m-2; inputs: minutes -59..0 from the peak; target: minutes 0..179, bands 0.1-0.8, 9.0-9.9, 13.0-13.9, 30.0-30.9 nm"))
    ex = pd.read_csv(R / "p2_zscore_full_p8/extra_rows.csv")                      # 395 flares (the 393 of the GOES runs are a subset)
    xr = ex.dataset_row.values
    np.savez_compressed(out / "outside_rule_flares.npz", event_id=eid[xr], peak_utc=F.peak_utc.values[xr].astype(str), dataset_row=xr.astype(int),
                        input_fism2_0p1_0p8nm=D["X"][xr].astype(np.float32), input_goes_xrsb=D["X_goes_b"][xr].astype(np.float32),
                        input_goes_xrsa=D["X_goes_a"][xr].astype(np.float32), target_fism2=D["Y"][xr].astype(np.float32),
                        note=np.array("flares that fail the isolation rule, predicted only (never trained on); same layout as test_flares_rebuilt.npz"))

    vr = sorted(set(valrows["default"].values()) | set(valrows["goes_b"].values()))
    np.savez_compressed(out / "validation_flares.npz", row=np.array(vr, dtype=int), event_id=eid[vr], peak_utc=F.peak_utc.values[vr].astype(str),
                        split_fism2_input=F.split_fism2_input.values[vr].astype(str), split_goes_input=F.split_goes_input.values[vr].astype(str),
                        input_fism2_0p1_0p8nm=D["X"][vr].astype(np.float32), input_goes_xrsb=D["X_goes_b"][vr].astype(np.float32),
                        input_goes_xrsa=D["X_goes_a"][vr].astype(np.float32), target_fism2=D["Y"][vr].astype(np.float32),
                        note=np.array("validation flares of either input path (122 each, 7 in common); used for early stopping and interval calibration"))

    def val_block(exp, m, inp):
        vs = [np.load(f, allow_pickle=False) for f in sorted(glob.glob(f"{R}/{exp}/val_{m}_s*.npz"))]
        if not vs:
            return None, None
        idx = vs[0]["idx"]; rows = np.array([valrows[inp][int(i)] for i in idx])
        return np.stack([v["pred_log"].astype(np.float32) for v in vs]), rows

    for name, (inp, exp) in MAIN.items():
        if not (R / exp).exists():
            print("skip", name, "(no results)"); continue
        z = {}
        for m in ARCH5:
            for pre, lab in (("oofL", "test_rebuilt"), ("oof", "test_authors_arrays")):
                A = seeds(R, exp, pre, m)
                if A is not None:
                    z[f"{lab}|{m}"] = A
            V, ve = val_block(exp, m, inp)
            if V is not None:
                z[f"validation|{m}"] = V; z[f"validation_row|{m}"] = ve; z[f"validation_event_id|{m}"] = eid[ve]
        np.savez_compressed(out / f"predictions_{name}.npz", **z); print(name, len(z), "arrays")
    z = {}
    for inp, lab in (("default", "fism2_input"), ("goes_convnn", "goes_fcn_converter_operational")):
        exp = "p2_zscore_full_p8" if inp == "default" else "p2_zscore_full_in-goes_convnn_p8"
        e2 = pd.read_csv(R / exp / "extra_rows.csv"); assert set(e2.dataset_row.values) <= set(xr)
        z[f"{lab}|event_id"] = eid[e2.dataset_row.values]; z[f"{lab}|row"] = e2.dataset_row.values.astype(int)
        for m in ("GRU-z", "FCN-z"):
            z[f"{lab}|outside_rule|{m}"] = seeds(R, exp, "oofN", m)
    np.savez_compressed(out / "predictions_outside_rule.npz", **z)
    z = {}
    for inp, lab in (("", "fism2_input"), ("_in-goes_b", "goes_direct"), ("_in-goes_convnn", "goes_fcn_converter_operational")):
        for k in (2, 5, 10):
            for m in ("GRU-z", "FCN-z"):
                A = seeds(R, f"p2_zscore_full{inp}_d{k}_p7", "oofL", m)
                if A is not None:
                    z[f"{lab}|issued_plus{k}min|test_rebuilt|{m}"] = A
    np.savez_compressed(out / "predictions_delayed_issue.npz", note=np.array("forecast issued k minutes after the peak; score lead >= k only"), **z)
    opsd = R / "p2_zscore_full_in-goes_convnn_ops_p11"
    if opsd.exists():
        orow = pd.read_csv(opsd / "ops_rows.csv"); z = {}
        for m in ("GRU-z", "FCN-z"):
            z[f"operational_archive_input|{m}"] = seeds(R, opsd.name, "oofO", m)
            z[f"science_quality_input|test_rebuilt|{m}"] = seeds(R, opsd.name, "oofL", m)
            z[f"science_quality_input|outside_rule|{m}"] = seeds(R, opsd.name, "oofN", m)
        np.savez_compressed(out / "predictions_operational_archive.npz", ops_row=orow.dataset_row.values.astype(int), ops_event_id=orow.event.values.astype(str), ops_kind=orow.kind.values.astype(str),
                            ops_variant=orow.variant.values.astype(str), **z)
    I = np.load(R / "full_server/conformal_intervals_none.npz", allow_pickle=False)
    np.savez_compressed(out / "conformal_intervals.npz", **{k: I[k].astype(np.float32) for k in I.files})
    S = np.load(R / "full_server/peak_detector_streams.npz", allow_pickle=False)
    np.savez_compressed(out / "goes_streams_detector.npz", event_id=S["event_id"], goes_class=S["cls"], goes_satellite=S["sat"],
                        kim2026_test=S["is_test"], xrsb=S["xrsb"].astype(np.float32), xrsa=S["xrsa"].astype(np.float32),
                        note=np.array("science-quality GOES XRS 1-min, W m-2, minutes -60..179 from the dataset peak"))
    with zipfile.ZipFile(out / "tables.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(glob.glob(str(R / "full_server/*.csv"))):
            zf.write(f, f"tables/{Path(f).name}")
        for f in a.extra or []:
            zf.write(f, f"paper/{Path(f).name}")
    print("written to", out)


if __name__ == "__main__":
    main()
