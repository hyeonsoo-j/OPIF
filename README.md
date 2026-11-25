# OPIF — Operational Post-peak Irradiance Forecaster for solar flares

Code for the manuscript

> H. Jeon and J. Jung, *OPIF: Toward an Operational Post-Peak Irradiance Forecaster for Solar Flares Using Inputs Available at Issue Time*,
> submitted to *Space Weather* (2026).

The study builds on the benchmark of **Kim et al. (2026)**, *Benchmarking deep learning models for real-time prediction of
post-peak solar flare irradiance using FISM2 data*, Space Weather, doi:10.1029/2026SW005183, and on their data package
(Zenodo doi:10.5281/zenodo.21400071). Their task: from the 60 minutes of 0.1–0.8 nm X-ray irradiance before the flare peak,
forecast the FISM2 irradiance in four bands (0.1–0.8, 9.0–9.9, 13.0–13.9, 30.0–30.9 nm) for the 180 minutes after the peak.

This repository re-builds that benchmark from public data and re-designs it for operations:

| step | what the code does | where |
|---|---|---|
| data | M/X flares 2003–2023 (1,606) from public FISM2 (LASP LISIRD) and NOAA GOES XRS; the 96 test flares of Kim et al. | `scripts/download/`, `src/data/` |
| models | FCN / GRU / LSTM / RNN / Transformer with global z-score scaling, 5-seed ensembles | `src/models/`, `src/experiments/`, `configs/` |
| input available at issue time | GOES XRS-A/B instead of the FISM2 0.1–0.8 nm band; a learned GOES→FISM2 converter | `src/data/pool.py`, `src/models/converter.py` |
| issue time | armed peak detector (declares a maximum 2 min after the peak), two-stage issue (+2 min, re-issue at +10 min) | `notebooks/peak_detector.py`, `notebooks/detector_analysis.py` |
| uncertainty | split-conformal 90 % intervals calibrated on 122 validation flares | `notebooks/split_conformal_full.py`, `notebooks/analyze_p7_p8.py` |
| application | 10 MHz HF absorption over Korea (D-RAP relation) | `notebooks/apply_drap.py` |
| non-isolated flares | forecasts valid until the next declared flare; re-issue rule | `notebooks/analyze_p7_p8.py`, `notebooks/analyze_p9.py`, `notebooks/analyze_p11.py` |
| statistics | hierarchical bootstrap over flares and training seeds, Holm correction | `notebooks/replicate_tests.py` |
| figures | figures of the paper (Figures 2–7, S1–S6) and of the seminar slides | `notebooks/paper1_figures.py`, `notebooks/talk_figures.py`, `paper/fig01_pipeline.tex` |
| Supporting Information | numbers computed without training (validation flares, level/shape, baselines, coverage levels, detector threshold, absorption scale) | `notebooks/si_*.py` |

Every experiment was specified, with its decision criterion, before it was run: see [`docs/decisions.md`](docs/decisions.md).

## Installation

```bash
git clone https://github.com/hyeonsoo-j/OPIF.git && cd OPIF
python3 -m pip install -r requirements.txt        # Python >= 3.10 (tested versions: docs/env.md)
```

Training runs on a GPU. The runners refuse to train silently on the CPU (`FISM_REQUIRE_GPU=1`, set in
`deploy/server/env.sh`; set `FISM_REQUIRE_GPU=0` only for small tests).

## Data

No data are stored in the repository; everything is downloaded from the public archives (see [`data/README.md`](data/README.md)).

```bash
# 1  the 96 test flares of Kim et al.: download 2_evaluation.zip from doi:10.5281/zenodo.21400071, unzip it to external/zenodo/raw/2_evaluation/
python scripts/zenodo_xy_from_package.py external/zenodo/raw/2_evaluation/3_evaluation/all_model_predictions_same_ytrue.npz

# 2  public FISM2 + GOES XRS for all M/X flares 2003-2023 (a few hours, resumable)
python scripts/download/run_all.py --stage test --limit 3     # smoke test
python scripts/download/run_all.py --stage test               # calibration against the authors' arrays
python scripts/download/run_all.py --stage all                # -> data/processed/events_latis.npz
python scripts/build_goes_post.py                             # GOES minutes after the peak (delayed issue)
```

The flare lists used (NOAA GOES XRS reports, HEK SWPC events) are kept in `external/flare_list/` as downloaded.

## Running the experiments

```bash
python run.py --exp p2_zscore --protocol full --input goes_convnn --seeds 0,1,2,3,4    # one configuration
```

`deploy/server/` holds the scripts that ran every experiment of the paper on a multi-GPU Linux server
(`run_everything.sh` runs steps 0–11; `10_run_paper1.sh` the operational experiments P1–P12; `12_run_control1.sh` the control
experiment fixed before its run on 2026-11-25: FISM2-input models with the train/validation assignment of the GOES-input models;
`13_run_detector_continuous.sh` the peak detector on two continuous years of GOES XRS-B, no training). They expect the repository at
`$REPO` (default: a folder `fism2_repro/` next to the scripts), e.g. `REPO=$PWD bash deploy/server/run_everything.sh`.
Results are written to `results/` (not tracked).

## Figures

```bash
python notebooks/paper1_figures.py --results results --out figures/paper        # Figures 2-7 and S1-S6 (vector PDF)
python notebooks/talk_figures.py --results results --out figures/talk           # seminar figures
xelatex paper/fig01_pipeline.tex                                                # Figure 1
```

Figures use Times New Roman and a fixed colour concept (grey = data and the Kim et al. setting, green = what we forecast,
orange = when a forecast is issued or ends; [`docs/figure_style.md`](docs/figure_style.md)). The scripts stop if Times New Roman
is not installed rather than drawing with another font.

## What is not included

Scripts of an early stage that read the authors' material through a web browser before the full-precision files and the
open-data rebuild were available (`scripts/browser/`, `scripts/decode_fism_dataset.py`) and a superseded figure script were removed
from this release; none of them is used by the paper. The decision log keeps the scientific record of that stage.

## Tests

```bash
python -m pip install pytest
python -m pytest -q tests
```

## Citation

Please cite the paper (reference to be added on publication) and this software (`CITATION.cff`; a Zenodo DOI is minted for
each release). Please also cite Kim et al. (2026) and their data package, and the data sources: FISM2 (Chamberlin et al., 2020;
LASP LISIRD) and NOAA GOES XRS (NOAA NCEI).

## License

Code: MIT ([`LICENSE`](LICENSE)). The data used are not redistributed here; they remain under the terms of their providers
(the Zenodo package of Kim et al. is CC BY 4.0).
