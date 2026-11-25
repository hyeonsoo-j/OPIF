# Data card

## 1. Data sets of the paper

| item | value |
|---|---|
| flare list | 1,606 M/X flares, January 2003 – March 2023 (119 X-class): NOAA GOES XRS flare reports (2003-01 – 2017-04) and the SWPC event list of the Heliophysics Event Knowledgebase (2012, 2014, 2015 and 2017-04 – 2023-03); `external/flare_list/`, `data/processed/flare_catalogue.csv` |
| peak minute | minute of the GOES XRS-B maximum within ±15 min of the listed peak |
| target | FISM2 flare product (LASP LISIRD), 1-min, minutes 0 … 179 after the peak, four bands: 0.1–0.8 nm (mean of the 0.1-nm bins), 9.0–9.9, 13.0–13.9, 30.0–30.9 nm (sum of ten 0.1-nm bins × 0.1 nm), W m⁻² |
| inputs | 60 minutes ending at the peak: FISM2 0.1–0.8 nm band (benchmark input), GOES XRS-B and XRS-A science-quality 1-min data (NOAA NCEI); GOES 8–15: 1,236 flares, GOES 16–18: 370; interior gaps ≤ 5 min filled log-linearly, longer gaps make the window unavailable |
| isolation rule | no other ≥ M1 flare from 60 min before to 180 min after the peak (Kim et al., 2026): 1,206 of the 1,604 flares with a GOES series pass, 398 (24.8 %) do not |
| test flares | the 96 test flares of Kim et al. (2026) (88 M, 8 X), scored twice: the authors' arrays (Zenodo doi:10.5281/zenodo.21400071) and the same flares rebuilt like the training flares; 95 have a complete GOES window |
| train / validation | the other isolated flares, by class in time order, 8:1 round robin: 988 / 122 for FISM2-input models; 982 / 122 for GOES-input models (the 6 flares without a GOES window are removed before the assignment; 867 training and 7 validation flares are common to both) |
| flares outside the rule | 395 (393 with a GOES window), predicted by models that never saw them |
| operational GOES archive | NOAA operational 1-min XRS for the same minutes (GOES 8–15 divided by the SWPC factors 0.7 / 0.85); 90 test flares and 381 flares outside the rule have a complete window |
| files | `data/processed/events_latis.npz` (built by `scripts/download/run_all.py`), `events_test96_rebuilt.npz`, `events_goes_post.npz`, `zenodo_test_xy.npz`, `zenodo_full.npz` |

## 2. Notes

- The authors' test arrays were first used as a checksummed transcript quantised to 1e-4 dex (`external/zenodo/`,
  `scripts/decode_zenodo_xy.py`); it differs from the published package by at most 0.0115 %. `scripts/zenodo_xy_from_package.py`
  builds the arrays from the package itself.
- In our merged flare list, 6 of the 96 test flares have another M flare inside the isolation window (20030613_M3.1, 20050910_X1.1,
  20120705_M6.1, 20210703_X1.5, 20230110_M5.1, 20230110_X1.0); the authors report none, probably a list-version difference.
  The test set is kept as published.
- The detector threshold 1e-5 W m⁻² is applied to the science-quality XRS-B scale. For GOES 8–15 that scale is 1.47 times the flux
  implied by the listed class (median; 1.03 for GOES 16–18), so the threshold corresponds to about C7 on the list scale of that era
  (Supporting Information Text S1, Table S8).
- The target is the FISM2 reconstruction, not an independent EUV observation.

## 3. Quality checks (`tests/test_data.py`, `results/verify/data_checks.csv`)

| check | result |
|---|---|
| missing / non-positive target values | none in the built data set |
| input and target share the peak minute (authors' arrays) | 96/96 |
| authors' input vs public FISM2 0.1–0.8 nm / GOES XRS-B | rms 0.032 / 0.18 dex |
| rebuilt targets vs the authors' arrays | rms 0.008–0.029 dex per band |
