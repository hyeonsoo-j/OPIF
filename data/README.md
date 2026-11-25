# data/

Not tracked. Built by the scripts (see the main README):

| file | made by | content |
|---|---|---|
| `processed/zenodo_test_xy.npz`, `processed/zenodo_full.npz` | `scripts/zenodo_xy_from_package.py` | the 96 test flares of Kim et al. (2026): `X_real`, `Y_true`, authors' predictions |
| `download/` | `scripts/download/run_all.py` | raw FISM2 (LASP LISIRD) and GOES XRS (NOAA NCEI) files |
| `processed/flare_catalogue.csv` | `src/data/flare_list.py` | M/X flares 2003-01 – 2023-03 from `external/flare_list/` |
| `processed/events_latis.npz` | `scripts/download/run_all.py --stage all` | 1,606 flares: 60-min inputs (FISM2, GOES XRS-A/B), 180-min FISM2 targets in 4 bands |
| `processed/events_goes_post.npz` | `scripts/build_goes_post.py` | GOES minutes after the peak (delayed issue) |
