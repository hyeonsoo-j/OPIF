# Environment (docs/env.md)

## Server that ran every training of the paper

```
Linux, 4 x NVIDIA RTX A6000 (48 GB), CUDA 12.4
Python 3.10.20, torch 2.6.0+cu124 (virtual environment created by deploy/server/1_setup.sh)
```

All trainings ran on the GPUs (`FISM_REQUIRE_GPU=1`, `deploy/server/env.sh`). A training with the same seed on the GPU is
bit-for-bit reproducible on the same hardware; on other hardware the seed-mean MAPE4 of a configuration moves by about 0.3 points
(standard deviation 0.26 points over 54 configurations, Supporting Information Text S3).

## Analysis and figures

Analyses that do not train (bootstrap tests, conformal intervals, detector, figures) run on a CPU. Versions used for the tables and
figures of the manuscript v6: Python 3.10.12, numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, matplotlib 3.10.9. Figures need Times New Roman
(`notebooks/sf_colors.py`).

Install: `python3 -m pip install -r requirements.txt`; for training install the CUDA build of torch that matches the server's driver.
