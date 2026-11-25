# Zenodo package inventory — 10.5281/zenodo.21400071 (v2, 2026-07-16, CC BY 4.0)

"Jeongheon-Kim/2026SW_Solar Irradiance Prediction Model: Dataset" — companion of
Kim et al. (2026), *Space Weather*, doi:10.1029/2026SW005183.

**Copies used.** The text content (scripts, README, CSV tables) was first transcribed into `external/zenodo/` and the test
arrays were used as a checksummed transcript (1e-4 dex). Per-event metrics were transcribed in compact form and verified against
column checksums computed from the original full-precision CSV (all 32 model×band blocks pass,
`scripts/assemble_zenodo_event_metrics.py`). From 2026-11-04 the full-precision zip files were used (SHA-256 below).

## File tree (from the zip central directories)

```
0_data.zip (12.5 kB)
  0_data/data_loader_w0_log10.py          (5.9 kB)  band-specific loader, w0
  0_data/data_loader_w1_log10.py          (4.1 kB)
  0_data/data_loader_w2_log10.py          (3.9 kB)
  0_data/data_loader_w3_log10.py          (3.9 kB)
  0_data/data_loader_multiband_log10.py   (5.8 kB)  4-band loader for RNN/LSTM/GRU/Transformer
1_tune_train.zip (31 kB)
  1_tune_train/train_fcn_w0_log10.py      (6.6 kB)  final FCN-w0 training (anchor model)
  1_tune_train/tune_fcn_w1_log10.py       (10.0 kB) Optuna search + final train, w1
  1_tune_train/tune_fcn_w2_log10.py       (9.7 kB)
  1_tune_train/tune_fcn_w3_log10.py       (9.7 kB)
  1_tune_train/tune_lstm_log10.py         (8.7 kB)
  1_tune_train/tune_gru_log10.py          (8.2 kB)
  1_tune_train/tune_rnn_log10.py          (8.3 kB)
  1_tune_train/tune_transformer_log10.py  (10.5 kB)
  1_tune_train/train_baseline_log10.py    (13.1 kB) persistence / power-law / LinearReg / RF
2_evaluation.zip (3.3 MB)
  3_evaluation/README_metrics.md                              (3.7 kB)
  3_evaluation/all_model_predictions_same_ytrue.npz           (3.15 MB)  X_real (96×60), Y_true (96×180×4),
                                                                         event_names, Y_<model> (96×180×4) ×8
  3_evaluation/event_metrics_same_predictions.csv             (375 kB)   3072 rows = 8 models × 96 events × 4 bands
  3_evaluation/summary_metrics_same_predictions.csv           (5.0 kB)   model × band: CC,RMSE,MRE,MAPE,APE,median/mean event RMSE
  3_evaluation/summary_metrics_and_skill_same_predictions.csv (8.6 kB)
  3_evaluation/skill_{rmse,mape,ape}_same_predictions.csv     (~0.8 kB each)
3_figures.zip (29.5 kB)   not inspected (figure scripts)
```

(No raw FISM2/GOES event files are included: the data loaders expect
`input_*.txt` (60 values) and `output_*.txt` (180 values per band) files
under a local `fism2/data/{input,output}/w?/{0_train,1_valid,2_test}` tree.)

## Test set: 96 events (not 94)

Event names are `YYYYMMDD_<class>` (e.g. `20031103_X3.9`). The list is in
`external/zenodo/event_metrics_compact_part1.txt` (first line) and in
`external/zenodo/event_metrics_same_predictions_compact.csv`. 88 M + 8 X.
An early task description said 94; the archive and the paper both say **96**.

## Score definitions (README_metrics.md + train_baseline_log10.py)

* All metrics in **physical units (W m⁻²)** after `10**pred`.
* **CC, RMSE**: pooled over 96 events × 180 steps per band (17 280 pairs);
  per-event CC/RMSE are also given in the event CSV.
* **MRE_i** = mean over 180 steps of 100·(ŷ−y)/y (signed); **MAPE_i** = same with
  |·|; both averaged over events. (`compute_scores` in the baseline script uses
  `pos = yt > 0`; FISM2 values are always > 0 so no ε is needed.)
* **APE_i** = 100·|ŷ₁ − y₁| / y₁ at the **first output step** (= flare peak);
  the baseline script's older `PE(%)` used the step of max(y) with sign.
* **Skill** = 1 − E_model / E_power-law for RMSE, MAPE, APE.

## What the scripts reveal about the data construction (used in src/)

* Input: 60 values of GOES 0.1–0.8 nm flux (`input_*.txt`), output: 180 values
  per band (`output_*.txt`). Both `log10`-transformed, **no** min-max/z-score.
* **The last input value equals the first output value** for w0: Power-law APE
  is identical for all 96 events (24.24 % = 1 − 2^−0.4005) and LinearReg APE is
  ~1e-13. Hence the input window ends **at the peak minute** and the output
  window starts at the same minute (shared sample), and the "GOES input" series
  is numerically identical to the FISM2 0.1–0.8 nm series at that minute.
* **Power-law baseline** (physical units): `f(t) = A·(t+1)^−α`, `t = 1…180`,
  `A = X_raw[-1]` (last *input* value — the X-ray flux — used for **all four
  bands**, which explains the terrible EUV power-law scores), α = median of
  per-event `curve_fit` on `y/y[0]` with bounds (0.01, 5), p0 = 0.5.
  Paper medians: 0.400, 0.022, 0.073, 0.010.
* **Persistence** in the authors' script = linear extrapolation of the last 10
  input minutes (not "hold the peak value"); EUV bands use `last_val × ratio`.
* **LinearReg / RF**: 60 log10 inputs → 720 log10 outputs (180×4) jointly;
  RF: 200 trees, max_depth 10, random_state 42, n_jobs −1.
* **FCN**: hidden dims from `BASE_H=[180,720,2880]×scale_factor`;
  `num_expansion` 1 → [h1], 2 → [h1,h2,h1], 3 → [h1,h2,h3,h2,h1];
  BatchNorm1d optional, Dropout (AlphaDropout for SELU); output Linear(·,180).
  w0 uses the **anchor** form: input = concat(60 log values, last value) → 61,
  output = anchor + delta. w1–w3: plain Linear(60,…), no anchor.
  Final w0 config: num_expansion 2, scale 1.280980 → [230, 922, 230], GELU,
  no BN, dropout 0.256843, lr 4.84e-4, batch 16 (4th-ranked Optuna trial,
  ~0.6 M params; best trial had 6.9 M).
* **GRU/LSTM/RNN**: `nn.GRU(1, H, L, dropout if L>1, batch_first)` → Dropout(p)
  on last hidden state → Linear(H, 720) → view(180, 4).
* **Transformer**: Linear(1,d) → sinusoidal PE (max_len 60, dropout) →
  `TransformerEncoder(d, nhead, ff=4d, dropout, batch_first)` ×L → **mean pool
  over time** → Linear(d, 720) → view(180,4).
* **Training**: Adam, MSELoss (log10 space), `ReduceLROnPlateau(patience=8,
  factor=0.5)` during tuning (100 epochs, early stop 15) and `patience=15`
  for final training (300 epochs, early stop 30), `clip_grad_norm_(1.0)`,
  best-validation checkpoint. Optuna: 50 trials (30 for Transformer).

## Author summary metrics [zenodo] (summary_metrics_same_predictions.csv)

| model | band | CC | RMSE | MRE % | MAPE % | APE % |
|---|---|---|---|---|---|---|
| Power-law | w0 | 0.748 | 1.160e-05 | 119.99 | 146.19 | 24.24 |
| Power-law | w1 | 0.197 | 5.055e-05 | −25.84 | 71.11 | 61.94 |
| Power-law | w2 | 0.377 | 4.162e-05 | 103.91 | 121.02 | 49.26 |
| Power-law | w3 | 0.091 | 5.234e-04 | −93.37 | 93.37 | 93.49 |
| LinearReg | w0 | 0.855 | 9.429e-06 | 9.56 | 35.74 | 0.00 |
| LinearReg | w1 | 0.347 | 8.876e-06 | 2.81 | 15.01 | 12.73 |
| LinearReg | w2 | 0.713 | 3.149e-06 | 3.27 | 12.84 | 15.28 |
| LinearReg | w3 | 0.244 | 5.750e-05 | 1.43 | 7.32 | 8.04 |
| RandomForest | w0 | 0.873 | 1.027e-05 | 8.37 | 33.86 | 22.19 |
| RandomForest | w1 | 0.657 | 6.680e-06 | 1.34 | 11.59 | 10.45 |
| RandomForest | w2 | 0.821 | 2.593e-06 | 1.57 | 9.62 | 13.80 |
| RandomForest | w3 | 0.658 | 3.747e-05 | 0.62 | 5.15 | 5.67 |
| FCN | w0 | 0.885 | 9.006e-06 | 5.27 | 34.50 | 0.46 |
| FCN | w1 | 0.556 | 7.396e-06 | 1.14 | 12.88 | 12.13 |
| FCN | w2 | 0.824 | 2.640e-06 | 0.61 | 10.05 | 14.68 |
| FCN | w3 | 0.662 | 3.726e-05 | 0.41 | 5.16 | 6.04 |
| LSTM | w0 | 0.851 | 1.073e-05 | 9.67 | 41.25 | 38.81 |
| LSTM | w1 | 0.460 | 8.129e-06 | −1.56 | 14.63 | 12.98 |
| LSTM | w2 | 0.779 | 3.020e-06 | −1.74 | 12.91 | 19.41 |
| LSTM | w3 | 0.327 | 4.923e-05 | −1.91 | 7.34 | 7.70 |
| GRU | w0 | 0.912 | 9.126e-06 | 7.42 | 33.49 | 23.60 |
| GRU | w1 | 0.639 | 6.946e-06 | 0.16 | 12.39 | 10.94 |
| GRU | w2 | 0.835 | 2.610e-06 | −0.09 | 9.89 | 15.89 |
| GRU | w3 | 0.626 | 3.965e-05 | −0.89 | 5.66 | 5.85 |
| RNN | w0 | 0.912 | 8.171e-06 | 12.13 | 35.69 | 35.44 |
| RNN | w1 | 0.608 | 7.533e-06 | −2.63 | 12.96 | 12.47 |
| RNN | w2 | 0.829 | 2.754e-06 | −3.91 | 11.07 | 18.55 |
| RNN | w3 | 0.574 | 4.358e-05 | −2.03 | 6.07 | 6.67 |
| Transformer | w0 | 0.896 | 8.315e-06 | 17.41 | 39.10 | 23.14 |
| Transformer | w1 | 0.652 | 6.741e-06 | 2.91 | 11.88 | 10.76 |
| Transformer | w2 | 0.811 | 2.699e-06 | 1.52 | 9.71 | 12.22 |
| Transformer | w3 | 0.627 | 4.111e-05 | −2.02 | 5.67 | 6.76 |

Skill tables (RMSE/MAPE/APE vs. power-law) are reproduced in the paper's Table 4
and stored in `external/zenodo/zenodo_skill_tables.md`.

Note the w3 **pooled** CC of the Transformer (0.627) versus its per-event CC
mean (19.36/96 = 0.20): pooled CC is dominated by the inter-event amplitude
differences, per-event CC measures the within-event temporal shape.

## SHA-256 of the zip files (external/zenodo/raw/)
```
6bdf935ee076be06de2f4e66599fc71449b2b0900c59b59bc18f4b72bbb1811e  0_data.zip
fadbad59cdb62b18a1762ed1adcb0f0a41f8bf21a480c0393ec2c1a34dd57640  1_tune_train.zip
08ef4b750ca03b2f01f3facb4e4b77ce80a7d8e6e072ad37b0112a903f7b1782  2_evaluation.zip
804b5e4375c0921afd1e4d421d269c91292181b8b527830e2cf2cdf88bbb7430  3_figures.zip
```
The earlier text transcription is superseded for analysis of the authors' predictions; it agrees with the full-precision arrays to <= 5.0e-5 dex (results/verify/transcription_check.json).
