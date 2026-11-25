# Decision log (docs/decisions.md)

Format: date — decision — rationale — alternatives considered.

Editorial note (2026-11-25, public release): notes about the working environment (network access, file transfer, who did what)
were removed from the entries of 2026-11-03 to 2026-11-20; the design, decision-rule and result items are unchanged. The unedited
log is kept with the project archive (server patch_v16). Entries from 2026-11-20 (P11/P12 results) on were added after that patch.

## 2026-11-03 — Starting point: the authors' public test material only; early protocol CV96
- **Decision.** Until the open-data rebuild of the training set (2026-11-05), the only data were the authors' public Zenodo
  material: scripts, metric tables, flare lists (NGDC/HEK) and the test arrays `X_real` (96x60) and `Y_true` (96x180x4). The test
  arrays were first used as a transcribed copy quantised to 1e-4 dex and checksum-verified (`external/zenodo/xy_chunk*.txt`,
  `scripts/decode_zenodo_xy.py`); the full-precision files differ from it by at most 0.0115 % (2026-11-04).
- **Rationale.** Analyses possible with the public test material only; no measurement is fabricated.
- **Consequence → protocol "CV96".** All training-based experiments (Phase 1 proxy reproduction,
  Phase 2 sensitivity tests, Phase 3 improvements) use a **10-fold nested cross-validation over
  the 96 Zenodo test events** (`src/data/cv96.py`): outer folds stratified by class and
  interleaved in time order (fold = rank mod 10, mimicking the paper's sequential allocation);
  inner train/val split 8:1 by the same rule. Every model sees ~78 training / 8 validation events
  (paper: 772 / 96) and is scored on out-of-fold predictions for all 96 events, so the scores are
  **not** a reproduction of the paper's numbers but a like-for-like *relative* comparison under a
  10x smaller training set. The authors' own per-event scores [zenodo] are analysed separately.
- **Alternatives.** Synthetic training data (rejected: results would not be about the Sun); leave-one-out (10x more compute,
  little gain); training on the 96 events and testing on the same events (leakage — rejected).
- Superseded on 2026-11-05 by protocol "full" on the open-data rebuild (below).

## 2026-11-03 — Event metadata for the 96 test events
- Event ids (`YYYYMMDD_<class>`) are matched to the catalogue; 4 ids are ambiguous (same class twice
  on one day). Two are resolved by the isolation rule (only one candidate passes it), two by the
  rise time seen in `X_real` (flag `ambiguous_resolved_by_rise_time`; 20040119_M1.0, 20131022_M1.0).
- Flare locations exist for 33/96 events only (NGDC rows with a position + HEK 2017–2023); the HEK
  records for 2012/2014/2015 have none. Location is therefore used only with a missing indicator.

## 2026-11-03 — Test set is 96 events (paper + Zenodo), not 94
- An early task description said 94; the Zenodo archive has 96 named test events (88 M, 8 X). We use 96.

## 2026-11-03 — Window alignment: input ends at the peak minute, output starts at the peak minute
- **Evidence [zenodo].** Power-law APE is 24.2396 % for *every* test event (= 1 − 2^−α with
  α = 0.4005) and LinearReg APE is ~1e-13, so y_true[0] == x_input[−1] exactly in the authors'
  data. We therefore define input = minutes [t_peak − 59, t_peak], output = [t_peak, t_peak + 179].
- **Consequence.** The "GOES X-ray input" and the FISM2 0.1–0.8 nm target are the same series at
  the shared minute. For the fallback data path, FISM2's own 0.1–0.8 nm band (which is built
  from GOES XRS) is an acceptable proxy for the GOES input; this is recorded as a deviation.

## 2026-11-03 — Power-law baseline follows the authors' script, not the paper's prose
- `A = last input X-ray value` for all four bands (authors' script), so EUV power-law predictions
  carry X-ray amplitude; α fitted on `y/y[0]` with `t = 1..180`, `(t+1)^−α`, bounds (0.01, 5).
  We implement exactly this as `powerlaw_authors`, and additionally a physically sensible
  `powerlaw_band` variant (A = band value at peak, which for EUV is unavailable in real time —
  so it is reported only as an oracle reference).

## 2026-11-03 — Flare list sources
- 2003-01 … 2017-04: NGDC GOES X-ray reports (M/X only; `goes-xrs-report_YYYY.txt`, 2015 from the
  "modifiedreplacedmissingrows" file). 2017-04 … 2023-03: HEK `FRM_NAME = SWPC` records
  (identical to NGDC where they overlap: 2003 → 2395 records / 180 M+X in both).
  Duplicated rows in the 2006 NGDC file are dropped.

## 2026-11-03 — Hyperparameters: use Table 3 + Zenodo scripts; depth typo for FCN-w1
- Table 3 lists FCN 9.0–9.9 nm as "[268, 1072, 268], depth 3"; 3 expansions would be 5 hidden
  layers. The hidden-dim vector is authoritative (depth 2 pattern [h1,h2,h1]).
- FCN-w0 uses [230, 922, 230] (int(180·1.28098)=230, int(720·1.28098)=922).

## 2026-11-03 — CV96 finding: paper-configured recurrent models collapse with ~78 training events
- **Observation [ours].** With the paper's preprocessing (raw log10, no standardisation) and Table-3
  hyper-parameters, GRU/LSTM/RNN trained on CV96 folds predict nearly the same curve for every
  event (cross-event CC of log10 w0 at 1 min after the peak: GRU −0.18; pooled CC w0 0.34 vs
  0.91 [zenodo]); FCN-w3 (BatchNorm, lr 2.4e-3) diverges in several folds (val MSE up to 2.6).
  FCN-w0 (anchor/residual form) and the power-law baseline do reproduce the authors' scores.
- **Decision.** Keep these runs as the faithful "paper-configuration" reproduction
  (`results/repro_cv96`), and test preprocessing as Phase-2 item 5: event peak normalisation
  (`target: peaknorm`: input and targets relative to the last input sample, the peak level fed
  as a standardised static feature) and global z-score (`standardize: true`).
- **Phase-3 base (greedy, by the pre-specified rule below; `results/improve/ledger.csv`, rows P2-5).**
  GRU: peak normalisation ADOPT vs paper preprocessing; z-score and hybrid normalisation are then
  *neutral* vs peak normalisation (MAPE4 17.35 / 17.83 vs 16.94, n.s.) → base **GRU-peaknorm**.
  FCN: peak normalisation ADOPT, and z-score ADOPT vs peak normalisation (MAPE4 16.37 vs 19.05,
  p = 1e-4) → base **FCN-z**. All Phase-3 GRU-family / modern-architecture variants are compared
  with GRU-peaknorm, FCN variants with FCN-z ("one change at a time";
  configs generated by `scripts/make_p3_configs.py --base peaknorm --base-fcn zscore`).

## 2026-11-03 — Pre-specified Phase-3 decision rule
- Primary metric: per-event MAPE averaged over the 4 bands (scale-free; closest to the log-MSE
  training loss). Secondary: per-event band-mean CC, pooled CC (paper metric), log10-RMSE.
- Test: two-sided Wilcoxon signed-rank on the 96 paired per-event primary values (3-seed mean);
  paired bootstrap 95 % CI (10 000 resamples). ADOPT if the primary improves with p < 0.05 and
  band-mean per-event CC does not degrade significantly; REJECT if significantly worse;
  otherwise neutral. No multiplicity correction across ideas (exploratory); stated in REPORT.
- Amplitude augmentation (I-3) is not applicable to peak-normalised inputs (the level is a separate
  feature); I-3 uses input-noise augmentation (0.01 dex) instead.

## 2026-11-03 — I-9 foundation models not executable
- The weights of Chronos/TimesFM/Moirai (`huggingface.co`) could not be downloaded in the analysis environment;
  the PyPI packages do not ship weights. I-9 is reported as not done; mirrors of weights on
  third-party GitHub repositories were not used (untrusted provenance).

## 2026-11-04 — Final model and I-10 scope
- The individually adopted GRU-line changes (45-min window, input noise, rise-time feature) did **not**
  combine (GRU-final vs GRU-peaknorm: MAPE4 16.81 vs 16.94, n.s.; per-event CC worse, p = 4e-4) and the
  GRU-final ensemble is not better than the FCN-z ensemble (16.31 vs 16.17, n.s.; CC worse). The final
  model is therefore the **FCN-z 5-seed ensemble** (paper FCN architectures/hyper-parameters, global
  z-score of log10 input and per-band targets, mean of 5 seeds in log space; 90 % intervals by
  cross-fold conformal prediction).
- I-10 was run on the best pipeline (FCN-z; per-fold, per-band TPE, 50 trials, inner validation only):
  MAPE4 16.24 vs 16.37 (p = 0.099, neutral) → not adopted. The queued GRU-final Optuna run (~2 h CPU) was
  cancelled because GRU-final itself was rejected.

## 2026-11-04 — Full-precision Zenodo packages
- The authors' packages `0_data.zip`, `1_tune_train.zip`, `2_evaluation.zip`, `3_figures.zip` were
  extracted to `external/zenodo/raw/` (path-traversal check, no scripts
  executed). `all_model_predictions_same_ytrue.npz` is read with `allow_pickle=False`; its `event_names`
  entry is a pickled object array and is **not** unpickled (untrusted) — the event order comes from the
  checksummed transcription and is confirmed numerically (max |Δ| 5.0e-5 dex = half the rounding step).
- Uses: evaluation-code validation on all 8 authors' models (machine precision), pooled-CC bootstrap,
  physics/lead-time/D-RAP analyses of the authors' predictions. The CV96 experiments were not re-run on
  the full-precision arrays (difference 0.0115 % ≪ model errors).
- The zips contain no training/validation data, so the CV96 protocol remains necessary.

## 2026-11-04 — Open-data rebuild of the training set
- **Decision.** Write downloaders (`scripts/download/`) for LaTiS `fism_flare_hr`, NOAA GOES XRS
  1-min science files and F10.7, run on a server with access to those hosts.
- **Calibration before bulk download.** The 96 Zenodo test events fix the authors' construction:
  `calibrate.py` picks the alignment shift (±20 min), the real-time peak rule, the 0.1-nm bin set and
  aggregation per band, and the input series (FISM2 0.1–0.8 nm vs GOES XRS-B, flags, ×0.7) by minimum
  log10 error; PASS requires < 0.01 dex median error and ≥ 85 % exact peak-rule agreement.
- **Observation [zenodo].** The authors' peak minute is the FISM2 0.1–0.8 nm maximum within ±15 min for
  86/96 events (10 events peak 1–3 min earlier), so no real-time rule may reproduce every alignment.
- **Protocol "full".** `src/data/pool.py` + `run.py --protocol full`: test = the authors' Zenodo arrays
  (indices 0..95), train/val = all other downloaded events passing the isolation rule, 8:1 within class
  in time order; results in `results/<exp>_full*/`. Options: input variant, leakage exclusion window.
- **Verification.** Offline mock servers built from the Zenodo arrays with a hidden rule: calibration
  recovers bins, aggregation, shifts, input and flag handling; build matches Zenodo to < 1e-5 dex;
  resume and no-overwrite tested. CV96 results unchanged (Power-law OOF identical).

## 2026-11-04 — Server bundle; GOES short-gap interpolation
- `deploy/server/` (shipped as `server_bundle_v1/`): numbered bash steps 0–6 (network check, venv setup +
  offline tests, smoke test in a separate download root, calibration with PASS gate, full download +
  versioned build `events_latis[_vN].npz`, full-protocol experiments E1–E5, results archive).
  Verified end to end against a local mock HTTP server before the first server run.
- GOES XRS input windows: interior gaps of <= 5 minutes (flagged/missing minutes) are interpolated
  log-linearly; longer gaps leave the variant unavailable. If an input variant is still missing for some
  test events, those events keep the Zenodo input and the count is printed (instead of aborting).
- Server URLs can be overridden with FISM_LATIS_URL / FISM_GOES_LEGACY_BASE / FISM_GOESR_BASE / FISM_F107_URL.

## 2026-11-05 — Real-server calibration: X-ray band not reproducible; score on two test sets
- **Evidence [ours, server download of the 96 test events].** LaTiS timestamps are on whole minutes and
  complete; EUV bands match the Zenodo targets to 0.2–1 % (median log offset w1 +0.002, w3 +0.0015 dex with
  10 bins = 0.1 nm x lower edges 9.0..9.9 / 30.0..30.9). The 0.1–0.8 nm band does not: Zenodo / public
  FISM2 (sum x 0.1 nm over 0.15..0.75 nm) is 10^0.144 ≈ 1.39 in the decay, ≈ 1.5–1.6 around the peak
  (time-varying, so not a bin/aggregation convention), and the authors' peak minute lies 1–2 min before
  the public-FISM2 maximum in most events. GOES XRS-B matches the Zenodo shape worse (median MAD 0.036 vs
  0.005 dex). Likely a different FISM2 version of the X-ray part (inference, not verified).
- **Build rules kept as auto-calibrated** (w0 'inner'+mean ≈ x1.43, EUV per calibration, peak = GOES-B
  argmax ±15 min, input = FISM2 w0): rebuilt test events then differ from Zenodo by median |Δ| 0.016 dex (X),
  0.013 / 0.005 / 0.009 / 0.002 dex (w0..w3); 96/96 test events rebuilt.
- **Protocol "full" now scores every model on two test sets** in one training run: the authors' Zenodo
  arrays (comparison with the paper; contains the construction mismatch) and the same 96 events rebuilt
  like the training data (`*_latis.csv`; like-for-like). Conclusions about models should rest on the
  rebuilt set; differences between the two sets measure the data-version effect.

## 2026-11-05 — Full-scale re-test of the Phase-3 ledger; final model kept
- **Leakage (C1).** Removing the 163 train/val flares within 24 h of a test flare costs 0.7–0.9 MAPE4 pt
  more than removing 163 random flares (GRU-pn +0.91, p < 1e-4, Holm 0.016; FCN-pn +0.65, p 0.03) ->
  the paper's interleaved time split is optimistic by about that amount; AR-level splits are recommended.
- **Ledger at full scale (rebuilt96; Holm over 44 tests).** Rejected: 15-min window, modern architectures
  (+0.67–0.85 vs FCN-z), FCN residual (+1.4), joint FCN (+1.2), static features for GRU (+1.0), FCN Optuna
  (+0.74–0.81, validation overfit). Neutral/small: GRU residual (−0.3), input noise (−0.2..−0.4), rise-time
  feature (−0.35), 30/45-min windows, combined GRU-final (−0.2 vs GRU-pn-ens5). Early forecast: peak error
  0.22–0.24 dex, w0 APE 41–47 % — unchanged conclusion.
- **Final model.** FCN-z 5-seed ensemble kept (MAPE4 13.78/13.41, APE 7.4 %). GRU-z-optuna 3-seed ensemble
  is best on CC4 (0.812) and MAPE4 (13.28/12.87) but not significantly better on MAPE4 (p 0.14–0.17) and
  worse on APE (+0.8, p < 0.01); proposed as a complementary member (average of both) for later work.
- **Applications on full-scale predictions.** D-RAP MAE 0.10 dB for FCN/FCN-z/GRU-pn (peak error ≤ 0.01 dB),
  0.11 dB for RF/GRU (peak 0.11 dB); adaptive cross-conformal 90 % coverage 0.89–0.90 (X-class w0 0.76–0.78).
  Validation predictions are not stored by the runner, so split conformal is left for the next run.

## 2026-11-09 — GPU follow-ups (G1–G4) and the final model
- Training now runs on the least-loaded healthy GPU by default (FISM_DEVICE=auto); CPU/GPU agreement measured on
  p3_ensemble: FCN-z seed-mean dMAPE4 −0.04/−0.07 (n.s.), GRU-peaknorm −0.36 (p < 0.001, GPU better) — within the seed
  spread, conclusions unchanged. GPU runs are repeatable (identical numbers across two runs).
- AR-level exclusion removes 47 % of the training events but costs GRU-pn only +0.23 MAPE4 (vs +1.28 for the ±24 h rule);
  the time-neighbour effect is therefore attributed to same-epoch events rather than AR identity (provisional: the
  size-matched random control for N=516 is still missing — 8_run_gpu_followups.sh could not read the base event count
  because run_logs/5_experiments.log was not in the new bundle folder; fixed to fall back to the log of step 8 itself).
- GRU-z Optuna with 100 trials / 5 seeds equals the paper hyper-parameters (dMAPE4 +0.04/+0.12, n.s.) and is worse than the
  30-trial result by 0.38 (p 0.02): the 30-trial edge was validation-selection noise. Optuna re-tuning: no gain (both models).
- Final model kept: FCN-z 5-seed ensemble (MAPE4 13.70/13.32, APE 7.5 %). Operational alternative: the average of the FCN-z
  and GRU-z ensembles (MAPE4 13.39/13.02, CC4 0.80, APE 7.7–8.0 %; vs FCN-z p = 0.10).

## 2026-11-09 — Closing runs R1–R3 (bundle v4 / step 9)
- AR-level exclusion vs size-matched random removal (516 events x3): GRU-pn −0.04/−0.23 (n.s.), FCN-pn +0.95/+0.67 (p 0.01/0.08).
  Same-AR flares are not a leakage source for GRU; the ±24 h effect (+0.9–1.0 beyond random) is attributed to same-epoch events.
  Draw-to-draw spread of the random removals is 0.9–1.8 MAPE4 pt — training-set composition matters as much as model choice.
- Split conformal (validation-calibrated, 122 events): adaptive 90 % intervals cover 0.89–0.97 on the test events for all four
  final candidates; X-class w0 0.78–0.85; D-RAP dB intervals 0.86–0.94. Recommended operational interval: conf-norm on the
  GRU-peaknorm ensemble or the FCN-z+GRU-z average.
- Operational recommendation: average of the FCN-z and GRU-z ensembles (MAPE4 −0.49 vs FCN-z, p ≈ 0.01; APE unchanged);
  single-model final remains the FCN-z 5-seed ensemble. The analysis glob bug (random-control pattern matched rnd163s1) is fixed.

## 2026-11-09 — Paper-1 (operational re-design) experiments added
- Code: `--input goes_conv|goes_convnn` (GOES XRS-A/B -> FISM2 0.1–0.8 nm converter, src/models/converter.py, fitted on
  the training events; validation RMSE printed), spec option `issue_delay` (input window ends k min after the peak;
  configs/paper1_delay.yaml), notebooks/peak_detector.py (causal k-minute-no-new-maximum rule on GOES-B, latency
  distribution), split_conformal_full.py `--cond peak|class`, notebooks/analyze_paper1.py, deploy/server/10_run_paper1.sh.
  Offline-tested on the mock bundle end to end; 28 tests pass (31 after patch_v10).
- Server run 2026-11-09 (patch_v9, WORKERS 8 on a shared GPU): P1/P2a hit CUDA out of memory and P2b died with
  "Cannot re-initialize CUDA in forked subprocess". Fixes: per-job OOM -> retry on the CPU; auto device choice needs >= 2 GB
  free and spreads workers; the FCN converter is fitted on the CPU in the parent. 29 tests.
- Rerun (patch_v9, WORKERS 4): P1/P2a/P3 complete; P2b died with "Unable to handle autograd's threading in combination
  with fork-based multiprocessing" and its workers had silently fallen back to the CPU. Root cause: fitting the FCN converter
  in the parent runs optimizer steps (torch.cuda.is_available() inside Adam poisons later forks for CUDA) and backward()
  (autograd starts its device threads, which forbids fork). **patch_v10:** the FCN converter is fitted in a separate
  'spawn' process and only the weights come back (converter.py); GPU choice reads free memory/utilisation from nvidia-smi
  matched by UUID (no CUDA context on every GPU, immune to CUDA_VISIBLE_DEVICES index shifts; the broken GPU is not
  enumerated by CUDA, so `CUDA_VISIBLE_DEVICES=0,1,3,4` had silently selected GPUs 0,1,4); `FISM_REQUIRE_GPU=1` (default
  in the bundle) waits for a free GPU instead of training on the CPU; OOM retries once on a re-selected GPU, then CPU;
  `FISM_MP_CONTEXT=spawn` escape hatch; `SKIP_GPUS=<nvidia-smi index>` builds a UUID-based CUDA_VISIBLE_DEVICES.
  Regression test runs the converter + forked workers in fresh interpreters.
- **patch_v11 (same day):** notebooks/split_conformal_full.py read the validation predictions of the GOES-input experiments
  against the DEFAULT event pool; an input variant drops the train/val events without that input, so the validation rows
  shift and the residuals were mis-aligned (GOES models: conf-norm coverage 0.99, width 1.3 dex in results_full_20261109_1520).
  The pool is now loaded per input (parsed from the exp id `_in-<input>`), with a check that the validation rows are that
  pool's validation split. P5/P6 rerun with the patch. 32 tests.
## 2026-11-11 — v6 all-GPU rerun adopted; replicate-aware testing; patch_v12
- **Base code = server_bundle_v6** (derived from v11): FCN converter fitted in a spawned child AND its forward
  pass in the parent in pure NumPy (scipy erf GELU), so the parent never imports torch before the workers fork; conformal pool per
  input. Reviewed line by line against v11 and kept; the additions on top are listed below. 33 tests.
- **The v6 clean run (results_full_20261111_0151) is the reference result set.** Verified independently: no failed step, no CPU
  fallback, all prediction files complete, summary and D-RAP tables recomputed from the arrays to 3e-14.
- **Testing protocol changed.** The paired sign-flip test on event MAPE4 conditions on the trained models; a 3-seed mean moves by
  +-0.3 pt (1 sigma) when the configuration is trained again (measured: CPU-era run vs v6 GPU run). Every comparison is now tested
  with a hierarchical bootstrap over test events (paired) and training replicates (notebooks/replicate_tests.py). Replicates:
  CPU-era + GPU seeds for E/C experiments (6), 5 seeds for P experiments, random-exclusion draws pooled. Rule: no claim for a
  difference below 0.5 pt from 3-seed means; a hyperparameter search is adopted only after the search itself has been repeated.
- **Withdrawn:** the recommendation of the FCN-z + GRU-z-optuna ensemble average as the operational model (its -0.5 pt came from
  one lucky 30-trial search; the repeated search selected other hyperparameters at the same validation loss and tested 1.2 pt worse).
  Final model unchanged: FCN-z 5-seed ensemble; GRU-z 5-seed ensemble is the equal-MAPE alternative.
- **Delayed-issuance models are scored on lead >= k only**, also for D-RAP (analyze_paper1.py, replicate_tests.py section 6,
  Fig. 8). The server table of the v6 run scored them on minutes 0..179, which include minutes given as input.
- **Conformal interval width** in conformal_split*.csv (`width_norm_*_dex`) is the full width (upper - lower), not the half-width;
  texts and the Fig. 7 axis label corrected.
- **patch_v12 (for server_bundle_v6):** env.sh counts cores without OMP_NUM_THREADS (the 10 h run had WORKERS=1: nproc obeys
  OMP_NUM_THREADS, which env.sh exports as 1 before the step scripts source it again); 6_pack_results.sh also packs
  data/processed/events_latis*.npz and events_test96_rebuilt.npz (validation targets, needed for interval plots);
  10_run_paper1.sh: P4b (detector v2, `--require-rise`) and `--save-intervals` on P5; peak_detector.py v2 rule;
  split_conformal_full.py `--save-intervals`; analyze_paper1.py lead >= k scoring; replicate_tests.py; paper1_figures.py.
## 2026-11-12 — patch_v12 run checked (results_full_20261112_1001); Fig. 2 with conformal intervals; detector v2
- The patch run (P4b, P5, P6; no training; 2 min 21 s; WORKERS 8, so the nproc fix works) is adopted. Compared file by file with
  the 0151 bundle: 1,583 files byte-identical (all prediction arrays; the conformal tables were recomputed to
  the same bytes); changed: paper1_drap.csv (lead >= k scoring, equals the locally corrected table to 1e-16) and logs; new: detector
  v2 tables, conformal_intervals_none.npz, data/processed/events_latis.npz and events_test96_rebuilt.npz.
- Interval arrays verified independently (centre = ensemble mean of the prediction arrays to 2e-7 dex; coverage and width recomputed
  from lo/hi equal the tables). The delivered dataset reproduces the split (988/122/96; GOES inputs 982/122/95), the stored validation
  rows and the converter validation RMSE (0.0894 / 0.0819 dex).
- **Fig. 2** now shows the operational forecast (GRU-z behind the FCN converter) with its 90 % conformal interval, the FISM2-input
  forecast of the same architecture and both input windows, on the rebuilt test set (median M-class and largest X-class event).
  `paper1_figures.py` uses data/processed/events_test96_rebuilt.npz (server file, has the GOES input) by default.
- **Detector v2 does not solve the premature declarations** (19.8 % -> 16.3 %): only 33 of the 123 window-edge events move to the
  correct peak. Read as a property of the flares (one in six has an earlier maximum above M1 within the preceding hour), not of the
  rule; the operational design therefore keeps the detector armed and re-issues. The end-to-end evaluation of that re-arming scheme
  is not done (it needs the GOES streams on the server).
- Counts corrected in the texts: 8 X-class test events (not 9), 9 X-class validation events (was "about ten").
## 2026-11-12 — patch_v13: armed peak detector (P4c)
- Why: the first-declaration rules (P4, P4b) stop at their first declaration; 16 % of the flares are declared at an earlier
  maximum. From the pre-peak GOES windows of the delivered dataset: in 221 of those 262 events the flux later exceeds the declared
  maximum before the dataset peak, in 32 it stays below but had fallen to half in between, in 8 neither. So a detector that keeps
  running should declare the flare's own maximum for about 98 % of the flares; the server run measures it with the full streams.
- Rule (notebooks/peak_detector.py detect_all): candidate = running maximum of the episode; declared when >= M1, no new maximum
  for k minutes, and >= rise_factor x the minimum of the episode BEFORE the maximum (so a decaying tail is never declared; the
  P4b rule compared with the running minimum of the whole stream and was satisfied by the decay itself, which caused its late
  declarations). A larger later maximum is declared again ('update'); a rise by rise_factor from the post-maximum minimum while
  still below the earlier maximum starts a new episode ('new').
- "Own peak declared" = a declaration whose maximum lies within +-5 min (--own-tol) of the dataset peak; latency = minute of the
  last such declaration. All declarations are written (peak_detector_declarations_v3.csv), so other definitions can be applied later.
- The median filter of this mode is causal (trailing 3 min). The centred filter of P4/P4b uses the next minute, so the "1 min
  median latency on a 3-min median" reported for them is not a real-time gain (real-time latency 2 min).
- `--save-streams` exports the XRS-A/B streams of all events for [peak-60, peak+180) (float64): detector settings can then be
  explored without the server, and the post-peak GOES minutes needed for GOES-driven delayed-issue models are available.
- Bundle: 10_run_paper1.sh step P4c; `LIGHT=1` packs only results/full_server and the logs (results_tables_<date>.tar.gz) for
  runs that train nothing; env.sh no longer fails under `set -e` when the repo zip is absent. 41 tests.
## 2026-11-12 — patch_v13 run checked (results_tables_20261112_1150); armed detector adopted in the paper
- Run: P4c only, 72 s, failed: none. The 25 existing tables and the interval arrays are byte-identical to the 1001 bundle.
- The exported streams reproduce the server's 21,841 declarations exactly, the v2 table completely and the v1 table except 18 of
  16,040 latencies (5 events): the 2026-11-10 code evaluated the declaration also on minutes without data, the current code skips
  them. All detector numbers in the texts are therefore recomputed from the streams with the current code
  (notebooks/detector_analysis.py -> peak_detector_rules.csv); v1 on-time 80.0 %, v2 82.4 % (unchanged to 0.1 pt).
- Reported as a trade-off, not as one tuned setting: rise factor 2 (the value fixed before the run) declares the flare's own
  maximum for 96.7 %, 1.5 for 98.4 %, 1.2 for 99.3 %, with 12.1 / 13.4 / 14.6 % of the flares having earlier declarations.
  The lower factors were evaluated after seeing the result for 2, on the same events; the text says so by giving all three.
- "End-to-end" score = existing GOES-driven predictions scored from the declaration minute on (test flares: all declared at +2 min;
  MAPE4 14.34 vs 14.31). Not done: retraining with input windows that end at the declared time; forecasts for the 398 flares that
  fail the benchmark's isolation rule (25 % of all M/X flares, never used for training or testing).
- Fig. 6d is now the outcome share per rule (five rules) instead of a latency histogram: the armed detector's latency is a single
  value (k), so a histogram carries no information.
## 2026-11-12 — patch_v14: delayed issue with real-time inputs (P7), forecasts for the flares outside the isolation rule (P8)
- Why P7: the gain of issuing k minutes after the peak (P3: -0.5..-1.0 pt at k=5, -1.2..-1.4 at k=10) was measured with the
  FISM2 input, whose minutes 1..k are the target itself and are not available in real time. The two-stage operation in the paper
  (issue at the declaration, re-issue later) therefore rested on an input the operational system does not have.
- Mechanism: `run.py --issue-delay k` (FISM_ISSUE_DELAY) shifts every input window in src/data/pool.py to minutes -59+k..k.
  FISM2 inputs take the 0.1-0.8 nm target minutes 1..k (identical to the spec option `issue_delay` of P3); GOES inputs take the
  GOES minutes 1..k from data/processed/events_goes_post.npz, built by scripts/build_goes_post.py from the streams exported by
  P4c with the gap rule of the input windows (log-linear interpolation over <= 5 missing minutes). The converters are fitted on
  the shifted windows (GOES minutes -59+k..k -> FISM2 minutes -59+k..k). Targets stay minutes 0..179; scores are read for lead >= k.
- Split: the train/val assignment is made on the events usable at the peak, for every delay; events without the minutes after
  the peak are dropped afterwards. Real dataset, GOES inputs: no event lost at k=2, 2 training events at k=5, 4 at k=10; the 122
  validation events and the 95 rebuilt test events are the same for every delay. (Re-deriving the 8:1 assignment after dropping
  events moves every later event between train and val; the converter's validation RMSE then changed from 0.088 to 0.071 at k=5
  only because other events were in the validation set.)
- P7 runs: k = 2, 5, 10 x {FISM2, GOES through the FCN converter, GOES XRS-B direct} x {GRU-z, FCN-z} x 5 seeds (90 trainings).
  The FISM2 arm with seeds 0-2 repeats P3 and checks the new mechanism on the server; seeds 3-4 are new. The linear converter is
  left out (the FCN converter is the operational candidate).
- Why P8: 398 of the 1,604 M/X flares with a GOES series (25 %) fail the isolation rule (another >= M1 flare within 60 min
  before or 180 min after the peak) and were never forecast. `run.py --predict-extra` (FISM_PREDICT_EXTRA) appends those with a
  complete input and target (395 for the FISM2 input, 393 for GOES) after the train/val rows; they are predicted, never fitted,
  and saved as oofN_<model>_s<seed>.npy with extra_rows.csv (pool index, dataset row, event). The peak-issued GRU-z / FCN-z are
  retrained with the same seeds for the FISM2 input and for GOES through the FCN converter (20 trainings); their test predictions
  must agree with the existing runs (p2_zscore_full_gpu5, p2_zscore_full_in-goes_convnn_p2nn), which checks the run.
- Analysis fixed before the run:
  P7 - MAPE4 on the rebuilt test set for lead >= k, delayed vs peak-issued model of the same input and architecture, hierarchical
  bootstrap over events (paired) and seeds; 18 comparisons, Holm-adjusted within this family. Secondary: band w0 alone, D-RAP
  absorption error on the same leads, conformal coverage and width of the delayed models. The paper keeps the re-issue step only
  if the GOES-driven delayed models improve on the GOES-driven peak-issued models (interval excluding 0); otherwise the text says
  that the gain needs the FISM2 input.
  P8 - MAPE4 of the flares outside the isolation rule against the test flares per input and architecture (bootstrap over events),
  split by the reason for failing the rule (earlier flare within 60 min / later flare within 180 min / both) and by class;
  coverage of the conformal intervals calibrated on the (isolated) validation events. Caveat to report: these flares come from the
  same years as the training flares; the subset whose [-60, +180] min span overlaps no train/val event is reported separately.
- Checked before delivery (real dataset): the peak-issued pools of the five inputs are bit-identical to the previous code (sha of
  X, Y, names, split); previous and current code give identical predictions (GRU-z, FCN-z, GRU-z-d5; 2 epochs, CPU);
  `--issue-delay 5` equals the spec option `issue_delay: 5`; `--predict-extra` leaves the test and validation predictions unchanged.
- Bundle: 10_run_paper1.sh steps P7 / P8 (`go_train` repeats a failed training call once with FISM_MP_CONTEXT=spawn; finished
  models are kept), `run.py --models` selects specs of a config, LIGHT=1 also packs the folders of the steps that trained
  (PACK_EXTRA), 11_verify_all_results.sh checks P7 / P8 once started. 46 tests.
## 2026-11-12 — patch_v14 run checked (results_tables_20261112_1355); P7 / P8 analysed as planned and adopted
- Run: P7 + P8, 28.5 min, 110 trainings on GPU (8 workers; 88 worker processes, all cuda), failed: none, no retry. The 30 existing
  tables are byte-identical to the 1150 bundle. FISM2 arm, seeds 0-2: 54/54 files bit-identical to P3. P8 validation predictions
  bit-identical to the existing peak-issued runs; test predictions within 2e-6 dex (inference batch now includes the extra rows).
  events_goes_post.npz, converter validation RMSE and the splits equal the local values.
- Analysis = notebooks/analyze_p7_p8.py, as fixed before the run. Its conformal routine reproduces the server's interval arrays to
  2.4e-7 dex; the headline numbers were recomputed with plain numpy (no project helpers) and agree to all printed digits.
- Decision rule applied: "keep the re-issue step only if the GOES-driven delayed models improve on the GOES-driven peak-issued
  models (interval excluding 0)". Outcome is mixed and is reported as such: at +10 min FCN-z improves with every input (Holm-adjusted
  p 0.002-0.02); GRU-z does not in MAPE4 (-0.68 [-1.83, 0.56] behind the FCN converter) but does in the 0.1-0.8 nm band (-6.0
  [-9.8, -2.2], adjusted p 0.03). Nothing at +2 min; nothing significant after adjustment at +5 min. The paper therefore states a
  sequence "first issue at detection with the peak-issued model, re-issue at +10 min", and no longer a 5-min update.
- The 5-seed FISM2 arm replaces the 3-seed P3 numbers for FCN-z / GRU-z in the texts. The GRU-z gain at +5 min that was reported as
  significant with 3 seeds (-0.77 [-1.52, -0.01]) is -0.62 [-1.39, 0.15] with 5 seeds; the text was corrected.
- Not pre-specified, labelled as such in the texts: delayed GOES-driven models vs the peak-issued FISM2-input model on the same
  leads (+0.4 / +0.3 pt at +10 min, intervals include 0).
- P8: the prevalence-weighted MAPE4 (about 16.0 % for the operational GRU-z) uses the share 398 / 1,604 and the seed-mean MAPE4 of
  the two groups (14.66 and 20.16); it is an estimate for flares with a complete input and target.
- Figures: new Fig. 9 (figures/paper1_v5/fig9_reissue_nonisolated.png); Fig. 8a now shows the GOES-driven delayed rows (P7) instead
  of the FISM2-input ones; Figs. 2-7 are byte-identical to v4. 47 tests (new: the analysis script on synthetic predictions).
## 2026-11-12 — recalibration of the intervals for all flares (design fixed before computing anything)
- Question: the 90 % intervals are calibrated on the 122 validation flares, which are all isolated. On the flares outside the
  isolation rule they cover 73-75 % of the 0.1-0.8 nm minutes (P8). Does a calibration set that represents all M/X flares restore the
  nominal coverage, and what does it cost in width? No training: the predictions of P8 (oofN_*, never used for fitting) are used.
- Which condition is allowed: whether a later flare will follow inside the 180-min window is not known at the issue time, so the
  intervals cannot be conditioned on the isolation status. One interval is calibrated for the whole flare population.
- Method: the existing conf-norm rule (per band x 30-min block, scores |r| / (sd + s0), 90 % quantile over the event-minutes of the
  calibration events) with event weights. Calibration events = the 122 validation flares (isolated) + flares outside the rule; the
  weights make the weighted share of the flares outside the rule equal to their share of all flares, p = 398 / 1,604 = 0.248 (each
  validation flare (1 - p) / 122, each flare outside the rule p / n). Weighted quantile = smallest score whose cumulative weight times
  n / (n + 1) reaches 0.90, which is the existing quantile when the weights are equal (checked in the code). s0 = median ensemble
  spread of the calibration events (unweighted).
- Evaluation without reuse: the flares outside the rule are grouped into clusters (successive peaks less than 24 h apart stay in one
  cluster), the clusters are assigned alternately to two folds in time order, and each fold is evaluated with intervals calibrated
  on the validation flares + the other fold. The isolated test flares are evaluated with the validation flares + all flares outside
  the rule. Reported: coverage and width (0.1-0.8 nm and four-band mean) for the test flares, the flares outside the rule (all, by
  reason, by 30-min block) and the population-weighted mean (1 - p) x test + p x outside, for the isolated-only and the all-flare
  calibration; four model / input combinations of P8, the operational GRU-z being the headline.
- Sensitivity, reported next to the main result: folds = first / second half of the clusters in time; unweighted pooling.
- What goes in the paper: the all-flare calibration is described as the operational calibration if the population-weighted
  0.1-0.8 nm coverage of the operational models lies in 0.88-0.92; the per-group coverages and the width are reported whatever they
  are (one interval for both groups is expected to over-cover the isolated and under-cover the other flares).
## 2026-11-12 — recalibration for all flares: result and decision
- Computed exactly as fixed above (notebooks/analyze_p7_p8.py -> paper1_p8_recal.csv, paper1_p8_recal_blocks.csv); 127 time clusters,
  folds 214 / 179 flares (GOES input). With equal weights the routine returns the existing intervals (asserted in the code). The
  headline numbers were recomputed with separate code (own weighted quantile, own folds) and agree to four decimals.
- Result (0.1-0.8 nm coverage, isolated-only -> all-flare weighted): population-weighted 0.895 -> 0.922 (operational GRU-z),
  0.879 -> 0.909 (operational FCN-z), 0.871 -> 0.907 and 0.868 -> 0.908 (FISM2 input); flares outside the rule 0.73-0.75 -> 0.78-0.80;
  isolated test flares 0.91-0.95 -> 0.95-0.97; population-weighted width +17.5 / +17.7 / +22.6 / +20.7 %.
- Adoption rule (population-weighted coverage of the operational models in 0.88-0.92): met by FCN-z (0.909), missed by GRU-z by
  0.002 (0.922). Applied as written: the all-flare calibration is NOT adopted as the operational calibration. It is reported as a
  result, because it shows what calibration can and cannot do: it restores the marginal coverage and cannot cover flares that have
  not happened at the issue time.
- Consequence for the text: the interval is stated to be valid until the detector declares the next flare; the remedy named in the
  paper is a new issue at that declaration. That re-issue scheme itself is still not evaluated (next optional step: score only the
  minutes before the next declaration).
- The calibration compared here is one interval for the whole population. A calibration conditional on something known at the issue
  time (e.g. an earlier flare within the last hour) was considered and not run: the under-coverage sits in the flares followed by
  another flare (0.69-0.71), which no quantity available at the issue time identifies.
## 2026-11-12 — re-issue at the next declaration: evaluation design (fixed before computing anything)
- Question: the text now says that a forecast (and its interval) is valid until the detector declares the next flare, and that a new
  forecast is issued then. How good are the forecasts over the minutes for which they are in force? No training; P8 predictions,
  the retained (isolated-flare) intervals and the exported GOES streams are used.
- Next declaration: the armed detector of P4c (k = 2, rise factor 2, raw XRS-B stream of [peak-60, peak+180)) is run on each flare;
  the next declaration is the first declaration whose maximum lies more than 5 min after the dataset peak (the definition of "another
  flare declared inside the window" used in P4c). T = its declaration minute relative to the dataset peak, 180 if there is none.
  The forecast issued at the peak is in force on minutes 0..T-1 ("valid minutes"); the later minutes belong to the next forecast.
- Scores on the valid minutes, pooled over event-minutes (each minute a forecast is in force counts once): MAPE4, MAPE of the
  0.1-0.8 nm band, coverage of the 90 % interval calibrated on isolated flares (0.1-0.8 nm and four-band mean). MAPE is computed per
  seed and averaged, coverage uses the 5-seed ensemble interval, as elsewhere. Groups: isolated test flares; flares outside the rule
  (all, by reason); all flares = the two groups combined with weights (share of the group) x (mean number of valid minutes), share of
  the flares outside the rule p = 398 / 1,604. Intervals: bootstrap over events (and seeds for MAPE). Flares without a GOES stream
  (1 test flare, 2 flares outside the rule for the FISM2 input) are left out. Reference: the same flares on all 180 minutes.
- Also reported: share of flares with a next declaration, median T, share of event-minutes in force; the last 10 valid minutes
  before the next declaration against the earlier valid minutes (the rise of the next flare precedes its declaration); 30-min
  blocks; rise factors 1.5 and 1.2 as sensitivity.
- Not evaluated here: the forecasts issued at the next declarations. A next flare that is a catalogued M/X flare is one of the flares
  already scored (as a test flare or a flare outside the rule, with its own valid minutes); declarations of maxima that are not in the
  catalogue have no forecast in this data set.
- Text rule: the paper says that re-issue restores the nominal coverage only if, for both operational models, the 0.1-0.8 nm
  coverage on valid minutes is >= 0.88 for all flares and >= 0.85 for the flares outside the rule. Otherwise it says that re-issue
  reduces but does not remove the loss and gives the numbers.
## 2026-11-12 — re-issue at the next declaration: result and decision
- Computed as fixed above (notebooks/analyze_p7_p8.py -> paper1_p8_reissue*.csv). The next declarations equal those of the server's
  declaration table (peak_detector_declarations_v3.csv, k = 2, raw) for all 488 flares; the pooled scores were recomputed with
  separate code and agree to four decimals. No flare had to be left out for a missing stream.
- A bug found while reading the first output and fixed before any number was used: the bootstrap of the full-window rows resampled a
  single event (the mask of the uncensored case had one row), so their intervals did not contain the point values. Point values
  were unaffected. The test now requires every interval to contain its point value.
- Result (0.1-0.8 nm coverage, all 180 min -> minutes in force): flares outside the rule 0.737 -> 0.829 and 0.730 -> 0.827 (operational
  GRU-z / FCN-z), 0.727 -> 0.843 and 0.745 -> 0.853 (FISM2 input); all flares 0.895 -> 0.936, 0.879 -> 0.918, 0.871 -> 0.917, 0.868 -> 0.908.
- Text rule applied as written: all flares >= 0.88 holds for both operational models, flares outside the rule >= 0.85 does not
  (0.829, 0.827). The paper says that re-issue removes more than half of the coverage loss of the flares outside the rule, not all of it.
- Added after seeing the tail table, and labelled descriptive: the minute-by-minute coverage aligned on the next declaration
  (paper1_p8_reissue_aligned.csv, Fig. 10b). It shows where the remaining loss is: the rise of the next flare, which precedes its
  declaration (2 min after its maximum) by 10-20 min.
- Not done: a rule that ends the forecast at the onset of the next rise (the armed detector already tracks the rise from the
  post-peak minimum); forecasts issued at declarations of maxima that are not catalogued flares.
- Figures: new Fig. 10 (figures/paper1_v6/fig10_reissue_validity.png); Figs. 2-9 byte-identical to v5.
## 2026-11-19 — three open points after the seminar deck: design fixed before computing (no server run)
Chosen on 2026-11-19: (1) evidence on the real-time data, (2) the rise of the next flare, (3) the X-class intervals. (1) is a
literature / documentation check and has no analysis design. (2) and (3) use the arrays of the 1355 bundle and the GOES streams only.
### (2) forecast ended at the onset of the next rise (notebooks/analyze_rise_xclass.py -> paper1_onset*.csv)
- Stream: raw 1-min XRS-B (peak_detector_streams.npz), as for the detector. Onset minute O_r of a forecast = first minute t >= 1 after
  the dataset peak with b_t >= r * min(b_0 .. b_t) (running minimum since the peak, the peak minute included); causal, no knowledge of
  what follows. r in {2, 1.5, 1.2} (the set of 2026-11-12).
- The forecast is in force on minutes 0 .. E_r - 1 with E_r = min(O_r, T), T = minute of the next declaration (armed detector, k = 2,
  rise factor 2: the rule of 2026-11-12, unchanged). After the onset no forecast is in force until a new one is issued.
- Scores as on 2026-11-12 (in_force(): pooled over event-minutes in force; coverage CI from an event bootstrap, MAPE CI from events
  and seeds; population combination with the share 398 / 1604 and the minutes in force), groups: isolated test flares, flares outside
  the rule (all / reason), all flares. Models GRU-z and FCN-z, operational input (GOES + FCN converter) and FISM2 input.
- Cost, reported next to every coverage: share of minutes in force; share of flares whose forecast is ended by an onset; and the
  0.1-0.8 nm coverage of the minutes that the onset rule discards (between E_r and T), separately for flares with and without a
  declared next flare. The second case shows how many good minutes are thrown away for rises that never become an M-class flare.
- Text rule: the onset rule replaces the declaration rule in the paper only if, for the largest r of the three that does so, both
  operational models reach a 0.1-0.8 nm coverage in force >= 0.85 for the flares outside the rule and >= 0.88 for all flares, AND
  the isolated test flares keep >= 90 % of their minutes in force (declaration rule: 97.6 %). Otherwise the declaration rule stays
  and the paper gives the trade-off.
### (3) X-class intervals (same script -> paper1_xclass*.csv)
- Events: X-class flares never used for fitting or calibration of the model under test: the test X flares and the X flares outside
  the isolation rule (25 in the data set; those with a complete GOES window for the operational input). Minutes in force under the
  declaration rule, so that a later flare is not counted as a miss of the class. Reference in every table: the M-class flares of the
  same group on the same kind of minutes. Band 0.1-0.8 nm (4-band coverage reported next to it).
- (A) descriptive, current intervals: coverage by class and 30-min block, share of the misses above / below the interval, coverage
  in bins of the peak class (M1-M2, M2-M5, M5-X1, >= X1), mean interval half-width against mean absolute error (dex) by class.
- (B) calibrations compared on these events: M0 current (conf-norm, 122 validation flares, no conditioning); M1 class-conditional
  on the validation flares (9 X); M2 peak-localized: weighted conformal quantile with weights exp(-d^2 / 2h^2), d = difference of
  log10 XRS-B at the peak between the calibration flare and the flare forecast, h = 0.3 dex (0.2 and 0.5 as sensitivity); M3
  class-conditional with a larger calibration set: validation flares + flares outside the rule (their minutes in force only), the
  flares outside the rule evaluated out-of-fold (time clusters of 2026-11-12, two folds), the test flares with all of them.
- "Class deficit" = coverage(M) - coverage(X), pooled over the test flares and the flares outside the rule, minutes in force.
  Text rule: a calibration is called a remedy only if, for both operational models, the deficit is <= 0.03, the M-class coverage of
  the isolated test flares stays within 0.88-0.96, and the X-class coverage CI (event bootstrap) is reported; the width change for
  X and M is given with it. Otherwise the paper keeps M0 and states the X-class deficit with its interval.
## 2026-11-19 — onset rule: first result, and one variant added after seeing it (descriptive)
- Computed as fixed above. Declaration-rule rows equal paper1_p8_reissue.csv exactly (24 rows). Onset r = 2: coverage in force of the
  flares outside the rule 0.829 -> 0.872 (operational GRU-z), 0.827 -> 0.868 (FCN-z), all flares 0.954 / 0.933; but the isolated test
  flares keep 83.1 % of their minutes in force (declaration 97.6 %): 42 % of them have a factor-2 rise from the post-peak minimum that
  never becomes an M-class flare, and the forecast covered 0.92 of the minutes given up there. Text rule applied as written: the
  coverage conditions hold for r = 2, the 90 % condition does not -> the declaration rule stays; the paper gives the trade-off.
- Added now, before computing it, and to be labelled as added afterwards: the same onset rule with a floor on the flux, i.e. onset =
  first minute with b_t >= 2 * running minimum AND b_t >= floor, floor in {C3, C5, M1} (3e-6, 5e-6, 1e-5 W m^-2), to see whether rises
  that stay small can be ignored. C5 is the one looked at first; the other two show the sensitivity. Same scores, same text rule.
- X-class tables computed as fixed above. On the minutes in force the X-class flares are not covered worse than the M-class ones
  (operational GRU-z 0.912 vs 0.860, FCN-z 0.888 vs 0.854; deficit M - X = -0.052 / -0.035), so by the text rule there is nothing to
  remedy and M0 stays. The pre-specified peak-class bins show the weak spot elsewhere: M5-X1 (45 flares) 0.75. Added now, before
  computing it and to be labelled descriptive: the coverage per peak-class bin for each calibration (does a peak-localized or
  class-conditional calibration repair M5-X1?).
## 2026-11-19 — patch_v15: design fixed before the run (P9 training with the flares outside the rule; P10 archive probe)
- Why: two questions a referee will ask. (i) The flares outside the isolation rule have MAPE4 ~20 % with models trained on isolated
  flares only: does training on them help? (ii) All GOES inputs so far are the science-quality archive (3-day latency): what does the
  operational archive look like? (ii) needs the file layout first, so this patch only probes it (P10, no analysis).
- P9 folds: the flares outside the rule that have a complete input and target, in time order, clustered with the 24-h gap rule of
  2026-11-12; fold = cluster index mod 2 (computed in src/data/pool.py, written to extra_rows.csv). Run f (0, 1): training rows = the
  isolated training flares + the flares outside the rule of fold f; validation rows unchanged (122 isolated flares, so early stopping and
  the interval calibration stay as they are); the converter stays fitted on the isolated training flares (identical to P8).
  GRU-z and FCN-z, seeds 0-4, inputs FISM2 and GOES + FCN converter: 2 x 2 x 2 x 5 = 40 GPU trainings.
- Scores: flares outside the rule out-of-fold only (fold 1 - f of run f); isolated test flares from both runs (mean of the two runs
  per seed). Baseline = the P8 runs (same seeds, same events). MAPE4 difference P9 - P8 with the hierarchical bootstrap over events
  (paired) and seeds (rt.hboot); Holm over the 8 comparisons (2 inputs x 2 models x {outside, test}). Next to it, descriptive: 0.1-0.8
  nm MAPE, MAPE4 and 0.1-0.8 nm interval coverage on the minutes in force (declaration rule, intervals calibrated on the validation
  flares), per reason for failing the rule.
- Text rule: the paper says that training on all flares helps only if, for BOTH operational models, MAPE4 of the flares outside the
  rule falls with adjusted p < 0.05 and the 95 % interval of the change for the isolated test flares does not lie entirely above 0.
  Then the operational numbers are re-stated with these models. Otherwise the paper reports the change and keeps the present models.
- The analysis script (notebooks/analyze_p9.py) is written when the results are back and follows this entry.

## 2026-11-20 — P9 result (patch_v15 run of 2026-11-20 09:42-09:52, results_tables_20261120_0952): training with the flares outside the rule does not help
- Run check: 40 trainings, all on GPU (32 trainer lines "device: cuda", none "cpu"; cuda:0-3), "finished. failed: none", the four P9
  folders PASS in 11_verify_all_results.sh. Training rows 1204 / 1167 (FISM2 input, fold 0 / 1) and 1196 / 1161 (GOES + FCN converter):
  988 + 216 / 179 and 982 + 214 / 179 as designed; validation rows unchanged (122). Folds in the result files equal those of
  src/data/pool.py and no 24-h time cluster is split between the folds (asserted in the script).
- Script: notebooks/analyze_p9.py, written after the run and following the entry of 2026-11-19. One reading had to be fixed while writing:
  "isolated test flares from both runs (mean of the two runs per seed)" = the mean of the two runs' event scores (MAPE4, errors, inside-the-
  interval flags), not a forecast averaged over the two runs (that would be a 2-model ensemble against a single baseline model).
  Test set of the family = the rebuilt test flares (95 with a GOES input, 96 with the FISM2 input), as in every operational table.
- Family of 8 (MAPE4 over all 180 min, P9 - P8, hierarchical bootstrap over events and seeds, n = 10000, Holm):
    GOES + FCN converter (operational)   outside the rule, out-of-fold (393)      isolated test flares (95)
      GRU-z   20.16 -> 20.22  +0.06 [-0.51, +0.68] p_adj 1.00     14.66 -> 15.51  +0.84 [+0.13, +1.62] p 0.022, p_adj 0.15
      FCN-z   20.56 -> 20.17  -0.39 [-0.87, +0.08] p_adj 0.61     15.42 -> 15.61  +0.19 [-0.26, +0.70] p_adj 1.00
    FISM2 input (benchmark)              outside (395)                            isolated test flares (96)
      GRU-z   18.29 -> 18.07  -0.22 [-0.73, +0.28] p_adj 1.00     13.47 -> 13.84  +0.37 [-0.23, +1.00] p_adj 1.00
      FCN-z   19.03 -> 18.95  -0.08 [-0.54, +0.37] p_adj 1.00     13.47 -> 14.12  +0.65 [+0.26, +1.05] p_adj 0.0096
  No comparison shows a fall for the flares outside the rule; the only adjusted-significant change is a rise for the isolated test
  flares (FCN-z, FISM2 input). 5-seed ensembles (not in the family): operational GRU-z test 14.31 -> 15.14 (+0.83 [+0.36, +1.31]),
  outside 19.88 -> 19.94; operational FCN-z test 15.24 -> 15.44, outside 20.39 -> 19.99 (-0.40 [-0.76, -0.03], unadjusted).
  zenodo96 (FISM2 input, not in the family): GRU-z +0.25 [-0.39, +0.86], FCN-z +0.63 [+0.23, +1.05].
- Text rule (both operational models: outside falls with adjusted p < 0.05 AND the test interval not entirely above 0): NOT met
  (GRU-z: outside +0.06, p_adj 1.00, test interval entirely above 0; FCN-z: outside -0.39, p_adj 0.61). The paper reports the change
  and keeps the present models (trained on the isolated flares). No operational number is re-stated.
- Descriptive, minutes in force (declaration rule), flares outside the rule, operational GRU-z / FCN-z: MAPE4 19.96 -> 20.26 / 20.29 ->
  19.97; 0.1-0.8 nm MAPE 44.4 -> 45.2 / 44.7 -> 43.9; 0.1-0.8 nm coverage 0.829 -> 0.842 / 0.827 -> 0.861 (FCN-z with intervals 14 % wider:
  0.683 -> 0.776 dex); all flares (population-weighted) coverage 0.936 -> 0.931 / 0.918 -> 0.921. By reason the change of MAPE4 (all 180
  min) of the operational models is within +-0.8 points (FISM2 input: 'both' -1.4 / -0.9, 41 flares); X class (25 flares) +1.9 / +0.8.
- Check that the added flares were used: on the flares a run was trained on (in-fold) MAPE4 is 18.2 / 18.4 against 20.2 / 20.6 without
  them, so the fit to those flares changed; the gain does not carry over to the other fold.
- Added after the result (post hoc, descriptive; paper1_p9_bias_posthoc.csv): signed error of log10 0.1-0.8 nm by 30-min block. For the
  flares outside the rule the under-prediction at 150-180 min shrinks from -0.219 to -0.176 dex (GRU-z) and -0.237 to -0.206 (FCN-z):
  the models raise the late part of every forecast a little. For the isolated test flares the same shift moves the bias from -0.032 to
  +0.002 (GRU-z) and adds error (0.1-0.8 nm MAPE 32.0 -> 34.4, +2.41 [+0.25, +4.90]). Reading: the error of these flares comes from a
  later flare that the 60 input minutes do not announce; more examples of it cannot be turned into a better forecast, only into a
  higher average tail. This supports handling them by the validity range and the re-issue, not by training.
- Independent re-computation (a separate script without project code): the 8 differences agree to 4 decimals, the
  intervals to the bootstrap noise (<= 0.03); MAPE in force of the operational models agrees to 3 decimals. The P8 rows of
  paper1_p9_inforce.csv equal the declaration rows of paper1_onset.csv exactly.

## 2026-11-20 — public-source check of the benchmark input and of real-time data
- The paper (Section 2.2 and the Availability Statement) says the input is real-time GOES 0.1-0.8 nm irradiance obtained from NOAA NCEI
  and the target is FISM2 from LISIRD. The authors' own package on Zenodo reads the input from files "input/w0/<split>/input_*_w0.txt"
  under their "fism2/data" tree, next to the outputs of the four bands (loader docstring: "log10(w0 X-ray flux), 60 timesteps"), and its
  README labels the array "GOES X-ray input sequences". In the arrays the last input value equals the first 0.1-0.8 nm target value for
  all 96 test flares, and the input follows the public FISM2 0.1-0.8 nm series (rms 0.032 dex) and not the NCEI science-quality XRS-B
  (rms 0.185 dex; ratio 0.71 an hour before the peak, 1.33 at the peak). What can be stated: the benchmark input is the FISM2 0.1-0.8 nm
  band (which FISM2 derives from GOES XRS-B), not the GOES XRS-B value a forecaster receives. Nothing more is claimed.
- FISM2 in real time: no public document found gives a latency or a real-time release of the FISM2 flare product (LASP space weather
  page, FISM2 ReadMe, 2021 Space Weather Workshop abstract, Woods et al. 2026 Solar Physics; the FISM2 paper pages could not be read).
  LASP only states that the flare product uses "GOES/XRS-B and its derivative" as proxies. The paper itself says FISM2 "does not
  currently provide operational real-time forecasts". Not established either way -> measured directly by the next patch (P12: the
  latest time stamp the public LaTiS service returns).
- GOES XRS in real time: NOAA's GOES-R XRS L2 ReadMe (6 Oct 2026): science-quality data "latency of three days or less"; operational
  data latency one day in the archive, with "significant issues that are not retroactively corrected" (static dark count against a
  time-dependent one in the science data). Users Guide: GOES 8-15 operational data carry the SWPC factors 0.7 (XRS-B) and 0.85 (XRS-A);
  GOES-R data do not. Woods et al. (2026) quote about 60 s latency for the operational XRS products used in real time.
- Related work found in the search: Yi et al. 2026 (arXiv 2608.20062) nowcast the PEAK flux of an ongoing flare from the preceding 60
  minutes of 1-min GOES 0.1-0.8 nm flux (attention LSTM), train on science-quality data and run on the SWPC real-time JSON feed. Same
  input convention and the same train-on-science / run-on-real-time practice as here; it stops at the peak, where this work starts.
  It gives no numbers for the difference between real-time and science-quality data.

## 2026-11-20 — patch_v16: design fixed before the run (P11 operational GOES inputs; P12 latest FISM2 time stamp)
- Why: every GOES input so far is NOAA's science-quality archive, released up to three days late. A forecaster receives the
  operational stream. NOAA archives it (P10 probe of 2026-11-20, 17 of 17 directories listed): GOES 16-18 daily files
  .../goesNN/l2/data/xrsf-l2-avg1m/YYYY/MM/dn_xrsf-l2-avg1m_gNN_dYYYYMMDD_v*.nc next to the science files; GOES 8-15 monthly files
  .../goes-space-environment-monitor/access/avg/YYYY/MM/goesNN/{netcdf,csv}/gNN_xrs_1m_YYYYMMDD_YYYYMMDD.{nc,csv}.
  The content of the GOES 8-15 files could not be inspected before the run, so the reader accepts the
  variable names of both generations, fetches the netCDF and the text file of each month (falling back to the text file if the netCDF
  file cannot be read), and the build step prints what it found; a wrong reading shows in the ratio table of (a).
- Operational input of a flare: the operational 1-min XRS-A/B of the satellite of its science-quality input, on the same minutes
  (dataset peak - 60 .. + 179). GOES 8-15: divided by the SWPC factors 0.7 (XRS-B) and 0.85 (XRS-A), which NOAA documents and a
  real-time user knows. Fill values and minutes with a non-zero quality flag are missing; interior gaps of <= 5 min are filled as in
  the dataset. Time stamps are used as archived ("as is"). Because the two archives may stamp a minute differently, the build step also
  estimates ONE integer lag per satellite (-2 .. +2 min, smallest median |dlog10| against the science input minutes; ties -> the lag
  nearest 0) from the isolated training/validation flares only (no test flare, no flare outside the rule) and writes "aligned" windows;
  if every lag is 0 they are not used.
- Run P11: GRU-z and FCN-z, GOES + FCN converter, seeds 0-4, trained exactly as P8 (science-quality inputs; converter fitted on the
  isolated training flares). The SAME trained model predicts the rebuilt test flares and the flares outside the rule with the
  science-quality input (oofL, oofN, as in P8) and with the operational input (twin rows -> oofO). 10 GPU trainings.
- Scores (script written when the results are back, following this entry):
  (a) inputs: share of flares with a complete operational window; log10 ratio operational / science over the 60 input minutes (median,
      rms, share of minutes with |d| > 0.05 dex) by satellite series (8-12, 13-15, 16-18); lag per satellite.
  (b) forecasts, family of 4: MAPE4 (all 180 min) operational - science input, same model and seed, flares with both inputs,
      2 models x {isolated test flares, flares outside the rule}; hierarchical bootstrap (events paired, seeds), Holm.
  (c) descriptive: 0.1-0.8 nm MAPE; minutes in force (declaration times of the science stream, as before): MAPE4 and coverage of the
      intervals (calibrated on the validation flares with science inputs); the detector on the operational stream (own peak declared,
      latency, earlier declarations) next to the science stream.
- Text rule: the paper says the forecasts hold with operational data if, for BOTH models, the upper end of the 95 % interval of the
  MAPE4 change on the isolated test flares, with as-is time stamps, is below +0.5 points (a quarter of the cost of replacing the FISM2
  input by GOES, +1.96). If only the aligned windows meet this, the paper says so and gives the lag as a time-stamp convention to apply.
  Otherwise the change is reported as the cost of real-time data, and a converter fitted on operational data is the follow-up.
- P12: latest time stamp of the public FISM2 flare product. The LaTiS service is asked for 0.1-0.2 nm in one 10-min window (12:00 UT)
  per day, going back from the run day by 0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365 days, then every day between
  the last empty and the first filled one. Result: the most recent day with data and its distance to the run day. No rule: a measured
  fact for the text (it answers whether the benchmark input could be had in real time).

## 2026-11-20 — P11 / P12 results (patch_v16 run, results_tables_20261120_1122) and decision
- (a) Inputs: a complete operational window exists for 90 of the 95 rebuilt test flares with a GOES window and 381 of the 393 flares
  outside the rule (15 of the 17 missing are GOES-13 months without operational files in the archive). XRS-B agrees with the
  science-quality value to rms 0.015-0.017 dex (median difference 0.000); the estimated lag is 0 min for all 9 satellites, so the
  aligned windows are not used. XRS-A differs more (rms about 0.15 dex; GOES 8-12 a constant 0.146 dex lower).
- (b) MAPE4 change, operational - science input, same model and seed, isolated test flares (90): GRU-z +0.17 [-0.13, +0.45],
  FCN-z +0.15 [-0.10, +0.38]; flares outside the rule: -0.05 for both. With independently resampled seeds the intervals widen to
  [-0.58, +0.97] and [-0.21, +0.49]; the pre-specified scoring pairs the seeds because both predictions of a seed come from the same
  weights.
- (c) 0.1-0.8 nm MAPE of the test flares +0.8 / +0.9 points (32.7 -> 33.5, 33.7 -> 34.4); coverage in force 0.959 -> 0.957 and
  0.936 -> 0.932; the detector on the operational stream declares the own peak for 96.3 % of 1,540 flares (science stream 96.6 %),
  with the same declaration minute for 95.0 %.
- Decision (text rule above): both upper ends are below +0.5 with as-is time stamps -> the paper states that the forecasts hold with
  the archived operational data, and that real-time gaps and late samples are not covered by this test.
- P12: the latest day with data in the public FISM2 flare product was 8.1 days before the run day (one measurement, 2026-11-20).

## 2026-11-25 — analyses requested by the internal review (no training); results
Designs as written in the review before computing; scripts `notebooks/si_*.py`, outputs in the Supporting Information.
- Assignment: the GOES-input pools drop the 6 flares without a GOES window before the 8:1 assignment -> 982 / 122; 867 training and
  7 validation flares are common with the FISM2-input pool (988 / 122). Disclosed in the paper; control experiment 1 below.
- Converter gain on the validation flares (same models; validation flares also stop the training, so not a clean test):
  FCN converter - GOES direct -0.36 [-0.90, +0.17] (test -0.76), GRU-z -0.00; test + validation (217) -0.54 [-0.94, -0.13].
  The paper no longer states "recovers about 40 %"; the test-flare gain is reported as found on the test flares.
- Level / shape: after removing one log10 offset per flare and band, GOES direct - FISM2 input drops from +2.05 to +0.14 (GRU-z) and
  +2.17 to +0.09 (FCN-z) points; the cost of the real-time input is a level, not a decay-shape, cost.
- Closed-form baselines on the 95 test flares, GOES input: training-mean profile 29.6, log regression on the last value and the
  window minimum 19.3, ridge regression of the 720 log targets on the 60 log inputs 16.6 (FISM2 input 29.8, 18.7, 15.0).
- Coverage at nominal 0.5 / 0.8 / 0.9 / 0.95 (0.1-0.8 nm, operational GRU-z): 0.51 / 0.85 / 0.95 / 0.97; FCN-z 0.49 / 0.84 / 0.93 / 0.96.
- Detector threshold on the list scale (A5; rule fixed in the review: keep the text if the own-peak share changes by < 2 points and
  coverage in force by < 0.02): own peak 96.7 -> 96.5 %, median latency 2 min unchanged, declarations not matching a listed M/X flare
  0.17 -> 0.07 per 4-h window, coverage in force changes <= 0.003 -> text kept, scale explained.
- Absorption reference scale (A2): with GOES XRS-B instead of the FISM2 band as the D-RAP input, the reference absorption itself
  changes by 0.09 dB on average (27 daytime test flares) and the forecast error becomes 0.16-0.17 dB -> Section 4.6 reads as error
  propagation, not as accuracy of an absorption forecast.
- Not done here: the detector on continuous GOES series (A1, needs full-day files); a time-extrapolation split (control 2) and an
  XRS-B target (control 3) are kept for the revision or follow-up work.

## 2026-11-25 — control experiment 1: design fixed before the run (FISM2 input with the GOES assignment)
- Why: the published comparison "GOES input - FISM2 input" (+1.96 [1.13, 2.82]) and "FCN converter - FISM2 input" (+1.20) mix the
  input with a slightly different train/validation assignment (6 flares removed before the round robin in the GOES pools).
- Run: `run.py --exp p2_zscore --protocol full --seeds 0,1,2,3,4 --split-like-goes --tag c1` (FISM2 input; `FISM_SPLIT_LIKE_GOES=1`
  assigns train/val among the flares with a GOES window only, which reproduces the GOES-pool assignment exactly; checked: 982 / 122,
  identical sets). Five architectures x five seeds, 25 GPU trainings (`deploy/server/12_run_control1.sh`).
- Family (Holm, 2): GOES direct - FISM2(c1), FCN converter - FISM2(c1); MAPE4 on the 95 rebuilt test flares with a GOES window;
  hierarchical bootstrap, flares paired, seeds of every model resampled independently, 10,000 replicates
  (`notebooks/control1_analysis.py`). Also: the benchmark-path interval re-calibrated on the now shared 122 validation flares.
- Decision rule: if the new GOES direct - FISM2 difference is within +-0.3 points of +1.96, the text keeps its sentence and only
  the numbers change; otherwise both values are reported and "about 2 points" is revised.

## 2026-11-25 — control experiment 1: result
- Run on the GPU server (4 GPUs, 25 trainings in 311 s; `deploy/server/12_run_control1.sh`). Assignment of the control run checked
  against the GOES pools by flare key: 982 / 122, identical sets.
- Pre-specified family (95 rebuilt test flares, Holm over 2): GOES direct - FISM2(c1) **+1.91 [+0.93, +2.85]**, Holm p = 0.0004
  (per architecture +1.80 to +2.15); FCN converter - FISM2(c1) +1.16 [+0.23, +2.02], Holm p = 0.014. Same assignment vs own
  assignment of the FISM2 input: +0.04 [-0.36, +0.46]. Output `results/full_server/control1.csv`.
- **Decision (rule above): within +-0.3 of +1.96 -> the sentence ("about 2 points") is kept, the numbers are updated** (paper:
  introduction, Section 4.2, Tables 1-2, limitations; Supporting Information Text S10, Table S9).
- Descriptive, not pre-specified (main-worker recheck from the server's per-event metrics and the forecasts): validation flares,
  now shared and paired, GOES direct - FISM2(c1) +2.63 [+1.52, +3.89], test + validation (217) +2.31 [+1.54, +3.14] (validation
  flares also stop the training); level / shape, ensembles: GRU-z +1.84 -> +0.09, FCN-z +1.77 -> +0.04 after removing one offset
  per flare and band; benchmark-path 90 % interval re-calibrated on the shared validation flares covers 0.911 (GRU-z) and 0.899
  (FCN-z) of the 0.1-0.8 nm test minutes (own assignment 0.919, 0.909).
- Data record: the forecasts are added as `predictions_fism2_input_goes_assignment.npz` (`scripts/build_zenodo_dataset.py`).
- Order of record and run: the design entry above was shipped to the server in `patch_v17.zip` (file `docs/decisions_append_20261125.md`,
  SHA-256 of the archive d29b1abeb28c3b80b73ae6f0d349f2aa7431d4827bf9307e97b808c182f43c43) before the run started. "Pre-specified"
  here means written into this log before the run, not registered with an external registry.
- Benchmark-path interval re-calibrated on the shared validation flares: planned in the design ("Also: ...") without a decision
  criterion, so it is a planned computation, not a test.

## 2026-11-25 — analysis A1, the detector on continuous series: design fixed before the run
- Why: the detector was evaluated only in the 4-h windows around listed M/X flares (paper limitation 2; internal review).
  Operational use needs the declaration rate on continuous data. No training.
- Data: two full years of the primary satellite, 2014 (GOES-15, maximum of cycle 24) and 2022 (GOES-16, rise of cycle 25).
  Primary product: the science-quality 1-min XRS-B files used everywhere in the paper (quality flags applied); secondary,
  descriptive: the archived operational 1-min product (GOES 8-15 SWPC scaling removed). Days without a file are gaps.
- Rule: the paper's armed detector unchanged (k = 2, r = 2, threshold 1e-5 W m-2, raw series). Sensitivity: catalogue-scale
  threshold 1.43e-5 for GOES-15.
- Matching: a declaration is matched if its maximum lies within +-5 min of the peak of a listed M/X flare (flare_catalogue.csv,
  all M/X flares, isolated or not). Unmatched declarations are classified in this order: gap (>= 10 missing minutes in the
  60 min before the maximum), spike (<= 2 minutes within +-10 min reach half the maximum), below_list_M1 (maximum below M1
  on the catalogue scale), near_listed (maximum within -60..+180 min of a listed peak), other.
- Report: data days, declarations per data day, share of listed M/X flares with data whose peak is declared, median latency,
  unmatched declarations per data day by class (`scripts/continuous_detector.py`, `deploy/server/13_run_detector_continuous.sh`).
- Decision rule (proposed by the internal review): primary product, threshold 1e-5: if unmatched declarations are <= 1 per data day in
  both years, the limitation sentence becomes a result sentence and an SI table is added; otherwise the numbers are reported
  as they are and the detector statements of the paper are weakened.

## 2026-11-25 — analysis A1, the detector on continuous series: result
- Output `results/full_server/continuous_detector_{summary,declarations,listed}.csv` (server run, no training). The summary
  re-derived from the two detail tables matches the server summary in every row (main-worker recheck).
- Primary (science-quality, threshold 1e-5): 2014 / GOES-15, 351 data days: 1.08 declarations per day, peaks of 215 of 219 listed
  M/X flares declared (98.2 %), median latency 2 min; unmatched 0.456 per day, of which 0.404 below M1 on the catalogue scale
  (list-scale C7.0-C10.0, median C8.1), gap + spike 0.017, near a listed flare 0.031, other 0.003. 2022 / GOES-16, 358 data days:
  0.54 per day, 182 of 192 (94.8 %), median latency 2 min; unmatched 0.031 per day (near a listed flare 0.025, other 0.006).
- **Decision (rule above): <= 1 unmatched declaration per data day in both years -> the limitation sentence becomes a result
  sentence (paper Section 4.3, limitation 2) and an SI table is added.**
- Sensitivity and secondary product: catalogue-scale threshold (2014) 0.037 unmatched per day with the same 215/219 declared;
  operational archive 1.07 / 0.54 per day, 0.457 / 0.031 unmatched per day, maximum minute identical for 364/378 (2014) and
  192/195 (2022) declarations. Of the 14 listed flares whose peak was not declared, 12 follow another declared flare by 17-130 min
  (no factor-2 rise above the preceding minimum).
- Record (added after the internal review of the result): `patch_v18.zip` SHA-256 d31e3c4efec71c6c20d4bcec3ba06c8611d3263a9d9da223064b717013fb1eb8;
  its design file equals the design entry above. The server log of the run (`run_logs/13_detector_continuous.log`) is not part of
  this repository. The `cls` label `spike` of `continuous_detector_declarations.csv` means "short maximum" (at most 2 minutes within
  +-10 min reach half the maximum), a shape class, not a data-error verdict; the paper calls it "short maximum". One of them
  (2014-01-28 15:26) is an M3.5 flare of the NOAA reports that is absent from the HEK list used for 2014 (the catalogue takes one
  source per year: HEK for 2012, 2014, 2015 and from 2017-04, NOAA reports otherwise). Counting it as listed: 216 of 220 peaks
  declared, 0.453 unmatched declarations per data day, gap + short maximum 0.014 per day; the decision does not change.
- The comment in `deploy/server/12_run_control1.sh` still says "pre-registered family"; the file is kept as it was run (it is part
  of `patch_v17.zip`, whose SHA-256 is given above). "Pre-specified" is the meaning.
