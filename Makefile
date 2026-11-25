# One-line entry points (CPU is enough; ~30 min for `reproduce` on 2 cores)
PY ?= python3

.PHONY: install data test reproduce verify improve final all download-smoke calibrate download-all full

install:
	$(PY) -m pip install -r requirements.txt

# decode the checksum-verified Zenodo test-set transcript and build events.nc + QC table
data:
	$(PY) scripts/assemble_zenodo_event_metrics.py
	$(PY) scripts/decode_zenodo_xy.py
	$(PY) scripts/make_events_nc.py

test:
	$(PY) -m pytest -q tests | tee logs/tests.log

# Phase 1 (CV96 proxy reproduction) + tables/figures
reproduce:
	$(PY) run.py --exp repro_cv96
	$(PY) scripts/report_repro.py

# Phase 2 verification
verify:
	$(PY) -I scripts/validate_metrics_authors.py        # needs external/zenodo/raw/2_evaluation (Zenodo zip)
	$(PY) notebooks/verify_authors_predictions.py
	$(PY) scripts/validate_metrics_powerlaw.py
	$(PY) notebooks/verify_powerlaw.py
	$(PY) notebooks/verify_significance_zenodo.py
	$(PY) notebooks/verify_skill_refs.py
	$(PY) notebooks/verify_physics_leadtime.py
	for e in p2_norm p2_norm2 p2_zscore p2_hybrid p2_window p2_blocked p2_scale; do $(PY) run.py --exp $$e; done
	$(PY) notebooks/verify_physics_leadtime.py p2_norm
	$(PY) notebooks/verify_physics_leadtime.py p2_zscore
	$(PY) notebooks/verify_leakage.py
	$(PY) notebooks/verify_phase2_runs.py

# Phase 3 improvements
improve:
	$(PY) scripts/make_p3_configs.py --base peaknorm --base-fcn zscore
	for e in p3_arch p3_resid p3_extreme p3_type p3_joint p3_features p3_early p3_ensemble p3_final; do $(PY) run.py --exp $$e; done
	$(PY) scripts/tune_optuna.py --config p3_final --spec FCN-z --family fcn --trials 50
	$(PY) notebooks/improve_uncertainty.py GRU-peaknorm p3_ensemble
	$(PY) notebooks/improve_uncertainty.py FCN-z p3_ensemble
	$(PY) notebooks/improve_uncertainty.py GRU-final p3_final
	$(PY) notebooks/improve_subsets.py
	$(PY) scripts/arch_costs.py
	$(PY) scripts/ledger.py

# Phase 4 application + final comparison
final:
	$(PY) notebooks/apply_drap.py repro_cv96:Power-law repro_cv96:Climatology repro_cv96:RandomForest repro_cv96:GRU repro_cv96:FCN p2_norm:GRU-peaknorm p2_zscore:FCN-z p3_ensemble:FCN-z-ens5 p3_final:GRU-final-ens5
	$(PY) scripts/final_compare.py

all: data test reproduce verify improve final

# ---- open data (run on a machine with internet access; see RUN_GUIDE_ko.md) ----
download-smoke:
	$(PY) -m pip install -r scripts/download/requirements-download.txt
	$(PY) scripts/download/run_all.py --stage test --limit 3

calibrate:
	$(PY) scripts/download/run_all.py --stage test

download-all:
	$(PY) scripts/download/run_all.py --stage all

# paper setting (train/val on downloaded events, test = Zenodo 96); WORKERS=8 make full
WORKERS ?= 4
full:
	bash scripts/run_full_suite.sh $(WORKERS)
