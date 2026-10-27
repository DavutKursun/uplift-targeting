.PHONY: setup setup-dev test splits ate synth train eval-dev policy-dev placebo smoke app

PY ?= python

setup:
	$(PY) -m pip install -r requirements.txt
setup-dev:
	$(PY) -m pip install -r requirements-dev.txt
test:
	$(PY) -m pytest -q
splits:           ## raw CSV → parquet cache → 4M-row stratified subsample → train/dev/test
	$(PY) scripts/download.py && $(PY) scripts/build_splits.py
ate:              ## arm sizes, outcome rates, ATE with CIs, randomization check (train)
	$(PY) scripts/ate_report.py
synth:            ## every learner against a synthetic truth (PEHE, Spearman)
	$(PY) scripts/synthetic_check.py
train:            ## all seven learners on visit, saved under models/
	$(PY) scripts/train.py --learners all --outcome visit
eval-dev:
	$(PY) scripts/evaluate.py --split dev
policy-dev:
	$(PY) scripts/policy_eval.py --split dev
placebo:
	$(PY) scripts/placebo.py
smoke:            ## the whole chain on a synthetic export (no real data)
	$(PY) -m pytest -q tests/test_scripts.py
app:
	$(PY) -m uvicorn app:app --port 7860
