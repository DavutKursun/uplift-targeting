.PHONY: setup setup-dev test splits ate synth train eval-dev tune plots-dev policy-dev paired-dev placebo smoke serving app bench api-test

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
tune:             ## at most four (leaves, min_data_in_leaf, rounds) configurations for S, R, DR on dev Qini
	$(PY) scripts/tune.py
plots-dev:        ## Qini curves and the decile plot of the best learner, from the saved models and dev results
	$(PY) scripts/plots.py --split dev
policy-dev:       ## budget curves raw and IPW, profit, break-even, segments, worked customers (dev)
	$(PY) scripts/policy_eval.py --split dev
paired-dev:       ## paired-bootstrap differences s_learner/r_learner minus response (dev)
	$(PY) scripts/paired.py --split dev
placebo:
	$(PY) scripts/placebo.py
smoke:            ## the whole chain on a synthetic export (no real data)
	$(PY) -m pytest -q tests/test_scripts.py
serving:          ## the dev policy grid, 2,000 anonymous dev customers and the version file under models/serving/
	$(PY) scripts/export_serving.py
app:
	$(PY) -m uvicorn app:app --port 7860
bench:            ## /score p50 / p95 in-process and over HTTP, batch-1,000 throughput
	$(PY) scripts/bench_latency.py
api-test:
	$(PY) -m pytest -q tests/test_app.py
