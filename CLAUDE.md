# uplift-targeting — who should get the campaign? Causal effect, not response probability

This file is the build spec. Read it fully before writing code. Work milestone by
milestone; after each milestone stop, show how to run it, and wait for Davut's OK.
Do not spawn review or refuter sub-agents; verify with the scripts and tests and
report directly. Section 0 says where we are.

## 0. Status (updated 2026-10-27 — Milestone 1 done)

**Starter and M1 are done and committed; start at Milestone 2 below.** Repo: `github.com/DavutKursun/uplift-targeting`
(public). `pytest -q` → 18 passed without the raw data. Every M1 number and table lives in `docs/DESIGN.md` §1–2;
`scripts/m1_diagnostics.py facts | balance | synthetic` reproduces them.

Implemented in the starter: `uplift/config.py`, `schema.py`, `data.py` (chunked loader → float32 parquet, stratified
subsample and splits, randomization check, ATE with CI), `synthetic.py` (a campaign with known τ(x): persuadables, sure
things, lost causes, sleeping dogs), `learners.py` (response model, S, T, X, transformed outcome, R, DR; one interface;
LightGBM; cross-fitting), `metrics.py` (Qini, AUUC, uplift by decile, policy value, stratified bootstrap, PEHE),
`policy.py` (budget → threshold, incremental outcomes per 1,000, profit, segments); `scripts/download.py`,
`build_splits.py`, `ate_report.py`, `synthetic_check.py`, `train.py`, `evaluate.py` (refuses `--split test` without
`--final`), `policy_eval.py`, `placebo.py`; `tests/`. NOT yet: `app.py`, Dockerfile/Space, model card, GIF.

Done in M1 (2026-10-27):
- **Data.** The Criteo AI Lab link is dead; the same `criteo-research-uplift-v2.1.csv.gz` comes from Criteo's own
  Hugging Face org (§3.1). Chunked CSV → parquet cache, 4M-row stratified subsample, splits 2,399,998 / 799,998 /
  800,004 (treated share 0.850 and visit rate 4.70% in each); `build_splits.py` took 16 s with a 1.04 GB peak RSS on
  the M2 / 8 GB. `data/splits/summary.json` committed; the test parquet has not been opened. The raw file is
  block-ordered by treatment (first 30% of rows all treated): never take "the first N rows".
- **The experiment (train).** visit 3.82% control → 4.85% treated, ATE +1.03 pp [0.97, 1.10], +27%; conversion
  0.196% → 0.302%, +0.106 pp [0.089, 0.122], +54%. The spec's §3.2 arm rates and relative uplifts were wrong (69% /
  37%) and were corrected to the measured values.
- **Randomization is imperfect.** max |SMD| = 0.049 (f3), 7 of 12 features above 0.02, all below 0.1; treatment is
  weakly predictable from the features (cross-fitted AUC 0.508, propensity 0.83–0.90); every covariate adjustment
  (exact strata, OLS, IPW, AIPW) puts the visit ATE near 0.8 pp, below the raw CI. Decision: the raw arm difference
  stays primary (as in the release paper); the adjusted number is reported next to it; the estimated-propensity
  ablation in M2 is a real robustness check; an IPW-weighted Qini is a candidate for M3; README and model card list
  it as a limitation of the data.
- **Synthetic ground truth (n = 200k).** All six uplift learners reach Spearman 0.66–0.88 with the true τ; the
  response model gets −0.08 and, at a 20% budget, fewer incremental visits than random targeting. With the starter's
  LightGBM values only the S-learner passed; the committed defaults in `learners.py` (lr 0.03, 15 / 7 leaves,
  min_data_in_leaf 2,000 / 5,000, 150 rounds) were chosen on the synthetic truth at 200k and 1M rows and are the M2
  starting point.

**Milestone 2 (do next):** all seven learners on `visit` (`make train`), dev Qini / AUUC with bootstrap CIs
(`make eval-dev`), uplift-by-decile tables, hyperparameters tuned once on dev Qini starting from the committed
defaults, the categorical ablation (§3.4), the estimated-propensity ablation (now a robustness check, §2), the
response-model contrast at every budget. Report the dev table with CIs and stop. Never open the test split.

macOS notes: Python 3.11 venv at `.venv`; 8 GB RAM. The raw CSV must be read in chunks (the loader does); never load it whole.

## 1. Goal

A marketing team can send a treatment (discount, email, ad) to only part of its customers. A **response model**
answers "who will convert?" and sends the offer to people who would have converted anyway. An **uplift model**
answers "whose behaviour does the offer *change*?" and sends it to the persuadables. This project builds the
second, measures it honestly on randomized-experiment data where the truth can be checked, and ships a
targeting service: for a customer, the estimated uplift, the segment, and treat / skip under a budget.

The README answers: *How much more do we get per 1,000 contacts by targeting on uplift instead of on response
probability?* *At a 20% budget, which share of the total incremental effect do we capture?* *Who should never be
contacted (sleeping dogs)?* *How sure are we (confidence intervals)?*

Portfolio purpose: causal thinking (prediction ≠ intervention), meta-learners implemented by hand, evaluation
without counterfactuals (Qini, uplift by decile, policy value), statistical honesty (bootstrap CIs, placebo
tests, one look at the test set), and a decision product on top. Rare in portfolios; interviewers notice.

## 2. Non-negotiables

- **Randomized data only for evaluation.** Criteo's treatment was randomized, so observed outcomes in the
  holdout give unbiased uplift for any targeting rule. No observational-data claims.
- **One look at the test split.** Hyperparameters, learner choice and budget curves are chosen on dev;
  `scripts/evaluate.py --split test` needs `--final` and runs once; the date goes into this file.
- **Ground truth where it exists.** Every learner is first validated on synthetic data with known τ(x) (PEHE,
  correlation with the truth); a learner that cannot recover a known effect is not trusted on real data.
- **The response model is always in the table.** The project's point is the gap between targeting on P(Y|T=1)
  and targeting on τ; it is reported at every budget.
- **Uncertainty is a number.** Qini coefficients and policy values carry stratified bootstrap 95% CIs
  (≥ 200 resamples); a placebo run (treatment labels shuffled) must give Qini ≈ 0 within its CI.
- **Two outcomes.** `visit` (4.7%, strong effect) is primary; `conversion` (0.29%, weak and noisy) is reported
  with its wide CIs as the honest second line, not hidden.
- Propensity is known (randomized, ≈ 0.85) and used as such in the X-, R- and DR-learners; estimating it is an
  ablation, not the default.
- No secrets exist in this project; Kaggle credentials stay in `~/.kaggle/`.

## 3. Data

### 3.1 Source

Criteo Uplift Prediction Dataset v2.1 (research release by Criteo AI Lab; ≈ 13.98M rows; 12 dense
features `f0`–`f11`, 4 continuous and 8 high-cardinality categorical stored as floats; `treatment` 0/1 with
85% treated; `visit`, `conversion`, `exposure` 0/1). Davut downloads it once: the Kaggle mirror
`arashnic/uplift-modeling` (file `criteo-uplift-v2.1.csv`, ≈ 3 GB) or the Criteo AI Lab page
(`criteo-research-uplift-v2.1.csv.gz`, ≈ 300 MB), and places the file in `data/raw/`. `scripts/download.py` only
checks the file and prints the routes. The loader accepts .csv and .csv.gz and reads in chunks.
2026-10-27: the Criteo AI Lab link (go.criteo.net) returns 404; the same file is served by Criteo's own Hugging Face
org, `https://huggingface.co/datasets/criteo/criteo-uplift` (311 MB gzip, CC BY-NC-SA 4.0, no login) — used for M1.

### 3.2 Expected facts (from the release paper) vs measured (M1, 2026-10-27, all 13,979,592 rows)

| Quantity | Expected | Measured |
|---|---|---|
| Rows | ≈ 13.98M | 13,979,592 ✓ |
| Treated share | ≈ 0.85 | 0.850 ✓ |
| Visit rate (all / control / treated) | ≈ 4.7% / ≈ 3.0% / ≈ 5.0%; relative uplift ≈ 69% | 4.70% / 3.82% / 4.85%; ATE +1.03 pp [1.01, 1.06]; relative +27.1% [26.2, 28.0] — the spec's arm rates were wrong, the all-rows rate is right |
| Conversion rate (all) | ≈ 0.29%; relative uplift ≈ 37% | 0.292%; control 0.194% / treated 0.309%; ATE +0.115 pp [0.109, 0.122]; relative +59% [54, 65] |
| Randomization | max \|SMD\| across features < 0.02 | max \|SMD\| = 0.049 (f3); 7 of 12 features above 0.02, all below 0.1. Treatment is predictable from features (see DESIGN.md §1): the arms are not perfectly balanced, and a feature-adjusted ATE on visit is ≈ 0.7 pp instead of 1.0 pp |

The raw treated-minus-control estimate stays the primary number (as in the release paper); the adjusted one is reported
next to it as the honest caveat, and every M2–M4 evaluation keeps using observed outcomes by arm.

### 3.3 Splits and subsample

No timestamps exist, so splits are random and stratified by (treatment, visit): train 60% / dev 20% / test 20%.
Default working set: a stratified subsample of 4M rows (train 2.4M, dev 0.8M, test 0.8M) written to
`data/splits/*.parquet` (gitignored); `--full` uses everything for the final M4 run if the Mac copes (report RAM).
`data/splits/summary.json` is committed.

### 3.4 Features

All twelve as floats. The 8 categorical-as-float columns are kept numeric for LightGBM (it splits on values);
an ablation in M2 treats the ones with ≤ 1,000 distinct values as categorical. Feature names are anonymous;
explanations in the product are "feature f3 is in its top decile" — stated as a limitation, with the
interpretable Hillstrom bonus (M6) as the remedy.

## 4. Learners (`uplift/learners.py`, LightGBM base models, same interface)

| Name | Idea | Why it is here |
|---|---|---|
| response | P(Y=1 \| X, T=1), target the highest | The wrong baseline everyone starts with |
| S-learner | one model f(X, T); τ = f(X,1) − f(X,0) | Simplest; regularisation can shrink τ to 0 |
| T-learner | μ1 on treated, μ0 on control; τ = μ1 − μ0 | Two models; noisy when control is small (15%) |
| X-learner | imputed effects on each arm, combined with the propensity | Built for imbalanced arms like this one |
| transformed outcome | regress Z = Y·(T−e)/(e(1−e)) on X | One regression, unbiased for τ, high variance |
| R-learner | cross-fitted residuals of Y and T; weighted regression for τ | Nie & Wager; the modern default |
| DR-learner | cross-fitted doubly-robust pseudo-outcomes; regress on X | Robust if either nuisance is right |

Cross-fitting uses 5 folds; nuisance models and the τ model are LightGBM with modest depth; hyperparameters
(leaves, min_data_in_leaf, learning rate, rounds) are chosen on dev Qini for the primary outcome, once, and
reused for the secondary outcome.

## 5. Evaluation (`uplift/metrics.py`, `scripts/evaluate.py`, `policy_eval.py`, `placebo.py`)

- **Ranking of uplift:** Qini curve and normalised Qini coefficient (area between the model curve and the
  random line, divided by the same area for a perfect ordering of observed outcomes), AUUC; bootstrap CIs.
- **Observed uplift by decile:** customers sorted by predicted τ into 10 bins; in each, mean(visit | treated) −
  mean(visit | control) with counts — the ground truth the RCT allows; predicted vs observed uplift per decile
  is the calibration check; the top decile should carry several times the ATE, the bottom one may be negative.
- **Policy value at budgets 5–50%:** incremental visits per 1,000 contacts and share of the total incremental
  effect captured, for: random, response model, each uplift learner; profit = value per visit × incremental
  visits − contact cost × contacts (parameters in config, stated as assumptions).
- **Sleeping dogs:** share of customers with predicted τ < 0 and the observed uplift in that group.
- **Placebo:** shuffle treatment within the split, retrain the best learner, report its Qini with CI (expect 0).
- **Synthetic ground truth:** PEHE and Spearman correlation with the true τ for every learner at n = 200k.
- **Secondary outcome:** the same tables for `conversion` with CIs; honest text about their width.
DESIGN.md: the learner table on dev and test, Qini plots, decile plots, the budget curve, the placebo result,
five worked customers (one per segment) and what we would do with a sure thing.

## 6. Serving (M5)

- FastAPI: `POST /score` with the 12 features → `{uplift, p_control, p_treated, segment, decision, budget,
  threshold, model, version, latency_ms}`; `POST /score/batch` (≤ 1,000); `GET /policy?budget=0.2` →
  threshold, expected incremental visits per 1,000 contacts, profit under the stated economics; `GET /health`.
- Gradio at /ui: tab "Customer" (pick a holdout customer or edit features; both the response model and the
  uplift model side by side, the segment, treat / skip); tab "Budget" (slider 5–50% → incremental visits,
  profit, threshold, the four-segment breakdown; response vs uplift curves); tab "Evidence" (Qini and decile
  plots from the committed results, the placebo, the CIs); tab "About" (RCT, limits, anonymous features).
- Hugging Face Space, Docker SDK, cpu-basic; models are small LightGBM boosters committed under `models/`.

## 7. Repository layout

```
uplift-targeting/
  app.py                       # FastAPI + Gradio (M5)
  uplift/ config.py, schema.py, data.py, synthetic.py, learners.py, metrics.py, policy.py
  scripts/ download.py, build_splits.py, ate_report.py, train.py, evaluate.py, policy_eval.py, placebo.py,
           bench_latency.py (M5), demo_gif.py (M5)
  data/raw/ (gitignored)  data/splits/ (parquet gitignored, summary.json committed)
  models/, eval/results/, tests/, docs/DESIGN.md, docs/model_card.md, assets/demo.gif, README.md,
  requirements*.txt, Makefile, LICENSE, .github/workflows/ci.yml
```

## 8. Milestones

**M1 — data and the experiment.** Download check, chunked load, subsample, splits, randomization check, ATE with
CIs on both outcomes, synthetic ground-truth validation of all learners (PEHE table).
**M2 — learners.** All seven on `visit`; dev Qini/AUUC with CIs; uplift-by-decile tables; hyperparameters chosen
once on dev; the categorical ablation; the response-model contrast.
**M3 — decisions.** Budget curves, profit under stated economics, segments, sleeping-dog analysis, placebo,
calibration of uplift by decile, the `conversion` outcome as the second line.
**M4 — final.** Test once for every table; DESIGN.md; model card; README results and the "response vs uplift"
decision table.
**M5 — serve and ship.** FastAPI + Gradio, latency, Space, GIF, README.
**M6 — optional.** Hillstrom (MineThatData, 64k rows, two email treatments): multi-treatment uplift with
interpretable features and a best-arm policy; uplift with a neural τ model; estimated propensity ablation.

## 9. Acceptance criteria

- Synthetic: every uplift learner reaches Spearman ≥ 0.6 with the true τ at n = 200k; the response model does not.
- Test (`visit`): the best uplift learner's normalised Qini beats the response model's with non-overlapping 95%
  CIs; the placebo Qini CI contains 0.
- Decile table on test: top-decile observed uplift ≥ 2 × ATE; the bottom decile's observed uplift ≤ ATE.
- Policy at a 20% budget: incremental visits per 1,000 contacts for uplift vs response vs random reported with
  CIs; the share of total incremental effect captured stated.
- `/score` p50 < 30 ms on the Space CPU; every response carries the segment and the decision under the budget.
- `pytest` green without the raw data; CI green; README on one screen before setup.

## 10. Out of scope

Observational (non-randomized) causal inference, multi-treatment on Criteo (single arm), long-term effects,
online learning. Listed as next steps.

## 11. Pitfalls to avoid

- Evaluating uplift with accuracy or AUC on the outcome: meaningless; use Qini / decile / policy value.
- Splitting without stratifying on treatment: the 15% control arm must be present in every split.
- Comparing learners at different budgets: fix the budget, then compare.
- Reporting conversion effects without CIs: at 0.29% base rate the CIs are the result.
- Reading the test split for anything before M4, including "just to check the size".
