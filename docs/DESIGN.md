# Design notes

Filled milestone by milestone; each section says what was measured, on which split, and what it changes.

## 1. The experiment (M1, 2026-10-27)

**Source and loading.** `criteo-research-uplift-v2.1.csv.gz` (311 MB) from Criteo's own Hugging Face organisation
(`criteo/criteo-uplift`; the link on the Criteo AI Lab page returned 404 that day). `scripts/build_splits.py` streams the
CSV in 1M-row chunks into a float32 / int8 parquet cache (125 MB, zstd), then subsamples and splits. On an Apple M2
(8 cores, 8 GB) the whole step took 16 s wall with a peak resident set of 1.04 GB (peak footprint 2.65 GB); the
raw text is never held whole. Everything below was produced by `scripts/ate_report.py` (train split) and
`scripts/m1_diagnostics.py` (`facts`, `balance`), whose outputs are in `eval/results/`.

**The raw file is block-ordered by treatment**, not shuffled: by tenths of the file the treated share runs 1.00, 1.00,
1.00, 0.99, 0.13, 0.87, 0.91, 0.80, 0.98, 0.83. Any "first N rows" shortcut would be a treated-only sample; the
stratified subsample and splits below ignore file order.

### 1.1 Population facts (all 13,979,592 rows) against the spec

These are descriptive statistics of the whole file, the same ones the release paper publishes; they include the test
rows but no modelling decision depends on them. Model work uses only train and dev.

| Quantity | Spec expected | Measured |
|---|---|---|
| Rows | ≈ 13.98M | 13,979,592 |
| Treated share | ≈ 0.85 | 0.850 (11,882,655 treated / 2,096,937 control) |
| Visit rate: all / control / treated | 4.7% / 3.0% / 5.0% | 4.70% / 3.82% / 4.85% |
| Visit ATE (absolute, 95% CI) | — | +1.03 pp [1.01, 1.06] |
| Visit relative uplift (95% CI, delta method on log ratio) | ≈ 69% | +27.1% [26.2, 28.0] |
| Conversion rate: all / control / treated | 0.29% / — / — | 0.292% / 0.194% / 0.309% |
| Conversion ATE (absolute, 95% CI) | — | +0.115 pp [0.109, 0.122] |
| Conversion relative uplift (95% CI) | ≈ 37% | +59.5% [54.3, 64.6] |
| Exposure rate: treated / control | — | 3.60% / 0 |
| Max \|SMD\| across the 12 features | < 0.02 | 0.049 (f3); 7 of 12 features above 0.02, none above 0.1 |

Row count, treated share and the all-rows outcome rates match the spec. The arm-level visit rates and both relative
uplifts in the spec do not: the control arm visits at 3.8%, not 3.0%, so the visit effect is +27%, not +69%, and the
conversion effect is +59%, not +37%. The spec table was corrected to the measured values (CLAUDE.md §3.2).

### 1.2 Subsample and splits

A stratified 4,000,000-row subsample (same (treatment, visit) mix as the file; seed 7), split 60 / 20 / 20 stratified
by (treatment, visit). `data/splits/summary.json` is committed; the parquet files are not.

| Split | Rows | Treated share | Visit rate |
|---|---|---|---|
| train | 2,399,998 | 0.850 | 4.699% |
| dev | 799,998 | 0.850 | 4.699% |
| test | 800,004 | 0.850 | 4.699% |

The test parquet has not been opened; its row count comes from the split builder's summary.

### 1.3 The experiment on train (`eval/results/ate_train.json`)

| Outcome | Control rate | Treated rate | ATE (95% CI) | Relative |
|---|---|---|---|---|
| visit | 3.820% (n = 359,999) | 4.854% (n = 2,039,999) | +1.034 pp [0.965, 1.104] | +27.1% |
| conversion | 0.196% | 0.302% | +0.106 pp [0.089, 0.122] | +53.7% |

The conversion line is the honest second outcome: at a 0.2% base rate the CI is a third of the effect on 2.4M rows,
and any per-customer estimate of it will be far wider.

### 1.4 Randomization check (train, `eval/results/randomization_train.csv`)

Standardized mean difference (treated minus control, over the pooled SD) per feature, sorted by |SMD|:

| Feature | SMD | Distinct values | | Feature | SMD | Distinct values |
|---|---|---|---|---|---|---|
| f3 | −0.048 | 422 | | f8 | −0.020 | 2,606 |
| f6 | −0.040 | 1,214 | | f2 | −0.008 | 559,005 |
| f5 | −0.029 | 81 | | f10 | +0.008 | 99,634 |
| f9 | +0.022 | 1,114 | | f4 | +0.006 | 163 |
| f7 | +0.022 | 123,046 | | f11 | −0.004 | 80 |
| f1 | +0.021 | 38 | | f0 | −0.002 | 1,119,705 |

Under the usual |SMD| < 0.1 bar the arms are balanced; under the spec's < 0.02 bar they are not. With 2.4M rows the
sampling SE of an SMD is about 0.002, so 0.048 on f3 is not noise.

**Does the imbalance matter?** Three checks, all on train, `eval/results/randomization_diagnostics_train.json`:

- *How predictable is treatment from the features?* A cross-fitted LightGBM propensity model reaches an out-of-fold
  AUC of 0.508; its 1st–99th percentile range is 0.83–0.90. The departure from a coin flip is small but real.
- *Where does it live?* The modal value of f3 (82% of rows, visit rate 3.0%) has a treated share of 0.848; the rarer
  f3 values, whose visit rates are 7–11%, have treated shares of 0.850–0.855. Groups that visit more were treated
  slightly more often, which inflates the raw treated-minus-control difference.
- *How much?* Visit ATE on train, raw vs adjusted:

| Estimator | ATE on visit (95% CI) |
|---|---|
| Raw difference of means (the RCT estimate) | +1.034 pp [0.965, 1.104] |
| Exact strata of f3 (288 strata, 99.98% of rows) | +0.892 pp [0.823, 0.962] |
| Exact strata of f3 × f5 × f1 (1,648 strata, 99.6% of rows) | +0.812 pp [0.743, 0.880] |
| OLS on treatment + 12 standardized features (HC0) | +0.719 pp [0.660, 0.779] |
| Hájek IPW with the cross-fitted propensity | +0.740 pp |
| AIPW (cross-fitted propensity and per-arm outcome models) | +0.772 pp [0.711, 0.833] |

Every adjustment lands near 0.8 pp, below the raw CI. Two readings are consistent with this: (i) the assignment was
not one per-user coin flip — the block structure of the file, with treated shares from 13% to 100% per block, looks
like several campaigns or periods with different treatment rates pooled without a block id, which makes the pooled
comparison confounded by whatever differs between blocks; (ii) some features were computed after assignment, in which
case adjusting on them removes part of the effect and the raw number is right. The file does not say which.

**Decision for this project.** The raw treated-minus-control estimate stays the primary number, as in the release
paper and every published benchmark on this data, and all M2–M4 evaluation keeps using observed outcomes by arm.
The adjusted ATE is reported next to it as the honest caveat, the estimated-propensity ablation in M2 becomes a real
robustness check rather than a formality, and an IPW-weighted Qini is a candidate for M3. The README and model card
state the imbalance as a limitation of the data, not of the method.

## 2. Ground truth on synthetic data (M1, 2026-10-27)

Real data never shows a customer's counterfactual, so every learner is first scored on a world where the individual
effect τ(x) is known (`uplift/synthetic.py`, `scripts/synthetic_check.py`). A learner that cannot recover a known
effect is not trusted on Criteo, whatever its Qini there.

**The world.** Twelve features (nine standard normal, three categorical-as-float like the real file); the untreated
visit probability is logistic in f1 and f3 (base rate ≈ 5%). The true effect has the four textbook segments:
persuadables where f0 is high (τ up to +5.7 pp), sure things at high f1 where the effect is multiplied towards zero,
lost causes at low f1, and sleeping dogs wherever f2 > 1.2 (τ shifted by −4 pp; 11.5% of customers). Treatment is a
coin flip with the Criteo share (85%). n = 200,000 (seed 11), split 60 / 40 with the same stratified rule as the real
data: learners fit on 120,000 rows, of which only 17,805 are control, and are scored on the other 80,000. True ATE
2.14 pp; the standard deviation of τ across customers is 1.93 pp, so the signal is small against binary outcome noise.

**Result with the committed defaults** (`eval/results/synthetic_check.csv`; Qini and policy value use the observed
outcomes of the held-out 80,000, exactly as on real data; the `truth` row scores customers by their true τ):

| Learner | Spearman with true τ | PEHE | Normalised Qini | Incremental visits / 1,000 contacts at a 20% budget | Fit (s) |
|---|---|---|---|---|---|
| truth | 1.000 | 0.000 | 0.077 | 51.5 | — |
| response model | **−0.081** | 0.122 | −0.006 | **6.3** | 0.9 |
| S-learner | 0.881 | 0.010 | 0.072 | 49.7 | 1.0 |
| T-learner | 0.662 | 0.027 | 0.066 | 43.7 | 0.9 |
| X-learner | 0.736 | 0.014 | 0.058 | 41.2 | 1.5 |
| transformed outcome | 0.748 | 0.016 | 0.072 | 46.3 | 0.8 |
| R-learner | 0.770 | 0.014 | 0.077 | 48.1 | 4.8 |
| DR-learner | 0.759 | 0.015 | 0.075 | 49.1 | 5.5 |

- **All six uplift learners pass the acceptance bar** (Spearman ≥ 0.6 with the truth at n = 200k): 0.66 (T) to 0.88 (S).
  The T-learner is the weakest, as expected with 17.8k control rows; the S-learner is the strongest here because the
  effect is smooth and additive in the features, which is the case it is built for.
- **The response model fails as intended.** Its ranking is slightly *anti*-correlated with the true effect (−0.08) and
  strongly correlated with the base rate (it ranks sure things first; `tests/test_core.py` asserts > 0.6 against p0).
  Its Qini is zero within noise, and at a 20% budget it buys 6.3 incremental visits per 1,000 contacts against 21.4 for
  random targeting (the ATE) and 49.7 for the S-learner: targeting by response probability is worse than no targeting
  on a world with sure things, which is the project's point in one line.
- Ranking quality carries over to decisions: the three best learners by Spearman capture 48–50 of the 51.5 incremental
  visits per 1,000 that the truth itself achieves at 20%.

**Why the defaults changed in M1** (`scripts/m1_diagnostics.py synthetic` → `eval/results/synthetic_settings.csv`).
With the starter's LightGBM values (learning rate 0.05, 31 / 15 leaves, min_data_in_leaf 500 / 1,000, 400 rounds)
only the S-learner cleared the bar at n = 200k; the pseudo-outcome learners (transformed outcome, R, DR) and the
per-arm learners (T, X) fit the noise in their targets, whose per-row variance is thirty times the variance of τ.
Spearman with the truth, same world:

| Settings | n (train rows) | Treated share | S | T | X | TO | R | DR |
|---|---|---|---|---|---|---|---|---|
| starter (loose) | 200k (120k) | 0.85 | 0.75 | 0.43 | 0.57 | 0.46 | 0.46 | 0.45 |
| starter (loose) | 200k (120k) | 0.50 | 0.75 | 0.44 | 0.57 | 0.49 | 0.51 | 0.51 |
| **committed** (lr 0.03, 15 / 7 leaves, min_data_in_leaf 2,000 / 5,000, 150 rounds) | 200k (120k) | 0.85 | 0.88 | 0.66 | 0.74 | 0.75 | 0.77 | 0.76 |
| starter (loose) | 1M (600k) | 0.85 | 0.92 | 0.64 | 0.76 | 0.74 | 0.75 | 0.74 |
| **committed** | 1M (600k) | 0.85 | 0.95 | 0.85 | 0.96 | 0.92 | 0.93 | 0.93 |

Balancing the arms (second row) does not help, so the 85 / 15 split is not the cause; more data helps (fourth row),
and stronger regularisation helps more at both sizes and cuts fit time by two thirds. The committed values are the
M2 starting point; M2 tunes once on dev Qini with real data, where τ is smaller still (ATE 1.0 pp).

**What this does and does not show.** The synthetic effect is a smooth function of three features; it validates the
seven implementations, the cross-fitting, the propensity handling and the default regularisation, and it shows the
response-model failure mode by construction. It says nothing about how strong the real heterogeneity is: that is M2's
question, answered only with observed outcomes by arm.

## 3. Learners on dev (M2)
Qini / AUUC with CIs, decile tables, hyperparameter choice, the categorical ablation, the propensity ablation.

## 4. Decisions (M3)
Budget curves, profit under the stated economics, segments, sleeping dogs, placebo, calibration of uplift by decile,
the `conversion` outcome.

## 5. Test results (M4) — run once on _(date)_

## 6. What I would do differently
