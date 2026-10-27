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

## 3. Learners on dev (M2, 2026-10-28)

All seven learners on `visit`: fit on train (2,399,998 rows, treated share 0.850), scored on dev (799,998 rows,
119,999 control). The dev ATE is 1.034 pp [0.914, 1.154], so **random targeting yields 10.3 incremental visits per
1,000 contacts at every budget**; that is the line every model is measured against. Qini is normalised (0 = random,
1 = the perfect ordering of observed outcomes); every CI is a stratified bootstrap with 200 resamples. One resample
loop now serves every statistic (`metrics.bootstrap_block`: the same resamples and numbers as before, three times
faster); evaluating seven models took 11.5 min with a 0.40 GB peak RSS. The test split was not opened.

### 3.1 The committed defaults (`eval/results/dev-summary-visit-defaults.md`)

Fit times with the M1 defaults on an Apple M2: response 6.0 s, S 6.6 s, T 5.5 s, X 11.1 s, transformed outcome 3.7 s,
R 26.9 s, DR 32.9 s; all seven in 105 s wall with a 0.91 GB peak RSS (1.30 GB peak footprint).

| Learner | Normalised Qini [95% CI] | AUUC | Incremental visits / 1,000 at 10% [CI] | at 20% [CI] | at 30% [CI] | Share of total effect at 10 / 20 / 30% |
|---|---|---|---|---|---|---|
| S-learner | 0.0908 [0.0776, 0.1030] | 0.00380 | 60.5 [51.5, 69.1] | 38.7 [32.7, 44.7] | 29.8 [25.8, 34.0] | 0.58 / 0.75 / 0.86 |
| X-learner | 0.0898 [0.0786, 0.1017] | 0.00376 | 61.5 [53.4, 68.8] | 39.4 [34.3, 44.7] | 28.8 [24.9, 32.5] | 0.59 / 0.76 / 0.83 |
| transformed outcome | 0.0897 [0.0770, 0.1012] | 0.00375 | 63.9 [55.8, 72.1] | 39.9 [34.8, 44.7] | 27.9 [24.0, 31.3] | 0.62 / 0.77 / 0.81 |
| R-learner | 0.0873 [0.0748, 0.0988] | 0.00365 | 65.0 [57.0, 72.4] | 39.0 [33.9, 43.9] | 28.8 [24.9, 32.8] | 0.63 / 0.75 / 0.84 |
| **response model** | 0.0872 [0.0743, 0.1004] | 0.00366 | 53.5 [43.7, 64.5] | 38.4 [32.7, 44.5] | 29.5 [25.6, 33.6] | 0.52 / 0.74 / 0.86 |
| DR-learner | 0.0869 [0.0735, 0.0990] | 0.00363 | 63.9 [55.6, 71.3] | 38.7 [33.4, 43.6] | 27.9 [24.0, 31.6] | 0.62 / 0.75 / 0.81 |
| T-learner | 0.0867 [0.0733, 0.0995] | 0.00363 | 59.8 [51.7, 66.8] | 38.9 [33.8, 43.7] | 29.3 [25.8, 32.8] | 0.58 / 0.75 / 0.85 |
| random | 0 | 0 | 10.3 | 10.3 | 10.3 | 0.10 / 0.20 / 0.30 |

Three things to read off this table. Every model, the response model included, finds 3.7–4 times the random uplift
at a 20% budget and captures three quarters of the total incremental effect with a fifth of the contacts. The seven
Qini values sit inside one another's CIs: the ranking signal on Criteo is real (every CI excludes zero by a wide
margin) but the learners are not separable at 800k dev rows. And the response model is a strong baseline here,
unlike on the synthetic world: the Criteo effect is close to proportional to the base rate (relative uplift ≈ 27%
in every slice we looked at), so ranking by P(visit | treated) ranks by uplift as well — except at the tight end,
where it costs: at a 10% budget the uplift learners find 60–65 incremental visits per 1,000 and the response model
53.5, with overlapping CIs.

### 3.2 Hyperparameters, chosen once on dev Qini (`scripts/tune.py` → `eval/results/tuning_dev_visit.csv`)

At most four configurations of (num_leaves, min_data_in_leaf, rounds) for the S-learner's single classifier and for
the τ regression of the R- and DR-learners; the nuisance models of R and DR keep the defaults and were cross-fitted
once per learner (32 s and 37 s) and reused for every τ configuration, so the comparison isolates the τ model. The
other four learners keep the defaults.

| Learner | Configuration (leaves, min_data_in_leaf, rounds) | Dev Qini | Incremental / 1,000 at 20% |
|---|---|---|---|
| S-learner | (15, 2,000, 150) default | 0.0908 | 38.7 |
| S-learner | **(31, 2,000, 300) kept** | **0.0941** | 41.0 |
| S-learner | (63, 1,000, 300) | 0.0934 | 38.6 |
| S-learner | (31, 500, 500) | 0.0932 | 40.6 |
| R-learner τ | (7, 5,000, 150) default | 0.0873 | 39.0 |
| R-learner τ | (15, 2,000, 300) | 0.0853 | 38.6 |
| R-learner τ | **(7, 20,000, 300) kept** | **0.0938** | 40.0 |
| R-learner τ | (3, 10,000, 500) | 0.0852 | 39.1 |
| DR-learner τ | (7, 5,000, 150) default | 0.0869 | 38.7 |
| DR-learner τ | (15, 2,000, 300) | 0.0853 | 39.5 |
| DR-learner τ | **(7, 20,000, 300) kept** | **0.0927** | 40.4 |
| DR-learner τ | (3, 10,000, 500) | 0.0842 | 38.9 |

The S-learner wants more capacity (31 leaves, 300 rounds: +0.003); the pseudo-outcome learners want *more*
regularisation of τ (min_data_in_leaf 20,000: +0.006 for both), and the less regularised configuration loses — the
same direction as the M1 synthetic finding, now on real data. Each gain is smaller than a single model's CI
half-width (± 0.012), but the comparison is paired on the same dev rows, and the choice was made once. The three
learners were retrained with the kept settings (`train.py --cls-params … --rounds 300` for S, `--reg-params …
--tau-rounds 300` for R and DR) and their bundles overwritten.

### 3.3 Ablations on the R- and DR-learners (`eval/results/ablations_dev_visit.md`)

Both ablations use the tuned τ settings, so "base" here is the installed bundle. The decision rule was fixed before
the numbers were seen: keep the variant whose Qini CI lower bound is higher; ties go to the simpler setting.

| Learner | Variant | Normalised Qini [95% CI] | Incremental / 1,000 at 20% [CI] | Fit (s) |
|---|---|---|---|---|
| R-learner | base: known propensity e = 0.85, numeric features | **0.0938 [0.0819, 0.1060]** | 40.0 [34.4, 45.0] | 43 |
| R-learner | (a) cross-fitted estimated propensity e(x) | 0.0923 [0.0807, 0.1046] | 39.6 [33.8, 44.5] | 72 |
| R-learner | (b) f1, f3, f4, f5, f11 as LightGBM categoricals (≤ 1,000 distinct values) | 0.0887 [0.0767, 0.1017] | 39.6 [34.1, 44.5] | 42 |
| DR-learner | base | **0.0927 [0.0817, 0.1055]** | 40.4 [35.2, 45.4] | 49 |
| DR-learner | (a) estimated propensity | 0.0928 [0.0810, 0.1050] | 40.4 [34.9, 45.6] | 88 |
| DR-learner | (b) categorical columns | 0.0887 [0.0778, 0.1021] | 39.4 [33.7, 44.4] | 42 |

**Decision: base carries to M3 for both learners** (CI lower bounds 0.0819 vs 0.0807 vs 0.0767 for R; 0.0817 vs
0.0810 vs 0.0778 for DR). Two readings:

- (a) is the robustness check promised in §1.4. Estimating e(x) instead of using 0.85 changes nothing measurable:
  the DR point estimate moves by +0.0001, the R one by −0.0015, both far inside the CI. The imbalance found in M1
  shifts the *level* of the effect (the adjusted ATE is 0.8 pp rather than 1.0 pp) but not the *ranking* of customers
  — the propensity varies only between 0.83 and 0.90 and is nearly orthogonal to the uplift signal. The known-e
  learners stay the default; the M1 caveat is about how much uplift we report, not about whom to target.
- (b) costs 0.004–0.005 Qini. With categorical handling the τ model splits four times as often on f4 (260 values)
  and a tenth as often on f3, which looks like LightGBM's categorical splits overfitting the coded columns at this
  signal level. Features stay numeric.

Reproduce with `train.py --learners r_learner,dr_learner --reg-params '{"num_leaves": 7, "min_data_in_leaf": 20000}'
--tau-rounds 300 --estimate-propensity` (or `--categorical 1000`) into `--models-dir models/ablations/<variant>`,
`evaluate.py --split dev --models-dir … --out eval/results/ablations/<variant>`, then `ablation_table.py`. The
ablation bundles are not committed.

### 3.4 Uplift by decile (`deciles_dev_visit_<learner>.csv`, `qini_curves_dev.png`, `deciles_dev_s_learner.png`)

Customers sorted by each model's score into ten equal bins (bin 1 = highest predicted uplift); in each, the observed
uplift is the treated visit rate minus the control visit rate, which the randomised assignment makes an unbiased
estimate of the effect the model placed there. The S-learner, best on dev by Qini:

| Decile | n | Control rows | Visit rate, control | Visit rate, treated | Observed uplift [95% CI] | Mean predicted uplift |
|---|---|---|---|---|---|---|
| 1 | 79,999 | 10,784 | 21.37% | 28.03% | **+6.66 pp [5.81, 7.50]** | +5.71 pp |
| 2 | 80,000 | 12,155 | 10.33% | 11.04% | +0.71 pp [0.12, 1.30] | +0.88 pp |
| 3 | 80,000 | 12,272 | 3.72% | 3.93% | +0.21 pp [−0.16, 0.57] | +0.28 pp |
| 4 | 80,000 | 12,350 | 1.29% | 1.51% | +0.22 pp [0.00, 0.44] | +0.11 pp |
| 5 | 80,000 | 12,253 | 0.68% | 0.73% | +0.06 pp [−0.10, 0.21] | +0.07 pp |
| 6 | 79,999 | 12,170 | 0.32% | 0.41% | +0.09 pp [−0.02, 0.21] | +0.03 pp |
| 7 | 80,000 | 12,037 | 0.19% | 0.21% | +0.02 pp [−0.07, 0.11] | +0.01 pp |
| 8 | 80,000 | 11,914 | 0.08% | 0.08% | +0.00 pp [−0.05, 0.06] | +0.00 pp |
| 9 | 80,000 | 12,054 | 0.06% | 0.08% | +0.03 pp [−0.02, 0.07] | +0.00 pp |
| 10 | 80,000 | 12,010 | 2.06% | 2.07% | +0.01 pp [−0.26, 0.29] | −0.02 pp |

- **Top decile ≥ 2 × ATE: passes, by a wide margin.** 6.66 pp is 6.4 × the dev ATE of 1.03 pp; the top 10% of
  customers hold 64% of the whole population's incremental visits. Every learner passes this check (top deciles
  5.98–6.66 pp for the uplift learners, 5.35 pp [4.41, 6.29] for the response model, whose top decile is the only one
  whose CI excludes 6 pp).
- **Bottom decile ≤ ATE: passes** for every learner (S: +0.01 pp [−0.26, 0.29]).
- **Is the bottom decile negative?** Not demonstrably. With the S-learner it is zero inside its CI; the R- and
  DR-learners put a negative point estimate there (−0.14 pp [−0.59, 0.30] and −0.08 pp [−0.52, 0.36]) but both CIs
  contain zero. At decile resolution the data shows no group of 80,000 customers whom the contact hurts; whether a
  smaller predicted-τ < 0 group exists is M3's sleeping-dog question, answered with its own CI.
- **Calibration.** Predicted and observed uplift agree in order everywhere and in level in deciles 2–10; the top
  decile is under-predicted by about 1 pp (5.7 predicted vs 6.7 observed). The transformed-outcome learner shows the
  usual high-variance signature instead, over-predicting its top decile (8.6 predicted vs 6.4 observed).
- **What the ordering is made of.** Deciles 1–2 are the high-base-rate customers (control visit rates 21% and 10%)
  and carry almost all of the effect; deciles 7–9 visit at 0.1% or less with or without the treatment (lost causes);
  decile 10 is a distinct group with a 2% base rate and no measurable effect. The Qini curves (`qini_curves_dev.png`)
  make the same point: the seven curves overlap almost everywhere, the response model's sits visibly below the
  S-learner's only in the first 20%.

### 3.5 The response-model contrast

At a 20% budget on dev, random targeting yields 10.3 incremental visits per 1,000 contacts and captures 20% of the
total incremental effect; the response model yields 38.4 [32.7, 44.5] and captures 74%; the best uplift learner (the
S-learner) yields 41.0 [35.5, 46.0] and captures 79%. The uplift gain over the response model at 20% is therefore
+2.6 visits per 1,000 contacts (+7%), with overlapping CIs. At a 10% budget the gap is three times larger: 53.5
[43.7, 64.5] against 66.6 [58.0, 73.9] per 1,000 (+13.1, +24%), with the shares of the total effect at 52% against
64%; at 30% it has closed (29.5 against 30.2). On Qini the two are 0.0872 [0.0743, 0.1004] and 0.0941 [0.0840,
0.1064]. So on Criteo the response model is a strong baseline, because the treatment effect is close to
proportional to the base rate and ranking by P(visit | treated) ranks by uplift as a by-product; the uplift learners
earn their keep where the budget is tight, which is exactly where a targeting decision bites, and are never worse.
The acceptance criterion written in M0 — the best uplift learner's Qini beats the response model's with
non-overlapping 95% CIs — is not met on dev and is unlikely to be met on test; the README will say so rather than
soften the criterion after the fact. A paired bootstrap of the difference (the two models scored on the same
resamples, which `bootstrap_block` already draws) is the right test for M3.

### 3.6 What carries to M3

- Seven bundles in `models/` (`eval/results/models_visit.json` records each one's settings and dev Qini): S-, R-
  and DR-learners with the settings of §3.2; the other four with the defaults. Best on dev by Qini: the S-learner
  (0.0941), with R (0.0938) and DR (0.0927) within 0.002.
- Known propensity and numeric features (§3.3). The S-learner's segments use its own f(X, 0) as P(visit | control);
  R and DR carry a control-outcome model for the same purpose.
- Evaluation: the raw Qini / policy value on observed outcomes by arm stays primary; M3 adds the IPW-weighted budget
  curve (cross-fitted propensity) as the robustness check to §1.4, paired-difference CIs, the placebo, and the
  `conversion` outcome.
- The dev split has now been used for learner choice, tuning and two ablations; every number above is a dev number
  and the test split is still unopened.

## 4. Decisions (M3)
Budget curves, profit under the stated economics, segments, sleeping dogs, placebo, calibration of uplift by decile,
the `conversion` outcome.

## 5. Test results (M4) — run once on _(date)_

## 6. What I would do differently
