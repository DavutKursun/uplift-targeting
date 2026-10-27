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

## 4. Decisions (M3, 2026-10-28)

Everything in this section is a dev number (799,998 customers, 119,999 in control); the test split is still unopened.
Deployed pair, by the rule fixed in CLAUDE.md §9 before any test run: the learner with the highest dev Qini whose CI
lower bound is within 0.005 of the best, ties to the simpler one. On dev that is the **S-learner** (0.0941, lower
bound 0.0840) with the **R-learner** (0.0938, lower bound 0.0819) as the cross-check; the DR-learner (0.0927, 0.0817)
also qualifies but is neither best nor simpler.

### 4.1 Paired differences against the response model (`scripts/paired.py` → `eval/results/paired_dev_visit.md`)

Two models scored on the same customers are compared on the same stratified resamples (`metrics.paired_block`), so
the CI of the difference is a paired bootstrap; two overlapping marginal CIs, as in §3.5, say nothing about the
difference. 200 resamples; `visit`; random targeting = 10.3 incremental visits per 1,000 at every budget.

| Difference | Normalised Qini | Incremental visits / 1,000 at 10% | at 20% | at 30% |
|---|---|---|---|---|
| S-learner − response | **+0.0069 [0.0007, 0.0135]** | **+13.0 [6.4, 19.2]** | **+2.6 [0.0, 4.8]** | +0.7 [−0.3, 2.0] |
| R-learner − response | +0.0067 [−0.0017, 0.0169] | **+9.5 [2.8, 15.8]** | +1.6 [−1.2, 4.3] | +0.3 [−1.4, 2.0] |

Bold: the CI excludes zero. The paired CI at the 10% budget is ± 6.4 wide where the marginal CIs of §3.1 were ± 8 to
± 10: the pairing removes the noise the two models share. The deployed learner clears the §9 criterion on dev (the 10%
difference excludes zero) and, as a bonus, so does its Qini difference; the 20% difference just excludes zero
(lower bound 0.005) and the 30% one does not. The R-learner clears the 10% criterion only. This is the quantitative
form of §3.5: uplift targeting pays where the budget is tight, and the advantage fades by 30%.

### 4.2 Budget curves, raw and IPW-weighted (`scripts/policy_eval.py` → `policy_dev_visit.csv`, `budget_curve_dev.png`, `profit_curve_dev.png`)

For random targeting, the response model and the deployed pair, at budgets 5–50%: the observed uplift inside the
targeted set (treated minus control visit rate) per 1,000 contacts, raw as everywhere else in this document, and
**IPW-weighted** with a propensity cross-fitted on dev itself (five folds, LightGBM; out-of-fold AUC 0.510, 1st–99th
percentile 0.84–0.89): each treated row in the targeted set weighted 1/ê, each control row 1/(1 − ê), Hájek-normalised.
This is the correction for the imbalance found in §1.4. CIs from 200 stratified resamples of the same loop.

| Budget | random raw / IPW | response raw / IPW | S-learner raw / IPW | R-learner raw / IPW | S-learner raw 95% CI |
|---|---|---|---|---|---|
| 5% | 5.5 / 2.2 | 72.1 / 75.3 | 91.4 / 89.2 | 97.3 / 94.7 | [78.9, 103.7] |
| 10% | 9.2 / 6.5 | 53.5 / 52.5 | 66.6 / 61.9 | 63.0 / 58.8 | [58.0, 74.0] |
| 15% | 9.9 / 7.2 | 42.8 / 39.1 | 49.2 / 44.2 | 47.5 / 42.1 | [41.9, 55.0] |
| 20% | 10.4 / 7.7 | 38.4 / 33.5 | 41.0 / 35.7 | 40.0 / 34.1 | [35.5, 46.0] |
| 30% | 10.0 / 7.4 | 29.5 / 23.9 | 30.2 / 24.7 | 29.8 / 23.8 | [26.7, 33.9] |
| 50% | 10.0 / 7.4 | 20.0 / 15.1 | 20.1 / 15.3 | 20.1 / 15.2 | [18.0, 22.4] |

The full table, with CIs for every model and both estimators and the share of the total effect, is `policy_dev_visit.csv`.

**Where raw and IPW differ by more than the CI.** Everywhere from the 20–25% budget upward, for all four models: the
IPW number is lower by a near-constant factor of about 0.75–0.80 (random 10.4 → 7.7 at 20%; S-learner 20.1 → 15.3 at
50%; the largest gap is 6.2 per 1,000, R-learner at 25%), while the raw CI half-width there is 2–5. At 5–10% the two
agree within 1–5 per 1,000, inside the CI. The factor is the one §1.4 found for the ATE (0.77 vs 1.03 pp): the
imbalance lives in the bulk of low-uplift customers, not in the high-uplift tail that a tight budget selects. So the
correction changes the *level* of every broad-budget number by about a quarter and changes no ranking and no
decision at the budgets that matter; both versions stay in the tables, raw as primary, IPW as the honest second line.

**Profit under the stated economics** (`config.Economics`: 2.00 of value per incremental visit, 0.10 per contact —
assumptions, not measurements; only their ratio matters). One more contact pays only while the targeted set still
yields more than contact_cost / value = **50 incremental visits per 1,000 contacts**, so the curves above say when to
stop. Cumulative profit per 1,000 customers of the population, from the 1%-step grid (point estimates):

| Model | Profit-maximising budget, raw (IPW) | Profit there, per 1,000 customers | Break-even budget, raw (IPW) |
|---|---|---|---|
| random | never profitable | −1.0 at 1% | — |
| response model | 5% (5%) | 2.2 (2.5) | 11% (11%) |
| S-learner | 8% (4%) | 4.8 (4.3) | 14% (12%) |
| R-learner | 5% (4%) | 4.7 (4.5) | 14% (12%) |

Under these assumptions the campaign should contact roughly the top 5–8% and never more than 12–14%; at its best
budget the deployed learner earns about twice the response model's best profit (4.8 vs 2.2 per 1,000 customers) and
the whole gain comes from the tight-budget region of §4.1. At 20% every policy already loses money, random loses at
every budget, and a value-to-cost ratio of 50 instead of 20 would move every break-even past 50% — the README states
the assumption next to the number.

### 4.3 Segments with the S-learner (`policy_dev_visit.json` → `segments`, `worked_customers_dev_visit.md`)

Segments from the model's own P(visit | control) = f(x, 0) and τ̂: persuadable τ̂ > δ; sleeping dog τ̂ < −δ; otherwise
sure thing if P(visit | control) ≥ p0_high else lost cause, with δ = 0.25 × ATE = 0.26 pp and p0_high = 2 × the control
visit rate = 7.64%. Inside each segment the observed uplift (treated minus control rate) is what the randomised
assignment allows us to check.

| Segment | Share | Customers | Control rows | Mean τ̂ | Mean P(visit \| control) | Observed uplift [95% CI] |
|---|---|---|---|---|---|---|
| persuadable | 25.3% | 202,574 | 29,482 | +2.68 pp | 13.6% | **+3.46 pp [3.04, 3.88]** |
| sure thing | 0.9% | 6,872 | 1,065 | +0.05 pp | 35.1% | +0.70 pp [−2.40, 3.81] |
| lost cause | 73.6% | 589,015 | 89,213 | +0.04 pp | 0.3% | +0.07 pp [0.03, 0.11] |
| sleeping dog | 0.2% | 1,537 | 239 | −0.53 pp | 48.3% | +0.08 pp [−6.79, 6.95] |
| τ̂ < 0 (any size) | 6.5% | 52,162 | 7,818 | −0.02 pp | 3.3% | −0.02 pp [−0.44, 0.39] |

- **Sleeping dogs are not demonstrated.** The 1,537 customers the model calls sleeping dogs show +0.08 pp with a CI
  of ± 7 pp (239 control rows), and the much larger τ̂ < 0 group shows −0.02 pp [−0.44, 0.39]: zero, not negative.
  Nothing in this data says that the ad hurts anyone; the label is the model's, and the product will show it with
  that caveat. The decision for those customers is the same either way: skip, because there is nothing to gain.
- **Sure things barely exist here** (0.9%): the customers most likely to visit anyway are also the ones the ad moves
  most (persuadables have a 13.6% base rate and a 3.5 pp effect), which is the proportional-effect structure that
  made the response model competitive in §3.5.
- **Lost causes are three quarters of the population** with a real but tiny effect, +0.07 pp [0.03, 0.11]: a contact
  there buys 0.0014 of value for 0.10 of cost. The persuadable quarter carries the campaign: its 3.46 pp effect is
  3.3 × the ATE and, as in §3.4, its top decile carries 6.7 pp.
- Calibration inside the segments is fair: the persuadables' mean τ̂ (2.68 pp) under-states their observed 3.46 pp,
  the same under-prediction at the top as in §3.4.

**Five worked customers** (one per segment plus one on the 20% threshold; features are Criteo's anonymous f0–f11, so a
row shows what the service returns, not why — the stated limitation, with the interpretable Hillstrom bonus as the
remedy). Scores from the S-learner; treat/skip at the 20% budget, threshold τ̂ ≥ 0.46 pp.

| Role | P(visit \| control) | P(visit \| treated) | τ̂ | Segment | Decision at 20% | Where it sits |
|---|---|---|---|---|---|---|
| persuadable | 33.3% | 59.0% | +25.7 pp | persuadable | treat | the highest τ̂ on dev (f6 = −21.0, f9 = 37.4) |
| sure thing | 82.8% | 82.9% | +0.1 pp | sure thing | skip | visits anyway; the response model would rank it first |
| lost cause | 0.04% | 0.04% | 0.0 pp | lost cause | skip | the lowest base rate on dev |
| sleeping dog | 51.3% | 49.4% | −1.8 pp | sleeping dog | skip | the most negative τ̂; the label is unverified (above) |
| borderline | 8.0% | 8.4% | +0.46 pp | persuadable | treat | exactly on the 20% threshold; skip at any tighter budget |

The full feature rows are in `worked_customers_dev_visit.md`.

### 4.4 Placebo (`scripts/placebo.py` → `placebo_dev_visit_s_learner.json`, `placebo_dev_visit_r_learner.json`)

Treatment labels shuffled within train, each learner refitted with its deployed settings, scored on the real dev
labels, 200 resamples:

| Learner | Placebo Qini [95% CI] | Contains 0 |
|---|---|---|
| S-learner (31 leaves, 300 rounds) | −0.0022 [−0.0136, 0.0095] | yes |
| R-learner (τ: 7 leaves, min_data_in_leaf 20,000, 300 rounds) | +0.0008 [−0.0127, 0.0126] | yes |

A pipeline that could manufacture a Qini from nothing would show it here; the real Qini of 0.094 is seven placebo
standard errors away from zero.

### 4.5 The `conversion` outcome (`dev-summary-conversion.md`, `paired_dev_conversion.md`, `deciles_dev_conversion_*.csv`)

The same pipeline on the rare outcome, with the settings chosen on `visit` (the spec's rule: tune once on the primary
outcome): S-learner, R-learner and the response model trained on `conversion`, evaluated on dev with 200 resamples.
Dev ATE 0.122 pp [0.094, 0.150]; random targeting = 1.22 incremental conversions per 1,000 contacts.

| Model | Normalised Qini [95% CI] | Conversions / 1,000 at 10% [CI] | at 20% [CI] | at 30% [CI] | Share of total at 10% |
|---|---|---|---|---|---|
| S-learner | 0.202 [0.140, 0.268] | 8.7 [5.8, 11.5] | 5.4 [3.8, 6.6] | 3.8 [2.8, 4.7] | 71% |
| response model | 0.197 [0.134, 0.255] | 8.8 [5.9, 11.8] | 5.2 [3.7, 6.6] | 3.7 [2.7, 4.6] | 72% |
| R-learner | 0.167 [0.110, 0.224] | 7.7 [5.1, 10.4] | 4.7 [3.3, 6.0] | 3.5 [2.6, 4.4] | 63% |

Paired differences, same resamples:

| Difference | Qini | / 1,000 at 10% | at 20% | at 30% |
|---|---|---|---|---|
| S-learner − response | +0.005 [−0.005, 0.019] | −0.1 [−0.7, 0.4] | +0.2 [−0.0, 0.4] | +0.1 [−0.0, 0.2] |
| R-learner − response | **−0.031 [−0.053, −0.000]** | **−1.1 [−2.0, −0.1]** | **−0.5 [−0.9, −0.0]** | −0.2 [−0.4, 0.1] |

The S-learner's top decile carries +0.87 pp [0.59, 1.15], seven times the ATE and 71% of the whole effect; its bottom
decile is −0.01 pp [−0.04, 0.02]. The R-learner's bottom decile holds +0.08 pp [0.01, 0.15] of *positive* observed
uplift: at a 0.3% base rate the pseudo-outcome regression misranks a converting group to the bottom, which is what
costs it a paired 1.1 conversions per 1,000 at the 10% budget against the response model.

**The honest sentence.** On conversion the CIs are the result: the effect is 1.2 conversions per 1,000 contacts, the
best policy's 20% number is 5.4 with a CI of ± 1.4 (a quarter of the estimate), the paired difference between the
S-learner and the response model is ± 0.5 per 1,000 around zero at every budget, and the R-learner is measurably
worse. Nothing on this outcome separates uplift targeting from response targeting; the ranking *is* real (every Qini
CI is far from zero, and it is more concentrated than for visits because conversions are rarer), but which model
produces it cannot be told apart on 800,000 customers, and the README will say so instead of printing 0.202 against
0.197 as if it meant something.

### 4.6 What carries to M4

- **Deployed pair:** S-learner deployed, R-learner as the cross-check (rule in CLAUDE.md §9). On conversion the
  same S-learner bundle is the one to serve; the R-learner is not.
- **The decision the product makes:** under the stated economics, contact the top 5–8% by predicted uplift and stop
  by 12–14%; at a 20% budget the campaign loses money with any model. The budget slider in M5 shows profit next to
  incremental visits so the user sees both.
- **Segments:** labels from the S-learner's own P(visit | control) and τ̂ with δ = ATE / 4 and p0_high = 2 × control
  rate; the sleeping-dog label is shown with the caveat that the data does not confirm harm (§4.3).
- **The single test run (M4):** `evaluate.py --split test --final` for visit and conversion, `paired.py --split test
  --final` with the pairs above and budgets 10 / 20 / 30%, `policy_eval.py --split test --final` with the dev
  budgets, `placebo.py` on test for the deployed learner, the segment table on test; every dev number above gets its
  test counterpart once, and the date goes into CLAUDE.md §0 and DESIGN.md §5.
- **Criterion on test:** the paired CI of S-learner − response at the 10% budget excludes zero; the 20%, 30% and
  Qini differences and the IPW versions are reported whatever they are; the placebo CI contains zero.

## 5. Test results (M4) — run once on 2026-10-28

The test split (800,004 customers, 120,000 in control) was opened once, on 2026-10-28, for this sequence and nothing
else: `evaluate.py --split test --final` for `visit` (all seven bundles) and `conversion` (three), `policy_eval.py
--split test --final` (budget curves raw and IPW, profit, segments), `paired.py --split test --final` for both
outcomes, `placebo.py --split test --final` for the S-learner (one fit on shuffled train labels), `plots.py --split
test --final`. Every setting, budget grid, δ (0.0025855) and p0_high (0.0764) was frozen at its dev value; the IPW
propensity was cross-fitted on test as it had been on dev. No script was run twice. `scripts/dev_test_table.py`
builds every table below from the result JSONs (`eval/results/dev_vs_test.md`); each block ends with the verdict.

### 5.1 Qini and per-1,000 — `visit` (dev 799,998 rows → test 800,004 rows, 200 resamples)

ATE: dev 1.034 pp [0.914, 1.154] → test 1.034 pp [0.914, 1.153]; random = 10.34 → 10.34 per 1,000.

| Learner | Dev Qini [CI] | Test Qini [CI] | Dev / 1,000 at 10% | Test at 10% | Dev at 20% | Test at 20% | Test at 30% |
|---|---|---|---|---|---|---|---|
| DR-learner | 0.0927 [0.0817, 0.1055] | 0.0854 [0.0743, 0.0972] | 61.5 [53.8, 68.9] | 58.5 [49.5, 66.8] | 40.4 [35.2, 45.4] | 39.8 [34.6, 44.3] | 29.1 [26.1, 32.3] |
| response model | 0.0872 [0.0743, 0.1004] | 0.0853 [0.0736, 0.0957] | 53.5 [43.7, 64.5] | 49.0 [39.2, 57.9] | 38.4 [32.7, 44.5] | 39.2 [33.6, 44.0] | 29.4 [25.4, 32.5] |
| S-learner | 0.0941 [0.0840, 0.1064] | 0.0853 [0.0746, 0.0955] | 66.5 [58.0, 73.9] | 62.9 [54.3, 71.5] | 41.0 [35.5, 46.0] | 38.9 [33.7, 43.5] | 28.4 [24.8, 31.4] |
| transformed outcome | 0.0897 [0.0770, 0.1012] | 0.0845 [0.0709, 0.0963] | 63.9 [55.8, 72.1] | 58.0 [50.2, 66.3] | 39.9 [34.8, 44.7] | 37.8 [32.8, 43.0] | 28.9 [25.0, 32.4] |
| X-learner | 0.0898 [0.0786, 0.1017] | 0.0845 [0.0716, 0.0959] | 61.5 [53.4, 68.8] | 61.2 [52.5, 68.8] | 39.4 [34.3, 44.7] | 37.8 [32.7, 42.1] | 29.3 [25.8, 32.3] |
| R-learner | 0.0938 [0.0819, 0.1060] | 0.0844 [0.0727, 0.0966] | 63.0 [54.7, 70.7] | 59.5 [50.2, 67.4] | 40.0 [34.4, 45.0] | 39.2 [33.8, 43.9] | 28.9 [25.7, 32.0] |
| T-learner | 0.0867 [0.0733, 0.0995] | 0.0759 [0.0634, 0.0873] | 59.8 [51.7, 66.8] | 53.6 [45.5, 62.7] | 38.9 [33.8, 43.7] | 36.5 [31.9, 40.8] | 27.9 [24.3, 31.0] |

**Does test confirm dev? Mostly.** Every test Qini sits inside its dev CI, every CI stays far from zero, and the
learners remain inseparable from one another. But every tuned learner lost 0.007–0.009 (S 0.0941 → 0.0853, R 0.0938 →
0.0844, DR 0.0927 → 0.0854) while the untuned ones lost 0.002–0.005 (response 0.0872 → 0.0853, X and TO −0.005): the
optimism of choosing settings on dev, worth about 0.005 of Qini, and on test the response model ties the S-learner on
Qini to four decimals. The per-1,000 numbers at 10% replicate inside their CIs (S 66.5 → 62.9, response 53.5 → 49.0).
The T-learner drops most (0.0867 → 0.0759) and is the only learner whose test value approaches its dev lower bound.

### 5.2 Paired differences — `visit`

| Difference | Statistic | Dev [CI] | Test [CI] | Test excludes 0 |
|---|---|---|---|---|
| S-learner − response model | Qini | 0.0069 [0.0007, 0.0135] | -0.0001 [-0.0053, 0.0067] | no |
| S-learner − response model | per 1,000 at 10% | 13.0 [6.4, 19.2] | 13.9 [6.5, 21.6] | yes |
| S-learner − response model | per 1,000 at 20% | 2.6 [0.0, 4.8] | -0.4 [-2.5, 2.1] | no |
| S-learner − response model | per 1,000 at 30% | 0.7 [-0.3, 2.0] | -0.9 [-2.2, 0.2] | no |
| R-learner − response model | Qini | 0.0067 [-0.0017, 0.0169] | -0.0009 [-0.0100, 0.0096] | no |
| R-learner − response model | per 1,000 at 10% | 9.5 [2.8, 15.8] | 10.6 [4.3, 17.5] | yes |
| R-learner − response model | per 1,000 at 20% | 1.6 [-1.2, 4.3] | -0.0 [-3.1, 2.9] | no |
| R-learner − response model | per 1,000 at 30% | 0.3 [-1.4, 2.0] | -0.5 [-2.0, 1.4] | no |

**Confirms the headline, not the halo.** S − response at 10%: +13.0 [6.4, 19.2] on dev → **+13.9 [6.5, 21.6] on test;
the CI excludes zero and the §9 criterion is met.** R − response at 10% replicates the same way (+9.5 → +10.6 [4.3,
17.5]). The two dev results that sat on the edge did not replicate, and their test values fall outside the dev CI:
the Qini difference (dev +0.0069 [0.0007, 0.0135] → test −0.0001 [−0.0053, 0.0067]) and the 20% difference (dev +2.6
[0.0, 4.8] → test −0.4 [−2.5, 2.1]). One explanation covers both: Qini integrates the whole budget range, the 20–100%
part of it is where the response model is as good, and the S-learner's dev edge there was partly the selection of its
settings on dev (its dev Qini was the maximum of four configurations). The advantage that survives is exactly the one
a tight-budget decision needs.

### 5.3 Budget curves — incremental visits per 1,000 contacts, raw / IPW (dev → test)

Propensity for IPW, cross-fitted on each split: dev AUC 0.5095 → test AUC 0.5093.

| Model | Budget | Dev raw [CI] | Test raw [CI] | Dev IPW | Test IPW |
|---|---|---|---|---|---|
| random | 5% | 5.5 [-0.4, 10.0] | 9.2 [3.9, 14.6] | 2.2 | 5.6 |
| random | 10% | 9.2 [5.9, 12.8] | 9.0 [5.2, 12.7] | 6.5 | 5.8 |
| random | 20% | 10.4 [8.0, 12.9] | 9.9 [7.4, 12.6] | 7.7 | 6.7 |
| random | 30% | 10.0 [7.9, 12.3] | 11.3 [9.0, 13.4] | 7.4 | 8.2 |
| response model | 5% | 72.1 [55.3, 87.0] | 55.5 [42.5, 69.4] | 75.3 | 58.8 |
| response model | 10% | 53.5 [43.7, 64.5] | 49.0 [39.2, 57.9] | 52.5 | 46.8 |
| response model | 20% | 38.4 [32.7, 44.5] | 39.2 [33.6, 44.0] | 33.5 | 33.1 |
| response model | 30% | 29.5 [25.6, 33.6] | 29.4 [25.4, 32.5] | 23.9 | 23.1 |
| S-learner | 5% | 91.4 [78.9, 103.7] | 84.7 [71.8, 99.4] | 89.2 | 80.8 |
| S-learner | 10% | 66.6 [58.0, 73.9] | 62.9 [54.3, 71.5] | 61.9 | 56.9 |
| S-learner | 20% | 41.0 [35.5, 46.0] | 38.9 [33.7, 43.5] | 35.7 | 33.1 |
| S-learner | 30% | 30.2 [26.7, 33.9] | 28.4 [24.8, 31.4] | 24.7 | 22.6 |
| R-learner | 5% | 97.3 [86.5, 110.4] | 88.1 [76.6, 100.8] | 94.7 | 84.0 |
| R-learner | 10% | 63.0 [54.7, 70.7] | 59.5 [50.2, 67.4] | 58.8 | 54.2 |
| R-learner | 20% | 40.0 [34.4, 45.0] | 39.2 [33.8, 43.9] | 34.1 | 32.9 |
| R-learner | 30% | 29.8 [26.0, 33.3] | 28.9 [25.7, 32.0] | 23.8 | 22.7 |

**Confirms, with the 5% point lower on test for every model.** The 5% numbers are the noisiest (40,000 customers, 6,000 in
control): response 72.1 → 55.5, S 91.4 → 84.7, R 97.3 → 88.1, all inside the dev CI, the response model's at its edge.
From 10% upward the curves replicate within 1–4 per 1,000. The IPW-weighted curves behave exactly as on dev: the same
propensity quality (AUC 0.509 vs 0.510), lower than raw by about a quarter from the 15–20% budget upward and beyond
the raw CI half-width there (the rows are listed in `policy_test_visit.json`), identical rankings.

### 5.4 Profit under the stated economics (dev → test)

Value per visit 2.0, contact cost 0.1 (assumptions); marginal break-even 50 per 1,000 contacts.

| Model | Best budget raw (IPW), dev → test | Max profit per 1,000 customers, dev → test | Break-even budget raw (IPW), dev → test |
|---|---|---|---|
| random | 1% (1%) → 1% (1%) | -1.0 → -0.9 | — (—) → — (—) |
| response model | 5% (5%) → 6% (5%) | 2.2 → 0.7 | 11% (11%) → 9% (8%) |
| S-learner | 8% (4%) → 6% (4%) | 4.8 → 3.6 | 14% (12%) → 14% (11%) |
| R-learner | 5% (4%) → 5% (5%) | 4.7 → 3.8 | 14% (12%) → 13% (11%) |

**Confirms the guidance; the response model's best case is weaker on test.** Best budgets 5–6% and break-even 13–14%
(11% IPW) for the uplift pair, 9% for the response model; at 20% every policy loses money on test as on dev. The
response model's maximum profit falls from 2.2 to 0.7 per 1,000 customers because its 5–6% per-1,000 fell from 72 to
55; the S-learner's falls from 4.8 to 3.6; on test the uplift learner's best profit is five times the response model's.
These are point estimates on a 1% grid under assumed economics: the budget guidance is the robust part, the profit
multiples are not.

### 5.5 Segments with the S-learner (dev → test; δ and p0_high fixed at the dev values)

| Segment | Dev share | Test share | Dev observed uplift [CI] | Test observed uplift [CI] |
|---|---|---|---|---|
| persuadable | 25.3% | 25.4% | +3.46 pp [3.04, 3.88] | +3.23 pp [2.80, 3.65] |
| sure_thing | 0.9% | 0.9% | +0.70 pp [-2.40, 3.81] | +2.05 pp [-1.11, 5.20] |
| lost_cause | 73.6% | 73.6% | +0.07 pp [0.03, 0.11] | +0.03 pp [-0.00, 0.07] |
| sleeping_dog | 0.2% | 0.2% | +0.08 pp [-6.79, 6.95] | +10.12 pp [3.35, 16.88] |
| τ̂ < 0 (any size) | 6.5% | 6.6% | -0.02 pp [-0.44, 0.39] | +0.31 pp [-0.10, 0.72] |

**Shares identical; three of four observed uplifts replicate; the sleeping-dog group reverses.** Persuadables +3.46 →
+3.23 pp (inside the dev CI), sure things and lost causes inside their wide CIs, the τ̂ < 0 group zero on dev and
+0.31 pp [−0.10, 0.72] on test. The 1,505 customers the model flags as sleeping dogs on test show **+10.1 pp [3.4,
16.9]** of observed uplift, outside the dev CI of [−6.8, 7.0] and clearly positive. Dev and test are two independent
samples of the same tiny group (242 control rows on test; a 49% base rate), and together they say the group is helped
by contact, not hurt: the model's negative τ̂ there (mean −0.5 pp) is noise around a large positive effect in a
high-base-rate region. On test the sleeping-dog label is not merely unverified, it is wrong for the group it names.
The product keeps the four-segment vocabulary but acts on this label only as "no predicted gain", and the README says so.

### 5.6 Placebo (treatment shuffled in train, deployed settings)

| Learner | Scored on | Placebo Qini [CI] | Contains 0 |
|---|---|---|---|
| S-learner | dev | -0.0022 [-0.0136, +0.0095] | yes |
| R-learner | dev | +0.0008 [-0.0127, +0.0126] | yes |
| S-learner | test | -0.0155 [-0.0287, -0.0024] | NO |

**The single-shuffle placebo criterion fails on test by a hair, and the criterion is the problem.** The same placebo
model (one shuffle of the train labels, seed 7, deployed settings) scores −0.0022 [−0.0136, 0.0095] on dev and
−0.0155 [−0.0287, −0.0024] on test. Two things are true at once: the sign is negative, so this is not leakage (which
would produce a positive Qini), and the test CI excludes zero. The bootstrap CI covers the sampling of customers with
the placebo ranking held fixed; it does not cover the randomness of the shuffle itself, which is the larger source of
variance here, because the true uplift is so concentrated (the top decile holds 60% of it) that any chance split on
the shuffled treatment flag inside a high-base-rate region moves a large block of true uplift to the top or the
bottom of the ranking. The right null is the spread of placebo Qini over independent shuffles, and that is now the criterion (CLAUDE.md §9,
changed at M5 with authorisation): `scripts/placebo.py --shuffles 5` fits five independent shuffles of the train labels
with the deployed settings and scores each on dev (`eval/results/placebo_null_dev_visit_s_learner.json`): Qini +0.027, -0.036, -0.051, -0.073, -0.032,
a null spanning **-0.073 to +0.027** (SD 0.037), five times wider than any single shuffle's
customer-bootstrap CI. **Verdict: the deployed model's test Qini, 0.0853, lies outside that spread; the criterion is met.**
The single-shuffle CIs stay in the table above as reported numbers; the test placebo's −0.0155 sits inside the null,
as a placebo should. The earlier wording, "the placebo Qini CI contains 0", was met on dev and not on test, and this
document records both.

### 5.7 Qini and per-1,000 — `conversion` (dev 799,998 rows → test 800,004 rows, 200 resamples)

ATE: dev 0.122 pp [0.094, 0.150] → test 0.125 pp [0.097, 0.152]; random = 1.22 → 1.25 per 1,000.

| Learner | Dev Qini [CI] | Test Qini [CI] | Dev / 1,000 at 10% | Test at 10% | Dev at 20% | Test at 20% | Test at 30% |
|---|---|---|---|---|---|---|---|
| S-learner | 0.2020 [0.1402, 0.2675] | 0.2175 [0.1568, 0.2824] | 8.7 [5.8, 11.5] | 9.5 [6.7, 12.2] | 5.4 [3.8, 6.6] | 5.6 [4.3, 6.9] | 4.0 [3.1, 4.9] |
| response model | 0.1974 [0.1341, 0.2549] | 0.2148 [0.1555, 0.2746] | 8.8 [5.9, 11.8] | 9.4 [6.8, 12.2] | 5.2 [3.7, 6.6] | 5.5 [4.2, 6.7] | 3.8 [3.0, 4.6] |
| R-learner | 0.1666 [0.1096, 0.2241] | 0.2000 [0.1389, 0.2668] | 7.7 [5.1, 10.4] | 8.8 [6.0, 11.8] | 4.7 [3.3, 6.0] | 5.2 [3.7, 6.6] | 3.8 [2.8, 4.7] |

**Confirms.** All three Qini values rise slightly on test (S 0.202 → 0.218, response 0.197 → 0.215, R 0.167 → 0.200)
and stay inside their dev CIs; the per-1,000 numbers move by less than one conversion.

### 5.8 Paired differences — `conversion`

| Difference | Statistic | Dev [CI] | Test [CI] | Test excludes 0 |
|---|---|---|---|---|
| S-learner − response model | Qini | 0.0046 [-0.0045, 0.0189] | 0.0026 [-0.0059, 0.0157] | no |
| S-learner − response model | per 1,000 at 10% | -0.1 [-0.7, 0.4] | 0.1 [-0.4, 0.6] | no |
| S-learner − response model | per 1,000 at 20% | 0.2 [-0.0, 0.4] | 0.1 [-0.1, 0.4] | no |
| S-learner − response model | per 1,000 at 30% | 0.1 [-0.0, 0.2] | 0.2 [0.0, 0.3] | yes |
| R-learner − response model | Qini | -0.0309 [-0.0527, -0.0001] | -0.0148 [-0.0420, 0.0096] | no |
| R-learner − response model | per 1,000 at 10% | -1.1 [-2.0, -0.1] | -0.6 [-1.4, 0.2] | no |
| R-learner − response model | per 1,000 at 20% | -0.5 [-0.9, -0.0] | -0.3 [-0.8, 0.0] | no |
| R-learner − response model | per 1,000 at 30% | -0.2 [-0.4, 0.1] | -0.1 [-0.4, 0.2] | no |

**Confirms that nothing separates them.** Seven of the eight test differences include zero; the eighth, S − response at
30%, is +0.17 conversions per 1,000 contacts [0.03, 0.31]: one of eight statistics, on a scale where the ATE is 1.25,
and not a basis for anything. The R-learner's dev deficit shrinks on test (Qini −0.031 → −0.015) and its CIs now
include zero.

### 5.9 Uplift by decile on test, deployed pair (`deciles_test_visit_s_learner.csv`, `deciles_test_visit_r_learner.csv`, `deciles_test_s_learner.png`)

Observed uplift per decile of the score (bin 1 = highest predicted uplift), treated minus control visit rate, with the mean predicted uplift:

| Decile | S-learner observed [95% CI] | S-learner predicted | R-learner observed [95% CI] | R-learner predicted |
|---|---|---|---|---|
| 1 | +6.29 pp [5.44, 7.13] | +5.73 pp | +5.95 pp [5.11, 6.79] | +6.05 pp |
| 2 | +0.79 pp [0.20, 1.38] | +0.88 pp | +1.05 pp [0.52, 1.59] | +1.06 pp |
| 3 | +0.08 pp [-0.30, 0.46] | +0.29 pp | +0.20 pp [-0.14, 0.53] | +0.36 pp |
| 4 | +0.17 pp [-0.05, 0.38] | +0.11 pp | +0.20 pp [-0.03, 0.43] | +0.15 pp |
| 5 | +0.12 pp [-0.04, 0.27] | +0.06 pp | +0.10 pp [-0.04, 0.24] | +0.07 pp |
| 6 | +0.09 pp [-0.01, 0.18] | +0.03 pp | +0.02 pp [-0.09, 0.14] | +0.04 pp |
| 7 | +0.03 pp [-0.05, 0.10] | +0.01 pp | +0.05 pp [-0.01, 0.11] | +0.02 pp |
| 8 | +0.01 pp [-0.05, 0.06] | +0.00 pp | +0.02 pp [-0.02, 0.07] | +0.01 pp |
| 9 | +0.01 pp [-0.04, 0.06] | +0.00 pp | -0.00 pp [-0.05, 0.05] | +0.01 pp |
| 10 | +0.22 pp [-0.05, 0.49] | -0.01 pp | +0.25 pp [-0.19, 0.69] | -0.18 pp |

**Confirms.** Top decile ≥ 2 × ATE: S +6.29 pp = 6.1 × the test ATE of 1.03 pp (dev 6.66), R +5.95 pp (dev 6.30); bottom
decile ≤ ATE: S +0.22 pp [−0.05, 0.49], R +0.25 pp [−0.19, 0.69] (dev +0.01 and −0.14). No decile where contact hurts,
on either split. Calibration as on dev: order right everywhere, the top decile under-predicted by about half a point.
The decile plot for the S-learner was rendered from this CSV (`plots.py` had drawn the DR-learner's, the top test Qini
by 0.0001, under its "best by Qini" rule; the test split was not read again).

### 5.10 The verdict on the acceptance criteria (CLAUDE.md §9)

| Criterion | Result on test |
|---|---|
| Synthetic: every uplift learner Spearman ≥ 0.6 with the truth, the response model not | met in M1 (0.66–0.88; −0.08) |
| Paired bootstrap, best uplift − response, incremental visits per 1,000 at 10%: CI excludes 0 | **met: +13.9 [6.5, 21.6]** |
| Differences at 20%, 30% and in Qini, reported whatever they are | −0.4 [−2.5, 2.1]; −0.9 [−2.2, 0.2]; −0.0001 [−0.0053, 0.0067] |
| Placebo: the deployed model's test Qini lies outside the spread of five placebo shuffles scored on dev (criterion changed at M5, authorised) | **met**: null -0.073 to +0.027, test Qini 0.0853. The earlier single-shuffle "CI contains 0" was met on dev (−0.0022 [−0.0136, 0.0095]) and not on test (−0.0155 [−0.0287, −0.0024]); see §5.6 |
| Decile table: top ≥ 2 × ATE, bottom ≤ ATE | met (6.1 × ATE; +0.22 pp) |
| Policy at 20%: per-1,000 for uplift vs response vs random with CIs, share of the total effect | 38.9 [33.7, 43.5] vs 39.2 [33.6, 44.0] vs 9.9 [7.4, 12.6]; shares 75% / 76% / 20% |
| `/score` p50 < 30 ms; pytest green without data; CI green; README on one screen | M5; 21 tests green; CI green; README rewritten |

What test adds to dev in one sentence: **the uplift advantage is real and worth about 14 incremental visits per
1,000 contacts at a 10% budget, it is gone by 20%, and nothing else about the deployed model changed except the
discovery that its sleeping-dog label points at customers who are helped.**

## 6. What I would do differently

- **Check the randomization before writing the spec.** The spec assumed a clean RCT (max |SMD| < 0.02) and arm rates
  from memory; both were wrong (§1). Half a day of M1 went into understanding an imbalance that a ten-line check on
  day one would have surfaced, and the IPW-weighted estimators (§4.2) would have been in the design from the start
  rather than added as a correction.
- **Write the acceptance criterion as a paired test from the beginning.** "Non-overlapping CIs" between two models
  scored on the same customers was the wrong test (§3.5 vs §4.1); the paired bootstrap costs nothing extra because the
  resamples are shared anyway. The criterion was changed at M3 with authorisation, before any test run, but a reader
  is right to ask whether it would have been changed had the dev result gone the other way. Fix criteria before the
  data, and fix the *right* ones.
- **Tune the τ model on a synthetic world first, then on dev.** The loose starter settings overfit noise (§2); the
  M1 synthetic check caught it, but only because the acceptance bar there was quantitative (Spearman ≥ 0.6). More of
  the pipeline deserves a known-truth check — in particular the segments, whose sleeping-dog label cannot be
  validated on Criteo (§4.3) but can on the synthetic world where sleeping dogs exist by construction.
- **Pick a dataset with interpretable features or run the Hillstrom bonus early.** Every worked customer in §4.3 is a
  row of anonymous numbers; the product can say *that* someone is persuadable, not *why*. The interpretable
  multi-treatment dataset was planned as M6 and should have been M2.
- **Budget the evaluation compute up front.** 200-resample bootstraps of seven models cost 8–12 minutes per split per
  outcome on a laptop; the joint resample loop (§3) and the paired block (§4.1) were written mid-project when the
  first bootstrap ran long. Writing `bootstrap_block` first would have saved an hour and a lot of care about which
  numbers were comparable.
- **Decide the deployed pair's rule before seeing dev.** It was fixed at M3 (CLAUDE.md §9), after the dev tables
  existed; it happens to be a rule that any sensible analyst would write, but it should have been in the M0 spec.
- **Report conversion as the first line, not the second**, in any real campaign where conversions are what gets paid
  for: §4.5 shows that the ranking is strong and the learner choice is undecidable at 800k customers. The honest
  product answer on a rare outcome is "use the simplest model and a bigger holdout", and the README now says so.
