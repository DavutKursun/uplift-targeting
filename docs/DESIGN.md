# Design notes

Filled milestone by milestone; each section says what was measured, on which split, and what it changes.

## 1. The experiment (M1)
Arm sizes, outcome rates, ATE with CIs for `visit` and `conversion`, randomization table, subsample and splits.

## 2. Ground truth on synthetic data (M1)
PEHE and Spearman with the true τ for every learner; why the response model fails there by construction.

## 3. Learners on dev (M2)
Qini / AUUC with CIs, decile tables, hyperparameter choice, the categorical ablation, the propensity ablation.

## 4. Decisions (M3)
Budget curves, profit under the stated economics, segments, sleeping dogs, placebo, calibration of uplift by decile,
the `conversion` outcome.

## 5. Test results (M4) — run once on _(date)_

## 6. What I would do differently
