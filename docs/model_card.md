# Model card — uplift-targeting

## Model details
- **Deployed model:** S-learner — one LightGBM classifier f(x, t) on the 12 Criteo features plus the treatment flag;
  uplift τ̂(x) = f(x, 1) − f(x, 0); P(visit | control) = f(x, 0). Settings chosen once on dev Qini: 31 leaves,
  min_data_in_leaf 2,000, learning rate 0.03, 300 rounds, L2 10, bagging 0.8. Bundle `models/s_learner-visit.joblib`
  (2.1 MB). A second bundle for the `conversion` outcome with the same settings.
- **Cross-check model:** R-learner (Nie & Wager) — cross-fitted outcome model, known propensity 0.85, weighted τ
  regression (7 leaves, min_data_in_leaf 20,000, 300 rounds). `models/r_learner-visit.joblib`.
- **Baseline kept in every table:** response model P(visit | x, treated).
- **Also trained and reported:** T-, X-, DR-learners and the transformed-outcome regression (`docs/DESIGN.md` §3).
- Version: the commit hash of this repository; date of the single test run: 2026-10-28.
- **Served at** https://davutkursun-uplift-targeting.hf.space (Hugging Face Docker Space, cpu-basic): `POST /score`, `POST /score/batch`,
  `GET /policy`, `GET /health`, demo UI at `/ui`. Model time about 1 ms per customer; round-trip p50 about 165 ms from a laptop
  (`eval/results/latency_space.json`). The τ̂ < −δ segment is served as `no_detectable_effect` with the sentence "negative predicted
  uplift was not confirmed on held-out data; do not treat as harm".

## Intended use
Portfolio demonstration of causal targeting: for a customer described by Criteo's 12 anonymous features, an estimated
uplift in visit probability, a segment label and a treat / skip decision under a budget. Not a production marketing
system; the data licence (CC BY-NC-SA 4.0) forbids commercial use of anything derived from it.

## Training data
Criteo Uplift Prediction Dataset v2.1 (13,979,592 rows; treatment randomised with an 85% treated share; outcomes
`visit` 4.70% and `conversion` 0.29%). A stratified 4,000,000-row subsample split 60 / 20 / 20 by (treatment, visit);
models fit on the 2,399,998-row train split, chosen on dev (799,998), evaluated once on test (800,004).

## Evaluation (test split, used once on 2026-10-28; `docs/DESIGN.md` §5)
| Quantity (visit, test split) | S-learner (deployed) | Response model | Random |
|---|---|---|---|
| Normalised Qini [95% CI] | 0.085 [0.075, 0.096] | 0.085 [0.074, 0.096] | 0 |
| Incremental visits / 1,000 contacts at 10% [CI] | 62.9 [54.3, 71.5] | 49.0 [39.2, 57.9] | 9.0 [5.2, 12.7] |
| at 20% [CI] | 38.9 [33.7, 43.5] | 39.2 [33.6, 44.0] | 9.9 [7.4, 12.6] |
| at 30% [CI] | 28.4 [24.8, 31.4] | 29.4 [25.4, 32.5] | 11.3 [9.0, 13.4] |
| Share of the total effect captured at 10% / 20% | 61% / 75% | 47% / 76% | 10% / 20% |
| Top-decile observed uplift | +6.29 pp [5.44, 7.13] (6.1 × ATE) | +4.90 pp [3.95, 5.84] | — |

Paired S-learner − response (same resamples): +13.9 [6.5, 21.6] per 1,000 at 10% (the acceptance criterion, met);
−0.4 [−2.5, 2.1] at 20%; −0.9 [−2.2, 0.2] at 30%; Qini −0.000 [−0.005, 0.007]. R-learner cross-check: +10.6 [4.3, 17.5]
at 10%. Profit under the stated economics (2.00 per visit, 0.10 per contact): best budget 6%, break-even 14%, at 20%
every policy loses. Placebo (labels shuffled before training): −0.002 [−0.014, 0.010] on dev, −0.016 [−0.029, −0.002]
on test for one shuffle (negative, not leakage; see `docs/DESIGN.md` §5.6). `conversion` (0.3% base rate): Qini
S-learner 0.218 [0.157, 0.282] vs response 0.215 [0.156, 0.275], paired difference +0.003 [−0.006, 0.016]: nothing
separates the two on the rare outcome. Full tables with their dev counterparts: `docs/DESIGN.md` §5.

## Metrics and how to read them
- Normalised Qini: area between the model's Qini curve and the random line, over the same area for the perfect
  ordering of observed outcomes; 0 = random, 1 = perfect. Every CI is a stratified bootstrap with 200 resamples.
- Incremental visits per 1,000 contacts at a budget: treated minus control visit rate inside the targeted top
  fraction, an unbiased estimate under randomisation. Reported raw and IPW-weighted (cross-fitted propensity).
- Paired differences between two models use the same resamples; that is the test for "better than the response model".

## Limitations
- **Imperfect randomisation.** Treatment is weakly predictable from the features (AUC 0.51; max |SMD| 0.05); every
  covariate adjustment lowers the visit ATE from 1.03 to about 0.8 pp. Raw numbers are primary (as in the dataset
  paper); IPW-weighted numbers are reported next to them and are about a quarter lower at broad budgets. Rankings
  and decisions do not change.
- **Sleeping dogs are not demonstrated.** The 0.2% of customers the model labels as harmed by contact show an observed
  uplift of +0.1 pp [−6.8, 7.0] on dev and +10.1 pp [3.4, 16.9] on test: helped, not hurt. The label is shown but acted on
  only as "no predicted gain".
- **Anonymous features.** The 12 inputs are unnamed floats; explanations are limited to "feature f3 is high".
- **Rare outcome.** On `conversion` (0.3% base rate) the CIs are as wide as the differences between models; nothing
  separates uplift targeting from response targeting there.
- **Single treatment, no timestamps, one dataset.** Multi-treatment, time-aware splits and other campaigns are out of scope.
- **Economics are assumptions.** Value per visit 2.0 and contact cost 0.10 are stated inputs, not measurements; only
  their ratio matters for the budget guidance.

## How to reproduce
`make splits ate synth train eval-dev policy-dev paired-dev placebo`, then the single test run documented in
`CLAUDE.md` §0 (M4) and `docs/DESIGN.md` §5. `pytest -q` needs no data.
