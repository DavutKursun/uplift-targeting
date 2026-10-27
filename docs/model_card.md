# Model card — uplift-targeting (filled at Milestone 4)

- **Models:** uplift meta-learners (S, T, X, transformed outcome, R, DR) with LightGBM base models; a response model as the baseline
- **Intended use:** portfolio demonstration of causal targeting on randomized-experiment data; not a production marketing system
- **Training data:** Criteo Uplift Prediction Dataset v2.1 (randomized treatment, anonymous features)
- **Evaluation:** stratified holdout, test used once; Qini, uplift by decile, policy value with bootstrap CIs; placebo check
- **Known limitations:** anonymous features (explanations are feature indices); single treatment; no timestamps (random split); the `conversion` outcome is rare and its estimates are wide
