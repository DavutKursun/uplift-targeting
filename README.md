# uplift-targeting

**Decides who should receive a marketing campaign by the effect it has on them, not by how likely they are to buy anyway.**

_(demo GIF and live link arrive in Milestone 5)_

## The problem
A campaign budget covers a fraction of the customers. Targeting the people most likely to convert wastes the
budget on those who would have converted without the offer, and can even hurt by contacting people the offer
annoys. The right question is causal: whose behaviour does the offer change?

## What this does
- Estimates each customer's uplift (treated minus untreated outcome probability) with seven learners, from the
  naive response model to cross-fitted R- and DR-learners, on randomized-experiment data
- Validates every learner against a synthetic world where the true effects are known
- Measures on the holdout what a randomized experiment allows: Qini, observed uplift by decile, incremental
  outcomes per 1,000 contacts at each budget, with bootstrap confidence intervals and a placebo check
- Turns the score into a decision: budget → threshold → treat / skip, with the persuadable / sure thing /
  lost cause / sleeping dog segment

## Results
_(filled at Milestone 4 — test split, used once)_

| Metric (outcome: visit) | Response model | Best uplift learner |
|---|---|---|
| Normalised Qini (95% CI) | | |
| Incremental visits per 1,000 contacts at a 20% budget | | |
| Share of the total incremental effect captured at 20% | | |
| Placebo Qini (treatment shuffled) | | |

## Data
Criteo Uplift Prediction Dataset v2.1 (≈ 14M rows, randomized treatment, 85% treated; outcomes `visit` and
`conversion`). Stratified random splits 60/20/20; a 4M-row working subsample by default.

## Run it yourself
```bash
pip install -r requirements-dev.txt && pytest -q      # no data needed
python scripts/download.py                            # where to put the raw file
make splits ate synth train eval-dev policy-dev placebo
```
