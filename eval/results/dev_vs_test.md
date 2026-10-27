# Dev → test, side by side (test run 2026-10-28)

### Qini and per-1,000 — `visit` (dev 799,998 rows → test 800,004 rows, 200 resamples)

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

### Paired differences — `visit`

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

### Qini and per-1,000 — `conversion` (dev 799,998 rows → test 800,004 rows, 200 resamples)

ATE: dev 0.122 pp [0.094, 0.150] → test 0.125 pp [0.097, 0.152]; random = 1.22 → 1.25 per 1,000.

| Learner | Dev Qini [CI] | Test Qini [CI] | Dev / 1,000 at 10% | Test at 10% | Dev at 20% | Test at 20% | Test at 30% |
|---|---|---|---|---|---|---|---|
| S-learner | 0.2020 [0.1402, 0.2675] | 0.2175 [0.1568, 0.2824] | 8.7 [5.8, 11.5] | 9.5 [6.7, 12.2] | 5.4 [3.8, 6.6] | 5.6 [4.3, 6.9] | 4.0 [3.1, 4.9] |
| response model | 0.1974 [0.1341, 0.2549] | 0.2148 [0.1555, 0.2746] | 8.8 [5.9, 11.8] | 9.4 [6.8, 12.2] | 5.2 [3.7, 6.6] | 5.5 [4.2, 6.7] | 3.8 [3.0, 4.6] |
| R-learner | 0.1666 [0.1096, 0.2241] | 0.2000 [0.1389, 0.2668] | 7.7 [5.1, 10.4] | 8.8 [6.0, 11.8] | 4.7 [3.3, 6.0] | 5.2 [3.7, 6.6] | 3.8 [2.8, 4.7] |

### Paired differences — `conversion`

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

### Budget curves — incremental visits per 1,000 contacts, raw / IPW (dev → test)

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

### Profit under the stated economics (dev → test)

Value per visit 2.0, contact cost 0.1 (assumptions); marginal break-even 50 per 1,000 contacts.

| Model | Best budget raw (IPW), dev → test | Max profit per 1,000 customers, dev → test | Break-even budget raw (IPW), dev → test |
|---|---|---|---|
| random | 1% (1%) → 1% (1%) | -1.0 → -0.9 | — (—) → — (—) |
| response model | 5% (5%) → 6% (5%) | 2.2 → 0.7 | 11% (11%) → 9% (8%) |
| S-learner | 8% (4%) → 6% (4%) | 4.8 → 3.6 | 14% (12%) → 14% (11%) |
| R-learner | 5% (4%) → 5% (5%) | 4.7 → 3.8 | 14% (12%) → 13% (11%) |

### Segments with the S-learner (dev → test; δ and p0_high fixed at the dev values)

| Segment | Dev share | Test share | Dev observed uplift [CI] | Test observed uplift [CI] |
|---|---|---|---|---|
| persuadable | 25.3% | 25.4% | +3.46 pp [3.04, 3.88] | +3.23 pp [2.80, 3.65] |
| sure_thing | 0.9% | 0.9% | +0.70 pp [-2.40, 3.81] | +2.05 pp [-1.11, 5.20] |
| lost_cause | 73.6% | 73.6% | +0.07 pp [0.03, 0.11] | +0.03 pp [-0.00, 0.07] |
| sleeping_dog | 0.2% | 0.2% | +0.08 pp [-6.79, 6.95] | +10.12 pp [3.35, 16.88] |
| τ̂ < 0 (any size) | 6.5% | 6.6% | -0.02 pp [-0.44, 0.39] | +0.31 pp [-0.10, 0.72] |

### Placebo (treatment shuffled in train, deployed settings)

| Learner | Scored on | Placebo Qini [CI] | Contains 0 |
|---|---|---|---|
| S-learner | dev | -0.0022 [-0.0136, +0.0095] | yes |
| R-learner | dev | +0.0008 [-0.0127, +0.0126] | yes |
| S-learner | test | -0.0155 [-0.0287, -0.0024] | NO |

