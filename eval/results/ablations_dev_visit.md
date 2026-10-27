# Ablations on dev — outcome `visit`

Rule, fixed before the numbers: keep the variant whose Qini CI lower bound is higher; ties → the simpler setting.

| learner    | variant     |   qini |   qini_ci_low |   qini_ci_high |   per_1000_at_20pct | per_1000_at_20pct_ci   |
|:-----------|:------------|-------:|--------------:|---------------:|--------------------:|:-----------------------|
| r_learner  | base        | 0.0938 |        0.0819 |         0.106  |               40    | [34.4, 45.0]           |
| r_learner  | propensity  | 0.0923 |        0.0807 |         0.1046 |               39.6  | [33.8, 44.5]           |
| r_learner  | categorical | 0.0887 |        0.0767 |         0.1017 |               39.63 | [34.1, 44.5]           |
| dr_learner | base        | 0.0927 |        0.0817 |         0.1055 |               40.39 | [35.2, 45.4]           |
| dr_learner | propensity  | 0.0928 |        0.081  |         0.105  |               40.4  | [34.9, 45.6]           |
| dr_learner | categorical | 0.0887 |        0.0778 |         0.1021 |               39.35 | [33.7, 44.4]           |

- **r_learner**: keep `base` (CI lower bounds: base 0.0819, propensity 0.0807, categorical 0.0767)
- **dr_learner**: keep `base` (CI lower bounds: base 0.0817, propensity 0.0810, categorical 0.0778)
