# Paired differences on test — outcome `conversion`, 800,004 rows, 200 resamples

Same stratified resamples for both models of each pair; a CI that excludes 0 is a paired-bootstrap difference at the 95% level.

| pair               | statistic         |   difference |   ci_low |   ci_high | excludes_zero   |
|:-------------------|:------------------|-------------:|---------:|----------:|:----------------|
| s_learner-response | qini              |       0.0026 |  -0.0059 |    0.0157 | False           |
| s_learner-response | per_1000_at_10pct |       0.1045 |  -0.3719 |    0.5928 | False           |
| s_learner-response | per_1000_at_20pct |       0.0836 |  -0.114  |    0.3575 | False           |
| s_learner-response | per_1000_at_30pct |       0.167  |   0.0339 |    0.3092 | True            |
| r_learner-response | qini              |      -0.0148 |  -0.042  |    0.0096 | False           |
| r_learner-response | per_1000_at_10pct |      -0.5977 |  -1.4288 |    0.1737 | False           |
| r_learner-response | per_1000_at_20pct |      -0.3399 |  -0.7814 |    0.0443 | False           |
| r_learner-response | per_1000_at_30pct |      -0.0528 |  -0.3511 |    0.2307 | False           |
