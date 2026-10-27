# Paired differences on dev — outcome `visit`, 799,998 rows, 200 resamples

Same stratified resamples for both models of each pair; a CI that excludes 0 is a paired-bootstrap difference at the 95% level.

| pair               | statistic         |   difference |   ci_low |   ci_high | excludes_zero   |
|:-------------------|:------------------|-------------:|---------:|----------:|:----------------|
| s_learner-response | qini              |       0.0069 |   0.0007 |    0.0135 | True            |
| s_learner-response | per_1000_at_10pct |      13.0402 |   6.4494 |   19.2302 | True            |
| s_learner-response | per_1000_at_20pct |       2.6046 |   0.0052 |    4.7734 | True            |
| s_learner-response | per_1000_at_30pct |       0.7234 |  -0.309  |    1.9884 | False           |
| r_learner-response | qini              |       0.0067 |  -0.0017 |    0.0169 | False           |
| r_learner-response | per_1000_at_10pct |       9.5103 |   2.8109 |   15.7787 | True            |
| r_learner-response | per_1000_at_20pct |       1.6354 |  -1.1857 |    4.2613 | False           |
| r_learner-response | per_1000_at_30pct |       0.2653 |  -1.3665 |    2.0395 | False           |
