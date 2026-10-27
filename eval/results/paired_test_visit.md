# Paired differences on test — outcome `visit`, 800,004 rows, 200 resamples

Same stratified resamples for both models of each pair; a CI that excludes 0 is a paired-bootstrap difference at the 95% level.

| pair               | statistic         |   difference |   ci_low |   ci_high | excludes_zero   |
|:-------------------|:------------------|-------------:|---------:|----------:|:----------------|
| s_learner-response | qini              |      -0.0001 |  -0.0053 |    0.0067 | False           |
| s_learner-response | per_1000_at_10pct |      13.9203 |   6.4766 |   21.5531 | True            |
| s_learner-response | per_1000_at_20pct |      -0.364  |  -2.5406 |    2.0712 | False           |
| s_learner-response | per_1000_at_30pct |      -0.9397 |  -2.23   |    0.1539 | False           |
| r_learner-response | qini              |      -0.0009 |  -0.01   |    0.0096 | False           |
| r_learner-response | per_1000_at_10pct |      10.5619 |   4.2633 |   17.4849 | True            |
| r_learner-response | per_1000_at_20pct |      -0.0209 |  -3.0753 |    2.9275 | False           |
| r_learner-response | per_1000_at_30pct |      -0.4875 |  -1.9868 |    1.3857 | False           |
