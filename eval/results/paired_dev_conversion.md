# Paired differences on dev — outcome `conversion`, 799,998 rows, 200 resamples

Same stratified resamples for both models of each pair; a CI that excludes 0 is a paired-bootstrap difference at the 95% level.

| pair               | statistic         |   difference |   ci_low |   ci_high | excludes_zero   |
|:-------------------|:------------------|-------------:|---------:|----------:|:----------------|
| s_learner-response | qini              |       0.0046 |  -0.0045 |    0.0189 | False           |
| s_learner-response | per_1000_at_10pct |      -0.1292 |  -0.6786 |    0.4072 | False           |
| s_learner-response | per_1000_at_20pct |       0.182  |  -0.0328 |    0.4206 | False           |
| s_learner-response | per_1000_at_30pct |       0.0775 |  -0.0316 |    0.2264 | False           |
| r_learner-response | qini              |      -0.0309 |  -0.0527 |   -0.0001 | True            |
| r_learner-response | per_1000_at_10pct |      -1.0809 |  -2.0489 |   -0.0806 | True            |
| r_learner-response | per_1000_at_20pct |      -0.4949 |  -0.8732 |   -0.0313 | True            |
| r_learner-response | per_1000_at_30pct |      -0.1671 |  -0.4183 |    0.1271 | False           |
