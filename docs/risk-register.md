| **Risk Description**                                     | **Impact** | **Likelihood** | **Mitigation / Action**                                                                                   |
| -------------------------------------------------------- | ---------- | -------------- | --------------------------------------------------------------------------------------------------------- |
| Noisy or incomplete review data                          | Medium     | Low            | Apply data cleaning, deduplication, and validation checks during preprocessing.                           |
| Imbalanced sentiment classes                             | High       | High           | Use class weighting, oversampling techniques, and monitor per-class performance metrics.                  |
| Lower accuracy on Spanish reviews                        | Medium     | Medium         | Use language detection and apply either a separate Spanish pipeline or lightweight translation.           |
| Model underperforms or overfits                          | High       | Medium         | Use cross-validation, start with simple models, and track validation metrics to detect overfitting early. |
| Weak recommendation quality                              | Medium     | Medium         | Tune similarity metrics; implement a popularity-based fallback for cold-start users.                      |
| Django app slow due to model size or inefficient queries | Medium     | Low            | Cache model objects, optimize database queries, and add DB indexes.                                       |
| Insufficient time/resources for advanced features        | High       | High           | Prioritize core system components; freeze scope early; implement optional features only if time remains.  |
| Fairness issues across languages or product categories   | High       | Medium         | Evaluate fairness metrics; retrain or rebalance data if disparities exceed thresholds.                    |
| Library or version conflicts                             | Medium     | Medium         | Pin dependency versions using `requirements.txt` or lockfiles; test setup in a clean environment.         |
| Database integration or migration errors                 | Medium     | Medium         | Test schema early; validate migrations regularly; maintain DB backups.                                    |
