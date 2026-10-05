**Performance under the mapped five-label scheme (tau = 0.5; counts pooled over runs). 'Pooled across datasets' sums TP/FP/FN and unit counts over both corpora (micro), not the arithmetic mean of dataset rows. N/A: the output mechanism does not produce source spans.**

| Dataset | Model | Accuracy (unit) | Precision (span) | Recall (span) | F1 (span) | Exact span match | Mean token IoU | Recoverable span rate |
|---|---|---|---|---|---|---|---|---|
| SubstanReview-derived | mock | 0.3262 | 0.2526 | 0.2310 | 0.2413 | 0.1333 | 0.3991 | 1.0000 |
| SubstanReview-derived | mock:noisy | 0.3119 | 0.2446 | 0.2143 | 0.2284 | 0.1119 | 0.3816 | 0.9973 |
