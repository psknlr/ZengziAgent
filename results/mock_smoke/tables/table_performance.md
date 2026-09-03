**Performance under the mapped five-label scheme (tau = 0.5; counts pooled over runs). 'Pooled across datasets' sums TP/FP/FN and unit counts over both corpora (micro), not the arithmetic mean of dataset rows. N/A: the output mechanism does not produce source spans.**

| Dataset | Model | Accuracy (unit) | Precision (span) | Recall (span) | F1 (span) | Exact span match | Mean token IoU | Recoverable span rate |
|---|---|---|---|---|---|---|---|---|
| SubstanReview-derived | mock | 0.3238 | 0.2519 | 0.2310 | 0.2410 | 0.0833 | 0.4005 | 1.0000 |
| SubstanReview-derived | mock:noisy | 0.3071 | 0.2400 | 0.2143 | 0.2264 | 0.0810 | 0.3803 | 0.9760 |
