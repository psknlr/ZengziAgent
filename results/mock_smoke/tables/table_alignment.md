**Alignment-aware span metrics (tau = 0.5). Without the Text Alignment Tool only verbatim matches are accepted; rejected spans count as false positives, so alignment effects enter span precision/recall/F1.**

| Dataset | Model | Configuration | Exact span match | Token IoU | Recoverable span rate | Rejected span rate | Fuzzy-recovered rate | Span precision | Span recall | Span F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| SubstanReview-derived | mock | w/o deterministic preprocessing | 0.1405 | 0.4015 | 1.0000 | 0.0000 | 0.1327 | 0.2526 | 0.2357 | 0.2438 |
| SubstanReview-derived | mock | w/o Text Alignment | 0.1500 | 0.3978 | 0.9696 | 0.0304 | 0.0000 | 0.2456 | 0.2310 | 0.2380 |
| SubstanReview-derived | mock | Direct LLM + Fixed Instruction Baseline | 0.0929 | 0.3191 | 0.7749 | 0.1969 | 0.0000 | 0.1995 | 0.1857 | 0.1924 |
| SubstanReview-derived | mock | Full ZengziAgent | 0.1333 | 0.3991 | 1.0000 | 0.0000 | 0.1406 | 0.2526 | 0.2310 | 0.2413 |
| SubstanReview-derived | mock:noisy | w/o deterministic preprocessing | 0.1119 | 0.3793 | 0.9948 | 0.0052 | 0.1680 | 0.2415 | 0.2190 | 0.2297 |
| SubstanReview-derived | mock:noisy | w/o Text Alignment | 0.1333 | 0.3663 | 0.9056 | 0.0944 | 0.0000 | 0.2270 | 0.2119 | 0.2192 |
| SubstanReview-derived | mock:noisy | Direct LLM + Fixed Instruction Baseline | 0.0429 | 0.1715 | 0.5182 | 0.4398 | 0.0000 | 0.1317 | 0.1119 | 0.1210 |
| SubstanReview-derived | mock:noisy | Full ZengziAgent | 0.1119 | 0.3816 | 0.9973 | 0.0027 | 0.1875 | 0.2446 | 0.2143 | 0.2284 |
