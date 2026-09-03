**Alignment-aware span metrics (tau = 0.5). Without the Text Alignment Tool only verbatim matches are accepted; rejected spans count as false positives, so alignment effects enter span precision/recall/F1.**

| Dataset | Model | Configuration | Exact span match | Token IoU | Recoverable span rate | Rejected span rate | Fuzzy-recovered rate | Span precision | Span recall | Span F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| SubstanReview-derived | mock | w/o deterministic preprocessing | 0.0857 | 0.4015 | 1.0000 | 0.0000 | 0.1327 | 0.2526 | 0.2357 | 0.2438 |
| SubstanReview-derived | mock | w/o Text Alignment | 0.0952 | 0.3978 | 0.9696 | 0.0304 | 0.0000 | 0.2456 | 0.2310 | 0.2380 |
| SubstanReview-derived | mock | Direct LLM + Fixed Instruction Baseline | 0.0619 | 0.3191 | 0.7749 | 0.1969 | 0.0000 | 0.1995 | 0.1857 | 0.1924 |
| SubstanReview-derived | mock | Full ZengziAgent | 0.0833 | 0.4005 | 1.0000 | 0.0000 | 0.1221 | 0.2519 | 0.2310 | 0.2410 |
| SubstanReview-derived | mock:noisy | w/o deterministic preprocessing | 0.0667 | 0.3793 | 0.9948 | 0.0052 | 0.1680 | 0.2415 | 0.2190 | 0.2297 |
| SubstanReview-derived | mock:noisy | w/o Text Alignment | 0.0905 | 0.3663 | 0.9056 | 0.0944 | 0.0000 | 0.2270 | 0.2119 | 0.2192 |
| SubstanReview-derived | mock:noisy | Direct LLM + Fixed Instruction Baseline | 0.0238 | 0.1715 | 0.5182 | 0.4398 | 0.0000 | 0.1317 | 0.1119 | 0.1210 |
| SubstanReview-derived | mock:noisy | Full ZengziAgent | 0.0810 | 0.3803 | 0.9760 | 0.0240 | 0.1227 | 0.2400 | 0.2143 | 0.2264 |
