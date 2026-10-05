**Supplementary: complete metrics for every configuration (counts pooled over runs).**

| Dataset | Model | ID | Configuration | N units | Accuracy | Precision | Recall | F1 | Exact span match | Token IoU | Rejected span rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SubstanReview-derived | mock | A1 | w/o Planner | 420 | 0.3262 | 0.2526 | 0.2310 | 0.2413 | 0.1333 | 0.3991 | 0.0000 |
| SubstanReview-derived | mock | A2 | w/o Task-Adaptive Prompting | 420 | 0.3262 | 0.2526 | 0.2310 | 0.2413 | 0.1333 | 0.3991 | 0.0000 |
| SubstanReview-derived | mock | A3 | w/o deterministic preprocessing | 420 | 0.3310 | 0.2526 | 0.2357 | 0.2438 | 0.1405 | 0.4015 | 0.0000 |
| SubstanReview-derived | mock | A4 | w/o Recorder validation/refinement | 420 | 0.3095 | 0.2486 | 0.2190 | 0.2329 | 0.1214 | 0.3670 | 0.0108 |
| SubstanReview-derived | mock | A5 | w/o Text Alignment | 420 | 0.3190 | 0.2456 | 0.2310 | 0.2380 | 0.1500 | 0.3978 | 0.0304 |
| SubstanReview-derived | mock | A6 | validation only (w/o refinement retry) | 420 | 0.3095 | 0.2606 | 0.2190 | 0.2380 | 0.1214 | 0.3670 | 0.0113 |
| SubstanReview-derived | mock | B0 | Direct LLM + Fixed Instruction Baseline | 420 | 0.2643 | 0.1995 | 0.1857 | 0.1924 | 0.0929 | 0.3191 | 0.1969 |
| SubstanReview-derived | mock | F | Full ZengziAgent | 420 | 0.3262 | 0.2526 | 0.2310 | 0.2413 | 0.1333 | 0.3991 | 0.0000 |
| SubstanReview-derived | mock:noisy | A1 | w/o Planner | 420 | 0.3119 | 0.2446 | 0.2143 | 0.2284 | 0.1119 | 0.3816 | 0.0027 |
| SubstanReview-derived | mock:noisy | A2 | w/o Task-Adaptive Prompting | 420 | 0.3119 | 0.2446 | 0.2143 | 0.2284 | 0.1119 | 0.3816 | 0.0027 |
| SubstanReview-derived | mock:noisy | A3 | w/o deterministic preprocessing | 420 | 0.3167 | 0.2415 | 0.2190 | 0.2297 | 0.1119 | 0.3793 | 0.0052 |
| SubstanReview-derived | mock:noisy | A4 | w/o Recorder validation/refinement | 420 | 0.2690 | 0.2362 | 0.1833 | 0.2064 | 0.0905 | 0.3006 | 0.0245 |
| SubstanReview-derived | mock:noisy | A5 | w/o Text Alignment | 420 | 0.2881 | 0.2270 | 0.2119 | 0.2192 | 0.1333 | 0.3663 | 0.0944 |
| SubstanReview-derived | mock:noisy | A6 | validation only (w/o refinement retry) | 420 | 0.2690 | 0.2628 | 0.1833 | 0.2160 | 0.0905 | 0.3006 | 0.0273 |
| SubstanReview-derived | mock:noisy | B0 | Direct LLM + Fixed Instruction Baseline | 420 | 0.1524 | 0.1317 | 0.1119 | 0.1210 | 0.0429 | 0.1715 | 0.4398 |
| SubstanReview-derived | mock:noisy | F | Full ZengziAgent | 420 | 0.3119 | 0.2446 | 0.2143 | 0.2284 | 0.1119 | 0.3816 | 0.0027 |
