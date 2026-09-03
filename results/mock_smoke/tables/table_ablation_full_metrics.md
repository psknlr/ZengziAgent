**Supplementary: complete metrics for every configuration (counts pooled over runs).**

| Dataset | Model | ID | Configuration | N units | Accuracy | Precision | Recall | F1 | Exact span match | Token IoU | Rejected span rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SubstanReview-derived | mock | A1 | w/o Planner | 420 | 0.3238 | 0.2526 | 0.2310 | 0.2413 | 0.0786 | 0.3994 | 0.0000 |
| SubstanReview-derived | mock | A2 | w/o Task-Adaptive Prompting | 420 | 0.3214 | 0.2519 | 0.2310 | 0.2410 | 0.0857 | 0.4022 | 0.0026 |
| SubstanReview-derived | mock | A3 | w/o deterministic preprocessing | 420 | 0.3310 | 0.2526 | 0.2357 | 0.2438 | 0.0857 | 0.4015 | 0.0000 |
| SubstanReview-derived | mock | A4 | w/o Recorder validation/refinement | 420 | 0.2881 | 0.2324 | 0.2048 | 0.2177 | 0.0619 | 0.3475 | 0.0541 |
| SubstanReview-derived | mock | A5 | w/o Text Alignment | 420 | 0.3190 | 0.2456 | 0.2310 | 0.2380 | 0.0952 | 0.3978 | 0.0304 |
| SubstanReview-derived | mock | A6 | validation only (w/o refinement retry) | 420 | 0.2857 | 0.2436 | 0.2048 | 0.2225 | 0.0643 | 0.3423 | 0.0595 |
| SubstanReview-derived | mock | B0 | Direct LLM + Fixed Instruction Baseline | 420 | 0.2643 | 0.1995 | 0.1857 | 0.1924 | 0.0619 | 0.3191 | 0.1969 |
| SubstanReview-derived | mock | F | Full ZengziAgent | 420 | 0.3238 | 0.2519 | 0.2310 | 0.2410 | 0.0833 | 0.4005 | 0.0000 |
| SubstanReview-derived | mock:noisy | A1 | w/o Planner | 420 | 0.3024 | 0.2359 | 0.2095 | 0.2219 | 0.0786 | 0.3759 | 0.0241 |
| SubstanReview-derived | mock:noisy | A2 | w/o Task-Adaptive Prompting | 420 | 0.3071 | 0.2372 | 0.2095 | 0.2225 | 0.0762 | 0.3799 | 0.0162 |
| SubstanReview-derived | mock:noisy | A3 | w/o deterministic preprocessing | 420 | 0.3167 | 0.2415 | 0.2190 | 0.2297 | 0.0667 | 0.3793 | 0.0052 |
| SubstanReview-derived | mock:noisy | A4 | w/o Recorder validation/refinement | 420 | 0.2286 | 0.2055 | 0.1595 | 0.1796 | 0.0452 | 0.2485 | 0.1411 |
| SubstanReview-derived | mock:noisy | A5 | w/o Text Alignment | 420 | 0.2881 | 0.2270 | 0.2119 | 0.2192 | 0.0905 | 0.3663 | 0.0944 |
| SubstanReview-derived | mock:noisy | A6 | validation only (w/o refinement retry) | 420 | 0.2190 | 0.2177 | 0.1524 | 0.1793 | 0.0429 | 0.2391 | 0.1803 |
| SubstanReview-derived | mock:noisy | B0 | Direct LLM + Fixed Instruction Baseline | 420 | 0.1524 | 0.1317 | 0.1119 | 0.1210 | 0.0238 | 0.1715 | 0.4398 |
| SubstanReview-derived | mock:noisy | F | Full ZengziAgent | 420 | 0.3071 | 0.2400 | 0.2143 | 0.2264 | 0.0810 | 0.3803 | 0.0240 |
