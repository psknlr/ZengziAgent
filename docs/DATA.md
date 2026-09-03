# Data formats and preparation

## SubstanReview-derived benchmark

`python scripts/prepare_data.py --substanreview` downloads
`annotation_final/{train,test}.jsonl` and the three IAA files from
<https://github.com/YanzhuGuo/SubstanReview> (Apache-2.0) into `data/substanreview/` and writes
`manifest_{train,test}.json` (record ids, SHA-256 of every text, label counts, dataset hash).

Release format: `{"id": int, "review": str, "label": [[start, end, "Eval_pos_1"], ...]}`.

Label mapping (`zengziagent/schema.py::canonical_label`): `Eval_pos_k → Eval_pos`,
`Eval_neg_k → Eval_neg`, `Jus_pos_k → Jus_pos`, `Jus_neg_k → Jus_neg`, `Major_claim → Major_Claim`;
the occurrence suffix `k` (claim–evidence link) is kept as metadata.  The released files contain
550 reviews / 4,160 spans: Eval_pos 1,468, Eval_neg 1,309, Jus_neg 942, Jus_pos 278,
Major_claim 163 – identical to Table 4 of the manuscript.  **Note:** the `Major_claim` label is
present in the public release; the manuscript text calling Major_Claim a "study-specific extension"
should be reconciled with this fact.

Evaluation uses the **test split (110 reviews, 816 units)**; demonstrations are selected from the
train split only (`Planner`, coverage-greedy, deterministic).

## eLife

1. `python scripts/prepare_data.py --elife-sample 40 --years 2016 2017 2018 2019 2020 --mailto you@org`
   samples article ids per year via OpenAlex (ISSN 2050-084X) and stores `sampled_articles.json`;
   or `--elife-ids ids.txt` with your own list (e.g. the 200 ids of the study).
2. Reviewer text is extracted from the highest available `elife-<id>-v<k>.xml`
   (`decision-letter` / `referee-report` sub-articles; `Reviewer #n` sections; editorial boilerplate
   dropped; author responses never included) into `data/elife/reviews.jsonl` with metadata
   (`article_id, doi, year, rounds, n_reports`).  `--granularity article` (default) yields one record
   per article, matching the 200 "reviews" of Table 4; `--granularity reviewer` yields one record per
   reviewer section.
3. Human gold annotations go to `data/elife/gold/elife_gold.jsonl` (same JSONL layout as
   SubstanReview or the canonical layout below) and held-out annotated reports used as
   demonstrations to `data/elife/gold/elife_demos.jsonl`.  Paths are set in `configs/experiment.yaml`.

## Canonical record layout

```json
{"review_id": "elife-30000", "dataset": "elife", "text": "...",
 "gold_spans": [{"start": 120, "end": 178, "label": "Eval_neg", "occurrence": 1}],
 "metadata": {"article_id": "30000", "doi": "10.7554/eLife.30000", "year": 2017, "rounds": [1]}}
```

## Prediction record layout (`predictions.jsonl`)

One JSON object per review with `attempts` (raw LLM outputs, validation reports, latency, token
usage, cached flag), `spans` (text, label, alignment status, processed and original offsets, token
range, similarity, method), `retries`, `flagged`, `final_xml`, prompt/spec hashes.  Baseline
predictions use the same layout with `config_id = "TB"` and `alignment_applicable = false` in the
manifest.
