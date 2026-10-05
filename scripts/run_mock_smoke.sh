#!/usr/bin/env bash
# End-to-end smoke test with the deterministic mock backend (no API keys, ~3 minutes).
# Outputs carry NO scientific meaning; they only validate the pipeline and the table machinery.
set -euo pipefail
OUT=results/mock_smoke
python scripts/prepare_data.py --substanreview
python -m zengziagent.experiments.run_ablation --backends mock mock:noisy --datasets substanreview --runs 2 --limit 30 --out $OUT/raw --cache $OUT/cache.sqlite
python -m zengziagent.experiments.run_reflection --backend mock --datasets substanreview --configs F --runs 2 --limit 30 --out $OUT/raw --cache $OUT/cache.sqlite
python -m zengziagent.experiments.evaluate --raw $OUT/raw --out $OUT/master
python -m zengziagent.experiments.analyze_ablation --master $OUT/master --out $OUT/tables --resamples 2000
python -m zengziagent.experiments.make_tables --master $OUT/master --out $OUT/tables
python -m zengziagent.experiments.make_figures --tables $OUT/tables --out $OUT/figures
python -m zengziagent.evaluation.audit $OUT/master/master_results.csv | tee $OUT/tables/audit_report.txt
