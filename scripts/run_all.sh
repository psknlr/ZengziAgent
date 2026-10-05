#!/usr/bin/env bash
# Full experimental protocol of the revised manuscript.
#   export OPENROUTER_API_KEY=...   (or POE_API_KEY / MINIMAX_API_KEY)
#   bash scripts/run_all.sh openrouter            # provider prefix for the backend aliases
# Every step is idempotent: finished runs are skipped, LLM responses are cached.
set -euo pipefail
PROVIDER="${1:-openrouter}"
# backends = the paper's three aliases when the provider serves them, else every alias it serves (minimax -> minimax:minimax)
mapfile -t BACKENDS < <(python -c "import sys; from zengziagent.llm import default_backends; print('\n'.join(default_backends(sys.argv[1])))" "$PROVIDER")
if [ ${#BACKENDS[@]} -eq 0 ] || [ -z "${BACKENDS[0]}" ]; then
  echo "no backend alias in configs/backends.yaml serves provider '$PROVIDER'" >&2
  exit 1
fi
echo "backends: ${BACKENDS[*]}"
WORKERS="${WORKERS:-4}"

echo "== 1. data"
python scripts/prepare_data.py --substanreview
# eLife gold file (human annotations) must exist at data/elife/gold/elife_gold.jsonl (see docs/DATA.md)

echo "== 2. main annotation + component ablation matrix (F, B0, A1-A6) x 3 runs"
python -m zengziagent.experiments.run_ablation --backends "${BACKENDS[@]}" --runs 3 --workers "$WORKERS"

echo "== 3. two-round reflection (R1 -> R2) for the full configuration"
for b in "${BACKENDS[@]}"; do
  python -m zengziagent.experiments.run_reflection --backend "$b" --configs F --runs 3 --workers "$WORKERS"
done

echo "== 4. transformer baselines (requires torch + transformers)"
for m in roberta electra albert gpt2; do
  for ds in substanreview elife; do
    python -m zengziagent.baselines.transformers_baseline --dataset "$ds" --model "$m" --seeds 13 42 2024 || echo "baseline $m/$ds skipped"
  done
done

echo "== 5. single-source evaluation -> master tables"
python -m zengziagent.experiments.evaluate

echo "== 6. statistics (paired bootstrap, McNemar, Holm), tables, figures, audit"
python -m zengziagent.experiments.analyze_ablation
python -m zengziagent.experiments.make_tables
python -m zengziagent.experiments.make_figures
python -m zengziagent.evaluation.audit results/master/master_results.csv | tee results/tables/audit_report.txt

echo "== 7. scientometric application (eLife): features + OpenAlex citations + analysis"
if [ -f data/elife/gold/elife_gold.jsonl ]; then
  python -m zengziagent.scientometrics.build_features \
    --predictions "results/raw/elife/$(echo "${BACKENDS[0]}" | tr ':' '-')/F/run0/predictions.jsonl" \
    --records data/elife/gold/elife_gold.jsonl --mailto "${OPENALEX_MAILTO:-}" \
    --out results/scientometrics/article_features.csv
  python -m zengziagent.scientometrics.analysis --features results/scientometrics/article_features.csv --out results/scientometrics
fi
echo "done: see results/tables and results/figures"
