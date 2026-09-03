"""Component-contribution matrix: datasets x backends x {F, B0, A1..A6} x runs.

Example::

    python -m zengziagent.experiments.run_ablation --backends openrouter:claude openrouter:gpt4o openrouter:gemini --runs 3
    python -m zengziagent.experiments.run_ablation --backends mock --datasets substanreview --runs 2 --limit 30
"""
from __future__ import annotations

import argparse
import sys

from ..llm.base import LLMRequestError
from ..utils import LOG, setup_logging
from .common import ExperimentConfig
from .configs import ORDER, get_config
from .run_annotation import build_backend, run_experiment


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="*", default=None, help="default: all datasets in experiment.yaml")
    ap.add_argument("--configs", nargs="*", default=ORDER)
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--cache", default="results/cache/llm_cache.sqlite")
    ap.add_argument("--experiment-config", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    setup_logging()
    exp = ExperimentConfig.load(args.experiment_config)
    datasets = args.datasets or exp.dataset_names()
    runs = args.runs or exp.runs
    for spec in args.backends:
        backend = build_backend(spec, args.cache)
        for ds in datasets:
            for cid in args.configs:
                cfg = get_config(cid, max_refinement_retries=exp.max_retries, alignment_similarity_threshold=exp.alignment_threshold)
                for k in range(runs):
                    try:
                        run_experiment(exp, ds, spec, cfg, k, out_root=args.out, limit=args.limit, workers=args.workers, force=args.force, backend=backend, cache_path=args.cache)
                    except FileNotFoundError as exc:
                        LOG.error("%s", exc)
                        break
                    except LLMRequestError as exc:
                        sys.exit(f"non-retryable provider error for {spec}: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
