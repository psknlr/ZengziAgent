"""Two-round reflection (R1 -> R2): re-run existing R1 predictions through a reflection pass.

Example::

    python -m zengziagent.experiments.run_reflection --backend openrouter:claude --datasets substanreview elife --configs F
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..llm.base import LLMRequestError
from ..utils import LOG, setup_logging
from .common import ExperimentConfig, run_dir
from .configs import get_config
from .run_annotation import build_backend, run_experiment


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--configs", nargs="*", default=["F"])
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
    backend = build_backend(args.backend, args.cache)
    for ds in datasets:
        for cid in args.configs:
            cfg = get_config(cid, max_refinement_retries=exp.max_retries, alignment_similarity_threshold=exp.alignment_threshold, reflection_rounds=2)
            for k in range(runs):
                r1 = run_dir(args.out, ds, args.backend, cid, k)
                if not (r1 / "predictions.jsonl").exists():
                    LOG.warning("no R1 predictions at %s; run run_annotation first", r1)
                    continue
                try:
                    run_experiment(exp, ds, args.backend, cfg, k, out_root=args.out, limit=args.limit, workers=args.workers, force=args.force, backend=backend, cache_path=args.cache, previous_run_dir=Path(r1))
                except LLMRequestError as exc:
                    sys.exit(f"non-retryable provider error: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
