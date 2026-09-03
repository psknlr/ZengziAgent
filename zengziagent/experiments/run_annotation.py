"""Run the annotation pipeline for dataset x backend x configuration x run.

Examples::

    python -m zengziagent.experiments.run_annotation --dataset substanreview --backend mock --configs F --runs 1 --limit 10
    python -m zengziagent.experiments.run_annotation --dataset all --backend openrouter:claude --configs F --runs 3
    python -m zengziagent.experiments.run_annotation --dataset elife --backend poe:gpt4o --configs F A2 A5 --runs 3 --workers 4

Outputs (per run directory ``results/raw/<dataset>/<backend>/<config>/run<k>/``):
``predictions.jsonl`` (one ReviewResult per review, including raw LLM outputs),
``manifest.json`` (model ids, sampling, dates, hashes), ``prompt_bundle.json`` and
``task_specification.json``.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from .. import __version__
from ..actor import PromptSynthesizer, build_prompt_bundle
from ..data.units import dataset_hash
from ..llm import ResponseCache, resolve_backend
from ..llm.base import LLMBackend, LLMRequestError
from ..pipeline import ZengziAgentPipeline
from ..planner import Planner
from ..schema import PipelineConfig, ReviewRecord
from ..utils import LOG, git_commit, read_json, read_jsonl, repo_root, setup_logging, utc_now, write_json, write_jsonl
from .common import ExperimentConfig, run_dir, strip_gold
from .configs import ORDER, get_config


def build_backend(spec: str, cache_path: Optional[str], **kwargs) -> LLMBackend:
    cache = ResponseCache(cache_path) if cache_path and not spec.startswith("mock") else None
    return resolve_backend(spec, cache=cache, **kwargs)


def run_is_complete(rd: Path, expected_ids: list[str]) -> bool:
    """A run directory is reused only if its manifest lists exactly the expected record ids and no errors."""
    man_path = rd / "manifest.json"
    if not man_path.exists():
        return False
    try:
        man = read_json(man_path)
    except (OSError, ValueError):  # unreadable / corrupt manifest -> re-run
        return False
    return list(man.get("record_ids", [])) == list(expected_ids) and int(man.get("n_errors", 0) or 0) == 0


def run_experiment(
    exp: ExperimentConfig,
    dataset: str,
    backend_spec: str,
    config: PipelineConfig,
    run_index: int,
    out_root: str | Path = "results/raw",
    limit: Optional[int] = None,
    workers: int = 1,
    force: bool = False,
    backend: Optional[LLMBackend] = None,
    cache_path: Optional[str] = "results/cache/llm_cache.sqlite",
    previous_run_dir: Optional[Path] = None,
) -> Path:
    """Annotate one dataset with one backend under one configuration (one run)."""
    rd = run_dir(out_root, dataset, backend_spec, config.config_id, run_index)
    if previous_run_dir is not None:
        rd = previous_run_dir / "round2"
    pred_path = rd / "predictions.jsonl"
    evals, demos = exp.load_dataset(dataset)
    if limit:
        evals = evals[:limit]
    expected_ids = [r.review_id for r in evals]
    if pred_path.exists() and not force:
        if run_is_complete(rd, expected_ids):
            LOG.info("skipping existing run %s (same record ids, no errors; use --force to redo)", rd)
            return rd
        LOG.info("re-running %s: existing run covers a different subset or contains errors", rd)
    backend = backend or build_backend(backend_spec, cache_path)
    capabilities = backend.probe_capabilities() if hasattr(backend, "probe_capabilities") else {"seed_sent": False}
    schema = exp.schema
    started = utc_now()
    # Stage 1 ------------------------------------------------------------------
    if config.use_planner:
        planner = Planner(schema, n_demonstrations=config.n_demonstrations, seed=0)
        spec = planner.build(dataset, strip_gold(evals), demo_pool=demos, exclude_ids={r.review_id for r in evals})
    else:
        spec = Planner.generic(schema, dataset)
    # Stage 2 ------------------------------------------------------------------
    synthesizer = PromptSynthesizer(backend) if config.prompt_synthesis == "llm" else None
    bundle = build_prompt_bundle(spec, backend, config, synthesizer=synthesizer)
    reflection_prompt = (repo_root() / "prompts" / "reflection_prompt.txt").read_text(encoding="utf-8").strip()
    pipeline = ZengziAgentPipeline(backend, config, bundle, allowed_labels=schema.label_names, sampling=exp.sampling, reflection_prompt=reflection_prompt)
    previous_xml: dict[str, str] = {}
    if previous_run_dir is not None:
        for row in read_jsonl(previous_run_dir / "predictions.jsonl"):
            previous_xml[row["review_id"]] = row.get("final_xml", "")
        evals = [r for r in evals if r.review_id in previous_xml]

    def work(rec: ReviewRecord):
        return pipeline.run(rec, run_index=run_index, previous_xml=previous_xml.get(rec.review_id) if previous_run_dir is not None else None)

    t0 = time.time()
    try:
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(work, evals))
        else:
            results = []
            for i, rec in enumerate(evals):
                results.append(work(rec))
                if (i + 1) % 10 == 0:
                    LOG.info("%s/%s/%s run%d: %d/%d reviews", dataset, backend_spec, config.config_id, run_index, i + 1, len(evals))
    except LLMRequestError as exc:
        LOG.error("aborting %s/%s/%s run%d without writing results: %s", dataset, backend_spec, config.config_id, run_index, exc)
        raise
    elapsed = time.time() - t0
    rd.mkdir(parents=True, exist_ok=True)
    write_jsonl(pred_path, (r.to_dict() for r in results))
    write_json(rd / "task_specification.json", spec.to_dict())
    write_json(rd / "prompt_bundle.json", bundle.to_dict())
    models_returned = sorted({a.model_returned for r in results for a in r.attempts if a.model_returned})
    manifest = {
        "package_version": __version__,
        "git_commit": git_commit(repo_root()),
        "dataset": dataset,
        "n_records": len(evals),
        "record_ids": [r.review_id for r in evals],
        "dataset_hash": dataset_hash(evals),
        "backend_spec": backend_spec,
        "provider": backend.provider,
        "model_requested": backend.model,
        "models_returned": models_returned,
        "backend_family": backend.family,
        "backend": backend.describe(),
        "backend_capabilities": capabilities,
        "seed_sent": bool(capabilities.get("seed_sent", False)) and exp.sampling.seed is not None,
        "config": config.to_dict(),
        "effective_retry_budget": config.max_refinement_retries if (config.use_validation and config.use_refinement) else 0,
        "sampling": exp.sampling.to_dict(),
        "request_params_sent": next((a.request_params for r in results for a in r.attempts if a.request_params), None),
        "limit": limit,
        "run_index": run_index,
        "round": 2 if previous_run_dir is not None else 1,
        "started_at": started,
        "finished_at": utc_now(),
        "elapsed_s": round(elapsed, 1),
        "prompt_hash": bundle.prompt_hash,
        "prompt_mode": bundle.mode,
        "spec_hash": spec.hash(),
        "spec_kind": spec.kind,
        "preprocessing": pipeline.preprocessor.name,
        "alignment": pipeline.aligner.describe(),
        "alignment_applicable": True,
        "n_errors": sum(1 for r in results if r.error),
        "n_truncated": sum(1 for r in results for a in r.attempts if a.finish_reason == "length"),
        "n_flagged": sum(1 for r in results if r.flagged),
        "total_retries": sum(r.retries for r in results),
        "n_llm_calls": sum(len(r.attempts) for r in results),
        "n_cached_calls": sum(1 for r in results for a in r.attempts if a.cached),
        "prompt_tokens": sum((a.prompt_tokens or 0) for r in results for a in r.attempts),
        "completion_tokens": sum((a.completion_tokens or 0) for r in results for a in r.attempts),
        "is_mock": backend.provider == "mock",
    }
    write_json(rd / "manifest.json", manifest)
    LOG.info("wrote %s (%d reviews, %.1fs, %d retries, %d flagged)", pred_path, len(results), elapsed, manifest["total_retries"], manifest["n_flagged"])
    return rd


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="substanreview", help="dataset name from configs/experiment.yaml, or 'all'")
    ap.add_argument("--backend", required=True, help="provider:alias | provider:model=<id> | mock[:profile]")
    ap.add_argument("--configs", nargs="*", default=["F"], help=f"configuration ids (default F); all = {ORDER}")
    ap.add_argument("--runs", type=int, default=None, help="number of repeated runs (default from experiment.yaml)")
    ap.add_argument("--run-index", type=int, default=None, help="run a single run index only")
    ap.add_argument("--limit", type=int, default=None, help="only the first N reviews (smoke tests)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--cache", default="results/cache/llm_cache.sqlite")
    ap.add_argument("--experiment-config", default=None)
    ap.add_argument("--prompt-synthesis", choices=["deterministic", "llm"], default=None, help="override the Actor's prompt synthesis mode")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check-model", action="store_true", help="verify the model id against the provider catalog before running")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    setup_logging(args.log_level)
    exp = ExperimentConfig.load(args.experiment_config)
    datasets = exp.dataset_names() if args.dataset == "all" else [args.dataset]
    configs = ORDER if args.configs == ["all"] else args.configs
    runs = args.runs or exp.runs
    run_indices = [args.run_index] if args.run_index is not None else list(range(runs))
    backend = build_backend(args.backend, args.cache)
    if args.check_model and hasattr(backend, "check_model_available"):
        ok = backend.check_model_available()
        LOG.info("model %s available at %s: %s", backend.model, backend.provider, ok)
        if ok is False:
            sys.exit(f"model '{backend.model}' is not listed by provider '{backend.provider}'")
    for ds in datasets:
        for cid in configs:
            overrides = {"max_refinement_retries": exp.max_retries, "alignment_similarity_threshold": exp.alignment_threshold}
            if not get_config(cid).use_refinement:
                overrides["max_refinement_retries"] = 0  # R = 0 for A4, A6, B0 (documented)
            if args.prompt_synthesis:
                overrides["prompt_synthesis"] = args.prompt_synthesis
            cfg = get_config(cid, **overrides)
            for k in run_indices:
                try:
                    run_experiment(exp, ds, args.backend, cfg, k, out_root=args.out, limit=args.limit, workers=args.workers, force=args.force, backend=backend, cache_path=args.cache)
                except FileNotFoundError as exc:
                    LOG.error("%s", exc)
                except LLMRequestError as exc:
                    sys.exit(f"non-retryable provider error: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
