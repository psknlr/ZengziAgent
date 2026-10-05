"""Shared helpers for the experiment runners: config loading, dataset resolution, paths."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from ..data.substanreview import load_substanreview
from ..data.units import load_records_jsonl
from ..schema import AnnotationSchema, ReviewRecord, SamplingParams
from ..utils import LOG, repo_root


@dataclass
class ExperimentConfig:
    raw: dict
    root: Path

    @classmethod
    def load(cls, path: Optional[str] = None) -> "ExperimentConfig":
        root = repo_root()
        p = Path(path) if path else root / "configs" / "experiment.yaml"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. The experiment scripts read configs/ and prompts/ from the repository checkout: "
                "run them from a clone installed with `pip install -e .` (or pass --experiment-config)."
            )
        with open(p, "r", encoding="utf-8") as fh:
            return cls(raw=yaml.safe_load(fh), root=root)

    @property
    def schema(self) -> AnnotationSchema:
        return AnnotationSchema.from_yaml(str(self.root / self.raw["schema"]))

    @property
    def sampling(self) -> SamplingParams:
        s = self.raw.get("sampling", {})
        return SamplingParams(temperature=float(s.get("temperature", 0.0)), top_p=float(s.get("top_p", 1.0)), max_tokens=int(s.get("max_tokens", 4096)), seed=s.get("seed"))

    @property
    def runs(self) -> int:
        return int(self.raw.get("runs", 3))

    @property
    def tau(self) -> float:
        return float(self.raw.get("evaluation", {}).get("iou_threshold", 0.5))

    @property
    def tau_grid(self) -> list[float]:
        return [float(x) for x in self.raw.get("evaluation", {}).get("iou_sensitivity", [0.3, 0.5, 0.7, 0.9, 1.0])]

    @property
    def rejected_policy(self) -> str:
        return self.raw.get("evaluation", {}).get("rejected_span_policy", "count_as_fp")

    @property
    def alignment_threshold(self) -> float:
        return float(self.raw.get("alignment", {}).get("similarity_threshold", 0.80))

    @property
    def max_retries(self) -> int:
        return int(self.raw.get("refinement", {}).get("max_retries", 2))

    @property
    def stats(self) -> dict:
        return self.raw.get("statistics", {"bootstrap_resamples": 10000, "seed": 12345, "confidence": 0.95})

    def dataset_names(self) -> list[str]:
        return list(self.raw.get("datasets", {}).keys())

    def load_dataset(self, name: str) -> tuple[list[ReviewRecord], list[ReviewRecord]]:
        """Return ``(evaluation_records, demonstration_pool)`` for a dataset name."""
        d = self.raw["datasets"][name]
        if d.get("loader") == "substanreview":
            root = self.root / d.get("root", "data/substanreview")
            evals = load_substanreview(root, d.get("eval_split", "test"))
            demos = load_substanreview(root, d.get("demo_split", "train"))
            return evals, demos
        path = self.root / d["path"]
        if not path.exists():
            raise FileNotFoundError(f"dataset '{name}': {path} not found (human-annotated gold file; see docs/DATA.md)")
        evals = load_records_jsonl(path, dataset=name)
        demos: list[ReviewRecord] = []
        dp = d.get("demo_path")
        if dp and (self.root / dp).exists():
            demos = load_records_jsonl(self.root / dp, dataset=name)
        else:
            LOG.warning("dataset '%s': no demonstration pool found; demonstrations will be empty", name)
        return evals, demos


def backend_slug(spec: str) -> str:
    """``openrouter:claude`` -> ``openrouter-claude``; ``openrouter:model=openai/gpt-4o`` -> ``openrouter-openai_gpt-4o``."""
    s = spec.replace("model=", "").replace(":", "-").replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)


def run_dir(out_root: str | Path, dataset: str, backend: str, config_id: str, run_index: int) -> Path:
    return Path(out_root) / dataset / backend_slug(backend) / config_id / f"run{run_index}"


def strip_gold(records: list[ReviewRecord]) -> list[ReviewRecord]:
    """Copies without gold spans - what the Planner/Analyzer are allowed to see."""
    return [ReviewRecord(r.review_id, r.dataset, r.text, [], dict(r.metadata)) for r in records]
