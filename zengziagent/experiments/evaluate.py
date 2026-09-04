"""Single evaluation script: raw predictions -> master results tables.

Scans ``results/raw/**/predictions.jsonl`` (plus ``round2`` reflection runs and transformer
baseline outputs written in the same format), recomputes every metric from the raw spans,
and writes:

* ``results/master/master_results.csv``   one row per dataset x backend x config x run x round x tau
                                           (+ ``dataset = pooled`` rows: counts summed over datasets)
* ``results/master/per_label.csv``         per-label unit accuracy and span P/R/F1
* ``results/master/review_counts.csv``     per-review counts (bootstrap input)
* ``results/master/unit_predictions.csv``  per-unit gold/pred labels (McNemar / agreement input)
* ``results/master/confusion.csv``         gold -> predicted label confusion counts
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..evaluation.metrics import aggregate, evaluate_review, pred_spans_from_result
from ..schema import LABELS
from ..tokenization import get_tokenizer
from ..utils import LOG, read_json, read_jsonl, setup_logging
from .common import ExperimentConfig


def discover_runs(raw_root: str | Path) -> list[dict]:
    runs = []
    for pred in sorted(Path(raw_root).rglob("predictions.jsonl")):
        man_path = pred.parent / "manifest.json"
        if not man_path.exists():
            LOG.warning("no manifest next to %s; skipping", pred)
            continue
        man = read_json(man_path)
        runs.append({"pred_path": pred, "manifest": man, "run_dir": pred.parent})
    return runs


def resolve_taus(taus: list[float] | None, exp: ExperimentConfig) -> list[float]:
    """The primary tau is always evaluated (per-review/unit files are written for it)."""
    return sorted(set(taus or exp.tau_grid) | {exp.tau})


def evaluate_all(raw_root: str | Path, out_root: str | Path, exp: ExperimentConfig, taus: list[float] | None = None, only_datasets: list[str] | None = None) -> pd.DataFrame:
    taus = resolve_taus(taus, exp)
    tok = get_tokenizer(exp.raw.get("alignment", {}).get("tokenizer", "auto"))
    policy = exp.rejected_policy
    dataset_cache: dict[str, dict] = {}
    master_rows: list[dict] = []
    label_rows: list[dict] = []
    review_rows: list[dict] = []
    unit_rows: list[dict] = []
    confusion_rows: list[dict] = []
    per_group_evals: dict[tuple, list] = defaultdict(list)  # for pooled rows

    for run in discover_runs(raw_root):
        man = run["manifest"]
        ds = man["dataset"]
        if only_datasets and ds not in only_datasets:
            continue
        if ds not in dataset_cache:
            try:
                evals, _ = exp.load_dataset(ds)
            except FileNotFoundError as exc:
                LOG.error("%s", exc)
                continue
            dataset_cache[ds] = {r.review_id: r for r in evals}
        records = dataset_cache[ds]
        if man.get("n_errors"):
            LOG.warning("%s: %s errored review(s) are scored as empty predictions; re-run this directory", run["run_dir"], man["n_errors"])
        preds = list(read_jsonl(run["pred_path"]))
        key_base = {
            "dataset": ds,
            "backend": man.get("backend_spec", man.get("backend", "?")),
            "provider": man.get("provider"),
            "model_requested": man.get("model_requested"),
            "models_returned": "|".join(man.get("models_returned", []) or []),
            "config_id": man.get("config", {}).get("config_id", man.get("config_id", "?")),
            "config_name": man.get("config", {}).get("name", man.get("config_name", "")),
            "run": int(man.get("run_index", 0)),
            "round": int(man.get("round", 1)),
            "alignment_applicable": bool(man.get("alignment_applicable", True)),
            "is_mock": bool(man.get("is_mock", False)),
            "prompt_hash": man.get("prompt_hash"),
            "spec_hash": man.get("spec_hash"),
            "dataset_hash": man.get("dataset_hash"),
            "n_llm_calls": man.get("n_llm_calls"),
            "total_retries": man.get("total_retries"),
            "n_flagged": man.get("n_flagged"),
            "n_errors": man.get("n_errors"),
        }
        for tau in taus:
            evs = []
            for p in preds:
                rec = records.get(p["review_id"])
                if rec is None:
                    LOG.warning("review %s not in dataset %s; skipping", p["review_id"], ds)
                    continue
                ev = evaluate_review(rec, pred_spans_from_result(p), tau=tau, tokenizer=tok, rejected_policy=policy)
                evs.append(ev)
                if tau == exp.tau:
                    review_rows.append({**{k: key_base[k] for k in ("dataset", "backend", "config_id", "run", "round")}, **ev.to_row()})
                    for u in ev.unit_records:
                        unit_rows.append({**{k: key_base[k] for k in ("backend", "config_id", "run", "round")}, **u})
                    for k, v in ev.confusion.items():
                        g, pl = k.split("->")
                        confusion_rows.append({**{kk: key_base[kk] for kk in ("dataset", "backend", "config_id", "run", "round")}, "gold": g, "pred": pl, "count": v})
            agg = aggregate(evs)
            master_rows.append({**key_base, "tau": tau, **{k: v for k, v in agg.items() if k not in ("per_label", "confusion")}})
            for lab in LABELS:
                pl = agg["per_label"][lab]
                label_rows.append({**{k: key_base[k] for k in ("dataset", "backend", "config_id", "run", "round")}, "tau": tau, "label": lab, **pl})
            per_group_evals[(key_base["backend"], key_base["config_id"], key_base["run"], key_base["round"], tau)].append((ds, evs, key_base))

    # pooled across datasets (sum of counts, never mean of ratios)
    for (backend, cid, run, rnd, tau), items in per_group_evals.items():
        if len({ds for ds, _, _ in items}) < 2:
            continue
        evs = [e for _, es, _ in items for e in es]
        base = dict(items[0][2])
        base.update(
            {
                "dataset": "pooled",
                "dataset_hash": "|".join(sorted({kb["dataset_hash"] or "" for _, _, kb in items})),
                "prompt_hash": "|".join(sorted({kb["prompt_hash"] or "" for _, _, kb in items})),
                "spec_hash": "|".join(sorted({kb["spec_hash"] or "" for _, _, kb in items})),
                "models_returned": "|".join(sorted({kb["models_returned"] or "" for _, _, kb in items})),
            }
        )
        for counter in ("n_llm_calls", "total_retries", "n_flagged", "n_errors"):
            vals = [kb.get(counter) for _, _, kb in items]
            base[counter] = sum(int(v) for v in vals if v is not None) if any(v is not None for v in vals) else None
        agg = aggregate(evs)
        master_rows.append({**base, "tau": tau, **{k: v for k, v in agg.items() if k not in ("per_label", "confusion")}})
        for lab in LABELS:
            label_rows.append({**{k: base[k] for k in ("dataset", "backend", "config_id", "run", "round")}, "tau": tau, "label": lab, **agg["per_label"][lab]})

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    master = pd.DataFrame(master_rows)
    if not master.empty:
        master = master.sort_values(["dataset", "backend", "config_id", "round", "run", "tau"]).reset_index(drop=True)
    master.to_csv(out_root / "master_results.csv", index=False)
    pd.DataFrame(label_rows).to_csv(out_root / "per_label.csv", index=False)
    pd.DataFrame(review_rows).to_csv(out_root / "review_counts.csv", index=False)
    pd.DataFrame(unit_rows).to_csv(out_root / "unit_predictions.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(out_root / "confusion.csv", index=False)
    with open(out_root / "evaluation_settings.json", "w", encoding="utf-8") as fh:
        json.dump({"taus": taus, "primary_tau": exp.tau, "rejected_span_policy": policy, "tokenizer": getattr(tok, "name", "?"), "unit_mode": "gold_span"}, fh, indent=2)
    LOG.info("master table: %d rows -> %s", len(master), out_root / "master_results.csv")
    return master


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out", default="results/master")
    ap.add_argument("--experiment-config", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--taus", nargs="*", type=float, default=None)
    args = ap.parse_args(argv)
    setup_logging()
    exp = ExperimentConfig.load(args.experiment_config)
    evaluate_all(args.raw, args.out, exp, taus=args.taus, only_datasets=args.datasets)


if __name__ == "__main__":  # pragma: no cover
    main()
