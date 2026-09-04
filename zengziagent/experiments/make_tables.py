"""Generate manuscript tables from the master results (never type numbers by hand).

Outputs in ``results/tables`` (CSV + Markdown + LaTeX):

* ``table_performance``        Table 11: dataset-specific + *Pooled across datasets* blocks; alignment
                                metrics ``N/A`` for systems whose output mechanism has no source spans
* ``table_performance_counts`` supplementary: N, N_correct, TP, FP, FN behind every row
* ``table_alignment``          Table 13: exact span match, mean token IoU, recoverable / rejected rates,
                                alignment-aware span P/R/F1 (full vs w/o Text Alignment)
* ``table_ablation_design``    the configuration matrix with the replacement of every removed component
* ``table_tau_sensitivity``    F1 as a function of the IoU threshold tau
* ``table_per_label``          per-label unit accuracy and span F1 (Figure 3 data)
* ``table_ablation_main`` etc. are produced by ``analyze_ablation``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..schema import LABELS
from ..utils import LOG, setup_logging
from .analyze_ablation import _markdown
from .common import ExperimentConfig
from .configs import ablation_table

DATASET_ORDER = {"elife": 0, "substanreview": 1, "pooled": 9}
DATASET_NAMES = {"elife": "eLife", "substanreview": "SubstanReview-derived", "pooled": "Pooled across datasets"}


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "N/A"
    return f"{x:.{nd}f}"


def _write(df: pd.DataFrame, out_dir: Path, name: str, caption: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    with open(out_dir / f"{name}.md", "w", encoding="utf-8") as fh:
        if caption:
            fh.write(f"**{caption}**\n\n")
        fh.write(_markdown(df, name))
    try:
        with open(out_dir / f"{name}.tex", "w", encoding="utf-8") as fh:
            fh.write(df.to_latex(index=False, float_format="%.4f", caption=caption or None, label=f"tab:{name}", escape=True))
    except Exception as exc:  # pragma: no cover
        LOG.warning("LaTeX export failed for %s: %s", name, exc)


def _primary(master: pd.DataFrame, tau: float, round_: int = 1) -> pd.DataFrame:
    m = master[(master["tau"] == tau) & (master["round"] == round_)].copy()
    return m


def _pool_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Sum counts over runs for each dataset x backend x config, then recompute metrics."""
    from ..evaluation.metrics import metrics_from_counts

    rows = []
    for (ds, backend, cid), g in df.groupby(["dataset", "backend", "config_id"]):
        s = g[["n_units", "n_correct", "tp", "fp", "fn", "n_pred", "n_exact", "n_normalized", "n_fuzzy", "n_rejected", "n_illegal", "exact_span_matches", "sum_best_iou"]].sum()
        m = metrics_from_counts(int(s.n_units), int(s.n_correct), int(s.tp), int(s.fp), int(s.fn))
        n_pred = s.n_pred
        rows.append(
            {
                "dataset": ds,
                "backend": backend,
                "config_id": cid,
                "config_name": g["config_name"].iloc[0],
                "model_requested": g["model_requested"].iloc[0],
                "models_returned": g["models_returned"].iloc[0],
                "alignment_applicable": bool(g["alignment_applicable"].iloc[0]),
                "is_mock": bool(g["is_mock"].iloc[0]),
                "n_runs": g["run"].nunique(),
                "n_units": int(s.n_units),
                "n_correct": int(s.n_correct),
                "tp": int(s.tp),
                "fp": int(s.fp),
                "fn": int(s.fn),
                "n_pred": int(n_pred),
                "n_rejected": int(s.n_rejected),
                **m,
                "exact_span_match_rate": s.exact_span_matches / s.n_units if s.n_units else float("nan"),
                "mean_token_iou": s.sum_best_iou / s.n_units if s.n_units else float("nan"),
                "recoverable_span_rate": (s.n_exact + s.n_normalized + s.n_fuzzy) / n_pred if n_pred else float("nan"),
                "rejected_span_rate": s.n_rejected / n_pred if n_pred else float("nan"),
                "fuzzy_span_rate": s.n_fuzzy / n_pred if n_pred else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_o"] = out["dataset"].map(DATASET_ORDER).fillna(5)
    return out.sort_values(["_o", "backend", "config_id"]).drop(columns="_o").reset_index(drop=True)


def make_tables(master_dir: Path, out_dir: Path, exp: ExperimentConfig) -> None:
    master = pd.read_csv(master_dir / "master_results.csv")
    if master.empty:
        LOG.error("empty master table")
        return
    tau = exp.tau
    prim = _pool_runs(_primary(master, tau))
    if prim.empty:
        LOG.error("no rows at tau=%s, round=1 in %s; nothing to tabulate", tau, master_dir)
        return
    # ---- Table 11 (performance) : full configuration + baselines
    perf = prim[prim["config_id"].isin(["F", "TB"])].copy()
    perf["Dataset"] = perf["dataset"].map(DATASET_NAMES).fillna(perf["dataset"])
    perf["Model"] = perf["backend"]
    perf["Accuracy (unit)"] = perf["accuracy"].map(_fmt)
    perf["Precision (span)"] = perf["precision"].map(_fmt)
    perf["Recall (span)"] = perf["recall"].map(_fmt)
    perf["F1 (span)"] = perf["f1"].map(_fmt)
    perf["Exact span match"] = [_fmt(v) if a else "N/A" for v, a in zip(perf["exact_span_match_rate"], perf["alignment_applicable"])]
    perf["Mean token IoU"] = [_fmt(v) if a else "N/A" for v, a in zip(perf["mean_token_iou"], perf["alignment_applicable"])]
    perf["Recoverable span rate"] = [_fmt(v) if a else "N/A" for v, a in zip(perf["recoverable_span_rate"], perf["alignment_applicable"])]
    cols = ["Dataset", "Model", "Accuracy (unit)", "Precision (span)", "Recall (span)", "F1 (span)", "Exact span match", "Mean token IoU", "Recoverable span rate"]
    _write(perf[cols], out_dir, "table_performance", f"Performance under the mapped five-label scheme (tau = {tau}; counts pooled over runs). 'Pooled across datasets' sums TP/FP/FN and unit counts over both corpora (micro), not the arithmetic mean of dataset rows. N/A: the output mechanism does not produce source spans.")
    counts = perf[["Dataset", "Model", "n_runs", "n_units", "n_correct", "tp", "fp", "fn", "n_pred", "n_rejected"]].rename(columns={"n_runs": "Runs", "n_units": "N units", "n_correct": "N correct", "tp": "TP", "fp": "FP", "fn": "FN", "n_pred": "N predicted spans", "n_rejected": "N rejected spans"})
    _write(counts, out_dir, "table_performance_counts", "Underlying counts for every performance row (Accuracy = N correct / N units; P = TP/(TP+FP); R = TP/(TP+FN)).")
    # ---- Table 13 (alignment) : F vs A5 (and B0) per dataset x backend
    al = prim[prim["config_id"].isin(["F", "A5", "B0", "A3"]) & prim["alignment_applicable"]].copy()
    al["Dataset"] = al["dataset"].map(DATASET_NAMES).fillna(al["dataset"])
    al["Configuration"] = al["config_name"]
    tbl = al[["Dataset", "backend", "Configuration", "exact_span_match_rate", "mean_token_iou", "recoverable_span_rate", "rejected_span_rate", "fuzzy_span_rate", "precision", "recall", "f1"]].rename(columns={"backend": "Model", "exact_span_match_rate": "Exact span match", "mean_token_iou": "Token IoU", "recoverable_span_rate": "Recoverable span rate", "rejected_span_rate": "Rejected span rate", "fuzzy_span_rate": "Fuzzy-recovered rate", "precision": "Span precision", "recall": "Span recall", "f1": "Span F1"})
    _write(tbl, out_dir, "table_alignment", f"Alignment-aware span metrics (tau = {tau}). Without the Text Alignment Tool only verbatim matches are accepted; rejected spans count as false positives, so alignment effects enter span precision/recall/F1.")
    # ---- ablation design table
    _write(pd.DataFrame(ablation_table()), out_dir, "table_ablation_design", "Configurations of the component-contribution study and what replaces each removed component.")
    # ---- tau sensitivity (F only)
    rows = []
    for (ds, backend), g in master[(master["config_id"] == "F") & (master["round"] == 1)].groupby(["dataset", "backend"]):
        for t, gg in g.groupby("tau"):
            s = gg[["n_units", "n_correct", "tp", "fp", "fn"]].sum()
            from ..evaluation.metrics import metrics_from_counts

            m = metrics_from_counts(int(s.n_units), int(s.n_correct), int(s.tp), int(s.fp), int(s.fn))
            rows.append({"Dataset": DATASET_NAMES.get(ds, ds), "Model": backend, "tau": t, "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"]})
    _write(pd.DataFrame(rows), out_dir, "table_tau_sensitivity", "Span-level metrics of the full system as a function of the token-IoU threshold tau (tau = 1.0 is exact token match).")
    # ---- per-label (Figure 3 data), F configuration, pooled over runs
    pl = pd.read_csv(master_dir / "per_label.csv")
    pl = pl[(pl["tau"] == tau) & (pl["round"] == 1) & (pl["config_id"] == "F")]
    rows = []
    for (ds, backend, lab), g in pl.groupby(["dataset", "backend", "label"]):
        s = g[["units", "correct", "tp", "fp", "fn"]].sum()
        from ..evaluation.metrics import metrics_from_counts

        m = metrics_from_counts(int(s.units), int(s.correct), int(s.tp), int(s.fp), int(s.fn))
        rows.append({"Dataset": DATASET_NAMES.get(ds, ds), "Model": backend, "Label": lab, "N units": int(s.units), "Unit accuracy": m["accuracy"], "Span precision": m["precision"], "Span recall": m["recall"], "Span F1": m["f1"]})
    per_label = pd.DataFrame(rows)
    if not per_label.empty:
        per_label["_l"] = per_label["Label"].map({l: i for i, l in enumerate(LABELS)})
        per_label = per_label.sort_values(["Dataset", "Model", "_l"]).drop(columns="_l")
    _write(per_label, out_dir, "table_per_label", "Per-label unit-level detection accuracy and span-level metrics (full configuration).")
    # ---- full ablation P/R/F1 (supplementary)
    supp = prim.copy()
    supp["Dataset"] = supp["dataset"].map(DATASET_NAMES).fillna(supp["dataset"])
    for col in ("exact_span_match_rate", "mean_token_iou", "rejected_span_rate"):
        supp[col] = [_fmt(v) if a else "N/A" for v, a in zip(supp[col], supp["alignment_applicable"])]
    supp = supp[["Dataset", "backend", "config_id", "config_name", "n_units", "accuracy", "precision", "recall", "f1", "exact_span_match_rate", "mean_token_iou", "rejected_span_rate"]].rename(columns={"backend": "Model", "config_id": "ID", "config_name": "Configuration", "n_units": "N units", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "exact_span_match_rate": "Exact span match", "mean_token_iou": "Token IoU", "rejected_span_rate": "Rejected span rate"})
    _write(supp, out_dir, "table_ablation_full_metrics", "Supplementary: complete metrics for every configuration (counts pooled over runs).")
    LOG.info("tables written to %s", out_dir)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", default="results/master")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--experiment-config", default=None)
    args = ap.parse_args(argv)
    setup_logging()
    make_tables(Path(args.master), Path(args.out), ExperimentConfig.load(args.experiment_config))


if __name__ == "__main__":  # pragma: no cover
    main()
