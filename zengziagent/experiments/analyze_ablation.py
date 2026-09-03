"""Component-contribution analysis with paired bootstrap CIs, McNemar tests and Holm correction.

Reads ``results/master/review_counts.csv`` and ``unit_predictions.csv`` and writes to
``results/tables``:

* ``ablation_main.csv|md``      per dataset x backend x ablation: ΔAccuracy, ΔF1, 95% CI, p, Holm-adjusted p, McNemar p
* ``ablation_contrast.csv|md``  ΔF1(eLife) − ΔF1(SubstanReview) per backend x ablation with bootstrap CI
* ``backend_pairwise.csv|md``   pairwise backend differences under the full configuration
* ``multirun_stability.csv|md`` mean ± SD across runs and inter-run agreement per configuration
* ``reflection_r1_r2.csv|md``   R1 -> R2 differences (overall and per label)
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.agreement import inter_run_agreement
from ..evaluation.metrics import metrics_from_counts
from ..evaluation.stats import bootstrap_contrast, holm, mcnemar, paired_bootstrap
from ..schema import LABELS
from ..utils import LOG, setup_logging
from .common import ExperimentConfig
from .configs import ABLATIONS, ORDER

COUNT_COLS = ["n_units", "n_correct", "tp", "fp", "fn"]
#: header rows written even when a table has no data (so downstream readers never see an empty file)
EMPTY_COLUMNS = {
    "ablation_main": ["dataset", "backend", "ablation", "ablation_name", "n_reviews", "n_units", "n_unit_run_pairs", "acc_full", "acc_ablated", "delta_accuracy", "acc_ci_low", "acc_ci_high", "f1_full", "f1_ablated", "delta_f1", "f1_ci_low", "f1_ci_high", "p_bootstrap_f1", "p_bootstrap_acc", "mcnemar_b", "mcnemar_c", "p_mcnemar", "precision_full", "recall_full", "precision_ablated", "recall_ablated", "p_adj_bootstrap_f1", "p_adj_bootstrap_acc", "p_adj_mcnemar"],
    "ablation_contrast": ["backend", "ablation", "ablation_name", "dataset_1", "dataset_2", "delta_f1_dataset_1", "delta_f1_dataset_2", "contrast", "ci_low", "ci_high", "p_value"],
    "backend_pairwise": ["dataset", "backend_a", "backend_b", "f1_a", "f1_b", "delta_f1", "ci_low", "ci_high", "p_bootstrap", "mcnemar_b", "mcnemar_c", "p_mcnemar", "p_adj_bootstrap", "p_adj_mcnemar"],
    "multirun_stability": ["dataset", "backend", "config_id", "n_runs", "accuracy_mean", "accuracy_sd", "f1_mean", "f1_sd", "inter_run_agreement", "inter_run_agreement_sd", "inter_run_fleiss_kappa"],
    "reflection_r1_r2": ["dataset", "backend", "config_id", "f1_r1", "f1_r2", "delta_f1_r2_minus_r1", "f1_ci_low", "f1_ci_high", "p_f1", "acc_r1", "acc_r2", "delta_acc", "acc_ci_low", "acc_ci_high", "p_acc"],
}


def _count_matrix(df: pd.DataFrame, runs_policy: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-review count matrix; ``pooled`` sums the counts over runs, ``run0`` uses the first run."""
    if runs_policy == "run0":
        df = df[df["run"] == df["run"].min()]
    g = df.groupby("review_id")[COUNT_COLS].sum().sort_index()
    return g, g.to_numpy(dtype=float)


def _aligned(a: pd.DataFrame, b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ids = sorted(set(a.index) & set(b.index))
    return a.loc[ids].to_numpy(dtype=float), b.loc[ids].to_numpy(dtype=float)


def _unit_correct(units: pd.DataFrame, runs_policy: str) -> pd.Series:
    """Paired unit-level correctness for McNemar.

    ``run0``: the first run.  ``pooled``: one value per *distinct* unit - correct when the unit
    was labelled correctly in at least half of the runs (majority collapse).  Treating every
    (unit, run) pair as an independent observation would multiply discordant counts by the
    number of runs and make the test anti-conservative.
    """
    if runs_policy == "run0":
        units = units[units["run"] == units["run"].min()]
        return units.set_index("unit_id")["correct"].astype(int).sort_index()
    return (units.groupby("unit_id")["correct"].mean() >= 0.5).astype(int).sort_index()


def analyze(master_dir: Path, out_dir: Path, exp: ExperimentConfig, runs_policy: str = "pooled", round_: int = 1) -> dict[str, pd.DataFrame]:
    st = exp.stats
    B = int(st.get("bootstrap_resamples", 10000))
    seed = int(st.get("seed", 12345))
    conf = float(st.get("confidence", 0.95))
    rc = pd.read_csv(master_dir / "review_counts.csv")
    units = pd.read_csv(master_dir / "unit_predictions.csv")
    rc = rc[rc["round"] == round_]
    units = units[units["round"] == round_]
    out: dict[str, pd.DataFrame] = {}

    # ---------------------------------------------------------------- ablations
    rows = []
    for (ds, backend), g in rc.groupby(["dataset", "backend"]):
        if "F" not in set(g["config_id"]):
            continue
        full_df, _ = _count_matrix(g[g["config_id"] == "F"], runs_policy)
        u_full = _unit_correct(units[(units["dataset"] == ds) & (units["backend"] == backend) & (units["config_id"] == "F")], runs_policy)
        fam = []
        for cid in ORDER:
            if cid == "F" or cid not in set(g["config_id"]):
                continue
            abl_df, _ = _count_matrix(g[g["config_id"] == cid], runs_policy)
            a, b = _aligned(full_df, abl_df)
            if len(a) == 0:
                continue
            f1 = paired_bootstrap(a, b, "f1", n_resamples=B, seed=seed, confidence=conf)
            acc = paired_bootstrap(a, b, "accuracy", n_resamples=B, seed=seed + 1, confidence=conf)
            u_abl = _unit_correct(units[(units["dataset"] == ds) & (units["backend"] == backend) & (units["config_id"] == cid)], runs_policy)
            idx = u_full.index.intersection(u_abl.index)
            mc = mcnemar(u_full.loc[idx].to_numpy(), u_abl.loc[idx].to_numpy()) if len(idx) else {"b": 0, "c": 0, "p_exact": float("nan")}
            mf = metrics_from_counts(*a.sum(0).astype(int))
            ma = metrics_from_counts(*b.sum(0).astype(int))
            fam.append(
                {
                    "dataset": ds,
                    "backend": backend,
                    "ablation": cid,
                    "ablation_name": ABLATIONS[cid].name,
                    "n_reviews": int(len(a)),
                    "n_units": int(len(u_full.index.intersection(u_abl.index))) if len(idx) else int(a[:, 0].sum()),
                    "n_unit_run_pairs": int(a[:, 0].sum()),
                    "acc_full": mf["accuracy"],
                    "acc_ablated": ma["accuracy"],
                    "delta_accuracy": acc["delta"],
                    "acc_ci_low": acc["ci_low"],
                    "acc_ci_high": acc["ci_high"],
                    "f1_full": mf["f1"],
                    "f1_ablated": ma["f1"],
                    "delta_f1": f1["delta"],
                    "f1_ci_low": f1["ci_low"],
                    "f1_ci_high": f1["ci_high"],
                    "p_bootstrap_f1": f1["p_value"],
                    "p_bootstrap_acc": acc["p_value"],
                    "mcnemar_b": mc["b"],
                    "mcnemar_c": mc["c"],
                    "p_mcnemar": mc["p_exact"],
                    "precision_full": mf["precision"],
                    "recall_full": mf["recall"],
                    "precision_ablated": ma["precision"],
                    "recall_ablated": ma["recall"],
                }
            )
        if fam:
            for key in ("p_bootstrap_f1", "p_bootstrap_acc", "p_mcnemar"):
                adj = holm([r[key] if not np.isnan(r[key]) else 1.0 for r in fam])
                for r, pa in zip(fam, adj):
                    r["p_adj_" + key[2:]] = pa
            rows.extend(fam)
    abl = pd.DataFrame(rows)
    out["ablation_main"] = abl

    # ------------------------------------------------------ dataset contrast
    crows = []
    if not abl.empty and abl["dataset"].nunique() >= 2:
        datasets = sorted(abl["dataset"].unique())
        for backend in sorted(abl["backend"].unique()):
            for cid in ORDER:
                if cid == "F":
                    continue
                sub = abl[(abl["backend"] == backend) & (abl["ablation"] == cid)]
                if sub["dataset"].nunique() < 2:
                    continue
                d1, d2 = datasets[:2] if "elife" not in datasets else ("elife", [d for d in datasets if d != "elife"][0])
                mats = {}
                for ds in (d1, d2):
                    g = rc[(rc["dataset"] == ds) & (rc["backend"] == backend)]
                    fdf, _ = _count_matrix(g[g["config_id"] == "F"], runs_policy)
                    adf, _ = _count_matrix(g[g["config_id"] == cid], runs_policy)
                    mats[ds] = _aligned(fdf, adf)
                res = bootstrap_contrast(mats[d1][0], mats[d1][1], mats[d2][0], mats[d2][1], "f1", n_resamples=B, seed=seed + 7, confidence=conf)
                crows.append({"backend": backend, "ablation": cid, "ablation_name": ABLATIONS[cid].name, "dataset_1": d1, "dataset_2": d2, "delta_f1_dataset_1": res["delta_1"], "delta_f1_dataset_2": res["delta_2"], "contrast": res["contrast"], "ci_low": res["ci_low"], "ci_high": res["ci_high"], "p_value": res["p_value"]})
    out["ablation_contrast"] = pd.DataFrame(crows)

    # --------------------------------------------------- backend pairwise (F)
    prow = []
    for ds, g in rc[rc["config_id"] == "F"].groupby("dataset"):
        backends = sorted(g["backend"].unique())
        fam = []
        for b1, b2 in combinations(backends, 2):
            m1, _ = _count_matrix(g[g["backend"] == b1], runs_policy)
            m2, _ = _count_matrix(g[g["backend"] == b2], runs_policy)
            a, b = _aligned(m1, m2)
            if len(a) == 0:
                continue
            f1 = paired_bootstrap(a, b, "f1", n_resamples=B, seed=seed + 3, confidence=conf)
            u1 = _unit_correct(units[(units["dataset"] == ds) & (units["backend"] == b1) & (units["config_id"] == "F")], runs_policy)
            u2 = _unit_correct(units[(units["dataset"] == ds) & (units["backend"] == b2) & (units["config_id"] == "F")], runs_policy)
            idx = u1.index.intersection(u2.index)
            mc = mcnemar(u1.loc[idx].to_numpy(), u2.loc[idx].to_numpy()) if len(idx) else {"p_exact": float("nan"), "b": 0, "c": 0}
            fam.append({"dataset": ds, "backend_a": b1, "backend_b": b2, "f1_a": f1["value_a"], "f1_b": f1["value_b"], "delta_f1": f1["delta"], "ci_low": f1["ci_low"], "ci_high": f1["ci_high"], "p_bootstrap": f1["p_value"], "mcnemar_b": mc["b"], "mcnemar_c": mc["c"], "p_mcnemar": mc["p_exact"]})
        if fam:
            for key in ("p_bootstrap", "p_mcnemar"):
                adj = holm([r[key] if not np.isnan(r[key]) else 1.0 for r in fam])
                for r, pa in zip(fam, adj):
                    r["p_adj_" + key[2:]] = pa
            prow.extend(fam)
    out["backend_pairwise"] = pd.DataFrame(prow)

    # ------------------------------------------------------ multi-run stability
    srows = []
    for (ds, backend, cid), g in rc.groupby(["dataset", "backend", "config_id"]):
        runs = sorted(g["run"].unique())
        per_run = []
        for k in runs:
            s = g[g["run"] == k][COUNT_COLS].sum()
            per_run.append(metrics_from_counts(*[int(x) for x in s]))
        accs = [m["accuracy"] for m in per_run]
        f1s = [m["f1"] for m in per_run]
        u = units[(units["dataset"] == ds) & (units["backend"] == backend) & (units["config_id"] == cid)]
        agreement = {"pairwise_agreement": float("nan"), "fleiss_kappa": float("nan")}
        if len(runs) >= 2:
            # fill before pivoting: pivot_table drops rows whose values are NaN in every run
            piv = u.assign(pl=u["pred_label"].fillna("None")).pivot_table(index="unit_id", columns="run", values="pl", aggfunc="first").fillna("None")
            agreement = inter_run_agreement([piv[k].tolist() for k in piv.columns])
        srows.append({"dataset": ds, "backend": backend, "config_id": cid, "n_runs": len(runs), "accuracy_mean": float(np.mean(accs)), "accuracy_sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0, "f1_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0, "inter_run_agreement": agreement["pairwise_agreement"], "inter_run_agreement_sd": agreement.get("pairwise_agreement_sd", float("nan")), "inter_run_fleiss_kappa": agreement["fleiss_kappa"]})
    out["multirun_stability"] = pd.DataFrame(srows)

    # ------------------------------------------------------------- R1 vs R2
    rc_all = pd.read_csv(master_dir / "review_counts.csv")
    units_all = pd.read_csv(master_dir / "unit_predictions.csv")
    rrows = []
    if 2 in set(rc_all["round"]):
        for (ds, backend, cid), g in rc_all.groupby(["dataset", "backend", "config_id"]):
            if not {1, 2} <= set(g["round"]):
                continue
            r1, _ = _count_matrix(g[g["round"] == 1], runs_policy)
            r2, _ = _count_matrix(g[g["round"] == 2], runs_policy)
            a, b = _aligned(r2, r1)
            if len(a) == 0:
                continue
            f1 = paired_bootstrap(a, b, "f1", n_resamples=B, seed=seed + 11, confidence=conf)
            acc = paired_bootstrap(a, b, "accuracy", n_resamples=B, seed=seed + 12, confidence=conf)
            row = {"dataset": ds, "backend": backend, "config_id": cid, "f1_r1": f1["value_b"], "f1_r2": f1["value_a"], "delta_f1_r2_minus_r1": f1["delta"], "f1_ci_low": f1["ci_low"], "f1_ci_high": f1["ci_high"], "p_f1": f1["p_value"], "acc_r1": acc["value_b"], "acc_r2": acc["value_a"], "delta_acc": acc["delta"], "acc_ci_low": acc["ci_low"], "acc_ci_high": acc["ci_high"], "p_acc": acc["p_value"]}
            uu = units_all[(units_all["dataset"] == ds) & (units_all["backend"] == backend) & (units_all["config_id"] == cid)]
            for lab in LABELS:
                for rnd in (1, 2):
                    sub = uu[(uu["round"] == rnd) & (uu["gold_label"] == lab)]
                    row[f"{lab}_acc_r{rnd}"] = float(sub["correct"].mean()) if len(sub) else float("nan")
                row[f"{lab}_delta"] = row[f"{lab}_acc_r2"] - row[f"{lab}_acc_r1"]
            rrows.append(row)
    out["reflection_r1_r2"] = pd.DataFrame(rrows)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        if df.empty and name in EMPTY_COLUMNS:
            df = pd.DataFrame(columns=EMPTY_COLUMNS[name])
            out[name] = df
        df.to_csv(out_dir / f"{name}.csv", index=False)
        with open(out_dir / f"{name}.md", "w", encoding="utf-8") as fh:
            fh.write(_markdown(df, name))
        LOG.info("wrote %s (%d rows)", out_dir / f"{name}.csv", len(df))
    return out


def _markdown(df: pd.DataFrame, name: str) -> str:
    if df.empty:
        return f"_{name}: no data_\n"
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", default="results/master")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--experiment-config", default=None)
    ap.add_argument("--runs-policy", choices=["pooled", "run0"], default="pooled", help="pooled: sum counts over the repeated runs (default); run0: first run only")
    ap.add_argument("--resamples", type=int, default=None, help="override the number of bootstrap resamples")
    args = ap.parse_args(argv)
    setup_logging()
    exp = ExperimentConfig.load(args.experiment_config)
    if args.resamples:
        exp.raw.setdefault("statistics", {})["bootstrap_resamples"] = args.resamples
    analyze(Path(args.master), Path(args.out), exp, runs_policy=args.runs_policy)


if __name__ == "__main__":  # pragma: no cover
    main()
