"""Numerical audit: every reported number must be derivable from the counts.

Checks performed on the master results table:

1. ``|F1 - 2PR/(P+R)| < 1e-6`` and P, R, Acc recomputed from N, N_correct, TP, FP, FN;
2. ``Accuracy == Recall`` diagnosis: reports the counts and explains when the identity is
   structural (unit-classification baselines whose predicted spans coincide with the
   evaluation units: every gold unit is either a TP or an FN, so ``TP = N_correct`` and
   ``TP + FN = N``);
3. pooled-vs-arithmetic-mean report for the "Pooled across datasets" block;
4. optional cross-check of a manuscript table (CSV) against the master table.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .metrics import metrics_from_counts


@dataclass
class Finding:
    level: str  # OK | WARN | ERROR | INFO
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.where}: {self.message}"


def audit_master(df: pd.DataFrame, tol: float = 1e-6) -> list[Finding]:
    findings: list[Finding] = []
    required = {"n_units", "n_correct", "tp", "fp", "fn", "accuracy", "precision", "recall", "f1"}
    missing = required - set(df.columns)
    if missing:
        return [Finding("ERROR", "columns", f"master table lacks {sorted(missing)}")]
    for _, r in df.iterrows():
        where = " / ".join(str(r.get(k, "")) for k in ("dataset", "backend", "config_id", "run") if k in df.columns)
        m = metrics_from_counts(int(r.n_units), int(r.n_correct), int(r.tp), int(r.fp), int(r.fn))
        for k in ("accuracy", "precision", "recall", "f1"):
            val = float(r[k])
            if math.isnan(m[k]) and math.isnan(val):
                continue
            if abs(m[k] - val) > tol:
                findings.append(Finding("ERROR", where, f"{k} reported {val:.6f} but counts give {m[k]:.6f}"))
        p, rc, f1 = float(r.precision), float(r.recall), float(r.f1)
        h = 2 * p * rc / (p + rc) if (p + rc) else 0.0
        if abs(h - f1) > tol:
            findings.append(Finding("ERROR", where, f"F1 {f1:.6f} != 2PR/(P+R) {h:.6f}"))
        if not math.isnan(float(r.accuracy)) and abs(float(r.accuracy) - rc) < 1e-9 and int(r.n_units) > 0:
            structural = int(r.tp) == int(r.n_correct) and int(r.tp) + int(r.fn) == int(r.n_units)
            if structural:
                findings.append(
                    Finding(
                        "INFO",
                        where,
                        "Accuracy equals Recall by construction: TP = N_correct and TP + FN = N (each evaluated unit is either a correctly labelled predicted span or a miss), i.e. a one-label-per-unit system whose predicted spans coincide with the units; Precision additionally counts predictions outside the gold units.",
                    )
                )
            else:
                findings.append(Finding("WARN", where, "Accuracy equals Recall numerically but not structurally; verify the counts."))
    # de-duplicate identical findings repeated across tau rows of the same run
    seen: set[tuple] = set()
    unique: list[Finding] = []
    for f in findings:
        key = (f.level, f.where, f.message)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    findings = unique
    if not [f for f in findings if f.level == "ERROR"]:
        findings.append(Finding("OK", "master", f"{len(df)} rows: all P/R/F1/Accuracy values are consistent with their counts"))
    return findings


def pooled_vs_mean_report(df: pd.DataFrame, group_cols=("backend", "config_id", "run"), dataset_col: str = "dataset") -> pd.DataFrame:
    """For every group, compare pooled micro metrics with the arithmetic mean of the dataset rows."""
    rows = []
    real = df[df[dataset_col] != "pooled"]
    for key, g in real.groupby(list(group_cols)):
        if g[dataset_col].nunique() < 2:
            continue
        n, c, tp, fp, fn = (int(g[k].sum()) for k in ("n_units", "n_correct", "tp", "fp", "fn"))
        pooled = metrics_from_counts(n, c, tp, fp, fn)
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        for k in ("accuracy", "precision", "recall", "f1"):
            row[f"{k}_pooled"] = pooled[k]
            row[f"{k}_mean_of_datasets"] = float(g[k].mean())
            row[f"{k}_diff"] = pooled[k] - float(g[k].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def compare_with_manuscript(master: pd.DataFrame, manuscript: pd.DataFrame, keys=("dataset", "backend", "config_id"), metrics=("accuracy", "precision", "recall", "f1"), tol: float = 5e-5) -> pd.DataFrame:
    """Cross-check numbers typed into the manuscript (CSV export of a table) against the master table."""
    m = master.merge(manuscript, on=list(keys), suffixes=("_master", "_manuscript"))
    rows = []
    for _, r in m.iterrows():
        for k in metrics:
            a, b = r.get(f"{k}_master"), r.get(f"{k}_manuscript")
            if a is None or b is None or (isinstance(b, float) and math.isnan(b)):
                continue
            rows.append({**{key: r[key] for key in keys}, "metric": k, "master": float(a), "manuscript": float(b), "abs_diff": abs(float(a) - float(b)), "ok": abs(float(a) - float(b)) <= tol})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Audit a master results CSV for internal numerical consistency.")
    ap.add_argument("master_csv")
    ap.add_argument("--manuscript-csv", default=None, help="CSV with dataset,backend,config_id,accuracy,precision,recall,f1 typed from the manuscript")
    args = ap.parse_args(argv)
    df = pd.read_csv(args.master_csv)
    findings = audit_master(df)
    n_err = 0
    for f in findings:
        print(f)
        n_err += f.level == "ERROR"
    rep = pooled_vs_mean_report(df)
    if not rep.empty:
        print("\nPooled micro metrics vs arithmetic mean of dataset rows (should differ in general):")
        print(rep.round(4).to_string(index=False))
    if args.manuscript_csv:
        cmp = compare_with_manuscript(df[df.get("run", 0) == df.get("run", 0)], pd.read_csv(args.manuscript_csv))
        print("\nManuscript cross-check:")
        print(cmp.to_string(index=False))
        n_err += int((~cmp["ok"]).sum()) if not cmp.empty else 0
    return 1 if n_err else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
