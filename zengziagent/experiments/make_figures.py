"""Figures generated from the master tables (matplotlib; no numbers typed by hand).

* ``fig_ablation_forest``      ΔF1 (ablated − full) with 95% bootstrap CIs, one panel per dataset
* ``fig_per_label_accuracy``   per-label unit accuracy by backend (Figure 3)
* ``fig_multirun_agreement``   mean inter-run agreement with SD error bars (replaces the heatmap grid)
* ``fig_reflection_delta``     R2 − R1 per-label accuracy differences (replaces the composite diagnostic)
* ``fig_tau_sensitivity``      F1 vs tau
All panels use >= 9 pt fonts at 89/183 mm widths and do not rely on colour alone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..schema import LABELS
from ..utils import LOG, setup_logging

MM = 1 / 25.4
HATCHES = ["", "//", "..", "xx", "\\\\", "++", "oo", "--"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 150, "savefig.dpi": 600})
    return plt


def fig_ablation_forest(tables: Path, out: Path) -> None:
    plt = _plt()
    p = tables / "ablation_main.csv"
    if not p.exists() or p.stat().st_size == 0:
        return
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return
    if df.empty:
        return
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(183 * MM, 80 * MM), sharey=True, squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = df[df["dataset"] == ds]
        backends = sorted(sub["backend"].unique())
        abls = [a for a in ["B0", "A1", "A2", "A3", "A4", "A5", "A6"] if a in set(sub["ablation"])]
        for bi, b in enumerate(backends):
            s = sub[sub["backend"] == b].set_index("ablation").reindex(abls)
            y = [i + (bi - (len(backends) - 1) / 2) * 0.22 for i in range(len(abls))]
            x = -s["delta_f1"].to_numpy()  # ablated - full
            lo = -s["f1_ci_high"].to_numpy()
            hi = -s["f1_ci_low"].to_numpy()
            ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="osD^v<>"[bi % 7], ms=4, capsize=2, lw=1, label=b)
        ax.axvline(0, color="0.3", lw=0.8, ls="--")
        ax.set_yticks(range(len(abls)))
        ax.set_yticklabels([f"{a}" for a in abls])
        ax.set_title(ds)
        ax.set_xlabel("ΔF1 (ablated − full) with 95% CI")
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    axes[0][0].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out / "fig_ablation_forest.png")
    fig.savefig(out / "fig_ablation_forest.pdf")
    plt.close(fig)


def fig_per_label_accuracy(tables: Path, out: Path) -> None:
    plt = _plt()
    p = tables / "table_per_label.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    if df.empty:
        return
    datasets = sorted(df["Dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(183 * MM, 70 * MM), sharey=True, squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = df[df["Dataset"] == ds]
        backends = sorted(sub["Model"].unique())
        w = 0.8 / max(1, len(backends))
        for bi, b in enumerate(backends):
            s = sub[sub["Model"] == b].set_index("Label").reindex(list(LABELS))
            xs = [i + (bi - (len(backends) - 1) / 2) * w for i in range(len(LABELS))]
            ax.bar(xs, s["Unit accuracy"].to_numpy(), width=w, label=b, hatch=HATCHES[bi % len(HATCHES)], edgecolor="black", lw=0.4)
        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=20)
        ax.set_ylim(0, 1)
        ax.set_title(ds)
        ax.set_ylabel("Unit-level accuracy")
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_per_label_accuracy.png")
    fig.savefig(out / "fig_per_label_accuracy.pdf")
    plt.close(fig)


def fig_multirun_agreement(tables: Path, out: Path) -> None:
    plt = _plt()
    p = tables / "multirun_stability.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df = df[df["config_id"] == "F"]
    if df.empty or df["n_runs"].max() < 2:
        return
    fig, ax = plt.subplots(figsize=(89 * MM, 60 * MM))
    labels = [f"{r.backend}\n{r.dataset}" for r in df.itertuples()]
    ax.bar(range(len(df)), df["inter_run_agreement"], yerr=df["f1_sd"], capsize=3, edgecolor="black", lw=0.4, color="0.75")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Inter-run label agreement")
    fig.tight_layout()
    fig.savefig(out / "fig_multirun_agreement.png")
    fig.savefig(out / "fig_multirun_agreement.pdf")
    plt.close(fig)


def fig_reflection_delta(tables: Path, out: Path) -> None:
    plt = _plt()
    p = tables / "reflection_r1_r2.csv"
    if not p.exists() or p.stat().st_size == 0:
        return
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(89 * MM, 60 * MM))
    w = 0.8 / max(1, len(df))
    for i, r in enumerate(df.itertuples()):
        deltas = [getattr(r, f"{lab}_delta") for lab in LABELS]
        xs = [k + (i - (len(df) - 1) / 2) * w for k in range(len(LABELS))]
        ax.bar(xs, deltas, width=w, label=f"{r.backend} ({r.dataset})", hatch=HATCHES[i % len(HATCHES)], edgecolor="black", lw=0.4)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=20)
    ax.set_ylabel("Δ unit accuracy (R2 − R1)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_reflection_delta.png")
    fig.savefig(out / "fig_reflection_delta.pdf")
    plt.close(fig)


def fig_tau_sensitivity(tables: Path, out: Path) -> None:
    plt = _plt()
    p = tables / "table_tau_sensitivity.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(89 * MM, 60 * MM))
    for i, ((ds, m), g) in enumerate(df.groupby(["Dataset", "Model"])):
        g = g.sort_values("tau")
        ax.plot(g["tau"], g["F1"], marker="osD^v<>"[i % 7], ms=4, lw=1, label=f"{m} ({ds})")
    ax.set_xlabel("Token-IoU threshold τ")
    ax.set_ylabel("Span F1")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_tau_sensitivity.png")
    fig.savefig(out / "fig_tau_sensitivity.pdf")
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args(argv)
    setup_logging()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = Path(args.tables)
    for fn in (fig_ablation_forest, fig_per_label_accuracy, fig_multirun_agreement, fig_reflection_delta, fig_tau_sensitivity):
        try:
            fn(tables, out)
        except Exception as exc:  # pragma: no cover
            LOG.error("%s failed: %s", fn.__name__, exc)
    LOG.info("figures written to %s: %s", out, sorted(p.name for p in out.glob('*.png')))


if __name__ == "__main__":  # pragma: no cover
    main()
