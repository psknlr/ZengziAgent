"""Multi-method scientometric analysis (Section 8; exploratory use-case demonstration).

Input: an article-level CSV with the feature columns of :mod:`features` plus ``citations``
and ``year``.  Methods (all parameters as in the manuscript):

* winsorisation of citation counts at the 99th percentile;
* negative binomial regression (NB2, alpha estimated) of citations on the four core features
  + year dummies;
* Spearman correlations with bootstrap 95% CIs (B = 2,000);
* quantile regression at tau in {0.25, 0.50, 0.75, 0.90};
* Mann-Whitney U (top citation decile vs rest) with rank-biserial effect size;
* K-means on the standardised six-feature set (k chosen by silhouette over 2..6),
  Kruskal-Wallis + pairwise Mann-Whitney with Benjamini-Hochberg correction;
* LASSO / Elastic Net (5-fold CV) predicting log(1 + c) from the eight-feature set.

Example::

    python -m zengziagent.scientometrics.analysis --features results/scientometrics/article_features.csv --out results/scientometrics
    python -m zengziagent.scientometrics.analysis --synthetic 200 --out results/scientometrics_synthetic   # code-path test only
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import LOG, setup_logging, write_json
from .features import CORE, EXTENDED, FULL


def winsorize(x: np.ndarray, q: float = 0.99) -> tuple[np.ndarray, float]:
    cap = float(np.quantile(x, q))
    return np.minimum(x, cap), cap


def nb_regression(df: pd.DataFrame, features: list[str], y: str = "citations_w") -> dict:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    formula = f"{y} ~ " + " + ".join(features) + " + C(year)"
    model = smf.negativebinomial(formula, data=df)
    res = model.fit(disp=0, maxiter=500)
    ci = res.conf_int()
    out = {"formula": formula, "alpha": float(res.params.get("alpha", np.nan)), "llf": float(res.llf), "aic": float(res.aic), "n": int(res.nobs), "coefficients": {}}
    for name in res.params.index:
        out["coefficients"][name] = {"coef": float(res.params[name]), "se": float(res.bse[name]), "z": float(res.tvalues[name]), "p": float(res.pvalues[name]), "ci_low": float(ci.loc[name, 0]), "ci_high": float(ci.loc[name, 1]), "irr": float(np.exp(res.params[name]))}
    return out


def spearman_bootstrap(df: pd.DataFrame, features: list[str], y: str = "citations", B: int = 2000, seed: int = 0) -> list[dict]:
    from scipy import stats

    rng = np.random.default_rng(seed)
    out = []
    n = len(df)
    for f in features:
        x = df[f].to_numpy(dtype=float)
        yy = df[y].to_numpy(dtype=float)
        ok = ~np.isnan(x)
        x, yy = x[ok], yy[ok]
        rho, p = stats.spearmanr(x, yy)
        boots = []
        for _ in range(B):
            idx = rng.integers(0, len(x), len(x))
            boots.append(stats.spearmanr(x[idx], yy[idx]).statistic)
        boots = np.array(boots)
        out.append({"feature": f, "rho": float(rho), "p": float(p), "ci_low": float(np.nanpercentile(boots, 2.5)), "ci_high": float(np.nanpercentile(boots, 97.5)), "n": int(len(x)), "B": B})
    return out


def quantile_regression(df: pd.DataFrame, features: list[str], y: str = "citations_w", taus=(0.25, 0.5, 0.75, 0.9)) -> list[dict]:
    import statsmodels.formula.api as smf

    formula = f"{y} ~ " + " + ".join(features) + " + C(year)"
    out = []
    for t in taus:
        res = smf.quantreg(formula, data=df).fit(q=t, max_iter=5000)
        ci = res.conf_int()
        for f in features:
            out.append({"tau": t, "feature": f, "coef": float(res.params[f]), "ci_low": float(ci.loc[f, 0]), "ci_high": float(ci.loc[f, 1]), "p": float(res.pvalues[f])})
    return out


def top_decile_test(df: pd.DataFrame, features: list[str], y: str = "citations", top_frac: float = 0.10) -> list[dict]:
    from scipy import stats

    thr = df[y].quantile(1 - top_frac)
    top = df[df[y] >= thr]
    rest = df[df[y] < thr]
    out = []
    for f in features:
        a, b = top[f].dropna().to_numpy(), rest[f].dropna().to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue
        U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        r_rb = 1 - 2 * U / (len(a) * len(b))
        out.append({"feature": f, "n_top": int(len(a)), "n_rest": int(len(b)), "median_top": float(np.median(a)), "median_rest": float(np.median(b)), "U": float(U), "p": float(p), "rank_biserial": float(r_rb)})
    return out


def kmeans_profiles(df: pd.DataFrame, features: list[str], y: str = "citations", k_range=range(2, 7), seed: int = 0) -> dict:
    from scipy import stats
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    sub = df.dropna(subset=features).copy()
    X = StandardScaler().fit_transform(sub[features].to_numpy(dtype=float))
    best = None
    sil = {}
    for k in k_range:
        if k >= len(sub):
            continue
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
        s = float(silhouette_score(X, km.labels_))
        sil[k] = s
        if best is None or s > best[0]:
            best = (s, k, km)
    if best is None:
        return {}
    _, k, km = best
    sub["cluster"] = km.labels_
    profiles = sub.groupby("cluster")[features].mean().round(4).to_dict(orient="index")
    groups = [g[y].to_numpy() for _, g in sub.groupby("cluster")]
    H, p = stats.kruskal(*groups) if len(groups) > 1 else (float("nan"), float("nan"))
    pairs = []
    for a, b in combinations(sorted(sub["cluster"].unique()), 2):
        ga, gb = sub[sub["cluster"] == a][y], sub[sub["cluster"] == b][y]
        U, pp = stats.mannwhitneyu(ga, gb, alternative="two-sided")
        pairs.append({"a": int(a), "b": int(b), "U": float(U), "p": float(pp)})
    if pairs:
        ps = np.array([x["p"] for x in pairs])
        order = np.argsort(ps)
        m = len(ps)
        adj = np.empty(m)
        prev = 1.0
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            val = min(prev, ps[i] * m / (rank + 1))
            adj[i] = val
            prev = val
        for x, a in zip(pairs, adj):
            x["p_bh"] = float(a)
    return {"k": int(k), "silhouette": sil, "cluster_sizes": sub["cluster"].value_counts().sort_index().to_dict(), "profiles": profiles, "kruskal_H": float(H), "kruskal_p": float(p), "pairwise": pairs, "assignments": sub[["article_id", "cluster"]].to_dict(orient="records") if "article_id" in sub else None}


def penalised_regression(df: pd.DataFrame, features: list[str], y: str = "citations", seed: int = 0) -> dict:
    from sklearn.linear_model import ElasticNetCV, LassoCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    sub = df.dropna(subset=features)
    X = StandardScaler().fit_transform(sub[features].to_numpy(dtype=float))
    target = np.log1p(sub[y].to_numpy(dtype=float))
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    out = {}
    for name, est in (("lasso", LassoCV(cv=cv, random_state=seed, max_iter=50000)), ("elastic_net", ElasticNetCV(cv=cv, random_state=seed, l1_ratio=[0.1, 0.5, 0.9], max_iter=50000))):
        est.fit(X, target)
        pred = est.predict(X)
        ss_res = float(((target - pred) ** 2).sum())
        ss_tot = float(((target - target.mean()) ** 2).sum())
        # cross-validated R^2
        cv_r2 = []
        for tr_idx, te_idx in cv.split(X):
            e2 = type(est)(cv=3, random_state=seed, max_iter=50000) if name == "lasso" else type(est)(cv=3, random_state=seed, l1_ratio=[0.1, 0.5, 0.9], max_iter=50000)
            e2.fit(X[tr_idx], target[tr_idx])
            p2 = e2.predict(X[te_idx])
            sr = float(((target[te_idx] - p2) ** 2).sum())
            st = float(((target[te_idx] - target[tr_idx].mean()) ** 2).sum())
            cv_r2.append(1 - sr / st if st else float("nan"))
        out[name] = {"alpha": float(est.alpha_), "coefficients": dict(zip(features, [float(c) for c in est.coef_])), "n_nonzero": int(np.sum(np.abs(est.coef_) > 1e-8)), "in_sample_r2": 1 - ss_res / ss_tot if ss_tot else float("nan"), "cv_r2_mean": float(np.nanmean(cv_r2)), "cv_r2_folds": [float(x) for x in cv_r2]}
    return out


def run_analysis(df: pd.DataFrame, out_dir: Path, seed: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df = df.dropna(subset=["citations", "year"])
    df["year"] = df["year"].astype(int)
    c = df["citations"].to_numpy(dtype=float)
    df["citations_w"], cap = winsorize(c, 0.99)
    vmr = float(c.var(ddof=1) / c.mean()) if c.mean() else float("nan")
    results: dict = {"n_articles": int(len(df)), "citation_mean": float(c.mean()), "citation_var": float(c.var(ddof=1)), "variance_to_mean_ratio": vmr, "winsor_cap_99": cap, "feature_sets": {"core": CORE, "extended": EXTENDED, "full": FULL}}
    try:
        results["negative_binomial"] = nb_regression(df, CORE)
    except Exception as exc:
        LOG.error("NB regression failed: %s", exc)
        results["negative_binomial"] = {"error": str(exc)}
    results["spearman"] = spearman_bootstrap(df, CORE, seed=seed)
    try:
        results["quantile_regression"] = quantile_regression(df, CORE)
    except Exception as exc:
        LOG.error("quantile regression failed: %s", exc)
        results["quantile_regression"] = {"error": str(exc)}
    results["top_decile_mann_whitney"] = top_decile_test(df, CORE)
    try:
        results["kmeans"] = kmeans_profiles(df, EXTENDED, seed=seed)
    except Exception as exc:
        LOG.error("k-means failed: %s", exc)
        results["kmeans"] = {"error": str(exc)}
    try:
        results["penalised"] = penalised_regression(df, FULL, seed=seed)
    except Exception as exc:
        LOG.error("penalised regression failed: %s", exc)
        results["penalised"] = {"error": str(exc)}
    corr = df[FULL].corr(method="pearson").round(3)
    results["feature_correlations"] = corr.to_dict()
    write_json(out_dir / "scientometric_results.json", results)
    pd.DataFrame(results["spearman"]).to_csv(out_dir / "spearman.csv", index=False)
    if isinstance(results["quantile_regression"], list):
        pd.DataFrame(results["quantile_regression"]).to_csv(out_dir / "quantile_regression.csv", index=False)
    pd.DataFrame(results["top_decile_mann_whitney"]).to_csv(out_dir / "top_decile.csv", index=False)
    if "coefficients" in results.get("negative_binomial", {}):
        pd.DataFrame(results["negative_binomial"]["coefficients"]).T.to_csv(out_dir / "nb_regression.csv")
    try:
        _figure(df, results, out_dir)
    except Exception as exc:  # pragma: no cover
        LOG.warning("figure failed: %s", exc)
    LOG.info("scientometric results written to %s", out_dir)
    return results


def _figure(df: pd.DataFrame, results: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 60 / 25.4))
    qr = results.get("quantile_regression")
    if isinstance(qr, list):
        q = pd.DataFrame(qr)
        for f in ("r_neg_val", "r_substan"):
            s = q[q["feature"] == f]
            axes[0].errorbar(s["tau"], s["coef"], yerr=[s["coef"] - s["ci_low"], s["ci_high"] - s["coef"]], marker="o", capsize=2, label=f)
        axes[0].axhline(0, color="0.4", lw=0.8, ls="--")
        axes[0].set_xlabel("quantile τ")
        axes[0].set_ylabel("coefficient")
        axes[0].legend(frameon=False, fontsize=7)
    thr = df["citations"].quantile(0.9)
    axes[1].boxplot([df[df["citations"] >= thr]["r_neg_val"].dropna(), df[df["citations"] < thr]["r_neg_val"].dropna()])
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(["top 10%", "rest"])
    axes[1].set_ylabel("negative evaluation proportion")
    nb = results.get("negative_binomial", {}).get("coefficients", {})
    names = [f for f in CORE if f in nb]
    if names:
        coefs = [nb[f]["coef"] for f in names]
        lo = [nb[f]["coef"] - nb[f]["ci_low"] for f in names]
        hi = [nb[f]["ci_high"] - nb[f]["coef"] for f in names]
        axes[2].errorbar(coefs, range(len(names)), xerr=[lo, hi], fmt="o", capsize=2)
        axes[2].set_yticks(range(len(names)))
        axes[2].set_yticklabels(names)
        axes[2].axvline(0, color="0.4", lw=0.8, ls="--")
        axes[2].set_xlabel("NB coefficient (95% CI)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_scientometrics.png", dpi=600)
    plt.close(fig)


def synthetic_features(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic data to exercise the code path (NOT for reporting)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        n_ep, n_en = rng.poisson(4), rng.poisson(6)
        n_jp, n_jn = rng.poisson(2), rng.poisson(5)
        n_mc = rng.poisson(1.2)
        from collections import Counter

        from .features import features_from_counts

        f = features_from_counts(Counter({"Eval_pos": n_ep, "Eval_neg": n_en, "Jus_pos": n_jp, "Jus_neg": n_jn, "Major_Claim": n_mc}))
        year = int(rng.choice([2016, 2017, 2018, 2019, 2020]))
        mu = np.exp(2.5 + 0.1 * (year - 2018))
        cites = rng.negative_binomial(2, 2 / (2 + mu))
        rows.append({"article_id": f"syn-{i}", "year": year, "doi": f"10.7554/eLife.{10000 + i}", "citations": int(cites), **f})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=None, help="article-level feature CSV (from build_features)")
    ap.add_argument("--synthetic", type=int, default=0, help="generate N synthetic articles (code-path test only)")
    ap.add_argument("--out", default="results/scientometrics")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    setup_logging()
    if args.synthetic:
        df = synthetic_features(args.synthetic, args.seed)
        LOG.warning("SYNTHETIC data: results are for testing the code path only")
    elif args.features:
        df = pd.read_csv(args.features)
    else:
        raise SystemExit("provide --features or --synthetic")
    run_analysis(df, Path(args.out), seed=args.seed)


if __name__ == "__main__":  # pragma: no cover
    main()
