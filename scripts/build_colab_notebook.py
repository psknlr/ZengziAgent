#!/usr/bin/env python3
"""Generate notebooks/ZengziAgent_Colab.ipynb (kept in sync with the CLI; run after CLI changes)."""
from __future__ import annotations

import json
from pathlib import Path

try:
    import nbformat as nbf
except ImportError:  # plain-JSON fallback (nbformat v4 layout)
    nbf = None


class _NB(dict):
    pass


def _new_notebook():
    if nbf is not None:
        return nbf.v4.new_notebook()
    return _NB(cells=[], metadata={}, nbformat=4, nbformat_minor=5)


def _md(source):
    if nbf is not None:
        return nbf.v4.new_markdown_cell(source)
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code(source):
    if nbf is not None:
        return nbf.v4.new_code_cell(source)
    return {"cell_type": "code", "metadata": {}, "source": source, "outputs": [], "execution_count": None}


nb = _new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"name": "ZengziAgent_Colab.ipynb", "provenance": [], "toc_visible": True},
    "accelerator": "GPU",
}
cells = []
md = lambda s: cells.append(_md(s))
code = lambda s: cells.append(_code(s))

md(r"""# ZengziAgent — fast reproduction of the revised experiments (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/psknlr/ZengziAgent/blob/claude/paper-revision-ablation-study-dgwirx/notebooks/ZengziAgent_Colab.ipynb)

This notebook runs the experiments requested by the reviewers end to end:

1. **Component-contribution study** — configurations F (full), B0 (direct LLM + fixed prompt), A1–A6 (w/o Planner, w/o task-adaptive prompting, w/o preprocessing, w/o validation/refinement, w/o text alignment, validation-only) on the SubstanReview-derived test split (and eLife if you upload the human gold file), with repeated runs.
2. **Two-round reflection** (R1 → R2).
3. **Single-source evaluation** → master CSV → paired-bootstrap CIs, McNemar, Holm-adjusted p-values, per-dataset tables, alignment metrics, τ sensitivity.
4. **Tables and figures** generated from the master CSV and a **numerical audit**.
5. Optional: transformer baselines (GPU) and the scientometric analysis.

**Fast mode** (default) uses a subset of reviews, 2 runs and 2,000 bootstrap resamples so that the whole notebook finishes in roughly 20–40 minutes with one backend. Switch `FAST = False` for the full protocol (110 SubstanReview reviews × 8 configurations × 3 runs per backend, 10,000 resamples).

You need one API key: **OpenRouter** (`OPENROUTER_API_KEY`), **Poe** (`POE_API_KEY`) or **MiniMax** (`MINIMAX_API_KEY`). Store it in Colab *Secrets* (🔑 icon in the left bar) under that name, or paste it when prompted. Set `PROVIDER = "mock"` to exercise the whole pipeline offline (no scientific meaning).""")

md("## 1. Setup — clone the repository and install dependencies")
code(r"""import os, sys, subprocess, pathlib
REPO_URL = "https://github.com/psknlr/ZengziAgent.git"
BRANCH = "claude/paper-revision-ablation-study-dgwirx"   # branch with the revised code
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
WORKDIR = pathlib.Path("/content/ZengziAgent" if IN_COLAB else os.environ.get("ZENGZI_WORKDIR", os.getcwd()))

if not (WORKDIR / "pyproject.toml").exists():
    subprocess.run(["git", "clone", "--branch", BRANCH, "--depth", "1", REPO_URL, str(WORKDIR)], check=True)
os.chdir(WORKDIR)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
print("working directory:", WORKDIR)
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)""")

md("## 2. Parameters\n\nChoose the provider, the backends, and fast vs. full protocol.")
code(r"""# ---- provider and backends -------------------------------------------------
PROVIDER = os.environ.get("ZENGZI_PROVIDER", "openrouter")   # "openrouter" | "poe" | "minimax" | "mock" (offline dry run)
BACKEND_ALIASES = ["claude", "gpt4o", "gemini"]   # aliases defined in configs/backends.yaml
# Exact model ids can be used instead of aliases, e.g. "model=openai/gpt-4o-2024-11-20"

# ---- protocol --------------------------------------------------------------
FAST = True                    # True: subset + 2 runs + 2,000 resamples;  False: full protocol
LIMIT = 30 if FAST else None   # number of test reviews per dataset (None = all 110 SubstanReview reviews)
RUNS = 2 if FAST else 3
CONFIGS = ["F", "B0", "A1", "A2", "A3", "A4", "A5", "A6"]   # component-contribution matrix
RESAMPLES = 2000 if FAST else 10000
WORKERS = 4                    # parallel API calls per run
RUN_REFLECTION = True          # R1 -> R2 for the full configuration
DATASETS = ["substanreview"]   # add "elife" after uploading data/elife/gold/elife_gold.jsonl (see section 3b)

if PROVIDER == "mock":
    BACKENDS = ["mock", "mock:noisy"]
else:
    BACKENDS = [f"{PROVIDER}:{a}" for a in BACKEND_ALIASES]
print("backends:", BACKENDS)
print("fast mode:", FAST, "| reviews per dataset:", LIMIT or "all", "| runs:", RUNS, "| configs:", CONFIGS)""")

md("## 3. API key\n\nRead from Colab Secrets (name = environment variable) or prompt once. Keys are never written to disk.")
code(r"""import getpass
KEY_ENV = {"openrouter": "OPENROUTER_API_KEY", "poe": "POE_API_KEY", "minimax": "MINIMAX_API_KEY", "minimax-cn": "MINIMAX_API_KEY"}
if PROVIDER != "mock":
    env_name = KEY_ENV[PROVIDER]
    key = os.environ.get(env_name)
    if not key:
        try:
            from google.colab import userdata  # type: ignore
            key = userdata.get(env_name)
        except Exception:
            key = None
    if not key:
        key = getpass.getpass(f"Paste your {env_name}: ")
    os.environ[env_name] = key.strip()
    print(f"{env_name} set ({len(os.environ[env_name])} characters)")
else:
    print("mock provider: no key needed")""")

md("### 3a. Data — download SubstanReview (public) and write the evaluation manifests")
code(r"""!python scripts/prepare_data.py --substanreview""")

md("""### 3b. (Optional) eLife gold annotations

The eLife reviewer text is public (`elife-article-xml`) but the study's human gold labels are not. To include eLife:
upload `elife_gold.jsonl` (and optionally `elife_demos.jsonl`) in the JSONL layout described in `docs/DATA.md`
to `data/elife/gold/`, then add `"elife"` to `DATASETS` above. To fetch reviewer text for annotation:
`!python scripts/prepare_data.py --elife-ids my_ids.txt` or `--elife-sample 40 --years 2016 2017 2018 2019 2020 --mailto you@org`.""")
code(r"""pathlib.Path("data/elife/gold").mkdir(parents=True, exist_ok=True)
if IN_COLAB and "elife" in DATASETS and not pathlib.Path("data/elife/gold/elife_gold.jsonl").exists():
    from google.colab import files  # type: ignore
    print("Upload elife_gold.jsonl (and elife_demos.jsonl):")
    up = files.upload()
    for name, content in up.items():
        pathlib.Path("data/elife/gold", name).write_bytes(content)
print("eLife gold present:", pathlib.Path("data/elife/gold/elife_gold.jsonl").exists())""")

md("### 3c. Check that the model ids exist at the provider (records the catalogue for the manifest)")
code(r"""from zengziagent.llm import resolve_backend
for spec in BACKENDS:
    be = resolve_backend(spec)
    ok = be.check_model_available() if hasattr(be, "check_model_available") else True
    print(f"{spec:28s} -> provider={be.provider:12s} model={be.model:40s} family={be.family:10s} listed={ok}")""")

md("## 4. Component-contribution study (main results + ablations)\n\nEach run directory gets `predictions.jsonl` (raw LLM outputs, validation reports, aligned spans), `manifest.json`, `prompt_bundle.json` and `task_specification.json`. Finished runs are skipped on re-execution; LLM responses are cached in SQLite.")
code(r"""import shlex, time
def sh(cmd):
    print("$", cmd); t0 = time.time()
    r = subprocess.run(cmd, shell=True)
    print(f"[exit {r.returncode}, {time.time()-t0:.0f}s]")
    if r.returncode: raise SystemExit(r.returncode)

limit = f"--limit {LIMIT}" if LIMIT else ""
sh(f"python -m zengziagent.experiments.run_ablation --backends {' '.join(BACKENDS)} --datasets {' '.join(DATASETS)} "
   f"--configs {' '.join(CONFIGS)} --runs {RUNS} --workers {WORKERS} {limit}")""")

md("## 5. Two-round reflection (R1 → R2) for the full configuration")
code(r"""if RUN_REFLECTION:
    for b in BACKENDS:
        sh(f"python -m zengziagent.experiments.run_reflection --backend {b} --datasets {' '.join(DATASETS)} --configs F --runs {RUNS} --workers {WORKERS} {limit}")""")

md("## 6. Evaluation (single source of truth) and statistics\n\n`evaluate` recomputes every metric from the raw spans; `analyze_ablation` adds paired-bootstrap CIs, exact McNemar tests and Holm correction; `make_tables` / `make_figures` render the manuscript tables and figures; `audit` verifies that every P/R/F1/Accuracy value is consistent with its counts.")
code(r"""sh("python -m zengziagent.experiments.evaluate")
sh(f"python -m zengziagent.experiments.analyze_ablation --resamples {RESAMPLES}")
sh("python -m zengziagent.experiments.make_tables")
sh("python -m zengziagent.experiments.make_figures")
sh("python -m zengziagent.evaluation.audit results/master/master_results.csv")""")

md("## 7. Results\n\n### Table 11 — performance (dataset-specific and pooled across datasets)")
code(r"""import pandas as pd
from IPython.display import display, Markdown, Image
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
display(Markdown(open("results/tables/table_performance.md").read()))
display(Markdown(open("results/tables/table_performance_counts.md").read()))""")

md("### Table 12 — component contribution: ΔAccuracy, ΔF1, 95% CI, Holm-adjusted p (per dataset × backend)")
code(r"""abl = pd.read_csv("results/tables/ablation_main.csv")
cols = ["dataset","backend","ablation","ablation_name","n_units","delta_accuracy","acc_ci_low","acc_ci_high","delta_f1","f1_ci_low","f1_ci_high","p_bootstrap_f1","p_adj_bootstrap_f1","p_mcnemar","p_adj_mcnemar"]
display(abl[cols].round(4))
display(Image("results/figures/fig_ablation_forest.png", width=900))""")

md("### Table 13 — alignment-aware span metrics (full vs. w/o Text Alignment vs. fixed-prompt baseline)")
code(r"""display(Markdown(open("results/tables/table_alignment.md").read()))
display(Markdown(open("results/tables/table_tau_sensitivity.md").read()))
display(Image("results/figures/fig_tau_sensitivity.png", width=600))""")

md("### Dataset contrast, backend comparison, multi-run stability, R1 → R2")
code(r"""def read_table(path):
    p = pathlib.Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return None
    return df if len(df) else None

for name in ["ablation_contrast", "backend_pairwise", "multirun_stability", "reflection_r1_r2"]:
    df = read_table(f"results/tables/{name}.csv")
    if df is not None:
        display(Markdown(f"**{name}**")); display(df.round(4))
    else:
        print(f"{name}: no data (e.g. a dataset contrast needs two datasets; R1/R2 needs the reflection step)")
for fig in ["fig_per_label_accuracy.png", "fig_multirun_agreement.png", "fig_reflection_delta.png"]:
    p = pathlib.Path("results/figures", fig)
    if p.exists(): display(Image(str(p), width=800))""")

md("### Prompts actually used (release these in the Supplementary Material)")
code(r"""import json, glob
b0 = sorted(glob.glob("results/raw/*/*/B0/run0/prompt_bundle.json"))
full = sorted(glob.glob("results/raw/*/*/F/run0/prompt_bundle.json"))
if b0:
    print("=== Fixed Instruction Baseline (B0 / A2) ===\n", json.load(open(b0[0]))["system_prompt"][:1500])
if full:
    print("\n=== Adaptive prompt P_{D,M} (F) — first 2500 characters ===\n", json.load(open(full[0]))["system_prompt"][:2500])""")

md("## 8. Optional — transformer baselines (Table 7; needs a GPU runtime, ~10–30 min per model)")
code(r"""RUN_BASELINES = False
BASELINE_MODELS = ["roberta", "electra", "albert", "gpt2"]
if RUN_BASELINES:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "transformers", "sentencepiece"], check=True)
    for m in BASELINE_MODELS:
        for ds in DATASETS:
            sh(f"python -m zengziagent.baselines.transformers_baseline --dataset {ds} --model {m} --seeds 13 42 2024 {limit}")
    sh("python -m zengziagent.experiments.evaluate && python -m zengziagent.experiments.make_tables")
    display(Markdown(open("results/tables/table_performance.md").read()))""")

md("## 9. Optional — scientometric application (eLife predictions + OpenAlex citations)\n\nRequires eLife records with `article_id`, `year`, `doi` metadata. The synthetic switch only exercises the code path.")
code(r"""RUN_SCIENTOMETRICS = False
SYNTHETIC_DEMO = False
OPENALEX_MAILTO = ""     # polite-pool contact e-mail
if RUN_SCIENTOMETRICS and "elife" in DATASETS:
    pred = sorted(glob.glob("results/raw/elife/*/F/run0/predictions.jsonl"))[0]
    sh(f"python -m zengziagent.scientometrics.build_features --predictions {pred} --records data/elife/gold/elife_gold.jsonl --mailto '{OPENALEX_MAILTO}' --out results/scientometrics/article_features.csv")
    sh("python -m zengziagent.scientometrics.analysis --features results/scientometrics/article_features.csv --out results/scientometrics")
    display(Image("results/scientometrics/fig_scientometrics.png", width=900))
elif SYNTHETIC_DEMO:
    sh("python -m zengziagent.scientometrics.analysis --synthetic 200 --out results/scientometrics_synthetic")""")

md("## 10. Download everything (raw predictions, manifests, master CSV, tables, figures)")
code(r"""import shutil
archive = shutil.make_archive("zengziagent_results", "zip", root_dir=".", base_dir="results")
print("archive:", archive, f"({os.path.getsize(archive)/1e6:.1f} MB)")
if IN_COLAB:
    from google.colab import files  # type: ignore
    files.download(archive)""")

nb["cells"] = cells
out = Path(__file__).resolve().parent.parent / "notebooks" / "ZengziAgent_Colab.ipynb"
out.parent.mkdir(exist_ok=True)
if nbf is not None:
    nbf.write(nb, str(out))
else:
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", out, "cells:", len(cells))
