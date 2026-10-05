"""Transformer baselines (Table 7): RoBERTa-base, ELECTRA-base, ALBERT-base, GPT-2-medium.

Each model is fine-tuned as a *unit classifier* over sentence units (five labels + ``None``)
with the manuscript's hyper-parameters (AdamW, weight decay 0.01, linear warm-up over 10% of
the steps, max length 512, early stopping on validation loss, effective batch size 32 via
gradient accumulation).  Predictions are written in the same ``predictions.jsonl`` format as
the LLM pipeline (a predicted span = a sentence unit with a non-``None`` label), so the same
evaluation script scores them; the manifest marks ``alignment_applicable = false`` because the
output mechanism has no free-form source spans (reported as N/A, not 0).

Requires ``pip install torch transformers``.  Example::

    python -m zengziagent.baselines.transformers_baseline --dataset substanreview --model roberta --seeds 13 42 2024
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Optional

from .. import __version__
from ..data.units import build_units, dataset_hash
from ..schema import LABELS, ReviewRecord
from ..utils import LOG, git_commit, repo_root, setup_logging, utc_now, write_json, write_jsonl
from ..experiments.common import ExperimentConfig

MODELS = {
    "roberta": {"checkpoint": "roberta-base", "params": "125M", "lr": 2e-5, "batch": 16, "epochs": 5},
    "electra": {"checkpoint": "google/electra-base-discriminator", "params": "110M", "lr": 2e-5, "batch": 16, "epochs": 5},
    "albert": {"checkpoint": "albert-base-v2", "params": "12M", "lr": 3e-5, "batch": 32, "epochs": 5},
    "gpt2": {"checkpoint": "gpt2-medium", "params": "355M", "lr": 1e-5, "batch": 8, "epochs": 3},
}
CLASSES = list(LABELS) + ["None"]


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_examples(records: list[ReviewRecord], context_window: int = 0) -> list[dict]:
    out = []
    for r in records:
        units = build_units(r, "sentence")
        for i, u in enumerate(units):
            ctx = " ".join(units[j].text for j in range(max(0, i - context_window), i)) if context_window else ""
            out.append({"review_id": r.review_id, "start": u.start, "end": u.end, "text": u.text, "context": ctx, "label": u.gold_label})
    return out


def train_and_predict(
    model_key: str,
    train_records: list[ReviewRecord],
    test_records: list[ReviewRecord],
    seed: int,
    out_dir: Path,
    dataset: str,
    checkpoint: Optional[str] = None,
    max_length: int = 512,
    effective_batch: int = 32,
    val_fraction: float = 0.1,
    patience: int = 2,
    context_window: int = 0,
    max_train_examples: Optional[int] = None,
    device: Optional[str] = None,
) -> Path:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    cfg = MODELS[model_key]
    ckpt = checkpoint or cfg["checkpoint"]
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    started = utc_now()
    # ---- data
    rng = random.Random(seed)
    ids = sorted({r.review_id for r in train_records})
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    val_ids = set(ids[:n_val])
    tr = make_examples([r for r in train_records if r.review_id not in val_ids], context_window)
    va = make_examples([r for r in train_records if r.review_id in val_ids], context_window)
    te = make_examples(test_records, context_window)
    if max_train_examples:
        tr = tr[:max_train_examples]
    LOG.info("%s: %d train / %d val / %d test sentence units (device=%s)", ckpt, len(tr), len(va), len(te), device)
    tok = AutoTokenizer.from_pretrained(ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=len(CLASSES))
    model.config.pad_token_id = tok.pad_token_id
    model.to(device)
    lab2id = {c: i for i, c in enumerate(CLASSES)}

    def collate(batch):
        if context_window and any(b["context"] for b in batch):
            # pair encoding inserts the model's own separator and truncates the context
            # ("only_first") rather than the sentence being classified; GPT-2 has no separator,
            # so a newline marks the boundary.  If the sentence alone exceeds max_length the
            # tokenizer cannot honour "only_first" and we fall back to "longest_first".
            ctxs = [(b["context"] or "") + ("" if tok.sep_token else "\n") for b in batch]
            texts = [b["text"] for b in batch]
            try:
                enc = tok(ctxs, texts, truncation="only_first", max_length=max_length, padding=True, return_tensors="pt")
            except Exception:
                enc = tok(ctxs, texts, truncation="longest_first", max_length=max_length, padding=True, return_tensors="pt")
        else:
            enc = tok([b["text"] for b in batch], truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([lab2id[b["label"]] for b in batch])
        return enc

    bs = cfg["batch"]
    accum = max(1, effective_batch // bs)
    dl_tr = DataLoader(tr, batch_size=bs, shuffle=True, collate_fn=collate)
    dl_va = DataLoader(va, batch_size=bs, shuffle=False, collate_fn=collate)
    dl_te = DataLoader(te, batch_size=bs, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total_steps = math.ceil(len(dl_tr) / accum) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    def evaluate_loss(dl):
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                tot += float(out.loss) * batch["labels"].shape[0]
                n += batch["labels"].shape[0]
        return tot / max(1, n)

    best_loss, best_state, bad = float("inf"), None, 0
    history = []
    for epoch in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        opt.zero_grad()
        for step, batch in enumerate(dl_tr):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / accum
            loss.backward()
            if (step + 1) % accum == 0 or step + 1 == len(dl_tr):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
        vl = evaluate_loss(dl_va)
        history.append({"epoch": epoch + 1, "val_loss": vl, "seconds": round(time.time() - t0, 1)})
        LOG.info("epoch %d val_loss %.4f (%.0fs)", epoch + 1, vl, time.time() - t0)
        if vl < best_loss - 1e-4:
            best_loss, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                LOG.info("early stopping at epoch %d", epoch + 1)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    # ---- predict
    model.eval()
    preds: list[int] = []
    with torch.no_grad():
        for batch in dl_te:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
            preds.extend(logits.argmax(-1).tolist())
    by_review: dict[str, list[dict]] = {}
    for ex, p in zip(te, preds):
        lab = CLASSES[p]
        if lab == "None":
            continue
        by_review.setdefault(ex["review_id"], []).append({"order": len(by_review.get(ex["review_id"], [])), "text": ex["text"], "label": lab, "label_raw": lab, "status": "exact", "start": ex["start"], "end": ex["end"], "original_start": ex["start"], "original_end": ex["end"], "token_start": None, "token_end": None, "similarity": 1.0, "method": "unit_classification", "index_hint": None, "recovered_text": ex["text"]})
    backend_spec = f"hf:{model_key}"
    rows = []
    for r in test_records:
        rows.append({"review_id": r.review_id, "dataset": dataset, "config_id": "TB", "backend": backend_spec, "model": ckpt, "run_index": seed, "round": 1, "retries": 0, "flagged": False, "attempts": [], "spans": by_review.get(r.review_id, []), "error": None})
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "predictions.jsonl", rows)
    manifest = {
        "package_version": __version__,
        "git_commit": git_commit(repo_root()),
        "dataset": dataset,
        "n_records": len(test_records),
        "record_ids": [r.review_id for r in test_records],
        "dataset_hash": dataset_hash(test_records),
        "backend_spec": backend_spec,
        "provider": "huggingface",
        "model_requested": ckpt,
        "models_returned": [ckpt],
        "backend_family": "transformer-baseline",
        "config": {"config_id": "TB", "name": f"Transformer baseline ({ckpt})"},
        "hyperparameters": {"learning_rate": cfg["lr"], "batch_size": cfg["batch"], "effective_batch_size": effective_batch, "epochs": cfg["epochs"], "max_length": max_length, "optimizer": "AdamW", "weight_decay": 0.01, "scheduler": "linear with 10% warm-up", "early_stopping_patience": patience, "val_fraction": val_fraction, "context_window": context_window, "classes": CLASSES},
        "seed": seed,
        "run_index": seed,
        "round": 1,
        "training_history": history,
        "best_val_loss": best_loss,
        "started_at": started,
        "finished_at": utc_now(),
        "alignment_applicable": False,
        "unit_construction": "sentence units labelled by the gold span with the largest token overlap (None otherwise)",
        "is_mock": False,
        "device": device,
    }
    write_json(out_dir / "manifest.json", manifest)
    LOG.info("wrote %s", out_dir / "predictions.jsonl")
    return out_dir


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="substanreview")
    ap.add_argument("--model", choices=sorted(MODELS), default="roberta")
    ap.add_argument("--checkpoint", default=None, help="override the Hugging Face checkpoint (e.g. a tiny model for smoke tests)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[13, 42, 2024])
    ap.add_argument("--train-jsonl", default=None, help="annotated training records (JSONL); default: the dataset's demonstration/train split")
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--context-window", type=int, default=0)
    ap.add_argument("--max-train-examples", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N test reviews (smoke tests)")
    ap.add_argument("--experiment-config", default=None)
    args = ap.parse_args(argv)
    setup_logging()
    exp = ExperimentConfig.load(args.experiment_config)
    test_records, train_records = exp.load_dataset(args.dataset)
    if args.train_jsonl:
        from ..data.units import load_records_jsonl

        train_records = load_records_jsonl(args.train_jsonl, dataset=args.dataset)
    if not train_records:
        raise SystemExit("no annotated training records available for this dataset")
    if args.limit:
        test_records = test_records[: args.limit]
    for seed in args.seeds:
        out_dir = Path(args.out) / args.dataset / f"hf-{args.model}" / "TB" / f"run{seed}"
        train_and_predict(args.model, train_records, test_records, seed, out_dir, args.dataset, checkpoint=args.checkpoint, max_length=args.max_length, context_window=args.context_window, max_train_examples=args.max_train_examples)


if __name__ == "__main__":  # pragma: no cover
    main()
