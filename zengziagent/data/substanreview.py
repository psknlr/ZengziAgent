"""SubstanReview (Guo et al., 2023) loader with the study's label mapping.

The released ``annotation_final/{train,test}.jsonl`` files contain 550 reviews (440/110)
with span labels ``Eval_pos_k``, ``Eval_neg_k``, ``Jus_pos_k``, ``Jus_neg_k`` and
``Major_claim``.  Occurrence suffixes ``_k`` link a justification to its evaluation and are
kept as metadata.  Label mapping is deterministic (``schema.canonical_label``).

Note for the manuscript: the release *does* contain a ``Major_claim`` label (163 spans),
matching the Major_Claim count in Table 4; the text of Section 4.1.2 should be checked
against this fact.
"""
from __future__ import annotations

from pathlib import Path

from ..schema import ReviewRecord
from ..utils import LOG
from .units import load_records_jsonl

RAW_BASE = "https://raw.githubusercontent.com/YanzhuGuo/SubstanReview/main/"
FILES = {
    "train": "annotation_final/train.jsonl",
    "test": "annotation_final/test.jsonl",
    "iaa_1": "annotation_IAA/annotator_1.jsonl",
    "iaa_2": "annotation_IAA/annotator_2.jsonl",
    "iaa_3": "annotation_IAA/annotator_3.jsonl",
    "license": "LICENSE",
}


def download_substanreview(dest: str | Path = "data/substanreview", force: bool = False) -> dict[str, Path]:
    import httpx  # noqa: WPS433

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for key, rel in FILES.items():
            target = dest / Path(rel).name if key != "license" else dest / "LICENSE"
            if key.startswith("iaa"):
                target = dest / f"iaa_annotator_{key[-1]}.jsonl"
            if target.exists() and not force:
                out[key] = target
                continue
            url = RAW_BASE + rel
            LOG.info("downloading %s", url)
            r = client.get(url)
            r.raise_for_status()
            target.write_bytes(r.content)
            out[key] = target
    return out


def load_substanreview(root: str | Path = "data/substanreview", split: str = "test") -> list[ReviewRecord]:
    root = Path(root)
    path = root / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python scripts/prepare_data.py --substanreview` first")
    records = load_records_jsonl(path, dataset="substanreview")
    for r in records:
        r.review_id = f"sr-{split}-{r.review_id}"
        r.metadata.setdefault("split", split)
        r.metadata.setdefault("source", "SubstanReview (Guo et al., 2023)")
    return records


def load_iaa(root: str | Path = "data/substanreview") -> dict[str, list[ReviewRecord]]:
    root = Path(root)
    out = {}
    for k in (1, 2, 3):
        p = root / f"iaa_annotator_{k}.jsonl"
        if p.exists():
            recs = []
            for rec in load_records_jsonl(p, dataset="substanreview-iaa"):
                rec.review_id = f"sr-iaa-{rec.review_id}"
                recs.append(rec)
            out[f"annotator_{k}"] = recs
    return out
