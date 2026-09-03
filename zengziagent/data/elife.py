"""eLife reviewer-report extraction from the public ``elife-article-xml`` repository.

Article XML files are fetched from
``https://raw.githubusercontent.com/elifesciences/elife-article-xml/master/articles/elife-<id>-v<k>.xml``
(the highest available version is used; version 1 files are frequently stubs without the
decision letter).  Reviewer text lives in ``<sub-article article-type="decision-letter">``
(2016-2020 style) or ``referee-report`` (2021+ public reviews); author responses
(``reply`` / ``author-comment``) are never annotation targets.

Gold labels for eLife are the study's own human annotations, stored in the same JSONL
schema as SubstanReview and joined by ``review_id``.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from ..schema import ReviewRecord
from ..utils import LOG, read_jsonl

RAW_BASE = "https://raw.githubusercontent.com/elifesciences/elife-article-xml/master/articles/"
_REVIEWER_RE = re.compile(r"^\s*Reviewer\s*#?\s*(\d+)\s*(?:\(.*?\))?\s*[:\-–—]?\s*$", re.I)
_REVIEWER_INLINE_RE = re.compile(r"^\s*Reviewer\s*#?\s*(\d+)\s*(?:\(.*?\))?\s*[:\-–—]\s*(.+)$", re.I | re.S)
_BOILERPLATE_PREFIXES = (
    "in the interests of transparency",
    "thank you for submitting",
    "thank you for sending",
    "thank you for resubmitting",
    "thank you for choosing to send",
    "your article has been",
    "the reviewers have discussed",
    "the reviewers have opted",
    "we are pleased to",
    "[editors' note",
    "[editors’ note",
)
_SECTION_HEADERS = {"summary:", "essential revisions:", "major comments:", "minor comments:", "reviewer #1:", "additional comments:"}


@dataclass
class ReviewerReport:
    article_id: str
    doi: str
    year: Optional[int]
    round_index: int
    sub_article_type: str
    reviewer: str  # "1", "2", "editor-consolidated" ...
    paragraphs: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(p for p in self.paragraphs if p.strip())

    @property
    def review_id(self) -> str:
        return f"elife-{self.article_id}-r{self.round_index}-{self.reviewer}"


def fetch_article_xml(article_id: str | int, cache_dir: str | Path = "data/elife/cache", max_version: int = 6, client=None) -> Optional[Path]:
    """Download the highest available version of an article's XML (cached)."""
    import httpx  # noqa: WPS433

    aid = f"{int(article_id):05d}"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(cache_dir.glob(f"elife-{aid}-v*.xml"))
    if existing:
        return existing[-1]
    own = client is None
    client = client or httpx.Client(timeout=120, follow_redirects=True)
    try:
        for v in range(max_version, 0, -1):
            url = f"{RAW_BASE}elife-{aid}-v{v}.xml"
            r = client.get(url)
            if r.status_code == 200 and len(r.content) > 1000:
                p = cache_dir / f"elife-{aid}-v{v}.xml"
                p.write_bytes(r.content)
                return p
            time.sleep(0.05)
    finally:
        if own:
            client.close()
    LOG.warning("no XML found for eLife article %s", aid)
    return None


def _text(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def parse_article(xml_bytes: bytes) -> dict:
    root = ET.fromstring(xml_bytes)
    meta = root.find("./front/article-meta")
    article_id = doi = None
    year = None
    title = None
    if meta is not None:
        for aid in meta.findall("article-id"):
            if aid.get("pub-id-type") == "publisher-id":
                article_id = (aid.text or "").strip()
            if aid.get("pub-id-type") == "doi":
                doi = (aid.text or "").strip()
        t = meta.find("./title-group/article-title")
        title = _text(t) if t is not None else None
        for pd in meta.findall("pub-date"):
            y = pd.find("year")
            if y is not None and y.text and y.text.strip().isdigit():
                year = int(y.text.strip())
                break
    subs = []
    for sa in root.iter("sub-article"):
        body = sa.find("body")
        paras: list[str] = []

        def walk(el) -> None:
            for child in el:
                if child.tag == "p":
                    txt = _text(child)
                    if txt:
                        paras.append(txt)
                else:
                    walk(child)

        if body is not None:
            walk(body)
        t = sa.find("./front-stub/title-group/article-title")
        subs.append({"type": sa.get("article-type"), "id": sa.get("id"), "title": _text(t) if t is not None else None, "paragraphs": paras})
    return {"article_id": article_id, "doi": doi, "year": year, "title": title, "sub_articles": subs}


def extract_reviewer_reports(article: dict, drop_boilerplate: bool = True) -> list[ReviewerReport]:
    """Split decision letters / referee reports into reviewer sections with round metadata."""
    reports: list[ReviewerReport] = []
    round_index = 0
    for sub in article["sub_articles"]:
        if sub["type"] not in {"decision-letter", "referee-report", "editor-report", "evaluation-summary"}:
            continue
        if sub["type"] == "decision-letter":
            round_index += 1
        rnd = round_index or 1
        sections: dict[str, list[str]] = {}
        current = "editor-consolidated" if sub["type"] == "decision-letter" else (sub["type"] + (":" + sub["id"] if sub["id"] else ""))
        for p in sub["paragraphs"]:
            low = p.lower().strip()
            if drop_boilerplate and any(low.startswith(b) for b in _BOILERPLATE_PREFIXES):
                continue
            m = _REVIEWER_RE.match(p)
            if m:
                current = m.group(1)
                continue
            m2 = _REVIEWER_INLINE_RE.match(p)
            if m2 and len(m2.group(2)) > 0:
                current = m2.group(1)
                sections.setdefault(current, []).append(m2.group(2).strip())
                continue
            sections.setdefault(current, []).append(p)
        for reviewer, paras in sections.items():
            paras = [p for p in paras if p.strip() and p.strip().lower() not in _SECTION_HEADERS]
            if not paras:
                continue
            reports.append(
                ReviewerReport(
                    article_id=article["article_id"] or "",
                    doi=article["doi"] or "",
                    year=article["year"],
                    round_index=rnd,
                    sub_article_type=sub["type"],
                    reviewer=str(reviewer),
                    paragraphs=paras,
                )
            )
    return reports


def build_review_records(
    article_ids: list[str | int],
    cache_dir: str | Path = "data/elife/cache",
    granularity: str = "article",
    dataset: str = "elife",
) -> list[ReviewRecord]:
    """Fetch + extract reviewer text.  ``granularity='article'`` concatenates all reviewer
    sections of an article's decision letter(s) into one record (the manuscript's 200
    eLife "reviews"); ``'reviewer'`` yields one record per reviewer section."""
    import httpx  # noqa: WPS433

    records: list[ReviewRecord] = []
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for aid in article_ids:
            path = fetch_article_xml(aid, cache_dir, client=client)
            if path is None:
                continue
            art = parse_article(path.read_bytes())
            reports = extract_reviewer_reports(art)
            if not reports:
                LOG.warning("article %s: no reviewer text found", aid)
                continue
            if granularity == "reviewer":
                for rep in reports:
                    records.append(
                        ReviewRecord(
                            review_id=rep.review_id,
                            dataset=dataset,
                            text=rep.text,
                            gold_spans=[],
                            metadata={"article_id": rep.article_id, "doi": rep.doi, "year": rep.year, "round": rep.round_index, "reviewer": rep.reviewer, "sub_article_type": rep.sub_article_type, "xml_file": path.name},
                        )
                    )
            else:
                parts = []
                for rep in reports:
                    header = f"[Round {rep.round_index} – Reviewer {rep.reviewer}]"
                    parts.append(header + "\n" + rep.text)
                records.append(
                    ReviewRecord(
                        review_id=f"elife-{art['article_id']}",
                        dataset=dataset,
                        text="\n\n".join(parts),
                        gold_spans=[],
                        metadata={"article_id": art["article_id"], "doi": art["doi"], "year": art["year"], "title": art["title"], "n_reports": len(reports), "rounds": sorted({r.round_index for r in reports}), "xml_file": path.name},
                    )
                )
    return records


def attach_gold(records: list[ReviewRecord], gold_path: str | Path) -> list[ReviewRecord]:
    """Join human gold spans (JSONL with review_id + gold_spans / label) onto extracted records."""
    from .units import load_records_jsonl

    gold = {r.review_id: r for r in load_records_jsonl(gold_path, dataset="elife")}
    out = []
    for r in records:
        g = gold.get(r.review_id)
        if g is None:
            continue
        if g.text != r.text:
            LOG.warning("gold text differs from extracted text for %s; using gold text as the source of truth", r.review_id)
            r.text = g.text
        r.gold_spans = g.gold_spans
        out.append(r)
    return out


def list_elife_articles_openalex(year: int, n: int, seed: int = 0, mailto: Optional[str] = None, per_page: int = 200, max_pages: int = 20) -> list[dict]:
    """Deterministically sample ``n`` eLife research articles from ``year`` via OpenAlex.

    Uses ``filter=primary_location.source.issn:2050-084X,publication_year:<year>,type:article``.
    """
    import httpx  # noqa: WPS433

    works: list[dict] = []
    cursor = "*"
    with httpx.Client(timeout=60) as client:
        for _ in range(max_pages):
            params = {
                "filter": f"primary_location.source.issn:2050-084X,publication_year:{year},type:article",
                "per-page": per_page,
                "cursor": cursor,
                "select": "id,doi,publication_year,title",
            }
            if mailto:
                params["mailto"] = mailto
            r = client.get("https://api.openalex.org/works", params=params)
            if r.status_code == 429:
                time.sleep(2.0)
                continue
            r.raise_for_status()
            data = r.json()
            for w in data.get("results", []):
                doi = (w.get("doi") or "").lower()
                m = re.search(r"10\.7554/elife\.(\d+)", doi)
                if m:
                    works.append({"openalex_id": w["id"], "doi": doi, "article_id": m.group(1), "year": w.get("publication_year"), "title": w.get("title")})
            cursor = (data.get("meta") or {}).get("next_cursor")
            if not cursor:
                break
            time.sleep(0.2)
    rng = random.Random(seed)
    works.sort(key=lambda w: w["article_id"])
    rng.shuffle(works)
    return works[:n]
