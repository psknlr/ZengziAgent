"""Citation retrieval from OpenAlex (five-year calendar window from the publication year)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from ..utils import LOG


def fetch_work(doi: str, mailto: Optional[str] = None, client=None, retries: int = 5) -> Optional[dict]:
    import httpx  # noqa: WPS433

    own = client is None
    client = client or httpx.Client(timeout=60)
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    params = {"mailto": mailto} if mailto else {}
    try:
        for attempt in range(retries):
            r = client.get(url, params=params)
            if r.status_code == 429:
                retry = r.headers.get("Retry-After")
                try:
                    delay = float(retry) if retry else 2.0 * (attempt + 1)
                except ValueError:
                    delay = 2.0 * (attempt + 1)
                time.sleep(min(delay, 60.0))
                continue
            if r.status_code == 404:
                return None  # genuinely unknown DOI
            r.raise_for_status()
            return r.json()
    finally:
        if own:
            client.close()
    raise RuntimeError(f"OpenAlex rate limit persisted for {doi} after {retries} attempts")


OPENALEX_HORIZON_YEARS = 10  # counts_by_year covers the last ten calendar years


def _retrieval_year(work: dict) -> int:
    ra = work.get("retrieved_at")
    if isinstance(ra, str) and len(ra) >= 4 and ra[:4].isdigit():
        return int(ra[:4])
    return time.gmtime().tm_year


def window_coverage(work: dict, pub_year: Optional[int] = None, window: int = 5) -> tuple[Optional[int], bool]:
    """``(publication_year, complete)``.

    OpenAlex lists only years with non-zero citations and only within the last ten calendar
    years (relative to retrieval).  A window is *complete* when the publication year lies inside
    that horizon (missing years then mean zero citations, and an uncited work with an empty list
    counts as 0); for older articles it is complete only if the observed years reach back to the
    publication year."""
    year = pub_year or work.get("publication_year")
    if year is None:
        return None, False
    horizon_start = _retrieval_year(work) - (OPENALEX_HORIZON_YEARS - 1)
    if year >= horizon_start:
        return year, True
    years = {int(c.get("year", 0)) for c in work.get("counts_by_year", []) or []}
    return year, bool(years) and min(years) <= year


def five_year_citations(work: dict, pub_year: Optional[int] = None, window: int = 5) -> Optional[int]:
    """Sum ``counts_by_year`` for ``pub_year .. pub_year + window - 1`` (calendar window).

    Returns ``None`` when the horizon returned by OpenAlex does not cover the publication year,
    so an incomplete window becomes a missing value rather than an undercount."""
    year, complete = window_coverage(work, pub_year, window)
    if year is None or not complete:
        return None
    return sum(int(c.get("cited_by_count", 0)) for c in (work.get("counts_by_year", []) or []) if year <= int(c.get("year", 0)) < year + window)


def fetch_citations(dois: list[str], cache_path: str | Path = "data/elife/citations_cache.json", mailto: Optional[str] = None, years: Optional[dict[str, int]] = None, window: int = 5) -> dict[str, dict]:
    import httpx  # noqa: WPS433

    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    with httpx.Client(timeout=60) as client:
        for doi in dois:
            key = doi.lower()
            entry = cache.get(key)
            if entry and entry.get("found"):
                # recompute the window from the archived counts so --window / corrected years apply
                w = {"counts_by_year": entry.get("counts_by_year", []), "publication_year": entry.get("publication_year"), "retrieved_at": entry.get("retrieved_at"), "cited_by_count": entry.get("cited_by_count_total")}
                entry["citations_5y"] = five_year_citations(w, (years or {}).get(doi), window)
                entry["window"] = window
                entry["window_complete"] = window_coverage(w, (years or {}).get(doi), window)[1]
                continue
            # entries with found=False are re-queried: a rate-limit burst must not become permanent
            w = fetch_work(doi, mailto=mailto, client=client)
            if w is None:
                LOG.warning("no OpenAlex record for %s", doi)
                cache[key] = {"doi": doi, "found": False}
            else:
                w["retrieved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                y, complete = window_coverage(w, (years or {}).get(doi), window)
                if complete and y is not None and y + window - 1 > _retrieval_year(w):
                    LOG.warning("%s: the %d-year window is not yet elapsed (published %s)", doi, window, y)
                if not complete:
                    LOG.warning("OpenAlex counts_by_year does not cover the publication year of %s; citations set to missing", doi)
                cache[key] = {"doi": doi, "found": True, "openalex_id": w.get("id"), "publication_year": w.get("publication_year"), "cited_by_count_total": w.get("cited_by_count"), "counts_by_year": w.get("counts_by_year", []), "citations_5y": five_year_citations(w, (years or {}).get(doi), window), "window": window, "window_complete": complete, "window_elapsed": bool(y is not None and y + window - 1 <= _retrieval_year(w)), "retrieved_at": w["retrieved_at"]}
            time.sleep(0.15)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1))
    return cache
