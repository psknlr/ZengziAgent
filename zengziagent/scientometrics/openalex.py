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
                time.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
    finally:
        if own:
            client.close()
    return None


def five_year_citations(work: dict, pub_year: Optional[int] = None, window: int = 5) -> Optional[int]:
    """Sum ``counts_by_year`` for ``pub_year .. pub_year + window - 1`` (calendar window)."""
    year = pub_year or work.get("publication_year")
    if year is None:
        return None
    return sum(int(c.get("cited_by_count", 0)) for c in work.get("counts_by_year", []) if year <= int(c.get("year", 0)) < year + window)


def fetch_citations(dois: list[str], cache_path: str | Path = "data/elife/citations_cache.json", mailto: Optional[str] = None, years: Optional[dict[str, int]] = None, window: int = 5) -> dict[str, dict]:
    import httpx  # noqa: WPS433

    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    with httpx.Client(timeout=60) as client:
        for doi in dois:
            key = doi.lower()
            if key in cache:
                continue
            w = fetch_work(doi, mailto=mailto, client=client)
            if w is None:
                LOG.warning("no OpenAlex record for %s", doi)
                cache[key] = {"doi": doi, "found": False}
            else:
                cache[key] = {"doi": doi, "found": True, "openalex_id": w.get("id"), "publication_year": w.get("publication_year"), "cited_by_count_total": w.get("cited_by_count"), "counts_by_year": w.get("counts_by_year", []), "citations_5y": five_year_citations(w, (years or {}).get(doi), window), "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            time.sleep(0.15)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1))
    return cache
