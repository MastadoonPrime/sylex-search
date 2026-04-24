"""Web search fallback for Sylex Search.

When the product index returns thin or empty results, this module
searches the web to fill the gap. Results are tagged with
source="web" so agents know they're unverified.

Good web results are auto-ingested into the index at low data_quality
so they appear in future searches but rank below curated entries.

Search providers (tried in order):
1. DuckDuckGo — free, no API key needed
2. Tavily — 1,000 free queries/month, needs TAVILY_API_KEY
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
_TIMEOUT = 3  # seconds — don't slow down the response

# Domains that are aggregators/listicles, not products themselves
_BLOCKLIST_DOMAINS = {
    "reddit.com", "g2.com", "capterra.com", "medium.com",
    "wikipedia.org", "youtube.com", "twitter.com", "x.com",
    "linkedin.com", "quora.com", "stackoverflow.com",
    "trustpilot.com", "producthunt.com", "alternativeto.net",
    "slant.co", "sourceforge.net", "facebook.com", "instagram.com",
    "tiktok.com", "pinterest.com", "yelp.com",
}

# ---------- Cache (mirrors db.py pattern) ----------

_web_cache: dict[str, tuple[float, list]] = {}
_WEB_CACHE_TTL = 300  # 5 minutes


def _cache_get(key: str) -> Optional[list]:
    if key in _web_cache:
        ts, val = _web_cache[key]
        if (time.time() - ts) < _WEB_CACHE_TTL:
            return val
        del _web_cache[key]
    return None


def _cache_set(key: str, val: list):
    _web_cache[key] = (time.time(), val)
    # Periodic cleanup
    if len(_web_cache) > 500:
        now = time.time()
        stale = [k for k, (ts, _) in _web_cache.items() if now - ts > _WEB_CACHE_TTL]
        for k in stale:
            del _web_cache[k]


# ---------- Domain helpers ----------

def _extract_domain(url: str) -> str:
    """Extract root domain from URL."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        # Strip www.
        if host.startswith("www."):
            host = host[4:]
        return host.lower()
    except Exception:
        return ""


def _is_blocked_domain(url: str) -> bool:
    """Check if URL belongs to a listicle/aggregator domain."""
    domain = _extract_domain(url)
    return any(domain == d or domain.endswith("." + d) for d in _BLOCKLIST_DOMAINS)


def _is_listicle_title(title: str) -> bool:
    """Detect listicle/comparison titles that aren't actual products."""
    patterns = [
        r"^top\s+\d+",
        r"^\d+\s+best",
        r"best\s+\d+",
        r"alternatives?\s+to",
        r"vs\.?\s+",
        r"comparison",
        r"review[s]?\s+(of|for|\d)",
    ]
    title_lower = title.lower()
    return any(re.search(p, title_lower) for p in patterns)


# ---------- Fit scoring for web results ----------

def _score_web_result(title: str, description: str, url: str, query_words: list[str]) -> int:
    """Heuristic fit score for a web result. No LLM calls.

    Web results start at 30 and cap at 60 — they should never
    outscore strong index matches.
    """
    score = 30
    title_lower = title.lower()
    desc_lower = description.lower()

    # Query word matches in title (+10 max)
    title_matches = sum(1 for w in query_words if w in title_lower)
    score += min(title_matches * 4, 10)

    # Query word matches in description (+5 max)
    desc_matches = sum(1 for w in query_words if w in desc_lower)
    score += min(desc_matches * 2, 5)

    # Looks like a product site, not a listicle (+5)
    if not _is_listicle_title(title) and not _is_blocked_domain(url):
        score += 5

    # Has a real domain (not a social/content site) (+5)
    domain = _extract_domain(url)
    if domain and "." in domain and len(domain) < 40:
        score += 5

    # Penalize very short descriptions
    if len(description) < 30:
        score -= 5

    return min(max(score, 10), 60)


def _normalize_results(raw_results: list[dict], query_words: list[str], num_results: int) -> list[dict]:
    """Filter, score, and normalize raw search results into compact format."""
    results = []
    for item in raw_results:
        title = item.get("title", "").strip()
        description = item.get("description", "").strip()
        url = item.get("url", "").strip()

        if not title or not url:
            continue

        if _is_blocked_domain(url):
            continue

        if _is_listicle_title(title):
            continue

        fit_score = _score_web_result(title, description, url, query_words)

        domain = _extract_domain(url)
        results.append({
            "name": title,
            "business": domain,
            "summary": description[:150] if description else "",
            "url": url,
            "price": "",
            "rating": None,
            "category": "uncategorized",
            "source": "web",
            "fit_score": fit_score,
        })

    results.sort(key=lambda r: r["fit_score"], reverse=True)
    return results[:num_results]


# ---------- Search providers ----------

def _search_duckduckgo(query: str, num_results: int) -> list[dict]:
    """Search via DuckDuckGo. Free, no API key needed."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.debug("ddgs/duckduckgo-search not installed, skipping DDG")
            return []

    try:
        raw = DDGS().text(query, max_results=num_results * 2)
        return [
            {"title": r.get("title", ""), "description": r.get("body", ""), "url": r.get("href", "")}
            for r in (raw or [])
        ]
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []


def _search_tavily(query: str, num_results: int) -> list[dict]:
    """Search via Tavily API. 1,000 free queries/month."""
    if not TAVILY_API_KEY:
        return []

    try:
        import requests as req
        resp = req.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": min(num_results * 2, 10),
                "search_depth": "basic",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {"title": r.get("title", ""), "description": r.get("content", ""), "url": r.get("url", "")}
            for r in data.get("results", [])
        ]
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return []


# ---------- Main search function ----------

def web_search(query: str, num_results: int = 5) -> list[dict]:
    """Search the web using available providers.

    Tries DuckDuckGo first (free, no key), falls back to Tavily.
    Returns compact result dicts tagged with source="web".
    Returns empty list if all providers fail.
    """
    if not query:
        return []

    cache_key = f"web:{query}:{num_results}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    query_words = [w.lower() for w in re.sub(r'[^\w\s]', ' ', query).split()
                   if len(w) >= 2]

    # Try providers in order
    for search_fn in [_search_duckduckgo, _search_tavily]:
        raw_results = search_fn(query, num_results)
        if raw_results:
            results = _normalize_results(raw_results, query_words, num_results)
            if results:
                _cache_set(cache_key, results)
                return results

    # All providers failed or returned nothing useful
    _cache_set(cache_key, [])
    return []


# ---------- Auto-ingest ----------

def auto_ingest(web_results: list[dict], query: str) -> None:
    """Auto-ingest good web results into the product index.

    Runs in a background thread so it never blocks the response.
    Only ingests results with decent fit scores and non-blocked domains.
    """
    if not web_results:
        return

    # Filter to ingestable results
    candidates = [
        r for r in web_results
        if r.get("fit_score", 0) >= 40
        and not _is_blocked_domain(r.get("url", ""))
        and not _is_listicle_title(r.get("name", ""))
    ]

    if not candidates:
        return

    def _do_ingest():
        try:
            # Late import to avoid circular dependency
            from db import upsert_business, upsert_product, get_db, _slugify

            db = get_db()

            for result in candidates:
                url = result.get("url", "")
                name = result.get("name", "")
                summary = result.get("summary", "")

                if not url or not name:
                    continue

                slug = _slugify(name)

                # Skip if already in the index (by URL or slug)
                existing = db.table("products").select("id").eq("url", url).limit(1).execute()
                if existing.data:
                    continue

                existing_slug = db.table("products").select("id").eq("slug", slug).limit(1).execute()
                if existing_slug.data:
                    continue

                # Create a business entry from the domain
                domain = _extract_domain(url)
                biz_slug = _slugify(domain or name)

                try:
                    biz_id = upsert_business(
                        name=domain or name,
                        slug=biz_slug,
                        website=url,
                    )

                    upsert_product(
                        business_id=biz_id,
                        slug=slug,
                        name=name,
                        description=summary,
                        short_description=summary[:100] if summary else "",
                        url=url,
                        source="web-discovered",
                        category="uncategorized",
                        data_quality=0.3,
                        data={
                            "discovered_from_query": query,
                            "discovered_at": datetime.now(timezone.utc).isoformat(),
                            "needs_enrichment": True,
                        },
                    )
                    logger.info(f"Auto-ingested web result: {name} ({url})")
                except Exception as e:
                    logger.warning(f"Failed to auto-ingest {name}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Auto-ingest batch failed: {e}")

    # Fire and forget
    thread = threading.Thread(target=_do_ingest, daemon=True)
    thread.start()
