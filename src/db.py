"""Supabase database layer for the product index.

Architecture:
- The `products` table is a THIN INDEX for search and filtering.
  It stores only what's needed to discover and rank products.
- Each product has a `data` JSONB column — a freeform blob that holds
  everything the product wants to say about itself. Different products
  have different shapes. An LLM can interpret any structured JSON, so
  we don't need to normalize everything into rigid columns.
- The `details` tool returns the blob as-is.
- The `discover` tool returns only index fields (compact, ~60 tokens).
"""

from __future__ import annotations

import hashlib
import ipaddress
import math
import os
import re
import secrets
import socket
import urllib.request
import json as _json
from datetime import datetime
from typing import Optional

from supabase import create_client, Client

SUPABASE_URL = os.environ.get("AC_SUPABASE_URL", "https://qislwyqxtjxybkgwaicq.supabase.co")
SUPABASE_KEY = os.environ.get("AC_SUPABASE_KEY", "")

_client: Optional[Client] = None


def get_db() -> Client:
    """Get or create the Supabase client."""
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def init_db():
    """Initialize the Supabase client. Tables already exist in Supabase."""
    get_db()


# ---------- Simple in-memory cache ----------

_cache: dict[str, tuple[float, any]] = {}
_CACHE_TTL = 300  # 5 minutes


def _cache_get(key: str):
    """Get from cache if not expired."""
    if key in _cache:
        ts, val = _cache[key]
        if (datetime.now().timestamp() - ts) < _CACHE_TTL:
            return val
        del _cache[key]
    return None


def _cache_set(key: str, val):
    """Set cache entry."""
    _cache[key] = (datetime.now().timestamp(), val)
    # Evict old entries if cache gets too large
    if len(_cache) > 500:
        now = datetime.now().timestamp()
        expired = [k for k, (ts, _) in _cache.items() if (now - ts) > _CACHE_TTL]
        for k in expired:
            del _cache[k]


def cache_clear():
    """Clear all cached results. Called via /cache-clear endpoint."""
    _cache.clear()


# ---------- Stop words for query cleaning ----------

_STOP_WORDS = {
    # English stop words
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "that", "this", "these", "those",
    "it", "its", "my", "your", "our", "their", "we", "they", "i",
    "me", "him", "her", "us", "them", "what", "which", "who", "whom",
    "so", "than", "too", "very", "just", "about", "above", "after",
    "before", "between", "into", "through", "during", "from", "up",
    "down", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same",
    # Agent/search context words — filler that doesn't help FTS matching
    "best", "top", "good", "great", "nice", "recommend", "recommendation",
    "need", "want", "looking", "find", "search", "help", "like",
    "small", "large", "medium", "big", "sized", "size",
    "use", "using", "used", "work", "works", "working",
    "modern", "new", "popular",
}


def _clean_query(query: str, use_and: bool = False) -> str:
    """Clean a natural-language query for Postgres websearch_to_tsquery.

    Strips special chars and stop words, returns cleaned string.
    When use_and=True, joins terms without OR (websearch_to_tsquery
    treats space-separated words as AND). When False, uses OR.
    """
    cleaned = re.sub(r'[^\w\s]', ' ', query)
    words = [w.strip().lower() for w in cleaned.split() if len(w.strip()) >= 1]
    content_words = [w for w in words if w not in _STOP_WORDS and len(w) >= 2]
    content_words += [w for w in words if len(w) == 1 and w.isdigit()]
    if not content_words:
        content_words = [w for w in words if len(w) >= 2]
    if not content_words:
        return ""
    if use_and:
        # websearch_to_tsquery treats spaces as AND by default
        return " ".join(content_words)
    # Use OR between terms so partial matches still surface results.
    return " OR ".join(content_words)


# ---------- Query classification ----------

# Signals that suggest a specific source
_SOURCE_SIGNALS = {
    "npm": {
        "keywords": {"npm", "node", "nodejs", "javascript", "js", "typescript", "ts",
                      "react", "vue", "angular", "svelte", "nextjs", "express", "deno", "bun",
                      "webpack", "vite", "eslint", "prettier", "jest", "mocha", "package"},
        "patterns": [r"^@[\w-]+/", r"\.js$", r"\.ts$"],
    },
    "pypi": {
        "keywords": {"pip", "python", "pypi", "django", "flask", "fastapi", "pandas",
                      "numpy", "scipy", "pytorch", "tensorflow", "jupyter", "pytest",
                      "conda", "virtualenv", "pydantic"},
        "patterns": [r"\.py$", r"^py[\w-]"],
    },
    "crates": {
        "keywords": {"rust", "cargo", "crate", "crates", "tokio", "serde", "async-std",
                      "actix", "wasm", "webassembly"},
        "patterns": [],
    },
    "github": {
        "keywords": {"github", "repo", "repository", "open-source", "opensource", "oss",
                      "stars", "fork"},
        "patterns": [r"[\w-]+/[\w-]+"],  # org/repo pattern
    },
    "saas": {
        "keywords": {"saas", "subscription", "pricing", "enterprise", "startup",
                      "crm", "erp", "helpdesk", "analytics", "marketing",
                      "collaboration", "workflow", "dashboard", "onboarding"},
        "patterns": [],
    },
}


def _classify_query(query: str) -> Optional[str]:
    """Detect if a query implies a specific source. Returns source name or None."""
    words = set(re.sub(r'[^\w\s/-]', ' ', query.lower()).split())

    best_source = None
    best_score = 0

    for source, signals in _SOURCE_SIGNALS.items():
        score = len(words & signals["keywords"])
        for pattern in signals.get("patterns", []):
            if re.search(pattern, query, re.IGNORECASE):
                score += 2
        if score > best_score:
            best_score = score
            best_source = source

    return best_source if best_score >= 1 else None


# Map query terms to subcategories for secondary retrieval
_SUBCATEGORY_MAP = {
    "database": ["database", "database-client"],
    "web framework": ["web-framework"],
    "framework": ["web-framework", "framework", "testing-framework"],
    "machine learning": ["machine-learning"],
    "deep learning": ["deep-learning"],
    "testing": ["testing-framework"],
    "test": ["testing-framework"],
    "cli": ["cli"],
    "cli framework": ["cli"],
    "command line": ["cli"],
    "security": ["security"],
    "auth": ["authentication"],
    "authentication": ["authentication"],
    "logging": ["logging"],
    "http": ["http", "http-client"],
    "http client": ["http-client"],
    "api": ["api"],
    "container": ["containerization"],
    "containerization": ["containerization"],
    "build tool": ["build-tools"],
    "build": ["build-tools"],
    "ci cd": ["ci-cd"],
    "orm": ["orm", "database-client"],
    "orm database": ["orm", "database-client"],
    "parser": ["parser"],
    "serialization": ["serialization"],
    "gui": ["gui"],
    "graphics": ["graphics"],
    "game": ["game-dev"],
    "networking": ["networking"],
    "messaging": ["messaging"],
    "encryption": ["cryptography", "security"],
    "crypto": ["cryptography"],
    "linter": ["linter", "code-quality"],
    "formatter": ["formatter"],
    "format": ["formatter"],
    "css": ["css-framework"],
    "css framework": ["css-framework"],
    "styling": ["css-framework"],
    "state management": ["state-management"],
    "state": ["state-management"],
    "package manager": ["package-manager"],
    "web server": ["web-framework"],
    "project management": ["issue-tracking", "all-in-one-workspace", "Issue Tracking & Agile Management", "Project Management & Team Collaboration", "work-management", "kanban"],
    "project manager": ["issue-tracking", "all-in-one-workspace", "Issue Tracking & Agile Management", "Project Management & Team Collaboration", "work-management"],
    "issue tracking": ["issue-tracking", "Issue Tracking & Agile Management"],
    "issue tracker": ["issue-tracking", "Issue Tracking & Agile Management"],
    "ai": ["machine-learning", "deep-learning"],
    "artificial intelligence": ["machine-learning"],
    "data science": ["machine-learning", "data-processing"],
    "nlp": ["machine-learning"],
    "natural language": ["machine-learning"],
    "crm": ["crm"],
    "email": ["email"],
    "analytics": ["analytics"],
    "monitoring": ["monitoring"],
    "devops": ["ci-cd", "containerization"],
    "deployment": ["deployment", "ci-cd"],
    "queue": ["messaging"],
    "message queue": ["messaging"],
}


def _infer_subcategories(query: str) -> list[str]:
    """Infer relevant subcategories from a query for secondary retrieval."""
    q = query.lower().strip()
    # Try exact match first
    if q in _SUBCATEGORY_MAP:
        return _SUBCATEGORY_MAP[q]
    # Try matching multi-word keys
    for key, subs in _SUBCATEGORY_MAP.items():
        if key in q or q in key:
            return subs
    # Try individual words
    matched = []
    for word in q.split():
        if word in _SUBCATEGORY_MAP:
            matched.extend(_SUBCATEGORY_MAP[word])
    return list(dict.fromkeys(matched))  # dedupe preserving order


# ---------- Core query functions ----------

_SELECT_COLS = (
    "id, name, slug, short_description, description, category, subcategory, "
    "price_model, price_min, price_max, has_free_tier, rating, review_count, "
    "data_quality, team_size_min, team_size_max, url, last_verified, source, "
    "data, businesses(name, website)"
)


def search_products(
    query: str,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    team_size: Optional[int] = None,
    has_free_tier: bool = False,
    source: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
    """Full-text search with optional filters. Returns compact results with fit scoring."""
    # Check cache
    cache_key = f"search:{query}:{category}:{subcategory}:{max_price}:{min_rating}:{team_size}:{has_free_tier}:{source}:{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    db = get_db()

    # Build common filter function to avoid repetition
    def _apply_filters(q):
        if category:
            q = q.ilike("category", category.lower().strip())
        if subcategory:
            q = q.ilike("subcategory", subcategory.lower().strip())
        if max_price is not None:
            q = q.or_(f"price_min.is.null,price_min.lte.{max_price}")
        if min_rating is not None:
            q = q.gte("rating", min_rating)
        if has_free_tier:
            q = q.or_("price_min.eq.0,price_min.is.null,price_model.in.(free,freemium),has_free_tier.eq.true")
        if source:
            q = q.eq("source", source.lower().strip())
        if team_size is not None:
            q = q.or_(f"team_size_min.is.null,team_size_min.lte.{team_size}")
            q = q.or_(f"team_size_max.is.null,team_size_max.gte.{team_size}")
        return q

    # Classify query to detect implied source
    implied_source = _classify_query(query) if query and not source else None
    fetch_mult = 8 if implied_source else 6
    fetch_limit = limit * fetch_mult

    # --- Primary retrieval: try AND first, fall back to OR ---
    # For multi-word queries, AND matching is more precise (fewer false positives).
    # If AND returns too few results, fall back to OR.
    rows = []
    if query:
        content_words = [w.lower() for w in re.sub(r'[^\w\s]', ' ', query).split()
                         if len(w) >= 2 and w.lower() not in _STOP_WORDS]
        use_and = len(content_words) >= 2  # Only use AND for multi-word queries

        if use_and:
            tsquery_and = _clean_query(query, use_and=True)
            if tsquery_and:
                q = db.table("products").select(_SELECT_COLS)
                q = _apply_filters(q)
                q = q.limit(fetch_limit)
                q = q.wfts("fts", tsquery_and)
                resp = q.execute()
                rows = resp.data if resp.data else []

        # Fall back to OR if AND returned too few, or single-word query
        if len(rows) < limit * 2:
            tsquery_or = _clean_query(query, use_and=False)
            if tsquery_or:
                seen_ids = {r.get("id") for r in rows}
                q = db.table("products").select(_SELECT_COLS)
                q = _apply_filters(q)
                q = q.limit(fetch_limit)
                q = q.wfts("fts", tsquery_or)
                resp = q.execute()
                if resp.data:
                    for r in resp.data:
                        if r.get("id") not in seen_ids:
                            seen_ids.add(r["id"])
                            rows.append(r)
    else:
        # No query — just filters
        q = db.table("products").select(_SELECT_COLS)
        q = _apply_filters(q)
        q = q.limit(fetch_limit)
        resp = q.execute()
        rows = resp.data if resp.data else []

    # --- Secondary retrieval: subcategory-based ---
    # Fetch products by inferred subcategory to catch well-known products
    # that FTS might miss. We fetch ALL products from each subcategory
    # and let the scoring algorithm rank them — subcategories like "database"
    # can have 100+ entries but we need PostgreSQL/MongoDB to surface.
    if query and not subcategory:
        inferred_subs = _infer_subcategories(query)
        if inferred_subs:
            seen_ids = {r.get("id") for r in rows}
            for sub in inferred_subs[:3]:
                q2 = db.table("products").select(_SELECT_COLS)
                q2 = q2.eq("subcategory", sub)
                if source:
                    q2 = q2.eq("source", source.lower().strip())
                if max_price is not None:
                    q2 = q2.or_(f"price_min.is.null,price_min.lte.{max_price}")
                q2 = q2.order("data_quality", desc=True)
                q2 = q2.limit(200)
                resp2 = q2.execute()
                if resp2.data:
                    for r in resp2.data:
                        if r.get("id") not in seen_ids:
                            seen_ids.add(r["id"])
                            rows.append(r)

    # Extract content words for relevance scoring
    query_words = []
    if query:
        query_words = [w.lower() for w in re.sub(r'[^\w\s]', ' ', query).split()
                       if len(w) >= 2 and w.lower() not in _STOP_WORDS]

    # Compute fit scores and build compact results
    results = []
    for row in rows:
        compact = _row_to_compact(row)
        compact["fit_score"] = _compute_fit_score(
            row, max_price=max_price, team_size=team_size,
            implied_source=implied_source, query_words=query_words,
        )
        results.append(compact)

    # Sort by fit score descending
    results.sort(key=lambda r: r["fit_score"], reverse=True)

    # Deduplicate — same name across different sources, keep highest scored
    results = _deduplicate(results)

    results = results[:limit]

    _cache_set(cache_key, results)

    # --- Web fallback: when index results are thin ---
    if query and _should_web_fallback(results, limit):
        try:
            from web_search import web_search, auto_ingest

            web_results = web_search(query, num_results=limit - len(results))
            if web_results:
                # Deduplicate against index results
                seen_names = {r["name"].lower() for r in results}
                seen_urls = {r.get("url", "").lower() for r in results}
                for wr in web_results:
                    if (wr["name"].lower() not in seen_names
                            and wr.get("url", "").lower() not in seen_urls):
                        results.append(wr)
                results = results[:limit]

                # Auto-ingest good results in the background
                auto_ingest(web_results, query)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Web fallback error: {e}")

    return results


def _should_web_fallback(results: list[dict], limit: int) -> bool:
    """Decide if we should fall back to web search.

    Triggers when:
    - No results at all, OR
    - Fewer than 3 results AND the best score is weak (< 40)
    """
    if not results:
        return True
    if len(results) < min(3, limit) and results[0].get("fit_score", 0) < 40:
        return True
    return False


def get_product(product_id: int) -> Optional[dict]:
    """Get full product details. Returns index fields + the product's data blob."""
    cached = _cache_get(f"product:{product_id}")
    if cached is not None:
        return cached

    db = get_db()
    resp = db.table("products").select(
        "*, businesses!inner(name, website)"
    ).eq("id", product_id).limit(1).execute()

    if not resp.data:
        return None

    row = resp.data[0]
    biz = row.get("businesses", {})

    result = {
        "id": row["id"],
        "name": row["name"],
        "business": biz.get("name", ""),
        "business_website": biz.get("website", ""),
        "description": row["description"],
        "url": row["url"],
        "category": row["category"],
        "subcategory": row["subcategory"],
        "price_model": row["price_model"],
        "price_min": row["price_min"],
        "price_max": row["price_max"],
        "has_free_tier": bool(row.get("has_free_tier")),
        "rating": row["rating"],
        "review_count": row["review_count"],
        "last_verified": row["last_verified"],
    }

    # The data blob — freeform JSON, shape varies per product
    data_blob = row.get("data", {})
    if isinstance(data_blob, str):
        try:
            data_blob = _json.loads(data_blob)
        except Exception:
            data_blob = {}

    # Surface MCP config at top level so agents can connect directly
    if data_blob.get("mcp"):
        result["mcp"] = data_blob["mcp"]

    # Strip internal fields from the public blob
    public_blob = {k: v for k, v in data_blob.items()
                   if k not in ("owner_token_hash", "pending_claim_code", "pending_claim_at")}
    result["data"] = public_blob

    _cache_set(f"product:{product_id}", result)
    return result


def compare_products(product_ids: list[int]) -> list[dict]:
    """Get full details for multiple products side-by-side."""
    db = get_db()
    resp = db.table("products").select(
        "*, businesses!inner(name, website)"
    ).in_("id", product_ids).execute()

    if not resp.data:
        return []

    results = []
    for row in resp.data:
        biz = row.get("businesses", {})
        item = {
            "id": row["id"],
            "name": row["name"],
            "business": biz.get("name", ""),
            "category": row["category"],
            "price_model": row["price_model"],
            "price_min": row["price_min"],
            "price_max": row["price_max"],
            "rating": row["rating"],
            "review_count": row["review_count"],
            "last_verified": row["last_verified"],
        }
        data_blob = row.get("data", {})
        if isinstance(data_blob, str):
            try:
                data_blob = _json.loads(data_blob)
            except Exception:
                data_blob = {}
        if data_blob.get("mcp"):
            item["mcp"] = data_blob["mcp"]
        item["data"] = {k: v for k, v in data_blob.items()
                        if k not in ("owner_token_hash", "pending_claim_code", "pending_claim_at")}
        results.append(item)

    return results


def get_categories() -> list[dict]:
    """Get all categories with product counts and subcategories."""
    cached = _cache_get("categories")
    if cached is not None:
        return cached

    db = get_db()
    resp = db.table("products").select(
        "category, subcategory"
    ).execute()

    if not resp.data:
        return []

    # Build category tree with counts
    cat_map: dict[str, dict] = {}
    for row in resp.data:
        cat = row.get("category") or "uncategorized"
        sub = row.get("subcategory")
        if cat not in cat_map:
            cat_map[cat] = {"category": cat, "count": 0, "subcategories": {}}
        cat_map[cat]["count"] += 1
        if sub:
            subs = cat_map[cat]["subcategories"]
            subs[sub] = subs.get(sub, 0) + 1

    # Format output
    results = []
    for cat in sorted(cat_map.keys()):
        entry = {
            "category": cat,
            "count": cat_map[cat]["count"],
        }
        if cat_map[cat]["subcategories"]:
            entry["subcategories"] = [
                {"name": s, "count": c}
                for s, c in sorted(cat_map[cat]["subcategories"].items(), key=lambda x: -x[1])
            ]
        results.append(entry)

    results.sort(key=lambda x: -x["count"])

    _cache_set("categories", results)
    return results


def get_alternatives(product_id: int, limit: int = 5) -> Optional[dict]:
    """Get alternatives for a product — same category, different product."""
    db = get_db()

    # First get the product to know its category
    prod_resp = db.table("products").select(
        "id, name, category, subcategory, price_min, price_max, rating, data"
    ).eq("id", product_id).limit(1).execute()

    if not prod_resp.data:
        return None

    product = prod_resp.data[0]
    cat = product.get("category")
    data = product.get("data") or {}

    # Check if the product's data blob has an alternatives list
    blob_alternatives = data.get("alternatives") or data.get("competitors") or []

    # Search same category, exclude this product
    q = db.table("products").select(_SELECT_COLS)
    if cat:
        q = q.eq("category", cat)
    q = q.neq("id", product_id)
    q = q.limit(limit * 2)

    resp = q.execute()
    rows = resp.data if resp.data else []

    # Score alternatives — prefer same subcategory, similar price range
    results = []
    for row in rows:
        compact = _row_to_compact(row)
        score = 50.0

        # Same subcategory bonus
        if row.get("subcategory") == product.get("subcategory"):
            score += 20

        # Mentioned in the product's alternatives list
        row_name = row.get("name", "").lower()
        if any(row_name in str(a).lower() for a in blob_alternatives):
            score += 25

        # Similar price range
        p_min = product.get("price_min")
        r_min = row.get("price_min")
        if p_min is not None and r_min is not None and p_min > 0:
            ratio = min(p_min, r_min) / max(p_min, r_min)
            score += ratio * 10

        # Rating bonus
        if row.get("rating"):
            score += (row["rating"] / 5.0) * 10

        compact["relevance_score"] = round(min(score, 100), 1)
        results.append(compact)

    results.sort(key=lambda r: r["relevance_score"], reverse=True)

    return {
        "product": {"id": product["id"], "name": product["name"], "category": cat},
        "alternatives": results[:limit],
    }


def get_product_count() -> int:
    """Get total product count (for health endpoint)."""
    cached = _cache_get("product_count")
    if cached is not None:
        return cached

    db = get_db()
    resp = db.table("products").select("id", count="exact").execute()
    count = resp.count if resp.count is not None else 0
    _cache_set("product_count", count)
    return count


def log_request(tool_name: str, query: str = None, product_ids: list[int] = None,
                latency_ms: int = None, client_id: str = None):
    """Log an API request for analytics. Best-effort, never fails the request."""
    try:
        db = get_db()
        db.table("api_usage").insert({
            "tool_name": tool_name,
            "query": query,
            "product_ids": product_ids,
            "latency_ms": latency_ms,
            "client_id": client_id,
        }).execute()
    except Exception:
        pass  # Never fail the request due to logging


def get_analytics(days: int = 7) -> dict:
    """Query api_usage table for analytics dashboard. Returns usage stats, top queries, funnels."""
    db = get_db()
    from datetime import timedelta

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()

    # Fetch all usage rows in the window
    resp = db.table("api_usage").select(
        "tool_name, query, product_ids, latency_ms, client_id, created_at"
    ).gte("created_at", cutoff).order("created_at", desc=True).limit(5000).execute()

    rows = resp.data if resp.data else []

    # --- Tool usage distribution ---
    tool_counts: dict[str, int] = {}
    for r in rows:
        t = r.get("tool_name") or "unknown"
        tool_counts[t] = tool_counts.get(t, 0) + 1

    # --- Top queries ---
    query_counts: dict[str, int] = {}
    for r in rows:
        q = (r.get("query") or "").strip().lower()
        if q:
            query_counts[q] = query_counts.get(q, 0) + 1
    top_queries = sorted(query_counts.items(), key=lambda x: -x[1])[:25]

    # --- Latency stats ---
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    latency_stats = {}
    if latencies:
        latencies_sorted = sorted(latencies)
        latency_stats = {
            "avg_ms": round(sum(latencies) / len(latencies), 1),
            "p50_ms": latencies_sorted[len(latencies_sorted) // 2],
            "p95_ms": latencies_sorted[int(len(latencies_sorted) * 0.95)],
            "max_ms": latencies_sorted[-1],
        }

    # --- Unique clients ---
    clients = {r.get("client_id") for r in rows if r.get("client_id")}

    # --- Discover → details conversion funnel ---
    # Group by client_id sessions (or just count overall)
    discover_count = tool_counts.get("discover", 0)
    details_count = tool_counts.get("details", 0)
    compare_count = tool_counts.get("compare", 0)
    alternatives_count = tool_counts.get("alternatives", 0)

    funnel = {
        "discover": discover_count,
        "details": details_count,
        "compare": compare_count,
        "alternatives": alternatives_count,
        "discover_to_details_pct": round(details_count / discover_count * 100, 1) if discover_count > 0 else 0,
        "discover_to_compare_pct": round(compare_count / discover_count * 100, 1) if discover_count > 0 else 0,
    }

    # --- Daily breakdown ---
    daily: dict[str, int] = {}
    for r in rows:
        day = (r.get("created_at") or "")[:10]
        if day:
            daily[day] = daily.get(day, 0) + 1
    daily_sorted = sorted(daily.items())

    # --- Most viewed products ---
    product_views: dict[int, int] = {}
    for r in rows:
        if r.get("tool_name") == "details" and r.get("product_ids"):
            for pid in r["product_ids"]:
                product_views[pid] = product_views.get(pid, 0) + 1
    top_products = sorted(product_views.items(), key=lambda x: -x[1])[:15]

    # Look up product names for top viewed
    top_product_details = []
    if top_products:
        pids = [p[0] for p in top_products]
        prod_resp = db.table("products").select("id, name").in_("id", pids).execute()
        name_map = {p["id"]: p["name"] for p in (prod_resp.data or [])}
        for pid, count in top_products:
            top_product_details.append({
                "product_id": pid,
                "name": name_map.get(pid, f"#{pid}"),
                "views": count,
            })

    # --- Session journeys ---
    # Group tool calls by client_id (session) to see agent behavior patterns
    sessions: dict[str, list[dict]] = {}
    for r in rows:
        cid = r.get("client_id")
        if cid:
            sessions.setdefault(cid, []).append(r)

    # Build session summaries
    session_summaries = []
    for sid, events in sessions.items():
        # Sort by time
        events.sort(key=lambda x: x.get("created_at", ""))
        tools_used = [e["tool_name"] for e in events if e.get("tool_name") not in ("session_start", "session_end")]
        queries = [e["query"] for e in events if e.get("query") and e.get("tool_name") == "discover"]

        # Find client type from session_start event
        client_type = "unknown"
        for e in events:
            if e.get("tool_name") == "session_start" and e.get("query"):
                client_type = e["query"]
                break

        started = events[0].get("created_at", "")
        ended = events[-1].get("created_at", "")

        if tools_used:  # Skip empty sessions
            session_summaries.append({
                "session_id": sid,
                "client_type": client_type,
                "started": started,
                "ended": ended,
                "tool_calls": len(tools_used),
                "journey": tools_used,
                "queries": queries,
            })

    session_summaries.sort(key=lambda x: x.get("started", ""), reverse=True)

    # --- Client type distribution ---
    client_types: dict[str, int] = {}
    for s in session_summaries:
        ct = s.get("client_type", "unknown")
        client_types[ct] = client_types.get(ct, 0) + 1

    return {
        "period_days": days,
        "total_requests": len(rows),
        "unique_sessions": len(sessions),
        "unique_clients": len(clients),
        "client_types": client_types,
        "tool_usage": tool_counts,
        "funnel": funnel,
        "top_queries": [{"query": q, "count": c} for q, c in top_queries],
        "top_products_viewed": top_product_details,
        "latency": latency_stats,
        "daily_requests": [{"date": d, "count": c} for d, c in daily_sorted],
        "recent_sessions": session_summaries[:20],
    }


def submit_feedback(
    feedback_type: str,
    message: str,
    query: str = None,
    expected: str = None,
    actual: str = None,
    agent_id: str = None,
):
    """Store agent feedback for search quality improvement. Best-effort."""
    try:
        db = get_db()
        db.table("feedback").insert({
            "feedback_type": feedback_type,
            "tool_name": "discover",
            "query": query,
            "expected": expected,
            "actual": actual,
            "message": message[:2000] if message else None,
            "agent_id": agent_id,
        }).execute()
    except Exception:
        pass  # Never fail the agent's request due to feedback storage


# ---------- Upsert helpers (used by crawler/seed scripts) ----------

# Index columns that go directly into the products table row
_INDEX_FIELDS = {
    "name", "slug", "short_description", "description", "category", "subcategory",
    "price_model", "price_min", "price_max", "price_unit", "has_free_tier",
    "rating", "review_count", "data_quality", "team_size_min", "team_size_max",
    "url", "last_verified", "source", "source_id",
}


def upsert_business(name: str, slug: str, website: str = None, **kwargs) -> int:
    """Insert or update a business. Returns the business ID."""
    db = get_db()

    # Check if exists
    resp = db.table("businesses").select("id").eq("slug", slug).limit(1).execute()
    if resp.data:
        bid = resp.data[0]["id"]
        update_data = {"name": name}
        if website:
            update_data["website"] = website
        # Merge any extra fields
        for k, v in kwargs.items():
            update_data[k] = v
        db.table("businesses").update(update_data).eq("id", bid).execute()
        return bid
    else:
        insert_data = {"name": name, "slug": slug}
        if website:
            insert_data["website"] = website
        for k, v in kwargs.items():
            insert_data[k] = v
        resp = db.table("businesses").insert(insert_data).execute()
        return resp.data[0]["id"]


def upsert_product(business_id: int, slug: str, name: str, **kwargs) -> int:
    """Insert or update a product. Separates index fields from blob data automatically.

    Any kwarg that matches an index column goes into the row directly.
    Everything else goes into the `data` JSONB blob.
    """
    db = get_db()

    # Separate index fields from blob data
    row_data = {"business_id": business_id, "slug": slug, "name": name}
    blob_data = {}

    for k, v in kwargs.items():
        if k in _INDEX_FIELDS:
            row_data[k] = v
        elif k == "data":
            # Explicit data blob passed — merge it
            if isinstance(v, dict):
                blob_data.update(v)
        else:
            blob_data[k] = v

    if blob_data:
        row_data["data"] = blob_data

    # Check if exists
    resp = db.table("products").select("id").eq("slug", slug).eq("business_id", business_id).limit(1).execute()
    if resp.data:
        pid = resp.data[0]["id"]
        db.table("products").update(row_data).eq("id", pid).execute()
        return pid
    else:
        resp = db.table("products").insert(row_data).execute()
        return resp.data[0]["id"]


# ---------- Self-service registration (agents-first) ----------

def _hash_token(token: str) -> str:
    """SHA-256 hash of an owner token. We never store the raw token."""
    return hashlib.sha256(token.encode()).hexdigest()


def _slugify(name: str) -> str:
    """Convert a product name to a URL-safe slug."""
    s = name.lower().strip()
    s = re.sub(r'[^a-z0-9]+', '-', s)
    return s.strip('-')[:120]


def register_product(
    name: str,
    description: str,
    url: str,
    category: str = "development",
    subcategory: str = None,
    source: str = "community",
    mcp_config: dict = None,
    agent_services: list = None,
    **extra,
) -> dict:
    """Register a new product/tool. Returns product_id + owner_token.

    The owner_token is the ONLY way to update or manage the listing.
    The agent MUST store it — we only store the hash.
    """
    db = get_db()
    slug = _slugify(name)

    # Check for duplicate slug
    resp = db.table("products").select("id").eq("slug", slug).limit(1).execute()
    if resp.data:
        return {"error": f"A product with slug '{slug}' already exists. Use 'claim' to take ownership of an existing listing."}

    # Generate owner token
    owner_token = f"sylex_{secrets.token_urlsafe(32)}"
    token_hash = _hash_token(owner_token)

    # Create or get business for this registration
    # Extract domain from URL for business slug
    biz_slug = slug
    biz_name = name
    biz_website = url
    business_id = upsert_business(name=biz_name, slug=f"community-{biz_slug}", website=biz_website)

    # Build data blob
    blob = {
        "owner_token_hash": token_hash,
        "registered_by": "agent",
        "registered_at": datetime.utcnow().isoformat(),
    }
    if mcp_config:
        blob["mcp"] = mcp_config
    if agent_services:
        blob["agent_services"] = agent_services
    for k, v in extra.items():
        blob[k] = v

    row = {
        "name": name,
        "slug": slug,
        "business_id": business_id,
        "description": description[:500] if description else None,
        "short_description": description[:120] if description else None,
        "url": url,
        "category": category,
        "source": source,
        "data": blob,
    }
    if subcategory:
        row["subcategory"] = subcategory

    resp = db.table("products").insert(row).execute()
    product_id = resp.data[0]["id"]

    return {
        "product_id": product_id,
        "slug": slug,
        "owner_token": owner_token,
        "message": "Registration successful. STORE YOUR OWNER TOKEN — you need it to update this listing. If you lose it, you can re-verify ownership via the 'claim' tool to get a new one.",
    }


def verify_owner(product_id: int, owner_token: str) -> tuple[bool, dict]:
    """Verify that an owner_token matches the stored hash for a product.
    Returns (is_valid, product_row).
    """
    db = get_db()
    resp = db.table("products").select("id, name, slug, data").eq("id", product_id).limit(1).execute()
    if not resp.data:
        return False, {"error": "Product not found."}

    product = resp.data[0]
    blob = product.get("data") or {}
    stored_hash = blob.get("owner_token_hash")

    if not stored_hash:
        return False, {"error": "This product has no owner. Use 'claim' to take ownership."}

    if _hash_token(owner_token) != stored_hash:
        return False, {"error": "Invalid owner token."}

    return True, product


def update_product_listing(product_id: int, owner_token: str, updates: dict) -> dict:
    """Update a product listing. Requires valid owner_token.

    Allowed updates: name, description, short_description, url, category,
    subcategory, price_model, has_free_tier, and any data-blob keys.
    """
    valid, result = verify_owner(product_id, owner_token)
    if not valid:
        return result

    db = get_db()
    allowed_index = {"name", "description", "short_description", "url", "category",
                     "subcategory", "price_model", "price_min", "price_max",
                     "has_free_tier"}

    row_updates = {}
    blob_updates = {}

    for k, v in updates.items():
        if k in ("owner_token", "owner_token_hash", "id", "slug"):
            continue  # Never allow these to be changed
        if k in allowed_index:
            row_updates[k] = v
        else:
            blob_updates[k] = v

    # Merge blob updates into existing data
    if blob_updates:
        existing_blob = result.get("data") or {}
        existing_blob.update(blob_updates)
        row_updates["data"] = existing_blob

    if not row_updates:
        return {"error": "No valid fields to update."}

    db.table("products").update(row_updates).eq("id", product_id).execute()

    return {"status": "updated", "product_id": product_id, "fields_updated": list(updates.keys())}


def set_mcp_config(product_id: int, owner_token: str, mcp_config: dict) -> dict:
    """Set MCP connection config on a product listing.

    mcp_config should contain transport details so other agents can connect:
    - For SSE: {"transport": "sse", "url": "https://..."}
    - For stdio: {"transport": "stdio", "command": "npx", "args": ["package-name"]}
    - For streamable-http: {"transport": "streamable-http", "url": "https://..."}
    """
    valid, result = verify_owner(product_id, owner_token)
    if not valid:
        return result

    db = get_db()
    existing_blob = result.get("data") or {}
    existing_blob["mcp"] = mcp_config
    existing_blob["mcp_updated_at"] = datetime.utcnow().isoformat()

    db.table("products").update({"data": existing_blob}).eq("id", product_id).execute()

    return {"status": "mcp_config_set", "product_id": product_id, "transport": mcp_config.get("transport", "unknown")}


def generate_claim_code(product_id: int) -> dict:
    """Generate a verification code for claiming an existing product.

    Returns a code the agent must place at a verifiable URL (npm package.json,
    GitHub repo description, DNS TXT record, or /.well-known/sylex-verify).
    """
    db = get_db()
    resp = db.table("products").select("id, name, slug, url, data").eq("id", product_id).limit(1).execute()
    if not resp.data:
        return {"error": "Product not found."}

    product = resp.data[0]
    blob = product.get("data") or {}
    is_reclaim = bool(blob.get("owner_token_hash"))

    # Generate claim code (short, easy to place)
    claim_code = f"sylex-verify-{secrets.token_hex(8)}"

    # Store it temporarily in the blob
    blob["pending_claim_code"] = claim_code
    blob["pending_claim_at"] = datetime.utcnow().isoformat()
    db.table("products").update({"data": blob}).eq("id", product_id).execute()

    product_url = product.get("url") or ""

    verification_methods = []
    if "github.com" in product_url:
        verification_methods.append({
            "method": "github",
            "instruction": f"Add '{claim_code}' to the repo description, About section, or a file named SYLEX_VERIFY in the repo root.",
        })
    if "npmjs.com" in product_url or blob.get("source") == "npm":
        verification_methods.append({
            "method": "npm",
            "instruction": f"Add '\"sylexVerify\": \"{claim_code}\"' to your package.json and publish.",
        })
    # Always offer website verification
    verification_methods.append({
        "method": "website",
        "instruction": f"Serve '{claim_code}' at {{your-domain}}/.well-known/sylex-verify (plain text response).",
    })
    verification_methods.append({
        "method": "dns",
        "instruction": f"Add a TXT record '_sylex-verify.{{your-domain}}' with value '{claim_code}'.",
    })

    result = {
        "product_id": product_id,
        "product_name": product.get("name"),
        "claim_code": claim_code,
        "verification_methods": verification_methods,
    }

    if is_reclaim:
        result["message"] = (
            "This product is already claimed. Re-verifying ownership will "
            "rotate the token — the old owner_token will stop working and "
            "a new one will be issued. Place the claim_code using any method "
            "above, then call 'claim' with verification_url to complete."
        )
        result["is_reclaim"] = True
    else:
        result["message"] = (
            "Place the claim_code using any method above, then call 'claim' "
            "with verification_url to complete."
        )

    return result


def _validate_url_safe(url: str) -> tuple[bool, str, str]:
    """Validate a URL is safe to fetch (SSRF protection).

    Returns (is_safe, error_message, resolved_ip). The resolved_ip is the
    validated IP address to connect to, eliminating DNS rebinding attacks
    by pinning the IP from validation through to the actual fetch.
    """
    from urllib.parse import urlparse

    # Block overly long URLs early (potential abuse)
    if len(url) > 2048:
        return False, "URL is too long (max 2048 characters).", ""

    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL format.", ""

    # Scheme must be HTTPS
    if parsed.scheme != "https":
        return False, "Only HTTPS URLs are allowed for verification.", ""

    # Must have a hostname
    hostname = parsed.hostname
    if not hostname:
        return False, "URL must include a hostname.", ""

    # Block obvious internal hostnames
    blocked_hosts = {
        "localhost", "127.0.0.1", "0.0.0.0", "::1",
        "metadata.google.internal", "metadata.google.com",
    }
    hostname_lower = hostname.lower()
    if hostname_lower in blocked_hosts:
        return False, "Internal/private URLs are not allowed.", ""

    # Block Railway internal networking and common cloud metadata
    if hostname_lower.endswith(".railway.internal"):
        return False, "Internal/private URLs are not allowed.", ""
    if hostname_lower.endswith(".internal"):
        return False, "Internal/private URLs are not allowed.", ""

    # Resolve the hostname and check for private IPs
    resolved_ip = ""
    try:
        # Resolve all IPs for the hostname
        addrinfos = socket.getaddrinfo(hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
        for family, _, _, _, sockaddr in addrinfos:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False, "URL resolves to a private/internal IP address.", ""
            # Block AWS/GCP/Azure metadata IPs explicitly
            if ip == ipaddress.ip_address("169.254.169.254"):
                return False, "Cloud metadata endpoints are not allowed.", ""
        # Pin the first resolved IP for use in the actual fetch
        if addrinfos:
            resolved_ip = addrinfos[0][4][0]
    except socket.gaierror:
        return False, f"Could not resolve hostname: {hostname}", ""

    if not resolved_ip:
        return False, "Could not resolve any IP for hostname.", ""

    return True, "", resolved_ip


def _fetch_url_pinned(url: str, resolved_ip: str, timeout: int = 10) -> str:
    """Fetch a URL using a pre-resolved IP to prevent DNS rebinding.

    Connects directly to the validated IP while sending the original
    hostname in the Host header and SNI for TLS. This closes the
    TOCTOU gap where DNS could rebind between validation and fetch.
    """
    import ssl
    from urllib.parse import urlparse

    parsed = urlparse(url)
    hostname = parsed.hostname
    port = parsed.port or 443
    path = parsed.path or "/"
    if parsed.query:
        path += f"?{parsed.query}"

    # Create a socket connection to the pinned IP
    is_ipv6 = ":" in resolved_ip
    af = socket.AF_INET6 if is_ipv6 else socket.AF_INET
    sock = socket.create_connection((resolved_ip, port), timeout=timeout)

    # Wrap with TLS, using the original hostname for SNI and cert verification
    ctx = ssl.create_default_context()
    ssl_sock = ctx.wrap_socket(sock, server_hostname=hostname)

    try:
        # Build HTTP/1.1 request manually
        request_line = f"GET {path} HTTP/1.1\r\n"
        headers = (
            f"Host: {hostname}\r\n"
            f"User-Agent: SylexSearch/1.0 ClaimVerifier\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )
        ssl_sock.sendall((request_line + headers).encode())

        # Read response
        response = b""
        while True:
            chunk = ssl_sock.recv(8192)
            if not chunk:
                break
            response += chunk
            if len(response) > 60_000:  # Safety cap
                break

        # Parse: split headers from body
        response_str = response.decode("utf-8", errors="replace")
        if "\r\n\r\n" in response_str:
            body = response_str.split("\r\n\r\n", 1)[1]
        else:
            body = response_str

        return body[:50_000]
    finally:
        ssl_sock.close()


def verify_claim(product_id: int, verification_url: str) -> dict:
    """Verify a claim by checking if the claim_code exists at the given URL.

    Returns product_id + owner_token on success.
    """
    db = get_db()
    resp = db.table("products").select("id, name, slug, url, data").eq("id", product_id).limit(1).execute()
    if not resp.data:
        return {"error": "Product not found."}

    product = resp.data[0]
    blob = product.get("data") or {}
    claim_code = blob.get("pending_claim_code")

    if not claim_code:
        return {"error": "No pending claim for this product. Call 'claim' first to get a verification code."}

    # SSRF protection: validate URL and pin the resolved IP
    url_safe, url_error, resolved_ip = _validate_url_safe(verification_url)
    if not url_safe:
        return {"error": f"Invalid verification URL: {url_error}"}

    # Fetch using the pinned IP (prevents DNS rebinding TOCTOU attacks)
    try:
        body = _fetch_url_pinned(verification_url, resolved_ip)
    except Exception as e:
        return {"error": f"Could not fetch verification URL: {str(e)}"}

    if claim_code not in body:
        return {"error": f"Claim code not found at {verification_url}. Make sure '{claim_code}' appears in the response."}

    # Verification passed — issue owner token
    owner_token = f"sylex_{secrets.token_urlsafe(32)}"
    token_hash = _hash_token(owner_token)

    blob["owner_token_hash"] = token_hash
    blob["claimed_at"] = datetime.utcnow().isoformat()
    blob["claimed_via"] = verification_url
    # Clean up claim state
    blob.pop("pending_claim_code", None)
    blob.pop("pending_claim_at", None)

    db.table("products").update({"data": blob}).eq("id", product_id).execute()

    return {
        "status": "claimed",
        "product_id": product_id,
        "product_name": product.get("name"),
        "owner_token": owner_token,
        "message": "Claim verified! STORE YOUR OWNER TOKEN — you need it for updates. Lost it? Re-verify via 'claim' anytime to get a new one (old token stops working).",
    }


# ---------- Agent services discovery ----------

def search_agent_services(service_type: str = "all", limit: int = 10) -> list[dict]:
    """Find products that provide agent infrastructure services.

    Searches the data blob for products with agent_services declarations,
    optionally filtered by service_type (memory, auth, billing, etc.).
    """
    cache_key = f"agent_services:{service_type}:{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    db = get_db()

    # Fetch products that have agent_services in their data blob
    # Supabase JSONB: filter for non-null agent_services
    q = db.table("products").select(
        "id, name, slug, short_description, description, category, url, data"
    ).not_.is_("data->agent_services", "null").limit(200)

    resp = q.execute()
    rows = resp.data if resp.data else []

    results = []
    for row in rows:
        data_blob = row.get("data") or {}
        if isinstance(data_blob, str):
            try:
                data_blob = _json.loads(data_blob)
            except Exception:
                continue

        services = data_blob.get("agent_services", [])
        if not services:
            continue

        for svc in services:
            svc_type = svc.get("service_type", "")
            if service_type != "all" and svc_type != service_type:
                continue

            results.append({
                "product_id": row["id"],
                "product_name": row["name"],
                "product_url": row.get("url"),
                "service_type": svc_type,
                "capabilities": svc.get("capabilities", []),
                "auth_method": svc.get("auth_method", "none"),
                "pricing": svc.get("pricing", "unknown"),
                "description": svc.get("description", ""),
                "mcp_config": svc.get("mcp_config"),
            })

    results = results[:limit]
    _cache_set(cache_key, results)
    return results


# ---------- Scoring helpers ----------

# Well-known products that should rank above obscure alternatives.
# These are household-name products that Wikidata entries lack popularity
# signals for. Values are added to the popularity score before log scaling.
# Even a small boost here is enough to break ties among dq=1.0 entries.
_NOTABLE_PRODUCTS = {
    # Databases
    "postgresql": 10_000_000, "mongodb": 10_000_000, "mysql": 10_000_000,
    "redis": 5_000_000, "sqlite": 5_000_000, "oracle database": 5_000_000,
    "microsoft sql server": 5_000_000, "elasticsearch": 3_000_000,
    "cassandra": 2_000_000, "neo4j": 1_000_000, "influxdb": 1_000_000,
    "clickhouse": 1_000_000, "couchdb": 500_000, "mariadb": 3_000_000,
    # ML / AI
    "tensorflow": 10_000_000, "pytorch": 10_000_000, "scikit-learn": 5_000_000,
    "keras": 5_000_000, "transformers": 5_000_000, "numpy": 10_000_000,
    "pandas": 10_000_000, "scipy": 5_000_000, "opencv": 3_000_000,
    # Project management / SaaS
    "jira": 5_000_000, "notion": 5_000_000, "asana": 3_000_000,
    "linear": 2_000_000, "trello": 3_000_000, "github": 10_000_000,
    "gitlab": 5_000_000, "slack": 5_000_000, "salesforce": 5_000_000,
    # Dev tools
    "docker": 10_000_000, "kubernetes": 10_000_000, "terraform": 5_000_000,
    "ansible": 3_000_000, "jenkins": 3_000_000, "grafana": 3_000_000,
    "prometheus": 3_000_000, "nginx": 5_000_000, "apache": 5_000_000,
    "webpack": 5_000_000, "vite": 3_000_000, "git": 10_000_000,
    # Languages / runtimes
    "python": 10_000_000, "javascript": 10_000_000, "typescript": 5_000_000,
    "rust": 5_000_000, "go": 5_000_000, "java": 10_000_000,
    # CLI tools
    "click": 1_000_000, "typer": 500_000, "commander": 500_000,
    "yargs": 1_000_000, "clap": 1_000_000,
    # State management
    "mobx": 1_000_000, "jotai": 500_000, "zustand": 1_000_000,
    "redux": 3_000_000,
}

def _compute_fit_score(
    row,
    max_price: Optional[float] = None,
    team_size: Optional[int] = None,
    implied_source: Optional[str] = None,
    query_words: Optional[list[str]] = None,
) -> float:
    """Compute a 0-100 fit score based on how well a product matches constraints.

    Scoring breakdown (max ~100):
      - Exact name match:     0-25 pts  (query IS the product name)
      - Term match relevance: 0-15 pts  (query words in name/desc)
      - Popularity:           0-20 pts  (downloads, stars, reviews — from data blob)
      - Data quality:         0-8 pts   (completeness of record)
      - Source boost:         -10 to +15 pts  (query implies a source)
      - Price match:          0-10 pts  (within budget)
      - Team size match:      0-7 pts   (fits team)
    """
    score = 15.0
    name_lower = (row.get("name") or "").lower().strip()
    data_blob = row.get("data") or {}
    subcategory = (row.get("subcategory") or "").lower()
    category = (row.get("category") or "").lower()

    # --- Exact/partial name match (0-25 points) ---
    # This is the single most important signal: if someone searches "numpy" they want numpy.
    # But for category queries like "database", we reduce the "contains" bonus —
    # "DataBase Oasis" having "database" in its name shouldn't outrank PostgreSQL.
    if query_words:
        query_joined = " ".join(query_words)
        name_normalized = re.sub(r'[^a-z0-9]', '', name_lower)
        query_normalized = re.sub(r'[^a-z0-9]', '', query_joined)
        # Check if query matches a known subcategory (category-style query)
        is_category_query = bool(_infer_subcategories(query_joined))
        if name_normalized == query_normalized:
            score += 25  # Exact match always strong
        elif not is_category_query:
            # Only award partial name matches for product-name queries, not category queries.
            # "database" in "DataBase Oasis" shouldn't outrank PostgreSQL.
            if name_normalized.startswith(query_normalized) or query_normalized.startswith(name_normalized):
                score += 18
            elif query_normalized in name_normalized:
                score += 12

    # --- Subcategory/category match (0-20 points) ---
    # "database" query → subcategory="database" is a near-perfect semantic match
    # This is critical because FTS returns many loosely-related results
    if query_words:
        query_joined = " ".join(query_words)
        query_hyphenated = "-".join(query_words)
        # Check subcategory match (strongest signal)
        sub_normalized = subcategory.replace("-", " ").replace("_", " ")
        if query_joined == sub_normalized or query_hyphenated == subcategory:
            score += 20
        elif query_joined in sub_normalized or sub_normalized in query_joined:
            score += 14
        elif any(w in subcategory.split("-") for w in query_words):
            score += 8
        # Category match (weaker)
        elif any(w in category.split("-") for w in query_words if len(w) >= 3):
            score += 5

    # --- Term match relevance (0-15 points) ---
    # WHERE the match occurs matters a lot. Name/subcategory match >> description match.
    if query_words:
        name_plus_sub = " ".join([name_lower, subcategory.replace("-", " "), category.replace("-", " ")])
        summary = (row.get("short_description") or "").lower()
        full_text = " ".join([name_plus_sub, summary, (row.get("description") or "").lower()])

        # Count matches in different zones
        name_matches = sum(1 for w in query_words if w in name_plus_sub)
        total_matches = sum(1 for w in query_words if w in full_text)

        if len(query_words) > 0:
            # Name/subcategory matches are worth 3x description-only matches
            name_ratio = name_matches / len(query_words)
            total_ratio = total_matches / len(query_words)
            score += name_ratio * 12 + total_ratio * 3

    # --- Popularity from data blob (0-20 points) ---
    # Each source stores popularity differently — extract and normalize
    popularity = _extract_popularity(row.get("source"), data_blob, row)
    if popularity > 0:
        # Log scale: 100 → ~5pts, 10k → ~10pts, 1M → ~15pts, 100M → ~20pts
        score += min(math.log10(max(popularity, 1)) / 4.0, 1.0) * 20

    # --- Notability boost (0-10 points) ---
    # Well-known products get extra points to break ties among entries
    # with identical data quality and popularity baselines.
    notable = _NOTABLE_PRODUCTS.get(name_lower, 0)
    if notable > 0:
        # 500K → ~3pts, 1M → ~4pts, 5M → ~7pts, 10M → ~10pts
        score += min(math.log10(notable) / 7.0, 1.0) * 10

    # --- Data quality (0-8 points) ---
    dq = row.get("data_quality") or 0.5
    score += dq * 8

    # --- Source boost (0-15 or -10 points) ---
    if implied_source:
        if row.get("source") == implied_source:
            score += 15
        else:
            score -= 10

    # --- Price match (0-10 points) ---
    price_min = row.get("price_min")
    if max_price is not None and price_min is not None:
        if price_min == 0:
            score += 10
        elif price_min <= max_price:
            ratio = 1 - (price_min / max_price)
            score += ratio * 7 + 3

    # --- Team size match (0-7 points) ---
    if team_size is not None:
        ts_min = row.get("team_size_min") or 1
        ts_max = row.get("team_size_max") or 100000
        if ts_min <= team_size <= ts_max:
            range_size = ts_max - ts_min
            if range_size > 0:
                position = (team_size - ts_min) / range_size
                if 0.1 <= position <= 0.7:
                    score += 7
                else:
                    score += 4

    return round(min(score, 100), 1)


def _extract_popularity(source: str, data: dict, row: dict) -> float:
    """Extract a popularity number from a product's data blob.

    Different sources store popularity differently:
      - npm: downloads_last_week in data blob
      - github: stars in data blob
      - crates: downloads or recent_downloads in data blob
      - pypi: no download stats yet, use review_count fallback
      - wikidata: well-known products get a baseline popularity
      - saas: well-known products get a baseline popularity
    """
    if not source:
        source = ""

    if source == "npm":
        # npm_score is a composite score (quality * popularity * maintenance)
        # Ranges roughly 0-200+, with popular packages >100
        npm_score = data.get("npm_score", 0) or 0
        if npm_score > 0:
            # Scale npm_score to be comparable with download counts
            # 150+ = extremely popular (react, express) → ~1M equivalent
            # 100-150 = popular → ~100K equivalent
            # 50-100 = moderate → ~10K equivalent
            return 10 ** (npm_score / 50)
        return data.get("downloads_last_week", 0) or 0

    if source == "github":
        return data.get("stars", 0) or 0

    if source == "crates":
        # recent_downloads is more relevant than all-time
        return data.get("recent_downloads", 0) or data.get("downloads", 0) or 0

    if source == "pypi":
        # PyPI data blob may have downloads; fall back to review_count
        downloads = data.get("downloads_last_week", 0) or data.get("downloads", 0) or 0
        if downloads > 0:
            return downloads
        rc = row.get("review_count", 0) or 0
        if rc > 0:
            return rc
        # Small baseline — PyPI packages without download data shouldn't
        # outrank packages that DO have popularity signals
        return 100

    if source == "saas":
        # SaaS products in our index are curated — they're all well-known
        rc = row.get("review_count", 0) or 0
        if rc > 0:
            return rc
        # Baseline: curated SaaS products are popular by definition
        dq = row.get("data_quality", 0) or 0
        return 50000 if dq >= 0.9 else 10000

    if source == "wikidata":
        # Wikidata entries are major, well-known software products
        # They have no download stats but are canonical — give them a
        # strong baseline so they don't get buried by niche packages
        dq = row.get("data_quality", 0) or 0
        return 100000 if dq >= 0.9 else 50000

    # Default: try review_count, then a small baseline
    return row.get("review_count", 0) or 100


def _deduplicate(results: list[dict]) -> list[dict]:
    """Remove duplicate entries across sources. Keep the highest-scored version.

    Deduplication is name-based (case-insensitive). When the same tool appears
    in npm, PyPI, GitHub, and SaaS, we keep the one with the best fit score.
    """
    seen_names: dict[str, int] = {}  # lowercase name -> index in output
    output = []

    for r in results:
        name_key = r["name"].lower().strip()
        # Also check without common prefixes/suffixes
        clean_key = re.sub(r'^(npm-|pypi-|crates-|github-)', '', name_key)

        # Check both the raw name and cleaned version
        match_key = None
        if name_key in seen_names:
            match_key = name_key
        elif clean_key in seen_names:
            match_key = clean_key

        if match_key is not None:
            # Already seen — keep higher scored one
            existing_idx = seen_names[match_key]
            if r["fit_score"] > output[existing_idx]["fit_score"]:
                output[existing_idx] = r
        else:
            seen_names[name_key] = len(output)
            seen_names[clean_key] = len(output)
            output.append(r)

    return output


def _row_to_compact(row) -> dict:
    """Minimal index representation — for discover results. ~70 tokens per result."""
    biz = row.get("businesses", {})
    price_min = row.get("price_min")
    price_max = row.get("price_max")

    price = ""
    if row.get("price_model") == "free":
        price = "Free"
    elif price_min is not None:
        price = f"${price_min}"
        if price_max and price_max != price_min:
            price += f"-${price_max}"

    result = {
        "id": row["id"],
        "name": row["name"],
        "business": biz.get("name", ""),
        "summary": row.get("short_description") or (row.get("description") or "")[:100],
        "price": price,
        "rating": row.get("rating"),
        "category": row.get("category"),
    }
    if row.get("subcategory"):
        result["subcategory"] = row["subcategory"]
    if row.get("has_free_tier"):
        result["has_free_tier"] = True
    if row.get("last_verified"):
        result["last_verified"] = row["last_verified"]
    if row.get("source") and row["source"] != "saas":
        result["source"] = row["source"]
    # Flag if this product has MCP connection config (agents can connect directly)
    data_blob = row.get("data") or {}
    if isinstance(data_blob, str):
        try:
            data_blob = _json.loads(data_blob)
        except Exception:
            data_blob = {}
    if data_blob.get("mcp"):
        result["has_mcp"] = True
        mcp = data_blob["mcp"]
        if mcp.get("transport"):
            result["mcp_transport"] = mcp["transport"]
    # Surface agent infrastructure services
    if data_blob.get("agent_services"):
        result["has_agent_services"] = True
        result["agent_service_types"] = [
            s.get("service_type") for s in data_blob["agent_services"]
            if s.get("service_type")
        ]
    return result
