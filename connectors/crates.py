#!/usr/bin/env python3
"""crates.io connector — pulls Rust packages into the index.

Uses the crates.io API to fetch popular Rust crates.

Usage:
    python3 connectors/crates.py --max 500
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from typing import Optional

import requests

from base import BaseConnector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("crates-connector")

CRATES_API = "https://crates.io/api/v1"

# Keyword to category mapping for Rust crates
KEYWORD_TO_CATEGORY = {
    "async": ("development", "async"),
    "web": ("development", "web"),
    "http": ("development", "http"),
    "api": ("development", "api"),
    "cli": ("development", "cli"),
    "database": ("development", "database"),
    "orm": ("development", "database"),
    "sql": ("development", "database"),
    "crypto": ("security", "cryptography"),
    "security": ("security", "security"),
    "auth": ("security", "authentication"),
    "test": ("testing", "testing-framework"),
    "testing": ("testing", "testing-framework"),
    "parser": ("development", "parser"),
    "serialization": ("data", "serialization"),
    "serde": ("data", "serialization"),
    "json": ("data", "data-processing"),
    "encoding": ("data", "encoding"),
    "gui": ("development", "gui"),
    "graphics": ("development", "graphics"),
    "game": ("development", "game-dev"),
    "network": ("communication", "networking"),
    "concurrency": ("development", "concurrency"),
    "embedded": ("development", "embedded"),
    "wasm": ("development", "webassembly"),
    "no-std": ("development", "embedded"),
    "machine-learning": ("ai-ml", "machine-learning"),
    "science": ("data", "scientific"),
    "math": ("data", "math"),
    "logging": ("development", "logging"),
    "error": ("development", "error-handling"),
    "config": ("development", "configuration"),
    "filesystem": ("development", "filesystem"),
    "os": ("development", "operating-system"),
    "ffi": ("development", "ffi"),
    "command-line": ("development", "cli"),
    "template": ("development", "template-engine"),
}


class CratesConnector(BaseConnector):
    SOURCE = "crates"
    BATCH_SIZE = 100  # crates.io max per_page

    def __init__(self):
        super().__init__()
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "sylex-search-connector/1.0 (https://github.com/MastadoonPrime/sylex-search)",
            "Accept": "application/json",
        })
        self._page = 1

    def fetch_batch(self, offset: int, limit: int) -> list[dict]:
        """Fetch popular crates sorted by downloads."""
        page = (offset // 100) + 1
        try:
            resp = self._session.get(
                f"{CRATES_API}/crates",
                params={
                    "page": page,
                    "per_page": min(limit, 100),
                    "sort": "downloads",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            crates = data.get("crates", [])
            time.sleep(1)  # Be gentle
            return crates

        except Exception as e:
            log.warning("crates.io fetch error: %s", e)
            time.sleep(5)
            return []

    def normalize(self, raw: dict) -> Optional[dict]:
        """Normalize a crates.io crate into our schema."""
        name = raw.get("name") or raw.get("id", "")
        if not name:
            return None

        description = raw.get("description") or ""
        homepage = raw.get("homepage") or ""
        repository = raw.get("repository") or ""
        documentation = raw.get("documentation") or ""
        downloads = raw.get("downloads", 0)
        recent_downloads = raw.get("recent_downloads", 0)
        max_version = raw.get("max_version") or raw.get("newest_version", "")
        keywords = raw.get("keywords", []) or []
        categories_raw = raw.get("categories", []) or []
        license_name = raw.get("license") or ""
        created = raw.get("created_at", "")
        updated = raw.get("updated_at", "")

        # Determine category from keywords
        category = "development"
        subcategory = "rust-crate"
        for kw in keywords:
            kw_lower = kw.lower().strip()
            if kw_lower in KEYWORD_TO_CATEGORY:
                category, subcategory = KEYWORD_TO_CATEGORY[kw_lower]
                break

        # Build slug
        slug = re.sub(r'[^a-z0-9-]', '-', name.lower())
        slug = re.sub(r'-+', '-', slug).strip("-")

        # Owner from repository URL
        owner = ""
        if repository and "github.com" in repository:
            parts = repository.rstrip("/").split("/")
            if len(parts) >= 4:
                owner = parts[3]

        biz_slug = owner or slug
        biz_slug = re.sub(r'[^a-z0-9-]', '-', biz_slug.lower())
        biz_slug = re.sub(r'-+', '-', biz_slug).strip("-")

        # Data quality
        quality = 0.4
        if description:
            quality += 0.15
        if downloads > 10000:
            quality += 0.1
        if downloads > 1000000:
            quality += 0.1
        if repository:
            quality += 0.05
        if documentation:
            quality += 0.05
        if keywords:
            quality += 0.05
        if license_name:
            quality += 0.05

        return {
            "name": name,
            "slug": slug,
            "source_id": name,
            "business_name": owner or name,
            "business_slug": biz_slug,
            "business_website": homepage or repository or None,
            "short_description": description[:200] if description else f"Rust crate: {name}",
            "description": description,
            "category": category,
            "subcategory": subcategory,
            "price_model": "free",
            "price_min": 0,
            "has_free_tier": True,
            "url": f"https://crates.io/crates/{name}",
            "data_quality": round(min(quality, 1.0), 2),
            "data": {
                "version": max_version,
                "downloads": downloads,
                "recent_downloads": recent_downloads,
                "license": license_name,
                "keywords": keywords[:15],
                "categories": categories_raw[:10],
                "repository": repository or None,
                "documentation": documentation or None,
                "homepage": homepage or None,
                "created_at": created[:10] if created else None,
                "updated_at": updated[:10] if updated else None,
                "type": "rust-crate",
            },
        }


def main():
    parser = argparse.ArgumentParser(description="crates.io connector for Agent Commerce")
    parser.add_argument("--max", type=int, default=500, help="Maximum crates to fetch")
    parser.add_argument("--offset", type=int, default=0, help="Start offset")
    args = parser.parse_args()

    connector = CratesConnector()
    connector.run(max_items=args.max, offset=args.offset)


if __name__ == "__main__":
    main()
