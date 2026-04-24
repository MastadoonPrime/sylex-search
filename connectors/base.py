"""Base connector interface for data sources.

Each connector pulls structured data from an external source and normalizes
it into the products table schema. Connectors are deterministic — no LLM
calls needed.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

# Add src to path for db imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from db import init_db, get_db, upsert_business, upsert_product

log = logging.getLogger("connector")


class BaseConnector(ABC):
    """Base class for all data source connectors."""

    SOURCE: str = ""  # e.g. "npm", "pypi", "wikidata"
    BATCH_SIZE: int = 100

    def __init__(self):
        init_db()
        self.db = get_db()
        self.stats = {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

    @abstractmethod
    def fetch_batch(self, offset: int, limit: int) -> list[dict]:
        """Fetch a batch of raw items from the source.

        Returns a list of source-native dicts. Each connector defines
        its own fetch logic (API calls, file reads, etc).
        """
        ...

    @abstractmethod
    def normalize(self, raw: dict) -> Optional[dict]:
        """Normalize a raw source item into our schema.

        Returns a dict with:
            - business_name, business_slug (for the business/org)
            - name, slug, source, source_id
            - short_description, description, category, subcategory
            - price_model, price_min, price_max, has_free_tier
            - rating, review_count, url
            - data_quality (0-1)
            - data: {} (freeform blob)

        Returns None to skip the item.
        """
        ...

    def upsert(self, item: dict) -> Optional[int]:
        """Upsert a normalized item into the database."""
        try:
            # Check if already exists by source + source_id
            source_id = item.get("source_id", item.get("slug"))
            existing = self.db.table("products").select("id").eq(
                "source", self.SOURCE
            ).eq("source_id", source_id).limit(1).execute()

            # Extract business info
            biz_name = item.pop("business_name", item["name"])
            biz_slug = item.pop("business_slug", item["slug"])
            biz_website = item.pop("business_website", None)

            # Upsert business
            bid = upsert_business(
                name=biz_name,
                slug=f"{self.SOURCE}-{biz_slug}",
                website=biz_website,
            )

            # Ensure source fields
            item["source"] = self.SOURCE
            item["source_id"] = source_id

            # Upsert product
            product_slug = f"{self.SOURCE}-{item.pop('slug', biz_slug)}"
            product_name = item.pop("name")

            pid = upsert_product(
                business_id=bid,
                slug=product_slug,
                name=product_name,
                **item,
            )

            if existing and existing.data:
                self.stats["updated"] += 1
            else:
                self.stats["inserted"] += 1

            return pid

        except Exception as e:
            log.error("Failed to upsert %s: %s", item.get("name", "?"), e)
            self.stats["errors"] += 1
            return None

    def run(self, max_items: int = 1000, offset: int = 0):
        """Run the connector — fetch, normalize, upsert in batches."""
        log.info("Starting %s connector (max=%d, offset=%d)", self.SOURCE, max_items, offset)
        total_fetched = 0

        while total_fetched < max_items:
            batch_limit = min(self.BATCH_SIZE, max_items - total_fetched)
            raw_batch = self.fetch_batch(offset=offset + total_fetched, limit=batch_limit)

            if not raw_batch:
                log.info("No more items from source")
                break

            for raw in raw_batch:
                normalized = self.normalize(raw)
                if normalized:
                    self.upsert(normalized)
                else:
                    self.stats["skipped"] += 1

            total_fetched += len(raw_batch)

            if len(raw_batch) < batch_limit:
                break  # Source exhausted

            log.info(
                "Progress: fetched=%d, inserted=%d, updated=%d, skipped=%d, errors=%d",
                total_fetched, self.stats["inserted"], self.stats["updated"],
                self.stats["skipped"], self.stats["errors"],
            )

        log.info(
            "Done. Total: fetched=%d, inserted=%d, updated=%d, skipped=%d, errors=%d",
            total_fetched, self.stats["inserted"], self.stats["updated"],
            self.stats["skipped"], self.stats["errors"],
        )
        return self.stats
