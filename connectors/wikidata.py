#!/usr/bin/env python3
"""Wikidata connector — pulls structured entities into the index.

Queries Wikidata SPARQL endpoint for software, businesses, organizations,
and other entities that agents commonly search for.

Usage:
    python3 connectors/wikidata.py --max 2000
    python3 connectors/wikidata.py --type software --max 500
    python3 connectors/wikidata.py --type companies --max 500
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
log = logging.getLogger("wikidata-connector")

SPARQL_URL = "https://query.wikidata.org/sparql"

# SPARQL queries for different entity types
QUERIES = {
    "software": """
        SELECT ?item ?itemLabel ?itemDescription ?website ?inception ?licenseLabel
               ?developerLabel ?genreLabel
        WHERE {{
            ?item wdt:P31/wdt:P279* wd:Q7397.
            OPTIONAL {{ ?item wdt:P856 ?website. }}
            OPTIONAL {{ ?item wdt:P571 ?inception. }}
            OPTIONAL {{ ?item wdt:P275 ?license. }}
            OPTIONAL {{ ?item wdt:P178 ?developer. }}
            OPTIONAL {{ ?item wdt:P136 ?genre. }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        ORDER BY ?item
        LIMIT {limit} OFFSET {offset}
    """,
    "companies": """
        SELECT ?item ?itemLabel ?itemDescription ?website ?inception
               ?countryLabel ?industryLabel ?employees ?revenue
        WHERE {{
            ?item wdt:P31/wdt:P279* wd:Q4830453.
            ?item wdt:P856 ?website.
            OPTIONAL {{ ?item wdt:P571 ?inception. }}
            OPTIONAL {{ ?item wdt:P17 ?country. }}
            OPTIONAL {{ ?item wdt:P452 ?industry. }}
            OPTIONAL {{ ?item wdt:P1128 ?employees. }}
            OPTIONAL {{ ?item wdt:P2139 ?revenue. }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        ORDER BY ?item
        LIMIT {limit} OFFSET {offset}
    """,
    "programming_languages": """
        SELECT ?item ?itemLabel ?itemDescription ?website ?inception
               ?designerLabel ?paradigmLabel ?influencedByLabel
        WHERE {{
            ?item wdt:P31 wd:Q9143.
            OPTIONAL {{ ?item wdt:P856 ?website. }}
            OPTIONAL {{ ?item wdt:P571 ?inception. }}
            OPTIONAL {{ ?item wdt:P287 ?designer. }}
            OPTIONAL {{ ?item wdt:P3966 ?paradigm. }}
            OPTIONAL {{ ?item wdt:P737 ?influencedBy. }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        ORDER BY ?item
        LIMIT {limit} OFFSET {offset}
    """,
    "databases": """
        SELECT ?item ?itemLabel ?itemDescription ?website ?inception
               ?developerLabel ?licenseLabel
        WHERE {{
            ?item wdt:P31/wdt:P279* wd:Q176165.
            OPTIONAL {{ ?item wdt:P856 ?website. }}
            OPTIONAL {{ ?item wdt:P571 ?inception. }}
            OPTIONAL {{ ?item wdt:P178 ?developer. }}
            OPTIONAL {{ ?item wdt:P275 ?license. }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        ORDER BY ?item
        LIMIT {limit} OFFSET {offset}
    """,
    "web_frameworks": """
        SELECT ?item ?itemLabel ?itemDescription ?website ?inception
               ?developerLabel ?langLabel ?licenseLabel
        WHERE {{
            ?item wdt:P31/wdt:P279* wd:Q1330336.
            OPTIONAL {{ ?item wdt:P856 ?website. }}
            OPTIONAL {{ ?item wdt:P571 ?inception. }}
            OPTIONAL {{ ?item wdt:P178 ?developer. }}
            OPTIONAL {{ ?item wdt:P277 ?lang. }}
            OPTIONAL {{ ?item wdt:P275 ?license. }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        ORDER BY ?item
        LIMIT {limit} OFFSET {offset}
    """,
    "operating_systems": """
        SELECT ?item ?itemLabel ?itemDescription ?website ?inception
               ?developerLabel ?licenseLabel ?familyLabel
        WHERE {{
            ?item wdt:P31/wdt:P279* wd:Q9135.
            OPTIONAL {{ ?item wdt:P856 ?website. }}
            OPTIONAL {{ ?item wdt:P571 ?inception. }}
            OPTIONAL {{ ?item wdt:P178 ?developer. }}
            OPTIONAL {{ ?item wdt:P275 ?license. }}
            OPTIONAL {{ ?item wdt:P361 ?family. }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        ORDER BY ?item
        LIMIT {limit} OFFSET {offset}
    """,
    "file_formats": """
        SELECT ?item ?itemLabel ?itemDescription ?website ?inception
               ?developerLabel ?mimeType ?extension
        WHERE {{
            ?item wdt:P31/wdt:P279* wd:Q235557.
            OPTIONAL {{ ?item wdt:P856 ?website. }}
            OPTIONAL {{ ?item wdt:P571 ?inception. }}
            OPTIONAL {{ ?item wdt:P178 ?developer. }}
            OPTIONAL {{ ?item wdt:P1163 ?mimeType. }}
            OPTIONAL {{ ?item wdt:P1195 ?extension. }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        ORDER BY ?item
        LIMIT {limit} OFFSET {offset}
    """,
    "protocols": """
        SELECT ?item ?itemLabel ?itemDescription ?website ?inception
        WHERE {{
            ?item wdt:P31/wdt:P279* wd:Q15836568.
            OPTIONAL {{ ?item wdt:P856 ?website. }}
            OPTIONAL {{ ?item wdt:P571 ?inception. }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        ORDER BY ?item
        LIMIT {limit} OFFSET {offset}
    """,
}

# Map Wikidata entity types to our categories
TYPE_TO_CATEGORY = {
    "software": ("development", "software"),
    "companies": ("business", "company"),
    "programming_languages": ("development", "programming-language"),
    "databases": ("development", "database"),
    "web_frameworks": ("development", "web-framework"),
    "operating_systems": ("development", "operating-system"),
    "file_formats": ("data", "file-format"),
    "protocols": ("communication", "protocol"),
}


def _extract_id(uri: str) -> str:
    """Extract QID from Wikidata URI."""
    if uri and "/" in uri:
        return uri.split("/")[-1]
    return uri or ""


class WikidataConnector(BaseConnector):
    SOURCE = "wikidata"
    BATCH_SIZE = 200  # SPARQL results per query

    def __init__(self, entity_type: str = "software"):
        super().__init__()
        self.entity_type = entity_type
        if entity_type not in QUERIES:
            raise ValueError(f"Unknown entity type: {entity_type}. Valid: {list(QUERIES.keys())}")
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "sylex-search-connector/1.0 (https://github.com/MastadoonPrime/sylex-search) Python/requests",
            "Accept": "application/json",
        })
        self._seen_ids = set()  # Dedupe within a run

    def fetch_batch(self, offset: int, limit: int) -> list[dict]:
        """Fetch entities from Wikidata SPARQL endpoint."""
        query = QUERIES[self.entity_type].format(limit=limit, offset=offset)

        try:
            resp = self._session.get(
                SPARQL_URL,
                params={"query": query, "format": "json"},
                timeout=60,
            )
            if resp.status_code == 429:
                log.warning("Rate limited, waiting 30s...")
                time.sleep(30)
                return self.fetch_batch(offset, limit)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", {}).get("bindings", [])
            time.sleep(2)  # Be gentle with Wikidata
            return results

        except Exception as e:
            log.error("SPARQL query failed: %s", e)
            time.sleep(10)
            return []

    def normalize(self, raw: dict) -> Optional[dict]:
        """Normalize a Wikidata SPARQL result."""
        item_uri = raw.get("item", {}).get("value", "")
        qid = _extract_id(item_uri)

        if not qid or qid in self._seen_ids:
            return None
        self._seen_ids.add(qid)

        name = raw.get("itemLabel", {}).get("value", "")
        # Skip items where label = QID (no English label)
        if not name or name.startswith("Q") and name[1:].isdigit():
            return None

        description = raw.get("itemDescription", {}).get("value", "")
        website = raw.get("website", {}).get("value", "")
        inception = raw.get("inception", {}).get("value", "")

        category, subcategory = TYPE_TO_CATEGORY.get(
            self.entity_type, ("general", "entity")
        )

        # Build slug
        slug = re.sub(r'[^a-z0-9-]', '-', name.lower())
        slug = re.sub(r'-+', '-', slug).strip("-")
        if not slug:
            slug = qid.lower()

        # Build data blob with available fields
        data_blob = {
            "wikidata_id": qid,
            "wikidata_url": item_uri,
            "type": self.entity_type,
        }

        if inception:
            data_blob["inception"] = inception[:10]  # Just the date part

        # Type-specific fields
        for field in ["licenseLabel", "developerLabel", "genreLabel", "countryLabel",
                       "industryLabel", "designerLabel", "paradigmLabel",
                       "influencedByLabel", "langLabel", "familyLabel",
                       "mimeType", "extension"]:
            val = raw.get(field, {}).get("value")
            if val:
                # Clean field name
                clean_name = field.replace("Label", "")
                data_blob[clean_name] = val

        if raw.get("employees", {}).get("value"):
            try:
                data_blob["employees"] = int(float(raw["employees"]["value"]))
            except (ValueError, TypeError):
                pass

        # Data quality — Wikidata entities are generally reliable
        quality = 0.5
        if description:
            quality += 0.15
        if website:
            quality += 0.15
        if inception:
            quality += 0.1
        if len(data_blob) > 4:
            quality += 0.1

        # Business name — use developer/designer if available
        biz_name = data_blob.get("developer", name)
        biz_slug = re.sub(r'[^a-z0-9-]', '-', biz_name.lower())
        biz_slug = re.sub(r'-+', '-', biz_slug).strip("-") or slug

        return {
            "name": name,
            "slug": slug,
            "source_id": qid,
            "business_name": biz_name,
            "business_slug": biz_slug,
            "business_website": website or None,
            "short_description": description[:200] if description else f"{name} ({self.entity_type})",
            "description": description or f"{name} — {self.entity_type} from Wikidata",
            "category": category,
            "subcategory": subcategory,
            "price_model": "free" if self.entity_type in ("software", "programming_languages", "protocols", "file_formats") else None,
            "url": website or item_uri,
            "data_quality": round(min(quality, 1.0), 2),
            "data": data_blob,
        }


def main():
    parser = argparse.ArgumentParser(description="Wikidata connector for Agent Commerce")
    parser.add_argument("--max", type=int, default=500, help="Max entities per type")
    parser.add_argument("--offset", type=int, default=0, help="Start offset")
    parser.add_argument("--type", type=str, default=None,
                        help=f"Entity type: {', '.join(QUERIES.keys())}. Default: all types")
    args = parser.parse_args()

    if args.type:
        types = [args.type]
    else:
        types = list(QUERIES.keys())

    total_stats = {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

    for entity_type in types:
        log.info("=== Starting %s ===", entity_type)
        connector = WikidataConnector(entity_type=entity_type)
        stats = connector.run(max_items=args.max, offset=args.offset)
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    log.info("=== ALL DONE === inserted=%d, updated=%d, skipped=%d, errors=%d",
             total_stats["inserted"], total_stats["updated"],
             total_stats["skipped"], total_stats["errors"])


if __name__ == "__main__":
    main()
