#!/usr/bin/env python3
"""MCP servers connector — pulls MCP server listings from awesome-mcp-servers.

Parses the curated markdown list from punkpeye/awesome-mcp-servers to ingest
real MCP server data into Sylex Search. This is the highest-priority connector
because our target users are agents looking for MCP tools.

Usage:
    python3 connectors/mcp_servers.py
    python3 connectors/mcp_servers.py --max 500
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
log = logging.getLogger("mcp-servers-connector")

AWESOME_MCP_RAW = "https://raw.githubusercontent.com/punkpeye/awesome-mcp-servers/main/README.md"

# Emoji to feature mapping
EMOJI_FEATURES = {
    "📇": "typescript",
    "🐍": "python",
    "🦀": "rust",
    "🏎️": "go",
    "#️⃣": "csharp",
    "☕": "java",
    "💎": "ruby",
    "☁️": "remote",
    "🏠": "local",
    "🍎": "macos",
    "🪟": "windows",
    "🐧": "linux",
}

# Section name to category mapping
SECTION_TO_CATEGORY = {
    "agent platforms": ("ai-ml", "agent-platform"),
    "art & culture": ("media", "art-culture"),
    "browser automation": ("automation", "browser"),
    "cloud platforms": ("cloud", "platform"),
    # Actual sections from awesome-mcp-servers README
    "aggregators": ("ai-ml", "aggregator"),
    "aerospace & astrodynamics": ("research", "aerospace"),
    "art & culture": ("media", "art-culture"),
    "architecture & design": ("design", "architecture"),
    "biology medicine and bioinformatics": ("research", "biology"),
    "browser automation": ("automation", "browser"),
    "cloud platforms": ("cloud", "platform"),
    "code execution": ("development", "code-execution"),
    "coding agents": ("development", "coding-agent"),
    "command line": ("development", "cli"),
    "communication": ("communication", "messaging"),
    "customer data platforms": ("business", "customer-data"),
    "databases": ("development", "database"),
    "data platforms": ("development", "data-platform"),
    "developer tools": ("development", "tools"),
    "delivery": ("business", "delivery"),
    "data science tools": ("development", "data-science"),
    "data visualization": ("development", "data-viz"),
    "embedded system": ("development", "embedded"),
    "education": ("education", "general"),
    "e-commerce": ("business", "ecommerce"),
    "environment & nature": ("research", "environment"),
    "file systems": ("development", "filesystem"),
    "finance & fintech": ("finance", "general"),
    "gaming": ("gaming", "general"),
    "government & public data": ("business", "government"),
    "health": ("healthcare", "general"),
    "hr": ("business", "hr"),
    "iot & home automation": ("iot", "home-automation"),
    "knowledge & memory": ("ai-ml", "knowledge-memory"),
    "legal": ("business", "legal"),
    "location services": ("location", "general"),
    "marketing": ("business", "marketing"),
    "monitoring": ("devops", "monitoring"),
    "multimedia process": ("media", "multimedia"),
    "notification": ("communication", "notification"),
    "project management & task management": ("project-management", "general"),
    "real estate": ("business", "real-estate"),
    "research & data": ("research", "general"),
    "search & data extraction": ("search", "general"),
    "security": ("security", "general"),
    "social media and community": ("social", "general"),
    "sports & sports analytics": ("sports", "general"),
    "testing": ("testing", "general"),
    "translation": ("ai-ml", "translation"),
    "travel & transportation": ("travel", "general"),
    "version control": ("development", "version-control"),
    "virtual environments": ("development", "virtual-env"),
    "workplace & productivity": ("productivity", "general"),
    "weather": ("weather", "general"),
    "other tools and integrations": ("general", "other"),
}


class MCPServersConnector(BaseConnector):
    """Connector for awesome-mcp-servers curated list."""

    SOURCE = "mcp-servers"
    BATCH_SIZE = 100

    def __init__(self):
        super().__init__()
        self._entries: list[dict] = []
        self._fetched = False

    def _parse_readme(self) -> list[dict]:
        """Parse the awesome-mcp-servers README into structured entries."""
        log.info("Fetching awesome-mcp-servers README...")
        resp = requests.get(AWESOME_MCP_RAW, timeout=30)
        resp.raise_for_status()
        lines = resp.text.split("\n")

        entries = []
        current_section = "other tools"

        for line in lines:
            # Track section headers (## or ###)
            # Format: ### 🔗 <a name="aggregators"></a>Aggregators
            section_match = re.match(r'^#{2,3}\s+(.+)', line)
            if section_match:
                raw_section = section_match.group(1).strip()
                # Remove HTML anchor tags: <a name="..."></a>
                raw_section = re.sub(r'<a[^>]*>.*?</a>', '', raw_section)
                # Remove emoji and special chars, keep letters, digits, spaces, &, -
                section_name = re.sub(r'[^\w\s&-]', '', raw_section).strip().lower()
                if section_name:
                    current_section = section_name

            # Parse list entries: - [name](url) description
            entry_match = re.match(
                r'^-\s+\[([^\]]+)\]\(([^)]+)\)\s*(.*)',
                line
            )
            if not entry_match:
                continue

            name = entry_match.group(1).strip()
            url = entry_match.group(2).strip()
            rest = entry_match.group(3).strip()

            # Skip non-GitHub entries (some are npm links, etc.)
            if "github.com" not in url and "npmjs.com" not in url:
                continue

            # Extract emojis for features
            features = []
            for emoji, feature in EMOJI_FEATURES.items():
                if emoji in rest:
                    features.append(feature)

            # Remove Glama badge markdown from description
            desc = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', rest).strip()
            # Remove remaining emojis and clean up
            desc = re.sub(r'[📇🐍🦀🏎️#️⃣☕💎☁️🏠🍎🪟🐧]', '', desc).strip()
            desc = re.sub(r'^[-–—]\s*', '', desc).strip()

            # Extract owner/repo from GitHub URL
            gh_match = re.match(r'https://github\.com/([^/]+)/([^/\s?#]+)', url)
            owner = gh_match.group(1) if gh_match else ""
            repo = gh_match.group(2) if gh_match else name

            entries.append({
                "name": name,
                "url": url,
                "description": desc,
                "section": current_section,
                "features": features,
                "owner": owner,
                "repo": repo,
            })

        log.info("Parsed %d entries from README", len(entries))
        return entries

    def _enrich_from_github(self, entry: dict) -> dict:
        """Optionally enrich with GitHub metadata (stars, language, etc.)."""
        owner = entry.get("owner", "")
        repo = entry.get("repo", "")
        if not owner or not repo:
            return entry

        try:
            resp = requests.get(
                f"https://api.github.com/repos/{owner}/{repo}",
                timeout=10,
                headers={"Accept": "application/vnd.github.v3+json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                entry["stars"] = data.get("stargazers_count", 0)
                entry["language"] = data.get("language", "")
                entry["github_description"] = data.get("description", "")
                entry["topics"] = data.get("topics", [])
                entry["license"] = (data.get("license") or {}).get("spdx_id", "")
                entry["updated_at"] = data.get("updated_at", "")
            elif resp.status_code == 403:
                log.warning("GitHub API rate limited, skipping enrichment")
                entry["_rate_limited"] = True
            time.sleep(0.5)  # Be nice to GitHub API
        except Exception as e:
            log.debug("GitHub enrichment failed for %s/%s: %s", owner, repo, e)

        return entry

    def fetch_batch(self, offset: int, limit: int) -> list[dict]:
        """Return a batch of parsed entries."""
        if not self._fetched:
            self._entries = self._parse_readme()
            self._fetched = True

        batch = self._entries[offset:offset + limit]

        # Enrich with GitHub data (but stop if rate limited)
        enriched = []
        for entry in batch:
            entry = self._enrich_from_github(entry)
            if entry.get("_rate_limited"):
                log.info("Rate limited — continuing without GitHub enrichment")
                enriched.append(entry)
                # Add remaining without enrichment
                enriched.extend(batch[len(enriched):])
                break
            enriched.append(entry)

        return enriched

    def normalize(self, raw: dict) -> Optional[dict]:
        """Normalize an awesome-mcp-servers entry into our schema."""
        name = raw.get("name", "")
        if not name:
            return None

        description = raw.get("description", "")
        gh_desc = raw.get("github_description", "")

        # Use GitHub description if local one is empty or very short
        if gh_desc and (not description or len(description) < 20):
            description = gh_desc

        # Determine category from section
        section = raw.get("section", "other tools")
        category, subcategory = SECTION_TO_CATEGORY.get(
            section, ("general", "mcp-server")
        )

        # Build features list for the data blob
        features = raw.get("features", [])
        language = raw.get("language", "")
        if language:
            features.append(f"language:{language.lower()}")

        # Determine price model (most MCP servers are free/open source)
        license_id = raw.get("license", "")
        has_free = bool(license_id) or "free" in description.lower()

        # Rating based on stars (rough heuristic)
        stars = raw.get("stars", 0)
        if stars >= 1000:
            rating = 4.5
        elif stars >= 100:
            rating = 4.0
        elif stars >= 10:
            rating = 3.5
        else:
            rating = 3.0

        owner = raw.get("owner", "")
        repo = raw.get("repo", name)

        return {
            "name": name,
            "slug": f"{owner}-{repo}" if owner else repo,
            "business_name": owner or name,
            "business_slug": owner or repo,
            "business_website": f"https://github.com/{owner}" if owner else None,
            "source_id": f"{owner}/{repo}" if owner else name,
            "short_description": description[:200] if description else f"MCP server: {name}",
            "description": description,
            "category": category,
            "subcategory": subcategory,
            "price_model": "free" if has_free else "unknown",
            "price_min": 0.0,
            "price_max": 0.0,
            "has_free_tier": has_free,
            "rating": rating,
            "review_count": stars,
            "url": raw.get("url", ""),
            "data_quality": 0.7 if description else 0.4,
            "data": {
                "type": "mcp-server",
                "features": features,
                "topics": raw.get("topics", []),
                "license": license_id,
                "stars": stars,
                "language": language,
                "section": section,
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest MCP servers from awesome-mcp-servers")
    parser.add_argument("--max", type=int, default=5000, help="Max items to ingest")
    parser.add_argument("--offset", type=int, default=0, help="Start offset")
    parser.add_argument("--no-enrich", action="store_true", help="Skip GitHub API enrichment")
    args = parser.parse_args()

    connector = MCPServersConnector()
    if args.no_enrich:
        connector._enrich_from_github = lambda e: e  # type: ignore
    stats = connector.run(max_items=args.max, offset=args.offset)
    print(f"\nResults: {stats}")
