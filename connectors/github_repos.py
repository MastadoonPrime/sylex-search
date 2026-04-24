#!/usr/bin/env python3
"""GitHub connector — pulls popular repositories into the index.

Uses the GitHub search API to find popular repos across different
languages and topics.

Usage:
    python3 connectors/github_repos.py --max 1000
    python3 connectors/github_repos.py --max 500 --language python
"""

from __future__ import annotations

import argparse
import logging
import os
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
log = logging.getLogger("github-connector")

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# Search queries to discover repos by topic
SEARCH_QUERIES = [
    # Frameworks & Libraries
    ("stars:>5000 language:javascript", "development", "javascript"),
    ("stars:>5000 language:typescript", "development", "typescript"),
    ("stars:>3000 language:python", "development", "python"),
    ("stars:>3000 language:go", "development", "go"),
    ("stars:>3000 language:rust", "development", "rust"),
    ("stars:>2000 language:java", "development", "java"),
    ("stars:>2000 language:c++", "development", "cpp"),
    ("stars:>2000 language:c#", "development", "csharp"),
    ("stars:>2000 language:ruby", "development", "ruby"),
    ("stars:>2000 language:swift", "development", "swift"),
    ("stars:>2000 language:kotlin", "development", "kotlin"),
    ("stars:>2000 language:php", "development", "php"),
    # Topics
    ("topic:machine-learning stars:>1000", "ai-ml", "machine-learning"),
    ("topic:deep-learning stars:>1000", "ai-ml", "deep-learning"),
    ("topic:llm stars:>500", "ai-ml", "llm"),
    ("topic:devops stars:>1000", "devops", "devops"),
    ("topic:docker stars:>1000", "devops", "containerization"),
    ("topic:kubernetes stars:>1000", "devops", "orchestration"),
    ("topic:database stars:>1000", "development", "database"),
    ("topic:security stars:>1000", "security", "security"),
    ("topic:testing stars:>1000", "testing", "testing-framework"),
    ("topic:cli stars:>1000", "development", "cli"),
    ("topic:api stars:>1000", "development", "api"),
    ("topic:framework stars:>2000", "development", "framework"),
    ("topic:react stars:>2000", "development", "frontend"),
    ("topic:vue stars:>1000", "development", "frontend"),
    ("topic:nextjs stars:>500", "development", "frontend"),
    ("topic:data-science stars:>1000", "data", "data-science"),
    ("topic:automation stars:>1000", "devops", "automation"),
]


class GitHubConnector(BaseConnector):
    SOURCE = "github"
    BATCH_SIZE = 100  # GitHub API max per_page

    def __init__(self, language: str = None):
        super().__init__()
        self._session = requests.Session()
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "sylex-search-connector/1.0 (https://github.com/MastadoonPrime/sylex-search)",
        }
        if GITHUB_TOKEN:
            headers["Authorization"] = f"token {GITHUB_TOKEN}"
        self._session.headers.update(headers)

        self._query_index = 0
        self._page = 1
        self._seen_repos = set()

        # If language specified, filter queries
        if language:
            lang_lower = language.lower()
            self._queries = [(q, c, s) for q, c, s in SEARCH_QUERIES
                             if lang_lower in q.lower()]
            if not self._queries:
                self._queries = [(f"stars:>1000 language:{language}", "development", language)]
        else:
            self._queries = list(SEARCH_QUERIES)

    def fetch_batch(self, offset: int, limit: int) -> list[dict]:
        """Search GitHub for popular repos."""
        results = []

        while len(results) < limit and self._query_index < len(self._queries):
            query, category, subcategory = self._queries[self._query_index]

            try:
                resp = self._session.get(
                    f"{GITHUB_API}/search/repositories",
                    params={
                        "q": query,
                        "sort": "stars",
                        "order": "desc",
                        "per_page": min(limit - len(results), 100),
                        "page": self._page,
                    },
                    timeout=30,
                )

                # Handle rate limiting
                remaining = int(resp.headers.get("X-RateLimit-Remaining", 1))
                if resp.status_code == 403 or remaining == 0:
                    reset_time = int(resp.headers.get("X-RateLimit-Reset", 0))
                    wait = max(reset_time - time.time(), 10)
                    log.warning("Rate limited, waiting %ds...", wait)
                    time.sleep(min(wait, 60))
                    continue

                resp.raise_for_status()
                data = resp.json()
                items = data.get("items", [])

                for item in items:
                    repo_id = item.get("id")
                    if repo_id not in self._seen_repos:
                        self._seen_repos.add(repo_id)
                        # Attach our category info
                        item["_category"] = category
                        item["_subcategory"] = subcategory
                        results.append(item)

                if len(items) < 100 or self._page >= 10:
                    # Move to next query
                    self._query_index += 1
                    self._page = 1
                else:
                    self._page += 1

                # Rate limiting — GitHub allows 10 search requests/min unauthenticated
                time.sleep(6 if not GITHUB_TOKEN else 1)

            except Exception as e:
                log.warning("GitHub search error: %s", e)
                self._query_index += 1
                self._page = 1
                time.sleep(5)

        return results

    def normalize(self, raw: dict) -> Optional[dict]:
        """Normalize a GitHub repo into our schema."""
        full_name = raw.get("full_name", "")
        name = raw.get("name", "")
        if not name or not full_name:
            return None

        description = raw.get("description") or ""
        owner = raw.get("owner", {})
        owner_login = owner.get("login", "")
        owner_type = owner.get("type", "User")  # User or Organization

        stars = raw.get("stargazers_count", 0)
        forks = raw.get("forks_count", 0)
        language = raw.get("language") or ""
        topics = raw.get("topics", []) or []
        license_info = raw.get("license") or {}
        license_name = license_info.get("spdx_id") or license_info.get("name") or ""
        homepage = raw.get("homepage") or ""
        archived = raw.get("archived", False)

        # Skip archived repos
        if archived:
            return None

        category = raw.get("_category", "development")
        subcategory = raw.get("_subcategory", "repository")

        # Refine category from topics
        if not raw.get("_category"):
            for topic in topics:
                topic_lower = topic.lower()
                if topic_lower in ("machine-learning", "deep-learning", "ai", "llm"):
                    category, subcategory = "ai-ml", topic_lower
                    break
                elif topic_lower in ("devops", "docker", "kubernetes"):
                    category, subcategory = "devops", topic_lower
                    break
                elif topic_lower in ("security", "cryptography"):
                    category, subcategory = "security", topic_lower
                    break

        # Build slug
        slug = re.sub(r'[^a-z0-9-]', '-', full_name.lower().replace("/", "-"))
        slug = re.sub(r'-+', '-', slug).strip("-")

        biz_slug = re.sub(r'[^a-z0-9-]', '-', owner_login.lower())
        biz_slug = re.sub(r'-+', '-', biz_slug).strip("-")

        # Data quality based on completeness
        quality = 0.4
        if description:
            quality += 0.15
        if stars > 100:
            quality += 0.1
        if stars > 1000:
            quality += 0.1
        if topics:
            quality += 0.05
        if license_name:
            quality += 0.05
        if homepage:
            quality += 0.05
        if language:
            quality += 0.05

        # Map stars to a 0-5 rating
        if stars >= 50000:
            rating = 5.0
        elif stars >= 10000:
            rating = 4.5
        elif stars >= 5000:
            rating = 4.0
        elif stars >= 1000:
            rating = 3.5
        elif stars >= 500:
            rating = 3.0
        else:
            rating = 2.5

        return {
            "name": name,
            "slug": slug,
            "source_id": full_name,
            "business_name": owner_login,
            "business_slug": biz_slug,
            "business_website": homepage or f"https://github.com/{owner_login}",
            "short_description": description[:200] if description else f"GitHub repository: {full_name}",
            "description": description,
            "category": category,
            "subcategory": subcategory,
            "price_model": "free",
            "price_min": 0,
            "has_free_tier": True,
            "rating": rating,
            "review_count": stars,  # Use stars as review count proxy
            "url": raw.get("html_url", f"https://github.com/{full_name}"),
            "data_quality": round(min(quality, 1.0), 2),
            "data": {
                "full_name": full_name,
                "stars": stars,
                "forks": forks,
                "language": language,
                "topics": topics[:15],
                "license": license_name if license_name != "NOASSERTION" else None,
                "homepage": homepage or None,
                "owner": owner_login,
                "owner_type": owner_type,
                "open_issues": raw.get("open_issues_count", 0),
                "created_at": (raw.get("created_at") or "")[:10],
                "updated_at": (raw.get("updated_at") or "")[:10],
                "default_branch": raw.get("default_branch", "main"),
                "type": "github-repo",
            },
        }


def main():
    parser = argparse.ArgumentParser(description="GitHub connector for Agent Commerce")
    parser.add_argument("--max", type=int, default=500, help="Maximum repos to fetch")
    parser.add_argument("--language", type=str, help="Filter by programming language")
    args = parser.parse_args()

    connector = GitHubConnector(language=args.language)
    connector.run(max_items=args.max)


if __name__ == "__main__":
    main()
