#!/usr/bin/env python3
"""npm registry connector — pulls Node.js packages into the index.

Uses the npm search API for discovery (most popular/relevant packages)
and per-package API for full metadata.

Usage:
    python3 connectors/npm.py --max 500
    python3 connectors/npm.py --max 1000 --offset 250
    python3 connectors/npm.py --popular  # fetch top packages by popularity
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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
log = logging.getLogger("npm-connector")

# npm search API
NPM_SEARCH_URL = "https://registry.npmjs.org/-/v1/search"
NPM_PACKAGE_URL = "https://registry.npmjs.org"
NPM_DOWNLOADS_URL = "https://api.npmjs.org/downloads/point/last-month"

# Category mapping — map npm keywords to our category system
KEYWORD_TO_CATEGORY = {
    # Development
    "framework": ("development", "framework"),
    "react": ("development", "frontend"),
    "vue": ("development", "frontend"),
    "angular": ("development", "frontend"),
    "svelte": ("development", "frontend"),
    "frontend": ("development", "frontend"),
    "ui": ("development", "frontend"),
    "component": ("development", "frontend"),
    "css": ("development", "frontend"),
    "styling": ("development", "frontend"),
    "backend": ("development", "backend"),
    "express": ("development", "backend"),
    "fastify": ("development", "backend"),
    "koa": ("development", "backend"),
    "server": ("development", "backend"),
    "api": ("development", "backend"),
    "rest": ("development", "backend"),
    "graphql": ("development", "backend"),
    "orm": ("development", "database"),
    "database": ("development", "database"),
    "sql": ("development", "database"),
    "mongodb": ("development", "database"),
    "redis": ("development", "database"),
    "compiler": ("development", "compiler"),
    "bundler": ("development", "bundler"),
    "webpack": ("development", "bundler"),
    "vite": ("development", "bundler"),
    "rollup": ("development", "bundler"),
    "esbuild": ("development", "bundler"),
    "cli": ("development", "cli"),
    "parser": ("development", "parser"),
    "typescript": ("development", "language"),

    # Testing
    "test": ("testing", "testing-framework"),
    "testing": ("testing", "testing-framework"),
    "jest": ("testing", "testing-framework"),
    "mocha": ("testing", "testing-framework"),
    "e2e": ("testing", "e2e"),
    "lint": ("testing", "linting"),
    "linter": ("testing", "linting"),
    "eslint": ("testing", "linting"),

    # DevOps / Deployment
    "docker": ("devops", "containerization"),
    "kubernetes": ("devops", "orchestration"),
    "ci": ("devops", "ci-cd"),
    "cd": ("devops", "ci-cd"),
    "deploy": ("deployment", "deployment"),
    "aws": ("deployment", "cloud"),
    "cloud": ("deployment", "cloud"),

    # Data
    "data": ("data", "data-processing"),
    "stream": ("data", "streaming"),
    "csv": ("data", "data-processing"),
    "json": ("data", "data-processing"),
    "xml": ("data", "data-processing"),
    "validation": ("data", "validation"),

    # Security
    "security": ("security", "security"),
    "auth": ("security", "authentication"),
    "authentication": ("security", "authentication"),
    "oauth": ("security", "authentication"),
    "jwt": ("security", "authentication"),
    "crypto": ("security", "cryptography"),
    "encryption": ("security", "cryptography"),

    # AI/ML
    "ai": ("ai-ml", "ai"),
    "machine-learning": ("ai-ml", "machine-learning"),
    "ml": ("ai-ml", "machine-learning"),
    "nlp": ("ai-ml", "nlp"),
    "neural": ("ai-ml", "neural-network"),
    "tensorflow": ("ai-ml", "machine-learning"),
    "openai": ("ai-ml", "ai"),
    "llm": ("ai-ml", "ai"),

    # Communication
    "email": ("communication", "email"),
    "smtp": ("communication", "email"),
    "websocket": ("communication", "realtime"),
    "socket": ("communication", "realtime"),
    "chat": ("communication", "chat"),
    "notification": ("communication", "notifications"),

    # Utility (fallback)
    "util": ("development", "utility"),
    "utility": ("development", "utility"),
    "helper": ("development", "utility"),
    "tool": ("development", "utility"),
}

# Popular search terms to discover packages
POPULAR_SEARCHES = [
    "react", "vue", "angular", "svelte", "next", "express", "fastify",
    "typescript", "webpack", "vite", "eslint", "prettier", "jest", "mocha",
    "axios", "lodash", "moment", "date-fns", "uuid", "chalk", "commander",
    "inquirer", "ora", "dotenv", "cors", "body-parser", "jsonwebtoken",
    "bcrypt", "mongoose", "sequelize", "prisma", "typeorm", "knex",
    "socket.io", "ws", "redis", "bull", "puppeteer", "playwright",
    "cheerio", "marked", "highlight", "d3", "chart", "three",
    "tailwind", "bootstrap", "material-ui", "styled-components",
    "redux", "zustand", "mobx", "jotai", "recoil",
    "graphql", "apollo", "trpc", "zod", "yup", "joi",
    "cypress", "vitest", "storybook", "turbo", "nx", "lerna",
    "aws-sdk", "firebase", "supabase", "stripe", "twilio",
    "nodemailer", "sharp", "ffmpeg", "pdf", "csv", "xml",
    "openai", "langchain", "ai", "tensorflow", "huggingface",
    "docker", "kubernetes", "serverless", "vercel",
    "electron", "tauri", "react-native", "expo",
    "markdown", "mdx", "remark", "rehype",
    "auth", "passport", "oauth", "jwt",
    "logger", "winston", "pino", "debug",
    "http", "fetch", "got", "superagent", "undici",
]


class NpmConnector(BaseConnector):
    SOURCE = "npm"
    BATCH_SIZE = 250  # npm search API max

    def __init__(self):
        super().__init__()
        self._search_index = 0
        self._search_offset = 0
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "sylex-search-connector/1.0 (https://github.com/MastadoonPrime/sylex-search)",
        })

    def fetch_batch(self, offset: int, limit: int) -> list[dict]:
        """Search npm for popular packages."""
        results = []

        while len(results) < limit and self._search_index < len(POPULAR_SEARCHES):
            term = POPULAR_SEARCHES[self._search_index]
            try:
                resp = self._session.get(
                    NPM_SEARCH_URL,
                    params={
                        "text": term,
                        "size": min(limit - len(results), 250),
                        "from": self._search_offset,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                objects = data.get("objects", [])

                for obj in objects:
                    results.append(obj)

                if len(objects) < 250:
                    # Move to next search term
                    self._search_index += 1
                    self._search_offset = 0
                else:
                    self._search_offset += len(objects)

                # Rate limiting — be gentle
                time.sleep(0.2)

            except Exception as e:
                log.warning("npm search error for '%s': %s", term, e)
                self._search_index += 1
                self._search_offset = 0
                time.sleep(1)

        return results

    def normalize(self, raw: dict) -> Optional[dict]:
        """Normalize an npm search result into our schema."""
        pkg = raw.get("package", {})
        name = pkg.get("name", "")

        if not name:
            return None

        # Skip scoped packages with very low quality (internal packages)
        # but keep popular scoped packages like @types/, @babel/, etc.
        if name.startswith("@") and raw.get("score", {}).get("detail", {}).get("popularity", 0) < 0.01:
            return None

        description = pkg.get("description", "")
        if not description:
            description = f"npm package: {name}"

        keywords = pkg.get("keywords", []) or []
        links = pkg.get("links", {})
        version = pkg.get("version", "")
        author = pkg.get("author", {})
        publisher = pkg.get("publisher", {})

        # Determine category from keywords
        category = "development"
        subcategory = "npm-package"
        for kw in keywords:
            kw_lower = kw.lower().strip()
            if kw_lower in KEYWORD_TO_CATEGORY:
                category, subcategory = KEYWORD_TO_CATEGORY[kw_lower]
                break

        # Score quality
        score = raw.get("score", {})
        detail = score.get("detail", {})
        npm_quality = detail.get("quality", 0.5)
        npm_popularity = detail.get("popularity", 0)
        npm_maintenance = detail.get("maintenance", 0.5)

        # Map npm score to our data_quality (0-1)
        data_quality = round(
            npm_quality * 0.4 + npm_popularity * 0.3 + npm_maintenance * 0.3,
            2,
        )

        # Build price info — all npm packages are free (open source)
        # But some have paid tiers or are wrappers around paid services

        # Extract author/org name
        author_name = ""
        if isinstance(author, dict):
            author_name = author.get("name", "")
        elif isinstance(author, str):
            author_name = author
        if not author_name and isinstance(publisher, dict):
            author_name = publisher.get("username", "")

        # Build slug
        slug = re.sub(r'[^a-z0-9-]', '-', name.lower().replace("/", "-").replace("@", ""))
        slug = re.sub(r'-+', '-', slug).strip("-")

        biz_slug = slug
        if author_name:
            biz_slug = re.sub(r'[^a-z0-9-]', '-', author_name.lower())
            biz_slug = re.sub(r'-+', '-', biz_slug).strip("-")

        return {
            "name": name,
            "slug": slug,
            "source_id": name,
            "business_name": author_name or name,
            "business_slug": biz_slug,
            "business_website": links.get("homepage"),
            "short_description": description[:200] if description else "",
            "description": description,
            "category": category,
            "subcategory": subcategory,
            "price_model": "free",
            "price_min": 0,
            "has_free_tier": True,
            "url": links.get("npm") or f"https://www.npmjs.com/package/{name}",
            "data_quality": data_quality,
            "data": {
                "version": version,
                "keywords": keywords[:20],
                "license": pkg.get("license") if isinstance(pkg.get("license"), str) else None,
                "homepage": links.get("homepage"),
                "repository": links.get("repository"),
                "npm_url": links.get("npm"),
                "npm_score": round(score.get("final", 0), 3),
                "npm_quality": round(npm_quality, 3),
                "npm_popularity": round(npm_popularity, 3),
                "npm_maintenance": round(npm_maintenance, 3),
                "author": author_name,
                "type": "npm-package",
            },
        }


def main():
    parser = argparse.ArgumentParser(description="npm connector for Agent Commerce")
    parser.add_argument("--max", type=int, default=500, help="Maximum packages to fetch")
    parser.add_argument("--offset", type=int, default=0, help="Start offset")
    parser.add_argument("--popular", action="store_true", help="Fetch popular packages")
    args = parser.parse_args()

    connector = NpmConnector()
    connector.run(max_items=args.max, offset=args.offset)


if __name__ == "__main__":
    main()
