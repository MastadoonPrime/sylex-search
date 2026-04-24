#!/usr/bin/env python3
"""PyPI connector — pulls Python packages into the index.

Uses PyPI JSON API for package metadata. Fetches the most popular/important
packages by crawling curated lists and search results.

Usage:
    python3 connectors/pypi.py --max 500
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from typing import Optional
from html.parser import HTMLParser

import requests

from base import BaseConnector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pypi-connector")

PYPI_API = "https://pypi.org/pypi"
PYPI_SIMPLE = "https://pypi.org/simple/"
PYPI_STATS = "https://pypistats.org/api/packages"

# Classifier to category mapping
CLASSIFIER_TO_CATEGORY = {
    "Framework": ("development", "framework"),
    "Framework :: Django": ("development", "web-framework"),
    "Framework :: Flask": ("development", "web-framework"),
    "Framework :: FastAPI": ("development", "web-framework"),
    "Framework :: Pytest": ("testing", "testing-framework"),
    "Topic :: Scientific/Engineering :: Artificial Intelligence": ("ai-ml", "machine-learning"),
    "Topic :: Scientific/Engineering": ("data", "scientific"),
    "Topic :: Software Development :: Libraries": ("development", "library"),
    "Topic :: Software Development :: Testing": ("testing", "testing-framework"),
    "Topic :: Software Development :: Build Tools": ("devops", "build-tools"),
    "Topic :: Software Development :: Version Control": ("devops", "version-control"),
    "Topic :: System :: Networking": ("communication", "networking"),
    "Topic :: Internet :: WWW/HTTP": ("development", "web"),
    "Topic :: Database": ("development", "database"),
    "Topic :: Security": ("security", "security"),
    "Topic :: Multimedia": ("media", "multimedia"),
    "Topic :: Text Processing": ("data", "text-processing"),
    "Topic :: Communications": ("communication", "messaging"),
    "Topic :: Office/Business": ("productivity", "office"),
}

# Top Python packages to index (by popularity/importance)
TOP_PACKAGES = [
    # Web frameworks
    "django", "flask", "fastapi", "tornado", "bottle", "falcon", "starlette",
    "sanic", "aiohttp", "uvicorn", "gunicorn", "celery",
    # Data science / ML
    "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn",
    "tensorflow", "pytorch", "keras", "xgboost", "lightgbm", "catboost",
    "transformers", "datasets", "tokenizers", "accelerate", "diffusers",
    "langchain", "openai", "anthropic", "ollama", "chromadb",
    "jupyter", "notebook", "ipython", "jupyterlab",
    # Data processing
    "polars", "dask", "vaex", "pyarrow", "parquet",
    "pydantic", "marshmallow", "attrs", "dataclasses-json",
    # Database
    "sqlalchemy", "psycopg2", "pymongo", "redis", "elasticsearch",
    "alembic", "tortoise-orm", "peewee", "databases",
    "supabase", "firebase-admin",
    # HTTP / API
    "requests", "httpx", "urllib3", "aiohttp", "grpcio",
    "beautifulsoup4", "scrapy", "selenium", "playwright",
    "lxml", "html5lib",
    # CLI / DevTools
    "click", "typer", "argparse", "rich", "tqdm", "colorama",
    "black", "ruff", "flake8", "mypy", "pylint", "isort",
    "pytest", "pytest-cov", "pytest-asyncio", "coverage", "tox", "nox",
    "poetry", "pipenv", "pip-tools", "setuptools", "wheel", "twine",
    # Cloud / DevOps
    "boto3", "google-cloud-storage", "azure-storage-blob",
    "docker", "kubernetes", "ansible", "fabric", "paramiko",
    "terraform-cdk",
    # Auth / Security
    "cryptography", "pyjwt", "python-jose", "passlib", "bcrypt",
    "authlib", "python-oauth2",
    # Utilities
    "python-dotenv", "pyyaml", "toml", "configparser",
    "pillow", "opencv-python", "imageio",
    "arrow", "pendulum", "python-dateutil",
    "loguru", "structlog", "logging",
    "jinja2", "mako",
    "pydantic-settings", "python-decouple",
    # Async
    "asyncio", "trio", "anyio", "twisted",
    # Messaging
    "kombu", "pika", "kafka-python",
    # File formats
    "openpyxl", "xlsxwriter", "python-docx", "reportlab", "pypdf",
    "markdown", "mistune",
    # API frameworks
    "graphene", "strawberry-graphql", "ariadne",
]


class PyPIConnector(BaseConnector):
    SOURCE = "pypi"
    BATCH_SIZE = 50  # Fetch one at a time via JSON API, but batch the list

    def __init__(self, use_top_list: bool = False):
        super().__init__()
        if use_top_list:
            self._package_list = self._fetch_top_packages()
        else:
            self._package_list = list(TOP_PACKAGES)
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "sylex-search-connector/1.0 (https://github.com/MastadoonPrime/sylex-search)",
        })

    def fetch_batch(self, offset: int, limit: int) -> list[dict]:
        """Fetch package metadata from PyPI JSON API."""
        results = []
        end = min(offset + limit, len(self._package_list))

        for i in range(offset, end):
            if i >= len(self._package_list):
                break

            pkg_name = self._package_list[i]
            try:
                resp = self._session.get(
                    f"{PYPI_API}/{pkg_name}/json",
                    timeout=15,
                )
                if resp.status_code == 404:
                    log.warning("Package not found: %s", pkg_name)
                    continue
                resp.raise_for_status()
                results.append(resp.json())
                time.sleep(0.1)  # Be gentle

            except Exception as e:
                log.warning("PyPI fetch error for '%s': %s", pkg_name, e)
                time.sleep(0.5)

        return results

    def normalize(self, raw: dict) -> Optional[dict]:
        """Normalize a PyPI package into our schema."""
        info = raw.get("info", {})
        name = info.get("name", "")

        if not name:
            return None

        summary = info.get("summary", "") or ""
        description = info.get("description", "") or ""
        # Truncate long descriptions
        if len(description) > 500:
            description = description[:500] + "..."

        # Extract category from classifiers
        classifiers = info.get("classifiers", []) or []
        category = "development"
        subcategory = "python-package"

        for classifier in classifiers:
            for prefix, (cat, sub) in CLASSIFIER_TO_CATEGORY.items():
                if classifier.startswith(prefix):
                    category = cat
                    subcategory = sub
                    break

        # Extract license
        license_name = info.get("license") or ""
        if not license_name:
            for c in classifiers:
                if c.startswith("License ::"):
                    license_name = c.split("::")[-1].strip()
                    break

        # Extract URLs
        project_urls = info.get("project_urls") or {}
        homepage = info.get("home_page") or project_urls.get("Homepage") or project_urls.get("homepage") or ""
        repo = project_urls.get("Repository") or project_urls.get("Source") or project_urls.get("GitHub") or ""
        docs = project_urls.get("Documentation") or project_urls.get("Docs") or ""

        # Author info
        author = info.get("author") or info.get("maintainer") or ""
        author_email = info.get("author_email") or info.get("maintainer_email") or ""

        # Version
        version = info.get("version", "")

        # Python version requirement
        requires_python = info.get("requires_python") or ""

        # Dependencies
        requires_dist = info.get("requires_dist") or []
        # Extract just package names from requirement strings
        deps = []
        for req in requires_dist[:30]:  # Cap at 30
            dep_name = re.split(r'[;><=!\s\[]', req)[0].strip()
            if dep_name and "extra ==" not in req:
                deps.append(dep_name)

        # Build slug
        slug = re.sub(r'[^a-z0-9-]', '-', name.lower())
        slug = re.sub(r'-+', '-', slug).strip("-")

        biz_slug = slug
        if author:
            biz_slug = re.sub(r'[^a-z0-9-]', '-', author.lower())
            biz_slug = re.sub(r'-+', '-', biz_slug).strip("-")

        # Keywords from classifiers
        keywords = []
        for c in classifiers:
            parts = c.split("::")
            if len(parts) >= 2:
                keywords.append(parts[-1].strip().lower())

        # Data quality based on info completeness
        quality = 0.4
        if summary:
            quality += 0.15
        if description and len(description) > 50:
            quality += 0.1
        if homepage or repo:
            quality += 0.1
        if requires_dist:
            quality += 0.1
        if classifiers and len(classifiers) >= 3:
            quality += 0.1
        if license_name:
            quality += 0.05

        return {
            "name": name,
            "slug": slug,
            "source_id": name,
            "business_name": author or name,
            "business_slug": biz_slug,
            "business_website": homepage or None,
            "short_description": summary[:200] if summary else f"Python package: {name}",
            "description": description[:500] if description else summary,
            "category": category,
            "subcategory": subcategory,
            "price_model": "free",
            "price_min": 0,
            "has_free_tier": True,
            "url": f"https://pypi.org/project/{name}/",
            "data_quality": round(min(quality, 1.0), 2),
            "data": {
                "version": version,
                "license": license_name[:100] if license_name else None,
                "requires_python": requires_python,
                "dependencies": deps[:20],
                "homepage": homepage or None,
                "repository": repo or None,
                "documentation": docs or None,
                "keywords": keywords[:15],
                "author": author,
                "type": "python-package",
            },
        }


    @staticmethod
    def _fetch_top_packages(count: int = 1000) -> list[str]:
        """Fetch top PyPI packages by download count from hugovk's dataset."""
        try:
            resp = requests.get(
                "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json",
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            packages = [row["project"] for row in data["rows"][:count]]
            log.info("Fetched %d top PyPI packages from hugovk dataset", len(packages))
            return packages
        except Exception as e:
            log.warning("Failed to fetch top packages list: %s, falling back to curated list", e)
            return list(TOP_PACKAGES)


def main():
    parser = argparse.ArgumentParser(description="PyPI connector for Agent Commerce")
    parser.add_argument("--max", type=int, default=500, help="Maximum packages to fetch")
    parser.add_argument("--offset", type=int, default=0, help="Start offset")
    parser.add_argument("--top", action="store_true", help="Use top-downloaded packages list (1000 packages)")
    args = parser.parse_args()

    connector = PyPIConnector(use_top_list=args.top)
    connector.run(max_items=args.max, offset=args.offset)


if __name__ == "__main__":
    main()
