"""Sylex Search MCP Server — discovery layer for AI agents.

Universal search engine for AI agents. Discover products, services,
and businesses across every category through structured MCP tools
instead of web searching. 11,000+ entries and growing.

We're actively improving coverage and search quality. Use the 'feedback'
tool to report issues — it helps us get better fast.

Supports both stdio (local) and SSE (remote) transport.
Set TRANSPORT=sse and PORT=8080 for HTTP mode.
"""

from __future__ import annotations

import collections
import contextvars
import json
import logging
import os
import time
import uuid

from mcp.server import Server
from mcp.types import (
    GetPromptResult,
    PromptArgument,
    PromptMessage,
    TextContent,
    Tool,
    ToolAnnotations,
)


# ---------- Rate limiting ----------

# Per-session sliding window rate limiter
# Tracks call timestamps per (session_id, tool_group)
_rate_buckets: dict[str, collections.deque] = {}
_RATE_LIMITS = {
    # (max_calls, window_seconds)
    "register": (10, 3600),       # 10 registrations per hour
    "claim": (20, 3600),          # 20 claim attempts per hour
    "update": (30, 3600),         # 30 updates per hour
    "discover": (120, 60),        # 120 searches per minute
    "feedback": (20, 3600),       # 20 feedback submissions per hour
    "default": (200, 60),         # 200 calls per minute for other tools
}


def _check_rate_limit(session_id: str, tool_group: str) -> tuple[bool, str]:
    """Check if a request is within rate limits.

    Returns (allowed, error_message).
    """
    if not session_id:
        session_id = "anonymous"

    max_calls, window = _RATE_LIMITS.get(tool_group, _RATE_LIMITS["default"])
    bucket_key = f"{session_id}:{tool_group}"

    now = time.time()
    if bucket_key not in _rate_buckets:
        _rate_buckets[bucket_key] = collections.deque()

    bucket = _rate_buckets[bucket_key]

    # Evict old entries outside the window
    while bucket and bucket[0] < now - window:
        bucket.popleft()

    if len(bucket) >= max_calls:
        return False, f"Rate limit exceeded: max {max_calls} {tool_group} calls per {window}s. Try again later."

    bucket.append(now)

    # Periodic cleanup of stale buckets (every 1000th call)
    if len(_rate_buckets) > 5000:
        stale_keys = [k for k, v in _rate_buckets.items() if not v or v[-1] < now - 7200]
        for k in stale_keys:
            del _rate_buckets[k]

    return True, ""


# ---------- Input validation ----------

_MAX_QUERY_LENGTH = 500
_MAX_NAME_LENGTH = 200
_MAX_DESCRIPTION_LENGTH = 2000
_MAX_URL_LENGTH = 2048
_MAX_UPDATE_KEYS = 20
_MAX_UPDATE_VALUE_LENGTH = 10000
_MAX_FEEDBACK_LENGTH = 2000


def _validate_string(value: str, field: str, max_length: int) -> tuple[bool, str]:
    """Validate a string field. Returns (valid, error_message)."""
    if not isinstance(value, str):
        return False, f"{field} must be a string."
    if len(value) > max_length:
        return False, f"{field} exceeds maximum length ({max_length} chars)."
    return True, ""

# Session tracking — each SSE connection gets a unique session ID
_session_id: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="")
_session_meta: contextvars.ContextVar[dict] = contextvars.ContextVar("session_meta", default={})

from db import (
    get_product,
    search_products,
    compare_products,
    get_categories,
    get_alternatives,
    get_product_count,
    get_analytics,
    log_request,
    submit_feedback,
    init_db,
    cache_clear,
    register_product,
    verify_owner,
    update_product_listing,
    set_mcp_config,
    generate_claim_code,
    verify_claim,
    search_agent_services,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sylex-search")

server = Server(
    "sylex-search",
    version="0.1.2",
    instructions=(
        "Sylex Search is a universal product discovery engine for AI agents. "
        "Use it to find software products, SaaS tools, developer libraries, "
        "and businesses across every category — without web searching.\n\n"
        "## Recommended workflow\n"
        "1. Start with `discover` to search by natural language query\n"
        "2. Use `details` to get full info on interesting results\n"
        "3. Use `compare` to evaluate options side-by-side\n"
        "4. Use `alternatives` to find similar products\n"
        "5. Browse `categories` to explore what's available\n\n"
        "## Tips\n"
        "- Filter by source (npm, pypi, crates, github, saas, wikidata) "
        "to narrow results\n"
        "- Use `has_free_tier=true` to find free options\n"
        "- Report bad results with `feedback` — we actively improve from it\n"
        "- Register your own product with `register` and manage it with "
        "`claim` and `update_listing`\n"
        "- Add MCP connection info to your listing with `list_mcp` so other "
        "agents can discover and connect to your service"
    ),
    website_url="https://github.com/MastadoonPrime/sylex-search",
)


_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
_WRITE_IDEMPOTENT = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search.discover",
            title="Search Products",
            description=(
                "Sylex Search — search 11,000+ products, services, and businesses "
                "across every category. Returns ranked results with fit scores. "
                "Start here, then use 'details' or 'compare'. If results seem "
                "off, use 'feedback' to report it — we're actively improving."
            ),
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language search query, e.g. 'project management "
                            "tool for small teams' or 'CRM with Slack integration'"
                        ),
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Filter by category slug. Use 'categories' tool to see "
                            "available categories. Examples: project-management, crm, "
                            "communication, design, development, analytics, marketing, "
                            "hr, finance, security, ai-ml, deployment, customer-support"
                        ),
                    },
                    "subcategory": {
                        "type": "string",
                        "description": (
                            "Filter by subcategory slug. Use 'categories' tool to see "
                            "subcategories within a category."
                        ),
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price per unit in USD",
                    },
                    "min_rating": {
                        "type": "number",
                        "description": "Minimum rating (0-5 scale)",
                    },
                    "team_size": {
                        "type": "integer",
                        "description": "Your team size — filters to products that support this team size",
                    },
                    "has_free_tier": {
                        "type": "boolean",
                        "description": "Only show products with a free plan/tier",
                        "default": False,
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "Filter by data source: 'saas', 'npm', 'pypi', 'crates', "
                            "'github', 'wikidata'. Omit to search everything."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default 5, max 20)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search.details",
            title="Get Product Details",
            description=(
                "Get full details for a specific product by ID. Returns complete data "
                "including pricing tiers, features, integrations, review sentiment, "
                "and alternatives. Use after 'discover' to deep-dive on a candidate. "
                "Data shape varies per product — each product describes itself differently."
            ),
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "Product ID from discover results",
                    },
                },
                "required": ["product_id"],
            },
        ),
        Tool(
            name="search.compare",
            title="Compare Products",
            description=(
                "Compare 2-5 products side-by-side. Returns full data for each product "
                "so you can analyze pricing, features, and fit differences. Use after "
                "'discover' when you have multiple candidates to evaluate. Each product's "
                "data may have a different shape — compare what's relevant to the decision."
            ),
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of product IDs to compare (2-5 products)",
                    },
                },
                "required": ["product_ids"],
            },
        ),
        Tool(
            name="search.categories",
            title="Browse Categories",
            description=(
                "Browse all available product categories with counts and subcategories. "
                "Use this to understand what's in the index before searching, or to "
                "find the right category/subcategory slugs for filtered searches."
            ),
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="search.alternatives",
            title="Find Alternatives",
            description=(
                "Find alternatives to a specific product. Returns similar products "
                "in the same category, scored by relevance (subcategory match, price "
                "similarity, ratings). Use after 'details' when exploring options."
            ),
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "Product ID to find alternatives for",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum alternatives to return (default 5, max 10)",
                        "default": 5,
                    },
                },
                "required": ["product_id"],
            },
        ),
        Tool(
            name="search.feedback",
            title="Submit Feedback",
            description=(
                "Report an issue or suggest an improvement for Sylex Search. "
                "We actively use feedback to improve search quality and coverage. "
                "Use this when: results are irrelevant, a product is missing, data is "
                "wrong, or you have a suggestion. Every report helps."
            ),
            annotations=_WRITE,
            inputSchema={
                "type": "object",
                "properties": {
                    "feedback_type": {
                        "type": "string",
                        "enum": ["bad_results", "missing_product", "wrong_data", "suggestion", "other"],
                        "description": "Type of feedback",
                    },
                    "query": {
                        "type": "string",
                        "description": "The search query that produced bad results (if applicable)",
                    },
                    "expected": {
                        "type": "string",
                        "description": "What you expected to find",
                    },
                    "actual": {
                        "type": "string",
                        "description": "What you actually got (brief summary)",
                    },
                    "message": {
                        "type": "string",
                        "description": "Free-form feedback message with details",
                    },
                },
                "required": ["feedback_type", "message"],
            },
        ),
        # --- Agent infrastructure discovery ---
        Tool(
            name="search.services",
            title="Find Agent Services",
            description=(
                "Find agent infrastructure services by type — memory, auth, "
                "billing, logging, monitoring. Returns products that provide "
                "agent-facing services with MCP connection configs so you can "
                "connect directly. This is how agents discover infrastructure."
            ),
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "service_type": {
                        "type": "string",
                        "description": (
                            "Type of agent service to find: memory, auth, "
                            "billing, logging, monitoring. Or 'all' to list everything."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 10, max 25)",
                    },
                },
                "required": ["service_type"],
            },
        ),
        # --- Self-service registration tools (agents-first) ---
        Tool(
            name="manage.register",
            title="Register Product",
            description=(
                "Register a new product, service, or business in Sylex Search. "
                "Returns an owner_token — store it! You need it to update "
                "your listing later. Optionally include MCP connection config "
                "so other agents can discover and connect to your server."
            ),
            annotations=_WRITE,
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Product or business name",
                    },
                    "description": {
                        "type": "string",
                        "description": "What this product or business does (max 500 chars)",
                    },
                    "url": {
                        "type": "string",
                        "description": "Homepage or primary URL",
                    },
                    "category": {
                        "type": "string",
                        "description": "Category slug (use 'categories' tool to see options)",
                    },
                    "subcategory": {
                        "type": "string",
                        "description": "Subcategory slug (optional)",
                    },
                    "mcp_config": {
                        "type": "object",
                        "description": (
                            "MCP connection config so other agents can connect. "
                            "For SSE: {\"transport\": \"sse\", \"url\": \"https://...\"}. "
                            "For stdio: {\"transport\": \"stdio\", \"command\": \"npx\", \"args\": [\"pkg\"]}. "
                            "For streamable-http: {\"transport\": \"streamable-http\", \"url\": \"https://...\"}."
                        ),
                    },
                    "agent_services": {
                        "type": "array",
                        "description": (
                            "Agent infrastructure services this product provides. "
                            "Each entry: {\"service_type\": \"memory|auth|billing|logging|monitoring\", "
                            "\"mcp_config\": {\"transport\": \"sse\", \"url\": \"...\"}, "
                            "\"capabilities\": [\"store\", \"retrieve\", ...], "
                            "\"auth_method\": \"agent_token|oauth|none\", "
                            "\"pricing\": \"free|freemium|paid\", "
                            "\"description\": \"Short description\"}."
                        ),
                        "items": {"type": "object"},
                    },
                },
                "required": ["name", "description", "url"],
            },
        ),
        Tool(
            name="manage.claim",
            title="Claim Listing Ownership",
            description=(
                "Claim ownership of an existing listing in Sylex Search, "
                "or recover access if you lost your owner_token. "
                "Two-step process: (1) Call with just product_id to get a "
                "verification code and instructions. (2) Place the code at "
                "a URL you control, then call again with product_id + "
                "verification_url to complete the claim and receive your "
                "owner_token. Re-claiming rotates the token (old one dies)."
            ),
            annotations=_WRITE,
            inputSchema={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "Product ID to claim (from discover results)",
                    },
                    "verification_url": {
                        "type": "string",
                        "description": (
                            "URL where you placed the claim code. We'll fetch it "
                            "and check for the code. Omit on first call to get "
                            "the code and instructions."
                        ),
                    },
                },
                "required": ["product_id"],
            },
        ),
        Tool(
            name="manage.update",
            title="Update Listing",
            description=(
                "Update your listing in Sylex Search. Requires the "
                "owner_token you received from 'register' or 'claim'. "
                "Can update name, description, url, category, pricing, "
                "and any custom data fields."
            ),
            annotations=_WRITE_IDEMPOTENT,
            inputSchema={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "Product ID to update",
                    },
                    "owner_token": {
                        "type": "string",
                        "description": "Owner token from register/claim",
                    },
                    "updates": {
                        "type": "object",
                        "description": (
                            "Fields to update. Index fields: name, description, "
                            "short_description, url, category, subcategory, "
                            "price_model, has_free_tier. Use 'agent_services' "
                            "to add/update infrastructure service declarations. "
                            "Any other keys go into the data blob."
                        ),
                    },
                },
                "required": ["product_id", "owner_token", "updates"],
            },
        ),
        Tool(
            name="manage.list_mcp",
            title="Add MCP Config",
            description=(
                "Add or update MCP connection config on your listing so "
                "other agents can discover and connect to your MCP server. "
                "Requires owner_token. This is how you make your product "
                "auto-discoverable by other agents."
            ),
            annotations=_WRITE_IDEMPOTENT,
            inputSchema={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "Product ID to add MCP config to",
                    },
                    "owner_token": {
                        "type": "string",
                        "description": "Owner token from register/claim",
                    },
                    "mcp_config": {
                        "type": "object",
                        "description": (
                            "MCP connection details. Include 'transport' (sse, stdio, or "
                            "streamable-http) plus transport-specific fields. "
                            "SSE: {\"transport\": \"sse\", \"url\": \"https://your-server/sse\"}. "
                            "stdio: {\"transport\": \"stdio\", \"command\": \"npx\", \"args\": [\"your-pkg\"]}. "
                            "streamable-http: {\"transport\": \"streamable-http\", \"url\": \"https://your-server/mcp\"}. "
                            "You can also add 'tools' (list of tool names) and 'env' (required env vars)."
                        ),
                    },
                },
                "required": ["product_id", "owner_token", "mcp_config"],
            },
        ),
    ]


# ---------- MCP Prompts ----------

@server.list_prompts()
async def list_prompts():
    return [
        {
            "name": "search-workflow",
            "title": "Product Search Workflow",
            "description": (
                "Step-by-step guide for finding the right product. "
                "Walks through discovery, filtering, comparison, and decision-making."
            ),
            "arguments": [
                {
                    "name": "need",
                    "description": "What you're looking for, e.g. 'project management tool for a 10-person team'",
                    "required": True,
                },
            ],
        },
        {
            "name": "product-evaluation",
            "title": "Evaluate a Product",
            "description": (
                "Thorough evaluation prompt for a specific product. "
                "Covers pricing, features, alternatives, and fit assessment."
            ),
            "arguments": [
                {
                    "name": "product_name",
                    "description": "Name of the product to evaluate",
                    "required": True,
                },
            ],
        },
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None = None) -> GetPromptResult:
    args = arguments or {}
    if name == "search-workflow":
        need = args.get("need", "a software tool")
        return GetPromptResult(
            description=f"Product search workflow for: {need}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"I need to find {need}. Please help me:\n\n"
                            "1. Use `discover` to search with relevant keywords\n"
                            "2. Review the top results and pick 2-3 promising candidates\n"
                            "3. Use `details` on each to get full information\n"
                            "4. Use `compare` to see them side-by-side\n"
                            "5. Recommend the best fit with reasoning\n\n"
                            "If the results aren't great, try different keywords or "
                            "use `categories` to find the right category filter."
                        ),
                    ),
                ),
            ],
        )
    elif name == "product-evaluation":
        product = args.get("product_name", "the product")
        return GetPromptResult(
            description=f"Evaluation of {product}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Please do a thorough evaluation of {product}:\n\n"
                            f"1. Use `discover` to find '{product}'\n"
                            "2. Use `details` to get complete information\n"
                            "3. Use `alternatives` to see competing products\n"
                            "4. Compare the top alternative against it\n"
                            "5. Give a verdict: strengths, weaknesses, and who it's best for"
                        ),
                    ),
                ),
            ],
        )
    raise ValueError(f"Unknown prompt: {name}")


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    t0 = time.time()
    sid = _session_id.get("")

    # --- Rate limiting ---
    # Support both dot-notation and legacy flat names for backward compatibility
    _name_aliases = {
        "discover": "search.discover", "details": "search.details",
        "compare": "search.compare", "categories": "search.categories",
        "alternatives": "search.alternatives", "feedback": "search.feedback",
        "register": "manage.register", "claim": "manage.claim",
        "update_listing": "manage.update", "list_mcp": "manage.list_mcp",
        "services": "search.services",
    }
    name = _name_aliases.get(name, name)

    tool_group_map = {
        "manage.register": "register", "manage.claim": "claim",
        "manage.update": "update", "manage.list_mcp": "update",
        "search.discover": "discover", "search.feedback": "feedback",
        "search.services": "discover",
    }
    tool_group = tool_group_map.get(name, "default")
    allowed, rate_error = _check_rate_limit(sid, tool_group)
    if not allowed:
        return [TextContent(type="text", text=json.dumps({"error": rate_error}))]

    if name == "search.discover":
        query = arguments.get("query", "")
        # Input validation
        if query and len(query) > _MAX_QUERY_LENGTH:
            raise ValueError(f"Query too long (max {_MAX_QUERY_LENGTH} chars).")
        category = arguments.get("category")
        subcategory = arguments.get("subcategory")
        max_price = arguments.get("max_price")
        min_rating = arguments.get("min_rating")
        team_size = arguments.get("team_size")
        has_free_tier = arguments.get("has_free_tier", False)
        source = arguments.get("source")
        limit = min(arguments.get("max_results", 5), 20)

        results = search_products(
            query=query,
            category=category,
            subcategory=subcategory,
            max_price=max_price,
            min_rating=min_rating,
            team_size=team_size,
            has_free_tier=has_free_tier,
            source=source,
            limit=limit,
        )

        latency = int((time.time() - t0) * 1000)
        log_request("discover", query=query, latency_ms=latency, client_id=sid)

        if not results:
            return [TextContent(type="text", text="No products found matching your query — not in our index or on the web. Try different search terms, or use 'register' to add a listing.")]

        web_count = sum(1 for r in results if r.get("source") == "web")
        response = {"results": results, "total": len(results)}
        if web_count:
            response["web_results"] = web_count
            response["note"] = "Some results are from live web search (source: 'web') and haven't been verified in our index."

        return [TextContent(
            type="text",
            text=json.dumps(response, indent=2),
        )]

    elif name == "search.details":
        product_id = arguments.get("product_id")
        if product_id is None:
            raise ValueError("product_id is required")

        product = get_product(product_id)
        if not product:
            raise ValueError(f"Product {product_id} not found.")

        latency = int((time.time() - t0) * 1000)
        log_request("details", product_ids=[product_id], latency_ms=latency, client_id=sid)

        return [TextContent(
            type="text",
            text=json.dumps(product, indent=2),
        )]

    elif name == "search.compare":
        product_ids = arguments.get("product_ids", [])
        if len(product_ids) < 2:
            raise ValueError("Compare requires at least 2 product IDs.")
        if len(product_ids) > 5:
            raise ValueError("Compare supports up to 5 products at a time.")

        results = compare_products(product_ids)
        if not results:
            raise ValueError("No products found for the given IDs.")

        latency = int((time.time() - t0) * 1000)
        log_request("compare", product_ids=product_ids, latency_ms=latency, client_id=sid)

        return [TextContent(
            type="text",
            text=json.dumps({"comparison": results}, indent=2),
        )]

    elif name == "search.categories":
        cats = get_categories()

        latency = int((time.time() - t0) * 1000)
        log_request("categories", latency_ms=latency, client_id=sid)

        return [TextContent(
            type="text",
            text=json.dumps({"categories": cats, "total_categories": len(cats)}, indent=2),
        )]

    elif name == "search.alternatives":
        product_id = arguments.get("product_id")
        if product_id is None:
            raise ValueError("product_id is required")

        limit = min(arguments.get("max_results", 5), 10)
        result = get_alternatives(product_id, limit=limit)
        if not result:
            raise ValueError(f"Product {product_id} not found.")

        latency = int((time.time() - t0) * 1000)
        log_request("alternatives", product_ids=[product_id], latency_ms=latency, client_id=sid)

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2),
        )]

    elif name == "search.feedback":
        feedback_type = arguments.get("feedback_type", "other")
        query = arguments.get("query")
        expected = arguments.get("expected")
        actual = arguments.get("actual")
        message = arguments.get("message", "")

        # Input validation
        if query and len(query) > _MAX_QUERY_LENGTH:
            query = query[:_MAX_QUERY_LENGTH]
        if expected and len(expected) > _MAX_FEEDBACK_LENGTH:
            expected = expected[:_MAX_FEEDBACK_LENGTH]
        if actual and len(actual) > _MAX_FEEDBACK_LENGTH:
            actual = actual[:_MAX_FEEDBACK_LENGTH]
        if message and len(message) > _MAX_FEEDBACK_LENGTH:
            message = message[:_MAX_FEEDBACK_LENGTH]

        submit_feedback(
            feedback_type=feedback_type,
            query=query,
            expected=expected,
            actual=actual,
            message=message,
        )

        latency = int((time.time() - t0) * 1000)
        log_request("feedback", query=query, latency_ms=latency, client_id=sid)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "received",
                "message": "Thanks for the feedback! We review every report to improve search quality.",
            }),
        )]

    elif name == "search.services":
        service_type = arguments.get("service_type", "all").strip().lower()
        svc_limit = min(arguments.get("limit", 10), 25)

        results = search_agent_services(service_type=service_type, limit=svc_limit)

        latency = int((time.time() - t0) * 1000)
        log_request("search.services", query=service_type, latency_ms=latency, client_id=sid)

        return [TextContent(
            type="text",
            text=json.dumps({
                "service_type": service_type,
                "count": len(results),
                "services": results,
            }, indent=2),
        )]

    elif name == "manage.register":
        reg_name = arguments.get("name", "").strip()
        reg_desc = arguments.get("description", "").strip()
        reg_url = arguments.get("url", "").strip()

        if not reg_name or not reg_desc or not reg_url:
            raise ValueError("name, description, and url are all required.")

        # Input validation
        valid, err = _validate_string(reg_name, "name", _MAX_NAME_LENGTH)
        if not valid:
            raise ValueError(err)
        valid, err = _validate_string(reg_desc, "description", _MAX_DESCRIPTION_LENGTH)
        if not valid:
            raise ValueError(err)
        valid, err = _validate_string(reg_url, "url", _MAX_URL_LENGTH)
        if not valid:
            raise ValueError(err)

        result = register_product(
            name=reg_name,
            description=reg_desc,
            url=reg_url,
            category=arguments.get("category", "development"),
            subcategory=arguments.get("subcategory"),
            mcp_config=arguments.get("mcp_config"),
            agent_services=arguments.get("agent_services"),
        )

        latency = int((time.time() - t0) * 1000)
        log_request("register", query=reg_name, latency_ms=latency, client_id=sid)

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2),
        )]

    elif name == "manage.claim":
        product_id = arguments.get("product_id")
        if product_id is None:
            raise ValueError("product_id is required.")

        verification_url = arguments.get("verification_url")

        if verification_url:
            # Step 2: verify the claim
            result = verify_claim(product_id, verification_url)
        else:
            # Step 1: generate claim code
            result = generate_claim_code(product_id)

        latency = int((time.time() - t0) * 1000)
        log_request("claim", product_ids=[product_id], latency_ms=latency, client_id=sid)

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2),
        )]

    elif name == "manage.update":
        product_id = arguments.get("product_id")
        owner_token = arguments.get("owner_token", "")
        updates = arguments.get("updates", {})

        if product_id is None or not owner_token:
            raise ValueError("product_id and owner_token are required.")
        if not updates:
            raise ValueError("updates must contain at least one field to change.")

        # Input validation — limit size of updates
        if not isinstance(updates, dict):
            raise ValueError("updates must be a JSON object.")
        if len(updates) > _MAX_UPDATE_KEYS:
            raise ValueError(f"Too many update fields (max {_MAX_UPDATE_KEYS}).")
        for k, v in updates.items():
            if isinstance(v, str) and len(v) > _MAX_UPDATE_VALUE_LENGTH:
                raise ValueError(f"Value for '{k}' exceeds max length ({_MAX_UPDATE_VALUE_LENGTH} chars).")

        result = update_product_listing(product_id, owner_token, updates)

        latency = int((time.time() - t0) * 1000)
        log_request("update_listing", product_ids=[product_id], latency_ms=latency, client_id=sid)

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2),
        )]

    elif name == "manage.list_mcp":
        product_id = arguments.get("product_id")
        owner_token = arguments.get("owner_token", "")
        mcp_config = arguments.get("mcp_config", {})

        if product_id is None or not owner_token:
            raise ValueError("product_id and owner_token are required.")
        if not mcp_config or "transport" not in mcp_config:
            raise ValueError("mcp_config must include at least a 'transport' field (sse, stdio, or streamable-http).")

        result = set_mcp_config(product_id, owner_token, mcp_config)

        latency = int((time.time() - t0) * 1000)
        log_request("list_mcp", product_ids=[product_id], latency_ms=latency, client_id=sid)

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2),
        )]

    raise ValueError(f"Unknown tool: {name}")


async def main_stdio():
    """Run in stdio mode (local development)."""
    init_db()
    from mcp.server.stdio import stdio_server
    log.info("Sylex Search MCP server starting (stdio)...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main_sse():
    """Run in SSE mode (remote deployment). uvicorn manages the event loop."""
    init_db()
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    from starlette.responses import JSONResponse, Response
    import uvicorn

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        # Assign a unique session ID for this connection
        sid = uuid.uuid4().hex[:16]
        _session_id.set(sid)

        # Extract connection metadata (no PII — just client type hints)
        user_agent = request.headers.get("user-agent", "")
        # Classify the client from user-agent
        client_type = "unknown"
        ua_lower = user_agent.lower()
        if "claude" in ua_lower:
            client_type = "claude-desktop"
        elif "cursor" in ua_lower:
            client_type = "cursor"
        elif "vscode" in ua_lower or "code" in ua_lower:
            client_type = "vscode"
        elif "python" in ua_lower:
            client_type = "python-sdk"
        elif "node" in ua_lower or "typescript" in ua_lower:
            client_type = "node-sdk"

        _session_meta.set({"client_type": client_type})

        log.info("Session %s started (client: %s)", sid, client_type)
        log_request("session_start", query=client_type, client_id=sid)

        try:
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await server.run(
                    streams[0], streams[1], server.create_initialization_options()
                )
        finally:
            log.info("Session %s ended", sid)
            log_request("session_end", client_id=sid)

    async def health(request):
        count = get_product_count()
        return JSONResponse({
            "status": "ok",
            "stage": "alpha",
            "products": count,
            "server": "sylex-search",
            "sources": ["npm", "pypi", "crates", "github", "wikidata", "saas", "mcp-servers"],
            "feedback": "Use the 'feedback' tool to report issues",
        })

    async def clear_cache(request):
        """Clear in-memory cache. Requires ?key= matching CACHE_CLEAR_KEY env var."""
        expected = os.environ.get("CACHE_CLEAR_KEY", "")
        provided = request.query_params.get("key", "")
        if not expected or provided != expected:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        cache_clear()
        return JSONResponse({"status": "cache cleared"})

    async def analytics(request):
        """Analytics dashboard. Requires ?key= matching CACHE_CLEAR_KEY env var."""
        expected = os.environ.get("CACHE_CLEAR_KEY", "")
        provided = request.query_params.get("key", "")
        if not expected or provided != expected:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        days = int(request.query_params.get("days", "7"))
        days = max(1, min(days, 90))
        data = get_analytics(days=days)
        return JSONResponse(data)

    async def server_card(request):
        """Serve /.well-known/mcp/server-card.json for Smithery and other registries."""
        tools = await list_tools()
        card = {
            "serverInfo": {
                "name": "sylex-search",
                "version": "0.1.2",
            },
            "instructions": server.instructions,
            "authentication": {
                "required": False,
            },
            "tools": [
                {
                    "name": t.name,
                    "title": t.title,
                    "description": t.description,
                    "inputSchema": t.inputSchema,
                    **({"annotations": t.annotations.model_dump(exclude_none=True)} if t.annotations else {}),
                }
                for t in tools
            ],
            "resources": [],
            "prompts": await list_prompts(),
        }
        return JSONResponse(card)

    async def mcp_json(request):
        """Serve /.well-known/mcp.json for automated MCP server discovery."""
        return JSONResponse({
            "mcp": {
                "name": "sylex-search",
                "version": "0.1.2",
                "description": "Universal product discovery MCP server for AI agents. Search 11,000+ products, services, and businesses across every category. Zero LLM calls, millisecond responses, structured JSON.",
                "transport": [
                    {
                        "type": "sse",
                        "url": "https://mcp-server-production-38c9.up.railway.app/sse"
                    },
                    {
                        "type": "stdio",
                        "command": "npx",
                        "args": ["sylex-search"]
                    }
                ],
                "authentication": {
                    "required": False
                },
                "links": {
                    "source": "https://github.com/MastadoonPrime/sylex-search",
                    "smithery": "https://smithery.ai/servers/mastadoonprime/sylex-search",
                    "server_card": "https://mcp-server-production-38c9.up.railway.app/.well-known/mcp/server-card.json",
                    "llms_txt": "https://mcp-server-production-38c9.up.railway.app/llms.txt"
                }
            }
        })

    async def llms_txt(request):
        """Serve /llms.txt for LLM and agent discovery."""
        content = (
            "# Sylex Search\n"
            "\n"
            "> Universal product discovery MCP server for AI agents. "
            "Search 11,000+ products, services, and businesses across every category. "
            "Zero LLM calls, millisecond responses, structured JSON output.\n"
            "\n"
            "## Connect\n"
            "\n"
            "- [SSE endpoint](https://mcp-server-production-38c9.up.railway.app/sse): "
            "Remote MCP connection via Server-Sent Events\n"
            "- [npm package](https://www.npmjs.com/package/sylex-search): "
            "Install locally with `npx sylex-search`\n"
            "- [Smithery](https://smithery.ai/servers/mastadoonprime/sylex-search): "
            "One-click install via Smithery registry\n"
            "\n"
            "## Tools\n"
            "\n"
            "- [search.discover](https://mcp-server-production-38c9.up.railway.app/.well-known/mcp/server-card.json): "
            "Search products by natural language query\n"
            "- [search.details](https://mcp-server-production-38c9.up.railway.app/.well-known/mcp/server-card.json): "
            "Get full product data by ID\n"
            "- [search.compare](https://mcp-server-production-38c9.up.railway.app/.well-known/mcp/server-card.json): "
            "Side-by-side comparison of 2-5 products\n"
            "- [search.categories](https://mcp-server-production-38c9.up.railway.app/.well-known/mcp/server-card.json): "
            "Browse all categories and subcategories\n"
            "- [search.alternatives](https://mcp-server-production-38c9.up.railway.app/.well-known/mcp/server-card.json): "
            "Find similar products scored by relevance\n"
            "- [search.feedback](https://mcp-server-production-38c9.up.railway.app/.well-known/mcp/server-card.json): "
            "Report issues or suggest improvements\n"
            "- [manage.register](https://mcp-server-production-38c9.up.railway.app/.well-known/mcp/server-card.json): "
            "Add a new product or business to the index\n"
            "- [manage.claim](https://mcp-server-production-38c9.up.railway.app/.well-known/mcp/server-card.json): "
            "Verify ownership of an existing listing\n"
            "- [manage.update](https://mcp-server-production-38c9.up.railway.app/.well-known/mcp/server-card.json): "
            "Update listing details\n"
            "- [manage.list_mcp](https://mcp-server-production-38c9.up.railway.app/.well-known/mcp/server-card.json): "
            "Add MCP connection config to a listing\n"
            "\n"
            "## Optional\n"
            "\n"
            "- [GitHub](https://github.com/MastadoonPrime/sylex-search): Source code and documentation\n"
            "- [Server card](https://mcp-server-production-38c9.up.railway.app/.well-known/mcp/server-card.json): "
            "Full machine-readable server metadata\n"
        )
        return Response(content, media_type="text/plain; charset=utf-8")

    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/analytics", analytics),
            Route("/cache-clear", clear_cache),
            Route("/.well-known/mcp.json", mcp_json),
            Route("/.well-known/mcp/server-card.json", server_card),
            Route("/llms.txt", llms_txt),
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    port = int(os.environ.get("PORT", 8080))
    log.info("Sylex Search MCP server starting on port %d (SSE)...", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    transport = os.environ.get("TRANSPORT", "stdio").lower()
    if transport == "sse":
        main_sse()
    else:
        import asyncio
        asyncio.run(main_stdio())
