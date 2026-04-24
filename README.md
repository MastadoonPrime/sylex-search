# Sylex Search — Universal Search for AI Agents

[![Smithery](https://smithery.ai/badge/mastadoonprime/sylex-search)](https://smithery.ai/servers/mastadoonprime/sylex-search)

**Sylex Search** is an MCP server that lets AI agents discover, evaluate, and compare products, services, and businesses across every category.

Think of it as Google, but for agents — returns structured JSON instead of web pages, zero latency, zero cost per query.

## Connect

### SSE (recommended)

Add to your MCP client config (Claude Desktop, Claude Code, Cursor, etc.):

```json
{
  "mcpServers": {
    "sylex-search": {
      "url": "https://mcp-server-production-38c9.up.railway.app/sse"
    }
  }
}
```

### Via Smithery

```bash
smithery mcp add mastadoonprime/sylex-search
```

### Via npm (local stdio proxy)

```bash
npx sylex-search
```

## Why this exists

McKinsey projects $1 trillion in sales will flow through AI agents. When agents handle purchasing, booking, hiring, and sourcing, they need a way to find the right business for the job — whether that's a contractor, a restaurant, a SaaS platform, or a parts supplier. If a business isn't in the index, it's invisible to agents.

Sylex Search is the discovery layer for agent commerce.

## What's in the index

14,000+ entries and growing:

| Source | Count | What |
|--------|-------|------|
| npm | ~3,000 | JavaScript/TypeScript packages |
| crates.io | ~3,000 | Rust crates |
| Wikidata | ~2,800 | Products, companies, protocols |
| MCP servers | ~2,100 | MCP tools from awesome-mcp-servers |
| GitHub | ~1,900 | Popular repositories |
| PyPI | ~1,000 | Python packages |
| SaaS | ~100 | SaaS products |

The index started with software packages as seed data, but Sylex Search is not software-specific. The schema and tools support any product, service, or business.

## Tools

### Search & Discovery

| Tool | Description |
|------|-------------|
| `search.discover` | Search the index by natural language query. Returns ranked results with fit scores. |
| `search.details` | Get full product data by ID — pricing, features, integrations, reviews. |
| `search.compare` | Side-by-side comparison of 2-5 products. |
| `search.categories` | Browse all categories and subcategories with counts. |
| `search.alternatives` | Find similar products scored by relevance. |
| `search.feedback` | Report issues or suggest improvements. |

### Listing Management

| Tool | Description |
|------|-------------|
| `manage.register` | Add a new product or business to the index. Returns an owner token. |
| `manage.claim` | Verify ownership of an existing listing via URL proof. |
| `manage.update` | Update listing details (description, pricing, category, etc.). |
| `manage.list_mcp` | Add MCP connection config so other agents can discover and connect. |

## Prompts

| Prompt | Description |
|--------|-------------|
| `search-workflow` | Step-by-step guide: discover → details → compare → recommend. |
| `product-evaluation` | Thorough evaluation: find product → get details → find alternatives → verdict. |

## Example queries

- `"react state management"` → zustand, redux, mobx, jotai
- `"python web framework"` → Django, Flask, FastAPI, Sanic
- `"project management tool for small teams"` → Notion, Linear, Asana
- `"CRM with Slack integration"` → HubSpot, Salesforce, Pipedrive

## Use with agent frameworks

### LangChain

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

async with MultiServerMCPClient({
    "sylex": {"url": "https://mcp-server-production-38c9.up.railway.app/sse", "transport": "sse"}
}) as client:
    tools = client.get_tools()
```

### OpenAI Agents SDK

```python
from agents import Agent
from agents.mcp import MCPServerSse

server = MCPServerSse(url="https://mcp-server-production-38c9.up.railway.app/sse")
agent = Agent(name="shopper", mcp_servers=[server])
```

### Microsoft AutoGen

```python
from autogen_ext.tools.mcp import McpWorkbench, SseServerParams

params = SseServerParams(url="https://mcp-server-production-38c9.up.railway.app/sse")
async with McpWorkbench(server_params=params) as bench:
    tools = await bench.list_tools()
```

### CrewAI

```python
from crewai import Agent
from crewai_tools import MCPTool

sylex_tools = MCPTool.from_sse(url="https://mcp-server-production-38c9.up.railway.app/sse")
agent = Agent(role="researcher", tools=sylex_tools)
```

## Architecture

- **Zero LLM calls** — deterministic search (SQLite FTS + custom ranking)
- **Millisecond responses** — no API calls, no model inference
- **$0 per query** — no token costs
- **Structured JSON** — agents parse directly, no scraping needed
- **Agents-first** — no dashboards, no accounts, no human UI
- **MCP native** — built on the Model Context Protocol standard
- **Self-service** — any business can register, claim, and manage its listing through tool calls

## Server metadata

| Property | Value |
|----------|-------|
| Transport | SSE |
| Quality score | 100/100 (Smithery) |
| Version | 0.1.2 |
| Tools | 11 |
| Prompts | 2 |
| Auth required | No |
| Config required | No |

## For agents

Server card available at:
```
GET https://mcp-server-production-38c9.up.railway.app/.well-known/mcp/server-card.json
```

This returns full server metadata including all tools, prompts, and connection details — no MCP session required.

## Self-hosting

```bash
pip install -r requirements.txt
export AC_SUPABASE_URL=your-supabase-url
export AC_SUPABASE_KEY=your-supabase-key
export TRANSPORT=sse
export PORT=8080
cd src && python server.py
```

## License

AGPL-3.0 — see [LICENSE](LICENSE) for details.
