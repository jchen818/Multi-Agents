#!/usr/bin/env bash
# SearXNG MCP server (stdio). uvx lives in ~/.local/bin, which a systemd service's PATH does not include.
UVX=$(command -v uvx || true)
[ -n "$UVX" ] || UVX="$HOME/.local/bin/uvx"
mkdir -p /home/mitac/Multi-Agents/logs
exec "$UVX" --with fastmcp==2.12.4 searxng-mcp-server --searxng-url "${SEARXNG_URL:-http://127.0.0.1:8888}" \
  2>>/home/mitac/Multi-Agents/logs/searxng-mcp.log
