#!/usr/bin/env bash
# SearXNG MCP server (stdio). uvx is installed in the project venv, which is not on PATH when the agents run as
# a systemd service or from a shell without the venv activated; look there first.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
UVX=""
for c in "$ROOT/.venv/bin/uvx" "$(command -v uvx 2>/dev/null)" "$HOME/.local/bin/uvx"; do
  [ -n "$c" ] && [ -x "$c" ] && { UVX=$c; break; }
done
mkdir -p "$ROOT/logs"
[ -n "$UVX" ] || { echo "run_searxng_mcp.sh: uvx not found (.venv/bin, PATH, ~/.local/bin)" >>"$ROOT/logs/searxng-mcp.log"; exit 127; }
exec "$UVX" --with fastmcp==2.12.4 searxng-mcp-server --searxng-url "${SEARXNG_URL:-http://127.0.0.1:8888}" \
  2>>"$ROOT/logs/searxng-mcp.log"
