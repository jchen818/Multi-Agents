#!/usr/bin/env bash
exec uvx --with fastmcp==2.12.4 searxng-mcp-server --searxng-url http://127.0.0.1:8888 2>>/home/mitac/Multi-Agents/logs/searxng-mcp.log
