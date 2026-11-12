#!/usr/bin/env bash
# AGFHC MCP server. Dry-run (simulated node only) unless AGFHC_MODE=live is set in agents.env.
set -a; [ -f /home/mitac/Multi-Agents/agents.env ] && . /home/mitac/Multi-Agents/agents.env; set +a
export AGFHC_MODE="${AGFHC_MODE:-dryrun}"
cd /home/mitac/Multi-Agents/tools/agfhc
exec /home/mitac/Multi-Agents/.venv/bin/python server.py 2>>/home/mitac/Multi-Agents/logs/agfhc.log
