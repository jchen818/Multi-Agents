#!/usr/bin/env bash
exec sudo -n -u redfish-agent /usr/local/bin/redfish-mcp-agents 2>>/home/mitac/Multi-Agents/logs/redfish-mcp.log
