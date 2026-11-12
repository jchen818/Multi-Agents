import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

TOOLS = "/home/mitac/Multi-Agents/tools"


NO_PASSTHROUGH = {"AGFHC_MODE"}   # dry-run vs live is set in agents.env only, never from a command line


def passthrough_env(prefixes=("AGFHC_", "SIM_")) -> dict:
    """Variables a stdio server should see from the caller. The MCP client otherwise passes only HOME, PATH,
    USER and similar, so e.g. SIM_SCENARIO=pass on the planner's command line would not reach the agfhc server."""
    return {k: v for k, v in os.environ.items() if k.startswith(prefixes) and k not in NO_PASSTHROUGH}


SERVERS = {
    "searxng": {"transport": "stdio", "command": f"{TOOLS}/run_searxng_mcp.sh", "args": []},
    "redfish": {"transport": "stdio", "command": f"{TOOLS}/run_redfish_mcp.sh", "args": []},
    "labcli": {"transport": "stdio", "command": f"{TOOLS}/run_labcli_mcp.sh", "args": []},
    "agfhc": {"transport": "stdio", "command": f"{TOOLS}/run_agfhc_mcp.sh", "args": [],
              "env": passthrough_env()},   # Phase 2; not given to Lab Ops
}
ALLOWED = {
    "searxng": {"search_web", "search_news", "fetch_url"},
    "redfish": {"list_bmcs", "list_systems", "get_system", "list_chassis",
                "get_health_summary", "get_thermal", "get_power",   # get_sensors removed: 140+ requests, hits 180 s timeout
                "get_component_inventory", "get_firmware_inventory",
                "get_boot_config", "get_log_entries"},
    "labcli": {"list_nodes", "run_readonly_cmd", "get_ipmi_sel", "get_ipmi_fru","get_ipmi_sensors"},
    "agfhc": {"list_test_nodes", "preflight", "list_recipe_tests", "plan_run", "start_run",
              "get_run_status", "analyze_run", "finalize_report"},
}
client = MultiServerMCPClient(SERVERS)

async def get_tools(servers=SERVERS):
    out = []
    for name in servers:
        out += [t for t in await client.get_tools(server_name=name) if t.name in ALLOWED[name]]
    return out

# Agent runs: one session per server, e.g.
# async with client.session("redfish") as s:
#     tools = [t for t in await load_mcp_tools(s) if t.name in ALLOWED["redfish"]]

