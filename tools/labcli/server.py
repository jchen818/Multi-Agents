"""labcli: read-only SSH + IPMI tools for test nodes (stdio MCP server)."""
import asyncio, os, shlex
from asyncio.subprocess import DEVNULL, PIPE
from pathlib import Path
import yaml
from mcp.server.fastmcp import FastMCP

ROOT = Path(os.environ.get("MULTI_AGENTS_ROOT", "/home/mitac/Multi-Agents"))
from functools import lru_cache

@lru_cache
def nodes() -> dict:
    return yaml.safe_load(open(ROOT / "config/nodes.yaml"))["nodes"]

@lru_cache
def secrets() -> dict:
    f = ROOT / "secrets/ipmi.yaml"
    return (yaml.safe_load(open(f)) or {}) if f.exists() else {}
FORBIDDEN = set(";|&><`$(){}\\*?!\n")

def _deny(short="", long=()):
    def rule(args):
        for a in args:
            if a.startswith("--") and any(a.startswith(l) for l in long):
                return f"flag {a} is not allowed"
            if a.startswith("-") and not a.startswith("--") and any(c in short for c in a[1:]):
                return f"flag {a} is not allowed"
    return rule

def _paths(*prefixes):
    def rule(args):
        for a in args:
            if a.startswith("-"):
                continue
            if ".." in a or not a.startswith(prefixes):
                return f"path {a} is not allowed (must start with {', '.join(prefixes)})"
    return rule

def _amd_smi(args):
    ok = {"version", "list", "static", "metric", "firmware", "bad-pages", "xgmi", "topology", "process", "partition"}
    if not args or args[0] not in ok:
        return f"amd-smi subcommand must be one of {sorted(ok)}"

def _ip(args):
    bad = {"add", "del", "delete", "set", "flush", "change", "replace", "append", "up", "down"}
    if bad & set(args):
        return "ip may only show state"
    words = [a for a in args if not a.startswith("-")]
    if words and words[0] not in {"addr", "address", "a", "link", "l", "route", "r", "neigh", "n"}:
        return "ip object must be addr, link, route or neigh"

def _ethtool(args):
    if len(args) == 1 and not args[0].startswith("-"):
        return None
    if len(args) == 2 and args[0] in {"-i", "-S", "-m", "-k"}:
        return None
    return "ethtool: only <iface> or -i/-S/-m/-k <iface>"

RULES = {
    **{p: None for p in ("uptime", "uname", "hostname", "free", "df", "lsblk", "lsmod", "lspci", "sensors", "nproc")},
    "dmesg": _deny(short="cCDEnwW", long=("--clear", "--read-clear", "--console", "--follow")),
    "journalctl": _deny(short="f", long=("--follow", "--vacuum", "--rotate", "--flush", "--sync",
                                          "--relinquish", "--smart-relinquish", "--setup-keys", "--update-catalog")),
    "rocm-smi": _deny(short="r", long=("--set", "--reset", "--load", "--save", "--gpureset", "--ras", "--autorespond")),
    "amd-smi": _amd_smi,
    "cat": _paths("/proc/", "/sys/", "/etc/os-release"),
    "ls": _paths("/proc/", "/sys/", "/dev/", "/opt/rocm", "/var/log/"),
    "ip": _ip,
    "ethtool": _ethtool,
}

def check_command(command: str) -> list[str]:
    """Return argv if the command is allowed, else raise ValueError."""
    if any(ch in FORBIDDEN for ch in command):
        raise ValueError("shell metacharacters (; | & > < ` $ ( ) etc.) are not allowed")
    argv = shlex.split(command)
    if not argv:
        raise ValueError("empty command")
    prog, args = argv[0], argv[1:]
    if prog not in RULES:
        raise ValueError(f"'{prog}' is not allowed. Allowed: {', '.join(sorted(RULES))}")
    why = RULES[prog](args) if RULES[prog] else None
    if why:
        raise ValueError(why)
    return argv

async def _run(cmd, env=None, timeout=60) -> str:
    proc = await asyncio.create_subprocess_exec(*cmd, stdin=DEVNULL, stdout=PIPE, stderr=PIPE, env=env)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return f"ERROR: timed out after {timeout}s"
    text = f"exit={proc.returncode}\n{out.decode(errors='replace')}"
    if err.strip():
        text += f"\n[stderr]\n{err.decode(errors='replace')[-2000:]}"
    return text

def _node(name):
    if name not in nodes():
        raise ValueError(f"unknown node '{name}'. Configured: {', '.join(sorted(nodes()))}")
    return nodes()[name]

mcp = FastMCP("labcli")

@mcp.tool()
def list_nodes() -> str:
    """List test nodes the agent may inspect (name, SSH host, BMC host). No secrets."""
    return yaml.safe_dump({k: {"ssh_host": v["ssh_host"], "bmc_host": v["bmc_host"]} for k, v in nodes().items()})

@mcp.tool()
async def run_readonly_cmd(node: str, command: str) -> str:
    """Run one allowlisted read-only command on a test node over SSH, e.g. 'amd-smi metric',
    'rocm-smi --showtemp', 'dmesg -T --level=err,warn', 'journalctl -k -n 200 --no-pager', 'lspci -nn'.
    No pipes, redirects or command chaining."""
    try:
        n, argv = _node(node), check_command(command)
    except ValueError as e:
        return f"REFUSED: {e}"
    return await _run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes",
                       f"{n['ssh_user']}@{n['ssh_host']}", "--", shlex.join(argv)])

async def _ipmi(node, *args):
    n = _node(node)
    s = secrets().get(node)
    if not s:
        return f"ERROR: no IPMI credentials configured for {node}"
    env = {"PATH": "/usr/bin:/bin", "IPMI_PASSWORD": s["ipmi_password"]}
    return await _run(["ipmitool", "-I", "lanplus", "-H", n["bmc_host"], "-U", s["ipmi_user"], "-E", *args],
                      env=env, timeout=45)

@mcp.tool()
async def get_ipmi_sel(node: str, last: int = 50) -> str:
    """Read the last N System Event Log entries (default 50, max 500) from a test node's BMC."""
    try:
        return await _ipmi(node, "sel", "elist", "last", str(max(1, min(last, 500))))
    except ValueError as e:
        return f"REFUSED: {e}"

@mcp.tool()
async def get_ipmi_fru(node: str) -> str:
    """Read FRU inventory (board, product, serial numbers) from a test node's BMC."""
    try:
        return await _ipmi(node, "fru", "print")
    except ValueError as e:
        return f"REFUSED: {e}"

if __name__ == "__main__":
    mcp.run()   # stdio

