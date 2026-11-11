"""Shared safety layer: wrap MCP tools and agent runs."""
import asyncio, ipaddress, os, re, socket, time
from pathlib import Path
from urllib.parse import urlparse

MAX_OUTPUT_CHARS = int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "8000"))
DEFAULT_TIMEOUT = 60
TIMEOUTS = {"redfish": 180, "searxng": 30}
KILL_FILE = Path(os.getenv("AGENTS_KILL_FILE", "/home/mitac/Multi-Agents/state/KILL"))

def killed() -> bool:
    return KILL_FILE.exists()

def to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)

def cap(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head, tail = int(limit * 0.7), int(limit * 0.3)
    return f"{text[:head]}\n[... truncated {len(text) - head - tail} chars ...]\n{text[-tail:]}"

def blocked_url(url: str):
    """Reason string if the URL must not be fetched, else None."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return "only http(s) URLs with a host are allowed"
    try:
        addrs = {ai[4][0] for ai in socket.getaddrinfo(p.hostname, None)}
    except socket.gaierror:
        return f"cannot resolve {p.hostname}"
    for a in addrs:
        ip = ipaddress.ip_address(a)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return f"{p.hostname} resolves to internal address {a}"
    return None

def guard_tool(tool, server: str):
    """Wrap a tool in place: kill switch, URL policy, timeout, output cap, errors as text."""
    orig = tool.coroutine
    timeout = TIMEOUTS.get(server, DEFAULT_TIMEOUT)
    pair = getattr(tool, "response_format", "content") == "content_and_artifact"
    ret = (lambda text, art=None: (text, art)) if pair else (lambda text, art=None: text)

    async def guarded(**kwargs):
        if killed():
            return ret("REFUSED: agents are stopped (kill switch is set)")
        if tool.name == "fetch_url":
            why = blocked_url(str(kwargs.get("url", "")))
            if why:
                return ret(f"REFUSED: {why}")
        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(orig(**kwargs), timeout=timeout)
        except asyncio.TimeoutError:
            return ret(f"ERROR: {tool.name} timed out after {timeout}s")
        except Exception as e:
            return ret(f"ERROR: {tool.name} failed: {type(e).__name__}: {e}"[:2000])
        content, artifact = result if isinstance(result, tuple) else (result, None)
        body = cap(to_text(content))
        return ret(f'<tool_output tool="{tool.name}" seconds="{time.monotonic() - t0:.1f}">\n{body}\n</tool_output>',
                   artifact)

    tool.coroutine = guarded
    return tool

async def endpoint_names(tools) -> list[str]:
    """Call list_bmcs once; return the configured endpoint names for the system prompt."""
    lb = next(t for t in tools if t.name == "list_bmcs")
    return re.findall(r'"endpoint":\s*"([^"]+)"', to_text(await lb.ainvoke({})))

def limit_middleware():
    from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
    steps = int(os.getenv("MAX_STEPS", "15"))
    return [ModelCallLimitMiddleware(run_limit=steps, exit_behavior="end"),
            ToolCallLimitMiddleware(run_limit=steps * 2, exit_behavior="end")]

async def run_with_deadline(agent, inputs, config):
    return await asyncio.wait_for(agent.ainvoke(inputs, config=config),
                                  timeout=int(os.getenv("MAX_RUN_SECONDS", "1800")))

