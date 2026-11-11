import asyncio
from langchain_core.tools import StructuredTool
import core.guard as g

SCHEMA = {"type": "object", "properties": {"url": {"type": "string"}}}

def fake(name, coro):
    return StructuredTool(name=name, description="test", args_schema=SCHEMA,
                          coroutine=coro, response_format="content_and_artifact")

def run(tool, args=None):
    return asyncio.run(tool.ainvoke(args or {}))

def test_timeout():
    async def slow(**kw): await asyncio.sleep(2); return ("x", None)
    g.DEFAULT_TIMEOUT = 0.5
    assert "timed out" in run(g.guard_tool(fake("slow", slow), "other"))

def test_cap():
    async def big(**kw): return ("A" * 20000, None)
    out = run(g.guard_tool(fake("big", big), "other"))
    assert "truncated" in out and len(out) < 9000

def test_error_becomes_text():
    async def boom(**kw): raise RuntimeError("bmc 500")
    assert "ERROR" in run(g.guard_tool(fake("boom", boom), "other"))

def test_fetch_url_blocks_lab():
    async def ok(**kw): return ("fetched", None)
    t = g.guard_tool(fake("fetch_url", ok), "searxng")
    for url in ("https://10.46.252.13/redfish/v1", "http://127.0.0.1:6333", "http://localhost:3001", "file:///etc/passwd"):
        assert "REFUSED" in run(t, {"url": url}), url
    assert g.blocked_url("https://rocm.docs.amd.com/") is None

def test_kill_switch(tmp_path):
    async def ok(**kw): return ("fine", None)
    g.KILL_FILE = tmp_path / "KILL"
    t = g.guard_tool(fake("t", ok), "other")
    assert "fine" in run(t)
    g.KILL_FILE.touch()
    assert "REFUSED" in run(t)

def test_middleware_imports():
    assert len(g.limit_middleware()) == 2

