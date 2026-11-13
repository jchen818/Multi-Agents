"""Promptfoo provider: run the real Lab Ops agent (live tools) for one case.

Turns in a multi-turn case are separated by "||" in the question.
Output is JSON: {"answer": <final text>, "calls": [{"name": ..., "args": {...}}, ...]}.
"""
import asyncio
import importlib.util
import json
import re
import sys
import uuid

sys.path.insert(0, "/home/mitac/Multi-Agents")

from contextlib import AsyncExitStack

from langfuse.langchain import CallbackHandler

from agents.lab_ops import build
from core.guard import run_with_deadline, to_text


async def _run(turns):
    async with AsyncExitStack() as stack:
        agent, *_ = await build(stack)
        thread = f"eval-{uuid.uuid4().hex[:8]}"
        cfg = {
            "callbacks": [CallbackHandler()],
            "configurable": {"thread_id": thread},
            "metadata": {"langfuse_session_id": thread, "langfuse_tags": ["eval-lab-ops"]},
            "run_name": "lab-ops-eval",
        }
        calls, answer = [], ""
        for q in turns:
            before = len((await agent.aget_state(cfg)).values.get("messages", []))
            res = await run_with_deadline(agent, {"messages": [("user", q)]}, cfg)
            for m in res["messages"][before:]:
                calls += [{"name": c["name"], "args": c["args"]}
                          for c in (getattr(m, "tool_calls", None) or [])]
            answer = to_text(res["messages"][-1].content)
        return {"answer": answer, "calls": calls}


def _labcli():
    spec = importlib.util.spec_from_file_location("labcli_eval", "/home/mitac/Multi-Agents/tools/labcli/server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ground_truth(v: dict) -> str:
    """Versions change with BKC/ROCm upgrades, so read the true value live (same read-only labcli command)
    instead of pinning it in the test file. live_cmd: command on live_node; live_re: what to extract."""
    if not v.get("live_cmd"):
        return ""
    out = asyncio.run(_labcli().run_readonly_cmd(v.get("live_node", "mitacai"), v["live_cmd"]))
    m = re.search(v.get("live_re", r"\S+"), out or "")
    return m.group(0) if m else ""


def call_api(prompt, options, context):
    turns = [t.strip() for t in prompt.split("||") if t.strip()]
    out = asyncio.run(_run(turns))
    out["expected"] = ground_truth((context or {}).get("vars") or {})
    return {"output": json.dumps(out)}
