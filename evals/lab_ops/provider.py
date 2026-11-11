"""Promptfoo provider: run the real Lab Ops agent (live tools) for one case.

Turns in a multi-turn case are separated by "||" in the question.
Output is JSON: {"answer": <final text>, "calls": [{"name": ..., "args": {...}}, ...]}.
"""
import asyncio
import json
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


def call_api(prompt, options, context):
    turns = [t.strip() for t in prompt.split("||") if t.strip()]
    return {"output": json.dumps(asyncio.run(_run(turns)))}
