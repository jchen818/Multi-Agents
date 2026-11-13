"""Promptfoo provider: run the real Phase 2 planner graph (gemma, agfhc MCP server in dry-run) for one case.

vars: node (default sim), decisions "reject" | "approve" | "edit:a,b;reject" (one per approval, ';'-separated),
      scenario (SIM_SCENARIO for the simulated run), seed (requirements of a recipe to pre-save: steps from seed_steps).
Output JSON: {status, tier, steps, based_on, approvals: [markdown...], report, recipe_saved, all_allowed, seed_id}.
Each case gets its own temporary recipe store (file lookup, no Mem0), so cases are repeatable and write nothing real.
"""
import asyncio
import json
import os
import sys
import tempfile
import uuid
from contextlib import AsyncExitStack
from pathlib import Path

ROOT = "/home/mitac/Multi-Agents"
sys.path.insert(0, ROOT)

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agents.test_planner.graph import build_graph, template
from core.guard import guard_tool
from core.mcp import ALLOWED, SERVERS, passthrough_env
from core.models import get_model
from core.recipes import RecipeStore


def decision(d: str) -> dict:
    d = d.strip()
    if d.startswith("edit:"):
        return {"action": "edit", "recipes": [r.strip() for r in d[5:].split(",") if r.strip()]}
    if d == "approve":
        return {"action": "approve"}
    return {"action": "reject", "reason": "eval"}


async def _run(v: dict) -> dict:
    os.environ["SIM_SCENARIO"] = v.get("scenario") or "fail"
    os.environ.setdefault("SIM_SLEEP", "2")
    store = RecipeStore(Path(tempfile.mkdtemp(prefix="eval-recipes-")))
    seed_id = ""
    if v.get("seed"):
        steps = [s for s in v.get("seed_steps", "all_lvl1").split("|")]
        rec = store.save(platform="mi325x", bkc="SIM-BIOS-1.0", requirements=v["seed"], node="sim", dryrun=True,
                         plan={"tier": v.get("seed_tier", "smoke"), "steps": [{"recipe": s, "reason": "seed"} for s in steps]},
                         results=[{"recipe": s, "run_id": "seed", "verdict": "PASS"} for s in steps], versions={})
        seed_id = rec["id"]
    client = MultiServerMCPClient({"agfhc": dict(SERVERS["agfhc"], env=passthrough_env())})   # env read per case
    thread = f"eval-planner-{uuid.uuid4().hex[:8]}"
    cfg = {"configurable": {"thread_id": thread}, "callbacks": [CallbackHandler()],
           "metadata": {"langfuse_session_id": thread, "langfuse_tags": ["eval-planner"]}, "run_name": "planner-eval"}
    async with AsyncExitStack() as stack:
        session = await stack.enter_async_context(client.session("agfhc"))
        loaded = [guard_tool(t, "agfhc") for t in await load_mcp_tools(session) if t.name in ALLOWED["agfhc"]]
        tools = {t.name: (lambda t: (lambda **kw: t.ainvoke(kw)))(t) for t in loaded}
        graph = build_graph(tools, get_model("planner"), get_model("executor"), InMemorySaver(),
                            poll_s=2, recipes=store)
        decisions = [d for d in (v.get("decisions") or "reject").split(";") if d.strip()]
        out = await graph.ainvoke({"requirements": v["requirements"], "node": v.get("node") or "sim"}, cfg)
        approvals = []
        while "__interrupt__" in out:
            approvals.append(out["__interrupt__"][0].value["markdown"])
            d = decisions.pop(0) if decisions else "reject"
            out = await graph.ainvoke(Command(resume=decision(d)), cfg)
    plan = out.get("plan") or {}
    allowed = set(template("mi325x")["allowed_recipes"])
    steps = [s["recipe"] for s in plan.get("steps") or []]
    return {"status": out.get("status"), "tier": plan.get("tier"), "steps": steps, "based_on": plan.get("based_on", ""),
            "approvals": approvals, "report": out.get("report", ""), "recipe_saved": out.get("recipe_saved", ""),
            "all_allowed": all(s in allowed for s in steps), "seed_id": seed_id}


def call_api(prompt, options, context):
    return {"output": json.dumps(asyncio.run(_run(context["vars"])))}
