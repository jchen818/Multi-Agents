"""Bake-off implementation 1: LangGraph (langchain create_agent + HumanInTheLoopMiddleware), as used in this repo.

    run_case(case, model) -> {"final": str, "ctx": Ctx, "model_calls": int}
"""
import uuid

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ModelCallLimitMiddleware
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from bakeoff import common as C


def tools_for(ctx):
    def mk(fn, name):
        return StructuredTool.from_function(lambda **kw: fn(ctx, **kw), name=name, description=C.TOOL_DOCS[name],
                                            args_schema=_schema(fn))
    return [mk(C.read_run, "read_run"), mk(C.grep_logs, "grep_logs"), mk(C.search_bugs, "search_bugs"),
            mk(C.field_values, "field_values"), mk(C.file_bug, "file_bug")]


def _schema(fn):
    import inspect
    from pydantic import create_model
    params = list(inspect.signature(fn).parameters.values())[1:]      # drop ctx
    return create_model(f"{fn.__name__}_args", **{p.name: (str, ...) for p in params})


async def run_case(case: dict, model, callbacks=None) -> dict:
    ctx = C.make_ctx(case)
    agent = create_agent(model, tools_for(ctx), system_prompt=C.SYSTEM, checkpointer=InMemorySaver(),
                         middleware=[HumanInTheLoopMiddleware({"file_bug": {"allowed_decisions": ["approve", "reject"]}}),
                                     ModelCallLimitMiddleware(run_limit=C.MAX_TURNS, exit_behavior="end")])
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}, "callbacks": callbacks or [],
           "recursion_limit": 4 * C.MAX_TURNS + 10}
    out = await agent.ainvoke({"messages": [("user", f"File a bug for the failed AGFHC run {case['id']} if one "
                                                     "is needed.")]}, cfg)
    while "__interrupt__" in out:
        req = out["__interrupt__"][0].value
        decisions = []
        for action in req["action_requests"]:
            ok, msg = C.review(ctx, action["args"])
            decisions.append({"type": "approve"} if ok else {"type": "reject", "message": msg})
        out = await agent.ainvoke(Command(resume={"decisions": decisions}), cfg)
    msgs = out["messages"]
    final = msgs[-1].content if msgs else ""
    final = final if isinstance(final, str) else " ".join(b.get("text", "") for b in final if isinstance(b, dict))
    return {"final": final, "ctx": ctx, "model_calls": sum(1 for m in msgs if m.type == "ai")}
