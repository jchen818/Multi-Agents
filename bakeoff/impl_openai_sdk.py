"""Bake-off implementation 2: OpenAI Agents SDK (Agent + function_tool(needs_approval=True) + RunState approvals),
on the same local vLLM model through its OpenAI-compatible Chat Completions API. Tracing to OpenAI is turned off.

The SDK's import name is `agents`, the same as this repo's package. bakeoff/run.py is started as a script, so the
repo root is only appended to sys.path and the installed SDK wins; with the repo root first, `import agents` would
load this repo's package instead.

    run_case(case, model_name, base_url) -> {"final": str, "ctx": Ctx, "model_calls": int}
"""
import json

from agents import Agent, OpenAIChatCompletionsModel, RunContextWrapper, Runner, function_tool, set_tracing_disabled
from openai import AsyncOpenAI

from bakeoff import common as C

set_tracing_disabled(True)          # the SDK exports traces to OpenAI by default; nothing leaves the lab


@function_tool(description_override=C.TOOL_DOCS["read_run"])
def read_run(w: RunContextWrapper[C.Ctx]) -> str:
    return C.read_run(w.context)


@function_tool(description_override=C.TOOL_DOCS["grep_logs"])
def grep_logs(w: RunContextWrapper[C.Ctx], pattern: str) -> str:
    return C.grep_logs(w.context, pattern)


@function_tool(description_override=C.TOOL_DOCS["search_bugs"])
def search_bugs(w: RunContextWrapper[C.Ctx], query: str) -> str:
    return C.search_bugs(w.context, query)


@function_tool(description_override=C.TOOL_DOCS["field_values"])
def field_values(w: RunContextWrapper[C.Ctx]) -> str:
    return C.field_values(w.context)


@function_tool(description_override=C.TOOL_DOCS["file_bug"], needs_approval=True)
def file_bug(w: RunContextWrapper[C.Ctx], summary: str, description: str, product: str, component: str,
             severity: str) -> str:
    return C.file_bug(w.context, summary, description, product, component, severity)


def make_agent(model) -> Agent:
    return Agent(name="bug-filer", instructions=C.SYSTEM, model=model,
                 tools=[read_run, grep_logs, search_bugs, field_values, file_bug])


def local_model(name: str, base_url: str):
    return OpenAIChatCompletionsModel(model=name, openai_client=AsyncOpenAI(base_url=base_url, api_key="EMPTY"))


async def run_case(case: dict, model) -> dict:
    ctx = C.make_ctx(case)
    agent = make_agent(model)
    result = await Runner.run(agent, f"File a bug for the failed AGFHC run {case['id']} if one is needed.",
                              context=ctx, max_turns=C.MAX_TURNS)
    calls = len(result.raw_responses)
    while result.interruptions:
        state = result.to_state()
        for item in result.interruptions:
            args = json.loads(getattr(item.raw_item, "arguments", "{}") or "{}")
            ok, msg = C.review(ctx, args)
            if ok:
                state.approve(item)
            else:
                state.reject(item, rejection_message=msg)
        result = await Runner.run(agent, state, context=ctx, max_turns=C.MAX_TURNS)
        calls += len(result.raw_responses)
    return {"final": str(result.final_output or ""), "ctx": ctx, "model_calls": calls}
