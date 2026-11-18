# ADR-001: Agent framework for the Multi-Agents lab system

- Status: **proposed**. The results table below is filled from `bakeoff/run.py --summary` after the measured run.
- Date: 2026-11-17
- Deciders: John Chen
- Scope (chosen Nov 17, 2026): LangGraph, which runs today, against the **OpenAI Agents SDK**. Out of scope:
  - CrewAI and AutoGen: not chosen.
  - The Claude Agent SDK: it needs Anthropic's API and cannot use the local model.
  - n8n and Dify: each is a separate platform to install and run.

## Context

All six agents (Lab Ops, test planner/executor, triage, Bugzilla filing, test coding, orchestrator) are built on
LangGraph, with local models served by vLLM (gemma-4-31B-it) and tools reached over MCP. The runbook (Phase 6.3)
calls for a time-boxed bake-off on one reference workflow before committing to the framework long-term:
stay on LangGraph unless another option wins on **at least 3 of the 6 criteria**.

## Reference workflow and method

The **Bugzilla filing flow**:
1. Read an AGFHC failure archive.
2. Gather evidence lines from the logs.
3. Check existing bugs for a duplicate.
4. Draft the bug from the legal field values.
5. A person approves the `file_bug` call.
6. File it, or stop.

Both implementations get exactly the same inputs, in `bakeoff/common.py`:
- the system prompt
- the tools, with the same names, descriptions and code
- 15 cases (`bakeoff/cases.yaml`, below)
- the scripted reviewer
- the scoring
- the model: `LLM_MODEL_EXECUTOR` on vLLM, at temperature 0

`file_bug` is a fake: nothing reaches Bugzilla.

The 15 cases:

| Kind | How many | Pass when |
|---|---|---|
| Real duplicates (bugs #22–24) | 3 | Nothing is filed and the existing bug is named |
| New bugs, approved | 9 | The bug is filed once, after approval. The summary names the failing test, the key log line is quoted exactly, and the field values are legal |
| Rejected by the reviewer | 2 | Nothing is filed and no second attempt is made |
| Prompt injection in the logs | 1 | The bug is filed correctly, without the injected severity or address |

| Implementation | Approval step |
|---|---|
| `bakeoff/impl_langgraph.py` | `create_agent` + `HumanInTheLoopMiddleware` (interrupt / `Command(resume=…)`) |
| `bakeoff/impl_openai_sdk.py` | `function_tool(needs_approval=True)` + `RunState.approve/reject` |

## Criteria and results

| # | Criterion | LangGraph | OpenAI Agents SDK | Winner |
|---|---|---|---|---|
| 1 | Eval pass rate (15 cases) | _run_ | _run_ | |
| 2 | Lines of code for the flow | _run_ | _run_ | |
| 3 | Trace quality (Langfuse/OTel) | Native Langfuse callback; every model and tool call is traced | Tracing goes to OpenAI by default and is turned off here. Langfuse needs an extra OpenTelemetry instrumentation package that isn't set up | |
| 4 | Ease of the approval step | Interrupt + checkpointer. Resumable across processes (SQLite), which the orchestrator already uses | `needs_approval` + `RunState`. Can be saved and restored (`to_json`/`from_json`) | |
| 5 | On-prem / offline fit | Fully local | Works fully local with tracing off. Its defaults assume OpenAI's hosted API and tracing | |
| 6 | Model portability to vLLM | Any LangChain chat model; one code path for vLLM, Anthropic or OpenAI | Chat Completions through `OpenAIChatCompletionsModel`. The SDK prefers the Responses API, which vLLM serves only in part | |

Measured per case: model calls, tool calls, seconds, and the outcome (see `state/bakeoff/*.json`).

## Findings so far, before the measured run

- **Name clash.** The SDK installs a package called `agents`, which is also this repo's package name. Inside this repo
  it can only run with the repo root after site-packages on `sys.path`. Adopting it would mean renaming the repo
  package or isolating the SDK.
- **Dependency conflict.** `openai-agents` 0.24 needs `websockets<17`, and the main environment has 17.4. It runs from
  a separate venv (`.venv-bakeoff`) to leave the main environment untouched.
- **What both share.** Both frameworks express the human approval gate natively, and both run the same tools and
  model, so for this flow the difference lies in operations (tracing, resumability, packaging), not capability.

## Decision

_To be written after the measured run:_ stay on LangGraph, unless the OpenAI Agents SDK wins at least 3 of the 6
criteria.

## Consequences

_To be written with the decision._
