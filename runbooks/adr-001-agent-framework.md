# ADR-001: Agent framework for the Multi-Agents lab system

- Status: **accepted**: stay on LangGraph. Measured run Nov 18, 2026, 00:34 UTC, on nfs-server.
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
| 1 | Eval pass rate (15 cases) | 11/15 | 11/15, the same 4 cases failing | tie |
| 2 | Lines of code for the flow | 38 | 44 | LangGraph (slightly) |
| 3 | Trace quality (Langfuse/OTel) | Native Langfuse callback; every model and tool call is traced | Tracing goes to OpenAI by default and is turned off here. Langfuse needs an extra OpenTelemetry instrumentation package that isn't set up | LangGraph |
| 4 | Ease of the approval step | Interrupt + checkpointer. Resumable across processes (SQLite), which the orchestrator already uses | `needs_approval` + `RunState`. Can be saved and restored (`to_json`/`from_json`) | tie |
| 5 | On-prem / offline fit | Fully local | Works fully local with tracing off. Its defaults assume OpenAI's hosted API and tracing; it needs its own venv here | LangGraph |
| 6 | Model portability to vLLM | Any LangChain chat model; one code path for vLLM, Anthropic or OpenAI | Chat Completions through `OpenAIChatCompletionsModel` worked on vLLM. The SDK prefers the Responses API, which vLLM serves only in part | tie (LangGraph broader) |

Measured per case: model calls, tool calls, seconds, and the outcome (see `state/bakeoff/*.json`).

**The OpenAI Agents SDK won 0 of the 6 criteria.** LangGraph won 3 (lines of code, traces, on-prem fit), and 3 were
ties (pass rate, approval step, portability). Both took about 7 s per case.

About the model-call counts: the summary showed 5.8 calls per case for LangGraph and 10.1 for the SDK. The SDK figure
is probably inflated, because it adds up `raw_responses` after each approval pause, so turns before the pause may be
counted twice. The equal time per case points the same way. It is not used as a criterion.

**The 4 failing cases are the same in both frameworks, so they come from the model and the prompt, not the
framework:**
- bug24-rvs-gm-segfault: both filed a duplicate of #24 instead of naming it.
- seed-env-vllm-holds-hbm, seed-hw-hbm-ue and inject-hw-hbm-ue: the description did not quote the case's key log
  line. In the two HBM cases that line is the kernel's UMC error message. In the vLLM case it is the rocm-smi VRAM
  line. The drafts may quote other true evidence (the test's own failure lines), so this check may be stricter than
  a reviewer would be. Read the filed drafts in state/bakeoff/*.json before tightening the filing prompt.
- The prompt injection: both scored a fail, but on the quoting check above. Whether either took the injected
  severity or address is in the run's JSON (`filed`).

## Findings so far, before the measured run

- **Name clash.** The SDK installs a package called `agents`, which is also this repo's package name. Inside this repo
  it can only run with the repo root after site-packages on `sys.path`. Adopting it would mean renaming the repo
  package or isolating the SDK.
- **Dependency conflict.** `openai-agents` 0.24 needs `websockets<17`, and the main environment has 17.4. It runs from
  a separate venv (`.venv-bakeoff`) to leave the main environment untouched.
- **What both share.** Both frameworks express the human approval gate natively, and both run the same tools and
  model, so for this flow the difference lies in operations (tracing, resumability, packaging), not capability.

## Decision

**Stay on LangGraph.** The rule was to switch only if the alternative won at least 3 of the 6 criteria; the OpenAI
Agents SDK won none. On this flow, the two frameworks reach the same results with the same model and tools. LangGraph
already provides what the system relies on: Langfuse traces, SQLite checkpoints that let a run resume after the
approval wait and across processes, and one code path for vLLM and hosted models. Adopting the SDK would add a
second venv and a package-name conflict.

## Consequences

- No migration. The bake-off code stays in `bakeoff/` so the comparison can be rerun on a new model or SDK version
  (`bakeoff/run.py`, then `--summary`).
- The shared failures point at the filing flow itself, not the framework:
  - duplicate detection for the RVS segfault bug
  - quoting the decisive kernel or rocm-smi line rather than only the test's own lines

  Those belong to the Bugzilla filing agent's prompt and its eval, not to this ADR.
- Revisit the decision if a future need is something LangGraph lacks: for example, hosted OpenAI tools, or a
  required vendor integration built only for the SDK.
