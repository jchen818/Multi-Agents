import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Roles and the agents that use them (agents.env: LLM_MODEL_<ROLE>, optional LLM_BASE_URL_<ROLE>):
#   planner  - test planner, triage
#   executor - test planner's executor, coder, hello
#   lab_ops  - Lab Ops chat agent; falls back to executor when LLM_MODEL_LAB_OPS is not set
FALLBACK = {"lab_ops": "executor"}


def resolve(role: str = "executor") -> tuple:
    """(model name, base URL) for a role. A role in FALLBACK without its own LLM_MODEL_<ROLE> uses the fallback
    role's model and URL; one with its own model uses LLM_BASE_URL_<ROLE>, else LLM_BASE_URL."""
    r = role.upper()
    if not os.getenv(f"LLM_MODEL_{r}") and role in FALLBACK:
        return resolve(FALLBACK[role])
    name = os.environ[f"LLM_MODEL_{r}"]
    return name, os.getenv(f"LLM_BASE_URL_{r}") or os.environ["LLM_BASE_URL"]


def get_model(role: str = "executor"):
    name, base = resolve(role)
    if name.startswith("claude") and os.getenv("ANTHROPIC_API_KEY"):
        return ChatAnthropic(model=name, temperature=0)
    return ChatOpenAI(model=name, base_url=base, api_key="EMPTY", temperature=0)
