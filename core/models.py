import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

def get_model(role: str = "executor"):
    r = role.upper()
    name = os.environ[f"LLM_MODEL_{r}"]
    if name.startswith("claude") and os.getenv("ANTHROPIC_API_KEY"):
        return ChatAnthropic(model=name, temperature=0)
    base = os.getenv(f"LLM_BASE_URL_{r}") or os.environ["LLM_BASE_URL"]
    return ChatOpenAI(model=name, base_url=base, api_key="EMPTY", temperature=0)

