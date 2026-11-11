import asyncio
from langgraph.prebuilt import create_react_agent
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langchain_mcp_adapters.tools import load_mcp_tools
from core.models import get_model
from core.mcp import client

SYSTEM = ("You are a lab assistant for MiTAC AI racks. Use tools to answer; never guess values. "
          "Always pass the Redfish endpoint name explicitly. Cite the tool output behind each claim. Be brief.")
TOOLS = {"list_bmcs", "list_systems", "get_system"}

async def main():
    async with client.session("redfish") as s:
        tools = [t for t in await load_mcp_tools(s) if t.name in TOOLS]
        agent = create_react_agent(get_model(), tools, prompt=SYSTEM)
        res = await agent.ainvoke(
            {"messages": [("user", "Which BMCs can you reach, and what are the model and power state of mitacai?")]},
            config={"callbacks": [CallbackHandler()], "recursion_limit": 31, "run_name": "hello-lab-ops"})
        for m in res["messages"]:
            calls = [c["name"] for c in getattr(m, "tool_calls", []) or []]
            text = m.content if isinstance(m.content, str) else str(m.content)
            print(f"{type(m).__name__:12} {calls or ''} {text[:300]}")
    get_client().flush()

asyncio.run(main())

