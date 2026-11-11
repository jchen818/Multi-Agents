from langgraph.types import interrupt
def power_cycle_node(node: str) -> str:
    if node not in RECOVERABLE_NODES:
        return f"REFUSED: {node} is not in RECOVERABLE_NODES"
    if interrupt({"action": "power_cycle", "node": node}) != "approve":
        return "DENIED by operator"
    return _ipmi(node, "chassis power cycle")
# resume: agent.invoke(Command(resume="approve"), config)

