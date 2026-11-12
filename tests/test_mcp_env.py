"""The agfhc stdio server must see SIM_*/AGFHC_* from the caller, but never AGFHC_MODE."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_agfhc_server_env_passthrough(monkeypatch):
    monkeypatch.setenv("SIM_SCENARIO", "pass")
    monkeypatch.setenv("AGFHC_TZ", "UTC")
    monkeypatch.setenv("AGFHC_MODE", "live")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-not-for-servers")
    import core.mcp as m
    m = importlib.reload(m)
    env = m.SERVERS["agfhc"]["env"]
    assert env["SIM_SCENARIO"] == "pass" and env["AGFHC_TZ"] == "UTC"
    assert "AGFHC_MODE" not in env and "LANGFUSE_SECRET_KEY" not in env
    assert all("env" not in m.SERVERS[s] for s in ("searxng", "redfish", "labcli"))
