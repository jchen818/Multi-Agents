"""Phase 2 planner graph against the real agfhc server in dry-run (simulated node) with a scripted model."""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "agfhc"))
from agents.test_planner.graph import build_graph, validate_plan, template, Plan, Replan  # noqa: E402

TOOLS = ["list_test_nodes", "preflight", "list_recipe_tests", "plan_run", "start_run",
         "get_run_status", "analyze_run", "finalize_report"]


class FakeModel:
    """Scripted answers: Plan objects, Replan objects, and free-text analyses, consumed in order."""

    def __init__(self, plans=(), replans=(), text="Analysis: see failure details."):
        self.q = {Plan: list(plans), Replan: list(replans)}
        self.text, self.calls = text, {Plan: 0, Replan: 0}

    def with_structured_output(self, schema, method=None):
        outer = self

        class R:
            async def ainvoke(self, _msgs):
                outer.calls[schema] += 1
                return outer.q[schema].pop(0)
        return R()

    async def ainvoke(self, _msgs):
        return AIMessage(content=self.text)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTI_AGENTS_ROOT", str(REPO))
    monkeypatch.setenv("AGFHC_MODE", "dryrun")
    monkeypatch.setenv("AGFHC_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENTS_KILL_FILE", str(tmp_path / "KILL"))
    monkeypatch.setenv("AGFHC_TMUX", "0")
    monkeypatch.setenv("SIM_SLEEP", "1")
    monkeypatch.delenv("SIM_GPU_BUSY", raising=False)
    spec = importlib.util.spec_from_file_location("agfhc_server_planner", REPO / "tools/agfhc/server.py")
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    def wrap(fn):
        async def call(**kw):
            r = fn(**kw)
            return await r if asyncio.iscoroutine(r) else r
        return call
    return {"tools": {n: wrap(getattr(srv, n)) for n in TOOLS}, "tmp": tmp_path}


def plan(*recipes, tier="smoke"):
    return Plan(tier=tier, steps=[{"recipe": r, "reason": "test"} for r in recipes])


def drive(graph, first, decisions, thread="t1"):
    cfg = {"configurable": {"thread_id": thread}}
    out, seen = asyncio.run(graph.ainvoke(first, cfg)), []
    while "__interrupt__" in out:
        payload = out["__interrupt__"][0].value
        seen.append(payload)
        out = asyncio.run(graph.ainvoke(Command(resume=decisions.pop(0)), cfg))
    return out, seen


def archives(tmp):
    d = tmp / "sim" / "archive"
    return sorted(d.glob("AGFHC_*")) if d.exists() else []


def test_approved_fail_run_hands_off(env, monkeypatch):
    monkeypatch.setenv("SIM_SCENARIO", "fail")
    m = FakeModel(plans=[plan("all_lvl1")])
    g = build_graph(env["tools"], m, m, InMemorySaver(), poll_s=0.5)
    out, seen = drive(g, {"requirements": "quick health check", "node": "sim"}, [{"action": "approve"}])
    assert "all_lvl1" in seen[0]["markdown"] and "~5m" in seen[0]["markdown"]
    assert out["status"] == "FAILED" and "hand off to triage" in out["report"]
    assert out["results"][0]["verdict"] == "FAIL"
    (a,) = archives(env["tmp"])
    assert "Analysis: see failure details." in (a / "report.html").read_text()


def test_two_step_pass_is_done(env, monkeypatch):
    monkeypatch.setenv("SIM_SCENARIO", "pass")
    m = FakeModel(plans=[plan("hbm_lvl1", "xgmi_lvl1", tier="subsystem_lvl1")])
    g = build_graph(env["tools"], m, m, InMemorySaver(), poll_s=0.5)
    out, _ = drive(g, {"requirements": "check HBM and XGMI", "node": "sim"}, [{"action": "approve"}])
    assert out["status"] == "DONE", out["report"]
    assert [r["recipe"] for r in out["results"]] == ["hbm_lvl1", "xgmi_lvl1"]
    assert len(archives(env["tmp"])) == 2


def test_reject_starts_nothing(env):
    m = FakeModel(plans=[plan("all_lvl1")])
    g = build_graph(env["tools"], m, m, InMemorySaver(), poll_s=0.5)
    out, _ = drive(g, {"requirements": "x", "node": "sim"}, [{"action": "reject", "reason": "wrong node"}])
    assert out["status"] == "REJECTED" and not out["results"]
    assert not (env["tmp"] / "sim" / "agfhc-runs").exists()


def test_invalid_plan_never_reaches_approval(env):
    m = FakeModel(plans=[plan("rm_everything"), plan("all_lvl9")])
    g = build_graph(env["tools"], m, m, InMemorySaver(), poll_s=0.5)
    out, seen = drive(g, {"requirements": "x", "node": "sim"}, [])
    assert out["status"] == "FAILED" and not seen and m.calls[Plan] == 2


def test_bad_edit_is_refused_then_approved(env, monkeypatch):
    monkeypatch.setenv("SIM_SCENARIO", "pass")
    m = FakeModel(plans=[plan("all_lvl1")])
    g = build_graph(env["tools"], m, m, InMemorySaver(), poll_s=0.5)
    out, seen = drive(g, {"requirements": "quick check", "node": "sim"},
                      [{"action": "edit", "recipes": ["all_burnin_24h"]}, {"action": "approve"}])
    assert "Edit refused" in seen[1]["markdown"] and "burn-in" in seen[1]["markdown"]
    assert [r["recipe"] for r in out["results"]] == ["all_lvl1"] and out["status"] == "DONE"


def test_real_node_refused_in_dryrun(env):
    m = FakeModel()
    g = build_graph(env["tools"], m, m, InMemorySaver(), poll_s=0.5)
    out, seen = drive(g, {"requirements": "x", "node": "mitacai"}, [])
    assert out["status"] == "REFUSED" and not seen and m.calls[Plan] == 0


def test_execution_error_replans_then_fails(env, monkeypatch):
    monkeypatch.setenv("SIM_GPU_BUSY", "1")
    m = FakeModel(plans=[plan("all_lvl1")], replans=[Replan(action="retry", reason="wait"), Replan(action="retry", reason="again")])
    g = build_graph(env["tools"], m, m, InMemorySaver(), poll_s=0.5)
    out, _ = drive(g, {"requirements": "x", "node": "sim"}, [{"action": "approve"}])
    assert out["status"] == "FAILED" and "stopped after 2 replans" in out["report"]
    assert m.calls[Replan] == 2 and not archives(env["tmp"])


def test_validate_plan_rules():
    tpl = template("mi325x")
    assert validate_plan({"steps": []}, tpl, "") == "plan has no steps"
    assert "burn-in" in validate_plan({"steps": [{"recipe": "all_burnin_4h"}]}, tpl, "standard run")
    assert validate_plan({"steps": [{"recipe": "all_burnin_4h"}]}, tpl, "4h burn-in") == ""
    assert validate_plan({"steps": [{"recipe": "all_lvl2"}]}, tpl, "") == ""
