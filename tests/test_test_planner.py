"""Phase 2 planner graph against the real agfhc server in dry-run (simulated node) with a scripted model."""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from pydantic import create_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "agfhc"))
from core.guard import guard_tool  # noqa: E402
from core.recipes import RecipeStore  # noqa: E402
from agents.test_planner.graph import build_graph, validate_plan, template, Plan, Replan  # noqa: E402

TOOLS = ["list_test_nodes", "preflight", "list_recipe_tests", "plan_run", "start_run",
         "get_run_status", "analyze_run", "finalize_report"]


def _schema(fn):
    import inspect
    fields = {p.name: (str, p.default if p.default is not inspect._empty else ...)
              for p in inspect.signature(fn).parameters.values()}
    return create_model(f"{fn.__name__}_args", **fields)


class FakeModel:
    """Scripted answers: Plan objects, Replan objects, and free-text analyses, consumed in order."""

    def __init__(self, plans=(), replans=(), text="Analysis: see failure details."):
        self.q = {Plan: list(plans), Replan: list(replans)}
        self.text, self.calls, self.prompts = text, {Plan: 0, Replan: 0}, []

    def with_structured_output(self, schema, method=None):
        outer = self

        class R:
            async def ainvoke(self, msgs):
                outer.calls[schema] += 1
                outer.prompts.append(msgs[-1][1])
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
    # Same path as production: each server function becomes a LangChain tool wrapped by core/guard.py
    def wrap(name, fn):
        async def coro(**kw):
            r = fn(**kw)
            return await r if asyncio.iscoroutine(r) else r
        tool = guard_tool(StructuredTool.from_function(coroutine=coro, name=name, description=name,
                                                       args_schema=_schema(fn)), "agfhc")
        return lambda **kw: tool.ainvoke(kw)
    return {"tools": {n: wrap(n, getattr(srv, n)) for n in TOOLS}, "tmp": tmp_path,
            "store": RecipeStore(tmp_path / "recipes")}


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
    g = build_graph(env["tools"], m, m, InMemorySaver(), poll_s=0.5, recipes=env["store"])
    out, seen = drive(g, {"requirements": "quick health check", "node": "sim"}, [{"action": "approve"}])
    assert "all_lvl1" in seen[0]["markdown"] and "~5m" in seen[0]["markdown"]
    assert out["status"] == "FAILED" and "hand off to triage" in out["report"]
    assert out["results"][0]["verdict"] == "FAIL"
    (a,) = archives(env["tmp"])
    assert "Analysis: see failure details." in (a / "report.html").read_text()


def test_two_step_pass_is_done(env, monkeypatch):
    monkeypatch.setenv("SIM_SCENARIO", "pass")
    m = FakeModel(plans=[plan("hbm_lvl1", "xgmi_lvl1", tier="subsystem_lvl1")])
    g = build_graph(env["tools"], m, m, InMemorySaver(), poll_s=0.5, recipes=env["store"])
    out, _ = drive(g, {"requirements": "check HBM and XGMI", "node": "sim"}, [{"action": "approve"}])
    assert out["status"] == "DONE", out["report"]
    assert [r["recipe"] for r in out["results"]] == ["hbm_lvl1", "xgmi_lvl1"]
    assert len(archives(env["tmp"])) == 2


def test_reject_starts_nothing(env):
    m = FakeModel(plans=[plan("all_lvl1")])
    g = build_graph(env["tools"], m, m, InMemorySaver(), poll_s=0.5, recipes=env["store"])
    out, _ = drive(g, {"requirements": "x", "node": "sim"}, [{"action": "reject", "reason": "wrong node"}])
    assert out["status"] == "REJECTED" and not out["results"]
    assert not (env["tmp"] / "sim" / "agfhc-runs").exists()


def test_invalid_plan_never_reaches_approval(env):
    m = FakeModel(plans=[plan("rm_everything"), plan("all_lvl9")])
    g = build_graph(env["tools"], m, m, InMemorySaver(), poll_s=0.5, recipes=env["store"])
    out, seen = drive(g, {"requirements": "x", "node": "sim"}, [])
    assert out["status"] == "FAILED" and not seen and m.calls[Plan] == 2


def test_bad_edit_is_refused_then_approved(env, monkeypatch):
    monkeypatch.setenv("SIM_SCENARIO", "pass")
    m = FakeModel(plans=[plan("all_lvl1")])
    g = build_graph(env["tools"], m, m, InMemorySaver(), poll_s=0.5, recipes=env["store"])
    out, seen = drive(g, {"requirements": "quick check", "node": "sim"},
                      [{"action": "edit", "recipes": ["all_burnin_24h"]}, {"action": "approve"}])
    assert "Edit refused" in seen[1]["markdown"] and "burn-in" in seen[1]["markdown"]
    assert [r["recipe"] for r in out["results"]] == ["all_lvl1"] and out["status"] == "DONE"


def test_real_node_refused_in_dryrun(env):
    m = FakeModel()
    g = build_graph(env["tools"], m, m, InMemorySaver(), poll_s=0.5, recipes=env["store"])
    out, seen = drive(g, {"requirements": "x", "node": "mitacai"}, [])
    assert out["status"] == "REFUSED" and not seen and m.calls[Plan] == 0


def test_execution_error_replans_then_fails(env, monkeypatch):
    monkeypatch.setenv("SIM_GPU_BUSY", "1")
    m = FakeModel(plans=[plan("all_lvl1")], replans=[Replan(action="retry", reason="wait"), Replan(action="retry", reason="again")])
    g = build_graph(env["tools"], m, m, InMemorySaver(), poll_s=0.5, recipes=env["store"])
    out, _ = drive(g, {"requirements": "x", "node": "sim"}, [{"action": "approve"}])
    assert out["status"] == "FAILED" and "stopped after 2 replans" in out["report"]
    assert m.calls[Replan] == 2 and not archives(env["tmp"])


def test_validate_plan_rules():
    tpl = template("mi325x")
    assert validate_plan({"steps": []}, tpl, "") == "plan has no steps"
    assert "burn-in" in validate_plan({"steps": [{"recipe": "all_burnin_4h"}]}, tpl, "standard run")
    assert validate_plan({"steps": [{"recipe": "all_burnin_4h"}]}, tpl, "4h burn-in") == ""
    assert validate_plan({"steps": [{"recipe": "all_lvl2"}]}, tpl, "") == ""


# ------------------------------------------------------------------ procedural memory (2.3)

def recipe_files(store, dryrun=True):
    d = store.base / "dryrun" if dryrun else store.base
    return sorted(d.glob("*/*/*.yaml"))


def test_executed_plan_is_saved_and_reused(env, monkeypatch):
    monkeypatch.setenv("SIM_SCENARIO", "pass")
    m = FakeModel(plans=[plan("hbm_lvl1", "xgmi_lvl1", tier="subsystem_lvl1")])
    g = build_graph(env["tools"], m, m, InMemorySaver(), poll_s=0.5, recipes=env["store"])
    out, _ = drive(g, {"requirements": "check HBM and XGMI after BKC update", "node": "sim"}, [{"action": "approve"}])
    (f,) = recipe_files(env["store"])
    rec = yaml.safe_load(f.read_text())
    assert rec["dryrun"] and rec["platform"] == "mi325x" and rec["bkc"] == "SIM-BIOS-1.0"
    assert [s["recipe"] for s in rec["steps"]] == ["hbm_lvl1", "xgmi_lvl1"]
    assert [o["verdict"] for o in rec["outcome"]] == ["PASS", "PASS"] and rec["versions"].get("ROCm") == "7.2.1"
    assert "Recipe saved" in out["report"] and out["recipe_saved"] == str(f)
    # second request: the planner is shown the recipe, adapts it, and the approver sees where it came from
    m2 = FakeModel(plans=[Plan(tier="subsystem_lvl1", based_on=rec["id"], steps=[{"recipe": "hbm_lvl1", "reason": "r"}])])
    g2 = build_graph(env["tools"], m2, m2, InMemorySaver(), poll_s=0.5, recipes=env["store"])
    out2, seen = drive(g2, {"requirements": "HBM and XGMI check", "node": "sim"}, [{"action": "reject"}], thread="t2")
    assert rec["id"] in m2.prompts[0] and "Previously approved plans" in m2.prompts[0]
    assert f"Based on approved recipe `{rec['id']}`" in seen[0]["markdown"]


def test_fail_verdict_still_saves_recipe_but_failed_execution_does_not(env, monkeypatch):
    monkeypatch.setenv("SIM_SCENARIO", "fail")
    m = FakeModel(plans=[plan("all_lvl1")])
    g = build_graph(env["tools"], m, m, InMemorySaver(), poll_s=0.5, recipes=env["store"])
    out, _ = drive(g, {"requirements": "smoke", "node": "sim"}, [{"action": "approve"}])
    assert out["status"] == "FAILED" and len(recipe_files(env["store"])) == 1   # plan executed; the GPU failed
    monkeypatch.setenv("SIM_GPU_BUSY", "1")
    m = FakeModel(plans=[plan("all_lvl2", tier="standard")], replans=[Replan(action="stop", reason="busy")])
    g = build_graph(env["tools"], m, m, InMemorySaver(), poll_s=0.5, recipes=env["store"])
    drive(g, {"requirements": "standard", "node": "sim"}, [{"action": "approve"}], thread="t2")
    assert len(recipe_files(env["store"])) == 1                                  # nothing ran: nothing saved


def test_rejected_plan_not_saved_and_unknown_based_on_cleared(env):
    m = FakeModel(plans=[Plan(tier="smoke", based_on="made-up-id", steps=[{"recipe": "all_lvl1", "reason": "r"}])])
    g = build_graph(env["tools"], m, m, InMemorySaver(), poll_s=0.5, recipes=env["store"])
    out, seen = drive(g, {"requirements": "x", "node": "sim"}, [{"action": "reject"}])
    assert "New plan (no matching approved recipe)" in seen[0]["markdown"] and out["plan"]["based_on"] == ""
    assert not recipe_files(env["store"])


def test_dryrun_and_live_recipes_never_mix(tmp_path):
    st = RecipeStore(tmp_path)
    kw = dict(platform="mi325x", bkc="25.06.08", plan={"tier": "smoke", "steps": [{"recipe": "all_lvl1"}]},
              results=[{"recipe": "all_lvl1", "run_id": "r1", "verdict": "PASS"}], versions={}, node="sim")
    st.save(requirements="quick health check", dryrun=True, **kw)
    assert st.find("quick health check", "mi325x", dryrun=True)
    assert st.find("quick health check", "mi325x", dryrun=False) == []
    assert st.find("quick health check", "mi350x", dryrun=True) == []


class FakeIndex:
    def __init__(self):
        self.added = []

    def add(self, text, agent_id, metadata):
        self.added.append((text, agent_id, metadata))

    def search(self, query, agent_id, platform, k):
        return [{"path": md["path"], "score": 0.9} for _, a, md in self.added if a == agent_id][:k] + \
               [{"path": "/etc/passwd", "score": 0.99}]


def test_index_is_used_and_paths_outside_the_store_are_ignored(tmp_path):
    idx = FakeIndex()
    st = RecipeStore(tmp_path, index=idx)
    rec = st.save(platform="mi325x", bkc="25.06.08", requirements="XGMI link check", dryrun=False, node="n1",
                  plan={"tier": "subsystem_lvl1", "steps": [{"recipe": "xgmi_lvl1"}]},
                  results=[{"recipe": "xgmi_lvl1", "run_id": "r", "verdict": "PASS"}], versions={})
    (text, agent, md), = idx.added
    assert agent == "test-planner" and md["type"] == "procedure" and "xgmi_lvl1" in text
    found = st.find("anything", "mi325x", dryrun=False)
    assert [r["id"] for r in found] == [rec["id"]]


def test_rerun_of_same_recipe_updates_history_not_a_new_file(env, monkeypatch):
    monkeypatch.setenv("SIM_SCENARIO", "pass")
    m = FakeModel(plans=[plan("all_lvl1")])
    g = build_graph(env["tools"], m, m, InMemorySaver(), poll_s=0.5, recipes=env["store"])
    drive(g, {"requirements": "quick health check", "node": "sim"}, [{"action": "approve"}])
    (f,) = recipe_files(env["store"])
    rid = yaml.safe_load(f.read_text())["id"]
    monkeypatch.setenv("SIM_SCENARIO", "fail")
    m2 = FakeModel(plans=[Plan(tier="smoke", based_on=rid, steps=[{"recipe": "all_lvl1", "reason": "r"}])])
    g2 = build_graph(env["tools"], m2, m2, InMemorySaver(), poll_s=0.5, recipes=env["store"])
    out, _ = drive(g2, {"requirements": "quick health check again", "node": "sim"}, [{"action": "approve"}], thread="t2")
    assert recipe_files(env["store"]) == [f] and "Recipe updated (used 2x)" in out["report"]
    rec = yaml.safe_load(f.read_text())
    assert rec["uses"] == 2 and [h["outcome"][0]["verdict"] for h in rec["history"]] == ["PASS", "FAIL"]
    # a changed step list is a new recipe
    m3 = FakeModel(plans=[Plan(tier="smoke", based_on=rid, steps=[{"recipe": "all_lvl2", "reason": "r"}])])
    g3 = build_graph(env["tools"], m3, m3, InMemorySaver(), poll_s=0.5, recipes=env["store"])
    drive(g3, {"requirements": "quick health check", "node": "sim"}, [{"action": "approve"}], thread="t3")
    assert len(recipe_files(env["store"])) == 2
