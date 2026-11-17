"""Phase 6.1 orchestrator: the planner, PREPARING exceptions and triage hand-off on the dry-run simulator."""
import asyncio
import json
import sys
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tests.test_test_planner import FakeModel, Plan, env, plan  # noqa: E402,F401  (env is a fixture)
from agents.test_planner.graph import _text, build_graph as planner_graph  # noqa: E402
from agents.orchestrator.graph import (Audit, archive_of, build_graph, can_run, readiness_exceptions,  # noqa: E402
                                       parse_expect)


def orchestrator(env, model, tmp_path, triage_run=None):
    async def preflight(node):
        return _text(await env["tools"]["preflight"](node=node))
    return build_graph(lambda: planner_graph(env["tools"], model, model, None, poll_s=0.5, recipes=env["store"]),
                       preflight, triage_run=triage_run, recipes=env["store"], checkpointer=InMemorySaver(),
                       home=tmp_path / "runs")


def drive(graph, first, decisions, audit):
    """Like the CLI: answer each interrupt, logging plan decisions (exception decisions are logged in the graph)."""
    cfg = {"configurable": {"thread_id": first["run_id"]}, "recursion_limit": 200}
    out, seen = asyncio.run(graph.ainvoke(first, cfg)), []
    while "__interrupt__" in out:
        payload = out["__interrupt__"][0].value
        seen.append(payload)
        d = decisions.pop(0)
        if payload.get("kind") != "exceptions":
            audit.log("plan_decision", action=d["action"])
        out = asyncio.run(graph.ainvoke(Command(resume=d), cfg))
    return out, seen


def start(run_id, req="quick health check", expect=None, nodes=("sim",)):
    return {"run_id": run_id, "requirements": req, "nodes": list(nodes), "expect": expect or {}}


def test_pass_run_has_one_human_event(env, tmp_path, monkeypatch):
    monkeypatch.setenv("SIM_SCENARIO", "pass")
    g = orchestrator(env, FakeModel(plans=[plan("all_lvl1")]), tmp_path)
    audit = Audit("r1", tmp_path / "runs")
    out, seen = drive(g, start("r1"), [{"action": "approve"}], audit)
    assert out["phase"] == "DONE", out.get("run_report")
    assert [p.get("kind") for p in seen] == [None]                       # only the plan approval
    assert [e["event"] for e in audit.human_events()] == ["plan_decision"]
    assert (audit.dir / "report.md").exists() and "DONE" in (audit.dir / "report.md").read_text()
    phases = [e.get("phase") for e in audit.events() if e["event"] == "state"]
    assert phases == ["REQUESTED", "PLANNING", "PLANNED", "PREPARING", "EXECUTING", "COLLECTED", "REPORTING"]
    assert out["recipe_saved"] and len(list((env["tmp"] / "recipes").rglob("*.yaml"))) == 1


def test_failed_run_goes_to_analysis(env, tmp_path, monkeypatch):
    monkeypatch.setenv("SIM_SCENARIO", "fail")
    seen_archives = []

    async def fake_triage(archive):
        seen_archives.append(archive)
        return {"ranked": [{"hyp": {"category": "environment/node state", "cause": "GPUs held by another process"},
                            "score": 0.8, "support_v": []}],
                "draft": {"summary": "hbm test fails to allocate"}, "_saved": str(tmp_path / "t")}
    g = orchestrator(env, FakeModel(plans=[plan("all_lvl1")]), tmp_path, triage_run=fake_triage)
    audit = Audit("r2", tmp_path / "runs")
    out, _ = drive(g, start("r2"), [{"action": "approve"}], audit)
    assert out["phase"] == "FAILED"
    assert len(seen_archives) == 1 and Path(seen_archives[0]).is_dir()          # FAILED -> ANALYZING, real archive
    assert out["triage"][0]["top"][0]["category"] == "environment/node state"
    md = (audit.dir / "report.md").read_text()
    assert "Failure analysis" in md and "agents.triage.file draft" in md and "not filed" in md
    assert "**environment/node state**: GPUs held by another process (score 0.8)" in md
    assert "### sim: `all_lvl1`" in md
    assert [e["event"] for e in audit.human_events()] == ["plan_decision"]


def test_busy_gpus_stop_the_run_without_a_decision(env, tmp_path, monkeypatch):
    monkeypatch.setenv("SIM_SCENARIO", "pass")
    monkeypatch.setenv("SIM_GPU_BUSY", "1")
    g = orchestrator(env, FakeModel(plans=[plan("all_lvl1")]), tmp_path)
    audit = Audit("r3", tmp_path / "runs")
    out, seen = drive(g, start("r3"), [{"action": "approve"}], audit)
    assert len(seen) == 1                                                      # no exception prompt: a hard stop
    assert out["phase"] == "REJECTED" and not out.get("node_results")
    assert "GPUs not idle" in (audit.dir / "report.md").read_text()
    assert not (env["tmp"] / "sim" / "agfhc-runs").exists()                    # nothing started


def test_abort_on_exceptions_starts_nothing(env, tmp_path, monkeypatch):
    monkeypatch.setenv("SIM_SCENARIO", "pass")
    g = orchestrator(env, FakeModel(plans=[plan("all_lvl1")]), tmp_path)
    audit = Audit("r3b", tmp_path / "runs")
    out, seen = drive(g, start("r3b", expect={"System BIOS": "V9.99"}),
                      [{"action": "approve"}, {"action": "abort", "reason": "wrong BKC"}], audit)
    assert seen[1]["kind"] == "exceptions" and out["phase"] == "REJECTED" and not out.get("node_results")
    assert not (env["tmp"] / "sim" / "agfhc-runs").exists()
    assert [e["event"] for e in audit.human_events()] == ["plan_decision", "exception_decision"]


def test_accepted_exception_runs_and_is_reported(env, tmp_path, monkeypatch):
    monkeypatch.setenv("SIM_SCENARIO", "pass")
    g = orchestrator(env, FakeModel(plans=[plan("all_lvl1")]), tmp_path)
    audit = Audit("r4", tmp_path / "runs")
    out, seen = drive(g, start("r4", expect={"System BIOS": "V9.99"}),
                      [{"action": "approve"}, {"action": "proceed", "reason": "BIOS update scheduled next week"}],
                      audit)
    assert any("expects V9.99" in e for e in seen[1]["exceptions"]["sim"])
    assert out["phase"] == "DONE"
    assert "BIOS update scheduled" in (audit.dir / "report.md").read_text()
    assert len(audit.human_events()) == 2


def test_exception_needs_a_reason(env, tmp_path, monkeypatch):
    monkeypatch.setenv("SIM_SCENARIO", "pass")
    g = orchestrator(env, FakeModel(plans=[plan("all_lvl1")]), tmp_path)
    audit = Audit("r5", tmp_path / "runs")
    out, _ = drive(g, start("r5", expect={"System BIOS": "V9.99"}),
                   [{"action": "approve"}, {"action": "proceed", "reason": "ok"}], audit)
    assert out["phase"] == "REJECTED"


def test_rejected_plan_ends_without_running(env, tmp_path):
    g = orchestrator(env, FakeModel(plans=[plan("all_lvl1")]), tmp_path)
    audit = Audit("r6", tmp_path / "runs")
    out, _ = drive(g, start("r6"), [{"action": "reject", "reason": "wrong node"}], audit)
    assert out["phase"] == "REJECTED" and not out.get("node_results")
    assert json.loads((audit.dir / "run.json").read_text())["phase"] == "REJECTED"


def test_kill_switch_stops_before_planning(env, tmp_path, monkeypatch):
    import core.guard
    monkeypatch.setattr(core.guard, "KILL_FILE", tmp_path / "KILL")
    (tmp_path / "KILL").write_text("stop")
    m = FakeModel(plans=[plan("all_lvl1")])
    g = orchestrator(env, m, tmp_path)
    out, seen = drive(g, start("r7"), [], Audit("r7", tmp_path / "runs"))
    assert out["phase"] == "STOPPED" and not seen and m.calls == {k: 0 for k in m.calls}


def test_four_nodes_one_approval_parallel_runs(env, tmp_path, monkeypatch):
    monkeypatch.setenv("SIM_NODES", "sim2,sim3,sim4")
    monkeypatch.setenv("SIM_SCENARIO", "pass")
    monkeypatch.setenv("SIM_SCENARIO_sim3", "fail")
    triaged = []

    async def fake_triage(archive):
        triaged.append(archive)
        return {"ranked": [], "draft": {}, "_saved": "x"}
    m = FakeModel(plans=[plan("all_lvl1")])
    g = orchestrator(env, m, tmp_path, triage_run=fake_triage)
    audit = Audit("r8", tmp_path / "runs")
    out, seen = drive(g, start("r8", nodes=("sim", "sim2", "sim3", "sim4")), [{"action": "approve"}], audit)
    assert len(seen) == 1 and m.calls[Plan] == 1                                # planned and approved once
    assert sorted(r["node"] for r in out["node_results"]) == ["sim", "sim2", "sim3", "sim4"]
    status = {r["node"]: r["status"] for r in out["node_results"]}
    assert status == {"sim": "DONE", "sim2": "DONE", "sim3": "FAILED", "sim4": "DONE"}
    assert out["phase"] == "FAILED" and len(triaged) == 1 and "/sim3/archive/" in triaged[0]
    for n in ("sim2", "sim3", "sim4"):                                          # own run root and archive each
        assert list((env["tmp"] / "sim" / n / "archive").glob("AGFHC_*"))
    assert [e["event"] for e in audit.human_events()] == ["plan_decision"]
    done = [e for e in audit.events() if e["event"] == "node_done"]
    assert len(done) == 4
    assert len(list((env["tmp"] / "recipes").rglob("*.yaml"))) == 1            # recipe saved once for the run
    md = (audit.dir / "report.md").read_text()
    assert "| sim3 | FAILED |" in md and "| sim4 | DONE |" in md


def test_exceptions_on_several_nodes_are_one_decision(env, tmp_path, monkeypatch):
    monkeypatch.setenv("SIM_NODES", "sim2,sim3")
    monkeypatch.setenv("SIM_SCENARIO", "pass")
    monkeypatch.setenv("SIM_BIOS_sim2", "SIM-BIOS-0.9")
    monkeypatch.setenv("SIM_BIOS_sim3", "SIM-BIOS-0.9")
    g = orchestrator(env, FakeModel(plans=[plan("all_lvl1")]), tmp_path)
    audit = Audit("r9", tmp_path / "runs")
    out, seen = drive(g, start("r9", nodes=("sim", "sim2", "sim3"), expect={"System BIOS": "SIM-BIOS-1.0"}),
                      [{"action": "approve"}, {"action": "proceed", "reason": "older BIOS accepted for this smoke run"}],
                      audit)
    assert len(seen) == 2 and sorted(seen[1]["exceptions"]) == ["sim2", "sim3"]
    assert out["phase"] == "DONE" and len(out["node_results"]) == 3
    assert len(audit.human_events()) == 2


def test_busy_gpus_on_one_node_stop_all(env, tmp_path, monkeypatch):
    monkeypatch.setenv("SIM_NODES", "sim2")
    monkeypatch.setenv("SIM_GPU_BUSY_sim2", "1")
    g = orchestrator(env, FakeModel(plans=[plan("all_lvl1")]), tmp_path)
    audit = Audit("r11", tmp_path / "runs")
    out, seen = drive(g, start("r11", nodes=("sim", "sim2")), [{"action": "approve"}], audit)
    assert out["phase"] == "REJECTED" and len(seen) == 1
    assert "sim2 cannot run AGFHC (GPUs not idle" in (audit.dir / "report.md").read_text()


def test_a_node_that_cannot_run_stops_everything(env, tmp_path, monkeypatch):
    monkeypatch.setenv("SIM_NODES", "sim2")
    g = orchestrator(env, FakeModel(plans=[plan("all_lvl1")]), tmp_path)
    audit = Audit("r10", tmp_path / "runs")
    out, _ = drive(g, start("r10", nodes=("sim", "nosuchnode")), [{"action": "approve"}], audit)
    assert out["phase"] == "REJECTED" and not out.get("node_results")
    assert "nosuchnode cannot run AGFHC" in (audit.dir / "report.md").read_text()
    assert not (env["tmp"] / "sim" / "agfhc-runs").exists()


def test_helpers():
    assert archive_of("**Archived** (12 files) to `localhost (dry-run):/srv/a/AGFHC_1/`") == "/srv/a/AGFHC_1"
    assert archive_of("**Archived** (3) to `/tmp/x/AGFHC_2/`") == "/tmp/x/AGFHC_2"
    pf = ("| VBIOS | 113 |\n**Recent amdgpu/RAS/AER kernel errors:**\n```\nnone\n```")
    v = {"VBIOS": "113", "System BIOS": "V1.04", "ROCm": "7.2.1"}
    assert readiness_exceptions(pf, v, {}) == []
    assert readiness_exceptions(pf, v, {"System BIOS": "V1.04"}) == []
    assert readiness_exceptions(pf, {**v, "VBIOS": "unknown"}, {}) == ["VBIOS version could not be read"]
    bad = pf.replace("none\n", "amdgpu: ring gfx timeout\n")
    assert readiness_exceptions(bad, v, {})[0].startswith("recent amdgpu")
    assert parse_expect(["System BIOS=V1.04"]) == {"System BIOS": "V1.04"}
    with pytest.raises(SystemExit):
        parse_expect(["nope"])
    assert can_run("**Can run AGFHC:** no - dry-run mode")[0] is False
    assert can_run("REFUSED: unknown node 'x'")[0] is False
    assert can_run("**Can run AGFHC:** yes\n**GPU usage:** VRAM in use on GPU[0] 97%")[0] is False
    assert can_run("**Can run AGFHC:** yes\n**GPU usage:** all 8 GPUs idle") == (True, "")
