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
from tests.test_test_planner import FakeModel, env, plan  # noqa: E402,F401  (env is a fixture)
from agents.test_planner.graph import build_graph as planner_graph  # noqa: E402
from agents.orchestrator.graph import (Audit, archive_of, build_graph, readiness_exceptions,  # noqa: E402
                                       parse_expect)


def orchestrator(env, model, tmp_path, triage_run=None):
    return build_graph(lambda prepare: planner_graph(env["tools"], model, model, None, poll_s=0.5,
                                                     recipes=env["store"], prepare=prepare),
                       triage_run=triage_run, checkpointer=InMemorySaver(), home=tmp_path / "runs")


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


def start(run_id, req="quick health check", expect=None):
    return {"run_id": run_id, "requirements": req, "node": "sim", "expect": expect or {}}


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
    assert phases == ["REQUESTED", "PREPARING", "EXECUTED", "REPORTING"]


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
    assert [e["event"] for e in audit.human_events()] == ["plan_decision"]


def test_busy_gpus_need_a_decision_and_abort_starts_nothing(env, tmp_path, monkeypatch):
    monkeypatch.setenv("SIM_SCENARIO", "pass")
    monkeypatch.setenv("SIM_GPU_BUSY", "1")
    g = orchestrator(env, FakeModel(plans=[plan("all_lvl1")]), tmp_path)
    audit = Audit("r3", tmp_path / "runs")
    out, seen = drive(g, start("r3"), [{"action": "approve"}, {"action": "abort", "reason": "node busy"}], audit)
    assert seen[1]["kind"] == "exceptions" and any("not idle" in e for e in seen[1]["exceptions"])
    assert out["phase"] == "REJECTED" and not out.get("results")
    assert not (env["tmp"] / "sim" / "agfhc-runs").exists()                    # nothing started
    assert [e["event"] for e in audit.human_events()] == ["plan_decision", "exception_decision"]


def test_accepted_exception_runs_and_is_reported(env, tmp_path, monkeypatch):
    monkeypatch.setenv("SIM_SCENARIO", "pass")
    g = orchestrator(env, FakeModel(plans=[plan("all_lvl1")]), tmp_path)
    audit = Audit("r4", tmp_path / "runs")
    out, seen = drive(g, start("r4", expect={"System BIOS": "V9.99"}),
                      [{"action": "approve"}, {"action": "proceed", "reason": "BIOS update scheduled next week"}],
                      audit)
    assert any("expects V9.99" in e for e in seen[1]["exceptions"])
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
    assert out["phase"] == "REJECTED" and not out.get("results")
    assert json.loads((audit.dir / "run.json").read_text())["phase"] == "REJECTED"


def test_kill_switch_stops_before_planning(env, tmp_path, monkeypatch):
    import core.guard
    monkeypatch.setattr(core.guard, "KILL_FILE", tmp_path / "KILL")
    (tmp_path / "KILL").write_text("stop")
    m = FakeModel(plans=[plan("all_lvl1")])
    g = orchestrator(env, m, tmp_path)
    out, seen = drive(g, start("r7"), [], Audit("r7", tmp_path / "runs"))
    assert out["phase"] == "STOPPED" and not seen and m.calls == {k: 0 for k in m.calls}


def test_helpers():
    assert archive_of("**Archived** (12 files) to `localhost (dry-run):/srv/a/AGFHC_1/`") == "/srv/a/AGFHC_1"
    assert archive_of("**Archived** (3) to `/tmp/x/AGFHC_2/`") == "/tmp/x/AGFHC_2"
    pf = ("**GPU usage:** all 8 GPUs idle\n| VBIOS | 113 |\n**Recent amdgpu/RAS/AER kernel errors:**\n```\nnone\n```")
    v = {"VBIOS": "113", "System BIOS": "V1.04", "ROCm": "7.2.1"}
    assert readiness_exceptions(pf, v, {}) == []
    assert readiness_exceptions(pf, v, {"System BIOS": "V1.04"}) == []
    assert readiness_exceptions(pf, {**v, "VBIOS": "unknown"}, {}) == ["VBIOS version could not be read"]
    bad = pf.replace("none\n", "amdgpu: ring gfx timeout\n")
    assert readiness_exceptions(bad, v, {})[0].startswith("recent amdgpu")
    assert parse_expect(["System BIOS=V1.04"]) == {"System BIOS": "V1.04"}
    with pytest.raises(SystemExit):
        parse_expect(["nope"])
