"""Phase 6.2 control plane on fakeredis: validation, ack-after-work, redelivery, dead letters, kill switch, triage
worker, and the orchestrator's triage over the bus."""
import asyncio
import json
import sys
from pathlib import Path

import pytest
from fakeredis import FakeAsyncRedis

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from core import bus as B  # noqa: E402
from core.bus import BadMessage, Bus, KILL_KEY, Worker, stream, validate  # noqa: E402
from agents.triage.worker import make_handler  # noqa: E402


def run(coro):
    return asyncio.run(coro)


def msg(**kw):
    base = {"id": "a1", "run_id": "run-1", "from": "orchestrator", "to": "triage", "type": "triage_request",
            "artifact_path": "", "note": "", "reply_to": "agents:orchestrator:in", "key": "k", "ts": "1"}
    return {**base, **kw}


def test_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ARTIFACT_ROOTS", str(tmp_path))
    validate(msg(artifact_path=str(tmp_path / "AGFHC_1")), "triage")
    for bad, why in [(msg(to="coder"), "addressed"), (msg(type="rm_rf"), "does not accept"),
                     (msg(artifact_path="/etc/passwd"), "outside"), (msg(run_id="a b;rm"), "run_id"),
                     (msg(reply_to="agents:x:in; DEL"), "reply_to"), ({**msg(), "cmd": "x"}, "unknown fields"),
                     (msg(note="x" * 5000), "too long"), (msg(artifact_path=str(tmp_path / ".." / "..")), "outside")]:
        with pytest.raises(BadMessage, match=why):
            validate(bad, "triage")


def test_send_refuses_what_the_receiver_would_reject():
    r = FakeAsyncRedis(decode_responses=True)
    with pytest.raises(BadMessage):
        run(Bus(r, "orchestrator").send("triage", "delete_everything", "run-1"))


def test_ack_only_after_the_work_and_retry_then_dead_letter(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ARTIFACT_ROOTS", str(tmp_path))
    r = FakeAsyncRedis(decode_responses=True)
    calls = []

    async def flaky(bus, m):
        calls.append(m["_attempt"])
        raise RuntimeError("model server down")

    async def go():
        await B.ensure_group(r, "triage")
        await Bus(r, "orchestrator").send("triage", "triage_request", "run-1", reply_to="agents:orchestrator:in")
        w = Worker(r, "triage", flaky, consumer="w1", visibility_s=0, block_ms=10, log=lambda *_: None)
        for _ in range(6):
            await w.step()
        pending = await r.xpending(stream("triage"), "triage")
        dead = await r.xrange(B.DEAD)
        return pending, dead
    pending, dead = run(go())
    assert calls == [1, 2, 3]                                   # retried, never acked while failing
    assert pending["pending"] == 0 and len(dead) == 1 and "failed 3 times" in dead[0][1]["reason"]


def test_a_dead_workers_message_is_claimed_by_another(tmp_path):
    r = FakeAsyncRedis(decode_responses=True)
    done = []

    async def ok(bus, m):
        done.append(m["id"])

    async def go():
        await B.ensure_group(r, "triage")
        await Bus(r, "orchestrator").send("triage", "triage_request", "run-1")
        # worker 1 reads it and dies without acking
        await r.xreadgroup("triage", "dead-worker", {stream("triage"): ">"}, count=1)
        w2 = Worker(r, "triage", ok, consumer="w2", visibility_s=0, block_ms=10, log=lambda *_: None)
        await w2.step()
        return await r.xpending(stream("triage"), "triage")
    pending = run(go())
    assert len(done) == 1 and pending["pending"] == 0


def test_bad_messages_go_to_dead_letters_unprocessed():
    r = FakeAsyncRedis(decode_responses=True)
    seen = []

    async def go():
        await B.ensure_group(r, "triage")
        await r.xadd(stream("triage"), msg(type="format_disk"))
        w = Worker(r, "triage", lambda b, m: seen.append(m), consumer="w", block_ms=10, log=lambda *_: None)
        await w.step()
        return await r.xrange(B.DEAD)
    dead = run(go())
    assert not seen and "does not accept" in dead[0][1]["reason"]


def test_kill_key_stops_workers_and_the_guard(monkeypatch):
    r = FakeAsyncRedis(decode_responses=True)
    seen = []

    async def go():
        await B.ensure_group(r, "triage")
        await Bus(r, "orchestrator").send("triage", "triage_request", "run-1")
        await r.set(KILL_KEY, "1")
        w = Worker(r, "triage", lambda b, m: seen.append(m), consumer="w", block_ms=10, log=lambda *_: None)
        return await w.step()
    assert run(go()) == 0 and not seen

    import core.guard as g

    class FakeSync:
        def exists(self, k):
            return 1
    monkeypatch.setenv("REDIS_URL", "redis://x")
    monkeypatch.setitem(g._KILL_CACHE, "client", FakeSync())
    monkeypatch.setitem(g._KILL_CACHE, "t", 0.0)
    assert g.killed() is True

    class Down:
        def exists(self, k):
            raise ConnectionError("redis down")
    monkeypatch.setitem(g._KILL_CACHE, "client", Down())
    monkeypatch.setitem(g._KILL_CACHE, "t", 0.0)
    assert g.killed() is False                                 # a Redis outage does not stop the agents


def test_triage_worker_replies_with_the_saved_triage(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ARTIFACT_ROOTS", str(tmp_path))
    arch = tmp_path / "AGFHC_1"
    arch.mkdir()
    r = FakeAsyncRedis(decode_responses=True)

    async def fake_triage(a):
        return {"ranked": [{"hyp": {"category": "hardware", "cause": "HBM ECC on GPU 5"}, "score": 0.9}],
                "draft": {"summary": "HBM errors on GPU 5"}}

    def save(state):
        d = tmp_path / "triage" / "AGFHC_1"
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def go():
        await B.ensure_group(r, "triage")
        orch = Bus(r, "orchestrator")
        await orch.send("triage", "triage_request", "run-1", artifact_path=str(arch),
                        reply_to="agents:orchestrator:in", key="run-1/sim/x")
        await orch.send("triage", "triage_request", "run-1", artifact_path=str(tmp_path / "missing"),
                        reply_to="agents:orchestrator:in", key="run-1/sim/y")
        w = Worker(r, "triage", make_handler(fake_triage, save), consumer="w", block_ms=10, log=lambda *_: None)
        await w.step()
        await w.step()
        return await orch.find_reply("run-1/sim/x"), await orch.find_reply("run-1/sim/y")
    ok, missing = run(go())
    assert ok["type"] == "triage_done" and json.loads(ok["note"])["top"][0]["category"] == "hardware"
    assert missing["type"] == "triage_failed" and "archive not found" in missing["note"]


def test_orchestrator_triage_over_the_bus(env, tmp_path, monkeypatch):
    """Dry-run on two nodes: sim3 fails, the orchestrator sends it to a triage worker running beside it over the
    bus, and the run's triage shows the worker's answer."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    from agents.orchestrator.graph import Audit, build_graph
    from agents.test_planner.graph import _text, build_graph as planner_graph
    from tests.test_orchestrator import start
    from tests.test_test_planner import FakeModel, plan
    monkeypatch.setenv("SIM_NODES", "sim3")
    monkeypatch.setenv("SIM_SCENARIO", "pass")
    monkeypatch.setenv("SIM_SCENARIO_sim3", "fail")
    monkeypatch.setenv("AGENT_ARTIFACT_ROOTS", str(env["tmp"]))
    r = FakeAsyncRedis(decode_responses=True)
    m = FakeModel(plans=[plan("all_lvl1")])

    async def fake_triage(a):
        return {"ranked": [{"hyp": {"category": "hardware", "cause": "HBM ECC on GPU 5"}, "score": 0.9}],
                "draft": {"summary": "HBM errors"}}

    async def preflight(node):
        return _text(await env["tools"]["preflight"](node=node))

    async def go():
        await B.ensure_group(r, "triage")
        stop = asyncio.Event()
        w = Worker(r, "triage", make_handler(fake_triage, lambda s: tmp_path / "t"), consumer="w",
                   visibility_s=600, block_ms=50, log=lambda *_: None)
        worker = asyncio.create_task(w.run(stop))
        g = build_graph(lambda: planner_graph(env["tools"], m, m, None, poll_s=0.5, recipes=env["store"]),
                        preflight, recipes=env["store"], checkpointer=InMemorySaver(), home=tmp_path / "runs",
                        bus=Bus(r, "orchestrator"), triage_timeout_s=30)
        cfg = {"configurable": {"thread_id": "rb"}, "recursion_limit": 200}
        out = await g.ainvoke(start("rb", nodes=("sim", "sim3")), cfg)
        while "__interrupt__" in out:
            out = await g.ainvoke(Command(resume={"action": "approve"}), cfg)
        stop.set()
        await worker
        return out
    out = run(go())
    assert out["phase"] == "FAILED"
    (t,) = out["triage"]
    assert t["node"] == "sim3" and t["top"][0]["cause"] == "HBM ECC on GPU 5" and not t.get("error")
    assert [e["via"] for e in Audit("rb", tmp_path / "runs").events() if e["event"] == "triage"] == ["bus"]


from tests.test_test_planner import env  # noqa: E402,F401  (fixture)
