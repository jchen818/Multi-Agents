"""Agents metrics (Phase 6.4): worker counters and heartbeat, the exporter's text, and the alert rules."""
import asyncio
import json
import re
import sys
from pathlib import Path

import fakeredis
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from core import bus as B  # noqa: E402
from core.metrics import HELP, collect_files, scrape  # noqa: E402


def test_worker_counts_outcomes_and_stays_alive_during_long_work():
    server = fakeredis.FakeServer()
    ar = fakeredis.FakeAsyncRedis(server=server, decode_responses=True)
    r = fakeredis.FakeRedis(server=server, decode_responses=True)
    seen_alive = []

    async def slow(bus, m):
        seen_alive.append(r.exists("agents:triage:alive:w"))
        if m["run_id"] == "run-bad":
            raise RuntimeError("model down")

    async def go():
        await B.ensure_group(ar, "triage")
        orch = B.Bus(ar, "orchestrator")
        await orch.send("triage", "triage_request", "run-ok")
        await orch.send("triage", "triage_request", "run-bad")
        await ar.xadd(B.stream("triage"), {"id": "x", "run_id": "r", "from": "orchestrator", "to": "triage",
                                           "type": "rm_rf"})
        stop = asyncio.Event()
        w = B.Worker(ar, "triage", slow, consumer="w", visibility_s=600, block_ms=10, log=lambda *_: None)
        t = asyncio.create_task(w.run(stop))
        await asyncio.sleep(0.5)
        stop.set()
        await t
    asyncio.run(go())
    assert seen_alive and all(seen_alive)                       # heartbeat set while handling
    assert not r.exists("agents:triage:alive:w")                # removed on a clean stop
    text = scrape(r, REPO / "nonexistent")
    assert 'agents_messages_total{agent="triage",outcome="done"} 1.0' in text
    assert 'agents_messages_total{agent="triage",outcome="retry"} 1.0' in text
    assert "agents_dead_letters_total 1.0" in text
    assert 'agents_queue_pending{agent="triage"} 1' in text    # the failed one waits for redelivery
    assert 'agents_up{agent="triage"} 0' in text and "agents_scrape_error 0" in text


def test_redis_down_is_a_metric_not_a_crash():
    class Down:
        def __getattr__(self, name):
            def f(*a, **k):
                raise ConnectionError("refused")
            return f
    text = scrape(Down(), REPO / "nonexistent")
    assert "agents_scrape_error 1" in text


def test_run_and_coder_state(tmp_path):
    runs = tmp_path / "state" / "runs"
    (runs / "r1").mkdir(parents=True)
    (runs / "r1" / "run.json").write_text(json.dumps({"phase": "DONE"}))
    (runs / "r2").mkdir()
    (runs / "r2" / "audit.jsonl").write_text("{}\n")
    cr = tmp_path / "state" / "coder" / "runs"
    cr.mkdir(parents=True)
    (cr / "a.json").write_text(json.dumps({"outcome": "ready", "pushed": False, "source": {"run_id": "r1"}}))
    (cr / "b.json").write_text(json.dumps({"outcome": "ready", "pushed": True, "source": {"run_id": "r1"}}))
    got = {(n, tuple(sorted(l.items()))): v for n, l, v in collect_files(tmp_path, now=10 ** 10)}
    assert got[("agents_runs", (("phase", "done"),))] == 1
    assert got[("agents_runs", (("phase", "in_progress"),))] == 1
    assert got[("agents_run_in_progress_oldest_seconds", ())] > 0
    assert got[("agents_coder_ready_to_push", ())] == 1


def test_alert_rules_use_exported_metrics():
    rules = yaml.safe_load((REPO / "deploy/prometheus/agents-alerts.yml").read_text())["groups"][0]["rules"]
    names = {r["alert"] for r in rules}
    assert {"AgentWorkerDown", "AgentQueueStuck", "AgentDeadLetters", "AgentsRedisUnreachable",
            "AgentsExporterDown"} <= names
    for r in rules:
        for m in re.findall(r"\b(agents_[a-z_]+)", r["expr"]):
            assert m in HELP, f"{r['alert']} uses {m}, which the exporter does not export"
    unit = (REPO / "deploy/systemd/agents-metrics.service").read_text()
    assert "core.metrics" in unit and "NoNewPrivileges=yes" in unit


def test_only_allowed_clients_get_metrics():
    import threading
    import urllib.error
    import urllib.request
    import socket
    from core import metrics as M
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    r = fakeredis.FakeRedis(decode_responses=True)
    threading.Thread(target=M.serve, args=(f"127.0.0.1:{port}", r, {"10.46.252.115"}), daemon=True).start()
    import time as _t
    _t.sleep(0.3)
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=3)
        raise AssertionError("an address outside the allow list got metrics")
    except urllib.error.HTTPError as e:
        assert e.code == 403
    s2 = socket.socket()
    s2.bind(("127.0.0.1", 0))
    port2 = s2.getsockname()[1]
    s2.close()
    threading.Thread(target=M.serve, args=(f"127.0.0.1:{port2}", r, {"127.0.0.1"}), daemon=True).start()
    _t.sleep(0.3)
    body = urllib.request.urlopen(f"http://127.0.0.1:{port2}/metrics", timeout=3).read().decode()
    assert "agents_scrape_error 0" in body
