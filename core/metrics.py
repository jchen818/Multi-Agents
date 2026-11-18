"""Prometheus exporter for the agents (Phase 6.4): one small process on nfs-server, scraped by the Prometheus on
mitacai. It reads Redis (queues, pending work, live workers, counters the workers keep in agents:metrics, the kill
switch) and the run/coder state files. Workers keep no HTTP server of their own. Read-only: it changes nothing.

    .venv/bin/python -m core.metrics --listen 10.46.252.118:9810      # or AGENTS_METRICS_LISTEN
    curl -s http://127.0.0.1:9810/metrics                              # only allowed clients get an answer

nfs-server has no firewall, so the exporter itself answers only the addresses in AGENTS_METRICS_ALLOW (default:
127.0.0.1 and the Prometheus host 10.46.252.115); everyone else gets 403.

Prometheus rules: deploy/prometheus/agents-alerts.yml.
"""
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(os.environ.get("MULTI_AGENTS_ROOT") or Path(__file__).resolve().parents[1])
AGENTS = ("triage", "coder", "orchestrator")
WORKERS = ("triage", "coder")
HELP = {
    "agents_up": ("gauge", "Live workers per agent (heartbeat in the last 60 s)"),
    "agents_queue_length": ("gauge", "Messages in the agent's stream (kept for history, capped)"),
    "agents_queue_pending": ("gauge", "Messages delivered to a worker and not yet acknowledged"),
    "agents_queue_oldest_pending_seconds": ("gauge", "Idle time of the oldest unacknowledged message"),
    "agents_messages_total": ("counter", "Messages handled by outcome (done, retry, dead)"),
    "agents_handle_seconds_sum": ("counter", "Total seconds spent handling messages"),
    "agents_handle_seconds_count": ("counter", "Handled deliveries"),
    "agents_dead_letters_total": ("counter", "Messages moved to agents:dead"),
    "agents_kill_switch": ("gauge", "1 when agents:kill or state/KILL is set"),
    "agents_runs": ("gauge", "Orchestrator runs by final phase (in_progress = no run.json yet)"),
    "agents_run_in_progress_oldest_seconds": ("gauge", "Age of the oldest orchestrator run without a report"),
    "agents_coder_ready_to_push": ("gauge", "Prepared coder changes waiting for a person to --push"),
    "agents_scrape_error": ("gauge", "1 if this scrape could not read Redis"),
}


def _labels(d: dict) -> str:
    return "{" + ",".join(f'{k}="{str(v)}"' for k, v in d.items()) + "}" if d else ""


def collect_redis(r) -> list:
    """[(metric, labels, value)] from Redis. r: a synchronous redis client (decode_responses=True)."""
    from core.bus import KILL_KEY, METRICS, stream
    out = []
    counters = r.hgetall(METRICS) or {}
    for a in AGENTS:
        s = stream(a)
        exists = r.exists(s)
        out.append(("agents_queue_length", {"agent": a}, r.xlen(s) if exists else 0))
        pending, oldest = 0, 0.0
        if exists:
            try:
                info = r.xpending(s, a)
                pending = int(info.get("pending", 0))
                if pending:
                    first = r.xpending_range(s, a, "-", "+", 50)
                    oldest = max((p.get("time_since_delivered", 0) for p in first), default=0) / 1000.0
            except Exception:
                pass                         # no consumer group yet
        out.append(("agents_queue_pending", {"agent": a}, pending))
        out.append(("agents_queue_oldest_pending_seconds", {"agent": a}, round(oldest, 1)))
        for oc in ("done", "retry", "dead"):
            out.append(("agents_messages_total", {"agent": a, "outcome": oc}, float(counters.get(f"{a}:{oc}", 0))))
        out.append(("agents_handle_seconds_sum", {"agent": a}, float(counters.get(f"{a}:seconds_sum", 0))))
        out.append(("agents_handle_seconds_count", {"agent": a}, float(counters.get(f"{a}:seconds_count", 0))))
    for a in WORKERS:
        out.append(("agents_up", {"agent": a}, sum(1 for _ in r.scan_iter(f"agents:{a}:alive:*"))))
    out.append(("agents_dead_letters_total", {}, float(counters.get("dead_total", 0))))
    kill = bool(r.exists(KILL_KEY)) or Path(os.getenv("AGENTS_KILL_FILE", str(ROOT / "state" / "KILL"))).exists()
    out.append(("agents_kill_switch", {}, int(kill)))
    return out


def collect_files(root: Path = ROOT, now: float = None) -> list:
    now = now or time.time()
    out, phases, oldest = [], {}, 0.0
    for d in sorted((root / "state" / "runs").glob("*")):
        f = d / "run.json"
        if f.exists():
            try:
                ph = (json.loads(f.read_text()).get("phase") or "unknown").lower()
            except ValueError:
                ph = "unknown"
        else:
            ph = "in_progress"
            audit = d / "audit.jsonl"
            started = audit.stat().st_ctime if audit.exists() else d.stat().st_ctime
            oldest = max(oldest, now - started)
        phases[ph] = phases.get(ph, 0) + 1
    for ph, n in sorted(phases.items()):
        out.append(("agents_runs", {"phase": ph}, n))
    out.append(("agents_run_in_progress_oldest_seconds", {}, round(oldest, 1)))
    ready = 0
    for p in (root / "state" / "coder" / "runs").glob("*.json"):
        try:
            rec = json.loads(p.read_text())
        except ValueError:
            continue
        ready += rec.get("outcome") == "ready" and not rec.get("pushed") and bool(rec.get("source"))
    out.append(("agents_coder_ready_to_push", {}, ready))
    return out


def render(samples: list) -> str:
    lines, seen = [], set()
    for name, labels, value in samples:
        if name not in seen and name in HELP:
            kind, text = HELP[name]
            lines += [f"# HELP {name} {text}", f"# TYPE {name} {kind}"]
            seen.add(name)
        lines.append(f"{name}{_labels(labels)} {value}")
    return "\n".join(lines) + "\n"


def scrape(r, root: Path = ROOT) -> str:
    samples = []
    try:
        samples += collect_redis(r)
        samples.append(("agents_scrape_error", {}, 0))
    except Exception:
        samples.append(("agents_scrape_error", {}, 1))
    samples += collect_files(root)
    return render(samples)


def allowed_clients() -> set:
    raw = os.environ.get("AGENTS_METRICS_ALLOW", "127.0.0.1,10.46.252.115")
    return {a.strip() for a in raw.split(",") if a.strip()}


def serve(listen: str, r, allow: set = None):
    host, port = listen.rsplit(":", 1)
    allow = allow if allow is not None else allowed_clients()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.client_address[0] not in allow:
                self.send_error(403)
                return
            if self.path.split("?")[0] != "/metrics":
                self.send_error(404)
                return
            body = scrape(r).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f"agents metrics on http://{listen}/metrics (allowed: {', '.join(sorted(allow))})")
    ThreadingHTTPServer((host, int(port)), H).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default=os.environ.get("AGENTS_METRICS_LISTEN", "127.0.0.1:9810"))
    ap.add_argument("--once", action="store_true", help="print one scrape and exit")
    a = ap.parse_args()
    import redis
    r = redis.from_url(os.environ["REDIS_URL"], decode_responses=True, socket_timeout=3, socket_connect_timeout=3)
    if a.once:
        sys.stdout.write(scrape(r))
        return
    serve(a.listen, r)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
