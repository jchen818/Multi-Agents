"""Control plane (Phase 6.2): Redis Streams between agents, artifacts by path on disk/NFS, a Redis kill switch.

Streams  agents:<name>:in, one per agent, each read by the consumer group <name>. A message is acknowledged
         (XACK) only after the work is done; if a worker dies mid-task, another worker claims the message once it
         has been pending longer than the agent's visibility timeout (XAUTOCLAIM). After MAX_DELIVERIES attempts a
         message goes to agents:dead with the reason, and is acknowledged so it stops coming back.
Messages flat string fields: id, run_id, from, to, type, artifact_path, note, reply_to, key, ts. Bulk data never
         travels on the bus: artifact_path points at a file or directory under an allowed root (state/,
         /srv/nfs/share/agent-runs, /srv/nfs/logs). Each agent accepts only its own message types.
Replies  go to the requester's reply_to stream. A requester waits by reading (XREAD, no group) for the reply whose
         `key` matches its request, so several runs can wait on the same stream without stealing replies.
Kill     the key agents:kill stops every worker before it takes new work, and core.guard.killed() honors it
         before every tool call (as well as the state/KILL file).

    .venv/bin/python -m core.bus status            # streams, pending, dead letters, live workers
    .venv/bin/python -m core.bus kill on|off       # set / clear agents:kill
    .venv/bin/python -m core.bus dead [N]          # last dead letters
"""
import asyncio
import json
import os
import re
import socket
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(os.environ.get("MULTI_AGENTS_ROOT") or Path(__file__).resolve().parents[1])
KILL_KEY = "agents:kill"
DEAD = "agents:dead"
MAX_DELIVERIES = 3
MAXLEN = 10000
FIELDS = ("id", "run_id", "from", "to", "type", "artifact_path", "note", "reply_to", "key", "ts")
NAME_RE = re.compile(r"[a-z][a-z0-9_-]{1,30}")
RUN_RE = re.compile(r"[A-Za-z0-9._:-]{1,120}")

# what each agent accepts on its stream
ACCEPTS = {
    "triage": {"triage_request"},
    "orchestrator": {"triage_done", "triage_failed"},
}
# how long a message may stay pending (worker busy or dead) before another worker claims it
VISIBILITY_S = {"triage": 45 * 60, "orchestrator": 5 * 60}


def stream(name: str) -> str:
    return f"agents:{name}:in"


def artifact_roots() -> list:
    extra = [p for p in os.environ.get("AGENT_ARTIFACT_ROOTS", "").split(":") if p]
    return [Path(p).resolve() for p in [str(ROOT / "state"), "/srv/nfs/share/agent-runs", "/srv/nfs/logs", *extra]]


class BadMessage(ValueError):
    pass


def validate(msg: dict, agent: str) -> dict:
    """Raise BadMessage unless the message is well formed and meant for this agent."""
    extra = set(msg) - set(FIELDS)
    if extra:
        raise BadMessage(f"unknown fields {sorted(extra)}")
    for f in ("id", "run_id", "from", "to", "type"):
        if not msg.get(f):
            raise BadMessage(f"missing {f}")
    if msg["to"] != agent:
        raise BadMessage(f"addressed to {msg['to']}, not {agent}")
    if not NAME_RE.fullmatch(msg["from"]):
        raise BadMessage("bad sender name")
    if not RUN_RE.fullmatch(msg["run_id"]):
        raise BadMessage("bad run_id")
    if msg["type"] not in ACCEPTS.get(agent, set()):
        raise BadMessage(f"{agent} does not accept type {msg['type']!r}")
    if msg.get("reply_to") and not re.fullmatch(r"agents:[a-z][a-z0-9_-]{1,30}:in", msg["reply_to"]):
        raise BadMessage("bad reply_to")
    ap = msg.get("artifact_path", "")
    if ap:
        p = Path(ap).resolve()
        if not any(p == r or r in p.parents for r in artifact_roots()):
            raise BadMessage(f"artifact_path {ap} is outside the allowed roots")
    if len(msg.get("note", "")) > 4000:
        raise BadMessage("note too long (put data in an artifact)")
    return msg


def connect(url: str = None):
    import redis.asyncio as redis
    url = url or os.environ.get("REDIS_URL")
    if not url:
        raise RuntimeError("REDIS_URL is not set (agents.env); see deploy/redis/")
    return redis.from_url(url, decode_responses=True, socket_connect_timeout=5)


class Bus:
    def __init__(self, redis, me: str):
        if not NAME_RE.fullmatch(me):
            raise ValueError(f"bad agent name {me!r}")
        self.r, self.me = redis, me

    async def send(self, to: str, type: str, run_id: str, artifact_path: str = "", note: str = "",
                   reply_to: str = "", key: str = "") -> dict:
        msg = {"id": uuid.uuid4().hex, "run_id": run_id, "from": self.me, "to": to, "type": type,
               "artifact_path": artifact_path, "note": note, "reply_to": reply_to, "key": key,
               "ts": str(time.time())}
        validate(msg, to)                     # refuse to send what the receiver would reject
        msg["stream_id"] = await self.r.xadd(stream(to), {k: v for k, v in msg.items() if k in FIELDS},
                                             maxlen=MAXLEN, approximate=True)
        return msg

    async def reply(self, req: dict, type: str, artifact_path: str = "", note: str = "") -> dict:
        to = (req.get("reply_to") or stream(req["from"])).split(":")[1]
        return await self.send(to, type, req["run_id"], artifact_path, note, key=req.get("key") or req["id"])

    async def find_reply(self, key: str, reply_stream: str = None, scan: int = 2000):
        """A reply already on the stream (e.g. a run resumed after the requester restarted)."""
        for _, f in await self.r.xrevrange(reply_stream or stream(self.me), count=scan):
            if f.get("key") == key:
                return f
        return None

    async def wait_reply(self, key: str, since: str, timeout_s: float, reply_stream: str = None, poll_ms: int = 2000):
        """Wait for the reply with this key that arrives after stream id `since`. None on timeout or kill."""
        s = reply_stream or stream(self.me)
        last, deadline = since, time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if await kill_set(self.r):
                return None
            got = await self.r.xread({s: last}, count=100, block=poll_ms)
            for _, entries in got or []:
                for sid, f in entries:
                    last = sid
                    if f.get("key") == key:
                        return f
        return None


async def kill_set(r) -> bool:
    try:
        return bool(await r.exists(KILL_KEY))
    except Exception:
        return False


async def ensure_group(r, name: str):
    try:
        await r.xgroup_create(stream(name), name, id="0", mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            raise


async def dead_letter(r, name: str, sid: str, fields: dict, reason: str):
    await r.xadd(DEAD, {"stream": stream(name), "stream_id": sid, "reason": reason[:500],
                        "message": json.dumps(fields)[:8000], "ts": str(time.time())}, maxlen=MAXLEN, approximate=True)
    await r.xack(stream(name), name, sid)


class Worker:
    """Consume agents:<name>:in in the consumer group <name>. handler(bus, msg) -> awaitable; raising leaves the
    message pending so it is retried after the visibility timeout (up to MAX_DELIVERIES). The handler also gets
    msg["_attempt"] and msg["_last"] (True on the final attempt: reply with a failure instead of raising)."""

    def __init__(self, redis, name: str, handler, consumer: str = None, visibility_s: float = None,
                 block_ms: int = 5000, log=print):
        self.r, self.name, self.handler = redis, name, handler
        self.consumer = consumer or f"{socket.gethostname()}-{os.getpid()}"
        self.visibility_ms = int(1000 * (visibility_s if visibility_s is not None else VISIBILITY_S.get(name, 600)))
        self.block_ms, self.log = block_ms, log
        self.bus = Bus(redis, name)

    async def deliveries(self, sid: str) -> int:
        p = await self.r.xpending_range(stream(self.name), self.name, sid, sid, 1)
        return int(p[0]["times_delivered"]) if p else 1

    async def handle(self, sid: str, fields: dict) -> str:
        """Process one delivery. Returns 'done', 'dead' or 'retry'."""
        n = await self.deliveries(sid)
        if n > MAX_DELIVERIES:
            await dead_letter(self.r, self.name, sid, fields, f"failed {n - 1} times")
            return "dead"
        try:
            msg = validate(dict(fields), self.name)
        except BadMessage as e:
            await dead_letter(self.r, self.name, sid, fields, f"rejected: {e}")
            return "dead"
        try:
            await self.handler(self.bus, {**msg, "_attempt": n, "_last": n >= MAX_DELIVERIES})
        except Exception as e:
            self.log(f"{self.name}: message {sid} attempt {n} failed: {type(e).__name__}: {e}")
            return "retry"
        await self.r.xack(stream(self.name), self.name, sid)
        return "done"

    async def step(self) -> int:
        """Claim stale messages, then read new ones. Returns the number handled."""
        if await kill_set(self.r):
            return 0
        await self.r.set(f"agents:{self.name}:alive:{self.consumer}", str(time.time()), ex=60)
        handled = 0
        _, claimed, *_ = await self.r.xautoclaim(stream(self.name), self.name, self.consumer,
                                                 min_idle_time=self.visibility_ms, start_id="0-0", count=5)
        for sid, f in claimed:
            if f:
                await self.handle(sid, f)
                handled += 1
        got = await self.r.xreadgroup(self.name, self.consumer, {stream(self.name): ">"}, count=1,
                                      block=self.block_ms)
        for _, entries in got or []:
            for sid, f in entries:
                await self.handle(sid, f)
                handled += 1
        return handled

    async def run(self, stop: asyncio.Event = None):
        await ensure_group(self.r, self.name)
        self.log(f"{self.name} worker {self.consumer} reading {stream(self.name)}")
        while not (stop and stop.is_set()):
            if await kill_set(self.r):
                await asyncio.sleep(5)
                continue
            if not await self.step():
                await asyncio.sleep(0.01)      # yield even if the client returns at once (no blocking read)


# ------------------------------------------------------------------ ops CLI

async def _cli(argv):
    r = connect()
    cmd = argv[0] if argv else "status"
    if cmd == "kill":
        if len(argv) < 2 or argv[1] not in ("on", "off"):
            raise SystemExit("usage: python -m core.bus kill on|off")
        if argv[1] == "on":
            await r.set(KILL_KEY, f"set {time.strftime('%Y-%m-%d %H:%M:%S')} by {os.environ.get('USER', '?')}")
        else:
            await r.delete(KILL_KEY)
        print(f"{KILL_KEY}: {await r.get(KILL_KEY) or 'clear'}")
    elif cmd == "dead":
        n = int(argv[1]) if len(argv) > 1 else 10
        for sid, f in await r.xrevrange(DEAD, count=n):
            print(f"{sid} {f.get('stream')} {f.get('reason')}\n    {f.get('message')[:300]}")
    else:
        print(f"{KILL_KEY}: {await r.get(KILL_KEY) or 'clear'}")
        for name in sorted(ACCEPTS):
            s = stream(name)
            length = await r.xlen(s) if await r.exists(s) else 0
            try:
                groups = await r.xinfo_groups(s) if length else []
            except Exception:
                groups = []
            pend = sum(int(g.get("pending", 0)) for g in groups)
            alive = [k.split(":")[-1] async for k in r.scan_iter(f"agents:{name}:alive:*")]
            print(f"{s:28} length {length:5}  pending {pend:3}  workers {', '.join(alive) or 'none'}")
        print(f"{DEAD:28} length {await r.xlen(DEAD) if await r.exists(DEAD) else 0:5}")
    await r.aclose()


if __name__ == "__main__":
    asyncio.run(_cli(sys.argv[1:]))
