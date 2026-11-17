"""Phase 6.4 deployment files: the systemd unit and the Redis compose file keep their safety settings."""
import re
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
UNIT = (REPO / "deploy/systemd/agents-triage-worker@.service").read_text()


def test_unit_runs_unprivileged_and_restarts():
    for line in ("User=@USER@", "NoNewPrivileges=yes", "PrivateTmp=yes", "ProtectSystem=full", "Restart=on-failure",
                 "KillSignal=SIGTERM", "WorkingDirectory=@ROOT@"):
        assert line in UNIT, line
    assert "agents.triage.worker" in UNIT
    assert not re.search(r"(?i)password|token|REDIS_URL=", UNIT)          # secrets stay in agents.env


def test_install_script_parses():
    subprocess.run(["bash", "-n", str(REPO / "deploy/systemd/install.sh")], check=True)


def test_redis_is_local_only_and_password_protected():
    c = yaml.safe_load((REPO / "deploy/redis/docker-compose.yml").read_text())["services"]["redis"]
    assert all(p.startswith("127.0.0.1:") for p in c["ports"])
    cmd = " ".join(c["command"])
    assert "--requirepass" in cmd and "noeviction" in cmd and "--appendonly yes" in cmd


def test_worker_stops_when_asked():
    import asyncio
    from fakeredis import FakeAsyncRedis
    from core.bus import Worker
    r = FakeAsyncRedis(decode_responses=True)

    async def go():
        stop = asyncio.Event()
        w = Worker(r, "triage", lambda b, m: None, consumer="w", block_ms=10, log=lambda *_: None)
        t = asyncio.create_task(w.run(stop))
        await asyncio.sleep(0.1)
        stop.set()
        await asyncio.wait_for(t, 2)
    asyncio.run(go())
