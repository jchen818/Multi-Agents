"""Coding agent as a service (Phase 6): consumes coder_task messages from agents:coder:in.

A task's artifact_path is a task file the orchestrator wrote (state/coder/tasks/<id>.md): a detection check to add
to the gpu-validation suite, drawn from a triage result. The worker runs the coding agent on a fresh worktree and
STOPS BEFORE PUSHING: the change waits, with its run record, until a person reviews it and runs
`python -m agents.coder.graph --push agent/<slug>` (which re-checks it, shows the diff and asks for a typed yes).

    .venv/bin/python -m agents.coder.worker             # runs until stopped; honors agents:kill
    sudo systemctl start agents-coder-worker             # as a service (deploy/systemd)
"""
import asyncio
import signal
import sys
import time
from pathlib import Path

from core.bus import Bus, Worker, connect

ROOT = Path(__file__).resolve().parents[2]


def make_handler(run_task, cfg: dict, log=print):
    """run_task(task_text) -> awaitable (state, branch, path)."""
    from agents.coder.graph import run_record, save_record

    async def handle(bus: Bus, msg: dict):
        task_file = Path(msg.get("artifact_path", ""))
        if not task_file.is_file():
            log(f"coder: task file missing: {task_file}")       # nothing to retry: drop it (acked)
            return
        task = task_file.read_text().strip()
        t0 = time.monotonic()
        state, branch, path = await run_task(task)
        rec = run_record(state, task, branch, path, int(time.monotonic() - t0), cfg,
                         source={"run_id": msg["run_id"], "task_file": str(task_file), "from": msg["from"]})
        saved = save_record(rec)
        log(f"coder: {msg['run_id']}: {rec['outcome']} {branch} {rec.get('reason', '')} -> {saved}")
    return handle


async def main():
    from agents.coder.graph import load_cfg, run
    from agents.coder.gitops import Repo, load_token
    from core.models import get_model
    cfg = load_cfg()
    if cfg.get("sandbox", {}).get("backend") != "docker":
        raise SystemExit("config/coder.yaml: sandbox.backend must be docker")
    repo, model = Repo(cfg, load_token()), get_model("executor")

    async def run_task(task):
        return await run(task, cfg, repo, model)

    r = connect()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    await Worker(r, "coder", make_handler(run_task, cfg)).run(stop)
    print("coder worker stopped")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    asyncio.run(main())
