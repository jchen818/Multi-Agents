"""Run the suite's pytest without trusting the code under test.

docker backend (the only one the CLI accepts): a throwaway container with no network, the checkout mounted read-only,
all capabilities dropped, no new privileges, bounded memory/CPU/pids/time, writable /tmp only. Tests run in replay
mode against recordings/; --node is never passed, and with no network a live run could not reach a node anyway.

local backend: plain subprocess, for this repo's own unit tests and the eval scorer in CI-less setups. Not isolated.
"""
import os
import subprocess
import sys
from pathlib import Path

PYTEST = ["python", "-m", "pytest", "-p", "no:cacheprovider", "-q", "-rfEsx", "-o", "addopts="]


def docker_cmd(root: Path, cfg: dict, args: list) -> list:
    sb = cfg.get("sandbox", {})
    return ["docker", "run", "--rm", "--network", "none", "--read-only",
            "--tmpfs", "/tmp:rw,size=256m", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "256", "--memory", str(sb.get("memory", "2g")), "--cpus", str(sb.get("cpus", "2")),
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp",
            "-v", f"{Path(root).resolve()}:/w:ro", "-w", "/w",
            sb.get("image", "multi-agents/coder-sandbox:1"), *PYTEST, *args]


def run_pytest(root: Path, cfg: dict, args: list) -> dict:
    """{'rc': int, 'output': str}. rc 0 = all passed (xfails allowed), 1 = failures, other = could not run."""
    for a in args:
        if a.startswith("-") or ".." in a:
            raise ValueError(f"pytest argument {a!r} not allowed")
    backend = cfg.get("sandbox", {}).get("backend", "docker")
    timeout = int(cfg.get("sandbox", {}).get("timeout_s", 300))
    if backend == "docker":
        cmd, env, cwd = docker_cmd(root, cfg, args), None, None
    elif backend == "local":
        cmd = [sys.executable, *PYTEST[1:], *args]
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        cwd = str(root)
    else:
        raise ValueError(f"unknown sandbox backend {backend!r}")
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"rc": 124, "output": f"pytest timed out after {timeout}s"}
    except FileNotFoundError as e:
        return {"rc": 127, "output": f"cannot start the sandbox: {e}"}
    return {"rc": r.returncode, "output": (r.stdout + ("\n" + r.stderr if r.stderr.strip() else "")).strip()}


def failed_tests(output: str) -> list:
    """Node ids of FAILED/ERROR lines in pytest -rs/-q output."""
    out = []
    for line in output.splitlines():
        if line.startswith(("FAILED ", "ERROR ")):
            out.append(line.split()[1].split(" - ")[0])
    return out


def failure_messages(output: str) -> dict:
    """{node id: the assertion message from pytest's short summary line} for failed/errored tests."""
    out = {}
    for line in output.splitlines():
        if line.startswith(("FAILED ", "ERROR ")):
            rest = line.split(" ", 1)[1]
            tid, _, msg = rest.partition(" - ")
            out[tid.strip()] = msg.strip()
    return out
