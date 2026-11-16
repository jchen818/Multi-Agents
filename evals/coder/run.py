"""Coder eval: run the agent on every task in tasks.yaml against a fresh worktree of gpu-validation, never pushing.

    .venv/bin/python evals/coder/run.py                 # all tasks (~5-10 min each on the local model)
    .venv/bin/python evals/coder/run.py power_cap numa_split
    .venv/bin/python evals/coder/run.py --keep          # keep the worktrees to read the code it wrote

Writes state/coder/eval/<timestamp>.json and prints a score. A 'ready' task only counts if its new checks FAIL on a
tampered copy of the recording: a check that passes on wrong data is not a check.
"""
import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.coder.graph import last_feedback, load_cfg, run  # noqa: E402
from agents.coder.sandbox import failed_tests, run_pytest  # noqa: E402

TASKS = Path(__file__).with_name("tasks.yaml")
PASS_MIN = 7          # of 9


def set_path(doc, dotted: str, value):
    keys = [int(k) if k.isdigit() else k for k in dotted.split(".")]
    cur = doc
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value


def tampered_copy(path: Path, tamper: dict) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="coder-eval-"))
    dst = tmp / "suite"
    shutil.copytree(path, dst, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    rec = sorted(p for p in (dst / "recordings").glob("*") if (p / "index.json").exists())[-1]
    f = rec / "out" / f"{tamper['file']}.txt"
    doc = json.loads(f.read_text())
    for k, v in tamper.get("set", {}).items():
        set_path(doc, k, v)
    f.write_text(json.dumps(doc))
    return dst


def score(case: dict, state: dict, path: Path, cfg: dict, runner=run_pytest) -> tuple:
    out, exp = state.get("outcome"), case["expect"]
    rep = state.get("report") or {}
    if exp == "refused":
        return out == "refused", f"outcome {out}: {state.get('reason', '')[:100]}"
    if out != "ready":
        return False, f"outcome {out}: {state.get('reason', '')[:100]}"
    checks_ = [t for t in rep.get("new_tests", []) if not t.startswith("tests/test_checks_catch.py")]
    if exp == "capture":
        if not rep.get("new_commands"):
            return False, "no new catalog command for data the recording lacks"
        skipped = "SKIPPED" in state["tests"]["tail"] or "skipped" in state["tests"]["tail"]
        return skipped, f"new commands {rep['new_commands']}; " + ("test skips until re-capture" if skipped
                                                                    else "expected the new test to skip")
    if state.get("findings"):
        return False, f"declared findings on a healthy recording: {list(state['findings'])}"
    if not checks_:
        return False, "no new check"
    bad = tampered_copy(path, case["tamper"])
    try:
        res = runner(bad, cfg, checks_)
    finally:
        shutil.rmtree(bad.parent, ignore_errors=True)
    caught = [t for t in checks_ if t in failed_tests(res["output"])]
    return bool(caught), f"tampered data caught by {caught or 'NONE of'} {checks_}"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg()
    cases = [c for c in yaml.safe_load(TASKS.read_text()) if not args.ids or c["id"] in args.ids]
    from agents.coder.gitops import STATE, Repo, load_token
    from core.models import get_model
    repo, model = Repo(cfg, load_token()), get_model("executor")
    rows = []
    for c in cases:
        t0 = time.monotonic()
        state, branch, path = await run(c["task"], cfg, repo, model)
        ok, why = score(c, state, path, cfg)
        rows.append({"id": c["id"], "ok": ok, "why": why, "outcome": state.get("outcome"),
                     "rounds": state.get("iteration"), "seconds": int(time.monotonic() - t0), "branch": branch,
                     "history": state.get("history"), "critique": state.get("critique"),
                     "summary": state.get("summary")})
        print(f"[{'PASS' if ok else 'FAIL'}] {c['id']:15s} {why}  ({rows[-1]['seconds']}s, rounds {state.get('iteration')})")
        if not ok and state.get("outcome") == "gave_up":
            print("\n".join("      " + l for l in last_feedback(state).splitlines()[-12:]))
        if not args.keep:
            repo.drop(path, branch)
    n = sum(r["ok"] for r in rows)
    print(f"\ncoder eval {n}/{len(rows)}" + (f" (gate needs {PASS_MIN}/9)" if len(rows) == 9 else ""))
    out = STATE / "eval" / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"saved {out}")
    sys.exit(0 if len(rows) < 9 or n >= PASS_MIN else 1)


if __name__ == "__main__":
    asyncio.run(main())
