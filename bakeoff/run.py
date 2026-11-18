"""Run the bake-off: the Bugzilla filing flow on 15 cases, once per framework, with the same local model.

    .venv/bin/python bakeoff/run.py                       # both frameworks, all 15 cases
    .venv/bin/python bakeoff/run.py --framework openai_sdk seed-hw-hbm-ue inject-hw-hbm-ue
    .venv/bin/python bakeoff/run.py --summary              # compare the latest result of each framework

The OpenAI Agents SDK lives in its own venv, .venv-bakeoff (it needs websockets<17, the main venv has 17, and its
import name `agents` is also this repo's package name):
    python3 -m venv .venv-bakeoff && .venv-bakeoff/bin/pip install openai-agents==0.24.0 pyyaml
run.py re-runs itself there for --framework openai_sdk. Start it as a script (not `-m`) so the repo root comes after
site-packages on sys.path. Results: state/bakeoff/<framework>-<time>.json. Nothing is filed; file_bug is a fake.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from bakeoff import common as C  # noqa: E402

OUT = ROOT / "state" / "bakeoff"
SDK_PY = ROOT / ".venv-bakeoff" / "bin" / "python"
IMPLS = {"langgraph": ROOT / "bakeoff/impl_langgraph.py", "openai_sdk": ROOT / "bakeoff/impl_openai_sdk.py"}


def loc(path: Path) -> int:
    """Lines of code: not blank, not a comment, not inside the module docstring."""
    lines, n, doc = path.read_text().splitlines(), 0, False
    for i, l in enumerate(lines):
        s = l.strip()
        if i == 0 and s.startswith('"""'):
            doc = not (s.count('"""') == 2)
            continue
        if doc:
            doc = '"""' not in s
            continue
        if s and not s.startswith("#"):
            n += 1
    return n


async def run_framework(name: str, ids=None) -> dict:
    name_model = os.environ["LLM_MODEL_EXECUTOR"]
    base = os.environ.get("LLM_BASE_URL_EXECUTOR") or os.environ["LLM_BASE_URL"]
    callbacks = []
    if name == "langgraph":
        from bakeoff.impl_langgraph import run_case
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=name_model, base_url=base, api_key="EMPTY", temperature=0)
        try:
            from langfuse.langchain import CallbackHandler
            callbacks = [CallbackHandler()]
        except Exception:
            pass

        async def one(c):
            return await run_case(c, model, callbacks)
    else:
        import agents as sdk
        if not hasattr(sdk, "Runner"):
            raise SystemExit("`import agents` found this repo's package, not the OpenAI Agents SDK: start "
                             "bakeoff/run.py as a script (python bakeoff/run.py), not with -m")
        from bakeoff.impl_openai_sdk import local_model, run_case
        model = local_model(name_model, base)

        async def one(c):
            return await run_case(c, model)
    rows = []
    for c in C.cases(ids):
        t0 = time.monotonic()
        try:
            r = await one(c)
            ok, why = C.score(r["ctx"], r["final"])
            row = {"case": c["id"], "expect": c["expect"], "passed": ok, "why": why, "final": r["final"][-300:],
                   "model_calls": r["model_calls"], "tool_calls": r["ctx"].tool_calls,
                   "approval_requests": len(r["ctx"].approval_requests), "filed": r["ctx"].filed}
        except Exception as e:
            row = {"case": c["id"], "expect": c["expect"], "passed": False, "why": f"crashed: {type(e).__name__}: {e}"[:300]}
        row["seconds"] = round(time.monotonic() - t0, 1)
        rows.append(row)
        print(f"[{name}] {'PASS' if row['passed'] else 'FAIL'} {c['id']:28s} {row['why'][:90]} ({row['seconds']}s)")
    res = {"framework": name, "model": name_model, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "loc": loc(IMPLS[name]),
           "passed": sum(r["passed"] for r in rows), "total": len(rows), "rows": rows,
           "traced_to_langfuse": bool(callbacks)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"[{name}] {res['passed']}/{res['total']} · {res['loc']} lines of code")
    return res


def sdk_here() -> bool:
    try:
        import agents
        return hasattr(agents, "Runner")
    except ImportError:
        return False


def summary():
    latest = {}
    for p in sorted(OUT.glob("*.json")):
        r = json.loads(p.read_text())
        latest[r["framework"]] = r
    if not latest:
        raise SystemExit("no results yet")
    cases = [c["id"] for c in C.cases()]
    print(f"{'case':30s} " + " ".join(f"{f:>12s}" for f in latest))
    for cid in cases:
        cells = []
        for r in latest.values():
            row = next((x for x in r["rows"] if x["case"] == cid), None)
            cells.append("-" if row is None else ("pass" if row["passed"] else "FAIL"))
        print(f"{cid:30s} " + " ".join(f"{c:>12s}" for c in cells))
    print()
    for f, r in latest.items():
        rows = r["rows"]
        calls = [x.get("model_calls", 0) for x in rows if "model_calls" in x]
        secs = [x["seconds"] for x in rows]
        print(f"{f:12s} {r['passed']}/{r['total']} passed · {r['loc']} LOC · model calls/case "
              f"{sum(calls) / max(1, len(calls)):.1f} · {sum(secs) / max(1, len(secs)):.0f} s/case · "
              f"Langfuse trace: {'yes' if r.get('traced_to_langfuse') else 'no'} · {r['when']}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--framework", choices=["langgraph", "openai_sdk", "all"], default="all")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    for f in (["langgraph", "openai_sdk"] if a.framework == "all" else [a.framework]):
        if f == "openai_sdk" and not sdk_here():
            if not SDK_PY.exists():
                raise SystemExit(f"OpenAI Agents SDK not installed; create {SDK_PY.parent.parent} (see the top of this file)")
            subprocess.run([str(SDK_PY), __file__, "--framework", "openai_sdk", *a.ids], check=True)
            continue
        await run_framework(f, a.ids or None)
    if a.framework == "all":
        summary()


if __name__ == "__main__":
    asyncio.run(main())
