"""Validation Orchestrator (Phase 6.1): one run from requirements to report, using the existing agents as workers.

    REQUESTED -> PLANNING -> AWAITING_APPROVAL -> PREPARING -> EXECUTING -> REPORTING -> DONE
                                                                   |
                                                                FAILED -> ANALYZING -> REPORTING

Workers: the Phase 2 test planner/executor (planning, the approval interrupt, execution on the AGFHC tools), a
PREPARING step that checks the node's readiness before the first test starts, and Phase 3 triage for every failed
step (ranked causes + a bug draft that a person files with agents.triage.file). Reporting writes the run report.

Two human decision points, both LangGraph interrupts and both written to the audit log:
  1. plan approval (the planner's own approve step: approve / edit / reject)
  2. readiness exceptions: GPUs busy, recent GPU kernel errors, unreadable firmware versions, or a version that
     differs from what the request expects (--expect "System BIOS=V1.04"). Expected versions come from the run
     request, never from code: they change with every BKC.
Nothing runs before approval; every run is resumable (SQLite checkpoints) and stops at the kill switch.

    .venv/bin/python -m agents.orchestrator.graph --node sim "quick health check of HBM and XGMI"
    .venv/bin/python -m agents.orchestrator.graph --node sim --expect "System BIOS=V1.04" "smoke test"
    .venv/bin/python -m agents.orchestrator.graph --resume <run_id>
    .venv/bin/python -m agents.orchestrator.graph --list
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from core.guard import killed

ROOT = Path(os.environ.get("MULTI_AGENTS_ROOT") or Path(__file__).resolve().parents[2])
RUNS = ROOT / "state" / "runs"
HUMAN_EVENTS = ("plan_decision", "exception_decision")


class State(TypedDict, total=False):
    # orchestrator
    run_id: str
    phase: str              # REQUESTED, RUNNING, ANALYZING, REPORTING, DONE, FAILED, REJECTED, STOPPED
    expect: dict            # versions the request expects, e.g. {"System BIOS": "V1.04"}
    exceptions: list        # readiness exceptions found in PREPARING
    exception_decision: dict
    triage: list            # [{recipe, run_id, archive, top, draft_summary, saved}]
    run_report: str
    # planner (shared keys: the planner graph runs as a subgraph on this state)
    requirements: str
    node: str
    platform: str
    preflight: str
    plan: dict
    current: int
    results: list
    replans: int
    last_error: str
    status: str
    report: str
    dryrun: bool
    versions: dict
    recipes: list
    recipe_saved: str


# ------------------------------------------------------------------ audit log

class Audit:
    """Append-only JSON lines per run: state transitions and every human decision."""

    def __init__(self, run_id: str, home: Path = RUNS):
        self.dir = Path(home) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "audit.jsonl"

    def log(self, event: str, **data):
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **data}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def events(self) -> list:
        return [json.loads(l) for l in self.path.read_text().splitlines()] if self.path.exists() else []

    def human_events(self) -> list:
        return [e for e in self.events() if e["event"] in HUMAN_EVENTS]


# ------------------------------------------------------------------ readiness (PREPARING)

def readiness_exceptions(preflight: str, versions: dict, expect: dict) -> list:
    """What a person must accept before tests start on this node. Read from the planner's preflight; nothing pinned."""
    out = []
    m = re.search(r"\*\*GPU usage:\*\*\s*(.+)", preflight or "")
    usage = (m.group(1).strip() if m else "unknown")
    if not usage.startswith("all ") or "idle" not in usage:
        out.append(f"GPUs not idle: {usage}")
    m = re.search(r"kernel errors:\*\*\s*```\s*(.*?)```", preflight or "", re.S)
    kern = (m.group(1).strip() if m else "")
    if kern and kern != "none":
        out.append("recent amdgpu/RAS/AER kernel errors:\n" + "\n".join(kern.splitlines()[-5:]))
    for key in ("VBIOS", "System BIOS", "ROCm"):
        if (versions.get(key) or "unknown").strip().lower() in ("unknown", "", "n/a"):
            out.append(f"{key} version could not be read")
    for key, want in (expect or {}).items():
        got = (versions.get(key) or "unknown").strip()
        if got != want.strip():
            out.append(f"{key} is {got}, the request expects {want}")
    return out


def make_prepare(audit_for):
    """The planner's prepare hook: runs after plan approval, before the first test."""

    def prepare(state):
        audit = audit_for(state)
        if killed():
            audit.log("stopped", where="PREPARING")
            return {"status": "STOPPED", "report": "stopped by the kill switch before any test started"}
        exc = readiness_exceptions(state.get("preflight", ""), state.get("versions") or {}, state.get("expect") or {})
        audit.log("state", phase="PREPARING", exceptions=exc)
        if not exc:
            return {"status": "EXECUTING", "exceptions": []}
        decision = interrupt({"kind": "exceptions", "node": state.get("node"), "exceptions": exc,
                              "markdown": "### Readiness exceptions on " + str(state.get("node")) + "\n"
                                          + "\n".join(f"- {e}" for e in exc)}) or {}
        audit.log("exception_decision", action=decision.get("action"), reason=decision.get("reason", ""),
                  exceptions=exc, by=decision.get("by", ""))
        if decision.get("action") == "proceed" and len(decision.get("reason", "").strip()) >= 8:
            return {"status": "EXECUTING", "exceptions": exc, "exception_decision": decision}
        return {"status": "REJECTED", "exceptions": exc, "exception_decision": decision,
                "report": f"not started: readiness exceptions not accepted ({decision.get('reason') or 'aborted'})"}
    return prepare


# ------------------------------------------------------------------ helpers

def archive_of(analysis: str) -> str:
    """Local archive dir from the executor's '**Archived** (...) to `host:/dir/`' line."""
    m = re.search(r"\*\*Archived\*\*[^`]*`([^`]+)`", analysis or "")
    if not m:
        return ""
    target = m.group(1)
    path = target.split(":", 1)[1] if ":" in target and not target.startswith("/") else target
    return path.rstrip("/")


def top_cause(c: dict) -> dict:
    """One ranked triage candidate ({hyp: {category, cause}, score, ...}) as {category, cause, score}."""
    h = c.get("hyp") or {}
    return {"category": h.get("category"), "cause": h.get("cause"), "score": round(float(c.get("score") or 0), 2)}


def run_markdown(state: dict) -> str:
    lines = [f"# Validation run {state['run_id']}", "",
             f"- Node: {state.get('node')} ({'dry-run' if state.get('dryrun', True) else 'live'})",
             f"- Requirements: {state.get('requirements')}",
             f"- Outcome: **{state.get('phase')}**"]
    if state.get("plan", {}).get("steps"):
        lines.append(f"- Plan ({state['plan'].get('tier')}): " + ", ".join(s["recipe"] for s in state["plan"]["steps"]))
    if state.get("exceptions"):
        d = state.get("exception_decision") or {}
        lines.append(f"- Readiness exceptions ({d.get('action', 'none')}: {d.get('reason', '')}): "
                     + "; ".join(e.splitlines()[0] for e in state["exceptions"]))
    lines += ["", "## Execution", state.get("report") or "(nothing ran)"]
    if state.get("triage"):
        lines += ["", "## Failure analysis"]
        for t in state["triage"]:
            lines.append(f"### `{t['recipe']}` run `{t['run_id']}`")
            if t.get("error"):
                lines.append(f"Triage not run: {t['error']}")
                continue
            for i, r in enumerate(t["top"], 1):
                lines.append(f"{i}. **{r['category']}**: {r['cause']} (score {r.get('score', '?')})")
            if not t["top"]:
                lines.append("No cause survived verification; see the full triage for the dropped candidates.")
            if t.get("draft_summary"):
                lines.append(f"\nBug draft (not filed): *{t['draft_summary']}*\n"
                             f"To file it: `.venv/bin/python -m agents.triage.file draft {t['archive']} --severity <sev>`"
                             f" then `create <token>` (a person files bugs).")
            lines.append(f"Full triage: `{t['saved']}/triage.md`")
    return "\n".join(lines)


# ------------------------------------------------------------------ graph

def build_graph(planner_graph_factory, triage_run=None, checkpointer=None, home: Path = RUNS):
    """planner_graph_factory(prepare) -> compiled planner graph (no checkpointer: it runs as a subgraph).
    triage_run(archive) -> awaitable triage state dict (agents.triage.graph output) or None to skip analysis."""

    def audit_for(state):
        return Audit(state["run_id"], home)

    planner = planner_graph_factory(make_prepare(audit_for))

    def requested(state: State):
        audit = audit_for(state)
        if killed():
            audit.log("stopped", where="REQUESTED")
            return {"phase": "STOPPED"}
        audit.log("state", phase="REQUESTED", node=state.get("node"), requirements=state.get("requirements"),
                  expect=state.get("expect") or {})
        return {"phase": "RUNNING", "triage": []}

    def after_planner(state: State):
        audit = audit_for(state)
        st = state.get("status")
        audit.log("state", phase="EXECUTED", planner_status=st,
                  results=[{k: r.get(k) for k in ("recipe", "run_id", "verdict")} for r in state.get("results") or []])
        if st == "FAILED" and any(r.get("verdict") == "FAIL" for r in state.get("results") or []):
            return Command(goto="analyze", update={"phase": "ANALYZING"})
        phase = {"DONE": "DONE", "REJECTED": "REJECTED", "REFUSED": "REJECTED", "STOPPED": "STOPPED"}.get(st, "FAILED")
        return Command(goto="report", update={"phase": phase})

    async def analyze(state: State):
        audit = audit_for(state)
        out = []
        for r in state.get("results") or []:
            if r.get("verdict") != "FAIL":
                continue
            if killed():
                audit.log("stopped", where="ANALYZING")
                break
            arch = archive_of(r.get("analysis", ""))
            item = {"recipe": r["recipe"], "run_id": r.get("run_id"), "archive": arch, "top": []}
            if triage_run is None:
                item["error"] = "triage disabled"
            elif not arch or not Path(arch).is_dir():
                item["error"] = f"archive not found on this host: {arch or 'no archive line'}"
            else:
                try:
                    t = await triage_run(arch)
                    item["top"] = [top_cause(c) for c in (t.get("ranked") or [])[:3]]
                    item["draft_summary"] = (t.get("draft") or {}).get("summary", "")
                    item["saved"] = t.get("_saved", "")
                except Exception as e:
                    item["error"] = f"{type(e).__name__}: {e}"[:300]
            audit.log("triage", recipe=item["recipe"], run_id=item["run_id"], top=item["top"], error=item.get("error"))
            out.append(item)
        return {"triage": out, "phase": "FAILED"}

    def report(state: State):
        audit = audit_for(state)
        md = run_markdown(state)
        (audit.dir / "report.md").write_text(md)
        (audit.dir / "run.json").write_text(json.dumps(
            {k: state.get(k) for k in ("run_id", "phase", "node", "requirements", "expect", "plan", "exceptions",
                                       "exception_decision", "triage", "status", "dryrun", "versions")},
            indent=1, default=str))
        audit.log("state", phase="REPORTING", outcome=state.get("phase"), report=str(audit.dir / "report.md"))
        return {"run_report": md}

    g = StateGraph(State)
    g.add_node("requested", requested)
    g.add_node("planner", planner)
    g.add_node("after_planner", after_planner)
    g.add_node("analyze", analyze)
    g.add_node("report", report)
    g.add_edge(START, "requested")
    g.add_conditional_edges("requested", lambda s: "report" if s.get("phase") == "STOPPED" else "planner",
                            ["report", "planner"])
    g.add_edge("planner", "after_planner")
    g.add_edge("analyze", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)


# ------------------------------------------------------------------ CLI

def ask(payload: dict) -> dict:
    """A person answers at the terminal. Input typed or pasted ahead of the prompt is dropped."""
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass
    import getpass
    who = getpass.getuser()
    if payload.get("kind") == "exceptions":          # the planner's ask_decision prints its own plan
        print("\n" + payload["markdown"])
        while True:
            a = input("\nproceed <reason> / abort <reason>: ").strip()
            if a.startswith("proceed "):
                if len(a[8:].strip()) >= 8:
                    return {"action": "proceed", "reason": a[8:].strip(), "by": who}
                print("give a reason (8+ characters) for accepting the exceptions")
            elif a.startswith("abort"):
                return {"action": "abort", "reason": a[5:].strip() or "aborted by operator", "by": who}
    from agents.test_planner.graph import ask_decision
    return {**ask_decision(payload), "by": who}


def parse_expect(items: list) -> dict:
    out = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep or not k.strip() or not v.strip():
            raise SystemExit(f"--expect needs KEY=VALUE, got {it!r}")
        out[k.strip()] = v.strip()
    return out


def list_runs(home: Path = RUNS):
    for d in sorted(home.glob("*")):
        f = d / "run.json"
        r = json.loads(f.read_text()) if f.exists() else {}
        human = len(Audit(d.name, home).human_events())
        print(f"{d.name}  {r.get('phase', 'IN PROGRESS'):10} node={r.get('node', '?')}  human events={human}  "
              f"{(r.get('requirements') or '')[:60]}")


async def main():
    ap = argparse.ArgumentParser(description="Phase 6 validation orchestrator")
    ap.add_argument("requirements", nargs="*")
    ap.add_argument("--node", default="sim")
    ap.add_argument("--expect", action="append", default=[], help='expected version, e.g. "System BIOS=V1.04"')
    ap.add_argument("--resume", help="continue a run waiting for a decision")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-research", action="store_true", help="triage without the web research step")
    args = ap.parse_args()
    if args.list:
        return list_runs()
    if not args.resume and not args.requirements:
        ap.error("give requirements (or --resume RUN_ID / --list)")

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from core.models import get_model
    from agents.test_planner.graph import build_graph as planner_graph, mcp_tools, recipe_store
    from agents.triage import graph as triage

    async with AsyncExitStack() as stack:
        (ROOT / "state").mkdir(exist_ok=True)
        saver = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(ROOT / "state/orchestrator.db")))
        tools = await mcp_tools(stack)
        store = recipe_store()
        web = None if args.no_research else await triage.production_research(stack)

        async def triage_run(archive):
            from agents.triage.episodes import EpisodeStore, _memory
            tg = triage.build_graph(get_model("planner"), bug_search=triage.production_bug_search(),
                                    episodes=EpisodeStore(memory=_memory()), research=web)
            out = await tg.ainvoke({"archive": archive, "exclude_bugs": []})
            out["_saved"] = str(triage.save(out))
            return out

        graph = build_graph(lambda prepare: planner_graph(tools, get_model("planner"), get_model("executor"), None,
                                                          recipes=store, prepare=prepare),
                            triage_run=triage_run, checkpointer=saver)
        run_id = args.resume or f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        cfg = {"configurable": {"thread_id": run_id}, "recursion_limit": 200}
        try:
            from langfuse.langchain import CallbackHandler
            cfg.update(callbacks=[CallbackHandler()], run_name="orchestrator",
                       metadata={"langfuse_session_id": run_id})
        except Exception:
            pass
        audit = Audit(run_id)
        if args.resume:
            snap = await graph.aget_state(cfg, subgraphs=True)
            pending = [i for t in snap.tasks for i in (t.interrupts or [])] or list(snap.interrupts or [])
            if not pending:
                raise SystemExit(f"run {run_id} is not waiting for a decision")
            inp = Command(resume=_decide(pending[0].value, audit))
        else:
            inp = {"run_id": run_id, "requirements": " ".join(args.requirements), "node": args.node,
                   "expect": parse_expect(args.expect)}
        print(f"run {run_id} · node {args.node} · audit {audit.path}")
        while True:
            out = await graph.ainvoke(inp, cfg)
            if "__interrupt__" in out:
                inp = Command(resume=_decide(out["__interrupt__"][0].value, audit))
                continue
            print("\n" + (out.get("run_report") or ""))
            print(f"\nphase={out.get('phase')} · report {audit.dir / 'report.md'} · "
                  f"human events {len(audit.human_events())}")
            break
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception:
        pass


def _decide(payload: dict, audit: Audit) -> dict:
    d = ask(payload)
    if payload.get("kind") != "exceptions":           # plan approval; exception decisions are logged in the graph
        audit.log("plan_decision", action=d.get("action"), reason=d.get("reason", ""),
                  recipes=d.get("recipes"), by=d.get("by", ""))
    return d


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    asyncio.run(main())
