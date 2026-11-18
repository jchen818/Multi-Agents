"""Validation Orchestrator (Phase 6): one validation request across one or more nodes, from requirements to report.

    REQUESTED -> PLANNING -> AWAITING_APPROVAL -> PREPARING -> EXECUTING (all nodes in parallel) -> REPORTING -> DONE
                                                                       |
                                                                    FAILED -> ANALYZING -> REPORTING

PLANNING / AWAITING_APPROVAL  the Phase 2 planner plans once, on the first node, and stops after approval. The
                              same approved plan runs on every node (each node still checks it fits its template).
PREPARING                     preflight on every node; a node that cannot run AGFHC or has busy GPUs stops the run
                              (no decision can accept those); readiness exceptions on any node (recent GPU kernel
                              errors, unreadable VBIOS/BIOS/ROCm, a version different from --expect) go to ONE
                              decision for all nodes.
EXECUTING                     one planner run per node, started together (LangGraph Send) with the pre-approved
                              plan; the run waits for every node (barrier) before going on.
ANALYZING                     Phase 3 triage for every failed step on every node; bug drafts are never filed here.
HAND-OFF                      a likely cause the gpu-validation suite could catch before AGFHC runs (node state,
                              firmware, XGMI, hardware, thermal, driver; score >= 0.5) becomes a coder task: a task
                              file, and with --bus a message to the coder worker, which prepares the change and
                              stops before pushing. AGFHC tool/config defects stay with the bug draft.
REPORTING                     state/runs/<run_id>/report.md + run.json; the recipe is saved once for the run.

Two human decision points, both interrupts, both in the audit log: plan approval, readiness exceptions.
Expected versions come from the run request (--expect), never from code: they change with every BKC.

    .venv/bin/python -m agents.orchestrator.graph --node sim "quick health check"
    SIM_NODES=sim2,sim3,sim4 .venv/bin/python -m agents.orchestrator.graph --node sim --node sim2 --node sim3 \\
        --node sim4 "quick health check"                          # four simulated nodes in dry-run
    .venv/bin/python -m agents.orchestrator.graph --bus --node sim "..."   # triage via the Redis worker (6.2)
    .venv/bin/python -m agents.orchestrator.graph --resume <run_id>
    .venv/bin/python -m agents.orchestrator.graph --list
"""
import argparse
import asyncio
import json
import operator
import os
import re
import sys
import time
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send, interrupt

from core.guard import killed

ROOT = Path(os.environ.get("MULTI_AGENTS_ROOT") or Path(__file__).resolve().parents[2])
RUNS = ROOT / "state" / "runs"
HUMAN_EVENTS = ("plan_decision", "exception_decision")
TASKS = ROOT / "state" / "coder" / "tasks"
# Causes the gpu-validation suite could catch before AGFHC runs (read-only checks): these become coder tasks.
# AGFHC tool/config defects are not the suite's business: they stay with the bug draft for a person.
DETECTABLE = {"environment/node state", "firmware/BKC", "XGMI link", "hardware", "thermal/power", "driver/ROCm"}
HANDOFF_MIN_SCORE = 0.5


class State(TypedDict, total=False):
    # orchestrator
    run_id: str
    nodes: list
    phase: str              # REQUESTED, PLANNING, PREPARING, EXECUTING, ANALYZING, DONE, FAILED, REJECTED, STOPPED
    expect: dict            # versions the request expects, e.g. {"System BIOS": "V1.04"}
    readiness: dict         # node -> {"runnable": bool, "why": str, "exceptions": [...], "versions": {...}}
    exception_decision: dict
    node_results: Annotated[list, operator.add]   # one entry per node, filled in parallel
    triage: list            # [{node, recipe, run_id, archive, top, draft_summary, saved}]
    handoffs: list          # [{category, cause, node, recipe, task_file, sent}] - coder tasks from triage
    run_report: str
    recipe_saved: str
    # planner keys (the planner runs as a subgraph on this state for PLANNING, on the first node)
    requirements: str
    node: str
    platform: str
    preflight: str
    plan: dict
    status: str
    report: str
    dryrun: bool
    versions: dict
    recipes: list
    stop_after_approval: bool


class NodeRun(TypedDict, total=False):
    """What one parallel branch gets."""
    run_id: str
    node: str
    requirements: str
    plan: dict


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
    """What a person may accept before tests start on this node (busy GPUs are not here: can_run blocks them).
    Read from the preflight; nothing pinned."""
    out = []
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
    ev = [str(x.get("text", ""))[:200] for x in (c.get("support_v") or [])[:2]]
    return {"category": h.get("category"), "cause": h.get("cause"), "score": round(float(c.get("score") or 0), 2),
            "evidence": ev}




def can_run(preflight: str) -> tuple:
    """(True, '') if the node can start AGFHC now, else (False, reason). Busy GPUs block too: start_run refuses
    them at start (another workload makes the results false), so no human decision can accept them."""
    if (preflight or "").startswith(("REFUSED", "ERROR")):
        return False, preflight.strip()[:300]
    m = re.search(r"\*\*Can run AGFHC:\*\*\s*(.+)", preflight or "")
    if not m:
        return False, "preflight did not say whether AGFHC can run"
    if not m.group(1).strip().startswith("yes"):
        return False, m.group(1).strip()
    u = re.search(r"\*\*GPU usage:\*\*\s*(.+)", preflight)
    usage = u.group(1).strip() if u else "unknown"
    if not (usage.startswith("all ") and "idle" in usage):
        return False, f"GPUs not idle: {usage}"
    return True, ""


def coder_task(item: dict, cause: dict) -> str:
    ev = "\n".join(f"  - {e}" for e in cause.get("evidence") or []) or "  - (see the triage report)"
    return (f"Triage of AGFHC run {item['run_id']} on {item['node']} ({item['recipe']}) found a likely cause: "
            f"[{cause['category']}] {cause['cause']}\nEvidence from the run's logs:\n{ev}\n\n"
            "Add a read-only check to the gpu-validation suite that detects this condition on a node before stress "
            "tests run, so it is caught early. Follow AGENTS.md: catalog commands only, no pinned versions, no node "
            "names, a catch test that fails on tampered data. If the data the check needs is not in the recording, "
            "add the read-only catalog command and skip until a re-capture. If an existing test already detects "
            "this, change nothing and name that test.")


def handoff_items(triage: list) -> list:
    """One coder task per detectable cause category in this run: the best-scored failure for it."""
    best = {}
    for it in triage or []:
        if it.get("error") or not it.get("top"):
            continue
        c = it["top"][0]
        if c.get("category") in DETECTABLE and float(c.get("score") or 0) >= HANDOFF_MIN_SCORE:
            if c["category"] not in best or c["score"] > best[c["category"]][1]["score"]:
                best[c["category"]] = (it, c)
    return [best[k] for k in sorted(best)]


def run_markdown(state: dict) -> str:
    nodes = state.get("nodes") or []
    lines = [f"# Validation run {state['run_id']}", "",
             f"- Nodes: {', '.join(nodes)} ({'dry-run' if state.get('dryrun', True) else 'live'})",
             f"- Requirements: {state.get('requirements')}",
             f"- Outcome: **{state.get('phase')}**"]
    if state.get("plan", {}).get("steps"):
        lines.append(f"- Plan ({state['plan'].get('tier')}), approved once for all nodes: "
                     + ", ".join(s["recipe"] for s in state["plan"]["steps"]))
    exc = {n: r["exceptions"] for n, r in (state.get("readiness") or {}).items() if r.get("exceptions")}
    blocked = {n: r["why"] for n, r in (state.get("readiness") or {}).items() if not r.get("runnable", True)}
    if blocked:
        lines.append("- Nodes that cannot run AGFHC: " + "; ".join(f"{n}: {w}" for n, w in blocked.items()))
    if exc:
        d = state.get("exception_decision") or {}
        lines.append(f"- Readiness exceptions ({d.get('action', 'none')}: {d.get('reason', '')}):")
        lines += [f"  - {n}: " + "; ".join(e.splitlines()[0] for e in es) for n, es in exc.items()]
    if state.get("recipe_saved"):
        lines.append(f"- Recipe: `{state['recipe_saved']}`")
    if state.get("node_results"):
        lines += ["", "## Execution", "", "| Node | Status | Steps |", "|---|---|---|"]
        for r in sorted(state["node_results"], key=lambda r: r["node"]):
            steps = ", ".join(f"{x['recipe']} {x.get('verdict') or 'not run'}" for x in r.get("results") or [])
            lines.append(f"| {r['node']} | {r.get('status')} | {steps or '-'} |")
        for r in sorted(state["node_results"], key=lambda r: r["node"]):
            lines += ["", f"### {r['node']}", r.get("report") or "(no report)"]
    elif state.get("report"):
        lines += ["", "## Planning", state["report"]]
    if state.get("triage"):
        lines += ["", "## Failure analysis"]
        for t in state["triage"]:
            lines.append(f"### {t['node']}: `{t['recipe']}` run `{t['run_id']}`")
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
    if state.get("handoffs"):
        lines += ["", "## Suggested suite checks (coding agent)"]
        for h in state["handoffs"]:
            how = ("queued for the coder worker; when it is ready, `python -m agents.coder.graph --list` shows it "
                   "and `--push <branch>` sends it for review" if h.get("sent") else
                   f"run `python -m agents.coder.graph --task-file {h['task_file']}`")
            lines.append(f"- [{h['category']}] {h['cause']} (from {h['node']} `{h['recipe']}`): {how}")
    return "\n".join(lines)


# ------------------------------------------------------------------ graph

def build_graph(planner_factory, preflight, triage_run=None, recipes=None, checkpointer=None, home: Path = RUNS,
                bus=None, triage_timeout_s: float = 3600, tasks_dir: Path = TASKS):
    """planner_factory() -> compiled Phase 2 planner graph without a checkpointer (used as the PLANNING subgraph and,
    with a pre-approved plan, once per node). preflight(node) -> awaitable preflight text (the agfhc tool).
    triage_run(archive) -> awaitable triage state, or None to skip analysis. recipes: RecipeStore to save the plan.
    bus: core.bus.Bus("orchestrator"): failed steps go to the triage worker over Redis instead of triage_run."""
    planning = planner_factory()
    node_planner = planner_factory()

    def audit_for(state):
        return Audit(state["run_id"], home)

    def requested(state: State):
        audit = audit_for(state)
        nodes = list(dict.fromkeys(state.get("nodes") or [state.get("node") or "sim"]))
        if killed():
            audit.log("stopped", where="REQUESTED")
            return {"phase": "STOPPED", "nodes": nodes}
        audit.log("state", phase="REQUESTED", nodes=nodes, requirements=state.get("requirements"),
                  expect=state.get("expect") or {})
        audit.log("state", phase="PLANNING", lead_node=nodes[0])
        return {"phase": "PLANNING", "nodes": nodes, "node": nodes[0], "stop_after_approval": True, "triage": []}

    def after_planning(state: State):
        st = state.get("status")
        audit_for(state).log("state", phase="PLANNED", planner_status=st)
        if st == "APPROVED":
            return Command(goto="prepare", update={"phase": "PREPARING"})
        phase = {"REJECTED": "REJECTED", "REFUSED": "REJECTED"}.get(st, "FAILED")
        return Command(goto="report", update={"phase": phase})

    async def prepare(state: State):
        audit = audit_for(state)
        if killed():
            audit.log("stopped", where="PREPARING")
            return Command(goto="report", update={"phase": "STOPPED"})
        from agents.test_planner.graph import parse_versions
        ready = {}
        for n in state["nodes"]:               # read-only; re-read when a run resumes after the decision
            pf = await preflight(n)
            ok, why = can_run(pf)
            v = parse_versions(pf)
            ready[n] = {"runnable": ok, "why": why, "versions": v,
                        "exceptions": readiness_exceptions(pf, v, state.get("expect") or {}) if ok else []}
        audit.log("state", phase="PREPARING",
                  readiness={n: {k: r[k] for k in ("runnable", "why", "exceptions")} for n, r in ready.items()})
        blocked = {n: r["why"] for n, r in ready.items() if not r["runnable"]}
        if blocked:
            return Command(goto="report", update={"readiness": ready, "phase": "REJECTED",
                                                  "report": "not started: " + "; ".join(f"{n} cannot run AGFHC ({w})"
                                                                                         for n, w in blocked.items())})
        exc = {n: r["exceptions"] for n, r in ready.items() if r["exceptions"]}
        if exc:
            md = "### Readiness exceptions\n" + "\n".join(
                f"- **{n}**: " + "; ".join(e.replace("\n", " / ") for e in es) for n, es in exc.items())
            decision = interrupt({"kind": "exceptions", "nodes": list(exc), "exceptions": exc, "markdown": md}) or {}
            audit.log("exception_decision", action=decision.get("action"), reason=decision.get("reason", ""),
                      exceptions=exc, by=decision.get("by", ""))
            if not (decision.get("action") == "proceed" and len(decision.get("reason", "").strip()) >= 8):
                return Command(goto="report", update={
                    "readiness": ready, "exception_decision": decision, "phase": "REJECTED",
                    "report": f"not started: readiness exceptions not accepted ({decision.get('reason') or 'aborted'})"})
            update = {"readiness": ready, "exception_decision": decision}
        else:
            update = {"readiness": ready}
        audit.log("state", phase="EXECUTING", nodes=state["nodes"])
        return Command(goto="fan_out", update={**update, "phase": "EXECUTING"})

    def fan_out(state: State):
        return {}

    def branches(state: State):
        return [Send("node_run", {"run_id": state["run_id"], "node": n, "requirements": state["requirements"],
                                  "plan": state["plan"]}) for n in state["nodes"]]

    async def node_run(b: NodeRun):
        audit = Audit(b["run_id"], home)
        if killed():
            audit.log("stopped", where="EXECUTING", node=b["node"])
            return {"node_results": [{"node": b["node"], "status": "STOPPED", "results": [], "report": "kill switch"}]}
        out = await node_planner.ainvoke({"requirements": b["requirements"], "node": b["node"],
                                          "approved_plan": b["plan"], "run_id": b["run_id"]},
                                         {"recursion_limit": 200})
        res = {"node": b["node"], "status": out.get("status"), "report": out.get("report", ""),
               "versions": out.get("versions") or {}, "dryrun": out.get("dryrun", True),
               "results": [{k: r.get(k) for k in ("recipe", "run_id", "verdict", "analysis", "error")}
                           for r in out.get("results") or []]}
        audit.log("node_done", node=b["node"], status=res["status"],
                  results=[{k: r.get(k) for k in ("recipe", "run_id", "verdict")} for r in res["results"]])
        return {"node_results": [res]}

    def collect(state: State):
        """Barrier: runs once every node branch is done."""
        audit = audit_for(state)
        rs = state.get("node_results") or []
        saved = ""
        full = [r for r in rs if r["results"] and len(r["results"]) == len(state["plan"]["steps"])
                and all(x.get("verdict") in ("PASS", "FAIL") for x in r["results"])]
        if recipes is not None and full:
            r = full[0]
            v = r.get("versions") or {}
            rec = recipes.save(platform=state.get("platform") or "mi325x", bkc=str(v.get("System BIOS") or "unknown"),
                               requirements=state["requirements"], plan=state["plan"], results=r["results"],
                               versions=v, node=",".join(sorted(x["node"] for x in rs)), dryrun=r.get("dryrun", True),
                               based_on=next((x for x in state.get("recipes") or []
                                              if x["id"] == state["plan"].get("based_on")), None))
            saved = rec["path"]
        all_done = len(rs) == len(state["nodes"]) and all(r["status"] == "DONE" for r in rs)
        failed = [r for r in rs if any(x.get("verdict") == "FAIL" for x in r["results"])]
        audit.log("state", phase="COLLECTED", done=all_done, failed_nodes=[r["node"] for r in failed])
        if failed:
            return Command(goto="analyze", update={"phase": "ANALYZING", "recipe_saved": saved})
        return Command(goto="report", update={"phase": "DONE" if all_done else "FAILED", "recipe_saved": saved,
                                              "dryrun": all(r.get("dryrun", True) for r in rs)})

    async def triage_via_bus(state, items):
        """Send one triage_request per failed step to the triage worker(s), then wait for every reply. A reply
        already on the stream (the run resumed) is reused instead of asking again."""
        sent = {}
        for it in items:
            key = f"{state['run_id']}/{it['node']}/{it['run_id']}"
            got = await bus.find_reply(key)
            if got is None:
                since = (await bus.r.xrevrange("agents:orchestrator:in", count=1) or [("0-0", {})])[0][0]
                try:
                    await bus.send("triage", "triage_request", state["run_id"], artifact_path=it["archive"],
                                   note=f"{it['node']} {it['recipe']}", reply_to="agents:orchestrator:in", key=key)
                except Exception as e:                  # e.g. the archive is outside the bus's artifact roots
                    it["_reply"] = {"type": "triage_failed", "note": f"not sent: {e}"}
                    it["_key"] = key
                    continue
                sent[key] = since
            else:
                it["_reply"] = got
            it["_key"] = key
        for it in items:
            if "_reply" not in it:
                it["_reply"] = await bus.wait_reply(it["_key"], sent[it["_key"]], triage_timeout_s)
            rep = it.pop("_reply")
            it.pop("_key")
            if rep is None:
                it["error"] = f"no triage reply within {int(triage_timeout_s)} s (is the triage worker running?)"
            elif rep.get("type") == "triage_failed":
                it["error"] = f"triage worker: {rep.get('note', '')[:300]}"
            else:
                note = json.loads(rep.get("note") or "{}")
                it.update(top=note.get("top") or [], draft_summary=note.get("draft_summary", ""),
                          saved=rep.get("artifact_path", ""))

    async def analyze(state: State):
        audit = audit_for(state)
        items = []
        for nr in sorted(state.get("node_results") or [], key=lambda r: r["node"]):
            for r in nr["results"]:
                if r.get("verdict") != "FAIL":
                    continue
                arch = archive_of(r.get("analysis", ""))
                item = {"node": nr["node"], "recipe": r["recipe"], "run_id": r.get("run_id"), "archive": arch,
                        "top": []}
                if not arch or not Path(arch).is_dir():
                    item["error"] = f"archive not found on this host: {arch or 'no archive line'}"
                elif bus is None and triage_run is None:
                    item["error"] = "triage disabled"
                items.append(item)
        todo = [it for it in items if "error" not in it]
        if killed():
            audit.log("stopped", where="ANALYZING")
            for it in todo:
                it["error"] = "stopped by the kill switch"
        elif bus is not None:
            await triage_via_bus(state, todo)
        else:
            for it in todo:
                if killed():
                    it["error"] = "stopped by the kill switch"
                    continue
                try:
                    t = await triage_run(it["archive"])
                    it["top"] = [top_cause(c) for c in (t.get("ranked") or [])[:3]]
                    it["draft_summary"] = (t.get("draft") or {}).get("summary", "")
                    it["saved"] = t.get("_saved", "")
                except Exception as e:
                    it["error"] = f"{type(e).__name__}: {e}"[:300]
        for it in items:
            audit.log("triage", node=it["node"], recipe=it["recipe"], run_id=it["run_id"], top=it["top"],
                      error=it.get("error"), via="bus" if bus is not None else "in-process")
        return {"triage": items, "phase": "FAILED",
                "dryrun": all(r.get("dryrun", True) for r in state.get("node_results") or [])}

    async def handoff(state: State):
        """Detectable causes become coder tasks: a task file, and with the bus a coder_task message. The coder
        worker prepares the change and stops before pushing; a person reviews and pushes it."""
        audit = audit_for(state)
        out = []
        for k, (it, c) in enumerate(handoff_items(state.get("triage")), 1):
            tasks_dir.mkdir(parents=True, exist_ok=True)
            f = tasks_dir / f"{state['run_id']}-{k}.md"
            f.write_text(coder_task(it, c) + "\n")
            h = {"category": c["category"], "cause": c["cause"], "node": it["node"], "recipe": it["recipe"],
                 "task_file": str(f), "sent": False}
            if bus is not None and not killed():
                try:
                    await bus.send("coder", "coder_task", state["run_id"], artifact_path=str(f),
                                   note=f"{c['category']}: {c['cause']}"[:500], key=f"{state['run_id']}/coder/{k}")
                    h["sent"] = True
                except Exception as e:
                    h["error"] = f"not sent: {e}"[:300]
            audit.log("handoff", **h)
            out.append(h)
        return {"handoffs": out}

    def report(state: State):
        audit = audit_for(state)
        md = run_markdown(state)
        (audit.dir / "report.md").write_text(md)
        (audit.dir / "run.json").write_text(json.dumps(
            {k: state.get(k) for k in ("run_id", "phase", "nodes", "requirements", "expect", "plan", "readiness",
                                       "exception_decision", "triage", "handoffs", "recipe_saved", "dryrun")}
            | {"node_results": [{k: v for k, v in r.items() if k != "report"} for r in state.get("node_results") or []]},
            indent=1, default=str))
        audit.log("state", phase="REPORTING", outcome=state.get("phase"), report=str(audit.dir / "report.md"))
        return {"run_report": md}

    g = StateGraph(State)
    for name, fn in (("requested", requested), ("planning", planning), ("after_planning", after_planning),
                     ("prepare", prepare), ("fan_out", fan_out), ("node_run", node_run), ("collect", collect),
                     ("analyze", analyze), ("handoff", handoff), ("report", report)):
        g.add_node(name, fn)
    g.add_edge(START, "requested")
    g.add_conditional_edges("requested", lambda s: "report" if s.get("phase") == "STOPPED" else "planning",
                            ["report", "planning"])
    g.add_edge("planning", "after_planning")
    g.add_conditional_edges("fan_out", branches, ["node_run"])
    g.add_edge("node_run", "collect")
    g.add_edge("analyze", "handoff")
    g.add_edge("handoff", "report")
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
        print(f"{d.name}  {r.get('phase', 'IN PROGRESS'):10} nodes={','.join(r.get('nodes') or ['?'])}  "
              f"human events={human}  "
              f"{(r.get('requirements') or '')[:60]}")


async def main():
    ap = argparse.ArgumentParser(description="Phase 6 validation orchestrator")
    ap.add_argument("requirements", nargs="*")
    ap.add_argument("--node", action="append", default=[], help="node to run on; repeat for several (default sim)")
    ap.add_argument("--expect", action="append", default=[], help='expected version, e.g. "System BIOS=V1.04"')
    ap.add_argument("--resume", help="continue a run waiting for a decision")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-research", action="store_true", help="triage without the web research step")
    ap.add_argument("--bus", action="store_true", help="send failed steps to the triage worker over Redis "
                                                       "(REDIS_URL; start it with python -m agents.triage.worker)")
    args = ap.parse_args()
    if args.list:
        return list_runs()
    if not args.resume and not args.requirements:
        ap.error("give requirements (or --resume RUN_ID / --list)")
    nodes = [n.strip() for item in (args.node or ["sim"]) for n in item.split(",") if n.strip()]

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from core.models import get_model
    from agents.test_planner.graph import build_graph as planner_graph, mcp_tools, recipe_store
    from agents.triage import graph as triage

    async with AsyncExitStack() as stack:
        (ROOT / "state").mkdir(exist_ok=True)
        saver = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(ROOT / "state/orchestrator.db")))
        tools = await mcp_tools(stack)
        store = recipe_store()
        # with --bus the triage worker does the research; the orchestrator does not need the web tools
        web = None if (args.no_research or args.bus) else await triage.production_research(stack)

        async def triage_run(archive):
            from agents.triage.episodes import EpisodeStore, _memory
            tg = triage.build_graph(get_model("planner"), bug_search=triage.production_bug_search(),
                                    episodes=EpisodeStore(memory=_memory()), research=web)
            out = await tg.ainvoke({"archive": archive, "exclude_bugs": []})
            out["_saved"] = str(triage.save(out))
            return out

        async def preflight(node):
            from agents.test_planner.graph import _text
            return _text(await tools["preflight"](node=node))

        bus = None
        if args.bus:
            from core.bus import Bus, connect
            bus = Bus(connect(), "orchestrator")
        graph = build_graph(lambda: planner_graph(tools, get_model("planner"), get_model("executor"), None,
                                                  recipes=store),
                            preflight, triage_run=None if bus else triage_run, recipes=store, checkpointer=saver,
                            bus=bus)
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
            inp = {"run_id": run_id, "requirements": " ".join(args.requirements), "nodes": nodes,
                   "expect": parse_expect(args.expect)}
        print(f"run {run_id} · nodes {', '.join(nodes)} · audit {audit.path}")
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
