"""Promptfoo provider: run the real triage graph (gemma + agfhc bug history) on one eval case.

vars: case = id in cases.yaml. bug cases hold their own bug out of retrieval; episodes come from an empty temporary
store so no case can see a confirmed answer. Output JSON: expected category, ranked categories/causes, top1/top3.
"""
import asyncio
import json
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = "/home/mitac/Multi-Agents"
sys.path.insert(0, ROOT)

from langfuse.langchain import CallbackHandler

from agents.triage.episodes import EpisodeStore
from agents.triage.graph import build_graph, production_bug_search
from core.models import get_model
from tools.triage.seed_cases import build, load


def _run(case_id: str) -> dict:
    defaults, cases = load()
    case = next(c for c in cases if c["id"] == case_id)
    tmp = Path(tempfile.mkdtemp(prefix="triage-eval-"))
    archive = Path(case["archive"]) if case.get("kind") == "real" else build(case, defaults, tmp)
    graph = build_graph(get_model("planner"), bug_search=production_bug_search(),
                        episodes=EpisodeStore(tmp / "episodes"))
    sid = f"eval-triage-{case_id}-{uuid.uuid4().hex[:6]}"
    out = asyncio.run(graph.ainvoke({"archive": str(archive), "exclude_bugs": case.get("exclude_bugs") or []},
                                    {"callbacks": [CallbackHandler()], "run_name": "triage-eval",
                                     "metadata": {"langfuse_session_id": sid, "langfuse_tags": ["eval-triage"]}}))
    ranked = [{"category": c["hyp"]["category"], "cause": c["hyp"]["cause"], "score": c.get("score")}
              for c in out.get("ranked") or []]
    cats = [r["category"] for r in ranked]
    rank = cats.index(case["category"]) + 1 if case["category"] in cats else None
    return {"case": case_id, "kind": case["kind"], "expected": case["category"], "expected_cause": case["cause"],
            "ranked": ranked, "rank": rank, "top1": rank == 1, "top3": bool(rank and rank <= 3),
            "dropped": len(out.get("dropped") or []), "calls": out.get("calls"), "report": out.get("report", "")[:6000]}


def call_api(prompt, options, context):
    return {"output": json.dumps(_run(context["vars"]["case"]))}
