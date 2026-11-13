"""Triage graph end to end on a real simulated FAIL archive, with a fake model that answers by prompt content."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from agents.triage.graph import (BugDraft, Critique, Hypotheses, Hypothesis, Judge, Score, build_graph,  # noqa: E402
                                 save)
from tests.test_triage_evidence import archive  # noqa: E402,F401  (module-scoped sim FAIL archive)

REAL = "pattern miscompare at 0x1f000"
CONSOLE = "FAIL: hbm_s16_ds-1-1"


def H(cat, cause, rx):
    return Hypothesis(category=cat, cause=cause, grep=[rx], confirm_if="see logs")


class Fake:
    """Responds per schema; records every prompt."""

    def __init__(self, judge_rejects=("thermal",)):
        self.prompts, self.judge_rejects = [], judge_rejects

    def with_structured_output(self, schema, method=None):
        outer = self

        class R:
            async def ainvoke(self, msgs):
                u = msgs[-1][1]
                outer.prompts.append((schema.__name__, u))
                return outer.answer(schema, u)
        return R()

    def answer(self, schema, u):
        if schema is Hypotheses:
            if "Make this candidate more specific" in u:
                return Hypotheses(hypotheses=[H("hardware", "refined: " + u.split("):", 1)[1].split("\n", 1)[0].strip(), "GPU-05")])
            return Hypotheses(hypotheses=[
                H("hardware", "HBM stack fault on GPU-05", "miscompare"),
                H("test tool/config defect", "hbm_s16_ds reads before it writes, as in bug #22", "TEST_HBM_STRESS_FAIL"),
                H("thermal/power", "GPU-05 throttled under load", "throttl"),
                H("XGMI link", "XGMI link down during the run", "xgmi"),
                H("driver/ROCm", "amdgpu RAS reporting fault", "RAS")])
        if schema is Score:
            if "throttled" in u:     # cites a line that does not exist
                return Score(score=0.65, support=[{"file": "dmesg_new.log", "text": "amdgpu: GPU-05 thermal throttling engaged"}],
                             reasoning="r")
            if "XGMI link down" in u:
                return Score(score=0.2, support=[], reasoning="no evidence")
            if "HBM stack fault" in u or "refined" in u:
                return Score(score=0.8 if "refined" in u else 0.7, support=[{"file": "results.json", "text": REAL}],
                             against=[], bugs=[22], reasoning="miscompare on GPU-05")
            if "reads before it writes" in u:
                return Score(score=0.6, support=[{"file": "console.log", "text": CONSOLE}], bugs=[22], reasoning="like #22")
            return Score(score=0.1, support=[{"text": CONSOLE}], reasoning="weak")
        if schema is Judge:
            return Judge(supported=not any(w in u for w in self.judge_rejects), why="checked")
        if schema is BugDraft:
            return BugDraft(summary="hbm_s16_ds FAIL on GPU-05 (all_lvl1)", description="Evidence: " + REAL)
        if schema is Critique:
            return Critique(ok=True)
        raise AssertionError(schema)


def run(graph, archive_dir, **kw):
    return asyncio.run(graph.ainvoke(dict({"archive": str(archive_dir)}, **kw)))


def test_triage_ranks_verified_hypotheses_and_drafts_a_bug(archive, tmp_path):
    seen = []
    search = lambda q, k, excl, terms: seen.append((q, excl, terms)) or [
        {"bug_id": 22, "status": "RESOLVED", "resolution": "FIXED", "summary": "hbm_ds_ntd of all_lvl5 failed",
         "terms": ["hbm_ds_ntd"], "text": "HBM_MISCOMPARE read before written"}]
    m = Fake()
    out = run(build_graph(m, bug_search=search), archive, exclude_bugs=[99])
    ranked = out["ranked"]
    assert ranked and ranked[0]["hyp"]["cause"].startswith("refined") and ranked[0]["score"] == 0.8
    assert all(x["ok"] for c in ranked for x in c["support_v"])
    assert any(c["hyp"]["category"] == "test tool/config defect" for c in ranked)
    why = {c["hyp"]["cause"]: c["why"] for c in out["dropped"]}
    assert "not found verbatim" in why["GPU-05 throttled under load"]          # fabricated line dropped
    assert why["XGMI link down during the run"] == "no supporting log line"
    assert "Bug 22" in out["history"] and seen[0][1] == [99] and "hbm_s16_ds" in seen[0][2]
    rep = out["report"]
    assert "verdict FAIL" in rep and f"`{REAL}` [logs/agfhc_logs/" in rep and "#22" in rep and "Dropped candidates" in rep
    assert out["draft"]["summary"].startswith("hbm_s16_ds") and out["draft"]["critique"]["ok"]
    assert out["calls"] <= 25
    first = next(u for s, u in m.prompts if s == "Hypotheses")
    assert "test tool/config defect" in first and "<evidence>" in first and "Bug 22" in first
    d = save(out, tmp_path)
    assert (d / "triage.md").read_text() == rep and json.loads((d / "triage.json").read_text())["draft"]


def test_judge_can_reject_and_nothing_survives_without_evidence(archive):
    m = Fake(judge_rejects=("HBM", "refined", "reads before", "amdgpu", "FAIL:"))
    out = run(build_graph(m), archive)
    assert out["ranked"] == [] and out["draft"] == {}
    assert "No hypothesis survived verification" in out["report"]
    assert "(bug history unavailable)" in out["history"]


def test_llm_call_budget_is_enforced(archive):
    out = run(build_graph(Fake(), max_calls=12), archive)
    assert out["calls"] <= 12
