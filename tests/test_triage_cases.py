"""Every triage eval case builds into a loadable archive whose key line exists verbatim (so it is solvable)."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from agents.triage.graph import CATEGORIES  # noqa: E402
from tools.triage.evidence import evidence_markdown, load_archive, verify_citations  # noqa: E402
from tools.triage.seed_cases import build, load  # noqa: E402

DEFAULTS, CASES = load()


def test_case_ids_unique_and_categories_valid():
    assert len({c["id"] for c in CASES}) == len(CASES) >= 14
    assert {c["category"] for c in CASES} == set(CATEGORIES)          # every category has a case
    assert all(c["kind"] in ("bug", "seeded", "real") for c in CASES)
    assert all(c.get("exclude_bugs") for c in CASES if c["kind"] == "bug")


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_case_builds_and_key_line_is_citable(case, tmp_path):
    ev = load_archive(build(case, DEFAULTS, tmp_path))
    assert ev.verdict == "FAIL" and [f["test"] for f in ev.failures] == [f"{x['test']}-1-1" for x in case["failed"]]
    (r,) = verify_citations(ev, [{"text": case["key"]}])
    assert r["ok"], f"key line not in archive: {case['key']}"
    md = evidence_markdown(ev)
    assert "GPU memory" in md and case["failed"][0]["status"] in md
    if case.get("gpu_usage") == "busy":
        assert "python3 (vllm serve)" in md
