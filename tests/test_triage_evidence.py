"""Triage evidence loader and citation check, against a real simulated FAIL archive."""
import asyncio
import importlib.util
import re
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "agfhc"))
from tools.triage.evidence import clean_lines, evidence_markdown, grep, load_archive, verify_citations  # noqa: E402


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tri")
    mp = pytest.MonkeyPatch()
    for k, v in dict(MULTI_AGENTS_ROOT=str(REPO), AGFHC_MODE="dryrun", AGFHC_STATE_DIR=str(tmp),
                     AGENTS_KILL_FILE=str(tmp / "KILL"), AGFHC_TMUX="0", SIM_SLEEP="0", SIM_SCENARIO="fail").items():
        mp.setenv(k, v)
    spec = importlib.util.spec_from_file_location("agfhc_server_tri", REPO / "tools/agfhc/server.py")
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    plan = asyncio.run(s.plan_run("sim", "all_lvl1"))
    started = asyncio.run(s.start_run(re.search(r"confirm_token: `([^`]+)`", plan).group(1)))
    rid = re.search(r"run_id=`([^`]+)`", started).group(1)
    for _ in range(30):
        if asyncio.run(s.get_run_status("sim", rid)).startswith("state=FINISHED"):
            break
        time.sleep(0.5)
    asyncio.run(s.analyze_run("sim", rid))
    mp.undo()
    (d,) = (tmp / "sim" / "archive").glob("AGFHC_*")
    return d


def test_load_archive(archive):
    ev = load_archive(archive)
    assert ev.verdict == "FAIL"
    assert any(n.endswith("console.log") for n in ev.docs) and any(n.endswith("results.json") for n in ev.docs)
    assert not any(n.endswith("run.sh") for n in ev.docs)
    (f,) = ev.failures
    assert f["test"] == "hbm_s16_ds-1-1" and f["status"] == "TEST_HBM_STRESS_FAIL" and f["subject"] == "GPU-05"
    assert any("miscompare" in l for l in f["lines"])


def test_grep_and_markdown(archive):
    ev = load_archive(archive)
    hits = grep(ev, r"TEST_HBM_STRESS_FAIL")
    assert hits and all(isinstance(i, int) for _, i, _ in hits)
    md = evidence_markdown(ev)
    assert "verdict **FAIL**" in md and "hbm_s16_ds-1-1" in md and re.search(r"\[logs/[^\]]+:\d+\]", md)


def test_citations_must_be_verbatim(archive):
    ev = load_archive(archive)
    r = verify_citations(ev, [
        {"text": "FAIL: hbm_s16_ds-1-1", "file": "console.log"},          # real, short whitespace differences
        {"text": "pattern miscompare at 0x1f000"},                         # real, any file
        {"text": "GPU-05 XGMI link down at 0x1f000"},                      # invented
        {"text": "FAIL"},                                                  # too short
        {"text": "pattern miscompare at 0x1f000", "file": "dmesg_new.log"},  # real text, wrong file
    ])
    assert [x["ok"] for x in r] == [True, True, False, False, False]
    assert r[0]["where"].startswith("logs/console.log:")
    assert "too short" in r[3]["why"]


def test_clean_lines_handles_ansi_and_redraws():
    assert clean_lines("\x1b[32mPASS\x1b[0m: a\r\nEXEC 1\rEXEC 2\rFAIL: b  \n\n") == ["PASS: a", "EXEC 1", "EXEC 2", "FAIL: b"]
