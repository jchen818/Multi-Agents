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


def test_summarize_patterns_collapses_repeats(archive):
    from tools.triage.evidence import summarize_patterns, Evidence
    ev = Evidence(root=archive, summary={}, docs={"logs/x.log": [
        "amdgpu 0000:05:00.0: ring gfx_0.0.0 timeout, signaled seq=101, emitted seq=103",
        "amdgpu 0000:15:00.0: ring gfx_0.0.0 timeout, signaled seq=7, emitted seq=9",
        "XGMI link 3 down"]})
    (n, shape, where), _ = summarize_patterns(ev)
    assert n == 2 and "ring gfx_0.<N>.<N> timeout" in shape and where == "logs/x.log:1"


# ------------------------------------------------------------------ read-only Bugzilla export

class FakeResp:
    def __init__(self, data):
        self.data, self.status_code = data, 200

    def json(self):
        return self.data


class FakeSession:
    """Answers the handful of GET endpoints the exporter uses; records every call."""

    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        path = url.split("/rest/", 1)[1]
        if path == "field/bug":
            return FakeResp({"fields": [{"name": "cf_customer", "display_name": "Customer", "is_custom": True},
                                        {"name": "cf_code_name", "display_name": "Code", "is_custom": True},
                                        {"name": "product", "display_name": "Product"}]})
        if path == "bug":
            off = int(params.get("offset", 0))
            return FakeResp({"bugs": [] if off else [
                {"id": 7, "summary": "MI325X GPU-05 HBM miscompare in AGFHC hbm_s16_ds", "product": "G8825Z5U2BC",
                 "status": "RESOLVED", "resolution": "FIXED", "cf_customer": "ACME", "cf_code_name": "A1"},
                {"id": 8, "summary": "BMC web UI slow", "product": "M2810Z5", "status": "NEW", "resolution": "",
                 "cf_customer": "---", "cf_code_name": "---"}]})
        m = __import__("re").match(r"bug/(\d+)/(comment|attachment)$", path)
        if m and m.group(2) == "comment" and m.group(1) == "8":
            return FakeResp({"bugs": {"8": {"comments": [{"count": 0, "creator": "c@x", "time": "2026-10-05",
                                                          "text": "Page load takes 20 s"}]}}})
        if m and m.group(2) == "comment":
            return FakeResp({"bugs": {m.group(1): {"comments": [
                {"count": 0, "creator": "a@x", "time": "2026-10-01", "text": "hbm_s16_ds fails on GPU-05"},
                {"count": 1, "creator": "b@x", "time": "2026-10-03", "text": "Root cause: bad HBM stack, replaced OAM"}]}}})
        if m:
            return FakeResp({"bugs": {m.group(1): [
                {"id": 70, "file_name": "dmesg.log", "content_type": "text/plain", "size": 30},
                {"id": 71, "file_name": "sos.tar.xz", "content_type": "application/x-xz", "size": 10}]} if m.group(1) == "7" else {}})
        if path == "bug/attachment/70":
            return FakeResp({"attachments": {"70": {"data": __import__("base64").b64encode(b"amdgpu: RAS error").decode()}}})
        return FakeResp({"error": True, "code": 404, "message": f"unexpected {path}"})


def test_bz_export_is_read_only_and_complete(tmp_path):
    from tools.triage.bz_export import BZ, export, stats
    sess = FakeSession()
    meta = export(BZ("http://bz", "KEY", session=sess), out=tmp_path, log=lambda *_: None)
    assert meta["bugs"] == 2 and meta["text_attachments_saved"] == 1
    lines = (tmp_path / "bugs.jsonl").read_text().splitlines()
    b7 = __import__("json").loads(lines[0])
    assert b7["custom"] == {"Customer": "ACME", "Code": "A1"} and "cf_customer" not in b7
    assert b7["comments"][1]["text"].startswith("Root cause")
    assert (tmp_path / b7["attachments"][0]["saved"]).read_bytes() == b"amdgpu: RAS error"
    assert b7["attachments"][1]["saved"] == ""                 # binary archive not downloaded
    assert all(p["api_key"] == "KEY" for _, p in sess.calls)  # key in the query string
    s = stats(tmp_path)
    assert "2 bugs, 3 comments" in s and "GPU-related" in s and "#7" in s and "#8" not in s.split("GPU-related")[1]


def test_nested_results_json_and_summary_fallback(tmp_path):
    import json as _j
    from tools.triage.evidence import load_archive
    d = tmp_path / "AGFHC_X"
    (d / "logs" / "agfhc_logs" / "agfhc_1").mkdir(parents=True)
    nested = {"summary": {"total_selected_tests": 6, "total_failed": 1},
              "suites": [{"name": "all_lvl1", "instances": {"items": [
                  {"test": "pcie_link_status", "run": 1, "iteration": 1, "state": "passed"},
                  {"test": "hbm_s16_ds", "run": 1, "iteration": 1, "state": "failed",
                   "primary_status_name": "TEST_HBM_STRESS_FAIL", "primary_action_subject": "GPU-05",
                   "extended_results": [{"test": ["FATL| ABORT: R1 GPU-00 failed to allocate buffer."]}]}]}}]}
    (d / "logs" / "agfhc_logs" / "agfhc_1" / "results.json").write_text(_j.dumps(nested))
    (d / "summary.json").write_text(_j.dumps({"verdict": "FAIL", "failed": ["hbm_s16_ds-1-1"]}))
    (f,) = load_archive(d).failures
    assert f == {"test": "hbm_s16_ds-1-1", "status": "TEST_HBM_STRESS_FAIL", "subject": "GPU-05",
                 "lines": ["FATL| ABORT: R1 GPU-00 failed to allocate buffer."]}
    (d / "logs" / "agfhc_logs" / "agfhc_1" / "results.json").unlink()      # no results.json: use summary.json
    (f,) = load_archive(d).failures
    assert f["test"] == "hbm_s16_ds-1-1"
