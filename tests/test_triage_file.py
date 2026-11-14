"""Triage draft -> Bugzilla: validated draft, then a human-confirmed create (ported from Bugzilla Agent 3.2.1)."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from agents.triage.file import Bugzilla, build_draft, create, match, save_draft  # noqa: E402


class R:
    def __init__(self, d):
        self.d, self.status_code = d, 200

    def json(self):
        return self.d


FIELDS = {"fields": [
    {"name": "bug_severity", "values": [{"name": "blocker"}, {"name": "major"}, {"name": "normal"}, {"name": "minor"}]},
    {"name": "rep_platform", "values": [{"name": "Server"}, {"name": "PC"}]},
    {"name": "op_sys", "values": [{"name": "All"}, {"name": "Linux"}]},
    {"name": "cf_customer", "display_name": "Customer", "is_custom": True, "is_on_bug_entry": True, "is_mandatory": True,
     "type": 2, "values": [{"name": "---"}, {"name": "Engineering team lab"}, {"name": "Roblox (Mateo)"}]},
    {"name": "cf_code_name", "display_name": "Code", "is_custom": True, "is_on_bug_entry": True, "is_mandatory": True,
     "type": 2, "values": [{"name": "8825"}, {"name": "C9i"}]}]}


class FakeBZ:
    def __init__(self):
        self.gets, self.posts = [], []

    def get(self, url, params=None, timeout=None):
        self.gets.append(url)
        path = url.split("/rest/", 1)[1]
        if path == "field/bug":
            return R(FIELDS)
        if path == "product" and (params or {}).get("type") == "enterable":
            return R({"products": [{"name": "G8825Z5U2BC"}, {"name": "B8056"}]})
        if path == "product":
            return R({"products": [{"components": [{"name": "General"}], "versions": [{"name": "unspecified"}]}]})
        return R({"error": True, "code": 404, "message": path})

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append((url.split("/rest/", 1)[1], json))
        return R({"id": 32} if url.endswith("/rest/bug") else {"ids": [70]})


TRIAGE = {"draft": {"summary": "hbm_s16_ds FAIL on GPU-05 (all_lvl1): buffer allocation failure",
                    "description": "Command: `/opt/amd/agfhc/agfhc -r all_lvl1`\nEvidence: failed to allocate buffer"}}
SUMMARY = {"host": "mitacai", "run_id": "20261015T203345Z-mitacai-all-lvl1-a6d7"}
CFG = {"product_by_host": {"mitacai": "G8825Z5U2BC"}, "defaults": {"hardware": "Server", "os": "All"}}


def bz(fake):
    return Bugzilla("http://bz", "RO-KEY", session=fake)


def test_match_is_the_agents_resolver():
    assert match("engineering", ["Engineering team lab", "Roblox (Mateo)"]) == ("Engineering team lab", [])
    assert match("8825", ["8825", "C9i"])[0] == "8825" and match("zzz", ["8825"])[0] is None


def test_not_ready_lists_missing_and_invalid_like_draft_bug():
    f = FakeBZ()
    d = build_draft(bz(f), TRIAGE, SUMMARY, CFG, fields={"Code": "9999"})
    assert not d["ready"]
    assert any(m.startswith("Severity (options: blocker, major") for m in d["missing"])
    assert any(m.startswith("Customer") for m in d["missing"])
    assert any("Code: '9999' is not allowed" in i for i in d["invalid"])
    assert not f.posts                                          # validation never writes


def test_ready_draft_then_create_only_with_every_guard(tmp_path):
    f = FakeBZ()
    d = build_draft(bz(f), TRIAGE, SUMMARY, CFG, severity="normal",
                    fields={"Customer": "engineering", "Code": "8825"}, attach_report="# Triage\n...")
    assert d["ready"], d
    b = d["body"]
    assert b["product"] == "G8825Z5U2BC" and b["component"] == "General" and b["version"] == "unspecified"
    assert b["cf_customer"] == "Engineering team lab" and b["platform"] == "Server" and b["op_sys"] == "All"
    assert "Drafted by the triage agent" in b["description"] and "Reviewed and filed" not in b["description"]
    assert d["critique"].startswith("not run")
    tok = save_draft(d, "AGFHC_20261015_133404", user="john", now=1000.0, drafts=tmp_path)
    assert oct((tmp_path / f"{tok}.json").stat().st_mode)[-3:] == "600"
    w = {"write_enabled": True, "filing_api_key": "JOHNS-KEY"}
    filing = Bugzilla("http://bz", "JOHNS-KEY", session=f)
    assert create(tok, filing, {}, tok, user="john", now=1001, drafts=tmp_path).startswith("BLOCKED: write_enabled")
    assert create(tok, filing, {"write_enabled": True}, tok, user="john", now=1001, drafts=tmp_path).startswith("BLOCKED: no filing_api_key")
    assert "different user" in create(tok, filing, w, tok, user="eve", now=1001, drafts=tmp_path)
    assert create(tok, filing, w, "yes", user="john", now=1001, drafts=tmp_path).startswith("NOT FILED")
    assert "expired" in create(tok, filing, w, tok, user="john", now=1000 + 16 * 60, drafts=tmp_path)
    assert not f.posts                                          # every refusal wrote nothing and kept the draft
    out = create(tok, filing, w, tok, user="john", now=1001, drafts=tmp_path)
    assert out.startswith("Filed Bug 32: http://bz/show_bug.cgi?id=32") and "Attached triage.md" in out
    assert [p for p, _ in f.posts] == ["bug", "bug/32/attachment"] and f.posts[0][1]["api_key"] == "JOHNS-KEY"
    assert f.posts[0][1]["description"].endswith("Reviewed and filed by john.")
    assert not (tmp_path / f"{tok}.json").exists()
    assert "not found" in create(tok, filing, w, tok, user="john", now=1002, drafts=tmp_path)   # single use


def test_no_draft_when_nothing_survived_triage():
    d = build_draft(bz(FakeBZ()), {"draft": {}}, SUMMARY, CFG, severity="normal",
                    fields={"Customer": "Engineering team lab", "Code": "8825"})
    assert not d["ready"] and any("nothing survived verification" in m for m in d["missing"])
