"""Triage research step: allowed domains only, verbatim quotes only, leads never change the ranking."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from agents.triage.graph import Finding, Findings, Queries, build_graph  # noqa: E402
from tests.test_triage_evidence import archive  # noqa: E402,F401
from tests.test_triage_graph import Fake  # noqa: E402
from tools.triage.research import allowed, verify_finding  # noqa: E402

NOTE = "Known issue: hbm_s16_ds may report TEST_HBM_STRESS_FAIL when GPU memory is already allocated by another process."
PAGES = {"https://rocm.docs.amd.com/en/latest/about/release-notes.html": f"<h1>ROCm release notes</h1>\n{NOTE}\nOther text.",
         "https://github.com/ROCm/agfhc/issues/42": "issue body unrelated to this run"}


def web():
    calls = {"search": [], "fetch": []}

    async def search(query):
        calls["search"].append(query)
        return ('<tool_output tool="search_web" seconds="0.4">\n1. ROCm release notes\n'
                'https://rocm.docs.amd.com/en/latest/about/release-notes.html\n2. forum https://forum.example.com/t/agfhc\n'
                '3. https://github.com/ROCm/agfhc/issues/42\n4. http://10.46.252.48/show_bug.cgi?id=22\n</tool_output>')

    async def fetch(url):
        calls["fetch"].append(url)
        return PAGES.get(url, "ERROR: not found")
    return SimpleNamespace(search=search, fetch=fetch), calls


class F(Fake):
    def answer(self, schema, u):
        if schema is Queries:
            return Queries(queries=["AGFHC hbm_s16_ds allocation failure known issue"])
        if schema is Findings:
            url = "https://rocm.docs.amd.com/en/latest/about/release-notes.html"
            return Findings(findings=[
                Finding(url=url, quote=NOTE, hypothesis=1, bearing="supports", note="matches the allocation failure"),
                Finding(url=url, quote="AGFHC 1.30.2 has a confirmed HBM firmware defect on MI325X", hypothesis=1,
                        bearing="supports"),                                        # invented
                Finding(url="https://forum.example.com/t/agfhc", quote=NOTE, hypothesis=1, bearing="supports")])
        return super().answer(schema, u)


def test_domain_allowlist():
    assert allowed("https://rocm.docs.amd.com/en/latest/x.html") and allowed("https://www.amd.com/en/support")
    assert allowed("https://github.com/ROCm/ROCm/releases") and not allowed("https://github.com/someone/fork")
    assert not allowed("http://rocm.docs.amd.com/x") and not allowed("https://forum.example.com/t/agfhc")
    assert not allowed("https://amd.com.evil.example/x")


def test_research_keeps_only_verbatim_quotes_from_allowed_pages(archive):
    w, calls = web()
    out = asyncio.run(build_graph(F(), research=w).ainvoke({"archive": str(archive)}))
    r = out["research"]
    assert r["ran"] and len(r["queries"]) == 3 and calls["search"] == r["queries"]
    assert set(calls["fetch"]) == set(PAGES)                         # forum and internal Bugzilla never fetched
    assert "https://forum.example.com/t/agfhc" in r["skipped"] and "http://10.46.252.48/show_bug.cgi?id=22" in r["skipped"]
    assert [f["quote"] for f in r["findings"]] == [NOTE]
    assert {x["why"] for x in r["rejected"]} == {"quote not found verbatim on the page", "url was not one of the fetched pages"}
    rep = out["report"]
    assert "### Release notes and docs (web; leads to check" in rep and NOTE in rep and "2 quote(s) dropped" in rep
    before = [c["hyp"]["cause"] for c in out["ranked"]]
    out2 = asyncio.run(build_graph(F()).ainvoke({"archive": str(archive)}))
    assert [c["hyp"]["cause"] for c in out2["ranked"]] == before      # research never changes the ranking
    assert out["calls"] <= 25


def test_no_research_for_hardware_only_or_without_tools(archive):
    class HW(F):
        def answer(self, schema, u):
            if schema.__name__ == "Judge":
                from agents.triage.graph import Judge
                return Judge(supported=("HBM stack" in u or "refined" in u), why="w")
            return super().answer(schema, u)
    w, calls = web()
    out = asyncio.run(build_graph(HW(), research=w).ainvoke({"archive": str(archive)}))
    assert {c["hyp"]["category"] for c in out["ranked"]} == {"hardware"}
    assert out["research"] == {"ran": False} and not calls["search"]
    assert asyncio.run(build_graph(F()).ainvoke({"archive": str(archive)}))["research"] == {"ran": False}


def test_verify_finding():
    pages = [{"url": "u", "text": "Line one.\n  Known   issue:  X fails on Y  when Z.\n"}]
    assert verify_finding({"url": "u", "quote": "known issue: X fails on Y when Z."}, pages) == ""
    assert verify_finding({"url": "u", "quote": "short"}, pages).startswith("quote shorter")
    assert verify_finding({"url": "v", "quote": "known issue: X fails on Y when Z."}, pages).startswith("url was not")
