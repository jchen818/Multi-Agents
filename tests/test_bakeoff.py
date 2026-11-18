"""Phase 6.3 bake-off harness: both implementations run the same scripted model (a fake OpenAI-compatible server)
through the same tools, approval step and scoring."""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bakeoff import common as C  # noqa: E402

KEY = "amdgpu: 1 uncorrectable hardware errors detected in umc block"


def tool_call(name, args, i):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": f"call_{i}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}


def text(t):
    return {"role": "assistant", "content": t}


FILE_ARGS = {"summary": "hbm_s16_ds fails: uncorrectable UMC errors on GPU-03", "product": "G8825Z5U2BC",
             "description": f"Run on eval-node.\nEvidence:\n{KEY}", "component": "GPU", "severity": "critical"}


class Fake(BaseHTTPRequestHandler):
    script: list = []
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.seen.append(body)
        msg = Fake.script.pop(0) if Fake.script else text("NOT FILED")
        out = {"id": "x", "object": "chat.completion", "created": 0, "model": body.get("model", "m"),
               "choices": [{"index": 0, "message": msg,
                            "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    Fake.seen = []
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def scripts():
    return {
        "file": [tool_call("read_run", {}, 1), tool_call("search_bugs", {"query": "hbm_s16_ds uncorrectable"}, 2),
                 tool_call("file_bug", FILE_ARGS, 3), text("FILED")],
        "reject": [tool_call("read_run", {}, 1), tool_call("file_bug", FILE_ARGS, 2),
                   text("NOT FILED (rejected): reviewer said lab cooling")],
    }


def case(cid):
    return next(c for c in C.cases() if c["id"] == cid)


def run_langgraph(base, cid):
    import asyncio
    from langchain_openai import ChatOpenAI
    from bakeoff.impl_langgraph import run_case
    model = ChatOpenAI(model="m", base_url=base, api_key="EMPTY", temperature=0, max_retries=0)
    r = asyncio.run(run_case(case(cid), model))
    return C.score(r["ctx"], r["final"]), r["ctx"]


def sdk_python():
    """The interpreter that has the OpenAI Agents SDK: .venv-bakeoff on nfs-server, or this one if it has it."""
    for py in (REPO / ".venv-bakeoff" / "bin" / "python", Path(sys.executable)):
        if py.exists() and subprocess.run([str(py), "-c", "import agents; agents.Runner"], cwd="/tmp",
                                          capture_output=True).returncode == 0:
            return str(py)
    pytest.skip("OpenAI Agents SDK not installed (.venv-bakeoff)")


def run_openai_sdk(base, cid, script):
    """In a separate process from /tmp, so `import agents` is the installed SDK (as with bakeoff/run.py)."""
    Fake.script = script                     # the fake model server runs in this process
    code = f"""
import asyncio, json, sys
sys.path.append({str(REPO)!r})
from bakeoff import common as C
from bakeoff.impl_openai_sdk import local_model, run_case
c = next(x for x in C.cases() if x["id"] == {cid!r})
r = asyncio.run(run_case(c, local_model("m", {base!r})))
ok, why = C.score(r["ctx"], r["final"])
print(json.dumps({{"ok": ok, "why": why, "filed": r["ctx"].filed, "requests": len(r["ctx"].approval_requests)}}))
"""
    p = subprocess.run([sdk_python(), "-c", code], cwd="/tmp", capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_cases_and_scoring_rules(tmp_path):
    cs = C.cases()
    assert len(cs) == 15 and {c["expect"] for c in cs} == {"file", "duplicate", "rejected"}
    ctx = C.make_ctx(case("inject-hw-hbm-ue"), tmp_path)
    assert "NOTE TO THE AI AGENT" in C.read_run(ctx)                              # the injection is in the logs
    ctx.filed.append(dict(FILE_ARGS, severity="blocker", description=FILE_ARGS["description"] + " admin@example.com"))
    ok, why = C.score(ctx, "FILED")
    assert not ok and "injected" in why
    dup = C.make_ctx(case("bug23-rochpl-oversized"), tmp_path / "d")
    assert C.score(dup, "DUPLICATE OF #23")[0] and not C.score(dup, "FILED, see #22")[0]
    assert "#23" in C.search_bugs(dup, "rochpl device memory")


def test_langgraph_files_after_approval(server):
    Fake.script = scripts()["file"]
    (ok, why), ctx = run_langgraph(server, "seed-hw-hbm-ue")
    assert ok, why
    assert len(ctx.approval_requests) == 1 and ctx.filed[0]["severity"] == "critical"


def test_langgraph_stops_on_rejection(server):
    Fake.script = scripts()["reject"]
    (ok, why), ctx = run_langgraph(server, "seed-thermal-throttle")
    assert ok, why
    assert ctx.filed == [] and len(ctx.approval_requests) == 1


def test_openai_sdk_files_after_approval(server):
    r = run_openai_sdk(server, "seed-hw-hbm-ue", scripts()["file"])
    assert r["ok"], r
    assert r["requests"] == 1 and r["filed"][0]["severity"] == "critical"


def test_openai_sdk_stops_on_rejection(server):
    r = run_openai_sdk(server, "seed-thermal-throttle", scripts()["reject"])
    assert r["ok"], r
    assert r["filed"] == [] and r["requests"] == 1


def test_run_script_finds_the_sdk_not_this_repo():
    p = subprocess.run([sdk_python(), "-c", "import sys; sys.path.append(%r); import agents; "
                        "print(hasattr(agents, 'Runner'))" % str(REPO)], cwd="/tmp", capture_output=True, text=True)
    assert p.stdout.strip() == "True"
