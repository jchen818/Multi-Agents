"""Test-coding agent: the worktree jail, the AGENTS.md diff checks, the push guard, and the plan/code/verify loop."""
import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from agents.coder import checks  # noqa: E402
from agents.coder.gitops import GitHub, PushRefused, Repo, check_branch, slug  # noqa: E402
from agents.coder.graph import Critique, Plan, build_graph, pr_text  # noqa: E402
from agents.coder.sandbox import docker_cmd, failed_tests  # noqa: E402
from agents.coder.workspace import Workspace  # noqa: E402

import yaml  # noqa: E402

CFG = {**yaml.safe_load((REPO / "config" / "coder.yaml").read_text()), "max_iterations": 2}
CFG["sandbox"] = {**CFG["sandbox"], "backend": "local", "timeout_s": 120}

MINI = {
    "AGENTS.md": "# Rules\n1. Read-only through the catalog.\n",
    "pytest.ini": "[pytest]\ntestpaths = tests\n",
    "gpuval/__init__.py": "",
    "gpuval/commands.py": 'COMMANDS = {\n    "gpu_count": "cat /sys/x",\n}\n',
    "config/platforms/mi325x.yaml": "name: mi325x\ngpu_count: 8\n",
    "conftest.py": "",
    "recordings/r1/out/gpu_count.txt": "8\n",
    "recordings/r1/index.json": "{}\n",
    "recordings/r1/out/amdsmi_static.txt": '{"gpu_data": [{"gpu": 0, "vram": {"size": {"value": 262128, "unit": "MB"}}}]}',
    "tests/__init__.py": "",
    "tests/test_basic.py": "def test_basic():\n    assert 8 == 8\n",
    "tests/test_checks_catch.py": "def test_catch_basic():\n    assert True\n",
}


def git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def mini(tmp_path):
    root = tmp_path / "suite"
    for rel, text in MINI.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    git(root.parent, "init", "-q", "-b", "main", str(root))
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    return root


# ------------------------------------------------------------------ workspace jail

def test_workspace_stays_inside_the_worktree(mini):
    ws = Workspace(mini, CFG)
    tools = {t.name: t for t in ws.tools()}
    assert tools["read_file"].invoke({"path": "../../etc/passwd"}).startswith("REFUSED")
    assert tools["read_file"].invoke({"path": "/etc/passwd"}).startswith("ERROR")      # becomes <root>/etc/passwd
    assert tools["read_file"].invoke({"path": ".git/config"}).startswith("REFUSED")
    assert "REFUSED" in tools["write_file"].invoke({"path": "recordings/r1/out/gpu_count.txt", "content": "7"})
    assert "REFUSED" in tools["write_file"].invoke({"path": "conftest.py", "content": "x"})
    assert "REFUSED" in tools["write_file"].invoke({"path": "AGENTS.md", "content": "x"})
    assert "REFUSED" in tools["write_file"].invoke({"path": "tests/run.sh", "content": "rm -rf /"})
    assert "REFUSED" in tools["run_tests"].invoke({"target": "--node 10.0.0.1"})
    assert "wrote" in tools["write_file"].invoke({"path": "tests/test_new.py", "content": "def test_n():\n  assert 1\n"})
    assert "[protected]" in tools["list_files"].invoke({"directory": ""})


def test_workspace_runs_tests_and_records_findings(mini):
    ws = Workspace(mini, CFG)
    out = ws.run_tests("tests/test_basic.py")
    assert out.startswith("exit 0"), out
    with pytest.raises(Exception):
        ws.declare_finding("tests/test_basic.py::test_basic", "short")
    ws.declare_finding("tests/test_basic.py::test_basic", "GPU 3 reports 750 W, the others 1000 W")
    assert "tests/test_basic.py::test_basic" in ws.findings


# ------------------------------------------------------------------ AGENTS.md diff checks

def diff_of(mini, files: dict) -> str:
    for rel, text in files.items():
        p = mini / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    git(mini, "add", "-A", "-N")
    return git(mini, "diff", "HEAD")


def run_checks(mini, files):
    ws = Workspace(mini, CFG)
    from agents.coder.graph import known_fields
    return checks.check(diff_of(mini, files), CFG, (mini / "gpuval/commands.py").read_text(), ws.protected,
                        versions=ws.versions, known_fields=known_fields(ws))


@pytest.mark.parametrize("files,expect", [
    ({"tests/test_x.py": 'def test_x(runner):\n    assert runner.run("gpu_count").out == "8"\n'}, None),
    ({"tests/test_x.py": 'NODE = "10.46.252.115"\n'}, "IP address"),
    ({"tests/test_x.py": 'def test_x():\n    assert host != "mitacai"\n'}, "node/host name"),
    ({"tests/test_x.py": 'def test_x(v):\n    assert v == "7.2.1"\n'}, "pinned version"),
    ({"tests/test_x.py": 'import pytest\n@pytest.mark.xfail\ndef test_x():\n    assert 0\n'}, "xfail"),
    ({"tests/test_x.py": 'import pytest\ndef test_x():\n    pytest.skip("flaky")\n'}, "remove this skip"),
    ({"tests/test_x.py": 'import pytest\ndef test_x():\n    pytest.skip("recording lacks it: re-capture")\n'}, None),
    ({"tests/test_x.py": 'def test_x(runner):\n    runner.run("dmesg_all")\n'}, "not in gpuval/commands.py"),
    ({"tests/test_basic.py": "def test_basic():\n    pass\n"}, "assert(s) removed"),
    ({"config/platforms/mi325x.yaml": "name: mi325x\ngpu_count: 4\n"}, "existing expectation changed"),
    ({"config/platforms/mi325x.yaml": "name: mi325x\ngpu_count: 8\nvram_mb: 262128\n"}, None),
    ({"config/platforms/mi325x.yaml": "# reformatted\nname:   mi325x\nvram_mb: 262128\ngpu_count: 8\n"}, None),
    ({"config/platforms/mi325x.yaml": "name: mi325x\nvram_mb: 262128\n"}, "gpu_count removed"),
    ({"recordings/r1/out/gpu_count.txt": "7\n"}, "protected path"),
    ({"tests/test_x.py": 'def test_x(static, platform):\n    assert all(g["vram"]["size"]["value"] == platform["gpu_count"] for g in static)\n'}, None),
    ({"tests/test_x.py": 'def test_x(static):\n    assert all(g["ras"].get("total_retired_pages", 0) == 0 for g in static)\n'},
     "field 'total_retired_pages' does not exist"),
    ({"conftest.py": "x = 1\n"}, "protected path"),
])
def test_diff_checks(mini, files, expect):
    rep = run_checks(mini, files)
    if expect is None:
        assert rep.ok, rep.problems
    else:
        assert any(expect in p for p in rep.problems), rep.problems


def test_recorded_data_index(mini):
    text, keys = Workspace(mini, CFG).recorded_data()
    assert "vram.size=262128 MB" in text and "gpu_count: text: 8" in text and "size" in keys


def test_new_catalog_command_is_reported(mini):
    rep = run_checks(mini, {"gpuval/commands.py": 'COMMANDS = {\n    "gpu_count": "cat /sys/x",\n'
                                                  '    "bad_pages": "amd-smi bad-pages --json",\n}\n',
                            "tests/test_bp.py": 'def test_bp(runner):\n    runner.run("bad_pages")\n'})
    assert rep.ok, rep.problems
    assert rep.new_commands == ["bad_pages"] and rep.new_tests == ["tests/test_bp.py::test_bp"]


# ------------------------------------------------------------------ sandbox

def test_docker_sandbox_is_locked_down(tmp_path):
    cmd = " ".join(docker_cmd(tmp_path, CFG, ["tests/test_x.py"]))
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "no-new-privileges", f"{tmp_path}:/w:ro",
                 "--pids-limit"):
        assert flag in cmd
    assert "--node" not in cmd
    assert failed_tests("FAILED tests/test_a.py::test_b - AssertionError: x\nERROR tests/test_c.py") == \
        ["tests/test_a.py::test_b", "tests/test_c.py"]


# ------------------------------------------------------------------ push guard

class FakeGH(GitHub):
    def __init__(self, remote, move_main=False):
        super().__init__("o/r", "t")
        self.remote, self.move_main, self.prs = remote, move_main, []

    def branch_sha(self, branch):
        if branch == "main" and self.move_main:
            self.main_calls = getattr(self, "main_calls", 0) + 1
            if self.main_calls >= 2:                 # someone pushed main while the agent was pushing
                return "0" * 40
        return git(self.remote, "rev-parse", f"refs/heads/{branch}")

    def open_pr(self, head, base, title, body, draft):
        self.prs.append((head, base, title, draft))
        return {"html_url": "https://example/pr/1"}


@pytest.fixture
def remote(mini, tmp_path):
    bare = tmp_path / "remote.git"
    git(tmp_path, "clone", "-q", "--bare", str(mini), str(bare))
    return bare


def test_branch_names():
    for bad in ("main", "agent/", "agent/../main", "feature/x", "agent/UPPER", "refs/heads/main"):
        with pytest.raises(PushRefused):
            check_branch(bad)
    check_branch("agent/" + slug("Check the power cap on every GPU!"))


def test_push_only_the_task_branch(remote, tmp_path):
    gh = FakeGH(remote)
    repo = Repo({**CFG, "repo": "o/r"}, "token", github=gh, home=tmp_path / "coder", url=str(remote))
    main_before = git(remote, "rev-parse", "refs/heads/main")
    repo.sync()
    branch, path = repo.new_worktree("add power cap check")
    (path / "tests" / "test_power.py").write_text("def test_p():\n    assert 1\n")
    sha = repo.commit(path, "Add power cap check")
    assert repo.push(path, branch) == sha
    assert git(remote, "rev-parse", f"refs/heads/{branch}") == sha
    assert git(remote, "rev-parse", "refs/heads/main") == main_before
    with pytest.raises(PushRefused):
        repo.push(path, "main")
    assert "Requested-by:" in git(path, "log", "-1", "--format=%B")


def test_push_stops_and_alerts_if_main_moves(remote, tmp_path):
    gh = FakeGH(remote, move_main=True)
    home = tmp_path / "coder"
    repo = Repo({**CFG, "repo": "o/r"}, "token", github=gh, home=home, url=str(remote))
    repo.sync()
    branch, path = repo.new_worktree("another check")
    (path / "tests" / "test_y.py").write_text("def test_y():\n    assert 1\n")
    repo.commit(path, "y")
    with pytest.raises(PushRefused, match="moved"):
        repo.push(path, branch)
    assert "moved" in (home / "ALERT").read_text()


def test_token_never_in_argv_or_config(remote, tmp_path):
    repo = Repo({**CFG, "repo": "o/r"}, "SECRET-TOKEN", github=FakeGH(remote), home=tmp_path / "c", url=str(remote))
    repo.sync()
    assert "SECRET-TOKEN" not in (repo.clone / ".git" / "config").read_text()
    assert "SECRET-TOKEN" not in repo.env["GIT_CONFIG_VALUE_0"]          # base64 in a header, via the environment


# ------------------------------------------------------------------ the loop

class FakeModel:
    """structured() calls: Plan, then Critique. Returns queued objects."""

    def __init__(self, plan, critiques=()):
        self.queue = {Plan: [plan], Critique: list(critiques)}

    def with_structured_output(self, schema, **kw):
        model = self

        class R:
            async def ainvoke(self, msgs):
                q = model.queue[schema]
                return {"parsed": q.pop(0) if len(q) > 1 else q[0], "raw": None}
        return R()


class ScriptedAgent:
    """Stands in for the tool-calling loop: each round writes the given files through the workspace tools."""

    def __init__(self, ws, rounds):
        self.ws, self.rounds = ws, rounds          # shared list: each round consumes one entry

    async def ainvoke(self, inputs, config=None):
        files, summary, finding = self.rounds.pop(0) if len(self.rounds) > 1 else self.rounds[0]
        tools = {t.name: t for t in self.ws.tools()}
        for rel, text in files.items():
            tools["write_file"].invoke({"path": rel, "content": text})
        if finding:
            tools["declare_finding"].invoke(finding)

        class M:
            content = summary
        return {"messages": [M()]}


GOOD = ({"tests/test_count.py": 'def test_count():\n    n = int(open("recordings/r1/out/gpu_count.txt").read())\n'
                                '    assert n == 8, f"{n} GPUs"\n',
         "tests/test_checks_catch.py": MINI["tests/test_checks_catch.py"] +
         "\n\ndef test_count_catch():\n    from tests.test_count import test_count\n    assert test_count\n"},
        "Check the GPU count\nReads gpu_count.", None)


def run_graph(mini, plan, rounds, critiques=(Critique(ok=True),)):
    ws = Workspace(mini, CFG)
    rounds = list(rounds)
    g = build_graph(FakeModel(plan, critiques), ws, CFG, agent_factory=lambda m, w, c: ScriptedAgent(w, rounds))
    return asyncio.run(g.ainvoke({"task": "check GPU count"}))


PLAN = Plan(refuse=False, data="gpu_count", files=["tests/test_count.py"], steps=["write it"])


def test_good_change_is_ready(mini):
    s = run_graph(mini, PLAN, [GOOD])
    assert s["outcome"] == "ready", s.get("report")
    assert s["iteration"] == 1 and s["tests"]["rc"] == 0
    title, body = pr_text(s, CFG)
    assert title == "Check the GPU count" and "## Checks" in body and "Findings" not in body


def test_problems_go_back_to_the_model_then_pass(mini):
    bad = ({"tests/test_count.py": 'def test_count():\n    assert "10.46.252.115"\n'}, "x", None)
    s = run_graph(mini, PLAN, [bad, GOOD])
    assert s["outcome"] == "ready" and s["iteration"] == 2


def test_gives_up_after_max_rounds(mini):
    bad = ({"tests/test_count.py": 'def test_count():\n    assert "10.46.252.115"\n'}, "x", None)
    s = run_graph(mini, PLAN, [bad])
    assert s["outcome"] == "gave_up" and s["iteration"] == 2
    assert any("IP address" in p for p in s["report"]["problems"])


def test_missing_catch_test_is_a_problem(mini):
    only = ({"tests/test_count.py": GOOD[0]["tests/test_count.py"]}, "x", None)
    s = run_graph(mini, PLAN, [only])
    assert s["outcome"] == "gave_up"
    assert any("no catch test" in p for p in s["report"]["problems"])


def test_refused_task_changes_nothing(mini):
    s = run_graph(mini, Plan(refuse=True, refuse_reason="resetting a GPU writes to hardware"), [GOOD])
    assert s["outcome"] == "refused" and "hardware" in s["reason"]
    assert git(mini, "status", "--porcelain") == ""


def test_new_failing_check_can_be_a_finding_and_opens_a_draft(mini):
    fail = dict(GOOD[0])
    fail["tests/test_count.py"] = 'def test_count():\n    assert 7 == 8, "GPU 3 missing"\n'
    finding = {"test": "tests/test_count.py::test_count", "reason": "the recording lists 7 GPUs; platform says 8"}
    s = run_graph(mini, PLAN, [(fail, "Check the GPU count", finding)])
    assert s["outcome"] == "ready", s["report"]
    title, body = pr_text(s, CFG)
    assert title.startswith("[finding]") and "Findings" in body


def test_existing_test_cannot_be_declared_a_finding(mini):
    broken = ({**GOOD[0], "tests/test_basic.py": "def test_basic():\n    assert 8 == 9\n"}, "x",
              {"test": "tests/test_basic.py::test_basic", "reason": "pretend this is real hardware trouble"})
    s = run_graph(mini, PLAN, [broken])
    assert s["outcome"] == "gave_up"
    assert any("only a NEW test" in p for p in s["report"]["problems"])


def test_critique_cannot_loop_forever(mini):
    s = run_graph(mini, PLAN, [GOOD], critiques=[Critique(ok=False, violations=["style nit"])])
    assert s["outcome"] == "ready" and s["iteration"] == 2      # max_iterations is 2 in these tests
    assert s["critique"]["unresolved"] == ["style nit"]
    assert "did not resolve" in pr_text(s, CFG)[1]
    assert [h["stage"] for h in s["history"]] == ["critique"]


def test_critique_sends_it_back(mini):
    s = run_graph(mini, PLAN, [GOOD], critiques=[Critique(ok=False, violations=["the check cannot fail"]),
                                                 Critique(ok=True)])
    assert s["outcome"] == "ready" and s["iteration"] == 2


# ------------------------------------------------------------------ eval scoring

def test_eval_scoring_rules(tmp_path):
    sys.path.insert(0, str(REPO / "evals" / "coder"))
    from run import score, set_path
    doc = {"gpu_data": [{"a": {"b": 1}}, {"a": {"b": 1}}]}
    set_path(doc, "gpu_data.1.a.b", 7)
    assert doc["gpu_data"][1]["a"]["b"] == 7
    assert score({"expect": "refused"}, {"outcome": "refused"}, tmp_path, CFG)[0]
    assert not score({"expect": "refused"}, {"outcome": "ready", "report": {}}, tmp_path, CFG)[0]
    assert not score({"expect": "ready"}, {"outcome": "gave_up", "reason": "x"}, tmp_path, CFG)[0]
    st = {"outcome": "ready", "report": {"new_tests": ["tests/test_a.py::test_a"], "new_commands": []},
          "findings": {"tests/test_a.py::test_a": "x" * 30}, "tests": {"tail": ""}}
    assert "findings" in score({"expect": "ready"}, st, tmp_path, CFG)[1]
    cap = {"outcome": "ready", "report": {"new_tests": ["tests/test_b.py::test_b"], "new_commands": ["bad_pages"]},
           "tests": {"tail": "SKIPPED [1] tests/test_b.py: recording lacks bad_pages: re-capture"}}
    assert score({"expect": "capture"}, cap, tmp_path, CFG)[0]
