"""End-to-end dry-run of the agfhc MCP server against the simulated node (sim/bin/agfhc). Touches no GPU."""
import asyncio
import importlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "agfhc"))


@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTI_AGENTS_ROOT", str(REPO))
    monkeypatch.setenv("AGFHC_MODE", "dryrun")
    monkeypatch.setenv("AGFHC_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENTS_KILL_FILE", str(tmp_path / "KILL"))
    monkeypatch.setenv("AGFHC_TMUX", "0")
    monkeypatch.setenv("SIM_SLEEP", "1")
    monkeypatch.delenv("SIM_GPU_BUSY", raising=False)
    spec = importlib.util.spec_from_file_location("agfhc_server", REPO / "tools" / "agfhc" / "server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(c):
    return asyncio.run(c)


def token(text):
    return re.search(r"confirm_token: `([^`]+)`", text).group(1)


def wait_finished(srv, run_id, limit=60):
    for _ in range(limit):
        st = run(srv.get_run_status("sim", run_id))
        if st.startswith("state=FINISHED"):
            return st
        time.sleep(1)
    raise AssertionError(f"run did not finish: {st}")


def full_run(srv, monkeypatch, scenario):
    monkeypatch.setenv("SIM_SCENARIO", scenario)
    plan = run(srv.plan_run("sim", "single_pass", operator="pytest"))
    assert "Planned AGFHC run (not started)" in plan and "DRY-RUN" in plan
    started = run(srv.start_run(token(plan)))
    assert started.startswith("STARTED"), started
    run_id = re.search(r"run_id=`([^`]+)`", started).group(1)
    wait_finished(srv, run_id)
    return run_id, run(srv.analyze_run("sim", run_id))


def test_list_nodes_dryrun(srv):
    out = srv.list_test_nodes()
    assert "**sim**" in out and "can run AGFHC" in out
    assert re.search(r"\*\*mitacai\*\*.*read-only", out)


def test_fail_scenario_archives_and_reports(srv, monkeypatch, tmp_path):
    run_id, out = full_run(srv, monkeypatch, "fail")
    assert "FAIL" in out and "Archived" in out, out
    arch = [d for d in (tmp_path / "sim" / "archive").iterdir() if d.name.startswith("AGFHC_")]
    assert len(arch) == 1
    summary = json.loads((arch[0] / "summary.json").read_text())
    assert summary["verdict"] == "FAIL" and summary["failed"]
    assert summary["results_source"] == "results.json"
    assert (arch[0] / "report.html").exists() and (arch[0] / "logs" / "console.log").exists()
    assert "GPU" in (arch[0] / "logs" / "gpu_usage_start.txt").read_text()      # GPU usage when the run started
    fin = run(srv.finalize_report("sim", run_id, "One test failed; see failure details."))
    assert "report.html" in fin
    assert "One test failed" in (arch[0] / "report.html").read_text()


def test_pass_scenario(srv, monkeypatch, tmp_path):
    _, out = full_run(srv, monkeypatch, "pass")
    summary = json.loads(next((tmp_path / "sim" / "archive").glob("AGFHC_*/summary.json")).read_text())
    assert summary["verdict"] == "PASS" and not summary["failed"], out


def test_refuses_real_node_in_dryrun(srv):
    assert run(srv.plan_run("mitacai", "single_pass")).startswith("REFUSED: dry-run mode")


def test_refuses_unknown_recipe_and_node(srv):
    assert "allowlist" in run(srv.plan_run("sim", "rm_rf_everything"))
    assert "disallowed characters" in run(srv.plan_run("sim", "a;b"))
    assert "unknown node" in run(srv.plan_run("ghost", "single_pass"))


def test_refuses_busy_gpus(srv, monkeypatch):
    monkeypatch.setenv("SIM_GPU_BUSY", "1")
    assert run(srv.plan_run("sim", "single_pass")).startswith("REFUSED")


def test_token_single_use_and_kill_switch(srv, monkeypatch, tmp_path):
    monkeypatch.setenv("SIM_SLEEP", "0")
    plan = run(srv.plan_run("sim", "single_pass"))
    (tmp_path / "KILL").touch()
    assert "kill switch" in run(srv.start_run(token(plan)))
    (tmp_path / "KILL").unlink()
    assert "unknown, used or expired" in run(srv.start_run("bogus"))
    plan = run(srv.plan_run("sim", "single_pass"))
    t = token(plan)
    first = run(srv.start_run(t))
    assert first.startswith("STARTED"), first
    assert "unknown, used or expired" in run(srv.start_run(t))
    wait_finished(srv, re.search(r"run_id=`([^`]+)`", first).group(1))


def test_recipe_missing_from_installed_agfhc_is_refused(srv, monkeypatch, tmp_path):
    import shutil
    data = tmp_path / "simdata"
    shutil.copytree(REPO / "sim" / "data", data)
    lst = data / "list.txt"
    lst.write_text("\n".join(l for l in lst.read_text().splitlines() if not l.startswith("all_lvl2 ")))
    monkeypatch.setenv("SIM_DATA", str(data))
    out = run(srv.plan_run("sim", "all_lvl2"))
    assert out.startswith("REFUSED") and "does not define recipe 'all_lvl2'" in out and "1.30.2" in out
    assert "Planned AGFHC run" in run(srv.plan_run("sim", "all_lvl1"))


def test_versions_include_bios(srv):
    out = run(srv.preflight("sim"))
    assert "| System BIOS | SIM-BIOS-1.0 |" in out
