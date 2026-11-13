"""Unit tests for tools/agfhc/agfhc_lib.py, ported from selfcheck_agfhc.py (Open WebUI AGFHC Test Runner v0.4.3).

The end-to-end, register_dut, tmux and Open WebUI exposure checks of the original belong to the MCP server
(tools/agfhc/server.py) and its dry-run tests, not to the pure helpers.
"""
import json
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "agfhc"))
import agfhc_lib as T  # noqa: E402


def check(cond, msg):
    assert cond, msg


SAMPLE_LOG = """
AGFHC v1.30.2  recipe: all_lvl1
[2026-10-14 10:00:01] gpu_info ............ PASS
[2026-10-14 10:01:10] hbm_bw .............. PASS
[2026-10-14 10:05:44] xgmi_a2a ............ FAIL
  GPU3 <-> GPU5 link bandwidth 41.2 GB/s below threshold 48.0 GB/s
[2026-10-14 10:06:00] pcie_link_check: SKIPPED
| dgemm_stress | PASSED |
Total errors: 0 errors
Overall result: FAIL
"""


def test_parser():
    p = T.parse_agfhc_log(SAMPLE_LOG)
    check(p["tests"].get("gpu_info") == "PASS", "parses dotted PASS")
    check(p["tests"].get("xgmi_a2a") == "FAIL", "parses FAIL")
    check(p["tests"].get("pcie_link_check") == "SKIP", "parses colon SKIPPED")
    check(p["tests"].get("dgemm_stress") == "PASS", "parses table PASSED")
    check(p["overall"] == "FAIL", "parses overall verdict")
    check(any("41.2 GB/s" in l for l in p["suspicious"]), "keeps threshold line as suspicious")
    check(not any("0 errors" in l for l in p["suspicious"]), "ignores '0 errors' noise")
    check("result" not in p["tests"] and "Overall" not in p["tests"], "no noise test names")


REAL_AGFHC_LOG = """
AGFHC version 1.30.2 - recipe all_lvl1
PASS: gpu_info-1-1 [00:00:02/00:00:30]
PASS: hbm_bw-1-1 [00:00:41/00:01:00]
FAIL: hbm_s16_ds-1-1 [00:00:06/00:01:20] TEST_HBM_STRESS_FAIL
SKIP: xgmi_a2a-1-1 [00:00:00/00:02:00] NOT_SUPPORTED
Return code: 13 (PROG_FAIL_IN_TEST_EXECUTION)
"""


def test_real_agfhc_format():
    p = T.parse_agfhc_log(REAL_AGFHC_LOG)
    check(p["tests"] == {"gpu_info-1-1": "PASS", "hbm_bw-1-1": "PASS", "hbm_s16_ds-1-1": "FAIL", "xgmi_a2a-1-1": "SKIP"},
          f"status-first AGFHC lines parsed: {p['tests']}")
    check(p["details"]["hbm_s16_ds-1-1"] == {"time": "00:00:06/00:01:20", "detail": "TEST_HBM_STRESS_FAIL"},
          "elapsed time + failure code captured")
    s = T.summarize(p, 13)
    check("**Verdict:** FAIL" in s and "hbm_s16_ds-1-1: FAIL [00:00:06/00:01:20] TEST_HBM_STRESS_FAIL" in s,
          "LLM summary includes failure detail")
    h = T.render_report_html({"tests": p["tests"], "test_details": p["details"], "counts": T.count_statuses(p["tests"]),
                              "verdict": "FAIL", "kernel_new": ["x"] * 3, "kernel_new_total": 912})
    check("TEST_HBM_STRESS_FAIL" in h and "00:00:06/00:01:20" in h, "report shows detail + time columns")
    check("<b>912</b>" in h and "last 3 of 912" in h, "report shows true kernel message total")


MITACAI_LOG = """
    PASS: pcie_link_status-1-1            [00:00:00/00:00:08]
    PASS: pcie_bidi_peak-1-1              [00:00:18/00:00:50]
    PASS: xgmi_a2a-1-1                    [00:00:09/00:00:25]
    FAIL: hbm_s16_ds-1-1                  [00:00:06/00:01:20] TEST_HBM_STRESS_FAIL
    PASS: hbm_bw-1-1                      [00:00:49/00:01:05]
"""


E = "\x1b"


COLORED_LOG = f"""Product: MI325X | GPUs: 8 | Mode: BM SPX NPS1 | Checks: dmesg+ ras+ perf+ cper+
pcie_link_status   PCIe Link Status          1 iteration                   0:00:08
xgmi_a2a           XGMI BW A2A               1 iteration                   0:00:25
PCIe Link Status
EXEC: pcie_link_status-1-1            [00:00:00/00:00:08]
{E}[92mPASS{E}[0m: pcie_link_status-1-1            [00:00:00/00:00:08]
EXEC: pcie_bidi_peak-1-1              [00:00:19/00:00:50]
{E}[92mPASS{E}[0m: pcie_bidi_peak-1-1              [00:00:19/00:00:50]
EXEC: xgmi_a2a-1-1                    [00:00:10/00:00:25]
{E}[92mPASS{E}[0m: xgmi_a2a-1-1                    [00:00:10/00:00:25]
{E}[92mPASS{E}[0m: gpu_info-1-1                    [00:00:01/00:00:10]
{E}[92mPASS{E}[0m: hbm_bw-1-1                      [00:00:49/00:01:05]
{E}[91mFAIL{E}[0m: hbm_s16_ds-1-1                  [00:00:25/00:01:20] TEST_HBM_STRESS_FAIL
Tests: 6 Total, 6 Executed, 0 Skipped | {E}[92m5 Passed{E}[0m, {E}[91m1 Failed{E}[0m
For more information on {E}[91mFAILs{E}[0m, see: {E}[91m__ROOTLOG__/agfhc_20261015-101151/results.json{E}[0m
Program exiting with return code PROG_FAIL_IN_TEST_EXECUTION [13]
"""


def test_colored_agfhc_output():
    log = COLORED_LOG.replace("__ROOTLOG__", "/root/agfhc/logs")
    p = T.parse_agfhc_log(log)
    c = T.count_statuses(p["tests"])
    check(c == {"PASS": 5, "FAIL": 1}, f"ANSI-colored result lines parsed: {c}")
    check(p["totals"] == {"total": 6, "executed": 6, "skipped": 0, "passed": 5, "failed": 1}, "AGFHC totals line parsed")
    check(T.totals_mismatch(p) == "", "no mismatch when lines match totals")
    check(p["config"].startswith("Product: MI325X") and p["logdir"] == "/root/agfhc/logs/agfhc_20261015-101151",
          "config + AGFHC log dir extracted")
    check(not any(l.startswith("EXEC") or "pcie_link_status   PCIe" in l or "Checks:" in l for l in p["suspicious"]),
          "EXEC/progress/banner lines not suspicious")
    check(any("TEST_HBM_STRESS_FAIL" in l for l in p["suspicious"]) and not any(E in l for l in p["suspicious"]),
          "FAIL line kept, colors stripped")
    p2 = T.parse_agfhc_log(log.replace(f"{E}[92mPASS{E}[0m: gpu_info-1-1", "gpu_info-1-1"))
    check("4 passed / 1 failed" in T.totals_mismatch(p2), "mismatch warning when a result line is missing")
    s = T.summarize(p, 13, results_failures='[{"test": "hbm_s16_ds", "result": "FAIL"}]')
    check("6 total" in s and "5 passed, 1 failed" in s and "results.json" in s, "LLM summary has totals + results.json")
    rj = T.results_json_failures(json.dumps({"tests": [{"name": "hbm_bw", "status": "PASS"},
                                                      {"name": "hbm_s16_ds", "status": "FAIL", "gpu": 3, "msg": "miscompare"}]}))
    check("hbm_s16_ds" in rj and "miscompare" in rj and "hbm_bw" not in rj, "results.json failing entries extracted")
    h = T.render_report_html({"tests": p["tests"], "agfhc_totals": p["totals"], "verdict": "FAIL",
                              "counts": T.count_statuses(p["tests"]), "results_failures": rj})
    check("<b>6</b><span>Tests</span>" in h and ">5</b><span>Passed</span>" in h and ">1</b><span>Failed</span>" in h,
          "report chips use AGFHC totals")
    check("results.json — failing entries" in h and "miscompare" in h, "report shows results.json failures")


def _ticker(name, n, limit, final):
    """AGFHC-style line: EXEC ticks redrawn with \r, then the final colored verdict, on ONE physical line."""
    ticks = "\r".join(f"EXEC: {name:<30}[00:00:{i:02d}/{limit}]" for i in range(n))
    return ticks + "\r" + final


CR_LOG_PART1 = (
    "Product: MI325X | GPUs: 8 | Mode: BM SPX NPS1 | Checks: dmesg+ ras+ perf+ cper+\n"
    "pcie_link_status   PCIe Link Status          1 iteration                   0:00:08\n"
    "pcie_bidi_peak     PCIe BW BiDi Peak         1 iteration                   0:00:50\n"
    "hbm_bw             HBM BW                    1 iteration                   0:01:05\n"
    "hbm_s16_ds         HBM Stress                1 iteration                   0:01:20\n"
    + _ticker("pcie_link_status-1-1", 1, "00:00:08", f"{E}[92mPASS{E}[0m: pcie_link_status-1-1 [00:00:00/00:00:08]") + "\n"
    + _ticker("pcie_bidi_peak-1-1", 20, "00:00:50", f"{E}[92mPASS{E}[0m: pcie_bidi_peak-1-1 [00:00:19/00:00:50]") + "\n"
    + "\r".join(f"EXEC: hbm_bw-1-1                    [00:00:{i:02d}/00:01:05]" for i in range(22))
)


CR_LOG_PART2 = (
    f"\r{E}[92mPASS{E}[0m: hbm_bw-1-1 [00:00:49/00:01:05]\n"
    + _ticker("hbm_s16_ds-1-1", 26, "00:01:20", f"{E}[91mFAIL{E}[0m: hbm_s16_ds-1-1 [00:00:25/00:01:20] TEST_HBM_STRESS_FAIL") + "\n"
    + f"Tests: 4 Total, 4 Executed, 0 Skipped | {E}[92m3 Passed{E}[0m, {E}[91m1 Failed{E}[0m\n"
    + "Program exiting with return code PROG_FAIL_IN_TEST_EXECUTION [13]\n"
)


def test_planned_table():
    pl = T.parse_planned(CR_LOG_PART1)
    check([t["name"] for t in pl] == ["pcie_link_status", "pcie_bidi_peak", "hbm_bw", "hbm_s16_ds"], "planned tests parsed in order")
    check(pl[3] == {"name": "hbm_s16_ds", "desc": "HBM Stress", "iters": 1, "limit": "0:01:20"}, "planned test fields")
    md = T.format_planned(pl)
    check("| 1 | `pcie_link_status` | PCIe Link Status | 1 | 0:00:08 |" in md and "**4 tests**" in md
          and "3m23s" in md, f"planned table + total time limit: {md.splitlines()[-1]}")


def test_progress_from_cr_log():
    check(CR_LOG_PART1.count("\n") == 7, "fixture: ticker lines are single physical lines")
    mid = T.progress_from_log(CR_LOG_PART1)
    check([d[:2] for d in mid["done"]] == [("PASS", "pcie_link_status-1-1"), ("PASS", "pcie_bidi_peak-1-1")],
          f"mid-run: finished tests found despite \\r ticker: {mid['done']}")
    check(mid["planned"] == 4 and mid["current"] == "hbm_bw-1-1 [00:00:21/00:01:05]",
          f"mid-run: planned count + LATEST tick of current test: {mid['planned']} / {mid['current']}")
    end = T.progress_from_log(CR_LOG_PART1 + CR_LOG_PART2)
    check([d[0] for d in end["done"]] == ["PASS", "PASS", "PASS", "FAIL"] and end["current"] == "",
          "end: 3 PASS + 1 FAIL, no current test")
    check(end["done"][3] == ("FAIL", "hbm_s16_ds-1-1", "00:00:25/00:01:20", "TEST_HBM_STRESS_FAIL"), "end: fail detail")
    check("3 Passed" in end["totals"] and E not in end["totals"], "end: totals line without color codes")
    h = T.format_progress("RUNNING", 185, mid, "America/Los_Angeles")
    check("RUNNING · 3m05s elapsed · 2/4 done · 2 PASS / 0 FAIL · now: `hbm_bw-1-1 [00:00:21/00:01:05]`" in h, f"headline: {h}")
    f = T.format_progress("FINISHED", 3790, end, "UTC", "13")
    check("**FINISHED** · 1h03m elapsed · 4/4 done · 3 PASS / 1 FAIL · exit code 13 · Tests: 4 Total" in f, f"final: {f}")
    t = T.format_test_line(end["done"][3])
    check(t == "  - ❌ FAIL `hbm_s16_ds-1-1` [00:00:25/00:01:20] TEST_HBM_STRESS_FAIL", f"nested test line: {t}")
    check(T.plain(h).startswith(h[2:12]) and "`" not in T.plain(h) and not T.plain(t).startswith("-"), "plain status text")


def test_indented_real_lines():
    p = T.parse_agfhc_log(MITACAI_LOG)
    check(len(p["tests"]) == 5 and p["tests"]["hbm_s16_ds-1-1"] == "FAIL" and p["tests"]["xgmi_a2a-1-1"] == "PASS",
          f"indented mitacai lines parsed: {T.count_statuses(p['tests'])}")
    check(p["details"]["pcie_link_status-1-1"]["time"] == "00:00:00/00:00:08", "indented line time captured")


def _bash(cmd):
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout


def test_dmesg_new_filter():
    d = tempfile.mkdtemp()
    with open(f"{d}/after", "w") as f:
        f.write("[  100.000001] amdgpu 0000:05:00.0: old message\n"
                "[  250.500000] amdgpu 0000:05:00.0: [drm] RAS: HBM ECC uncorrectable error\n"
                "[ 1000.250000] pcieport 0000:00:01.1: AER: Corrected error\n")
    out = _bash(f"awk -v t0=200.0 {shlex.quote(T.DMESG_NEW_AWK)} {d}/after")
    check(out.splitlines() == ["[+50.5s] amdgpu 0000:05:00.0: [drm] RAS: HBM ECC uncorrectable error",
                               "[+800.2s] pcieport 0000:00:01.1: AER: Corrected error"],
          f"only messages after test start, as offsets: {out.splitlines()}")
    # legacy ISO snapshots whose timestamps drifted between reads -> nothing new
    with open(f"{d}/b", "w") as f:
        f.write("2026-10-13T04:06:25,684203-07:00 workqueue: svm_range_restore_work [amdgpu] hogged CPU\n")
    with open(f"{d}/a", "w") as f:
        f.write("2026-10-13T04:06:25,684209-07:00 workqueue: svm_range_restore_work [amdgpu] hogged CPU\n"
                "2026-10-15T09:58:00,000000-07:00 amdgpu: RAS new\n")
    out = _bash(f"comm -13 <(sed -E 's/^[^ ]+ //' {d}/b | sort) <(sed -E 's/^[^ ]+ //' {d}/a | sort)")
    check(out.strip() == "amdgpu: RAS new", f"legacy ISO drift ignored, real new line kept: {out.strip()!r}")
    shutil.rmtree(d)


def test_summary():
    p = T.parse_agfhc_log(SAMPLE_LOG)
    s = T.summarize(p, 1, "2026-10-14T10:05:40 amdgpu 0000:c1:00.0: RAS: xgmi link error\n", budget=6000)
    check("**Verdict:** FAIL" in s and "xgmi_a2a: FAIL" in s and "RAS" in s, "summary has verdict, failures, dmesg")
    check(len(T.summarize(p, 1, "x" * 50000, budget=500)) <= 500, "summary respects char budget")
    check("UNKNOWN" in T.summarize({"tests": {}, "suspicious": []}, 0), "empty parse -> UNKNOWN, not PASS")


def test_helpers():
    c = T.build_run_cmd("{bin} -r {recipe} {extra}", "/opt/amd/agfhc/agfhc", "all_lvl1", "/x", "-i 0,1")
    check(c == "/opt/amd/agfhc/agfhc -r all_lvl1 -i 0,1", f"run cmd built: {c}")
    c2 = T.build_run_cmd("{bin} -r {recipe} {extra}", "/opt/a", "r1; rm -rf /", "/x")
    check("'r1; rm -rf /'" in c2, "injection in recipe is quoted")
    s = T.split_sections("@@RC\n@@MANIFEST\n{}\n@@LOG\nline1\nline2\n@@DMESG\n")
    check(s["RC"] == "" and s["LOG"] == "line1\nline2" and s["DMESG"] == "", "section split incl. empty sections")
    kv = T.parse_kv("AGFHC=AGFHC v1.30.2\nROCM=7.2.1\nGPU_COUNT=\nnoise line\nlower=x")
    check(kv == {"AGFHC": "AGFHC v1.30.2", "ROCM": "7.2.1", "GPU_COUNT": "unknown"}, f"parse_kv: {kv}")
    check("| ROCm | 7.2.1 |" in T.format_versions(kv), "versions table")
    # 2026-10-15 16:30:00 UTC = 09:30:00 PDT
    n = T.archive_folder_name(1792081800, "AGFHC_%Y%m%d_%H%M%S", "America/Los_Angeles")
    check(n == "AGFHC_20261015_093000", f"archive folder name in local tz: {n}")
    check(T.archive_folder_name(1792081800, "AGFHC_%Y%m%d_%H%M%S", "Bad/Zone") == "AGFHC_20261015_163000",
          "bad timezone falls back to UTC")
    check(T.fmt_duration(100, 3825) == "1h 02m 05s", "duration format")
    h = T.md_to_html("## Verdict\n**FAIL** on `xgmi_a2a`\n\n- GPU3 <script>\n- GPU5\n\n1. rerun\n```\nraw <b>\n```")
    check("<h4>Verdict</h4>" in h and "<strong>FAIL</strong>" in h and "<code>xgmi_a2a</code>" in h, "md headings/bold/code")
    check("<ul>" in h and "<ol>" in h and "&lt;script&gt;" in h and "raw &lt;b&gt;" in h, "md lists + escaping")


def _results_json(fail_subject="GPU-05"):
    ok = lambda t, tt: {"state": "passed", "primary_status_name": "TEST_PASS", "time_total": tt, "test": t,
                        "run": 1, "iteration": 1, "extended_results": []}
    return json.dumps({
        "summary": {"total_selected_tests": 6, "total_executed_tests": 6, "total_passed": 5, "total_failed": 1,
                    "total_skipped": 0, "status_code_text": "PROG_FAIL_IN_TEST_EXECUTION", "status_code_numeric": 13,
                    "failed_instances": ["hbm_s16_ds-1-1"]},
        "results": [
            ok("pcie_link_status", "00:00:00"), ok("pcie_bidi_peak", "00:00:19"), ok("xgmi_a2a", "00:00:10"),
            ok("hbm_bw", "00:00:49"),
            {"state": "failed", "primary_status_name": "TEST_HBM_STRESS_FAIL", "primary_status_value": 12001,
             "primary_action_subject": fail_subject, "time_total": "00:00:05", "test": "hbm_s16_ds", "run": 1,
             "iteration": 1,
             "extended_results": [{"action_subject": fail_subject, "test": [
                 "2026-10-15 11:53:20.402 FATL| ABORT: R1 GPU-00 failed to allocate buffer.", "Aborted (core dumped)"]}]},
            ok("gfx_dgemm", "00:01:18")]})


NOISY_CONSOLE = (
    "    PASS: pcie_link_status-1-1 [00:00:00/00:00:08]\n    PASS: pcie_bidi_peak-1-1 [00:00:19/00:00:50]\n"
    "    PASS: xgmi_a2a-1-1 [00:00:10/00:00:25]\n    PASS: hbm_bw-1-1 [00:00:49/00:01:05]\n"
    "    FAIL: hbm_s16_ds-1-1 [00:00:05/00:01:20] TEST_HBM_STRESS_FAIL\n    PASS: gfx_dgemm-1-1 [00:01:18/00:01:30]\n"
    # sub-tool output that used to be counted as tests:
    "BABEL ............ PASS\nGST | PASS\n2026-10-15 11:53:20.402 FATL| ABORT: R1 GPU-00 failed to allocate buffer.\n"
    "Tests: 6 Total, 6 Executed, 0 Skipped | 5 Passed, 1 Failed\n")


def test_results_json_authoritative():
    rj = T.results_json_tests(_results_json())
    check(len(rj["tests"]) == 6 and T.count_statuses(rj["tests"]) == {"PASS": 5, "FAIL": 1},
          f"results.json: 6 instances, 5 PASS / 1 FAIL: {rj['tests']}")
    check(rj["details"]["hbm_s16_ds-1-1"] == {"time": "00:00:05", "detail": "TEST_HBM_STRESS_FAIL (GPU-05)"},
          "failed instance carries status code + blamed GPU")
    check(rj["details"]["hbm_bw-1-1"]["detail"] == "", "passed instances have no detail")
    check(rj["totals"] == {"total": 6, "executed": 6, "skipped": 0, "passed": 5, "failed": 1}, "totals from results.json")
    noisy = T.parse_agfhc_log(NOISY_CONSOLE)
    check(any(n in noisy["tests"] for n in ("BABEL", "GST", "FATL")), "fixture: console parsing alone picks up sub-tool noise")
    m = T.merge_results(T.parse_agfhc_log(NOISY_CONSOLE), _results_json())
    check(sorted(m["tests"]) == sorted(rj["tests"]) and m["source"] == "results.json", "merge: exactly the 6 real tests")
    check(T.totals_mismatch(m) == "", "no mismatch warning once results.json is used")
    check(m["details"]["hbm_s16_ds-1-1"]["time"] == "00:00:05/00:01:20", "console time (elapsed/limit) kept when available")
    f = T.merge_results(T.parse_agfhc_log(MITACAI_LOG), "")
    check(f["source"] == "console.log" and len(f["tests"]) == 5, "falls back to console.log without results.json")


def test_gpu_busy():
    mem = ("============ Memory Usage (VRAM%) ============\n" + "".join(
        f"GPU[{i}]\t\t: GPU Memory Allocated (VRAM%): {92 if i in (4, 5, 6, 7) else 0}\n"
        f"GPU[{i}]\t\t: GPU Memory Read/Write Activity (%): 0\n" for i in range(8)))
    pids = ("============ KFD Processes ============\nKFD process information:\n"
            "PID\tPROCESS NAME\tGPU(s)\tVRAM USED\tSDMA USED\tCU OCCUPANCY\n"
            "48211\tpython3\t4\t961167872000\t0\t0\n")
    u = T.parse_gpu_usage(mem, pids, 5)
    check(u["busy"] == [(4, 92), (5, 92), (6, 92), (7, 92)] and u["checked"] == 8 and u["procs"] == ["48211 python3 4 961167872000 0 0"],
          f"busy GPUs + processes parsed: {u}")
    w = T.gpu_busy_warning(u)
    check("NOT idle" in w and "GPU[5] 92%" in w and "48211 python3" in w and "failed to allocate buffer" in w, "busy warning")
    idle = T.parse_gpu_usage(mem.replace(": 92", ": 0"), "No KFD PIDs currently running\n", 5)
    check(idle["busy"] == [] and idle["procs"] == [] and T.gpu_busy_warning(idle) == "", "idle GPUs -> no warning")
    check(T.parse_gpu_usage("", "", 5)["checked"] == 0, "no rocm-smi data -> unknown, not 'idle'")


def test_inventory_helpers():
    rows = T.parse_inventory("Tools\tAGFHC\tAGFHC version 1.30.2\nFirmware\tSystem BIOS\t ()\nbad line\n"
                             "GPU firmware\tVBIOS\tA\nGPU firmware\tVBIOS\tA\nGPU firmware\tPSP\tX\nGPU firmware\tPSP\tY\n"
                             "Package\trocm\t7.2.1\nCustom\tthing\t1\n")
    check(["Firmware", "System BIOS", "unknown"] in rows and len(rows) == 8, "parse_inventory cleans empty values")
    agg = dict(T.aggregate_inventory(rows))
    gf = {i["component"]: i for i in agg["GPU firmware (all GPUs)"]}
    check(gf["VBIOS"]["value"] == "A  (×2)" and not gf["VBIOS"]["mismatch"], "identical per-GPU values collapse")
    check(gf["PSP"]["mismatch"] and "X (×1) · Y (×1)" == gf["PSP"]["value"], "per-GPU mismatch flagged")
    check(list(agg)[0] == "Test & ROCm tools" and "Custom" in agg, "category order + unknown categories kept")
    check(T.inventory_mismatches(rows) == ["PSP: X (×1) · Y (×1)"], "mismatch list")


def test_report_html():
    p = T.parse_agfhc_log(SAMPLE_LOG)
    s = {"run_id": "r1", "host": "mi325x-1", "address": "10.0.0.11", "platform": "mi325x", "recipe": "all_lvl1",
         "cmd": "agfhc -r all_lvl1", "operator": "johnc", "start_epoch": 1792081800, "end_epoch": 1792085000, "rc": 1,
         "verdict": "FAIL", "counts": T.count_statuses(p["tests"]), "tests": p["tests"], "suspicious": p["suspicious"],
         "kernel_new": ["amdgpu RAS <xgmi> error"], "versions": {"AGFHC": "1.30.2", "ROCM": "7.2.1"},
         "timezone": "America/Los_Angeles", "analysis": "### Likely cause\n- XGMI link GPU3↔GPU5",
         "files": [{"path": "logs/console.log", "size": 2048}], "nfs_url": "10.46.252.71:/srv/nfs/logs/AGFHC_x/",
         "generated": "now"}
    h = T.render_report_html(s)
    check(h.startswith("<!doctype html>") and 'class="badge v-FAIL"' in h, "report has verdict badge")
    check("0h 53m 20s" in h and "2026-10-15 09:30:00 PDT" in h, "report has local times + duration")
    check("7.2.1" in h and "Likely cause" in h and 'href="logs/console.log"' in h, "report has versions, analysis, log links")
    check("&lt;xgmi&gt;" in h and "<xgmi>" not in h, "report escapes log content")


def test_log_analyzer_paths_pure():
    m = ("10.46.252.71:/srv/nfs /mnt/logs nfs4 rw 0 0\n10.46.252.71:/srv/nfs/logs /data/agfhc nfs4 rw 0 0\n"
         "10.46.252.99:/srv/nfs /other nfs4 rw 0 0\n/dev/sda1 /srv/nfs ext4 rw 0 0\n")
    check(T.local_root_from_mounts(m, {"10.46.252.71"}, "/srv/nfs/logs") == "/data/agfhc", "most specific NFS export wins")
    check(T.local_root_from_mounts(m.splitlines()[0], {"10.46.252.71"}, "/srv/nfs/logs/") == "/mnt/logs/logs",
          "parent export mounted at /mnt/logs -> /mnt/logs/logs")
    check(T.local_root_from_mounts(m, {"1.2.3.4"}, "/srv/nfs/logs") == "", "other servers / local disks ignored")
    files = [{"path": p, "size": 1} for p in (
        "logs/console.log", "logs/agfhc_logs/agfhc_20261015-112416/error.json",
        "logs/agfhc_logs/agfhc_20261015-112416/results.json", "logs/agfhc_logs/agfhc_20261015-112416/tests/hbm_s16_ds-1-1/stdout.log",
        "logs/agfhc_logs/agfhc_20261015-112416/tests/hbm_bw-1-1/stdout.log")]
    fm = T.failure_file_map(files, ["hbm_s16_ds-1-1"], "/mnt/logs/logs/AGFHC_x")
    check("`/mnt/logs/logs/AGFHC_x/logs/agfhc_logs/agfhc_20261015-112416/error.json`" in fm
          and "Files of failed test `hbm_s16_ds-1-1`:" in fm and "tests/hbm_s16_ds-1-1/stdout.log" in fm
          and "hbm_bw-1-1" not in fm, "file map lists key files + only the failed test's files")


def test_misc_pure_helpers():
    check(T.ip_allowed("10.46.252.120", "10.46.252.0/24") and not T.ip_allowed("8.8.8.8", "10.46.252.0/24"), "ip_allowed")
    check(T.platform_from_model("AMD Instinct MI325X") == "mi325x" and T.platform_from_model("") == "unknown", "platform")
    check(T.redact("pw=abc123", "abc123") == "pw=******", "redact")
    rid = "20261112T174708Z-mi325x-1-all-lvl1-ab58"
    check(T.tmux_session_name(rid) == "agfhc-174708-ab58", f"tmux session name: {T.tmux_session_name(rid)}")


def test_parse_list_real_output():
    out = T.parse_list((Path(__file__).resolve().parents[1] / "sim/data/list.txt").read_text())
    check(out["version"] == "1.30.2", f"version {out['version']}")
    check(len(out["tests"]) == 30 and "hbm_s16_ds" in out["tests"], f"tests {len(out['tests'])}")
    check(len(out["recipes"]) == 40 and "all_lvl1" in out["recipes"] and "Name" not in out["recipes"], f"recipes {len(out['recipes'])}")
    check(T.parse_list("garbage") == {"version": "", "tests": [], "recipes": []}, "unparseable -> empty")
