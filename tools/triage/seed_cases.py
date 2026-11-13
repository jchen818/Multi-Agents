"""Build triage eval archives from evals/triage/cases.yaml, in the same layout the agfhc server archives real runs:
summary.json + logs/{console.log, dmesg_new.log, gpu_usage_start.txt, agfhc_logs/agfhc_<ts>/{results.json, journal.log}}.

    .venv/bin/python -m tools.triage.seed_cases state/triage_eval     # builds every case
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "evals" / "triage" / "cases.yaml"
TZ = "America/Los_Angeles"
START = "2026-11-02 08:00:00"          # every seeded event happens after this run start

BUSY = ("============ Memory Usage (VRAM%) ============\n"
        + "".join(f"GPU[{i}]\t\t: GPU Memory Allocated (VRAM%): 92\n" for i in range(8))
        + "============ KFD Processes ============\nPID\tPROCESS NAME\tGPU(s)\tVRAM USED\n"
          "48211\tpython3 (vllm serve)\t0,1,2,3,4,5,6,7\t1922335744000\n")
IDLE = ("============ Memory Usage (VRAM%) ============\n"
        + "".join(f"GPU[{i}]\t\t: GPU Memory Allocated (VRAM%): 0\n" for i in range(8))
        + "No KFD PIDs currently running\n")


def load(path: Path = CASES) -> tuple:
    data = yaml.safe_load(open(path))
    return data["defaults"], data["cases"]


def build(case: dict, defaults: dict, out: Path) -> Path:
    start = datetime.strptime(START, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo(TZ))
    if case.get("kind") == "bug":                          # bug evidence carries its own (older) dates
        start = start.replace(year=2026, month=10, day=1)
    ts = start.strftime("%Y%m%d-%H%M%S")
    recipe = case.get("recipe") or defaults["recipe"]
    d = out / f"AGFHC_EVAL_{case['id']}"
    agl = d / "logs" / "agfhc_logs" / f"agfhc_{ts}"
    agl.mkdir(parents=True, exist_ok=True)
    failed = [f"{f['test']}-1-1" for f in case["failed"]]
    versions = dict(defaults["versions"], **(case.get("versions") or {}))
    summary = {"run_id": f"eval-{case['id']}", "host": "eval-node", "platform": "mi325x", "recipe": recipe,
               "cmd": f"/opt/amd/agfhc/agfhc -r {recipe}", "verdict": "FAIL", "rc": 13, "failed": failed,
               "start_epoch": int(start.timestamp()), "end_epoch": int(start.timestamp()) + 600, "timezone": TZ,
               "versions": versions, "counts": {"FAIL": len(failed)}, "eval_case": case["id"]}
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    console = [f"Log Directory: /root/agfhc/logs/agfhc_{ts}",
               f"Product: MI325X | GPUs: 8 | Mode: BM SPX NPS1 | Checks: dmesg+ ras+ perf+ cper+"]
    console += [f"FAIL: {f['test']}-1-1".ljust(36) + f"[{f.get('time', '00:00:01')}/{f.get('limit', '00:01:00')}] {f['status']}"
                for f in case["failed"]]
    console.append(f"Tests: {len(failed)} Total, {len(failed)} Executed, 0 Skipped | 0 Passed, {len(failed)} Failed")
    (d / "logs" / "console.log").write_text("\n".join(console) + "\n")
    results = {"summary": {"total": len(failed), "failed": len(failed)},
               "results": [{"state": "failed", "primary_status_name": f["status"], "primary_action_subject": f["subject"],
                            "time_total": f.get("time", "00:00:01"), "test": f["test"], "run": 1, "iteration": 1,
                            "extended_results": [{"action_subject": f["subject"], "test": f["lines"]}]}
                           for f in case["failed"]]}
    (agl / "results.json").write_text(json.dumps(results, indent=1))
    jts = start.strftime("%Y%m%d-%H%M%S")
    journal = [f"{jts} INFO | {f['test']} | {f['subject']}: {line}" for f in case["failed"] for line in f["lines"]]
    if case.get("journal"):
        journal.insert(0, f"{jts} {case['journal']}")
    (agl / "journal.log").write_text("\n".join(journal) + "\n")
    (d / "logs" / "dmesg_new.log").write_text(case.get("dmesg_new") or "")
    if case.get("gpu_usage"):
        (d / "logs" / "gpu_usage_start.txt").write_text(BUSY if case["gpu_usage"] == "busy" else IDLE)
    return d


def build_all(out: Path, path: Path = CASES) -> list:
    defaults, cases = load(path)
    return [build(c, defaults, out) for c in cases]


if __name__ == "__main__":
    for p in build_all(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "state" / "triage_eval")):
        print(p)
