"""AGFHC pure helpers, ported unchanged from the Open WebUI "AGFHC Test Runner" v0.4.3 (johnc, MiTAC).

Parsing (console.log, results.json, planned-test table, live progress), verdicts, inventory aggregation,
GPU-busy detection, the remote shell snippets (inventory script, dmesg filter) and the HTML report.
No I/O except what callers pass in; the SSH/MCP layer lives in tools/agfhc/server.py.
Tests: tests/test_agfhc_lib.py (ported from selfcheck_agfhc.py).
"""

import html
import json
import re
import secrets
import shlex
from datetime import datetime, timezone
from typing import Any, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None


TOOL_VERSION = "0.4.3"
RUN_ROOT = "/var/tmp/agfhc-runs"
TOKEN_TTL_S = 300
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.:,\-]{1,128}$")
SAFE_ARGS = re.compile(r"^[A-Za-z0-9_.:,=/ \-]{0,256}$")
SAFE_PATH = re.compile(r"^/[A-Za-z0-9_./\-]{1,255}$")

VERSION_KEYS = [
    ("AGFHC", "AGFHC"),
    ("ROCM", "ROCm"),
    ("AMDGPU", "amdgpu driver"),
    ("AMDSMI", "amd-smi"),
    ("KERNEL", "Kernel"),
    ("OS", "OS"),
    ("GPU_MODEL", "GPU model"),
    ("GPU_COUNT", "GPU count"),
    ("VBIOS", "VBIOS"),
    ("HOSTNAME", "Hostname"),
]

# Full driver / firmware / tool inventory for the report. Prints "Category<TAB>Component<TAB>Version" lines.
# Placeholders __SUDO__ and __AGFHC_VER__ are filled in by Tools._inventory_script(). Every probe is optional.
INVENTORY_SCRIPT = r"""
PATH="$PATH:/usr/sbin:/sbin:/opt/rocm/bin"
p(){ printf '%s\t%s\t%s\n' "$1" "$2" "$3"; }
first(){ grep -m1 . | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'; }
# ---- tools
p Tools AGFHC "$(__AGFHC_VER__ 2>&1 | first)"
p Tools ROCm "$(cat /opt/rocm/.info/version 2>/dev/null)"
command -v amd-smi >/dev/null && p Tools amd-smi "$(amd-smi version 2>/dev/null | first)"
command -v rocm-smi >/dev/null && p Tools rocm-smi "$(rocm-smi --version 2>/dev/null | grep -im1 version | sed 's/.*[Vv]ersion[: ]*//')"
command -v hipconfig >/dev/null && p Tools HIP "$(hipconfig --version 2>/dev/null | first)"
p Tools Python "$(python3 --version 2>&1 | awk '{print $2}')"
# ---- drivers
p Driver "amdgpu (kernel module)" "$(cat /sys/module/amdgpu/version 2>/dev/null || modinfo -F version amdgpu 2>/dev/null)"
p Driver "Linux kernel" "$(uname -r)"
for n in /sys/class/net/*; do
  i=$(basename "$n"); [ -e "$n/device/driver" ] || continue
  drv=$(basename "$(readlink -f "$n/device/driver")"); dv=""; fw=""; bus=""
  if command -v ethtool >/dev/null; then
    e=$(ethtool -i "$i" 2>/dev/null)
    dv=$(printf '%s\n' "$e" | awk -F': ' '/^version:/{print $2}')
    fw=$(printf '%s\n' "$e" | awk -F': ' '/^firmware-version:/{print $2}')
    bus=$(printf '%s\n' "$e" | awk -F': ' '/^bus-info:/{print $2}')
  fi
  [ -z "$dv" ] && dv=$(cat "/sys/module/$drv/version" 2>/dev/null)
  [ -z "$dv" ] && dv=$(modinfo -F version "$drv" 2>/dev/null)
  [ -z "$dv" ] && dv="in-kernel ($(uname -r))"
  p Driver "NIC $i ($drv${bus:+, $bus})" "$dv"
  [ -n "$fw" ] && [ "$fw" != "N/A" ] && p "NIC firmware" "$i ($drv)" "$fw"
done
# ---- GPU firmware (one line per GPU; aggregated later so mismatches show up)
if command -v amd-smi >/dev/null; then
  amd-smi static --vbios 2>/dev/null | awk '/^[[:space:]]*VERSION:/{print "GPU firmware\tVBIOS\t" $2}'
  amd-smi static --vbios 2>/dev/null | awk '/^[[:space:]]*PART_NUMBER:/{print "GPU firmware\tVBIOS part number\t" $2}'
  amd-smi firmware 2>/dev/null | awk '/FW_ID:/{id=$2} /FW_VERSION:/{print "GPU firmware\t" id "\t" $2}'
fi
# ---- platform firmware
d=/sys/class/dmi/id
p Firmware "System BIOS" "$(cat $d/bios_version 2>/dev/null)$( [ -r $d/bios_date ] && echo " ($(cat $d/bios_date))")"
bmc=$(timeout 10 __SUDO__ipmitool mc info 2>/dev/null | awk -F': ' '/Firmware Revision/{print $2}')
p Firmware "BMC" "${bmc:-unknown (needs ipmitool + sudo)}"
p Firmware "CPU microcode" "$(grep -m1 microcode /proc/cpuinfo | awk -F': ' '{print $2}')"
for c in /sys/class/nvme/nvme*; do
  [ -e "$c" ] || continue
  p "Storage firmware" "$(basename "$c") $(sed 's/[[:space:]]*$//' "$c/model" 2>/dev/null)" "$(sed 's/[[:space:]]*$//' "$c/firmware_rev" 2>/dev/null)"
done
# ---- platform
p Platform System "$(cat $d/sys_vendor 2>/dev/null) $(cat $d/product_name 2>/dev/null)"
p Platform Board "$(cat $d/board_vendor 2>/dev/null) $(cat $d/board_name 2>/dev/null)"
p Platform CPU "$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
p Platform OS "$(. /etc/os-release 2>/dev/null; echo "$PRETTY_NAME")"
# ---- installed ROCm / AMD packages
{ if command -v dpkg-query >/dev/null; then dpkg-query -W -f='${Package}\t${Version}\n' 2>/dev/null;
  elif command -v rpm >/dev/null; then rpm -qa --qf '%{NAME}\t%{VERSION}-%{RELEASE}\n' 2>/dev/null; fi; } \
 | grep -E '^(rocm|amdgpu|amd-smi|hip|rccl|rocblas|hipblas|miopen|rocm-validation-suite|transferbench|agfhc|rocprofiler|roctracer|rocminfo|rdc|half|comgr|hsa)' \
 | sort -u | while IFS="$(printf '\t')" read -r n v; do p Package "$n" "$v"; done
"""

INVENTORY_ORDER = [
    ("Tools", "Test & ROCm tools"),
    ("Driver", "Drivers"),
    ("GPU firmware", "GPU firmware (all GPUs)"),
    ("Firmware", "System firmware"),
    ("NIC firmware", "NIC firmware"),
    ("Storage firmware", "Storage firmware"),
    ("Platform", "Platform"),
    ("Package", "Installed ROCm / AMD packages"),
]

# ---------------------------------------------------------------------------
# Pure helpers (no I/O) — unit-tested by selfcheck_agfhc.py
# ---------------------------------------------------------------------------


def parse_inventory(text: str) -> list:
    """Parse 'Category<TAB>Component<TAB>Version' lines into [[cat, comp, ver], ...]."""
    rows = []
    for line in (text or "").splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        cat, comp, ver = (x.strip() for x in parts)
        if not cat or not comp:
            continue
        if ver in ("", "()", "( )") or re.fullmatch(r"\(?\s*\)?", ver):
            ver = "unknown"
        rows.append([cat, comp, ver])
    return rows


def aggregate_inventory(rows: list) -> list:
    """Group rows by category (report order) and collapse per-GPU duplicates.

    Returns [(category_label, [{"component", "value", "mismatch"}...]), ...]. A component reported with
    different values (e.g. one GPU on other firmware) is flagged mismatch=True.
    """
    from collections import OrderedDict

    groups: "OrderedDict[str, OrderedDict[str, list]]" = OrderedDict()
    for cat, comp, ver in rows:
        groups.setdefault(cat, OrderedDict()).setdefault(comp, []).append(ver)
    labels = dict(INVENTORY_ORDER)
    order = [c for c, _ in INVENTORY_ORDER] + [c for c in groups if c not in labels]
    out = []
    for cat in order:
        if cat not in groups:
            continue
        items = []
        for comp, vals in groups[cat].items():
            distinct = list(dict.fromkeys(vals))
            if len(distinct) == 1:
                v = distinct[0] + (f"  (×{len(vals)})" if len(vals) > 1 else "")
                items.append({"component": comp, "value": v, "mismatch": False})
            else:
                v = " · ".join(f"{d} (×{vals.count(d)})" for d in distinct)
                items.append({"component": comp, "value": v, "mismatch": True})
        out.append((labels.get(cat, cat), items))
    return out


def inventory_mismatches(rows: list) -> list:
    return [
        f"{comp}: {it['value']}"
        for _, its in aggregate_inventory(rows)
        for it in its
        if it["mismatch"]
        for comp in [it["component"]]
    ]


DEFAULT_RESULT_REGEX = (
    r"(?P<test>[A-Za-z][\w.\-:/]{2,})\s*(?:\.{2,}|:|\||-|\s)\s*\[?\s*"
    r"(?P<status>PASS(?:ED)?|FAIL(?:ED)?|ERROR|SKIP(?:PED)?|ABORT(?:ED)?|TIMEOUT)\b"
)
# AGFHC v1.30 style: "FAIL: hbm_s16_ds-1-1 [00:00:06/00:01:20] TEST_HBM_STRESS_FAIL" (status first).
# Always tried before the result_regex valve (which handles "test_name ..... PASS" style).
STATUS_FIRST_REGEX = re.compile(
    r"^\W*(?:\[[^\]]*\]\s*)?(?P<status>PASS(?:ED)?|FAIL(?:ED)?|ERROR|SKIP(?:PED)?|ABORT(?:ED)?|TIMEOUT)\s*:\s*"
    r"(?P<test>[A-Za-z][\w.\-:/]*)\s*(?:\[(?P<time>[^\]]*)\])?\s*(?P<detail>.*)$",
    re.IGNORECASE,
)
STATUS_MAP = {
    "PASS": "PASS",
    "PASSED": "PASS",
    "FAIL": "FAIL",
    "FAILED": "FAIL",
    "ERROR": "ERROR",
    "SKIP": "SKIP",
    "SKIPPED": "SKIP",
    "ABORT": "ERROR",
    "ABORTED": "ERROR",
    "TIMEOUT": "ERROR",
}
# Failure words only (not topic words like "pcie"/"xgmi"/"ras", which appear in every AGFHC test name).
ERROR_HINTS = re.compile(
    r"(error|fail|fault|timeout|uncorrect|\bhang|\breset|page retire|bad page|throttl|\bmce\b|"
    r"below threshold|above threshold|exceed|mismatch|degrad|out of range|not found|missing)",
    re.IGNORECASE,
)
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b?\[[0-9;]*m")
# "Tests: 6 Total, 6 Executed, 0 Skipped | 5 Passed, 1 Failed"
AGFHC_TOTALS_RE = re.compile(
    r"Tests:\s*(?P<total>\d+)\s*Total,\s*(?P<executed>\d+)\s*Executed,\s*(?P<skipped>\d+)\s*Skipped\s*\|\s*"
    r"(?P<passed>\d+)\s*Passed,\s*(?P<failed>\d+)\s*Failed",
    re.IGNORECASE,
)
AGFHC_LOGDIR_RE = re.compile(r"(/[^\s'\"]+/agfhc_\d{8}-\d{6})")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


# Kernel messages worth correlating with GPU tests (whole-word match, so "ras" doesn't hit "erase").
DMESG_FILTER = "amdgpu|ras|aer|xgmi|mce|ecc|hbm|pcieport"
# Keep only dmesg lines logged after the run started, using the kernel's monotonic [seconds] stamp
# (stable across reads, unlike --time-format iso), and print them as "[+N.Ns since start] message".
DMESG_NEW_AWK = (
    "{ if (match($0, /^\\[ *[0-9]+\\.[0-9]+\\]/)) { ts = substr($0, 2, RLENGTH - 2) + 0; "
    'if (ts > t0) printf "[+%.1fs] %s\\n", ts - t0, substr($0, RLENGTH + 2) } }'
)
NOISE_TESTS = {"result", "results", "status", "summary", "overall", "total", "test"}


def parse_agfhc_log(text: str, result_regex: str = DEFAULT_RESULT_REGEX) -> dict:
    """Extract per-test verdicts and suspicious lines from an AGFHC console log.

    Returns {"tests": {name: status}, "details": {name: {"time", "detail"}}, "overall": str|None,
    "suspicious": [lines]}. Last verdict for a test wins (AGFHC prints progress then final status).
    """
    rx = re.compile(result_regex, re.IGNORECASE)
    tests: dict[str, str] = {}
    details: dict[str, dict] = {}
    overall = None
    suspicious: list[str] = []
    totals: Optional[dict] = None
    config = ""
    logdir = ""
    for raw in strip_ansi(text).splitlines():
        line = raw.strip()
        if not line or re.match(r"^(EXEC|RUN|RUNNING)\s*:", line, re.IGNORECASE):
            continue  # progress ticks
        m_tot = AGFHC_TOTALS_RE.search(line)
        if m_tot:
            totals = {k: int(v) for k, v in m_tot.groupdict().items()}
            continue
        if line.startswith("Product:") and not config:
            config = line[:200]
            continue
        m_dir = AGFHC_LOGDIR_RE.search(line)
        if m_dir and not logdir:
            logdir = m_dir.group(1)
        m_sf = STATUS_FIRST_REGEX.match(line)
        if m_sf and m_sf.group("test").lower() not in NOISE_TESTS:
            name = m_sf.group("test").rstrip(".:-")
            tests[name] = STATUS_MAP.get(m_sf.group("status").upper(), "ERROR")
            details[name] = {
                "time": (m_sf.group("time") or "").strip(),
                "detail": m_sf.group("detail").strip()[:200],
            }
            if tests[name] in ("FAIL", "ERROR"):
                suspicious.append(line)
            continue
        m_over = re.search(
            r"\b(overall|final|summary)\b[^A-Za-z]*(?:result|status)?\s*[:=\-]?\s*"
            r"(PASS(?:ED)?|FAIL(?:ED)?|ERROR)\b",
            line,
            re.IGNORECASE,
        )
        if m_over:
            overall = STATUS_MAP[m_over.group(2).upper()]
            continue
        m = rx.search(line)
        if m:
            name = m.group("test").strip(" .:|-")
            if name.lower() not in NOISE_TESTS:
                tests[name] = STATUS_MAP.get(m.group("status").upper(), "ERROR")
                if tests[name] in ("FAIL", "ERROR"):
                    suspicious.append(line)
                continue
        if ERROR_HINTS.search(line) and not re.search(r"\b0 errors?\b", line, re.I):
            suspicious.append(line)
    if totals and not overall:
        overall = "FAIL" if totals["failed"] else ("PASS" if totals["passed"] else None)
    return {
        "tests": tests,
        "details": details,
        "overall": overall,
        "suspicious": suspicious,
        "totals": totals,
        "config": config,
        "logdir": logdir,
    }


def totals_mismatch(parsed: dict) -> str:
    """Warn when per-test lines disagree with AGFHC's own 'Tests: … Passed, … Failed' line."""
    t = parsed.get("totals")
    if not t:
        return ""
    c = count_statuses(parsed.get("tests", {}))
    got_p, got_f = c.get("PASS", 0), c.get("FAIL", 0) + c.get("ERROR", 0)
    if (got_p, got_f) != (t["passed"], t["failed"]):
        return (
            f"AGFHC reports {t['passed']} passed / {t['failed']} failed of {t['total']}, but only "
            f"{got_p} passed / {got_f} failed result lines were found in the console log."
        )
    return ""


RESULT_STATE_MAP = {
    "passed": "PASS",
    "pass": "PASS",
    "success": "PASS",
    "ok": "PASS",
    "failed": "FAIL",
    "fail": "FAIL",
    "failure": "FAIL",
    "skipped": "SKIP",
    "skip": "SKIP",
    "not_run": "SKIP",
    "queued": "SKIP",
}


def results_json_tests(text: str) -> dict:
    """Authoritative per-test results from AGFHC results.json.

    Test instances are dicts with "test", "run", "iteration" and "state" (e.g. state "failed", test "hbm_s16_ds",
    run 1, iteration 1 -> instance "hbm_s16_ds-1-1", the same name the console uses). The run summary dict has
    total_selected_tests / total_passed / total_failed / ...  Returns {"tests", "details", "totals"} (empty if unusable).
    """
    try:
        data = json.loads(text)
    except Exception:
        return {"tests": {}, "details": {}, "totals": None}
    tests: dict = {}
    details: dict = {}
    totals = None

    def walk(o):
        nonlocal totals
        if isinstance(o, dict):
            if "total_selected_tests" in o and totals is None:
                g = lambda k: int(o.get(k) or 0)
                totals = {
                    "total": g("total_selected_tests"),
                    "executed": g("total_executed_tests"),
                    "skipped": g("total_skipped"),
                    "passed": g("total_passed"),
                    "failed": g("total_failed"),
                }
            if isinstance(o.get("test"), str) and isinstance(o.get("state"), str):
                name = o["test"]
                if o.get("run") is not None and o.get("iteration") is not None:
                    name = f"{name}-{o['run']}-{o['iteration']}"
                st = RESULT_STATE_MAP.get(o["state"].strip().lower(), "ERROR")
                tests[name] = st
                d = o.get("primary_status_name") or ""
                if d in ("", "TEST_PASS", "PASS", "SUCCESS") or st == "PASS":
                    d = ""
                subj = o.get("primary_action_subject") or ""
                if d and subj:
                    d = f"{d} ({subj})"
                details[name] = {
                    "time": str(o.get("time_total") or ""),
                    "detail": d[:200],
                }
                return
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(data)
    return {"tests": tests, "details": details, "totals": totals}


def merge_results(parsed: dict, results_text: str) -> dict:
    """Prefer results.json per-test data over console parsing; keep console-only extras (config, suspicious)."""
    rj = (
        results_json_tests(results_text)
        if (results_text or "").strip()
        else {"tests": {}}
    )
    if rj["tests"]:
        cd = parsed.get("details", {})
        parsed["details"] = {
            n: {
                "time": cd.get(n, {}).get("time") or rj["details"][n]["time"],
                "detail": rj["details"][n]["detail"] or cd.get(n, {}).get("detail", ""),
            }
            for n in rj["tests"]
        }
        parsed["tests"] = dict(rj["tests"])
        parsed["source"] = "results.json"
    else:
        parsed["source"] = "console.log"
    if not parsed.get("totals") and rj.get("totals"):
        parsed["totals"] = rj["totals"]
    if parsed.get("totals") and not parsed.get("overall"):
        parsed["overall"] = (
            "FAIL"
            if parsed["totals"]["failed"]
            else ("PASS" if parsed["totals"]["passed"] else None)
        )
    return parsed


def parse_gpu_usage(mem_text: str, pids_text: str, threshold: int = 5) -> dict:
    """From `rocm-smi --showmemuse` and `rocm-smi --showpids`: GPUs with VRAM in use and the processes using GPUs."""
    busy, checked = [], 0
    for m in re.finditer(
        r"GPU\[(\d+)\]\s*:\s*GPU Memory Allocated \(VRAM%\)\s*:\s*(\d+)",
        strip_ansi(mem_text),
    ):
        checked += 1
        if int(m.group(2)) >= threshold:
            busy.append((int(m.group(1)), int(m.group(2))))
    procs = []
    for line in strip_ansi(pids_text).splitlines():
        f = line.split()
        if len(f) >= 3 and f[0].isdigit() and not line.lstrip().startswith("GPU"):
            procs.append(" ".join(f[:6]))
    return {"busy": busy, "procs": procs[:12], "checked": checked}


def gpu_busy_warning(u: dict) -> str:
    if not u["busy"] and not u["procs"]:
        return ""
    parts = []
    if u["busy"]:
        parts.append(
            "VRAM in use on " + ", ".join(f"GPU[{g}] {p}%" for g, p in u["busy"])
        )
    if u["procs"]:
        parts.append("GPU processes: " + "; ".join(u["procs"]))
    return (
        "⚠️ GPUs are NOT idle: "
        + " | ".join(parts)
        + ". AGFHC needs idle GPUs; tests that allocate GPU memory "
        "(e.g. hbm_s16_ds 'failed to allocate buffer') will fail. Stop these workloads first (e.g. an inference "
        "container like vLLM) or expect false failures."
    )


def results_json_failures(text: str, limit: int = 3000) -> str:
    """Pull failing entries out of AGFHC results.json without assuming its schema."""
    try:
        data = json.loads(text)
    except Exception:
        return strip_ansi(text)[:limit]
    hits: list = []

    def walk(o):
        if len(hits) >= 20:
            return
        if isinstance(o, dict):
            vals = " ".join(
                str(v) for v in o.values() if isinstance(v, (str, int, float, bool))
            )
            if re.search(r"\bFAIL", vals, re.IGNORECASE):
                hits.append(o)
                return
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(data)
    return json.dumps(hits, indent=1, default=str)[:limit] if hits else ""


def compute_verdict(parsed: dict, rc: Optional[int]) -> str:
    tests = parsed.get("tests", {})
    bad = any(v in ("FAIL", "ERROR") for v in tests.values())
    if parsed.get("overall"):
        return parsed["overall"]
    if rc is None:
        return "RUNNING/UNKNOWN"
    if rc != 0 or bad:
        return "FAIL"
    if tests:
        return "PASS"
    return "UNKNOWN"


def count_statuses(tests: dict) -> dict:
    counts: dict[str, int] = {}
    for s in tests.values():
        counts[s] = counts.get(s, 0) + 1
    return counts


def parse_kv(text: str) -> dict:
    """Parse KEY=value lines (version script output)."""
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            if re.fullmatch(r"[A-Z_]{2,20}", k):
                out[k] = v.strip() or "unknown"
    return out


def format_versions(v: dict) -> str:
    """Markdown table of versions for the chat."""
    rows = ["| Component | Version |", "|---|---|"]
    for key, label in VERSION_KEYS:
        rows.append(f"| {label} | {v.get(key, 'unknown') or 'unknown'} |")
    return "\n".join(rows)


def summarize(
    parsed: dict,
    rc: Optional[int],
    dmesg: str = "",
    budget: int = 6000,
    results_failures: str = "",
) -> str:
    """Render a compact markdown report for the LLM, bounded to `budget` chars."""
    tests = parsed.get("tests", {})
    counts = count_statuses(tests)
    bad = {k: v for k, v in tests.items() if v in ("FAIL", "ERROR")}
    verdict = compute_verdict(parsed, rc)
    if verdict == "UNKNOWN":
        verdict = "UNKNOWN (no test verdicts parsed — check result_regex valve)"
    t = parsed.get("totals")

    out = [
        f"**Verdict:** {verdict}   **exit code:** {rc if rc is not None else 'n/a'}",
        (
            f"**AGFHC totals (authoritative):** {t['total']} total, {t['executed']} executed, {t['skipped']} skipped, "
            f"{t['passed']} passed, {t['failed']} failed"
            if t
            else "**AGFHC totals:** not found in console log"
        ),
        "**Per-test lines parsed:** "
        + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"),
    ]
    warn = totals_mismatch(parsed)
    if warn:
        out.append(f"⚠️ {warn}")
    if parsed.get("config"):
        out.append(f"**AGFHC config:** {parsed['config']}")
    if bad:
        out.append("**Failed / errored tests:**")
        det = parsed.get("details", {})
        out += [
            f"- {k}: {v}"
            + (f" [{det[k]['time']}]" if det.get(k, {}).get("time") else "")
            + (f" {det[k]['detail']}" if det.get(k, {}).get("detail") else "")
            for k, v in bad.items()
        ]
    if results_failures.strip():
        out.append("**AGFHC results.json — failing entries (excerpt):**")
        out.append("```json\n" + results_failures[:2500] + "\n```")
    if parsed.get("suspicious"):
        uniq = list(dict.fromkeys(parsed["suspicious"]))
        out.append(f"**Suspicious log lines ({len(uniq)} unique, first 40):**")
        out.append("```\n" + "\n".join(l[:300] for l in uniq[:40]) + "\n```")
    if dmesg.strip():
        lines = dmesg.strip().splitlines()
        out.append(
            f"**Kernel (amdgpu/RAS/AER) messages during run ({len(lines)}, last 30):**"
        )
        out.append("```\n" + "\n".join(l[:300] for l in lines[-30:]) + "\n```")
    passed = [k for k, v in tests.items() if v == "PASS"]
    if passed:
        out.append("**Passed:** " + ", ".join(passed))
    text = "\n".join(out)
    if len(text) > budget:
        text = text[: budget - 60] + "\n…[truncated to analysis_char_budget]"
    return text


def build_run_cmd(
    template: str, binary: str, recipe: str, run_dir: str, extra_args: str = ""
) -> str:
    extra = " ".join(shlex.quote(a) for a in (extra_args or "").split())
    cmd = template.format(
        bin=shlex.quote(binary),
        recipe=shlex.quote(recipe),
        run_dir=shlex.quote(run_dir),
        extra=extra,
    )
    return re.sub(r"\s+", " ", cmd).strip()


def split_sections(out: str) -> dict:
    """Split '@@NAME' delimited remote output into {NAME: body}."""
    parts: dict[str, str] = {}
    cur = None
    buf: list[str] = []
    for line in out.splitlines():
        m = re.fullmatch(r"@@([A-Z_]+)", line.strip())
        if m:
            if cur:
                parts[cur] = "\n".join(buf)
            cur, buf = m.group(1), []
        elif cur:
            buf.append(line)
    if cur:
        parts[cur] = "\n".join(buf)
    return parts


def new_run_id(host: str, recipe: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = re.sub(r"[^A-Za-z0-9]+", "-", f"{host}-{recipe}").strip("-")[:48]
    return f"{ts}-{safe}-{secrets.token_hex(2)}"


def _tz(name: str):
    if ZoneInfo and name:
        try:
            return ZoneInfo(name)
        except Exception:
            pass
    return timezone.utc


def fmt_epoch(epoch: Optional[int], tzname: str) -> str:
    if not epoch:
        return "n/a"
    dt = datetime.fromtimestamp(int(epoch), _tz(tzname))
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def fmt_duration(start: Optional[int], end: Optional[int]) -> str:
    if not start or not end or end < start:
        return "n/a"
    s = int(end) - int(start)
    return f"{s // 3600}h {(s % 3600) // 60:02d}m {s % 60:02d}s"


def archive_folder_name(start_epoch: Optional[int], fmt: str, tzname: str) -> str:
    """AGFHC_<timestamp> from the run's start time (falls back to now)."""
    dt = (
        datetime.fromtimestamp(int(start_epoch), _tz(tzname))
        if start_epoch
        else datetime.now(_tz(tzname))
    )
    name = dt.strftime(fmt)
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", name)


def human_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024 or unit == "GB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return f"{n} B"


def _inline_md(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return s


def md_to_html(md: str) -> str:
    """Tiny, safe markdown subset: headings, bullets, numbered lists, code fences, bold, code, paragraphs."""
    out: list[str] = []
    para: list[str] = []
    list_tag: Optional[str] = None
    code: Optional[list[str]] = None

    def flush():
        nonlocal para, list_tag
        if para:
            out.append("<p>" + " ".join(_inline_md(p) for p in para) + "</p>")
            para = []
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    for line in (md or "").splitlines():
        if line.strip().startswith("```"):
            if code is None:
                flush()
                code = []
            else:
                out.append("<pre>" + html.escape("\n".join(code)) + "</pre>")
                code = None
            continue
        if code is not None:
            code.append(line)
            continue
        h = re.match(r"^\s*(#{1,4})\s+(.*)", line)
        ul = re.match(r"^\s*[-*]\s+(.*)", line)
        ol = re.match(r"^\s*\d+[.)]\s+(.*)", line)
        if h:
            flush()
            lvl = min(len(h.group(1)) + 2, 6)
            out.append(f"<h{lvl}>{_inline_md(h.group(2))}</h{lvl}>")
        elif ul or ol:
            tag = "ul" if ul else "ol"
            if para:
                out.append("<p>" + " ".join(_inline_md(p) for p in para) + "</p>")
                para = []
            if list_tag != tag:
                if list_tag:
                    out.append(f"</{list_tag}>")
                out.append(f"<{tag}>")
                list_tag = tag
            out.append(f"<li>{_inline_md((ul or ol).group(1))}</li>")
        elif not line.strip():
            flush()
        else:
            if list_tag:
                out.append(f"</{list_tag}>")
                list_tag = None
            para.append(line.strip())
    if code is not None:
        out.append("<pre>" + html.escape("\n".join(code)) + "</pre>")
    flush()
    return "\n".join(out)


REPORT_CSS = """
:root{--fg:#1d232b;--muted:#5d6876;--line:#dfe3e8;--bg:#f6f7f9;--card:#fff;
--pass:#1f7a3f;--pass-bg:#e5f4ea;--fail:#b42318;--fail-bg:#fde8e6;--warn:#8a5a00;--warn-bg:#fff3d6;--skip:#5d6876;--skip-bg:#eef0f3}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 48px}
header{display:flex;flex-wrap:wrap;gap:16px;align-items:center;justify-content:space-between;margin-bottom:20px}
h1{font-size:22px;margin:0}
.sub{color:var(--muted);font-size:13px;margin-top:2px}
.badge{display:inline-block;font-weight:700;letter-spacing:.04em;border-radius:6px;padding:8px 18px;font-size:18px}
.v-PASS{background:var(--pass-bg);color:var(--pass)} .v-FAIL,.v-ERROR{background:var(--fail-bg);color:var(--fail)}
.v-other{background:var(--warn-bg);color:var(--warn)}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 20px}
.chip{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 16px;min-width:110px}
.chip b{display:block;font-size:22px;line-height:1.2}
.chip span{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media (max-width:760px){.grid{grid-template-columns:1fr}}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:16px}
h2{font-size:15px;margin:0 0 10px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600;width:34%}
table.results th,table.inv th{width:auto}
table.inv td:first-child{width:40%;color:var(--fg)}
tr.mism td{background:var(--fail-bg)}
section h3{font-size:13.5px;margin:16px 0 4px}
section h3:first-child{margin-top:0}
td code,p code,li code{background:var(--bg);padding:1px 5px;border-radius:4px;font-size:12.5px}
.st{font-weight:700;font-size:12px;border-radius:4px;padding:2px 8px}
.st-PASS{background:var(--pass-bg);color:var(--pass)} .st-FAIL,.st-ERROR{background:var(--fail-bg);color:var(--fail)}
.st-SKIP{background:var(--skip-bg);color:var(--skip)}
pre{background:#11161c;color:#e6e9ee;padding:12px;border-radius:8px;overflow:auto;font-size:12.5px;line-height:1.45;max-height:420px;white-space:pre-wrap;word-break:break-word}
details summary{cursor:pointer;color:var(--muted);margin:4px 0}
.muted{color:var(--muted)}
.analysis h3,.analysis h4{margin:14px 0 6px}
a{color:#1f5fbf}
footer{color:var(--muted);font-size:12px;margin-top:24px}
@media print{body{background:#fff}section{break-inside:avoid}pre{max-height:none}}
"""


def render_report_html(s: dict) -> str:
    """Render the final HTML report from a summary dict (see _build_summary)."""
    e = html.escape
    verdict = s.get("verdict", "UNKNOWN")
    vclass = f"v-{verdict}" if verdict in ("PASS", "FAIL", "ERROR") else "v-other"
    tz = s.get("timezone", "UTC")
    tests = s.get("tests", {})
    counts = s.get("counts", {})

    def row(k, v, code=False):
        val = f"<code>{e(str(v))}</code>" if code else e(str(v))
        return f"<tr><th>{e(k)}</th><td>{val}</td></tr>"

    run_rows = "".join(
        [
            row("DUT", f"{s.get('host')} ({s.get('address', '?')})"),
            row("Platform", s.get("platform", "?")),
            row("Recipe", s.get("recipe", "?"), True),
            row("Command", s.get("cmd", "?"), True),
            row("Run ID", s.get("run_id", "?"), True),
            row("Operator", s.get("operator") or "n/a"),
            row("Start", fmt_epoch(s.get("start_epoch"), tz)),
            row("End", fmt_epoch(s.get("end_epoch"), tz)),
            row("Duration", fmt_duration(s.get("start_epoch"), s.get("end_epoch"))),
            row("Exit code", s.get("rc") if s.get("rc") is not None else "n/a"),
            row("AGFHC config", s.get("agfhc_config") or "n/a"),
            row("AGFHC log folder (DUT)", s.get("agfhc_logdir") or "n/a", True),
            row(
                "Live session (DUT tmux)", s.get("tmux_session") or "none (nohup)", True
            ),
        ]
    )
    vers = s.get("versions", {})
    ver_rows = "".join(
        row(label, vers.get(key, "unknown") or "unknown") for key, label in VERSION_KEYS
    )

    tot = s.get("agfhc_totals")
    if tot:  # AGFHC's own "Tests: N Total … | P Passed, F Failed" line is authoritative
        chips = [
            f'<div class="chip"><b>{tot["total"]}</b><span>Tests</span></div>',
            f'<div class="chip"><b class="st-PASS" style="background:none">{tot["passed"]}</b><span>Passed</span></div>',
            f'<div class="chip"><b class="st-FAIL" style="background:none">{tot["failed"]}</b><span>Failed</span></div>',
        ]
        if tot.get("skipped"):
            chips.append(
                f'<div class="chip"><b class="st-SKIP" style="background:none">{tot["skipped"]}</b><span>Skipped</span></div>'
            )
    else:
        chips = [
            f'<div class="chip"><b>{len(tests)}</b><span>Tests parsed</span></div>'
        ]
        for st in ("PASS", "FAIL", "ERROR", "SKIP"):
            if counts.get(st):
                chips.append(
                    f'<div class="chip"><b class="st-{st}" style="background:none">{counts[st]}</b><span>{st}</span></div>'
                )
    chips.append(
        f'<div class="chip"><b>{s.get("kernel_new_total", len(s.get("kernel_new", [])))}</b>'
        f"<span>New kernel msgs</span></div>"
    )
    inv_groups = aggregate_inventory(s.get("inventory", []))
    n_mism = sum(1 for _, its in inv_groups for it in its if it["mismatch"])
    if n_mism:
        chips.append(
            f'<div class="chip"><b class="st-FAIL" style="background:none">{n_mism}</b><span>Version mismatches</span></div>'
        )

    inv_parts = []
    mism_tag = ' <span class="st st-FAIL">MISMATCH</span>'
    for label, items in inv_groups:
        trs = "".join(
            f'<tr{" class=mism" if it["mismatch"] else ""}><td>{e(it["component"])}</td>'
            f'<td><code>{e(it["value"])}</code>{mism_tag if it["mismatch"] else ""}</td></tr>'
            for it in items
        )
        tbl = f'<table class="inv"><tr><th>Component</th><th>Version</th></tr>{trs}</table>'
        if label.startswith("Installed"):
            inv_parts.append(
                f"<details><summary><b>{e(label)}</b> ({len(items)})</summary>{tbl}</details>"
            )
        else:
            inv_parts.append(f"<h3>{e(label)}</h3>{tbl}")
    inv_html = (
        "".join(inv_parts)
        + f'<p class="muted">Inventory {e(s.get("inventory_note", ""))}.</p>'
        if inv_parts
        else '<p class="muted">Inventory not available for this run.</p>'
    )

    order = {"FAIL": 0, "ERROR": 1, "SKIP": 2, "PASS": 3}
    if tests:
        det = s.get("test_details", {})
        trs = "".join(
            f'<tr><td><code>{e(n)}</code></td><td><span class="st st-{e(st)}">{e(st)}</span></td>'
            f'<td>{e(det.get(n, {}).get("time", ""))}</td>'
            f'<td>{"<code>" + e(det[n]["detail"]) + "</code>" if det.get(n, {}).get("detail") else ""}</td></tr>'
            for n, st in sorted(
                tests.items(), key=lambda kv: (order.get(kv[1], 9), kv[0])
            )
        )
        results = (
            f'<table class="results"><tr><th>Test</th><th>Result</th><th>Time (elapsed/limit)</th>'
            f"<th>Detail</th></tr>{trs}</table>"
        )
    else:
        results = (
            '<p class="muted">No per-test verdicts were parsed from the console log. '
            "Check the <code>result_regex</code> valve against <code>logs/console.log</code>.</p>"
        )

    analysis = s.get("analysis") or ""
    analysis_html = (
        f'<section class="analysis"><h2>Analysis</h2>{md_to_html(analysis)}</section>'
        if analysis.strip()
        else ""
    )

    kern = s.get("kernel_new", [])
    k_total = s.get("kernel_new_total", len(kern))
    k_note = (
        f'<p class="muted">Showing the last {len(kern)} of {k_total}. All of them are in '
        f"<code>logs/dmesg_new.log</code>.</p>"
        if k_total > len(kern)
        else ""
    )
    kern_html = (
        f"{k_note}<pre>{e(chr(10).join(kern))}</pre>"
        if kern
        else '<p class="muted">No new amdgpu / RAS / AER / XGMI / MCE kernel messages during the run.</p>'
    )
    lw = s.get("agfhc_logs_warning", "")
    logs_warn_html = (
        f'<section style="border-color:var(--fail)"><b style="color:var(--fail)">⚠ {e(lw)}</b></section>'
        if lw
        else ""
    )
    warn = s.get("totals_warning", "")
    warn_html = (
        f'<p class="st st-FAIL" style="display:block;padding:8px 10px">⚠ {e(warn)}</p>'
        if warn
        else ""
    )
    rj = s.get("results_failures", "")
    rj_html = (
        f"<section><h2>AGFHC results.json — failing entries</h2><pre>{e(rj)}</pre></section>"
        if rj
        else ""
    )
    susp = s.get("suspicious", [])
    susp_html = (
        f"<details{' open' if len(susp) <= 15 else ''}><summary>{len(susp)} lines</summary>"
        f"<pre>{e(chr(10).join(susp[:300]))}</pre></details>"
        if susp
        else '<p class="muted">None.</p>'
    )

    files = s.get("files", [])
    if files:
        lis = "".join(
            f'<tr><td><a href="{e(f["path"])}">{e(f["path"])}</a></td><td>{e(human_bytes(f.get("size", 0)))}</td></tr>'
            for f in files[:500]
        )
        files_html = (
            f'<table class="results"><tr><th>File</th><th>Size</th></tr>{lis}</table>'
        )
    else:
        files_html = '<p class="muted">No files listed.</p>'

    title = f"AGFHC Report — {s.get('host', '?')} — {s.get('recipe', '?')}"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)}</title><style>{REPORT_CSS}</style></head>
<body><div class="wrap">
<header><div><h1>AGFHC Test Report</h1>
<div class="sub">{e(s.get('host', '?'))} · {e(s.get('recipe', '?'))} · {e(fmt_epoch(s.get('start_epoch'), tz))}</div>
<div class="sub">Archive: <code>{e(s.get('nfs_url', ''))}</code></div></div>
<div class="badge {vclass}">{e(verdict)}</div></header>
<div class="chips">{''.join(chips)}</div>
{logs_warn_html}
{analysis_html}
<div class="grid">
<section><h2>Run</h2><table>{run_rows}</table></section>
<section><h2>Key versions</h2><table>{ver_rows}</table></section>
</div>
<section><h2>Test results</h2>{warn_html}{results}<p class="muted">Source: {e(s.get('results_source', 'console.log'))}</p></section>
{rj_html}
<section><h2>Driver, firmware &amp; tool versions</h2>{inv_html}</section>
<section><h2>New kernel messages during run</h2>{kern_html}</section>
<section><h2>Suspicious log lines</h2>{susp_html}</section>
<section><h2>Archived logs</h2>{files_html}</section>
<footer>Generated {e(s.get('generated', ''))} by AGFHC Test Runner v{e(s.get('tool_version', TOOL_VERSION))} (Open WebUI).</footer>
</div></body></html>
"""


def fmt_secs(sec: Optional[int]) -> str:
    if sec is None or sec < 0:
        return "?"
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


PLANNED_TEST_RE = re.compile(
    r"^(?P<name>[A-Za-z][\w.\-]*)\s{2,}(?P<desc>.+?)\s{2,}(?P<iters>\d+)\s+iterations?\s+(?P<limit>\d+:\d{2}:\d{2})\s*$"
)


def parse_planned(text: str) -> list:
    """AGFHC's test table printed at the start of a run:
    'pcie_link_status   PCIe Link Status          1 iteration                   0:00:08'
    """
    out, seen = [], set()
    for l in strip_ansi(text).splitlines():
        m = PLANNED_TEST_RE.match(l.strip())
        if m and m.group("name") not in seen:
            seen.add(m.group("name"))
            out.append(
                {
                    "name": m.group("name"),
                    "desc": m.group("desc").strip(),
                    "iters": int(m.group("iters")),
                    "limit": m.group("limit"),
                }
            )
    return out


def _hms_to_s(t: str) -> int:
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def format_planned(tests: list) -> str:
    """Markdown table of a recipe's tests with the summed time limit."""
    if not tests:
        return ""
    rows = [
        "| # | Test | Description | Iterations | Time limit |",
        "|---|---|---|---|---|",
    ]
    rows += [
        f"| {i} | `{t['name']}` | {t['desc']} | {t['iters']} | {t['limit']} |"
        for i, t in enumerate(tests, 1)
    ]
    total = sum(_hms_to_s(t["limit"]) for t in tests)
    rows.append(
        f"\n**{len(tests)} tests**, combined time limit ≈ {fmt_secs(total)} (actual runs are usually shorter)."
    )
    return "\n".join(rows)


def progress_from_log(text: str) -> dict:
    """Live progress from a (partial) AGFHC console log.

    AGFHC redraws its EXEC ticker with carriage returns, so one physical line can hold many ticks followed by the
    final PASS/FAIL; splitting on \\r as well as \\n (str.splitlines does both) gives one record per tick.
    Returns {"done": [(status, test, time, detail), ...] in finish order, "planned": int|None,
             "current": "test [elapsed/limit]" or "", "totals": "Tests: … Failed" or ""}.
    """
    clean = strip_ansi(text)
    parsed = parse_agfhc_log(clean)
    done = [
        (
            st,
            n,
            parsed["details"].get(n, {}).get("time", ""),
            parsed["details"].get(n, {}).get("detail", ""),
        )
        for n, st in parsed["tests"].items()
    ]
    planned_tests = parse_planned(clean)
    planned = len(planned_tests) or None
    current = ""
    finished = set(parsed["tests"])
    for l in clean.splitlines():
        m = re.match(r"^\s*EXEC\s*:\s*(\S+)\s*(\[[^\]]*\])?", l)
        if m:
            current = (
                ""
                if m.group(1) in finished
                else f"{m.group(1)} {m.group(2) or ''}".strip()
            )
    m_t = re.search(r"Tests:.*?Failed", clean)
    return {
        "done": done,
        "planned": planned,
        "planned_tests": planned_tests,
        "current": current,
        "totals": m_t.group(0) if m_t else "",
    }


def format_progress(
    state: str,
    elapsed: Optional[int],
    prog: dict,
    tzname: str,
    rc: Optional[str] = None,
) -> str:
    """Headline for one poll, e.g.
    '- 10:24:05 · RUNNING · 3m00s elapsed · 3/6 done · 3 PASS / 0 FAIL · now: hbm_bw-1-1 [00:00:21/00:01:05]'
    """
    now = datetime.now(_tz(tzname)).strftime("%H:%M:%S")
    done = prog.get("done", [])
    npass = sum(1 for d in done if d[0] == "PASS")
    nfail = sum(1 for d in done if d[0] in ("FAIL", "ERROR"))
    parts = [
        state if state == "RUNNING" else f"**{state}**",
        f"{fmt_secs(elapsed)} elapsed",
    ]
    if prog.get("planned") and prog["planned"] >= len(done):
        parts.append(f"{len(done)}/{prog['planned']} done")
    elif done:
        parts.append(f"{len(done)} done")
    parts.append(f"{npass} PASS / {nfail} FAIL")
    if state == "RUNNING" and prog.get("current"):
        parts.append(f"now: `{prog['current'][:90]}`")
    if state == "FINISHED":
        parts.append(f"exit code {rc if rc is not None else '?'}")
        if prog.get("totals"):
            parts.append(prog["totals"])
    return f"- {now} · " + " · ".join(parts)


def format_test_line(d: tuple) -> str:
    st, name, t, detail = d
    icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "❌", "SKIP": "⏭"}.get(st, "•")
    return (
        f"  - {icon} {st} `{name}`"
        + (f" [{t}]" if t else "")
        + (f" {detail}" if detail else "")
    )


def plain(md: str) -> str:
    """Markdown progress line -> plain text for the Open WebUI status line."""
    return re.sub(r"[*`]", "", md).strip().lstrip("-").strip()


def _to_int(x) -> Optional[int]:
    x = (str(x) if x is not None else "").strip()
    return int(x) if x.lstrip("-").isdigit() else None


def local_root_from_mounts(
    mounts_text: str, nfs_addresses: set, nfs_logs_dir: str
) -> str:
    """Map '<nfs host>:<nfs_logs_dir>' to a local path using /proc/mounts lines like
    '10.46.252.71:/srv/nfs /mnt/logs nfs4 rw,...'. The most specific export wins. Returns '' if none covers it.
    """
    want = nfs_logs_dir.rstrip("/")
    best_len, best = -1, ""
    for line in mounts_text.splitlines():
        f = line.split()
        if len(f) < 3 or not f[2].startswith("nfs") or ":" not in f[0]:
            continue
        host, export = f[0].rsplit(":", 1)
        export = export.rstrip("/")
        if host.strip("[]") not in nfs_addresses:
            continue
        if want != export and not want.startswith(export + "/"):
            continue
        if len(export) > best_len:
            best_len, best = (
                len(export),
                f[1].replace("\\040", " ").rstrip("/") + want[len(export) :],
            )
    return best


# Files worth reading first when a test fails (AGFHC v1.30 layout under agfhc_logs/agfhc_<ts>/).
KEY_LOG_FILES = (
    "error.json",
    "results.json",
    "journal.log",
    "dmesg_full.log",
    "console.log",
    "system_monitor.csv",
)


def failure_file_map(
    files: list, failed_tests: list, local_dir: str, limit: int = 40
) -> str:
    """Markdown list of exact local paths the Log Analyzer should open for this run's failures."""
    paths = [f["path"] for f in files]
    out = []
    key = [
        p
        for p in paths
        if "/agfhc_logs/" in p and p.rsplit("/", 1)[-1] in KEY_LOG_FILES
    ]
    if key:
        out.append("AGFHC run files:")
        out += [f"- `{local_dir}/{p}`" for p in key[:12]]
    for t in failed_tests:
        tp = [p for p in paths if f"/tests/{t}/" in p]
        if tp:
            out.append(f"Files of failed test `{t}`:")
            out += [f"- `{local_dir}/{p}`" for p in tp[:limit]]
    out.append(
        f"Run console: `{local_dir}/logs/console.log`, new kernel messages: `{local_dir}/logs/dmesg_new.log`"
    )
    return "\n".join(out)


SAFE_USER = re.compile(r"^[a-z_][a-z0-9_.\-]{0,31}$")


def ip_allowed(address: str, cidrs: str) -> bool:
    import ipaddress

    try:
        ip = ipaddress.ip_address(address.strip())
    except ValueError:
        return False
    for c in (x.strip() for x in (cidrs or "").split(",")):
        if c:
            try:
                if ip in ipaddress.ip_network(c, strict=False):
                    return True
            except ValueError:
                continue
    return False


def platform_from_model(model: str) -> str:
    m = re.search(r"MI\s?(\d{3}[A-Z]?)", model or "", re.IGNORECASE)
    return f"mi{m.group(1).lower()}" if m else "unknown"


def sudoers_line(user: str, agfhc_bin: str) -> str:
    return f"{user} ALL=(root) NOPASSWD: {agfhc_bin}, /usr/bin/dmesg, /usr/bin/pkill, /usr/bin/tar, /usr/bin/ipmitool"


def redact(text: str, secret: str) -> str:
    return text.replace(secret, "******") if secret else text


def tmux_session_name(run_id: str) -> str:
    """agfhc-<HHMMSS UTC>-<4 hex>, e.g. agfhc-174708-ab58 (tmux names can't contain '.' or ':')."""
    hhmmss = run_id[9:15] if len(run_id) > 15 and run_id[8] == "T" else "run"
    return re.sub(r"[^A-Za-z0-9_-]", "-", f"agfhc-{hhmmss}-{run_id[-4:]}")


def attach_commands(user: str, address: str, session: str) -> str:
    return (
        f"watch live (read-only): `ssh -t {user}@{address} tmux attach -r -t {session}`  ·  "
        f"interactive: `ssh -t {user}@{address} tmux attach -t {session}` (Ctrl-b d to detach; "
        f"Ctrl-C there would STOP the test)"
    )


def _uv_get(user: Optional[dict], key: str, default: Any = None) -> Any:
    """__user__['valves'] may be a pydantic object, a dict, or missing."""
    if not user:
        return default
    uv = user.get("valves")
    if uv is None:
        return default
    if isinstance(uv, dict):
        return uv.get(key, default)
    return getattr(uv, key, default)
