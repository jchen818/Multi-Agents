"""agfhc: AGFHC plan / approve / run / analyze / archive as a stdio MCP server (Phase 2).

Ported from the Open WebUI "AGFHC Test Runner" v0.4.3 (johnc, MiTAC); pure helpers live in agfhc_lib.py.
Differences from the Open WebUI tool:
  * Dry-run by default (AGFHC_MODE=dryrun): only the simulated node "sim" can run. It executes the real
    launch / collect / archive shell scripts locally against sim/bin/agfhc (authentic AGFHC 1.30 output).
  * Live mode (AGFHC_MODE=live): only nodes with `test_node: true` in config/nodes.yaml can run.
  * Busy GPUs, an active run, an unknown recipe or a non-test node are hard refusals (not warnings).
  * system ssh with BatchMode + StrictHostKeyChecking (not paramiko AutoAddPolicy).
  * Not exposed: register_dut (installs keys/sudoers from a chat password) and stop (operator-only).
  * Kill switch: state/KILL refuses start_run.
"""
import asyncio
import getpass
import json
import os
import re
import secrets
import shlex
import subprocess
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import yaml
from mcp.server.fastmcp import FastMCP

import agfhc_lib as L

VERSION = "0.5.0-mcp"
ROOT = Path(os.environ.get("MULTI_AGENTS_ROOT", "/home/mitac/Multi-Agents"))
MODE = os.environ.get("AGFHC_MODE", "dryrun").strip().lower()
STATE = Path(os.environ.get("AGFHC_STATE_DIR", str(ROOT / "state")))
SIM_DIR = STATE / "sim"
TOKEN_TTL_S = int(os.environ.get("AGFHC_TOKEN_TTL_S", "1800"))   # long enough for a human approval
USE_TMUX = os.environ.get("AGFHC_TMUX", "1") == "1"
TZ = os.environ.get("AGFHC_TZ", "America/Los_Angeles")
KILL_FILE = Path(os.environ.get("AGENTS_KILL_FILE", str(ROOT / "state/KILL")))
_TOKENS: dict = {}


# ------------------------------------------------------------------ configuration

@lru_cache
def _template(platform: str) -> dict:
    p = ROOT / "config" / "plan_templates" / f"{platform}.yaml"
    if not p.exists():
        raise ValueError(f"no plan template for platform '{platform}' ({p})")
    return yaml.safe_load(open(p))


def _config() -> dict:
    return yaml.safe_load(open(ROOT / "config" / "nodes.yaml")) or {}


def _nodes() -> dict:
    nodes = dict(_config().get("nodes") or {})
    if MODE == "dryrun":
        nodes["sim"] = {"ssh_host": "localhost", "ssh_user": getpass.getuser(), "platform": "mi325x", "sim": True}
    return nodes


def _node(name: str) -> dict:
    nodes = _nodes()
    if name not in nodes:
        raise ValueError(f"unknown node '{name}'. Configured: {', '.join(sorted(nodes))}")
    return nodes[name]


def _runnable(name: str) -> str:
    """'' if AGFHC may run on this node in the current mode, else the reason."""
    n = _node(name)
    if MODE == "dryrun":
        return "" if n.get("sim") else (f"dry-run mode: only the simulated node 'sim' can run AGFHC "
                                         f"('{name}' is a real machine)")
    if MODE != "live":
        return f"unknown AGFHC_MODE '{MODE}'"
    if n.get("sim"):
        return "the simulated node is not available in live mode"
    if n.get("test_node") is not True:
        return f"'{name}' is not marked `test_node: true` in config/nodes.yaml"
    return ""


def _run_root(name: str) -> str:
    n = _node(name)
    if n.get("sim"):
        return str(SIM_DIR / "agfhc-runs")
    return _template(n.get("platform", "mi325x"))["agfhc"].get("run_root", L.RUN_ROOT).rstrip("/")


def _agfhc(name: str) -> dict:
    """AGFHC paths for a node: template values, overridden for the simulated node."""
    n = _node(name)
    a = dict(_template(n.get("platform", "mi325x"))["agfhc"])
    if n.get("sim"):
        a.update(bin=str(ROOT / "sim/bin/agfhc"), log_root=str(SIM_DIR / "rootlog"), sudo="")
    else:
        a["sudo"] = "sudo -n "
    return a


def _archive_target(name: str) -> dict:
    """{'node': archive node name, 'dir': base dir, 'address': for display}."""
    if _node(name).get("sim"):
        return {"node": "sim", "dir": str(SIM_DIR / "archive"), "address": "localhost (dry-run)"}
    arch = _config().get("archive") or {}
    if not arch.get("node") or not arch.get("dir"):
        raise RuntimeError("live archiving needs `archive: {node: <name in nodes>, dir: /srv/nfs/logs}` in config/nodes.yaml")
    return {"node": arch["node"], "dir": arch["dir"].rstrip("/"), "address": _node(arch["node"]).get("ssh_host")}


def _sim_env() -> dict:
    env = dict(os.environ)
    env.update(PATH=f"{ROOT / 'sim/bin'}:{env.get('PATH', '/usr/bin:/bin')}", SIM_DATA=str(ROOT / "sim/data"),
               FAKE_ROOTLOG=str(SIM_DIR / "rootlog"))
    return env


def _sim_exports() -> str:
    """Exports written into run.sh so a run started inside an existing tmux server still sees the simulator."""
    e = _sim_env()
    keys = ("PATH", "SIM_DATA", "FAKE_ROOTLOG", "SIM_SCENARIO", "SIM_SLEEP", "SIM_GPU_BUSY")
    return "".join(f"export {k}={shlex.quote(e[k])}\n" for k in keys if k in e)


# ------------------------------------------------------------------ execution

def _argv(name: str, script: str) -> list:
    n = _node(name)
    if n.get("sim"):
        return ["bash", "-c", script]
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes",
            f"{n['ssh_user']}@{n['ssh_host']}", "--", "bash -c " + shlex.quote(script)]


def _exec(name: str, script: str, timeout: int = 120, stdin: bytes = None):
    env = _sim_env() if _node(name).get("sim") else None
    try:
        p = subprocess.run(_argv(name, script), input=stdin, capture_output=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    return p.returncode, p.stdout.decode(errors="replace"), p.stderr.decode(errors="replace")


def _stream_copy(src: str, src_cmd: str, dst: str, dst_cmd: str):
    """Pipe src_cmd's stdout (on src) into dst_cmd (on dst) through this host. Returns (src_rc, dst_rc, bytes, err)."""
    s = subprocess.Popen(_argv(src, src_cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env=_sim_env() if _node(src).get("sim") else None)
    d = subprocess.Popen(_argv(dst, dst_cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env=_sim_env() if _node(dst).get("sim") else None)
    total = 0
    while True:
        chunk = s.stdout.read(262144)
        if not chunk:
            break
        d.stdin.write(chunk)
        total += len(chunk)
    d.stdin.close()
    s_rc, d_rc = s.wait(), d.wait()
    return s_rc, d_rc, total, (s.stderr.read() + d.stderr.read()).decode(errors="replace")


def _write_file(name: str, path: str, data: bytes):
    rc, _, err = _exec(name, f"cat > {shlex.quote(path)}", stdin=data)
    if rc != 0:
        raise RuntimeError(f"write {path} on {name} failed: {err.strip()[:300]}")


def _read_file(name: str, path: str) -> bytes:
    rc, out, err = _exec(name, f"cat {shlex.quote(path)}")
    if rc != 0:
        raise RuntimeError(f"read {path} on {name} failed: {err.strip()[:300]}")
    return out.encode()


# ------------------------------------------------------------------ remote scripts (ported from _Core)

def _version_script(name: str) -> str:
    a = _agfhc(name)
    ver = a["version_cmd"].format(bin=shlex.quote(a["bin"]))
    return (
        "export PATH=$PATH:/opt/rocm/bin; "
        f"echo \"AGFHC=$({a['sudo']}{ver} 2>&1 | grep -m1 . | tr -d '\\r' || echo unknown)\"; "
        "echo \"ROCM=$(cat /opt/rocm/.info/version 2>/dev/null || amd-smi version 2>/dev/null | grep -oE 'ROCm version: [0-9.]+' | awk '{print $3}' || echo unknown)\"; "
        'echo "AMDGPU=$(cat /sys/module/amdgpu/version 2>/dev/null || amd-smi version 2>/dev/null | grep -oE \'amdgpu version: [0-9.]+\' | awk \'{print $3}\' || echo unknown)"; '
        "echo \"AMDSMI=$(amd-smi version 2>/dev/null | head -1 | tr -d '\\r' || echo unknown)\"; "
        'echo "KERNEL=$(uname -r)"; '
        'echo "OS=$(. /etc/os-release 2>/dev/null; echo ${PRETTY_NAME:-unknown})"; '
        "echo \"GPU_MODEL=$(amd-smi static --asic 2>/dev/null | grep -m1 -i market_name | cut -d: -f2- | sed 's/^ *//' || echo unknown)\"; "
        "echo \"GPU_COUNT=$(amd-smi list 2>/dev/null | grep -c '^GPU' || echo unknown)\"; "
        "echo \"VBIOS=$(amd-smi static --vbios 2>/dev/null | grep -iE '^ *VERSION:' | awk '{print $2}' | sort -u | paste -sd, - || echo unknown)\"; "
        'echo "HOSTNAME=$(hostname)"'
    )


def _inventory_script(name: str) -> str:
    a = _agfhc(name)
    ver = a["version_cmd"].format(bin=shlex.quote(a["bin"]))
    return L.INVENTORY_SCRIPT.replace("__SUDO__", a["sudo"]).replace("__AGFHC_VER__", a["sudo"] + ver)


def _gpu_usage(name: str) -> dict:
    script = ("export PATH=$PATH:/opt/rocm/bin; echo '@@MEM'; rocm-smi --showmemuse 2>/dev/null; "
              "echo '@@PIDS'; rocm-smi --showpids 2>/dev/null")
    _, out, _ = _exec(name, script)
    p = L.split_sections(out)
    return L.parse_gpu_usage(p.get("MEM", ""), p.get("PIDS", ""), 5)


def _collect_logdir_sh(name: str) -> str:
    """Copy AGFHC's own log folder into agfhc_logs/ (path from console.log, else by start time). Ported unchanged."""
    a = _agfhc(name)
    s, root = a["sudo"], shlex.quote(a["log_root"].rstrip("/"))
    grab = (f'{s}tar -C "$(dirname "$ld")" -cf - "$(basename "$ld")" 2>>agfhc_collect.err '
            "| tar -C agfhc_logs -xf - 2>>agfhc_collect.err")
    return (
        "mkdir -p agfhc_logs; : > agfhc_collect.err; "
        "ld=$(sed 's/\\x1b\\[[0-9;]*m//g' console.log 2>/dev/null | tr '\\r' '\\n' "
        "| grep -oE '/[^ ]+/agfhc_[0-9]{8}-[0-9]{6}' | head -1); "
        'if [ -z "$ld" ]; then st=$(cat start_epoch 2>/dev/null || echo 0); '
        "  for i in $(seq -5 120); do n=agfhc_$(date -d @$((st + i)) +%Y%m%d-%H%M%S); ld=" + root + "/$n; "
        '    [ -e "agfhc_logs/$n" ] && break; ' + grab + '; [ -d "agfhc_logs/$n" ] && break; ld=\'\'; done; '
        'elif [ ! -e "agfhc_logs/$(basename "$ld")" ]; then ' + grab + "; fi; "
        'if [ -n "$ld" ] && [ -d "agfhc_logs/$(basename "$ld")" ]; then echo "OK $ld" > agfhc_logdir.txt; '
        'else echo "MISSING path=${ld:-not-found}" > agfhc_logdir.txt; fi; '
    )


def _recipe_tests(name: str, recipe: str) -> dict:
    """Tests of a recipe before running it: the last run's planned table on this node, else the recipe file."""
    q, a, rr = shlex.quote, _agfhc(name), _run_root(name)
    script = (
        f"for d in $(ls -1dt {q(rr)}/*/ 2>/dev/null); do "
        f"grep -q '\"recipe\": \"'{q(recipe)}'\"' \"$d/manifest.json\" 2>/dev/null || continue; "
        '[ -s "$d/console.log" ] || continue; echo \'@@FROM\'; basename "$d"; echo \'@@BANNER\'; '
        "sed 's/\\x1b\\[[0-9;]*m//g' \"$d/console.log\" | tr '\\r' '\\n' | grep -v '^EXEC' | head -150; break; done; "
        f"echo '@@RECIPEFILE'; f=$(find {q(os.path.dirname(a['bin']))} -maxdepth 6 -type f "
        f"\\( -name {q(recipe + '.yml')} -o -name {q(recipe + '.yaml')} \\) 2>/dev/null | head -1); "
        '[ -n "$f" ] && { echo "$f"; head -c 4000 "$f"; }'
    )
    _, out, _ = _exec(name, script)
    parts = L.split_sections(out)
    tests = L.parse_planned(parts.get("BANNER", ""))
    if tests:
        return {"tests": tests, "source": f"the last `{recipe}` run on {name} (`{parts.get('FROM', '').strip()}`)", "raw": ""}
    raw = parts.get("RECIPEFILE", "").strip()
    return {"tests": [], "source": "recipe file" if raw else "", "raw": raw}


def _recipe_tests_md(name: str, recipe: str) -> str:
    r = _recipe_tests(name, recipe)
    if r["tests"]:
        return f"**Tests in `{recipe}`** (from {r['source']}):\n\n{L.format_planned(r['tests'])}"
    if r["raw"]:
        return f"**Tests in `{recipe}`** (recipe file, no previous run here):\n```\n{r['raw'][:3000]}\n```"
    return (f"**Tests in `{recipe}`**: not known yet (no previous run on {name}, no recipe file found). "
            "AGFHC prints its test table when the run starts.")


def _check_recipe(name: str, recipe: str, extra_args: str):
    if not L.SAFE_NAME.match(recipe or ""):
        raise ValueError(f"recipe '{recipe}' contains disallowed characters")
    allowed = _template(_node(name).get("platform", "mi325x")).get("allowed_recipes") or []
    if recipe not in allowed:
        raise ValueError(f"recipe '{recipe}' is not in the plan template allowlist ({len(allowed)} recipes)")
    if not L.SAFE_ARGS.match(extra_args or ""):
        raise ValueError("extra_args contains disallowed characters")


def _progress(name: str, run_id: str) -> dict:
    d = shlex.quote(f"{_run_root(name)}/{run_id}")
    script = (f"cd {d} || exit 4; s=$(cat start_epoch 2>/dev/null || date +%s); e=$(cat end_epoch 2>/dev/null || date +%s); "
              "echo '@@ELAPSED'; echo $((e - s)); echo '@@STATE'; "
              'if [ -f rc ]; then echo FINISHED; elif kill -0 "$(cat pid 2>/dev/null)" 2>/dev/null; then echo RUNNING; '
              "else echo STOPPED_UNEXPECTEDLY; fi; echo '@@RC'; cat rc 2>/dev/null; "
              "echo '@@LOG'; tail -c 300000 console.log 2>/dev/null")
    rc, out, _ = _exec(name, script)
    if rc == 4:
        raise FileNotFoundError(f"no run {run_id} on {name}")
    parts = {"ELAPSED": "", "STATE": "", "RC": "", "LOG": ""}
    chunks = re.split(r"(?m)^@@(ELAPSED|STATE|RC|LOG)\n", out)
    for i in range(1, len(chunks) - 1, 2):
        parts[chunks[i]] = chunks[i + 1]
    return {"state": parts["STATE"].strip() or "?", "elapsed": L._to_int(parts["ELAPSED"]),
            "rc": parts["RC"].strip() or None, "prog": L.progress_from_log(parts["LOG"])}


def _collect(name: str, run_id: str) -> dict:
    d = shlex.quote(f"{_run_root(name)}/{run_id}")
    script = (
        f"cd {d} || exit 4; [ -f rc ] && {{ {_collect_logdir_sh(name)} }} ; "
        "echo '@@LOGDIR'; cat agfhc_logdir.txt 2>/dev/null; echo '@@TMUX'; cat tmux_session 2>/dev/null; "
        "echo '@@RC'; cat rc 2>/dev/null; echo '@@MANIFEST'; cat manifest.json 2>/dev/null; "
        "echo '@@RESULTS'; f=$(find agfhc_logs -name results.json 2>/dev/null | head -1); "
        '[ -n "$f" ] && head -c 300000 "$f"; echo; '
        "echo '@@VERSIONS'; cat versions.txt 2>/dev/null; echo '@@INVENTORY'; cat inventory.tsv 2>/dev/null; "
        "echo '@@START'; cat start_epoch 2>/dev/null; echo '@@END'; cat end_epoch 2>/dev/null; "
        "echo '@@ARCHIVE'; cat archive_path 2>/dev/null; echo '@@LOG'; tail -c 400000 console.log 2>/dev/null; "
        "echo '@@DMESGN'; wc -l < dmesg_new.log 2>/dev/null || echo 0; echo '@@DMESG'; tail -n 2000 dmesg_new.log 2>/dev/null"
    )
    rc, out, _ = _exec(name, script, timeout=180)
    if rc == 4:
        raise FileNotFoundError(f"no run {run_id} on {name}")
    return L.split_sections(out)


def _build_summary(name: str, run_id: str, parts: dict) -> dict:
    i = L._to_int
    try:
        manifest = json.loads(parts.get("MANIFEST", "") or "{}")
    except json.JSONDecodeError:
        manifest = {}
    rc = i(parts.get("RC"))
    parsed = L.merge_results(L.parse_agfhc_log(parts.get("LOG", "")), parts.get("RESULTS", ""))
    tests = parsed["tests"]
    logdir = (parts.get("LOGDIR") or "").strip()
    kernel = [l for l in parts.get("DMESG", "").splitlines() if l.strip()]
    return {
        "tool_version": VERSION, "run_id": run_id, "host": name, "mode": MODE,
        "address": manifest.get("address", _node(name).get("ssh_host")), "platform": manifest.get("platform", "?"),
        "recipe": manifest.get("recipe", "?"), "cmd": manifest.get("cmd", "?"), "operator": manifest.get("operator", ""),
        "start_epoch": i(parts.get("START")), "end_epoch": i(parts.get("END")), "rc": rc,
        "verdict": L.compute_verdict(parsed, rc), "counts": L.count_statuses(tests), "tests": tests,
        "failed": [k for k, v in tests.items() if v in ("FAIL", "ERROR")],
        "suspicious": list(dict.fromkeys(parsed["suspicious"]))[:300],
        "kernel_new": kernel[-300:], "kernel_new_total": i(parts.get("DMESGN")) or len(kernel),
        "test_details": parsed.get("details", {}), "results_source": parsed.get("source", "console.log"),
        "tmux_session": (parts.get("TMUX") or "").strip(), "agfhc_totals": parsed.get("totals"),
        "totals_warning": L.totals_mismatch(parsed), "agfhc_config": parsed.get("config", ""),
        "agfhc_logdir": logdir[3:] if logdir.startswith("OK ") else parsed.get("logdir", ""),
        "agfhc_logs_warning": (f"AGFHC's own log folder was NOT collected ({logdir[8:]})." if logdir.startswith("MISSING") else ""),
        "results_failures": L.results_json_failures(parts["RESULTS"]) if parts.get("RESULTS", "").strip() else "",
        "versions": L.parse_kv(parts.get("VERSIONS", "")), "timezone": TZ,
        "inventory": L.parse_inventory(parts.get("INVENTORY", "")), "inventory_note": "collected at run start",
        "analysis": "", "files": [], "archive_dir": (parts.get("ARCHIVE") or "").strip(), "nfs_url": "",
        "generated": datetime.now(L._tz(TZ)).strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


def _archive(name: str, run_id: str, summary: dict) -> str:
    """Create AGFHC_<ts> on the archive target, copy the run dir, write summary.json + report.html."""
    t = _archive_target(name)
    q, base, rr = shlex.quote, t["dir"], _run_root(name)
    existing = summary.get("archive_dir", "")
    if existing and L.SAFE_PATH.match(existing):
        d, copy_logs = existing, False
        try:
            old = json.loads(_read_file(t["node"], f"{d}/summary.json"))
            summary["analysis"] = summary.get("analysis") or old.get("analysis", "")
        except Exception:
            pass
    else:
        nm = L.archive_folder_name(summary.get("start_epoch"), "AGFHC_%Y%m%d_%H%M%S", TZ)
        alt = f"{nm}_{re.sub(r'[^A-Za-z0-9]+', '-', name)}_{run_id[-4:]}"
        script = (f'base={q(base)}; mkdir -p "$base" 2>/dev/null; [ -d "$base" ] && [ -w "$base" ] || {{ echo NOBASE; exit 5; }}; '
                  f'd="$base"/{q(nm)}; if [ -e "$d" ] && [ "$(cat "$d/.run_id" 2>/dev/null)" != {q(run_id)} ]; then d="$base"/{q(alt)}; fi; '
                  f'mkdir -p "$d/logs" && echo {q(run_id)} > "$d/.run_id" && echo "$d"')
        rc, out, err = _exec(t["node"], script)
        if rc == 5:
            raise RuntimeError(f"{base} does not exist or is not writable on {t['node']}")
        if rc != 0 or not out.strip():
            raise RuntimeError(f"could not create archive folder: {err.strip() or out.strip()}")
        d, copy_logs = out.strip().splitlines()[-1], True
        _exec(name, f"echo {q(d)} > {q(rr + '/' + run_id)}/archive_path")
    summary["archive_dir"], summary["nfs_url"] = d, f"{t['address']}:{d}/"
    note = "logs already archived"
    if copy_logs:
        s_rc, d_rc, n, err = _stream_copy(
            name, f"tar -C {q(rr + '/' + run_id)} --exclude=./.lock --exclude=./pid --exclude=./archive_path -cf - .",
            t["node"], f"tar -C {q(d)}/logs -xf -")
        if d_rc != 0:
            raise RuntimeError(f"log copy failed (src rc={s_rc}, dst rc={d_rc}): {err.strip()[:300]}")
        note = f"copied {L.human_bytes(n)} of logs"
    _, out, _ = _exec(t["node"], f"cd {q(d)} && find logs -type f -printf '%p\\t%s\\n' | sort | head -500")
    summary["files"] = [{"path": p, "size": int(sz)} for p, sz in (l.split("\t", 1) for l in out.splitlines() if "\t" in l)
                        if sz.strip().isdigit()]
    _write_file(t["node"], f"{d}/summary.json", json.dumps(summary, indent=2).encode())
    _write_file(t["node"], f"{d}/report.html", L.render_report_html(summary).encode())
    return note


def _banner() -> str:
    return ("**DRY-RUN** (simulated node, no GPU touched)" if MODE == "dryrun" else "**LIVE** (real AGFHC on a test node)")


# ------------------------------------------------------------------ MCP tools

mcp = FastMCP("agfhc")


@mcp.tool()
def list_test_nodes() -> str:
    """List nodes and whether AGFHC may run on each in the current mode (dry-run: only 'sim')."""
    rows = [f"agfhc server v{VERSION} · mode {_banner()}"]
    for name, n in sorted(_nodes().items()):
        why = _runnable(name)
        rows.append(f"- **{name}** ({n.get('ssh_host')}, platform {n.get('platform', '?')}): "
                    + ("can run AGFHC" if not why else f"read-only ({why})"))
    return "\n".join(rows)


@mcp.tool()
async def preflight(node: str) -> str:
    """Read a node's versions (AGFHC, ROCm, amdgpu, kernel, GPU model/count, VBIOS), GPU idle state, active run lock
    and recent amdgpu/RAS/AER kernel errors. Read-only; run it before planning."""
    try:
        n, rr = _node(node), _run_root(node)
    except ValueError as e:
        return f"REFUSED: {e}"
    a = _agfhc(node)
    script = (f"echo '@@VERSIONS'; {_version_script(node)}; echo '@@ACTIVE'; cat {shlex.quote(rr)}/.host.lock 2>/dev/null || echo none; "
              f"echo '@@KERNEL'; {a['sudo']}dmesg --level=err,crit,alert,emerg 2>/dev/null | grep -iwE '{L.DMESG_FILTER}' | tail -10 || true")
    rc, out, err = await asyncio.to_thread(_exec, node, script)
    if rc == 255:
        return f"ERROR: cannot reach {node}: {err.strip()[:300]}"
    p = L.split_sections(out)
    usage = await asyncio.to_thread(_gpu_usage, node)
    busy = L.gpu_busy_warning(usage)
    idle = (f"all {usage['checked']} GPUs idle" if usage["checked"] and not busy else
            busy or "GPU usage unknown (no rocm-smi data)")
    why = _runnable(node)
    return (f"### Preflight: {node} · {_banner()}\n**Can run AGFHC:** {'yes' if not why else 'no - ' + why}\n"
            f"**GPU usage:** {idle}\n\n{L.format_versions(L.parse_kv(p.get('VERSIONS', '')))}\n\n"
            f"**Active run:** {p.get('ACTIVE', '').strip() or 'none'}\n"
            f"**Recent amdgpu/RAS/AER kernel errors:**\n```\n{p.get('KERNEL', '').strip() or 'none'}\n```")


@mcp.tool()
async def list_recipe_tests(node: str, recipe: str) -> str:
    """List the tests in an AGFHC recipe (name, description, iterations, time limit) without running it."""
    try:
        _check_recipe(node, recipe, "")
    except ValueError as e:
        return f"REFUSED: {e}"
    return await asyncio.to_thread(_recipe_tests_md, node, recipe)


@mcp.tool()
async def plan_run(node: str, recipe: str, extra_args: str = "", operator: str = "") -> str:
    """Prepare (do NOT start) an AGFHC run: checks preconditions, reads live versions and the recipe's tests, and returns
    a confirm_token. Show the plan to the user and call start_run only after the user approves."""
    try:
        why = _runnable(node)
        if why:
            return f"REFUSED: {why}"
        _check_recipe(node, recipe, extra_args)
    except ValueError as e:
        return f"REFUSED: {e}"
    rr, a = _run_root(node), _agfhc(node)
    _, lock, _ = await asyncio.to_thread(_exec, node, f"cat {shlex.quote(rr)}/.host.lock 2>/dev/null")
    if lock.strip():
        return f"REFUSED: {node} already has an active AGFHC run ({lock.strip()})"
    usage = await asyncio.to_thread(_gpu_usage, node)
    busy = L.gpu_busy_warning(usage)
    if busy:
        return f"REFUSED: {busy}"
    rc, out, err = await asyncio.to_thread(_exec, node, _version_script(node))
    if rc == 255:
        return f"ERROR: cannot reach {node}: {err.strip()[:300]}"
    versions = L.parse_kv(out)
    tests_md = await asyncio.to_thread(_recipe_tests_md, node, recipe)
    run_id = L.new_run_id(node, recipe)
    cmd = L.build_run_cmd(a["run_cmd"], a["bin"], recipe, f"{rr}/{run_id}", extra_args)
    token = secrets.token_urlsafe(8)
    _TOKENS[token] = {"node": node, "recipe": recipe, "cmd": cmd, "run_id": run_id, "operator": operator,
                      "exp": time.time() + TOKEN_TTL_S}
    t = _archive_target(node)
    return (f"**Planned AGFHC run (not started)** · {_banner()}\n\n**Versions on {node} now:**\n{L.format_versions(versions)}\n\n"
            f"{tests_md}\n\n**GPU usage:** all {usage['checked']} GPUs idle\n"
            f"- recipe: `{recipe}`\n- run_id: `{run_id}`\n- command: `{a['sudo']}{cmd}`\n"
            f"- archive: `{t['address']}:{t['dir']}/AGFHC_<start time>/`\n"
            f"- confirm_token: `{token}` (single use, expires in {TOKEN_TTL_S // 60} min)\n\n"
            "Show this plan to the user and start only after explicit approval.")


@mcp.tool()
async def start_run(confirm_token: str) -> str:
    """Start a planned AGFHC run in the background. Requires the confirm_token from plan_run AND the user's explicit
    approval. Returns the run_id at once; poll with get_run_status, then analyze_run when FINISHED."""
    if KILL_FILE.exists():
        return "REFUSED: agents are stopped (kill switch is set)"
    plan = _TOKENS.pop(confirm_token, None)
    if not plan or plan["exp"] < time.time():
        return "REFUSED: confirm_token is unknown, used or expired. Call plan_run again."
    node, run_id = plan["node"], plan["run_id"]
    why = _runnable(node)
    if why:
        return f"REFUSED: {why}"
    busy = L.gpu_busy_warning(await asyncio.to_thread(_gpu_usage, node))
    if busy:
        return f"REFUSED (re-checked at start): {busy}"
    n, a, rr = _node(node), _agfhc(node), _run_root(node)
    q, s, d = shlex.quote, a["sudo"], f"{rr}/{run_id}"
    manifest = json.dumps({"run_id": run_id, "host": node, "address": n.get("ssh_host"), "platform": n.get("platform", "?"),
                           "recipe": plan["recipe"], "cmd": plan["cmd"], "operator": plan["operator"], "mode": MODE,
                           "started_utc": datetime.now(timezone.utc).isoformat()})
    cmd_str = f"{s}{plan['cmd']}"
    tsess = L.tmux_session_name(run_id)
    post = (f"{s}dmesg 2>/dev/null | grep -iwE '{L.DMESG_FILTER}' > dmesg_after.log\n"
            f'awk -v t0="$(cat start_uptime 2>/dev/null || echo 0)" {q(L.DMESG_NEW_AWK)} dmesg_after.log > dmesg_new.log\n'
            f"{_collect_logdir_sh(node)}\nrm -f {q(rr)}/.host.lock .lock\n")
    run_sh = ("#!/bin/bash\n" + (_sim_exports() if n.get("sim") else "") + f"cd {q(d)} || exit 1\necho $$ > pid\n"
              'if [ -n "$AGFHC_TMUX" ] && command -v script >/dev/null 2>&1; then\n'
              f"  script -q -f -e -c {q(cmd_str)} console.log\n  rc=$?\nelse\n  {cmd_str} > console.log 2>&1\n  rc=$?\nfi\n"
              "echo $rc > rc; date +%s > end_epoch\n" + post)
    launch = (f"printf '%s' {q(run_sh)} > {q(d)}/run.sh; chmod 700 {q(d)}/run.sh; "
              f"if [ {'1' if USE_TMUX else '0'} = 1 ] && command -v tmux >/dev/null 2>&1 && "
              f"tmux new-session -d -s {q(tsess)} -x 200 -y 50 {q('AGFHC_TMUX=1 bash ' + d + '/run.sh')}; then "
              f"echo {q(tsess)} > {q(d)}/tmux_session; for i in $(seq 50); do [ -s {q(d)}/pid ] && break; sleep 0.1; done; "
              f"echo STARTED_TMUX; else nohup bash {q(d)}/run.sh > /dev/null 2>&1 & sleep 0.2; echo STARTED; fi")
    script = (f"set -e; mkdir -p {q(rr)} {q(d)}/agfhc_logs; "
              f"if [ -e {q(rr)}/.host.lock ]; then echo BUSY $(cat {q(rr)}/.host.lock); exit 3; fi; "
              f"echo {q(run_id)} > {q(rr)}/.host.lock; touch {q(d)}/.lock; echo {q(manifest)} > {q(d)}/manifest.json; "
              f"date +%s > {q(d)}/start_epoch; {{ {_version_script(node)}; }} > {q(d)}/versions.txt 2>/dev/null || true; "
              f"{{ {_inventory_script(node)}\n}} > {q(d)}/inventory.tsv 2>/dev/null || true; "
              f"cut -d' ' -f1 /proc/uptime > {q(d)}/start_uptime; "
              f"{s}dmesg 2>/dev/null | grep -iwE '{L.DMESG_FILTER}' > {q(d)}/dmesg_before.log || true; " + launch)
    rc, out, err = await asyncio.to_thread(_exec, node, script)
    if rc == 3:
        return f"NOT STARTED: {node} already has an active AGFHC run ({out.strip()})"
    if rc != 0:
        return f"ERROR: launch failed (rc={rc}): {(out + err).strip()[:600]}"
    live = (f"\nLive view (read-only): `ssh -t {n['ssh_user']}@{n['ssh_host']} tmux attach -r -t {tsess}`"
            if "STARTED_TMUX" in out and not n.get("sim") else
            (f"\nLive view: `tmux attach -r -t {tsess}` on nfs-server" if "STARTED_TMUX" in out else ""))
    return f"STARTED · {_banner()} · run_id=`{run_id}` on {node}.{live}\nPoll get_run_status(node, run_id) until FINISHED, then analyze_run."


@mcp.tool()
async def get_run_status(node: str, run_id: str) -> str:
    """One progress snapshot of a run: state, elapsed time, tests done / planned, PASS/FAIL so far, current test."""
    if not L.SAFE_NAME.match(run_id or ""):
        return "REFUSED: invalid run_id"
    try:
        snap = await asyncio.to_thread(_progress, node, run_id)
    except (FileNotFoundError, ValueError) as e:
        return f"REFUSED: {e}"
    head = L.format_progress(snap["state"], snap["elapsed"], snap["prog"], TZ, snap["rc"])
    lines = [L.format_test_line(d) for d in snap["prog"]["done"]]
    nxt = ("Next: analyze_run." if snap["state"] == "FINISHED" else
           "Still running; poll again in about a minute." if snap["state"] == "RUNNING" else
           "The run ended without an exit code; call analyze_run to see what was logged.")
    return f"state={snap['state']}\n{head}\n" + "\n".join(lines) + f"\n{nxt}"


@mcp.tool()
async def analyze_run(node: str, run_id: str) -> str:
    """Collect and condense a run's results (verdict, per-test results, failure details, new kernel messages,
    versions). If the run is finished, also archive logs + summary.json + report.html. Then write an analysis and
    call finalize_report."""
    if not L.SAFE_NAME.match(run_id or ""):
        return "REFUSED: invalid run_id"
    try:
        parts = await asyncio.to_thread(_collect, node, run_id)
    except (FileNotFoundError, ValueError) as e:
        return f"REFUSED: {e}"
    summary = _build_summary(node, run_id, parts)
    parsed = L.merge_results(L.parse_agfhc_log(parts.get("LOG", "")), parts.get("RESULTS", ""))
    report = L.summarize(parsed, summary["rc"], parts.get("DMESG", ""), 6000, summary.get("results_failures", ""))
    if summary["rc"] is None:
        note = "Run not finished yet; logs are archived when analyze_run is called after it finishes."
    else:
        try:
            copied = await asyncio.to_thread(_archive, node, run_id, summary)
            note = f"**Archived** ({copied}) to `{summary['nfs_url']}`\n**Report:** `{summary['nfs_url']}report.html`"
        except Exception as e:
            note = f"WARNING: archiving failed: {e}"
    mism = L.inventory_mismatches(summary.get("inventory", []))
    inv = (f"**Inventory:** {len(summary.get('inventory', []))} items"
           + (f"; version MISMATCH: {'; '.join(mism[:10])}" if mism else "; no mismatches across GPUs/devices"))
    if summary.get("agfhc_logs_warning"):
        inv += f"\nWARNING: {summary['agfhc_logs_warning']}"
    return (f"### AGFHC run {run_id} on {node} · {_banner()}\nRecipe: `{summary['recipe']}` · Duration: "
            f"{L.fmt_duration(summary['start_epoch'], summary['end_epoch'])}\n\n**Versions during run:**\n"
            f"{L.format_versions(summary['versions'])}\n{inv}\n\n{report}\n\n{note}")


@mcp.tool()
async def finalize_report(node: str, run_id: str, analysis: str) -> str:
    """Add the agent's written analysis (markdown) to the archived report.html. Call after analyze_run archived the run."""
    if not L.SAFE_NAME.match(run_id or ""):
        return "REFUSED: invalid run_id"
    try:
        _, out, _ = await asyncio.to_thread(_exec, node, f"cat {shlex.quote(_run_root(node) + '/' + run_id)}/archive_path 2>/dev/null")
        t = _archive_target(node)
    except (ValueError, RuntimeError) as e:
        return f"REFUSED: {e}"
    d = out.strip()
    if not d or not L.SAFE_PATH.match(d):
        return "REFUSED: run not archived yet; call analyze_run after it finishes."
    summary = json.loads(await asyncio.to_thread(_read_file, t["node"], f"{d}/summary.json"))
    summary["analysis"] = (analysis or "")[:20000]
    summary["generated"] = datetime.now(L._tz(TZ)).strftime("%Y-%m-%d %H:%M:%S %Z")
    await asyncio.to_thread(_write_file, t["node"], f"{d}/summary.json", json.dumps(summary, indent=2).encode())
    await asyncio.to_thread(_write_file, t["node"], f"{d}/report.html", L.render_report_html(summary).encode())
    return f"Report updated with analysis: `{summary.get('nfs_url', d + '/')}report.html`"


if __name__ == "__main__":
    mcp.run()   # stdio
