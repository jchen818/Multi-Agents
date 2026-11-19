#!/usr/bin/env python3
"""Check every node in config/nodes.yaml the way the agents reach it. Read-only; prints no secrets.

    .venv/bin/python scripts/check_nodes.py              # all nodes
    .venv/bin/python scripts/check_nodes.py mi350x-4     # one node

Per node: the config itself (keys, typos, no passwords in a file that goes to GitHub), SSH as the agents run it
(BatchMode, strict host keys, through ProxyJump if ~/.ssh/config says so), ROCm / amdgpu / GPU count, the IPMI
account in secrets/ipmi.yaml (a sensor read at USER privilege, and that it is NOT allowed to change power), and what
the agents can do with it (AGFHC plan template, gpu-validation platform file).
"""
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
KNOWN = {"ssh_host", "ssh_user", "bmc_host", "platform", "test_node", "sim"}
SECRET_WORDS = re.compile(r"pass|secret|token|key", re.I)
OK, BAD, WARN = "ok  ", "FAIL", "warn"


def ssh(n: dict, cmd: str, timeout=25):
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes",
            f"{n['ssh_user']}@{n['ssh_host']}", "--", "bash -c " + shlex.quote(cmd)]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except FileNotFoundError:
        return 127, "", "ssh not installed on this host"


def ipmi(n: dict, cred: dict, *args, timeout=30):
    env = {"PATH": "/usr/bin:/bin", "IPMI_PASSWORD": str(cred.get("ipmi_password", ""))}
    try:
        p = subprocess.run(["ipmitool", "-I", "lanplus", "-H", n["bmc_host"], "-U", str(cred.get("ipmi_user", "")),
                            "-E", *args], capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except FileNotFoundError:
        return 127, "", "ipmitool not installed on this host"


def check(name: str, n: dict, creds: dict) -> list:
    out = []

    def say(level, what):
        out.append((level, what))

    # --- config
    unknown = set(n) - KNOWN
    for k in sorted(unknown):
        if SECRET_WORDS.search(k) or k in ("bmc_user", "ipmi_user"):
            say(BAD, f"config: '{k}' does not belong in nodes.yaml (it goes to GitHub); put credentials in secrets/ipmi.yaml")
        else:
            say(BAD, f"config: unknown key '{k}' (typo? known: {', '.join(sorted(KNOWN))})")
    for k in ("ssh_host", "ssh_user", "bmc_host"):
        if not n.get(k):
            say(BAD, f"config: missing {k}")
    platform = n.get("platform") or "mi325x"
    if "platform" not in n:
        say(WARN, "config: no platform; the agents assume mi325x")
    if not n.get("ssh_host") or not n.get("ssh_user"):
        return out

    # --- ssh, the way the agents connect
    rc, o, e = ssh(n, "hostname")
    if rc != 0:
        say(BAD, f"ssh {n['ssh_user']}@{n['ssh_host']}: {(e or o)[:200]}")
        return out
    say(OK, f"ssh as the agents run it -> {o}")
    rc, o, _ = ssh(n, "cat /opt/rocm/.info/version 2>/dev/null")
    if o and re.match(r"^\d+\.\d+", o):
        say(OK, f"ROCm {o}")
    else:
        rc2, o2, _ = ssh(n, "ls -d /opt/rocm* 2>/dev/null | tr '\\n' ' '; echo '|'; ls /opt/rocm/.info 2>/dev/null | tr '\\n' ' '")
        dirs, info = (o2.split("|") + [""])[:2]
        say(WARN, f"no /opt/rocm/.info/version (ROCm dirs: {dirs.strip() or 'none'}; .info has: "
                  f"{info.strip() or 'nothing'}): the agents read the ROCm version from that file")
    rc, o, _ = ssh(n, "cat /sys/module/amdgpu/version 2>/dev/null || echo none")
    say(OK if o != "none" else WARN, f"amdgpu driver {o if o != 'none' else 'not loaded'}")
    rc, o, _ = ssh(n, "PATH=$PATH:/opt/rocm/bin; amd-smi list 2>/dev/null | grep -c '^GPU:' || true")
    say(OK if o not in ("", "0") else WARN, f"GPUs seen by amd-smi: {o or 0}")
    rc, o, _ = ssh(n, "sudo -n true 2>/dev/null && echo yes || echo no")
    if n.get("test_node"):
        say(OK if o == "yes" else BAD, f"passwordless sudo (AGFHC needs it on a test node): {o}")

    # --- IPMI with the agents' account
    cred = creds.get(name)
    if not cred:
        say(WARN, "IPMI: no entry in secrets/ipmi.yaml (Lab Ops IPMI tools will refuse this node)")
    elif str(cred.get("ipmi_user", "")).lower() in ("root", "admin", "administrator"):
        say(BAD, f"IPMI: secrets/ipmi.yaml uses '{cred['ipmi_user']}': use a dedicated USER-privilege account")
    else:
        rc, o, e = ipmi(n, cred, "sdr", "type", "Temperature")
        if rc == 0 and o:
            say(OK, f"IPMI read as {cred['ipmi_user']}: {len(o.splitlines())} temperature sensors")
        else:
            say(BAD, f"IPMI read as {cred['ipmi_user']} on {n['bmc_host']}: {(e or o or 'no output')[:200]}")
        rc, o, e = ipmi(n, cred, "channel", "getaccess", "1")
        block = next((b for b in re.split(r"(?m)^(?=User ID\s*:)", o or "")
                      if re.search(rf"User Name\s*:\s*{re.escape(str(cred['ipmi_user']))}\s*$", b, re.M)), "")
        m = re.search(r"Privilege Level\s*:\s*(\S+)", block)
        if m:
            lvl = m.group(1).upper()
            say(OK if lvl == "USER" else WARN, f"IPMI privilege of the agents' account: {lvl}"
                + ("" if lvl == "USER" else " (USER is enough for reads and cannot change power)"))

    # --- what the agents can do with it
    tpl = ROOT / "config" / "plan_templates" / f"{platform}.yaml"
    say(OK if tpl.exists() else WARN, f"AGFHC plan template for {platform}: " +
        ("present" if tpl.exists() else "missing (the planner refuses this node until it exists)"))
    if n.get("test_node"):
        say(WARN, "test_node: true -> AGFHC may run here in live mode; make sure nothing else uses these GPUs")
    return out


def main():
    cfg = yaml.safe_load(open(ROOT / "config" / "nodes.yaml")) or {}
    nodes = cfg.get("nodes") or {}
    sec = ROOT / "secrets" / "ipmi.yaml"
    creds = (yaml.safe_load(open(sec)) or {}) if sec.exists() else {}
    if sec.exists() and oct(sec.stat().st_mode)[-3:] != "600":
        print(f"{WARN} secrets/ipmi.yaml is mode {oct(sec.stat().st_mode)[-3:]}: chmod 600 it")
    raw = (ROOT / "config" / "nodes.yaml").read_text()
    if re.search(r"(?im)^\s*\w*(pass|secret|token)\w*\s*:", raw):
        print(f"{BAD} config/nodes.yaml contains a password-like key: remove it (the repo is public)")
    names = sys.argv[1:] or sorted(nodes)
    worst = 0
    for name in names:
        if name not in nodes:
            print(f"\n{name}: not in config/nodes.yaml")
            worst = 1
            continue
        print(f"\n== {name}")
        for level, what in check(name, nodes[name], creds):
            print(f"  {level} {what}")
            worst = max(worst, 1 if level == BAD else 0)
    sys.exit(worst)


if __name__ == "__main__":
    main()
