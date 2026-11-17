"""AGENTS.md rules that plain code can check on a diff. A problem here blocks the push; the model must fix it.

Rule 1 read-only via the catalog: runner.run("x") must name a command in gpuval/commands.py (tests/test_catalog.py
        then proves every catalog command is read-only; that file is protected).
Rule 2 no pinned versions: no version-like literals added to tests/ or config/platforms/.
Rule 3 no node names or IPs anywhere in added lines.
Rule 4 recordings and the known-findings wiring are protected; new skips only for "recording lacks it, re-capture".
Rule 5 no weakening: existing platform expectations are not changed, asserts are not removed, no xfail added.
"""
import re
from dataclasses import dataclass, field

IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
VERSION = re.compile(r"""["'](?:v|V)?\d+\.\d+(?:\.\d+)+(?:[-+.~]\w+)*["']""")
RUN_NAME = re.compile(r"""\brun\(\s*["']([\w-]+)["']""")
SKIP = re.compile(r"pytest\.skip\(|mark\.skip\b|mark\.skipif\b|importorskip\(")
XFAIL = re.compile(r"\bxfail\b")
FIELD = re.compile(r"""\[\s*["']([A-Za-z_][\w]*)["']\s*\]|\.get\(\s*["']([A-Za-z_][\w]*)["']""")
CMD_KEY = re.compile(r"""^\s*["']([\w-]+)["']\s*:\s*["']""", re.M)


@dataclass
class Report:
    problems: list = field(default_factory=list)       # block the push
    new_commands: list = field(default_factory=list)   # catalog additions: the recording must be re-captured
    new_tests: list = field(default_factory=list)
    files: list = field(default_factory=list)
    unknown_fields: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.problems


def parse_diff(text: str) -> dict:
    """{path: {'added': [(line_no, text)], 'removed': [text], 'new': bool}}"""
    files, cur, n, is_new = {}, None, 0, False
    for line in text.splitlines():
        if line.startswith("diff --git "):
            cur, is_new = None, False
        elif cur is None and line.startswith("--- "):
            is_new = line.strip() == "--- /dev/null"
        elif cur is None and line.startswith("+++ "):
            path = line[4:].strip()
            path = path[2:] if path.startswith("b/") else path
            cur = files.setdefault(path, {"added": [], "removed": [], "new": is_new})
        elif cur is None and line.startswith("new file mode"):
            is_new = True
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            n = int(m.group(1)) if m else 0
        elif cur is not None:
            if line.startswith("+"):
                cur["added"].append((n, line[1:]))
                n += 1
            elif line.startswith("-"):
                cur["removed"].append(line[1:])
            elif not line.startswith("\\"):
                n += 1
    return files


def _yaml_keys(text: str) -> set:
    import yaml
    out = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                out.add(str(k))
                walk(v)
    try:
        walk(yaml.safe_load(text) or {})
    except yaml.YAMLError:
        pass
    return out


def changed_expectations(old_text: str, new_text: str) -> list:
    """Keys of a platform YAML whose value changed or vanished (new keys are fine). Formatting and comments ignored."""
    import yaml
    try:
        old, new = yaml.safe_load(old_text) or {}, yaml.safe_load(new_text) or {}
    except yaml.YAMLError as e:
        return [f"not valid YAML any more: {e}"]
    out = []

    def walk(o, n, path):
        if isinstance(o, dict):
            for k, v in o.items():
                if not isinstance(n, dict) or k not in n:
                    out.append(f"{path}{k} removed")
                else:
                    walk(v, n[k], f"{path}{k}.")
        elif o != n:
            out.append(f"{path.rstrip('.')}: {o!r} -> {n!r}")
    walk(old, new, "")
    return out


def check(diff_text: str, cfg: dict, catalog_text: str, protected, versions=None, known_fields=None) -> Report:
    """protected: callable(path) -> bool (Workspace.protected).
    versions: optional callable(path) -> (old_text, new_text), used to compare platform YAML by value.
    known_fields: optional set of every key in the recording and the platform config; a test that reads a field
    no recording has (and adds no new command to get it) checks invented data: it can never fail."""
    rep = Report()
    files = parse_diff(diff_text)
    rep.files = sorted(files)
    names = [re.compile(rf"\b{re.escape(n)}\b", re.I) for n in cfg.get("forbidden_names", [])]
    catalog = set(CMD_KEY.findall(catalog_text))

    for path, ch in files.items():
        if protected(path):
            rep.problems.append(f"{path}: protected path changed (rule 1/4: engineers own it)")
        for no, text in ch["added"]:
            where = f"{path}:{no}"
            if IPV4.search(text):
                rep.problems.append(f"{where}: IP address in the suite (rule 3): {text.strip()[:120]}")
            for rx in names:
                if rx.search(text):
                    rep.problems.append(f"{where}: node/host name in the suite (rule 3): {text.strip()[:120]}")
            if path.startswith(("tests/", "config/platforms/")) and VERSION.search(text):
                rep.problems.append(f"{where}: looks like a pinned version (rule 2): {text.strip()[:120]}")
            if path.startswith("tests/") or path.startswith("gpuval/"):
                for name in RUN_NAME.findall(text):
                    if name not in catalog:
                        rep.problems.append(f"{where}: runner.run('{name}') is not in gpuval/commands.py (rule 1)")
                if XFAIL.search(text):
                    rep.problems.append(f"{where}: xfail added; only an engineer marks known findings (rule 4/5)")
                if SKIP.search(text) and not re.search(r"captur|recording", text, re.I):
                    rep.problems.append(
                        f"{where}: remove this skip: `{text.strip()[:120]}`. A test may only skip when runner.run "
                        f"returns rc 127 (the recording lacks the command), with a message containing 're-capture "
                        f"with tools/capture.py'. If the check depends on the hardware layout (sockets, NUMA nodes, "
                        f"GPU count), put the expected value in config/platforms/<platform>.yaml as a NEW key and "
                        f"assert against it instead of skipping (rule 4)")
            if known_fields is not None and path.startswith("tests/") and not path.endswith("test_checks_catch.py"):
                for m in FIELD.finditer(text):
                    key = m.group(1) or m.group(2)
                    if key not in known_fields:
                        rep.unknown_fields.append((where, key))
            if path.startswith("tests/") and re.match(r"\s*def (test_\w+)", text):
                rep.new_tests.append(f"{path}::{re.match(r'\s*def (test_\w+)', text).group(1)}")
        if path.startswith("tests/") and not ch["new"]:
            gone = sum(1 for t in ch["removed"] if re.match(r"\s*assert\b", t))
            came = sum(1 for _, t in ch["added"] if re.match(r"\s*assert\b", t))
            if gone > came:
                rep.problems.append(f"{path}: {gone - came} assert(s) removed from an existing test (rule 5)")
        if path.startswith("config/platforms/") and not ch["new"]:
            if versions is not None:
                changed = changed_expectations(*versions(path))
            else:
                changed = [t.strip() for t in ch["removed"] if t.strip() and not t.strip().startswith("#")]
            if changed:
                rep.problems.append(f"{path}: existing expectation changed (rule 5): {'; '.join(changed)[:200]}. "
                                    f"Add new keys only; keep every existing line.")
        if path == "gpuval/commands.py":
            before = set(CMD_KEY.findall("\n".join(ch["removed"])))
            after = set(CMD_KEY.findall("\n".join(t for _, t in ch["added"])))
            rep.new_commands += sorted(after - before)
            for gone in sorted(before - after):
                rep.problems.append(f"gpuval/commands.py: command '{gone}' removed or renamed")
    if rep.unknown_fields and not rep.new_commands:
        for where, key in rep.unknown_fields:
            rep.problems.append(f"{where}: field '{key}' does not exist in the recording (see the recorded data "
                                f"list). A check on a missing field can never fail. Use a field the recording has, "
                                f"or add a read-only command that returns this data (then the test skips until a "
                                f"re-capture)")
    return rep
