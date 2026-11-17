"""The coding agent's only view of the repo: one git worktree. Paths are checked; nothing outside it is reachable.

The model gets list_files / read_file / search / write_file / git_diff / run_tests / declare_finding. It never gets a
shell, git, or the network. Writes are refused for protected paths (config/coder.yaml) and non-text files.
"""
import fnmatch
import json
import re
import subprocess
from pathlib import Path

from langchain_core.tools import StructuredTool

from agents.coder.sandbox import run_pytest

READ_LIMIT = 60_000


class Refused(Exception):
    pass


class Workspace:
    def __init__(self, root: Path, cfg: dict, runner=run_pytest):
        self.root = Path(root).resolve()
        self.cfg = cfg
        self.runner = runner
        self.findings: dict = {}           # test node id -> why the model believes the failure is real
        self.last_tests = None

    # ---------------------------------------------------------------- paths
    def path(self, rel: str) -> Path:
        rel = (rel or "").strip().lstrip("/")
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise Refused(f"{rel!r} is outside the repository")
        if ".git" in p.relative_to(self.root).parts:
            raise Refused("the .git directory is off limits")
        return p

    def rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix()

    def protected(self, rel: str) -> bool:
        for pat in self.cfg.get("protected", []):
            if pat.endswith("/") and (rel + "/").startswith(pat):
                return True
            if rel == pat or fnmatch.fnmatch(rel, pat):
                return True
        return False

    # ---------------------------------------------------------------- tools
    def list_files(self, directory: str = "") -> str:
        """List the files under a directory of the repository (default: everything except recordings/out)."""
        base = self.path(directory)
        if not base.is_dir():
            return f"ERROR: {directory!r} is not a directory"
        out = []
        for p in sorted(base.rglob("*")):
            r = self.rel(p)
            if p.is_dir() or ".git" in p.parts or "__pycache__" in r or "/out/" in r:
                continue
            out.append(r + ("  [protected]" if self.protected(r) else ""))
        return "\n".join(out[:400]) or "(empty)"

    def read_file(self, path: str) -> str:
        """Read one text file from the repository."""
        p = self.path(path)
        if not p.is_file():
            return f"ERROR: no file {path!r}"
        text = p.read_text(errors="replace")
        if len(text) > READ_LIMIT:
            return text[:READ_LIMIT] + f"\n[... {len(text) - READ_LIMIT} more chars; use search to find a part ...]"
        return text

    def search(self, pattern: str, directory: str = "") -> str:
        """Regex search over text files (file:line: text), e.g. a JSON key in recordings/*/out/*.txt."""
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"ERROR: bad regex: {e}"
        hits = []
        for p in sorted(self.path(directory).rglob("*")):
            if not p.is_file() or ".git" in p.parts or "__pycache__" in p.parts:
                continue
            try:
                for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{self.rel(p)}:{i}: {line.strip()[:200]}")
                        if len(hits) >= 80:
                            return "\n".join(hits) + "\n[... more matches; narrow the pattern ...]"
            except OSError:
                continue
        return "\n".join(hits) or "no matches"

    def write_file(self, path: str, content: str) -> str:
        """Create or replace a whole text file. Protected paths (recordings/, conftest.py, ...) are refused."""
        p = self.path(path)
        r = self.rel(p)
        if self.protected(r):
            raise Refused(f"{r} is protected (engineers own it). If a recording lacks data, say so in the summary.")
        if p.suffix not in self.cfg.get("writable_suffixes", [".py", ".yaml", ".md"]):
            raise Refused(f"{r}: only {', '.join(self.cfg['writable_suffixes'])} files can be written")
        if len(content.encode()) > self.cfg.get("max_file_bytes", 100_000):
            raise Refused(f"{r}: file too large")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content if content.endswith("\n") else content + "\n")
        return f"wrote {r} ({len(content.splitlines())} lines)"

    def newest_recording(self):
        recs = sorted(p for p in (self.root / "recordings").glob("*") if (p / "index.json").exists())
        return recs[-1] if recs else None

    def recorded_data(self, limit: int = 14000) -> tuple:
        """(text, keys): what the newest recording holds - every JSON field path of the first GPU with a sample
        value, and the first lines of plain-text outputs - plus the set of all JSON keys in it."""
        rec = self.newest_recording()
        if rec is None:
            return "(no recording)", set()
        seen = set()
        lines, keys = [f"recording {rec.name}: catalog name -> fields (GPU 0 shown; every GPU has the same)"], set()
        for f in sorted((rec / "out").glob("*.txt")):
            name = f.stem
            m = re.match(r"(.+?)_[0-9a-f]{4}_[0-9a-f]{2}_[0-9a-f]{2}\.\d$", name)
            if m:                                 # per-GPU output (<name>_<bdf>): one example is enough
                if m.group(1) in seen:
                    continue
                seen.add(m.group(1))
                name = f"{m.group(1)} (per GPU, e.g. {name[len(m.group(1)) + 1:]})"
            text = f.read_text(errors="replace")
            try:
                doc = json.loads(text)
                if not isinstance(doc, (dict, list)):
                    raise ValueError("scalar")
            except ValueError:
                head = " | ".join(l.strip() for l in text.splitlines()[:3] if l.strip())[:160]
                lines.append(f"{name}: text: {head or '(empty)'}")
                continue
            _keys(doc, keys)
            first = doc.get("gpu_data", doc) if isinstance(doc, dict) else doc
            if isinstance(first, list) and first and isinstance(first[0], list):
                first = first[0]
            sample = first[0] if isinstance(first, list) and first else first
            flat = []
            _flatten(sample, "", flat)
            wrap = "gpu_data[i]" if isinstance(doc, dict) and "gpu_data" in doc else ("[i]" if isinstance(doc, list) else "")
            lines.append(f"{name} (json{', ' + wrap if wrap else ''}): " + "; ".join(flat)[:6000])
        text = "\n".join(lines)
        return (text[:limit] + "\n[... truncated ...]" if len(text) > limit else text), keys

    def na_fields(self) -> set:
        """Keys whose value is N/A (or empty) everywhere in the newest recording: data this platform does not report.
        A check on such a field can never fail here."""
        rec = self.newest_recording()
        seen, real = set(), set()
        for f in sorted((rec / "out").glob("*.txt")) if rec else []:
            try:
                doc = json.loads(f.read_text(errors="replace"))
            except ValueError:
                continue
            _leaves(doc, seen, real)
        return seen - real

    def versions(self, rel: str) -> tuple:
        """(text at the base commit, text now) of one file; '' when it does not exist."""
        r = subprocess.run(["git", "-C", str(self.root), "show", f"HEAD:{rel}"], capture_output=True, text=True)
        p = self.root / rel
        return (r.stdout if r.returncode == 0 else ""), (p.read_text() if p.exists() else "")

    def git_diff(self) -> str:
        """The change so far, against the base branch (new files included)."""
        return diff(self.root)

    def run_tests(self, target: str = "") -> str:
        """Run pytest in the sandbox (replay mode, no network). target: a test file or node id, or empty for all."""
        if target:
            target = target.strip()
            if not re.fullmatch(r"tests/[\w/.-]+\.py(::[\w\[\]-]+)*", target):
                raise Refused("target must be a test file or node id under tests/, e.g. tests/test_links.py::test_x")
            self.path(target.split("::")[0])
        res = self.runner(self.root, self.cfg, [target] if target else [])
        if not target:
            self.last_tests = res
        return f"exit {res['rc']}\n{res['output'][-6000:]}"

    def declare_finding(self, test: str, reason: str) -> str:
        """Declare that a NEW test fails on the recording because the hardware/stack really is wrong (AGENTS.md rule
        5), not because the test is wrong. The pull request is then opened as a draft for an engineer to judge."""
        if not re.fullmatch(r"tests/[\w/.-]+\.py::[\w\[\]-]+", test.strip()):
            raise Refused("test must be a node id like tests/test_x.py::test_y")
        if len(reason.strip()) < 20:
            raise Refused("give the evidence: which values in the recording are wrong and what they should be")
        self.findings[test.strip()] = reason.strip()
        return f"recorded finding for {test.strip()}"

    def tools(self):
        def wrap(fn):
            def call(**kw):
                try:
                    return fn(**kw)
                except Refused as e:
                    return f"REFUSED: {e}"
                except Exception as e:
                    return f"ERROR: {type(e).__name__}: {e}"[:2000]
            return StructuredTool.from_function(call, name=fn.__name__, description=fn.__doc__,
                                                args_schema=_schema(fn))
        return [wrap(f) for f in (self.list_files, self.read_file, self.search, self.write_file, self.git_diff,
                                  self.run_tests, self.declare_finding)]


def _keys(doc, out: set):
    if isinstance(doc, dict):
        for k, v in doc.items():
            out.add(str(k))
            _keys(v, out)
    elif isinstance(doc, list):
        for v in doc:
            _keys(v, out)


NA = ("N/A", "n/a", "NA", "", None)


def _leaves(doc, seen: set, real: set):
    """seen: every leaf key; real: leaf keys with at least one value that is not N/A."""
    if isinstance(doc, dict):
        for k, v in doc.items():
            if isinstance(v, (dict, list)):
                real.add(str(k))                  # e.g. ppt0.socket_power_limit = {value, unit}: real data
                _leaves(v, seen, real)
            else:
                seen.add(str(k))
                if v not in NA:
                    real.add(str(k))
    elif isinstance(doc, list):
        for v in doc:
            _leaves(v, seen, real)


def _flatten(x, path, out, depth=0):
    if isinstance(x, dict) and depth < 6:
        if set(x) == {"value", "unit"}:           # shown as the dict it is: tests read ["value"] / ["unit"]
            out.append(f"{path}={{value: {x['value']}, unit: {x['unit']}}}")
            return
        for k, v in x.items():
            _flatten(v, f"{path}.{k}" if path else str(k), out, depth + 1)
    elif isinstance(x, list) and depth < 6:
        if x and isinstance(x[0], (dict, list)):
            _flatten(x[0], f"{path}[i]", out, depth + 1)
        else:
            out.append(f"{path}=[{len(x)} items]")
    else:
        out.append(f"{path}={str(x)[:24]}")


def _schema(fn):
    import inspect
    from pydantic import create_model
    fields = {}
    for name, prm in inspect.signature(fn).parameters.items():
        fields[name] = (str, prm.default if prm.default is not inspect.Parameter.empty else ...)
    return create_model(f"{fn.__name__}_args", **fields)


def diff(root: Path, base: str = "HEAD") -> str:
    """Working-tree diff against base, including untracked files (git add -N keeps them unstaged)."""
    subprocess.run(["git", "-C", str(root), "add", "-A", "-N"], check=True, capture_output=True)
    r = subprocess.run(["git", "-C", str(root), "diff", "--no-color", base], capture_output=True, text=True)
    return r.stdout or "(no changes)"


def changed_files(root: Path, base: str = "HEAD") -> list:
    subprocess.run(["git", "-C", str(root), "add", "-A", "-N"], check=True, capture_output=True)
    r = subprocess.run(["git", "-C", str(root), "diff", "--name-only", base], capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if l]
