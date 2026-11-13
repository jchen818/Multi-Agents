"""Evidence for Phase 3 triage: load an archived AGFHC run, search its logs, and verify cited log lines.

An archive is the AGFHC_<YYYYmmdd_HHMMSS>/ folder the agfhc server writes (summary.json, report.html, logs/...).
Every line the triage agent cites must appear verbatim in one of these logs (verify_citations); this check is
deterministic, so a hypothesis can never rest on a made-up log line.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
TEXT_SUFFIXES = {".log", ".json", ".txt", ".tsv", ".err", ""}
SKIP = {"run.sh", "start_epoch", "end_epoch", "start_uptime", "rc", ".run_id", "tmux_session"}
MAX_FILE_BYTES = 4_000_000
MIN_CITATION_CHARS = 12          # "FAIL" alone proves nothing


def clean_lines(text: str) -> list:
    """Lines as a reader sees them: ANSI codes removed, carriage-return redraws split, trailing space trimmed."""
    out = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        for part in raw.split("\r"):
            line = ANSI.sub("", part).rstrip()
            if line:
                out.append(line)
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", ANSI.sub("", s)).strip()


@dataclass
class Evidence:
    root: Path
    summary: dict
    docs: dict = field(default_factory=dict)        # relative path -> list of cleaned lines
    failures: list = field(default_factory=list)    # from results.json: test, status, subject, lines

    @property
    def verdict(self) -> str:
        return self.summary.get("verdict", "UNKNOWN")

    def doc_names(self) -> list:
        return sorted(self.docs)


def load_archive(path) -> Evidence:
    root = Path(path)
    summary = json.loads((root / "summary.json").read_text())
    docs = {}
    logs = root / "logs"
    for p in sorted(logs.rglob("*")) if logs.exists() else []:
        if not p.is_file() or p.name in SKIP or p.suffix not in TEXT_SUFFIXES:
            continue
        if p.stat().st_size > MAX_FILE_BYTES:
            data = p.read_bytes()[-MAX_FILE_BYTES:]   # keep the end: failures are usually there
        else:
            data = p.read_bytes()
        if b"\x00" in data[:4096]:
            continue
        lines = clean_lines(data.decode(errors="replace"))
        if lines:
            docs[str(p.relative_to(root))] = lines
    failures = []
    for name, lines in docs.items():
        if name.endswith("results.json"):
            try:
                rj = json.loads("\n".join(lines))
            except json.JSONDecodeError:
                continue
            for r in rj.get("results", []):
                if r.get("state") not in ("passed", "skipped"):
                    ext = [l for e in r.get("extended_results") or [] for l in (e.get("test") or [])]
                    failures.append({"test": f"{r.get('test')}-{r.get('run', 1)}-{r.get('iteration', 1)}",
                                     "status": r.get("primary_status_name"), "subject": r.get("primary_action_subject"),
                                     "lines": ext[:20]})
    return Evidence(root=root, summary=summary, docs=docs, failures=failures)


def grep(ev: Evidence, pattern: str, max_hits: int = 40, files: str = "") -> list:
    """[(file, line_no, line)] matching a case-insensitive regex (or plain text if the regex is invalid)."""
    try:
        rx = re.compile(pattern, re.I)
    except re.error:
        rx = re.compile(re.escape(pattern), re.I)
    hits = []
    for name in ev.doc_names():
        if files and files not in name:
            continue
        for i, line in enumerate(ev.docs[name], 1):
            if rx.search(line):
                hits.append((name, i, line[:400]))
                if len(hits) >= max_hits:
                    return hits
    return hits


def verify_citations(ev: Evidence, citations: list) -> list:
    """Each citation {'text': ..., 'file': optional} must be a verbatim part of one log line (whitespace-normalized).
    Returns [{'text', 'ok', 'where' | 'why'}] in order."""
    out = []
    for c in citations:
        text = _norm(c.get("text", "") if isinstance(c, dict) else str(c))
        want = (c.get("file") or "") if isinstance(c, dict) else ""
        if len(text) < MIN_CITATION_CHARS:
            out.append({"text": text, "ok": False, "why": f"too short to prove anything (< {MIN_CITATION_CHARS} chars)"})
            continue
        where = None
        for name in ev.doc_names():
            if want and not (name == want or name.endswith("/" + want) or Path(name).name == Path(want).name):
                continue
            for i, line in enumerate(ev.docs[name], 1):
                if text in _norm(line):
                    where = f"{name}:{i}"
                    break
            if where:
                break
        out.append({"text": text, "ok": bool(where), **({"where": where} if where else
                    {"why": "not found verbatim in " + (want or "any log")})})
    return out


def evidence_markdown(ev: Evidence, budget: int = 9000) -> str:
    """Condensed evidence for the model. Log lines carry their [file:line] so they can be cited exactly."""
    s = ev.summary
    parts = [f"## Run {s.get('run_id')} on {s.get('host')} · recipe `{s.get('recipe')}` · verdict **{ev.verdict}** "
             f"(rc {s.get('rc')})",
             "**Versions:** " + "; ".join(f"{k}={v}" for k, v in (s.get("versions") or {}).items()),
             f"**Counts:** {s.get('counts')}"]
    if ev.failures:
        parts.append("### Failing tests (results.json)")
        for f in ev.failures:
            parts.append(f"- `{f['test']}`: {f['status']} on {f['subject'] or '?'}")
            parts += [f"    {l}" for l in f["lines"][:8]]
    kern = [h for h in grep(ev, r".", 60, "dmesg_new")]
    parts.append("### New kernel messages during the run" + ("" if kern else ": none"))
    parts += [f"[{n}:{i}] {l}" for n, i, l in kern[:40]]
    bad = grep(ev, r"\b(FAIL|ERROR|FATL|ABORT|miscompare|timeout|RAS|XGMI|PCIe|AER|MCE)\b", 60)
    parts.append("### Error-like lines")
    parts += [f"[{n}:{i}] {l}" for n, i, l in bad]
    parts.append("### Log files: " + ", ".join(f"{n} ({len(ev.docs[n])} lines)" for n in ev.doc_names()))
    text = "\n".join(parts)
    return text if len(text) <= budget else text[:budget] + "\n[... truncated ...]"
