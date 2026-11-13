"""Reflexion loop for triage (Phase 3.3): engineer-confirmed root causes become episode notes triage reads next time.

An episode is written only when an engineer confirms the cause; triage never labels its own runs. Each note records
the failure signature, what triage guessed (and whether it was right), the confirmed cause, and what evidence pointed
to it. Files are the record ($TRIAGE_EPISODES_DIR, default state/triage/episodes/<archive>.yaml); Mem0
(agents_memory, type=episode, agent_id=triage) indexes them for similarity search.

    .venv/bin/python -m agents.triage.episodes pending
    .venv/bin/python -m agents.triage.episodes confirm state/archives/AGFHC_20261015_133404 \
        --category "environment/node state" --cause "HBM held by the vLLM model server on mitacai" \
        --evidence "failed to allocate buffer" --note "mitacai serves models; never a test node"
    .venv/bin/python -m agents.triage.episodes list
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("MULTI_AGENTS_ROOT") or Path(__file__).resolve().parents[2])
TRIAGE = ROOT / "state" / "triage"
AGENT_ID = "triage"


def _words(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9_]+", (s or "").lower()) if len(w) > 2}


def signature(triage_json: dict, summary: dict) -> str:
    """What the failure looked like, in the terms the next triage will search with."""
    failed = summary.get("failed") or []
    return (f"{summary.get('recipe')} on {summary.get('platform') or summary.get('versions', {}).get('GPU_MODEL', '')}: "
            f"failed {', '.join(failed) or '-'}; AGFHC {summary.get('versions', {}).get('AGFHC', '?')}, "
            f"ROCm {summary.get('versions', {}).get('ROCM', '?')}")


class EpisodeStore:
    def __init__(self, base: Path = None, memory=None):
        self.base = Path(base or os.environ.get("TRIAGE_EPISODES_DIR") or TRIAGE / "episodes")
        self.m = memory        # mem0 Memory or None

    def path(self, archive_name: str) -> Path:
        return self.base / f"{archive_name}.yaml"

    def confirm(self, archive: Path, category: str, cause: str, evidence: str = "", note: str = "",
                by: str = "", triage_dir: Path = None) -> dict:
        from agents.triage.graph import CATEGORIES
        if category not in CATEGORIES:
            raise ValueError(f"category must be one of: {', '.join(CATEGORIES)}")
        archive = Path(archive)
        summary = json.loads((archive / "summary.json").read_text())
        tj_path = (triage_dir or TRIAGE) / archive.name / "triage.json"
        tj = json.loads(tj_path.read_text()) if tj_path.exists() else {}
        ranked = tj.get("ranked") or []
        guesses = [{"rank": i, "category": c["hyp"]["category"], "cause": c["hyp"]["cause"],
                    "right_category": c["hyp"]["category"] == category} for i, c in enumerate(ranked, 1)]
        hit = next((g["rank"] for g in guesses if g["right_category"]), None)
        ep = {"archive": archive.name, "run_id": summary.get("run_id"), "host": summary.get("host"),
              "signature": signature(tj, summary), "failed": summary.get("failed") or [],
              "versions": {k: summary.get("versions", {}).get(k) for k in ("AGFHC", "ROCM", "AMDGPU", "VBIOS", "BIOS")},
              "confirmed": {"category": category, "cause": cause, "evidence": evidence, "note": note,
                            "by": by or os.environ.get("USER", ""), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
              "triage": {"had_report": bool(tj), "guesses": guesses, "right_category_rank": hit,
                         "open_checks": [c["hyp"]["cause"] for c in tj.get("dropped") or []
                                         if c["hyp"]["category"] == category][:3]}}
        self.base.mkdir(parents=True, exist_ok=True)
        self.path(archive.name).write_text(yaml.safe_dump(ep, sort_keys=False, allow_unicode=True))
        if self.m is not None:
            try:
                self.m.add(self.text(ep), agent_id=AGENT_ID, infer=False,
                           metadata={"type": "episode", "archive": archive.name, "category": category})
            except Exception as e:          # the file is the record
                ep["index_error"] = f"{type(e).__name__}: {e}"[:300]
        return ep

    @staticmethod
    def text(ep: dict) -> str:
        c, t = ep["confirmed"], ep["triage"]
        guess = (f"triage ranked the right category #{t['right_category_rank']}" if t["right_category_rank"]
                 else "triage missed the right category" if t["had_report"] else "no triage report")
        return (f"Episode {ep['archive']}: {ep['signature']}. Confirmed cause [{c['category']}]: {c['cause']}. "
                f"Evidence: {c['evidence'] or '-'}. {guess}. Note: {c['note'] or '-'}")

    def all(self) -> list:
        return [yaml.safe_load(p.read_text()) for p in sorted(self.base.glob("*.yaml"))] if self.base.exists() else []

    def similar(self, query: str, k: int = 5, exclude_archive: str = "") -> list:
        """Closest confirmed episodes. Mem0 when available, else word overlap over the files."""
        eps = {e["archive"]: e for e in self.all() if e["archive"] != exclude_archive}
        found = []
        if self.m is not None:
            try:
                res = self.m.search(query, filters={"agent_id": AGENT_ID, "type": "episode"}, limit=k + 1)
                rows = res.get("results", res) if isinstance(res, dict) else res
                for r in rows or []:
                    a = (r.get("metadata") or {}).get("archive")
                    if a in eps and a not in [f["archive"] for f in found]:
                        found.append(eps[a])
            except Exception:
                found = []
        if not found:
            q = _words(query)
            scored = sorted(((len(q & _words(self.text(e))) / max(1, len(q)), e) for e in eps.values()),
                            key=lambda x: -x[0])
            found = [e for s, e in scored if s > 0]
        return found[:k]


def pending(archives_dir: Path = ROOT / "state" / "archives", store: EpisodeStore = None) -> list:
    """FAIL archives that have a triage report but no confirmed cause yet."""
    store = store or EpisodeStore()
    out = []
    for d in sorted(Path(archives_dir).glob("AGFHC_*")):
        try:
            v = json.loads((d / "summary.json").read_text()).get("verdict")
        except Exception:
            continue
        if v != "PASS" and not store.path(d.name).exists():
            out.append((d.name, (TRIAGE / d.name / "triage.json").exists()))
    return out


def _memory():
    try:
        from core.memory import get_memory
        return get_memory()
    except Exception as e:
        print(f"warning: Mem0 unavailable ({type(e).__name__}: {e}); episode saved to file only")
        return None


def main():
    ap = argparse.ArgumentParser(description="Confirm triage root causes (Phase 3.3)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("confirm")
    c.add_argument("archive")
    c.add_argument("--category", required=True)
    c.add_argument("--cause", required=True)
    c.add_argument("--evidence", default="", help="the log line or fact that settled it")
    c.add_argument("--note", default="", help="what would have pointed to it sooner")
    sub.add_parser("list")
    sub.add_parser("pending")
    args = ap.parse_args()
    if args.cmd == "confirm":
        ep = EpisodeStore(memory=_memory()).confirm(Path(args.archive), args.category, args.cause, args.evidence, args.note)
        print(EpisodeStore.text(ep))
        print(f"saved {EpisodeStore().path(ep['archive'])}" + (f" (index error: {ep['index_error']})" if ep.get("index_error") else ""))
    elif args.cmd == "list":
        for e in EpisodeStore().all():
            print(EpisodeStore.text(e))
    else:
        rows = pending()
        print("\n".join(f"{a}  pending ({'triage report ready' if t else 'not triaged yet'})" for a, t in rows)
              or "no pending archives")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
