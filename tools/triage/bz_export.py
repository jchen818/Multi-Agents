"""Read-only Bugzilla export for Phase 3 triage (step 3.1).

Pulls every bug (any status/resolution) with its comments and attachment list, and saves small text attachments,
into state/bugs/: bugs.jsonl (one bug per line) and att/<bug>/<attachment id>_<name>.
GET requests only. The API key comes from secrets/bugzilla.yaml and goes in the query string
(Apache on the Bugzilla host strips the X-BUGZILLA-API-KEY header, as in the Bugzilla Agent tool).

    .venv/bin/python -m tools.triage.bz_export            # export everything
    .venv/bin/python -m tools.triage.bz_export --stats    # summary of what was exported
"""
import argparse
import base64
import collections
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(os.environ.get("MULTI_AGENTS_ROOT") or Path(__file__).resolve().parents[2])
OUT = ROOT / "state" / "bugs"
BUG_FIELDS = ("id,summary,product,component,version,status,resolution,severity,priority,platform,op_sys,"
              "creation_time,last_change_time,creator,assigned_to,keywords,see_also,depends_on,blocks,dupe_of")
ATT_FIELDS = "id,file_name,summary,content_type,size,is_obsolete,creation_time,creator"
TEXT_TYPES = re.compile(r"^(text/|application/(json|x-yaml|xml|x-log))")
MAX_TEXT_ATTACHMENT = 512 * 1024
SAFE = re.compile(r"[^\w.\-]")


class BZ:
    """Minimal read-only client. Same conventions as the Bugzilla Agent tool (v3.2.1)."""

    def __init__(self, url: str, api_key: str, timeout: int = 30, session=None):
        self.base, self.key, self.timeout = url.rstrip("/"), api_key, timeout
        self.s = session or requests.Session()

    def get(self, path: str, params: dict = None) -> dict:
        p = dict(params or {}, api_key=self.key)
        r = self.s.get(f"{self.base}/rest/{path.lstrip('/')}", params=p, timeout=self.timeout)
        try:
            data = r.json()
        except ValueError:
            raise RuntimeError(f"HTTP {r.status_code} non-JSON from {path}")
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"Bugzilla error {data.get('code')}: {data.get('message')}")
        return data

    def custom_fields(self) -> list:
        return [f for f in self.get("field/bug").get("fields", []) if f.get("is_custom") and not f.get("is_obsolete")]

    def all_bugs(self, extra_fields: list, page: int = 500) -> list:
        bugs, offset = [], 0
        while True:
            chunk = self.get("bug", {"include_fields": BUG_FIELDS + "," + ",".join(extra_fields),
                                     "limit": page, "offset": offset, "order": "bug_id"}).get("bugs", [])
            bugs += chunk
            if len(chunk) < page:
                return bugs
            offset += page

    def comments(self, bug_id) -> list:
        return self.get(f"bug/{bug_id}/comment").get("bugs", {}).get(str(bug_id), {}).get("comments", [])

    def attachments(self, bug_id) -> list:
        return self.get(f"bug/{bug_id}/attachment", {"include_fields": ATT_FIELDS}).get("bugs", {}).get(str(bug_id), [])

    def attachment_data(self, att_id) -> bytes:
        d = self.get(f"bug/attachment/{att_id}").get("attachments", {}).get(str(att_id), {})
        return base64.b64decode(d.get("data", ""))


def load_client(path: Path = None) -> BZ:
    cfg = yaml.safe_load(open(path or ROOT / "secrets" / "bugzilla.yaml"))
    if not cfg.get("api_key") or "<" in str(cfg["api_key"]):
        raise SystemExit("secrets/bugzilla.yaml has no real api_key")
    return BZ(cfg["url"], str(cfg["api_key"]))


def export(bz: BZ, out: Path = OUT, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    custom = bz.custom_fields()
    labels = {f["name"]: f.get("display_name") or f["name"] for f in custom}
    bugs = bz.all_bugs(list(labels))
    log(f"{len(bugs)} bugs; custom fields: {', '.join(labels.values()) or 'none'}")
    saved_atts, tmp = 0, out / "bugs.jsonl.tmp"
    with open(tmp, "w") as fh:
        for i, b in enumerate(bugs, 1):
            bid = b["id"]
            b["custom"] = {labels[k]: b.pop(k) for k in list(b) if k in labels}
            b["comments"] = [{"count": c.get("count"), "creator": c.get("creator"), "time": c.get("time"),
                              "text": c.get("text") or ""} for c in bz.comments(bid)]
            atts = bz.attachments(bid)
            for a in atts:
                a["saved"] = ""
                if a.get("is_obsolete") or (a.get("size") or 0) > MAX_TEXT_ATTACHMENT:
                    continue
                if not TEXT_TYPES.match(str(a.get("content_type", ""))) and not re.search(
                        r"\.(log|txt|json|csv|out|dmesg|sel)$", a.get("file_name", ""), re.I):
                    continue
                p = out / "att" / str(bid) / f"{a['id']}_{SAFE.sub('_', a.get('file_name') or 'attachment')}"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(bz.attachment_data(a["id"]))
                a["saved"] = str(p.relative_to(out))
                saved_atts += 1
            b["attachments"] = atts
            fh.write(json.dumps(b) + "\n")
            if i % 25 == 0:
                log(f"  {i}/{len(bugs)}")
    tmp.replace(out / "bugs.jsonl")
    meta = {"exported": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "url": bz.base, "bugs": len(bugs),
            "text_attachments_saved": saved_atts, "custom_fields": labels}
    (out / "export.json").write_text(json.dumps(meta, indent=2))
    return meta


def stats(out: Path = OUT) -> str:
    bugs = [json.loads(l) for l in open(out / "bugs.jsonl")]
    by = lambda k: collections.Counter(str(b.get(k) or "-") for b in bugs).most_common(20)
    gpu = re.compile(r"\b(gpu|agfhc|rvs|rocm|amdgpu|xgmi|hbm|mi3\d\dx?|ras|aer|pcie|amd-smi|rocm-smi)\b", re.I)
    gpu_bugs = [b for b in bugs if gpu.search(b.get("summary", "") + " " + " ".join(c["text"] for c in b["comments"]))]
    lines = [f"{len(bugs)} bugs, {sum(len(b['comments']) for b in bugs)} comments, "
             f"{sum(1 for b in bugs for a in b['attachments'] if a.get('saved'))} text attachments saved",
             "status/resolution: " + ", ".join(f"{s}/{r}={n}" for (s, r), n in collections.Counter(
                 (b.get("status"), b.get("resolution") or "-") for b in bugs).most_common(12)),
             "products: " + ", ".join(f"{p}={n}" for p, n in by("product")),
             f"GPU-related (summary or comments mention GPU/AGFHC/ROCm/XGMI/HBM/RAS/PCIe...): {len(gpu_bugs)}"]
    lines += [f"  #{b['id']} [{b.get('status')}/{b.get('resolution') or '-'}] {b.get('product')}: {b.get('summary', '')[:80]}"
              for b in gpu_bugs[:40]]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stats", action="store_true", help="summarize the last export instead of exporting")
    args = ap.parse_args()
    if not args.stats:
        meta = export(load_client())
        print(f"exported {meta['bugs']} bugs, {meta['text_attachments_saved']} text attachments to {OUT}")
    print(stats())


if __name__ == "__main__":
    sys.exit(main())
