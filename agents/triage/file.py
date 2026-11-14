"""Hand a triage bug draft to Bugzilla through the Bugzilla Agent's draft -> confirm flow (ported from v3.2.1).

draft:  reads state/triage/<archive>/triage.json, validates every field against Bugzilla's own field list for the
        product (GET only), and either lists what is missing/invalid or saves a draft with a short-lived token.
create: files a saved draft. A person runs it at a terminal and types the token back to confirm. It needs
        write_enabled: true and the engineer's own filing_api_key in secrets/bugzilla.yaml (the read-only export key
        is never used to file), the same Unix user who drafted it, and an unexpired token. Triage agents never call it.

    .venv/bin/python -m agents.triage.file draft state/archives/AGFHC_20261015_133404 --severity normal \\
        --field Code=8825 --field "Customer=Engineering team lab"
    .venv/bin/python -m agents.triage.file show <token>
    .venv/bin/python -m agents.triage.file create <token>
"""
import argparse
import difflib
import getpass
import json
import os
import secrets
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("MULTI_AGENTS_ROOT") or Path(__file__).resolve().parents[2])
TRIAGE = ROOT / "state" / "triage"
DRAFTS = ROOT / "state" / "bugzilla" / "drafts"
CONFIG = ROOT / "config" / "bugzilla.yaml"
SECRETS = ROOT / "secrets" / "bugzilla.yaml"
DRAFT_TTL_S = 15 * 60
MAX_ATTACHMENT = 1000 * 1024


# ------------------------------------------------------------------ ported helpers (Bugzilla Agent 3.2.1)

def match(value, choices: list) -> tuple:
    """Resolve a user-supplied value against legal choices. Returns (match or None, suggestions)."""
    if value is None:
        return None, []
    v = str(value).strip().lower()
    if not v:
        return None, []
    low = {c.lower(): c for c in choices}
    if v in low:
        return low[v], []
    pre = [c for c in choices if c.lower().startswith(v)]
    if len(pre) == 1:
        return pre[0], []
    sub = [c for c in choices if v in c.lower()]
    if len(sub) == 1:
        return sub[0], []
    close_low = difflib.get_close_matches(v, list(low.keys()), n=5, cutoff=0.6)
    close = [low[c] for c in close_low]
    if len(close) == 1 and difflib.SequenceMatcher(None, v, close_low[0]).ratio() >= 0.85:
        return close[0], []
    return None, (pre or sub or close)[:8]


def legal_values(field: dict, product) -> list:
    vals = []
    for v in field.get("values") or []:
        name = v.get("name")
        if not name or name == "---" or v.get("is_active") is False:
            continue
        vis = v.get("visibility_values") or []
        if product is not None and vis and field.get("value_field") == "product" and product not in vis:
            continue
        vals.append(name)
    return vals


def field_visible(field: dict, product: str) -> bool:
    if field.get("visibility_field") == "product":
        vis = field.get("visibility_values") or []
        return not vis or product in vis
    return True


def form_spec(fields: dict, product: str, defaults: dict) -> list:
    """Every field to collect for this product, in form order (severity, hardware, OS, then custom fields)."""
    spec = []
    for fname, param, label, default in (("bug_severity", "severity", "Severity", defaults.get("severity", "")),
                                         ("rep_platform", "platform", "Hardware", defaults.get("hardware", "Server")),
                                         ("op_sys", "op_sys", "OS", defaults.get("os", "All"))):
        f = fields.get(fname, {})
        spec.append({"param": param, "label": label, "required": True, "type": "select",
                     "values": legal_values(f, product), "default": default})
    for f in fields.values():
        if not f.get("is_custom") or not f.get("is_on_bug_entry") or f.get("is_obsolete") or not field_visible(f, product):
            continue
        kind = {2: "select", 3: "multiselect"}.get(f.get("type"), "text")
        spec.append({"param": f["name"], "label": f.get("display_name") or f["name"],
                     "required": bool(f.get("is_mandatory")), "type": kind,
                     "values": legal_values(f, product) if kind != "text" else [], "default": ""})
    return spec


# ------------------------------------------------------------------ client

class Bugzilla:
    """GET for validation; POST only from create(). Key in the query/body: Apache strips the header on this host."""

    def __init__(self, url: str, key: str, session=None, timeout: int = 30):
        import requests
        self.base, self.key, self.timeout = url.rstrip("/"), key, timeout
        self.s = session or requests.Session()

    def _parse(self, r):
        try:
            data = r.json()
        except ValueError:
            raise RuntimeError(f"HTTP {r.status_code} non-JSON")
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"Bugzilla error {data.get('code')}: {data.get('message')}")
        return data

    def get(self, path, params=None):
        return self._parse(self.s.get(f"{self.base}/rest/{path}", params=dict(params or {}, api_key=self.key),
                                      timeout=self.timeout))

    def post(self, path, body):
        return self._parse(self.s.post(f"{self.base}/rest/{path}", json=dict(body, api_key=self.key),
                                       headers={"X-BUGZILLA-API-KEY": self.key}, timeout=self.timeout))

    def fields(self) -> dict:
        return {f["name"]: f for f in self.get("field/bug").get("fields", [])}

    def enterable_products(self) -> list:
        return [p["name"] for p in self.get("product", {"type": "enterable", "include_fields": "name,is_active"})
                .get("products", []) if p.get("is_active", True)]

    def product_detail(self, product: str) -> dict:
        ps = self.get("product", {"names": product}).get("products", [])
        if not ps:
            raise RuntimeError(f"product '{product}' not found")
        return {"components": [c["name"] for c in ps[0].get("components", []) if c.get("is_active", True)],
                "versions": [v["name"] for v in ps[0].get("versions", []) if v.get("is_active", True)]}


def load_config() -> dict:
    cfg = yaml.safe_load(open(CONFIG)) if CONFIG.exists() else {}
    sec = yaml.safe_load(open(SECRETS)) if SECRETS.exists() else {}
    return {**(cfg or {}), "url": (sec or {}).get("url") or (cfg or {}).get("url"), "api_key": (sec or {}).get("api_key"),
            "filing_api_key": (sec or {}).get("filing_api_key"), "write_enabled": bool((sec or {}).get("write_enabled"))}


# ------------------------------------------------------------------ draft

def build_draft(bz: Bugzilla, triage: dict, summary_json: dict, cfg: dict, product: str = "", component: str = "",
                version: str = "", severity: str = "", fields: dict = None, attach_report: str = "") -> dict:
    """{'ready': bool, 'missing': [...], 'invalid': [...], 'body': {...}, 'preview': [(label, value)], 'attachment'}"""
    d = triage.get("draft") or {}
    host = summary_json.get("host", "")
    want = product or (cfg.get("product_by_host") or {}).get(host) or cfg.get("default_product") or ""
    want = (cfg.get("product_aliases") or {}).get(want, want)
    out = {"ready": False, "missing": [], "invalid": [], "body": {}, "preview": [], "attachment": None}
    products = bz.enterable_products()
    prod, sugg = match(want, products)
    if not prod:
        out["missing" if not want else "invalid"].append(
            f"Product{' ' + repr(want) if want else ''}" + (f" (did you mean: {', '.join(sugg)})" if sugg else
                                                            f" (options: {', '.join(products[:15])})"))
        return out
    spec, detail = form_spec(bz.fields(), prod, cfg.get("defaults") or {}), bz.product_detail(prod)
    body, preview = {"product": prod}, [("Product", prod)]
    given = {"severity": severity} if severity else {}
    given.update(fields or {})
    fixed = {"severity": "severity", "hardware": "platform", "platform": "platform", "os": "op_sys", "op_sys": "op_sys"}
    assigned, leftover = {}, []
    for k, v in given.items():
        kl = str(k).strip().lower()
        param = fixed.get(kl)
        if not param:
            exact = [s for s in spec if s["label"].lower() == kl or s["param"].lower() == kl]
            partial = [s for s in spec if kl in s["label"].lower() and s["param"] not in fixed.values()]
            hit = exact[0] if exact else (partial[0] if len(partial) == 1 else None)
            param = hit["param"] if hit else None
        if param:
            assigned[param] = v
        else:
            leftover.append(str(k))
    for s in spec:
        raw = assigned.get(s["param"], s["default"] or None)
        if raw in (None, "", []):
            if s["required"]:
                out["missing"].append(s["label"] + (f" (options: {', '.join(s['values'][:15])})" if s["values"] else ""))
            continue
        if s["type"] in ("select", "multiselect") and s["values"]:
            items = [x.strip() for x in str(raw).split(",")] if s["type"] == "multiselect" else [raw]
            got = []
            for it in items:
                m, sg = match(it, s["values"])
                if m:
                    got.append(m)
                else:
                    out["invalid"].append(f"{s['label']}: '{it}' is not allowed."
                                          + (f" Did you mean: {', '.join(sg)}?" if sg else f" Options: {', '.join(s['values'][:15])}"))
            if got:
                body[s["param"]] = got if s["type"] == "multiselect" else got[0]
                preview.append((s["label"], ", ".join(got)))
        else:
            body[s["param"]] = str(raw)
            preview.append((s["label"], str(raw)))
    if leftover:
        out["invalid"].append(f"Unknown field(s) for {prod}: {', '.join(leftover)}")
    for label, param, vals, val in (("Component", "component", detail["components"], component),
                                    ("Version", "version", detail["versions"], version)):
        if len(vals) == 1:
            body[param] = vals[0]
            preview.append((label, vals[0]))
        elif not val:
            out["missing"].append(f"{label} (options: {', '.join(vals)})")
        else:
            m, sg = match(val, vals)
            if m:
                body[param] = m
                preview.append((label, m))
            else:
                out["invalid"].append(f"{label}: '{val}' is not allowed. Options: {', '.join(sg or vals)}")
    summary, description = (d.get("summary") or "").strip(), (d.get("description") or "").strip()
    if not summary:
        out["missing"].append("Summary (the triage report has no bug draft; nothing survived verification)")
    elif len(summary) > 255:
        out["invalid"].append(f"Summary is {len(summary)} characters; the limit is 255.")
    note = ("\n\n---\nDrafted by the triage agent from archive " + str(summary_json.get("run_id")) +
            ". Causes above are hypotheses ranked from verified log lines.")
    body["summary"], body["description"] = summary, description + note if description else ""
    if attach_report and len(attach_report.encode()) <= MAX_ATTACHMENT:
        import base64
        out["attachment"] = {"file_name": "triage.md", "content_type": "text/plain", "summary": "Triage report",
                             "data": base64.b64encode(attach_report.encode()).decode()}
    cr = d.get("critique") or {}
    out["critique"] = ("revised once for: " + "; ".join(d["revised_for"]) if d.get("revised_for") else
                       "ok" if cr.get("ok") else ("UNRESOLVED: " + "; ".join(cr.get("problems") or ["no critique"]) if cr
                                                  else "not run (draft from an older triage run; rerun triage)"))
    out.update(body=body, preview=preview, ready=not out["missing"] and not out["invalid"])
    return out


def save_draft(draft: dict, archive: str, user: str = None, now: float = None, drafts: Path = DRAFTS) -> str:
    drafts.mkdir(parents=True, exist_ok=True)
    os.chmod(drafts, 0o700)
    now = now or time.time()
    for p in drafts.glob("*.json"):                       # drop expired drafts
        try:
            if json.loads(p.read_text())["expires"] < now:
                p.unlink()
        except Exception:
            pass
    token = secrets.token_hex(4)
    p = drafts / f"{token}.json"
    p.write_text(json.dumps({"user": user or getpass.getuser(), "archive": archive, "expires": now + DRAFT_TTL_S,
                             "body": draft["body"], "attachment": draft.get("attachment"), "preview": draft["preview"],
                             "critique": draft.get("critique", "")}))
    os.chmod(p, 0o600)
    return token


def create(token: str, bz: Bugzilla, cfg: dict, confirm: str, user: str = None, now: float = None,
           drafts: Path = DRAFTS) -> str:
    """File a saved draft. Every refusal leaves the draft in place."""
    if not cfg.get("write_enabled"):
        return "BLOCKED: write_enabled is not true in secrets/bugzilla.yaml (read-only by default). The draft was kept."
    if not cfg.get("filing_api_key"):
        return "BLOCKED: no filing_api_key (your own Bugzilla key) in secrets/bugzilla.yaml. The draft was kept."
    p = drafts / f"{(token or '').strip()}.json"
    if not p.exists():
        return "ERROR: draft token not found. Run draft again."
    d = json.loads(p.read_text())
    if d["expires"] < (now or time.time()):
        return "ERROR: draft token expired. Run draft again."
    if d["user"] != (user or getpass.getuser()):
        return "ERROR: this draft belongs to a different user."
    if (confirm or "").strip() != token.strip():
        return "NOT FILED: confirmation did not match the token."
    d["body"]["description"] += f"\nReviewed and filed by {d['user']}."
    try:
        bug_id = bz.post("bug", d["body"]).get("id")
    except Exception as e:
        return f"ERROR: bug was NOT filed: {e}"
    p.unlink()
    msg = f"Filed Bug {bug_id}: {bz.base}/show_bug.cgi?id={bug_id}"
    if d.get("attachment"):
        try:
            bz.post(f"bug/{bug_id}/attachment", dict(d["attachment"], ids=[bug_id], is_patch=False))
            msg += "\nAttached triage.md."
        except Exception as e:
            msg += f"\nWARNING: filed, but the attachment failed: {e}"
    return msg


def preview_text(token: str, d: dict, similar: list) -> str:
    lines = [f"DRAFT READY. token={token} (valid {DRAFT_TTL_S // 60} min). Nothing has been filed.", ""]
    lines += [f"{k}: {v}" for k, v in d["preview"]]
    lines += [f"Summary: {d['body']['summary']}", "Description:", d["body"]["description"]]
    lines.append(f"\nSelf-critique: {d.get('critique') or 'unknown'}")
    if d.get("attachment"):
        lines.append("Attachment: triage.md (the full triage report)")
    lines += ["", "Similar existing bugs (from the last export):"]
    lines += [f"  #{h['bug_id']} [{h.get('status')}/{h.get('resolution') or '-'}] {h.get('summary', '')[:80]}"
              + (f" (terms: {', '.join(h['terms'])})" if h.get("terms") else " (wording only)") for h in similar] or ["  none"]
    lines += ["", "Review it. To file it under your own Bugzilla account:",
              f"  .venv/bin/python -m agents.triage.file create {token}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Triage bug draft -> Bugzilla (draft, then human-confirmed create)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft")
    d.add_argument("archive")
    d.add_argument("--product", default="")
    d.add_argument("--component", default="")
    d.add_argument("--version", default="")
    d.add_argument("--severity", default="")
    d.add_argument("--field", action="append", default=[], help='Label=value, e.g. "Customer=Engineering team lab"')
    s = sub.add_parser("show")
    s.add_argument("token")
    c = sub.add_parser("create")
    c.add_argument("token")
    args = ap.parse_args()
    cfg = load_config()
    if args.cmd == "draft":
        archive = Path(args.archive)
        tj = TRIAGE / archive.name / "triage.json"
        if not tj.exists():
            raise SystemExit(f"no triage report for {archive.name}; run agents.triage.graph on it first")
        triage = json.loads(tj.read_text())
        report = (TRIAGE / archive.name / "triage.md").read_text() if (TRIAGE / archive.name / "triage.md").exists() else ""
        flds = dict(f.split("=", 1) for f in args.field if "=" in f)
        bz = Bugzilla(cfg["url"], cfg["api_key"])          # read-only key: validation only
        dr = build_draft(bz, triage, json.loads((archive / "summary.json").read_text()), cfg, args.product,
                         args.component, args.version, args.severity, flds, report)
        if not dr["ready"]:
            print("NOT READY. Add these, then run draft again with all of them:")
            print("\n".join([f"- Missing: {m}" for m in dr["missing"]] + [f"- Invalid: {i}" for i in dr["invalid"]]))
            raise SystemExit(2)
        try:
            from tools.triage.bug_index import OpenAIEmbedder, hybrid_search, qdrant
            from tools.triage.evidence import load_archive
            fails = load_archive(archive).failures            # test names AND status codes (TEST_HBM_STRESS_FAIL)
            q = dr["body"]["summary"] + " " + " ".join(f"{f['test']} {f['status']}" for f in fails)
            similar = hybrid_search(qdrant(), OpenAIEmbedder(), q, 5,
                                    extra_terms=[f["test"].rsplit("-", 2)[0] for f in fails] + [f["status"] or "" for f in fails])
        except Exception:
            similar = []
        token = save_draft(dr, archive.name)
        print(preview_text(token, json.loads((DRAFTS / f"{token}.json").read_text()), similar))
    elif args.cmd == "show":
        p = DRAFTS / f"{args.token}.json"
        print(preview_text(args.token, json.loads(p.read_text()), []) if p.exists() else "no such draft")
    else:
        if not sys.stdin.isatty():
            raise SystemExit("create must be run by a person at a terminal")
        p = DRAFTS / f"{args.token}.json"
        if p.exists():
            print(preview_text(args.token, json.loads(p.read_text()), []).rsplit("\nReview it.", 1)[0])
        confirm = input(f"\nType the token ({args.token}) to file this bug under your Bugzilla account: ")
        print(create(args.token, Bugzilla(cfg["url"], cfg.get("filing_api_key") or ""), cfg, confirm))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
