"""Index the exported Bugzilla bugs into Qdrant `agents_bugs` for triage retrieval (Phase 3.1).

Each bug becomes several chunks: a header chunk (summary, product, status, custom fields, first comment) and one
chunk per later comment and text attachment (long ones split). Root causes are written in comments, not a field,
so comments are the main signal. Embeddings: qwen3-embed via the OpenAI-compatible endpoint (EMBED_BASE_URL).

    .venv/bin/python -m tools.triage.bug_index build
    .venv/bin/python -m tools.triage.bug_index search "hbm_s16_ds miscompare GPU-05"
"""
import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(os.environ.get("MULTI_AGENTS_ROOT") or Path(__file__).resolve().parents[2])
BUGS = ROOT / "state" / "bugs"
COLLECTION = "agents_bugs"
CHUNK = 1500
NS = uuid.UUID("6f1c2a40-0b7e-4d8a-9f43-2b1d7c0e5a11")


def _split(text: str, size: int = CHUNK) -> list:
    text = (text or "").strip()
    if len(text) <= size:
        return [text] if text else []
    parts, cur = [], ""
    for para in text.split("\n"):
        while len(para) > size:
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(para[:size])
            para = para[size:]
        if len(cur) + len(para) + 1 > size:
            parts.append(cur)
            cur = para
        else:
            cur = f"{cur}\n{para}" if cur else para
    if cur:
        parts.append(cur)
    return parts


def chunks(bug: dict, base: Path = BUGS) -> list:
    """[(kind, text, extra payload)] for one exported bug."""
    custom = ", ".join(f"{k}={v}" for k, v in (bug.get("custom") or {}).items() if v not in (None, "", "---"))
    head = (f"Bug {bug['id']}: {bug.get('summary', '')}\nProduct {bug.get('product')} / {bug.get('component')} | "
            f"{bug.get('status')}/{bug.get('resolution') or '-'}" + (f" | {custom}" if custom else ""))
    out = []
    comments = bug.get("comments") or []
    first = comments[0]["text"] if comments else ""
    for i, part in enumerate(_split(first, CHUNK - len(head)) or [""]):
        out.append(("summary" if i == 0 else "description", f"{head}\n{part}".strip(), {"comment": 0}))
    for c in comments[1:]:
        for part in _split(c.get("text", ""), CHUNK - len(head)):
            out.append(("comment", f"{head}\n[comment #{c.get('count')} {str(c.get('time', ''))[:10]}]\n{part}",
                        {"comment": c.get("count")}))
    for a in bug.get("attachments") or []:
        if a.get("saved") and (base / a["saved"]).exists():
            body = (base / a["saved"]).read_bytes()[:8000].decode(errors="replace")
            for part in _split(body, CHUNK - len(head))[:3]:
                out.append(("attachment", f"{head}\n[attachment {a.get('file_name')}]\n{part}",
                            {"attachment": a.get("file_name")}))
    return out


class OpenAIEmbedder:
    """qwen3-embed on vLLM rejects the `dimensions` parameter, so send model + input only (as core/memory.py)."""

    def __init__(self):
        from openai import OpenAI
        self.c = OpenAI(base_url=os.environ["EMBED_BASE_URL"], api_key="EMPTY")
        self.model = os.environ.get("EMBED_MODEL", "qwen3-embed")

    def __call__(self, texts: list) -> list:
        out = []
        for i in range(0, len(texts), 32):
            out += [d.embedding for d in self.c.embeddings.create(model=self.model, input=texts[i:i + 32]).data]
        return out


def qdrant():
    from qdrant_client import QdrantClient
    return QdrantClient(url=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"))


def build(client, embed, bugs_file: Path = BUGS / "bugs.jsonl", collection: str = COLLECTION, log=print) -> int:
    from qdrant_client import models
    if not collection.startswith("agents_"):
        raise ValueError("agents write only to agents_* collections")   # never touch open-webui_* collections
    bugs = [json.loads(l) for l in open(bugs_file)]
    rows = [(b, kind, text, extra) for b in bugs for kind, text, extra in chunks(b, bugs_file.parent)]
    vecs = embed([t for _, _, t, _ in rows])
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(collection, vectors_config=models.VectorParams(size=len(vecs[0]), distance=models.Distance.COSINE))
    points = [models.PointStruct(
        id=str(uuid.uuid5(NS, f"{b['id']}:{kind}:{i}")), vector=v,
        payload=dict(extra, bug_id=b["id"], kind=kind, text=t, summary=b.get("summary", ""), product=b.get("product"),
                     component=b.get("component"), status=b.get("status"), resolution=b.get("resolution") or ""))
        for i, ((b, kind, t, extra), v) in enumerate(zip(rows, vecs))]
    client.upsert(collection, points)
    log(f"indexed {len(bugs)} bugs as {len(points)} chunks into {collection}")
    return len(points)


def search(client, embed, query: str, k: int = 5, exclude_bugs=(), product: str = None,
           collection: str = COLLECTION) -> list:
    """Best chunk per bug, best bugs first: [{'bug_id', 'score', 'summary', 'status', 'resolution', 'text'}].
    exclude_bugs keeps a case's own bug out during evals."""
    from qdrant_client import models
    must_not = [models.FieldCondition(key="bug_id", match=models.MatchAny(any=list(exclude_bugs)))] if exclude_bugs else []
    must = [models.FieldCondition(key="product", match=models.MatchValue(value=product))] if product else []
    flt = models.Filter(must=must or None, must_not=must_not or None) if (must or must_not) else None
    hits = client.query_points(collection, query=embed([query])[0], query_filter=flt, limit=k * 6,
                               with_payload=True).points
    best = {}
    for h in hits:
        b = h.payload["bug_id"]
        if b not in best:
            best[b] = {"bug_id": b, "score": round(h.score, 3), "summary": h.payload["summary"],
                       "status": h.payload["status"], "resolution": h.payload["resolution"], "text": h.payload["text"]}
    return list(best.values())[:k]


SPECIFIC = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*[A-Za-z0-9]")


def specific_terms(text: str, limit: int = 12) -> list:
    """Test/recipe names, status codes, part numbers: tokens with a digit, underscore, inner capital or hyphen
    (the same idea as the Bugzilla Agent's _is_specific)."""
    seen, out = set(), []
    for t in SPECIFIC.findall(text or ""):
        low = t.lower()
        if low in seen or len(low) < 4:
            continue
        if any(c.isdigit() for c in t) or "_" in t or "-" in t or any(c.isupper() for c in t[1:]):
            seen.add(low)
            out.append(low)
    return out[:limit]


def hybrid_search(client, embed, query: str, k: int = 5, exclude_bugs=(), bugs_file: Path = BUGS / "bugs.jsonl",
                  collection: str = COLLECTION, extra_terms=()) -> list:
    """Semantic hits re-ranked by exact specific terms (test names like hbm_s16_ds, codes like TEST_HBM_STRESS_FAIL).
    Semantic scores alone sit in a narrow band (0.59-0.74 in the first build), so a bug that merely shares the word
    'failed' could outrank the one that names the same test; exact technical terms decide the order instead."""
    terms = list(dict.fromkeys(specific_terms(query) + [t.lower() for t in extra_terms if len(t) >= 4]))
    texts = {}
    if bugs_file.exists():
        for line in open(bugs_file):
            b = json.loads(line)
            texts[b["id"]] = (b.get("summary", "") + " " + " ".join(c["text"] for c in b.get("comments") or [])).lower()
    if texts:                       # a term in a quarter or more of all bugs (rocm, agfhc, the GPU model) is noise
        cap = max(3, len(texts) // 4)
        terms = [t for t in terms if sum(t in x for x in texts.values()) <= cap]
    sem = {h["bug_id"]: h for h in search(client, embed, query, k=max(k * 4, 20), exclude_bugs=exclude_bugs,
                                          collection=collection)}
    for bid, text in texts.items():          # bugs that name a term but were not in the semantic top
        if bid not in sem and bid not in exclude_bugs and any(t in text for t in terms):
            sem[bid] = {"bug_id": bid, "score": 0.0, "summary": text[:80], "status": "", "resolution": "", "text": ""}
    for h in sem.values():
        hit = [t for t in terms if t in texts.get(h["bug_id"], "")]
        h["terms"] = hit
        h["rank"] = round(h["score"] + 0.15 * min(len(hit), 4), 3)
    return sorted(sem.values(), key=lambda h: -h["rank"])[:k]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["build", "search"])
    ap.add_argument("query", nargs="*")
    args = ap.parse_args()
    client, embed = qdrant(), OpenAIEmbedder()
    if args.cmd == "build":
        build(client, embed)
    else:
        for h in hybrid_search(client, embed, " ".join(args.query)):
            print(f"#{h['bug_id']} {h['rank']:.3f} (semantic {h['score']:.3f}, terms {', '.join(h['terms']) or '-'}) "
                  f"[{h['status']}/{h['resolution'] or '-'}] {h['summary'][:70]}")
            print("    " + h["text"].replace("\n", " ")[:240])


if __name__ == "__main__":
    sys.exit(main())
