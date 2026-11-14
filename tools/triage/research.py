"""Web research for triage (Phase 3): AMD release notes and docs, as leads with verbatim quotes.

Only pages on allowed documentation domains are fetched (ROCm docs, AMD sites, the ROCm GitHub org). Page text is data:
every quote the model takes from a page must appear on that page verbatim (whitespace-normalized), like log citations.
Fetching goes through the guarded SearXNG MCP tools, so internal addresses stay refused.
"""
import os
import re
from urllib.parse import urlparse

DEFAULT_DOMAINS = ("rocm.docs.amd.com", "instinct.docs.amd.com", "rocm.blogs.amd.com", "docs.amd.com", "amd.com",
                   "github.com/ROCm")
URL = re.compile(r"https?://[^\s<>\"')\]]+")
WRAP = re.compile(r'\A\s*<tool_output tool="[^"]*" seconds="[^"]*">\n?(.*?)\n?</tool_output>\s*\Z', re.S)
MIN_QUOTE = 20


def domains() -> tuple:
    env = os.environ.get("TRIAGE_RESEARCH_DOMAINS")
    return tuple(d.strip() for d in env.split(",") if d.strip()) if env else DEFAULT_DOMAINS


def allowed(url: str, doms: tuple = None) -> bool:
    p = urlparse(url)
    if p.scheme != "https" or not p.hostname:
        return False
    host, path = p.hostname.lower(), p.path
    for d in doms or domains():
        h, _, pre = d.partition("/")
        if (host == h or host.endswith("." + h)) and (not pre or path.lower().startswith("/" + pre.lower())):
            return True
    return False


def _text(x) -> str:
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(x, list):
        x = "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in x)
    m = WRAP.match(str(x))
    return m.group(1) if m else str(x)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


async def gather(search, fetch, queries: list, max_pages: int = 4, page_chars: int = 30000) -> dict:
    """{'queries': [...], 'pages': [{'url','query','text'}], 'skipped': [urls not on allowed domains]}"""
    seen, pages, skipped = set(), [], []
    for q in queries:
        res = _text(await search(query=q))
        for u in URL.findall(res):
            u = u.rstrip(".,;")
            if u in seen:
                continue
            seen.add(u)
            if not allowed(u):
                skipped.append(u)
                continue
            if len(pages) >= max_pages:
                continue
            body = _text(await fetch(url=u))
            if body and not body.startswith(("REFUSED", "ERROR")):
                pages.append({"url": u, "query": q, "text": body[:page_chars]})
    return {"queries": list(queries), "pages": pages, "skipped": skipped[:20]}


def verify_finding(f: dict, pages: list) -> str:
    """'' if the quote is verbatim on the page it claims, else why not."""
    page = next((p for p in pages if p["url"] == f.get("url")), None)
    if page is None:
        return "url was not one of the fetched pages"
    q = norm(f.get("quote", ""))
    if len(q) < MIN_QUOTE:
        return f"quote shorter than {MIN_QUOTE} characters"
    return "" if q in norm(page["text"]) else "quote not found verbatim on the page"
