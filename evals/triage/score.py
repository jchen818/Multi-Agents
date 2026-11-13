"""Summarize a triage eval run: top-1 / top-3 by kind (accuracy counts real + bug cases; seeded is regression)."""
import json
import sys
from collections import defaultdict

rows = [json.loads(r["response"]["output"]) for r in json.load(open(sys.argv[1]))["results"]["results"]
        if r.get("response") and r["response"].get("output")]
by = defaultdict(list)
for o in rows:
    by["real+bug" if o["kind"] in ("real", "bug") else "seeded"].append(o)
for k, rs in by.items():
    n = len(rs)
    print(f"{k:9s} n={n:2d}  top-1 {sum(o['top1'] for o in rs)}/{n}  top-3 {sum(o['top3'] for o in rs)}/{n}")
print()
for o in sorted(rows, key=lambda o: (o["kind"], o["case"])):
    got = " > ".join(r["category"] for r in o["ranked"][:3]) or "nothing verified"
    mark = "1" if o["top1"] else ("3" if o["top3"] else "x")
    print(f"[{mark}] {o['kind']:6s} {o['case']:30s} expected {o['expected']:24s} got {got}")
