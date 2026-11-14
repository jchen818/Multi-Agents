"""agents_bugs index: chunking, build into in-memory Qdrant, search, exclusion of a case's own bug."""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.triage.bug_index import build, chunks, search, _split  # noqa: E402

qdrant_client = pytest.importorskip("qdrant_client")


def fake_embed(texts):
    """Bag of words hashed into 64 dims: enough for 'same words -> close vectors'."""
    out = []
    for t in texts:
        v = [0.0] * 64
        for w in re.findall(r"[a-z0-9_]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        n = math.sqrt(sum(x * x for x in v)) or 1
        out.append([x / n for x in v])
    return out


BUGS = [
    {"id": 22, "summary": "hbm_ds_ntd of all_lvl5 failed in AGFHC test", "product": "G8825Z5U2BC", "component": "GPU",
     "status": "RESOLVED", "resolution": "FIXED", "custom": {"Customer": "---"},
     "comments": [{"count": 0, "time": "2026-09-15", "text": "hbm_ds_ntd fails"},
                  {"count": 3, "time": "2026-10-06", "text": "HBM_MISCOMPARE read before pattern written; all words zero"}],
     "attachments": []},
    {"id": 23, "summary": "AGFHC rochpl in gfx_lvl4 failures on 8825", "product": "G8825Z5U2BC", "component": "GPU",
     "status": "RESOLVED", "resolution": "FIXED", "custom": {},
     "comments": [{"count": 0, "time": "2026-09-15", "text": "Device memory allocation failed for A and b " * 80}],
     "attachments": [{"file_name": "hpl.log", "saved": "att/23/1_hpl.log"}]},
    {"id": 30, "summary": "lost ip shared nic issue", "product": "R2513G6", "component": "BMC",
     "status": "RESOLVED", "resolution": "FIXED", "custom": {}, "comments": [{"count": 0, "time": "x", "text": "BMC shared NIC"}],
     "attachments": []},
]


@pytest.fixture
def store(tmp_path):
    (tmp_path / "att" / "23").mkdir(parents=True)
    (tmp_path / "att" / "23" / "1_hpl.log").write_text("HPL ERROR from process # 0 ... Requested 241.767 GiBs total")
    f = tmp_path / "bugs.jsonl"
    f.write_text("\n".join(json.dumps(b) for b in BUGS))
    client = qdrant_client.QdrantClient(":memory:")
    n = build(client, fake_embed, f, log=lambda *_: None)
    return client, n


def test_chunks_cover_header_comments_attachments(tmp_path):
    (tmp_path / "att" / "23").mkdir(parents=True)
    (tmp_path / "att" / "23" / "1_hpl.log").write_text("Requested 241.767 GiBs")
    kinds = [k for k, _, _ in chunks(BUGS[1], tmp_path)]
    assert kinds[0] == "summary" and "description" in kinds and kinds[-1] == "attachment"
    assert all(t.startswith("Bug 23: AGFHC rochpl") for _, t, _ in chunks(BUGS[1], tmp_path))
    assert all(len(p) <= 1500 for p in _split("x" * 4000))


def test_search_finds_the_right_bug_and_can_exclude_it(store):
    client, n = store
    assert n >= 5
    hits = search(client, fake_embed, "HBM_MISCOMPARE hbm_ds_ntd words zero")
    assert hits[0]["bug_id"] == 22 and len({h["bug_id"] for h in hits}) == len(hits)
    assert 22 not in [h["bug_id"] for h in search(client, fake_embed, "HBM_MISCOMPARE hbm_ds_ntd", exclude_bugs=[22])]
    assert search(client, fake_embed, "241.767 GiBs requested", k=1)[0]["bug_id"] == 23
    assert {h["bug_id"] for h in search(client, fake_embed, "NIC", product="R2513G6")} == {30}


def test_refuses_non_agents_collection(tmp_path):
    f = tmp_path / "bugs.jsonl"
    f.write_text(json.dumps(BUGS[2]))
    with pytest.raises(ValueError):
        build(qdrant_client.QdrantClient(":memory:"), fake_embed, f, collection="open-webui_files", log=lambda *_: None)


def test_hybrid_ranks_exact_terms_first(store, tmp_path):
    from tools.triage.bug_index import hybrid_search, specific_terms
    client, _ = store
    f = tmp_path / "bugs.jsonl"
    assert specific_terms("rochpl device memory allocation failed on GPU-05, TEST_HBM_STRESS_FAIL") == \
        ["gpu-05", "test_hbm_stress_fail"]
    hits = hybrid_search(client, fake_embed, "hbm_ds_ntd failed", k=3, bugs_file=f)
    assert hits[0]["bug_id"] == 22 and "hbm_ds_ntd" in hits[0]["terms"]
    assert 22 not in [h["bug_id"] for h in hybrid_search(client, fake_embed, "hbm_ds_ntd", exclude_bugs=[22], bugs_file=f)]


def test_common_terms_do_not_count(tmp_path):
    from tools.triage.bug_index import build, hybrid_search
    bugs = [{"id": i, "summary": f"rocm agfhc issue {i}", "product": "P", "component": "C", "status": "RESOLVED",
             "resolution": "FIXED", "custom": {}, "comments": [{"count": 0, "time": "t", "text": "rocm-7.2.1 seen"}],
             "attachments": []} for i in range(1, 13)]
    bugs.append({"id": 99, "summary": "hbm_s16_ds allocation failure", "product": "P", "component": "C",
                 "status": "RESOLVED", "resolution": "FIXED", "custom": {}, "comments": [], "attachments": []})
    f = tmp_path / "bugs.jsonl"
    f.write_text("\n".join(json.dumps(b) for b in bugs))
    c = qdrant_client.QdrantClient(":memory:")
    build(c, fake_embed, f, log=lambda *_: None)
    hits = hybrid_search(c, fake_embed, "hbm_s16_ds failed on rocm-7.2.1", k=13, bugs_file=f)
    assert all("rocm-7.2.1" not in h["terms"] for h in hits)            # in 12 of 13 bugs: not a signal
    assert next(h for h in hits if h["bug_id"] == 99)["terms"] == ["hbm_s16_ds"]
