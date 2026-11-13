"""Phase 3.3: confirmed causes become episodes that later triage runs see."""
import asyncio
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from agents.triage.episodes import EpisodeStore, pending  # noqa: E402
from agents.triage.graph import build_graph, save  # noqa: E402
from tests.test_triage_evidence import archive  # noqa: E402,F401
from tests.test_triage_graph import Fake  # noqa: E402


def triaged(archive_dir, tmp_path):
    out = asyncio.run(build_graph(Fake()).ainvoke({"archive": str(archive_dir)}))
    save(out, tmp_path / "triage")
    return out


def test_confirm_records_guesses_and_rank(archive, tmp_path):
    triaged(archive, tmp_path)
    st = EpisodeStore(tmp_path / "eps")
    ep = st.confirm(archive, "test tool/config defect", "hbm_s16_ds reads before it writes",
                    evidence="pattern miscompare at 0x1f000", note="check AGFHC version first",
                    by="john", triage_dir=tmp_path / "triage")
    assert ep["triage"]["had_report"] and ep["triage"]["right_category_rank"] == 2   # refined hardware was #1
    assert ep["confirmed"]["by"] == "john" and "hbm_s16_ds-1-1" in ep["signature"]
    on_disk = yaml.safe_load(st.path(archive.name).read_text())
    assert on_disk["confirmed"]["cause"] == "hbm_s16_ds reads before it writes"
    assert "ranked the right category #2" in st.text(ep)
    with pytest.raises(ValueError):
        st.confirm(archive, "cosmic rays", "x", triage_dir=tmp_path / "triage")


def test_episodes_reach_the_next_triage_but_not_their_own(archive, tmp_path):
    st = EpisodeStore(tmp_path / "eps")
    st.confirm(archive, "environment/node state", "HBM held by the vLLM model server", triage_dir=tmp_path / "none")
    other = json.loads(json.dumps(yaml.safe_load(st.path(archive.name).read_text())))
    other["archive"] = "AGFHC_20261023_134551"
    st.path(other["archive"]).write_text(yaml.safe_dump(other))
    m = Fake()
    out = asyncio.run(build_graph(m, episodes=st).ainvoke({"archive": str(archive)}))
    assert "Confirmed past failures" in out["history"] and "AGFHC_20261023_134551" in out["history"]
    assert f"Episode {archive.name}" not in out["history"]          # its own answer is never shown to it
    first = next(u for s, u in m.prompts if s == "Hypotheses")
    assert "vLLM model server" in first


class FakeMem:
    def __init__(self):
        self.rows = []

    def add(self, text, agent_id, infer, metadata):
        self.rows.append({"memory": text, "metadata": metadata, "agent_id": agent_id})

    def search(self, query, filters, limit):
        return {"results": [r for r in self.rows if r["agent_id"] == filters["agent_id"]][:limit]}


def test_mem0_index_used_when_available(archive, tmp_path):
    mem = FakeMem()
    st = EpisodeStore(tmp_path / "eps", memory=mem)
    st.confirm(archive, "hardware", "bad HBM stack on GPU-05", triage_dir=tmp_path / "none")
    assert mem.rows[0]["metadata"] == {"type": "episode", "archive": archive.name, "category": "hardware"}
    assert [e["archive"] for e in st.similar("anything")] == [archive.name]


def test_pending_lists_unconfirmed_fail_archives(archive, tmp_path):
    st = EpisodeStore(tmp_path / "eps")
    assert pending(archive.parent, st) == [(archive.name, False)]
    st.confirm(archive, "hardware", "x", triage_dir=tmp_path / "none")
    assert pending(archive.parent, st) == []
