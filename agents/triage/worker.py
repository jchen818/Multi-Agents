"""Triage as a service (Phase 6.2): consumes triage_request messages from agents:triage:in.

A request's artifact_path is an AGFHC archive directory. The worker runs the Phase 3 triage graph on it, saves
triage.md / triage.json under state/triage/<archive>/, and replies triage_done with artifact_path = that directory
and a short JSON note (top causes, bug draft summary). It files nothing. On the last delivery attempt a failure is
replied as triage_failed instead of being retried again.

    .venv/bin/python -m agents.triage.worker            # runs until stopped; honors agents:kill
"""
import asyncio
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from core.bus import Bus, Worker, connect

ROOT = Path(__file__).resolve().parents[2]


def summary_note(state: dict) -> str:
    top = [{"category": c["hyp"]["category"], "cause": c["hyp"]["cause"], "score": round(float(c.get("score") or 0), 2)}
           for c in (state.get("ranked") or [])[:3]]
    return json.dumps({"top": top, "draft_summary": (state.get("draft") or {}).get("summary", "")})[:3900]


def make_handler(run_triage, save):
    """run_triage(archive) -> awaitable triage state; save(state) -> saved directory."""

    async def handle(bus: Bus, msg: dict):
        archive = msg.get("artifact_path", "")
        try:
            if not archive or not Path(archive).is_dir():
                raise FileNotFoundError(f"archive not found: {archive or '(none)'}")
            state = await run_triage(archive)
            saved = save(state)
        except Exception as e:
            if msg.get("_last") or isinstance(e, FileNotFoundError):
                await bus.reply(msg, "triage_failed", note=f"{type(e).__name__}: {e}"[:1000])
                return
            raise
        await bus.reply(msg, "triage_done", artifact_path=str(saved), note=summary_note(state))
    return handle


async def main():
    from core.models import get_model
    from agents.triage import graph as triage
    from agents.triage.episodes import EpisodeStore, _memory
    async with AsyncExitStack() as stack:
        web = await triage.production_research(stack)

        async def run_triage(archive):
            g = triage.build_graph(get_model("planner"), bug_search=triage.production_bug_search(),
                                   episodes=EpisodeStore(memory=_memory()), research=web)
            return await g.ainvoke({"archive": archive, "exclude_bugs": []})

        r = connect()
        await Worker(r, "triage", make_handler(run_triage, triage.save)).run()


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    asyncio.run(main())
