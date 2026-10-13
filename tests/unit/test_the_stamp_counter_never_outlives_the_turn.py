"""The stamp counter is gone from the metadata on EVERY exit a ``StopPipeline`` makes.

``_stamp`` numbers the calls of a turn (``StageMetrics.seq``) and keeps its count on
``ctx.metadata["_call_seq"]``. It is bookkeeping, not a contract: the number that matters lives
on each metrics row, and ``ctx.metadata`` is what a host persists. ``_finish`` removes the
counter — but ``_finish`` is the road of a turn that ENDS, and a turn a ``StopPipeline`` stops
takes another one: the ``except`` in ``run_turn`` returned the context as it was, counter
included. The one test there was (``test_pipeline.py``) covers the ordinary exit only.

Measured on ``b2bf3c3`` (this file, run against that tree): a stop from ``after_noumeno`` left
``_call_seq == 1`` on the metadata, from ``after_id`` ``3``, and so on for every hook that fires
after the first stage, and for a stop raised from inside the executor.

Two things make it more than an untidy key:

* a host that persists ``ctx.metadata`` stores a counter nobody reads;
* a carrier that hands a turn's metadata to the next one makes the next turn's numbering START
  from it — the first call of that turn is not ``seq == 1``.

Every stop point is a twin, and each proves its own premise first: the probe reads the counter
at the instant of the stop, so "absent afterwards" is the clearing's doing and not a turn that
never stamped anything. The two stop points where nothing can be left behind (``before_turn``:
nothing was stamped yet; ``after_turn``: ``_finish`` already removed it) are the controls.
"""

from __future__ import annotations

import pytest
from cogno_anima.types import PipelineContext, ToolExecution

from cogno_soma import Hooks, Pipeline, StopPipeline, TurnConfig
from cogno_soma import pipeline as pipeline_mod
from tests.conftest import (FakeEgo, FakeID, FakeNER, FakeNoumeno, FakeSuperego,
                            RecordingDispatcher, StubBackend, StubEmbedder)

KEY = pipeline_mod._SEQ_KEY

READ = ToolExecution(tool="read_ledger", arguments={}, result="3 rows", ok=True,
                     side_effect=False)

#: hook → what the turn needs for that hook to fire: (route, the judge approves on its Nth call).
HOOKS_THAT_FIRE_AFTER_A_STAMP = {
    "after_noumeno": ("SUPEREGO", 1),
    "after_ner": ("SUPEREGO", 1),
    "after_id": ("SUPEREGO", 1),
    "after_ego": ("EGO", 1),
    "on_commit": ("EGO", 1),
    "on_rollback": ("EGO", 2),          # fires between a rejected attempt and its retry
    "after_superego": ("SUPEREGO", 1),
}


class _Probe:
    """A hook that reads the counter at the instant it fires, then stops the turn."""

    def __init__(self) -> None:
        self.seen: "list[object]" = []

    def __call__(self, ctx) -> None:
        self.seen.append(ctx.metadata.get(KEY))
        raise StopPipeline(reason="completed", response="Please hold.")


class _StopsInsideProcess:
    """A host's wrapper around the executor: the inner stage runs, then the wrapper stops."""

    name = "ego"

    def __init__(self, seen: "list[object]") -> None:
        self._inner, self._seen = FakeEgo(tool_calls=[READ]), seen

    async def process(self, ctx, backend, dispatcher, *, system_prompt):
        ctx = await self._inner.process(ctx, backend, dispatcher, system_prompt=system_prompt)
        self._seen.append(ctx.metadata.get(KEY))
        raise StopPipeline(reason="completed", response="Please hold.")


def _pipe(route: str, approve_after: int, ego=None) -> Pipeline:
    return Pipeline(embedder=StubEmbedder(), noumeno=FakeNoumeno(), ner=FakeNER(),
                    id_stage=FakeID(route=route),
                    ego=ego or FakeEgo(tool_calls=[READ]),
                    superego=FakeSuperego(approve_after=approve_after))


def _cfg(hooks=None) -> TurnConfig:
    return TurnConfig(gen_backend=StubBackend(), ego_backend=StubBackend(), ego_prompt="x",
                      voice_prompt="v", hooks=hooks, max_corrections=2)


async def _stopped_at(hook: str):
    route, approve_after = HOOKS_THAT_FIRE_AFTER_A_STAMP.get(hook, ("SUPEREGO", 1))
    probe = _Probe()
    ctx = await _pipe(route, approve_after).run_turn(
        PipelineContext(user_input="hi"), _cfg(Hooks(**{hook: probe})),
        dispatcher=RecordingDispatcher())
    return ctx, probe


# ── the twin, one per stop point ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("hook", sorted(HOOKS_THAT_FIRE_AFTER_A_STAMP))
async def test_twin_a_stop_from_a_hook_leaves_no_counter_on_the_metadata(hook):
    ctx, probe = await _stopped_at(hook)
    assert len(probe.seen) == 1, "the premise: the hook fired, once, and stopped the turn"
    assert ctx.superego_result.response == "Please hold."
    assert isinstance(probe.seen[0], int) and probe.seen[0] > 0, (
        "the premise: the counter WAS on the metadata when the stop was raised")
    assert KEY not in ctx.metadata


async def test_twin_a_stop_from_inside_the_executor_leaves_no_counter_on_the_metadata():
    seen: "list[object]" = []
    ctx = await _pipe("EGO", 1, ego=_StopsInsideProcess(seen)).run_turn(
        PipelineContext(user_input="hi"), _cfg(), dispatcher=RecordingDispatcher())
    assert ctx.superego_result.response == "Please hold."
    assert seen and isinstance(seen[0], int) and seen[0] > 0
    assert KEY not in ctx.metadata
    # …and the pass that stop interrupted is still recorded, after the counter was read for it.
    assert ctx.ego_result.metrics.seq == seen[0] + 1


# ── the controls: the two stop points where there is nothing to leave behind ───────────────

async def test_control_before_turn_stops_before_anything_was_stamped():
    """Absent in both worlds, for a reason that is not the clearing: no stage ran."""
    ctx, probe = await _stopped_at("before_turn")
    assert probe.seen == [None]
    assert KEY not in ctx.metadata and not any(m.seq for m in ctx.stage_metrics)


async def test_control_after_turn_fires_after_the_ordinary_exit_removed_it():
    """Absent in both worlds: ``_finish`` removes the counter BEFORE it fires ``after_turn``."""
    ctx, probe = await _stopped_at("after_turn")
    assert probe.seen == [None]
    assert KEY not in ctx.metadata and any(m.seq > 0 for m in ctx.stage_metrics)


# ── what the clearing must not take with it ────────────────────────────────────────────────

@pytest.mark.parametrize("hook", sorted(HOOKS_THAT_FIRE_AFTER_A_STAMP))
async def test_the_numbers_already_stamped_stay_on_their_rows(hook):
    """The counter goes; the ``seq`` each call was given before the stop is still on its
    metrics row, none of them past the count the probe read, and nothing else on the metadata
    is touched — a host key that merely looks like the counter included."""
    route, approve_after = HOOKS_THAT_FIRE_AFTER_A_STAMP[hook]
    probe = _Probe()
    start = PipelineContext(user_input="hi")
    start.metadata.update({"tenant": "acme", "_call_seq_of_the_host": 7})
    ctx = await _pipe(route, approve_after).run_turn(
        start, _cfg(Hooks(**{hook: probe})), dispatcher=RecordingDispatcher())
    seqs = {m.seq for m in ctx.stage_metrics if m.seq}
    assert ctx.noumeno.metrics.seq == 1 and 1 in seqs and max(seqs) <= probe.seen[0]
    assert ctx.metadata["tenant"] == "acme" and ctx.metadata["_call_seq_of_the_host"] == 7


# ── the consequence: a carried metadata no longer shifts the next turn's numbering ─────────

async def test_the_next_turn_of_a_carrier_that_keeps_the_metadata_starts_at_one():
    """A carrier that hands the whole metadata on (``SessionRunner`` does not; a host's own
    may). Turn 1 is stopped after the ID, three stamps in; turn 2 runs on what turn 1 left."""
    first, probe = await _stopped_at("after_id")
    assert probe.seen == [3]

    second = PipelineContext(user_input="and now?")
    second.metadata.update(first.metadata)
    second = await _pipe("SUPEREGO", 1).run_turn(second, _cfg(),
                                                 dispatcher=RecordingDispatcher())
    assert second.noumeno.metrics.seq == 1
