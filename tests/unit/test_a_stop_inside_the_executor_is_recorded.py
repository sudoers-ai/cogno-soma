"""A ``StopPipeline`` raised from INSIDE the executor's ``process`` still leaves that pass on record.

``Pipeline(ego=…)`` takes any stage of the documented shape, and a host wraps the executor: the
inner stage runs to its end (``ctx.ego_result`` is set, the tools ran), then a check of the host's
decides the turn ends here and raises ``StopPipeline`` — from inside ``process``. The loop
recorded a pass on the line AFTER the ``await``::

    ctx = await self._ego.process(...)
    _stamp(ctx, ctx.ego_result.metrics, attempt=attempt, prompt=...)      # never reached
    ctx.turn_executions.extend(ctx.ego_result.tools_executed)             # never reached

so a pass that ended that way kept its cost row with ``seq == 0`` and no host label, and what it
executed never joined ``turn_executions``. Measured downstream with a probe: about 1 executor
call in 205 has this shape.

What does NOT change, and is pinned here as a constant rather than claimed as a repair:
``committed_this_turn`` already answered ``True`` on such a turn, because it reads the surviving
``ego_result`` as a floor. The accumulator and the stamp are what were missing.

The record is made by the ONE function the ordinary path uses (``_record_ego_pass``), and only
for a result THIS call produced — a stage that stops before replacing the result leaves the
previous attempt's row and executions exactly as they were.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap

import cogno_anima.metakeys as mk
import pytest
from cogno_anima.errors import ToolExecutionError
from cogno_anima.stages.ego import EgoStage
from cogno_anima.types import (PipelineContext, ToolExecution, ToolResult,
                               committed_this_turn)

from cogno_soma import Hooks, Pipeline, StopPipeline, TurnConfig, pipeline as pipeline_module
from tests.conftest import (FakeEgo, FakeID, FakeNER, FakeNoumeno, FakeSuperego,
                            RecordingDispatcher, StubBackend, StubEmbedder)

QUESTION = "Which of the two did you mean?"
EGO_LABEL = "9f3c1a"

READ = ToolExecution(tool="read_ledger", arguments={}, result="3 rows", ok=True,
                     side_effect=False)
WRITE = ToolExecution(tool="add_entry", arguments={"amount": 45}, result="entry 7 recorded",
                      ok=True, side_effect=True)


class _Scripted:
    """One reply per call, then the last one — the text transport of a REAL executor."""

    def __init__(self, replies) -> None:
        self.replies, self.model, self.calls = list(replies), "scripted", 0

    async def generate(self, system: str, prompt: str):
        self.calls += 1
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return reply, 40, 12


class _WritingDispatcher:
    """One tool, and calling it WRITES."""

    def __init__(self) -> None:
        self.executed: "list[tuple[str, dict]]" = []

    def tools_schema(self) -> "list[dict]":
        return [{"type": "function", "function": {
            "name": "add_entry", "description": "Record one entry.",
            "parameters": {"type": "object", "properties": {"amount": {"type": "number"}}}}}]

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        self.executed.append((name, arguments))
        return ToolResult(output="entry 7 recorded", ok=True, side_effect=True)


class _Wrapped:
    """The host-shaped executor wrapper: the inner stage runs to its END, then — on the listed
    invocations — the wrapper raises, from inside ``process``. ``before=True`` raises BEFORE
    the inner stage ran instead, so the result on the context is still the previous pass's."""

    name = "ego"

    def __init__(self, inner, *, on=(1,), raises=None, before=False) -> None:
        self._inner, self._on, self._before = inner, tuple(on), before
        self._raises = raises or (lambda: StopPipeline(reason="completed", response=QUESTION))
        self.invocations = 0

    async def process(self, ctx, backend, dispatcher, *, system_prompt):
        self.invocations += 1
        stops = self.invocations in self._on
        if stops and self._before:
            raise self._raises()
        ctx = await self._inner.process(ctx, backend, dispatcher, system_prompt=system_prompt)
        if stops:
            raise self._raises()
        return ctx


def _ego_rows(ctx) -> list:
    """The executor's cost rows, each once (a row can sit in two slots), in call order."""
    rows = {id(m): m for m in ctx.stage_metrics if m.stage == "ego"}
    return sorted(rows.values(), key=lambda m: m.seq)


async def _turn(ego, *, superego=None, dispatcher=None, ego_backend=None, budget=1,
                hooks=None, raw=None):
    pipe = Pipeline(embedder=StubEmbedder(), noumeno=FakeNoumeno("Record an entry of 45."),
                    ner=FakeNER(intent_class="ACTION_REQUEST"), id_stage=FakeID(route="EGO"),
                    ego=ego, superego=superego or FakeSuperego())
    cfg = TurnConfig(gen_backend=StubBackend(), ego_backend=ego_backend or StubBackend(),
                     ego_prompt="x", limits_prompt="limits", voice_prompt="voice",
                     max_corrections=budget, hooks=hooks)
    ctx = raw if raw is not None else PipelineContext(user_input="record an entry of 45")
    ctx.metadata[mk.PROMPT_SHAS] = {"ego": EGO_LABEL}
    return await pipe.run_turn(ctx, cfg, dispatcher=dispatcher or RecordingDispatcher())


def _real_writing_turn(**wrapper_kw):
    """The REAL executor on the text path: it calls ``add_entry`` once, the dispatcher writes,
    it drafts — and only then does the wrapper decide."""
    dispatcher = _WritingDispatcher()
    backend = _Scripted(['<TOOL_CALL>{"tool": "add_entry", "args": {"amount": 45}}</TOOL_CALL>',
                         "The entry was recorded."])
    return _Wrapped(EgoStage(), **wrapper_kw), dispatcher, backend


# ── RED TWIN — the write is on the turn, and the row has its place and its label ────────────

async def test_red_twin_a_stop_after_the_executor_wrote_leaves_the_write_and_the_stamp():
    ego, dispatcher, backend = _real_writing_turn()
    superego = FakeSuperego()
    ctx = await _turn(ego, superego=superego, dispatcher=dispatcher, ego_backend=backend)

    # the premise: the tool RAN, once, and the turn was stopped from inside the executor
    assert dispatcher.executed == [("add_entry", {"amount": 45})]
    assert ctx.stop_reason == "completed" and ctx.superego_result.response == QUESTION
    assert superego._evals == 0 and mk.JUDGE_ATTEMPTS not in ctx.metadata

    # what this pass executed is on the TURN (was [])
    assert [(t.tool, t.ok, t.side_effect) for t in ctx.turn_executions] == [
        ("add_entry", True, True)]
    # …and its cost row has its place, its attempt and the host's label (was seq 0, sha "")
    row = ctx.ego_result.metrics
    assert (row.seq, row.attempt, row.prompt_sha) == (4, 1, EGO_LABEL)
    assert row.tokens_in > 0 and row.tokens_out > 0
    # the last row is the stop's own reply: no model wrote it, so nothing sequenced it
    assert [(m.stage, m.seq) for m in ctx.stage_metrics] == [
        ("noumeno", 1), ("ner", 2), ("id", 3), ("ego", 4), ("superego_voice", 0)]


async def test_constant_committed_this_turn_was_already_true_on_the_floor():
    """Not a repair: the predicate reads the surviving ``ego_result`` as a floor, so it said
    ``True`` before this change too. Pinned so nobody reads the twin above as "the commit bit
    was wrong" — the accumulator was, and the predicate never depended on it alone."""
    ego, dispatcher, backend = _real_writing_turn()
    ctx = await _turn(ego, dispatcher=dispatcher, ego_backend=backend)
    assert committed_this_turn(ctx) is True
    ctx.turn_executions.clear()
    assert committed_this_turn(ctx) is True, "the floor alone answers it"


async def test_red_twin_a_stop_on_the_retry_adds_that_pass_after_the_first():
    """Attempt 1 read and was rejected; attempt 2 wrote and the wrapper stopped. Both passes
    are on the turn, in order and once each, and each row keeps its own attempt."""
    ego = _Wrapped(FakeEgo(tool_calls=[[READ], [WRITE]]), on=(2,))
    ctx = await _turn(ego, superego=FakeSuperego(approve=False), budget=2)

    assert ego.invocations == 2 and ctx.stop_reason == "completed"
    assert [t.tool for t in ctx.turn_executions] == ["read_ledger", "add_entry"]   # was 1 entry
    assert [(m.seq, m.attempt, m.prompt_sha) for m in _ego_rows(ctx)] == [
        (4, 1, EGO_LABEL), (6, 2, EGO_LABEL)]                    # the second was (0, 0, "")
    assert len(ctx.metadata[mk.JUDGE_ATTEMPTS]) == 1, "only attempt 1 was judged"


# ── the control: a turn nobody stops is recorded exactly as it always was ───────────────────

async def test_control_a_turn_that_is_not_stopped_is_recorded_once_and_in_order():
    ego, dispatcher, backend = _real_writing_turn(on=())
    ctx = await _turn(ego, dispatcher=dispatcher, ego_backend=backend)

    assert ctx.stop_reason == "completed" and ctx.superego_result.response == "final reply"
    assert [t.tool for t in ctx.turn_executions] == ["add_entry"]
    row = ctx.ego_result.metrics
    assert (row.seq, row.attempt, row.prompt_sha) == (4, 1, EGO_LABEL)
    # the judge ran fifth; it sits in retry_metrics, which the context lists last
    assert [(m.stage, m.seq) for m in ctx.stage_metrics] == [
        ("noumeno", 1), ("ner", 2), ("id", 3), ("ego", 4), ("superego_voice", 6),
        ("superego_judge", 5)]


async def test_control_a_wrapper_that_never_stops_changes_nothing_of_the_turn():
    """The same turn with and without the wrapper, on deterministic doubles: every byte of the
    context is the same. The wrapper is the only thing this file adds to a turn."""
    plain = await _turn(FakeEgo(tool_calls=[WRITE], drafts=["Recorded."]))
    wrapped = await _turn(_Wrapped(FakeEgo(tool_calls=[WRITE], drafts=["Recorded."]), on=()))
    assert plain.turn_executions and plain.model_dump_json() == wrapped.model_dump_json()


# ── the guard: only a result THIS call produced is recorded ─────────────────────────────────

async def test_a_stop_before_any_result_records_nothing():
    """Attempt 1 stopped before the stage built a result: there is no row and no execution to
    record, and the stop is still the stop."""
    ctx = await _turn(_Wrapped(FakeEgo(tool_calls=[WRITE]), before=True))
    assert ctx.stop_reason == "completed" and ctx.superego_result.response == QUESTION
    assert ctx.ego_result is None and ctx.turn_executions == []
    assert [(m.stage, m.seq) for m in ctx.stage_metrics] == [
        ("noumeno", 1), ("ner", 2), ("id", 3), ("superego_voice", 0)]


async def test_a_stop_before_the_retry_replaced_the_result_does_not_record_the_first_pass_twice():
    """The NEGATIVE of the twin. Attempt 1 was recorded by the ordinary path; attempt 2 stopped
    before building anything, so the result on the context is still attempt 1's. Recording it
    again would list its executions twice and re-stamp its row as attempt 2."""
    ego = _Wrapped(FakeEgo(tool_calls=[[WRITE], [READ]]), on=(2,), before=True)
    ctx = await _turn(ego, superego=FakeSuperego(approve=False), budget=2)

    assert ego.invocations == 2 and ctx.stop_reason == "completed"
    assert [t.tool for t in ctx.turn_executions] == ["add_entry"], "once, not twice"
    assert [(m.seq, m.attempt) for m in _ego_rows(ctx)] == [(4, 1)]


async def test_a_stop_in_the_after_ego_hook_is_outside_the_executor_and_records_once():
    """A neighbour that must not move: the hook fires AFTER the loop, where the ordinary path
    has already recorded the pass. One record, not two."""
    def stop(ctx):
        raise StopPipeline(reason="completed", response=QUESTION)

    ctx = await _turn(FakeEgo(tool_calls=[WRITE]), hooks=Hooks(after_ego=stop))
    assert ctx.superego_result.response == QUESTION
    assert [t.tool for t in ctx.turn_executions] == ["add_entry"]
    assert [(m.seq, m.attempt) for m in _ego_rows(ctx)] == [(4, 1)]


# ── only the STOP is caught: everything else unwinds as it did ──────────────────────────────

@pytest.mark.parametrize("unwinding", [asyncio.CancelledError, RuntimeError])
async def test_anything_that_is_not_a_stop_still_unwinds_and_is_not_recorded(unwinding):
    """``asyncio.CancelledError`` (a ``BaseException``) and a stray error both leave ``run_turn``
    as they always did. The except is for the ONE signal ``run_turn`` turns into a returned
    context; a cancelled turn returns nothing, so it is not this code's to tidy."""
    ego = _Wrapped(FakeEgo(tool_calls=[WRITE]), raises=lambda: unwinding("not a stop"))
    raw = PipelineContext(user_input="record an entry of 45")
    with pytest.raises(unwinding):
        await _turn(ego, raw=raw)
    # the premise: the stage DID build a result with a write before it raised
    assert [t.tool for t in raw.ego_result.tools_executed] == ["add_entry"]
    assert raw.turn_executions == [] and raw.ego_result.metrics.seq == 0


# ── the boundary: a stop raised by a TOOL never arrives here as a stop ──────────────────────

class _StoppingTool(_WritingDispatcher):
    async def execute(self, name: str, arguments: dict) -> ToolResult:
        self.executed.append((name, arguments))
        raise StopPipeline(reason="completed", response=QUESTION)


async def test_boundary_a_stop_raised_by_a_tool_is_wrapped_by_the_executor_and_is_not_a_stop():
    """Out of this change's reach, and pinned so the reach is not overstated. cogno-anima's
    executor wraps anything that escapes ``dispatcher.execute`` in ``ToolExecutionError``, so a
    ``StopPipeline`` raised by a tool leaves ``run_turn`` as an ERROR, with no result built —
    there is no pass to record. A host that stops from inside the executor does it from a
    wrapper around the stage, which is the shape this file repairs."""
    dispatcher = _StoppingTool()
    backend = _Scripted(['<TOOL_CALL>{"tool": "add_entry", "args": {"amount": 45}}</TOOL_CALL>',
                         "The entry was recorded."])
    raw = PipelineContext(user_input="record an entry of 45")
    with pytest.raises(ToolExecutionError) as caught:
        await _turn(EgoStage(), dispatcher=dispatcher, ego_backend=backend, raw=raw)
    assert isinstance(caught.value.__cause__, StopPipeline)
    assert dispatcher.executed and raw.ego_result is None and raw.turn_executions == []


# ── a record that cannot be made never turns a stop into a crash ────────────────────────────

class _Unreadable:
    """A stand-in result whose executions cannot be read."""

    metrics = None

    @property
    def tools_executed(self):
        raise RuntimeError("unreadable result")


class _LeavesAnUnreadableResult:
    name = "ego"

    def __init__(self, *, stops: bool) -> None:
        self._stops = stops

    async def process(self, ctx, backend, dispatcher, *, system_prompt):
        ctx.ego_result = _Unreadable()
        if self._stops:
            raise StopPipeline(reason="completed", response=QUESTION)
        return ctx


async def test_a_record_that_fails_does_not_turn_the_stop_into_a_crash():
    ctx = await _turn(_LeavesAnUnreadableResult(stops=True))
    assert ctx.stop_reason == "completed" and ctx.superego_result.response == QUESTION
    assert ctx.turn_executions == []


async def test_control_the_same_unreadable_result_does_break_the_ordinary_path():
    """The control of the test above: the result really is unreadable. On a turn nobody stops
    the ordinary path raises on it, as it always did — so the test above is not passing
    because nothing went wrong."""
    with pytest.raises(RuntimeError, match="unreadable result"):
        await _turn(_LeavesAnUnreadableResult(stops=False))


# ── ONE definition: the stop path calls the function the ordinary path calls ────────────────

def _calls(tree, name: str) -> int:
    return sum(1 for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name)


def test_the_pass_is_recorded_by_one_function_and_the_loop_calls_it_on_both_paths():
    module = ast.parse(inspect.getsource(pipeline_module))
    extends = [n for n in ast.walk(module)
               if isinstance(n, ast.Attribute) and n.attr == "extend"
               and isinstance(n.value, ast.Attribute) and n.value.attr == "turn_executions"]
    assert len(extends) == 1, "what a pass executed joins the turn in ONE place"

    record = ast.parse(inspect.getsource(pipeline_module._record_ego_pass))
    assert _calls(record, "_stamp") == 1 and len(
        [n for n in ast.walk(record) if isinstance(n, ast.Attribute) and n.attr == "extend"]) == 1

    loop = ast.parse(textwrap.dedent(inspect.getsource(pipeline_module.Pipeline._run_ego_loop)))
    assert _calls(loop, "_record_ego_pass") == 2, "the ordinary path and the stop path"
    assert _calls(loop, "_stamp") == 0, "the loop stamps the executor only through it"
