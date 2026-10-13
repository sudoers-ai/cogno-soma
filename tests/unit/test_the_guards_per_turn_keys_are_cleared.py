"""A turn that never calls the scope guard carries none of the guard's per-turn keys.

``SuperegoStage.check_input_scope`` (cogno-anima) leaves three facts on ``ctx.metadata``, where
a trace writer can reach them: the inventory of the prompt it sent (``mk.SCOPE_PROMPT_BLOCKS``),
the digest of that prompt (``mk.SCOPE_PROMPT_SHA``) and how its verdict was read
(``mk.SCOPE_VERDICT_READ``). All three are PER-TURN, and the guard keeps them honest itself —
but only when it is CALLED. Two exits of ``run_turn`` never call it: a turn with no scope
prompt, and a turn the PII gate ends first. On those, a value a carrier left on the metadata
would be read as this turn's: a digest and an inventory of a prompt that was never rendered.

The third key was cleared before those exits already. This file is the same rule for the other
two, and one derivation so a fourth key cannot join the guard unseen.

A defence, not a served defect: ``SessionRunner`` builds a fresh context each turn and carries
a whitelist these keys are not on. A different carrier would have to make that choice itself.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import cogno_anima.metakeys as mk
import pytest
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import PipelineContext

from cogno_soma import Hooks, Pipeline, SessionRunner, StopPipeline, TurnConfig
from tests.conftest import (FakeEgo, FakeID, FakeNER, FakeNoumeno, RecordingDispatcher,
                            StubBackend, StubEmbedder)

STALE_SHA = "0" * 12
STALE_BLOCKS = [{"block": "scope_rules", "chars": 99999}]
STALE = {mk.SCOPE_PROMPT_SHA: STALE_SHA, mk.SCOPE_PROMPT_BLOCKS: STALE_BLOCKS}

# (scope prompt, the ID double, which exit this is)
NEVER_CALLS_THE_GUARD = [
    ("", None, "no scope prompt: the guard is never called"),
    ("only plans", FakeID(route="EGO", blocked=True), "the PII gate ends the turn before it"),
]


class _Voice:
    model = "voice"

    async def generate(self, system: str, prompt: str):
        return "The plan costs 10.", 30, 6


def _context(metadata=None) -> PipelineContext:
    ctx = PipelineContext(user_input="how much is the plan?")
    ctx.metadata.update({k: (list(v) if isinstance(v, list) else v)
                         for k, v in (metadata or {}).items()})
    return ctx


async def _turn(*, scope_prompt="", id_stage=None, metadata=None, intent="INFORMATION_REQUEST",
                hooks=None):
    """One turn with the REAL guard (the keys are anima's to write; a double writes none)."""
    pipe = Pipeline(embedder=StubEmbedder(), noumeno=FakeNoumeno("How much is the plan?"),
                    ner=FakeNER(intent_class=intent), id_stage=id_stage or FakeID(route="SUPEREGO"),
                    ego=FakeEgo(), superego=SuperegoStage())
    cfg = TurnConfig(gen_backend=StubBackend(), ego_backend=StubBackend(), ego_prompt="x",
                     voice_backend=_Voice(), scope_prompt=scope_prompt, hooks=hooks)
    return await pipe.run_turn(_context(metadata), cfg, dispatcher=RecordingDispatcher())


# ── the twin, one per key and per exit ─────────────────────────────────────────────────────

@pytest.mark.parametrize("key", [mk.SCOPE_PROMPT_SHA, mk.SCOPE_PROMPT_BLOCKS])
@pytest.mark.parametrize("scope_prompt, id_stage, why", NEVER_CALLS_THE_GUARD)
async def test_twin_a_turn_that_never_calls_the_guard_carries_no_stale_prompt_record(
        scope_prompt, id_stage, why, key):
    ctx = await _turn(scope_prompt=scope_prompt, id_stage=id_stage, metadata=STALE)
    assert key not in ctx.metadata, why


@pytest.mark.parametrize("scope_prompt, id_stage, why", NEVER_CALLS_THE_GUARD)
async def test_the_premise_those_two_exits_really_do_not_call_the_guard(
        scope_prompt, id_stage, why):
    """The twin's denominator: with nothing stale on the way in, nothing is on the way out —
    so on these exits no writer ran, and the absence above is the clearing's doing."""
    ctx = await _turn(scope_prompt=scope_prompt, id_stage=id_stage)
    assert mk.SCOPE_PROMPT_SHA not in ctx.metadata and mk.SCOPE_PROMPT_BLOCKS not in ctx.metadata
    assert not any(m.stage == "superego_scope" for m in ctx.stage_metrics), why


# ── the control: a turn that calls the guard keeps what the guard wrote ────────────────────

async def test_control_a_consulted_guard_leaves_its_own_record_not_the_stale_one():
    """The clearing is BEFORE the guard, never after. What the turn carries out is what the
    guard writes when called directly over the same turn — a real digest and a real inventory."""
    ctx = await _turn(scope_prompt="only plans", metadata=STALE)

    direct = _context()
    await FakeNoumeno("How much is the plan?").process(direct, None)
    await FakeNER().process(direct, None)
    result = await SuperegoStage().check_input_scope(direct, StubBackend(),
                                                     scope_prompt="only plans")
    assert result.prompt_sha and result.prompt_blocks, "the premise: a prompt was built"

    assert ctx.metadata[mk.SCOPE_PROMPT_SHA] == result.prompt_sha != STALE_SHA
    assert ctx.metadata[mk.SCOPE_PROMPT_BLOCKS] == result.prompt_blocks != STALE_BLOCKS


async def test_control_a_guard_that_is_called_and_bypasses_says_so_in_its_own_words():
    """Anima's rule for a bypass (a SOCIAL turn builds no prompt): the inventory is ``[]`` and
    the digest is absent. That is the guard speaking, and the clearing does not overrule it:
    ``[]`` — "called, no prompt built" — stays apart from absent — "never called"."""
    ctx = await _turn(scope_prompt="only plans", metadata=STALE, intent="SOCIAL")
    assert ctx.metadata[mk.SCOPE_PROMPT_BLOCKS] == []
    assert mk.SCOPE_PROMPT_SHA not in ctx.metadata


# ── a key from somewhere else is not this rule's to touch ──────────────────────────────────

@pytest.mark.parametrize("scope_prompt, id_stage, why", NEVER_CALLS_THE_GUARD)
async def test_nothing_else_on_the_metadata_is_cleared_with_them(scope_prompt, id_stage, why):
    """The adverse half: the host's own keys, and one that merely LOOKS like the guard's,
    come out as they went in."""
    mine = {mk.PROMPT_SHAS: {"scope": "9f3c1a"}, "scope_prompt_sha_of_the_host": "kept",
            "tenant": "acme"}
    ctx = await _turn(scope_prompt=scope_prompt, id_stage=id_stage, metadata={**STALE, **mine})
    for key, value in mine.items():
        assert ctx.metadata[key] == value, why


# ── the boundary: a stop BEFORE the gates is a third exit, and it is not covered ───────────

async def test_boundary_a_hook_that_stops_before_the_gates_ends_the_turn_before_the_clearing():
    """Out of this change's reach, and pinned so the reach is not overstated. The clearing sits
    where the first one did: after the ID, before the gates. A hook that raises
    ``StopPipeline`` earlier ends the turn first, and whatever the context arrived with — the
    verdict read included — is still on it. A carrier of the host's own drops the keys itself."""
    def stop(ctx):
        raise StopPipeline(reason="completed", response="Please hold.")

    stale = {**STALE, mk.SCOPE_VERDICT_READ: "string_bool"}
    ctx = await _turn(scope_prompt="only plans", metadata=stale, hooks=Hooks(after_ner=stop))
    assert ctx.superego_result.response == "Please hold."
    assert {k: ctx.metadata[k] for k in stale} == stale


# ── the declared scope: the session runner never carried them ──────────────────────────────

async def test_the_session_runner_does_not_carry_the_guards_record_to_the_next_turn():
    """Why this is a defence: turn 1 calls the guard, the session is persisted and resumed
    under a configuration with no scope prompt, and the runner's whitelist already keeps
    turn 1's record out of turn 2 — and out of the state a host persists."""
    pipe = Pipeline(embedder=StubEmbedder(), noumeno=FakeNoumeno("How much is the plan?"),
                    ner=FakeNER(), id_stage=FakeID(route="SUPEREGO"), ego=FakeEgo(),
                    superego=SuperegoStage())

    def cfg(scope_prompt: str) -> TurnConfig:
        return TurnConfig(gen_backend=StubBackend(), ego_backend=StubBackend(), ego_prompt="x",
                          voice_backend=_Voice(), scope_prompt=scope_prompt)

    guarded = SessionRunner(pipe, cfg("only plans"), dispatcher=RecordingDispatcher())
    first = await guarded.run("how much is the plan?")
    assert first.metadata[mk.SCOPE_PROMPT_SHA] and first.metadata[mk.SCOPE_PROMPT_BLOCKS]
    assert "scope_prompt" not in repr(guarded.state)

    resumed = SessionRunner(pipe, cfg(""), dispatcher=RecordingDispatcher(),
                            state=guarded.state)
    second = await resumed.run("and the yearly one?")
    assert mk.SCOPE_PROMPT_SHA not in second.metadata
    assert mk.SCOPE_PROMPT_BLOCKS not in second.metadata


# ── the derivation: every key the guard writes is one the pipeline clears ──────────────────

def _is_metakey(node) -> bool:
    return (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "mk")


def _metakeys(fn, *, assigned: bool) -> "set[str]":
    """The ``mk.<NAME>`` keys a function REMOVES (``….pop(mk.X, …)``) and, with ``assigned``,
    also the ones it ASSIGNS (``…[mk.X] = …``). A key it only reads is neither."""
    found: "set[str]" = set()
    for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(fn)))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "pop" and node.args and _is_metakey(node.args[0])):
            found.add(node.args[0].attr)
        if (assigned and isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
                and _is_metakey(node.slice)):
            found.add(node.slice.attr)
    return found


def test_every_metakey_the_guard_writes_is_cleared_before_the_exits_that_skip_it():
    """Read off both sources. The guard's keys come from anima's own function; the pipeline's
    clearings from ``run_turn``. A key the guard starts writing tomorrow fails here until the
    pipeline clears it too."""
    written = _metakeys(SuperegoStage.check_input_scope, assigned=True)
    assert {"SCOPE_PROMPT_BLOCKS", "SCOPE_PROMPT_SHA", "SCOPE_VERDICT_READ"} <= written, (
        "the reading found less than the three keys known today: it is not reading the guard")
    assert "SCOPE_PENDING_REQUEST" not in written, "a key the guard only READS is not its record"

    cleared = _metakeys(Pipeline.run_turn, assigned=False)
    assert written <= cleared, f"written by the guard, never cleared: {sorted(written - cleared)}"
