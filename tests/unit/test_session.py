"""Tests for SessionRunner: multi-turn threading + serializable state + dispatcher."""

import pytest

from cogno_soma import Pipeline, SessionRunner, TurnConfig
from cogno_soma.errors import SomaError

from tests.conftest import (
    FakeEgo,
    FakeID,
    FakeNER,
    FakeNoumeno,
    FakeSuperego,
    RecordingDispatcher,
)


def _pipe(embedder, *, route="SUPEREGO", rewritten="rewritten text", goal="goal-A", domains=None):
    return Pipeline(
        embedder=embedder,
        noumeno=FakeNoumeno(rewritten=rewritten),
        ner=FakeNER(goal=goal, domains=domains or ["finance"]),
        id_stage=FakeID(route=route),
        ego=FakeEgo(),
        superego=FakeSuperego(),
    )


def _cfg(backend):
    return TurnConfig(gen_backend=backend, ego_backend=backend, ego_prompt="x")


async def test_turn_number_increments(stub_embedder, stub_backend):
    sess = SessionRunner(_pipe(stub_embedder), _cfg(stub_backend),
                         dispatcher=RecordingDispatcher())
    await sess.run("first")
    await sess.run("second")
    assert sess.turn_number == 2


async def test_carry_threads_id_state_and_goal(stub_embedder, stub_backend):
    sess = SessionRunner(_pipe(stub_embedder, goal="track expenses"), _cfg(stub_backend),
                         dispatcher=RecordingDispatcher())
    await sess.run("turn one")
    # state captured the goal + id_state for the next turn
    st = sess.state
    assert st["carry"]["last_goal"] == "track expenses"
    assert st["carry"]["id_state"] == {"seen": True}
    assert st["carry"]["active_domains"] == ["finance"]


async def test_history_feeds_last_rewritten(stub_embedder, stub_backend):
    captured: list[str] = []
    pipe = _pipe(stub_embedder, rewritten="REWRITTEN-1")

    # wrap noumeno to capture what metadata the 2nd turn saw
    orig = pipe._noumeno.process

    async def spy(ctx, backend):
        captured.append(ctx.metadata.get("last_rewritten", ""))
        return await orig(ctx, backend)

    pipe._noumeno.process = spy  # type: ignore[method-assign]
    sess = SessionRunner(pipe, _cfg(stub_backend), dispatcher=RecordingDispatcher())
    await sess.run("t1")
    await sess.run("t2")
    assert captured == ["", "REWRITTEN-1"]  # 1st turn no history, 2nd sees prior rewrite


async def test_persona_and_module_stamped(stub_embedder, stub_backend):
    pipe = _pipe(stub_embedder)
    seen: dict = {}

    orig = pipe._noumeno.process

    async def spy(ctx, backend):
        seen.update(ctx.metadata)
        return await orig(ctx, backend)

    pipe._noumeno.process = spy  # type: ignore[method-assign]
    sess = SessionRunner(pipe, _cfg(stub_backend), dispatcher=RecordingDispatcher(),
                         persona_id="VET", mcp_module="veterinary", force_language="pt-BR")
    ctx = await sess.run("oi")
    assert seen["active_persona_id"] == "VET"
    assert seen["active_mcp_module"] == "veterinary"
    assert ctx.force_language == "pt-BR"


async def test_memories_injected_as_ego_context(stub_embedder, stub_backend):
    pipe = _pipe(stub_embedder)
    seen: dict = {}

    orig = pipe._id.process

    async def spy(ctx, embedder):
        seen["ego_context"] = ctx.metadata.get("ego_context")
        return await orig(ctx, embedder)

    pipe._id.process = spy  # type: ignore[method-assign]
    sess = SessionRunner(pipe, _cfg(stub_backend), dispatcher=RecordingDispatcher())
    await sess.run("what's my balance", memories=["balance is 100", "currency BRL"])
    # The SOURCES instruction always leads; memories land in their own labelled layer.
    assert "[SOURCES]" in seen["ego_context"]
    assert seen["ego_context"].endswith("[MEMORIES]\nbalance is 100\ncurrency BRL")


async def test_transcript_feeds_conversation_history(stub_embedder, stub_backend):
    # the 2nd turn must see the prior exchange (user text + the voiced assistant reply) so a
    # follow-up like a bare name resolves against what was actually said.
    pipe = _pipe(stub_embedder)
    seen: dict = {}
    orig = pipe._id.process

    async def spy(ctx, embedder):
        seen["ego_context"] = ctx.metadata.get("ego_context")
        return await orig(ctx, embedder)

    pipe._id.process = spy  # type: ignore[method-assign]
    sess = SessionRunner(pipe, _cfg(stub_backend), dispatcher=RecordingDispatcher())
    await sess.run("quero marcar com o cardiologista")
    await sess.run("Vinicius Vale")
    hist = seen["ego_context"]
    assert "[RECENT CONVERSATION]" in hist
    assert "User: quero marcar com o cardiologista" in hist
    assert "Assistant: final reply" in hist            # the voiced reply, not just the user text
    # and the transcript is in the serializable state (survives a worker handoff), now with a ts
    row = sess.state["transcript"][-1]
    assert row[:2] == ["Vinicius Vale", "final reply"] and isinstance(row[2], float)


async def test_stale_exchange_drops_out_of_verbatim_window(stub_embedder, stub_backend):
    # THE 2026-07 DOCTOR'S-AGENDA FABRICATION: a listing from days ago must NOT sit verbatim in
    # the next turn's context (that is where the voicer copied it over an empty fresh read). An
    # exchange older than the burst gap leaves [RECENT CONVERSATION] entirely.
    pipe = _pipe(stub_embedder)
    seen: dict = {}
    orig = pipe._id.process

    async def spy(ctx, embedder):
        seen["ego_context"] = ctx.metadata.get("ego_context")
        return await orig(ctx, embedder)

    pipe._id.process = spy  # type: ignore[method-assign]
    sess = SessionRunner(pipe, _cfg(stub_backend), dispatcher=RecordingDispatcher())
    await sess.run("aqui está sua agenda: consulta 13/07", now=1000.0)   # the "old listing" turn
    await sess.run("Oi", now=1000.0 + 3 * 24 * 3600)                     # 3 days later
    ctx_seen = seen["ego_context"]
    assert "13/07" not in ctx_seen                    # the stale listing is gone
    assert "[RECENT CONVERSATION]" not in ctx_seen    # nothing verbatim across the gap
    assert "[SOURCES]" in ctx_seen                     # but the sources guard still leads
    # a same-burst follow-up DOES stay verbatim
    await sess.run("e amanhã?", now=1000.0 + 3 * 24 * 3600 + 60)
    assert "User: Oi" in seen["ego_context"]


async def test_layers_are_ordered_by_authority(stub_embedder, stub_backend):
    pipe = _pipe(stub_embedder)
    seen: dict = {}
    orig = pipe._id.process

    async def spy(ctx, embedder):
        seen["ego_context"] = ctx.metadata.get("ego_context")
        return await orig(ctx, embedder)

    pipe._id.process = spy  # type: ignore[method-assign]
    sess = SessionRunner(pipe, _cfg(stub_backend), dispatcher=RecordingDispatcher())
    await sess.run("oi", memories=["prefers PIX"],
                   prior_summary="Earlier: discussed a July booking.",
                   graph_context="Dr. Vale — cardiologist")
    ctx_seen = seen["ego_context"]
    order = [ctx_seen.index(lbl) for lbl in
             ("[SOURCES]", "[EARLIER CONTEXT]", "[MEMORIES]", "[KNOWLEDGE GRAPH]")]
    assert order == sorted(order)                      # authority order preserved


async def test_transcript_window_is_bounded(stub_embedder, stub_backend):
    sess = SessionRunner(_pipe(stub_embedder), _cfg(stub_backend),
                         dispatcher=RecordingDispatcher(), max_history=2)
    for i in range(5):
        await sess.run(f"turn {i}")
    assert len(sess.state["transcript"]) == 2           # only the last 2 exchanges kept
    assert len(sess.state["history"]) == 2              # history bounded too (only its tail is read)


async def test_state_round_trip_resumes_session(stub_embedder, stub_backend):
    sess = SessionRunner(_pipe(stub_embedder, goal="g1"), _cfg(stub_backend),
                         dispatcher=RecordingDispatcher())
    await sess.run("t1")
    snapshot = sess.state

    # a fresh worker reconstructs from the persisted snapshot
    sess2 = SessionRunner(_pipe(stub_embedder, goal="g2"), _cfg(stub_backend),
                          dispatcher=RecordingDispatcher(), state=snapshot)
    assert sess2.turn_number == 1
    await sess2.run("t2")
    assert sess2.turn_number == 2
    assert sess2.state["carry"]["last_goal"] == "g2"


async def test_dispatcher_factory_called_per_turn(stub_embedder, stub_backend):
    built: list[RecordingDispatcher] = []

    def factory():
        d = RecordingDispatcher()
        built.append(d)
        return d

    sess = SessionRunner(_pipe(stub_embedder), _cfg(stub_backend), dispatcher_factory=factory)
    await sess.run("t1")
    await sess.run("t2")
    assert len(built) == 2  # a fresh dispatcher per turn


async def test_run_dispatcher_override_wins(stub_embedder, stub_backend):
    override = RecordingDispatcher()
    sess = SessionRunner(_pipe(stub_embedder), _cfg(stub_backend),
                         dispatcher=RecordingDispatcher())
    ctx = await sess.run("t1", dispatcher=override)
    assert ctx is not None  # ran with the override, no error


async def test_missing_dispatcher_raises(stub_embedder, stub_backend):
    sess = SessionRunner(_pipe(stub_embedder), _cfg(stub_backend))
    with pytest.raises(SomaError, match="no dispatcher"):
        await sess.run("t1")


async def test_metadata_override_merges_last(stub_embedder, stub_backend):
    pipe = _pipe(stub_embedder)
    seen: dict = {}

    orig = pipe._noumeno.process

    async def spy(ctx, backend):
        seen.update(ctx.metadata)
        return await orig(ctx, backend)

    pipe._noumeno.process = spy  # type: ignore[method-assign]
    sess = SessionRunner(pipe, _cfg(stub_backend), dispatcher=RecordingDispatcher())
    await sess.run("t1", metadata={"ego_max_steps": 8, "turn_number": 99})
    assert seen["ego_max_steps"] == 8
    assert seen["turn_number"] == 99  # host override beats the auto-increment
