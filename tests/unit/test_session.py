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


async def test_pii_hint_carried_one_turn_rolling(stub_embedder, stub_backend):
    """PII shared this turn arms ``pii_session_hint`` for the NEXT turn only (the parent
    runner's ``_last_pii_risk``, rolling): the ID's anaphoric fast-path + lenient goal
    threshold read this key, and it was never fed by anything — the whole path was dead
    in production until this carry (found porting the parent's memory bench)."""
    from tests.conftest import FakeNER
    seen: list = []
    pipe = _pipe(stub_embedder)
    pipe._ner = FakeNER(pii=["NATIONAL_ID"])

    orig = pipe._id.process

    async def spy(ctx, embedder):
        seen.append(ctx.metadata.get("pii_session_hint"))
        return await orig(ctx, embedder)
    pipe._id.process = spy

    sess = SessionRunner(pipe, _cfg(stub_backend), dispatcher=RecordingDispatcher())
    await sess.run("meu CPF é 529.982.247-25")     # turn 1: PII shared → arms the hint
    assert seen[-1] is None                        # ...but only for the NEXT turn
    pipe._ner = FakeNER(pii=[])                    # turn 2: no new PII in this turn
    await sess.run("confirma se ficou certo?")
    assert seen[-1] is True                        # armed by turn 1's carry
    await sess.run("e o horário de amanhã?")       # turn 3: turn 2 had no PII
    assert seen[-1] is None                        # rolling: the hint decayed


async def test_pii_hint_host_override_wins(stub_embedder, stub_backend):
    """``run(metadata=...)`` merges last — a host that injects its own hint value is
    authoritative over the carry (same contract as every other carried key)."""
    from tests.conftest import FakeNER
    seen: list = []
    pipe = _pipe(stub_embedder)
    pipe._ner = FakeNER(pii=["CREDENTIAL"])

    orig = pipe._id.process

    async def spy(ctx, embedder):
        seen.append(ctx.metadata.get("pii_session_hint"))
        return await orig(ctx, embedder)
    pipe._id.process = spy

    sess = SessionRunner(pipe, _cfg(stub_backend), dispatcher=RecordingDispatcher())
    await sess.run("senha: hunter2222")
    await sess.run("segue igual?", metadata={"pii_session_hint": False})
    assert seen[-1] is False                       # host override beat the carry


async def test_pii_hint_survives_state_round_trip(stub_embedder, stub_backend):
    """The carry is part of ``.state`` — a rebuilt runner (multi-worker host) keeps the
    armed hint for the next turn."""
    from tests.conftest import FakeNER
    pipe = _pipe(stub_embedder)
    pipe._ner = FakeNER(pii=["PHONE"])
    sess = SessionRunner(pipe, _cfg(stub_backend), dispatcher=RecordingDispatcher())
    await sess.run("meu telefone é (11) 98482-1841")
    assert sess.state["carry"]["pii_session_hint"] is True

    seen: list = []
    pipe2 = _pipe(stub_embedder)
    orig = pipe2._id.process

    async def spy(ctx, embedder):
        seen.append(ctx.metadata.get("pii_session_hint"))
        return await orig(ctx, embedder)
    pipe2._id.process = spy
    sess2 = SessionRunner(pipe2, _cfg(stub_backend), dispatcher=RecordingDispatcher(),
                          state=sess.state)
    await sess2.run("pode confirmar?")
    assert seen[-1] is True


async def test_sources_instruction_keeps_memories_assertable(stub_embedder, stub_backend):
    """The anti-staleness rule must stay scoped to VOLATILE data. Its first wording forbade
    asserting ANY fact not backed by a fresh tool result — measured live: the operator's
    referral note was recalled into [MEMORIES] (rank #1) and the model still answered
    'não tenho acesso', obeying the instruction to the letter, twice, then stayed
    consistent with its own denial. Durable contact facts are saved to be USED."""
    captured: list[str] = []
    pipe = _pipe(stub_embedder, route="EGO")     # SUPEREGO-routed turns skip the executor
    orig = pipe._ego.process

    async def spy(ctx, backend, dispatcher, *, system_prompt):
        captured.append(str(ctx.metadata.get("ego_context", "")))
        return await orig(ctx, backend, dispatcher, system_prompt=system_prompt)
    pipe._ego.process = spy

    sess = SessionRunner(pipe, _cfg(stub_backend), dispatcher=RecordingDispatcher())
    await sess.run("quem te passou meu contato?",
                   memories=["Nota do operador: veio através do José Manzoli."])
    src = captured[0]
    assert "VOLATILE" in src                      # staleness rule survives, scoped
    assert "DURABLE" in src and "MAY state" in src
    assert "outrank" in src                       # beats the model's own earlier denial
    assert "[MEMORIES]" in src and "José Manzoli" in src


async def test_with_no_verbatim_window_the_recap_is_the_THREAD_not_background(
        stub_embedder, stub_backend):
    """"EARLIER CONTEXT is background" holds only while a RECENT CONVERSATION sits above it.

    With an empty burst window the recap is the ONLY account of the conversation the model
    gets, and calling it background tells the model to ignore everything it knows. Measured on
    a real WhatsApp conversation (2026-08): a 24h gap emptied the window, the recap was all
    that reached the model, and it re-opened the conversation from the start — greeting a
    contact it was mid-diagnosis with. A messaging session never rotates (`session_id` is
    derived from tenant+channel+sender), so this is not the rare case: it is every gap longer
    than the burst, for every contact.

    Mutation: use one instruction unconditionally and this dies."""
    captured: list[str] = []
    pipe = _pipe(stub_embedder, route="EGO")
    orig = pipe._ego.process

    async def spy(ctx, backend, dispatcher, *, system_prompt):
        captured.append(str(ctx.metadata.get("ego_context", "")))
        return await orig(ctx, backend, dispatcher, system_prompt=system_prompt)
    pipe._ego.process = spy

    sess = SessionRunner(pipe, _cfg(stub_backend), dispatcher=RecordingDispatcher())
    await sess.run("e aí?", prior_summary="Estávamos vendo os horários da clínica.")
    src = captured[0]
    assert "[RECENT CONVERSATION]" not in src, "premise: the burst window is empty"
    assert "ONLY account" in src
    assert "do NOT restart the conversation" in src
    assert "EARLIER CONTEXT / KNOWLEDGE GRAPH are background" not in src


async def test_once_the_conversation_IS_flowing_the_recap_goes_back_to_background(
        stub_embedder, stub_backend):
    """The other direction, and it must hold or the fix would promote a stale recap over the
    live thread — the 2026-07 doctor's-agenda fabrication in reverse."""
    captured: list[str] = []
    pipe = _pipe(stub_embedder, route="EGO")
    orig = pipe._ego.process

    async def spy(ctx, backend, dispatcher, *, system_prompt):
        captured.append(str(ctx.metadata.get("ego_context", "")))
        return await orig(ctx, backend, dispatcher, system_prompt=system_prompt)
    pipe._ego.process = spy

    sess = SessionRunner(pipe, _cfg(stub_backend), dispatcher=RecordingDispatcher())
    await sess.run("bom dia")                                   # seeds the burst window
    await sess.run("e aí?", prior_summary="Estávamos vendo os horários.")
    src = captured[-1]
    assert "[RECENT CONVERSATION]" in src, "premise: the window carried forward"
    assert "EARLIER CONTEXT / KNOWLEDGE GRAPH are background" in src
    assert "ONLY account" not in src


async def test_the_volatile_bar_survives_in_BOTH_wordings(stub_embedder, stub_backend):
    """Whatever else changes, a recap must never be restated as current data. Losing that on
    the no-transcript path would re-open the fabrication this layering exists to stop."""
    from cogno_soma.session import (_SOURCES_INSTRUCTION,
                                    _SOURCES_INSTRUCTION_NO_TRANSCRIPT)
    for text in (_SOURCES_INSTRUCTION, _SOURCES_INSTRUCTION_NO_TRANSCRIPT):
        assert "VOLATILE" in text
        assert "never restate them" in text
        assert "outrank" in text          # the referral-denial fix, on both paths
