"""Control-flow tests for ``Pipeline.run_turn`` using fake stages."""


from cogno_soma import Pipeline, TurnConfig

from tests.conftest import (
    FakeEgo,
    FakeID,
    FakeNER,
    FakeNoumeno,
    FakeSuperego,
    StubBackend,
)


def _pipeline(embedder, **stages):
    return Pipeline(
        embedder=embedder,
        noumeno=stages.get("noumeno", FakeNoumeno()),
        ner=stages.get("ner", FakeNER()),
        id_stage=stages.get("id_stage", FakeID(route="SUPEREGO")),
        ego=stages.get("ego", FakeEgo()),
        superego=stages.get("superego", FakeSuperego()),
    )


def _cfg(stub_backend, **kw):
    base = dict(gen_backend=stub_backend, ego_backend=stub_backend, ego_prompt="exec")
    base.update(kw)
    return TurnConfig(**base)


def _ctx(text="hi"):
    from cogno_anima.types import PipelineContext
    return PipelineContext(user_input=text)


class _SideEffectEgo(FakeEgo):
    """A FakeEgo whose trace records a committed mutating call (has_side_effects=True)."""

    async def process(self, ctx, backend, dispatcher, *, system_prompt):
        from cogno_anima.types import EgoResult, EgoStep, ToolExecution
        ctx = await super().process(ctx, backend, dispatcher, system_prompt=system_prompt)
        ctx.ego_result = EgoResult(
            steps=[EgoStep(index=0, path="native",
                           tool_calls=[ToolExecution(tool="book_appointment", ok=True,
                                                     side_effect=True)])],
            metrics=ctx.ego_result.metrics)
        return ctx


async def test_non_task_path_voices_response(stub_embedder, stub_backend, dispatcher):
    """SUPEREGO route (no EGO): goes straight to voice, never runs the EGO."""
    ego = FakeEgo()
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="SUPEREGO"), ego=ego)
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend), dispatcher=dispatcher)
    assert ctx.superego_result.response == "final reply"
    assert ctx.stop_reason == "completed"
    assert ego.invocations == 0


async def test_confirmed_calls_force_the_ego_route(stub_embedder, stub_backend, dispatcher):
    """Gate-B completion: a bare "sim" routes to SUPEREGO, but a pending confirmed call MUST
    still run the EGO so the approved action is executed (else it is silently dropped)."""
    ego = FakeEgo()
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="SUPEREGO"), ego=ego,
                     superego=FakeSuperego(approve=True))
    ctx = _ctx()
    ctx.metadata.update({"ego_confirmed": True,
                         "ego_confirmed_calls": [{"tool": "book_appointment", "arguments": {}}]})
    ctx = await pipe.run_turn(ctx, _cfg(stub_backend), dispatcher=dispatcher)
    assert ego.invocations == 1                       # EGO ran despite the SUPEREGO route


async def test_ego_route_runs_loop_then_voices(stub_embedder, stub_backend, dispatcher):
    ego = FakeEgo()
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=ego,
                     superego=FakeSuperego(approve=True))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend), dispatcher=dispatcher)
    assert ego.invocations == 1
    assert ctx.superego_result.response == "final reply"
    assert ctx.ego_result is not None


async def test_pii_critical_blocks_before_ego(stub_embedder, stub_backend, dispatcher):
    ego = FakeEgo()
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="SUPEREGO", blocked=True), ego=ego)
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend), dispatcher=dispatcher)
    assert ctx.stop_reason == "pii_blocked"
    assert ctx.superego_result.blocked is True
    assert ego.invocations == 0


async def test_scope_guard_blocks(stub_embedder, stub_backend, dispatcher):
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"),
                     superego=FakeSuperego(scope_blocked=True, refusal="nope"))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, scope_prompt="scope"), dispatcher=dispatcher)
    assert ctx.stop_reason == "scope_blocked"
    assert ctx.superego_result.response == "nope"
    assert ctx.superego_result.blocked is True


async def test_scope_guard_allows_then_continues(stub_embedder, stub_backend, dispatcher):
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="SUPEREGO"),
                     superego=FakeSuperego(scope_blocked=False))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, scope_prompt="scope"), dispatcher=dispatcher)
    assert ctx.stop_reason == "completed"
    assert ctx.superego_result.response == "final reply"


async def test_correction_loop_retries_until_budget(stub_embedder, stub_backend, dispatcher):
    """Judge rejects every attempt → loop runs max_corrections times. The EGO only READ (no
    side effect) so the turn ends in needs_clarification (voiced), not a dead-end handoff."""
    ego = FakeEgo()                         # EgoResult has no steps → has_side_effects is False
    sup = FakeSuperego(approve=False)
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=ego, superego=sup)
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=3), dispatcher=dispatcher)
    assert ego.invocations == 3
    assert ctx.needs_handoff is False
    assert ctx.stop_reason == "needs_clarification"
    assert ctx.superego_result.response == "final reply"   # voiced → conversation stays alive


async def test_reject_after_side_effect_hands_off(stub_embedder, stub_backend, dispatcher):
    """Judge rejects AND the EGO already committed a mutating call (side_effect) → hand off.
    Fail-closed: never voice an unverified action as done."""
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=_SideEffectEgo(),
                     superego=FakeSuperego(approve=False))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=3), dispatcher=dispatcher)
    assert ctx.needs_handoff is True
    assert ctx.stop_reason == "human_handoff"


async def test_correction_loop_recovers_and_voices(stub_embedder, stub_backend, dispatcher):
    """Judge rejects the first attempt, approves the second → voices, no handoff."""
    ego = FakeEgo()
    sup = FakeSuperego(approve_after=2)
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=ego, superego=sup)
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=3), dispatcher=dispatcher)
    assert ego.invocations == 2
    assert ctx.needs_handoff is False
    assert ctx.superego_result.response == "final reply"


async def test_correction_rerun_does_not_replay_confirmed_calls(stub_embedder, stub_backend,
                                                                dispatcher):
    """Gate-B replay is once-only: the confirmed calls execute on attempt 1 (their outcome is
    in the trace); a judge-rejected correction re-run must NOT replay them — a rejected-but-
    successful call would execute twice (double booking)."""
    seen_calls_per_attempt = []

    class RecordingEgo(FakeEgo):
        async def process(self, ctx, backend, dispatcher, *, system_prompt):
            seen_calls_per_attempt.append(ctx.metadata.get("ego_confirmed_calls"))
            return await super().process(ctx, backend, dispatcher, system_prompt=system_prompt)

    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="SUPEREGO"), ego=RecordingEgo(),
                     superego=FakeSuperego(approve_after=2))
    ctx = _ctx()
    ctx.metadata.update({"ego_confirmed": True,
                         "ego_confirmed_calls": [{"tool": "book_appointment", "arguments": {}}]})
    ctx = await pipe.run_turn(ctx, _cfg(stub_backend, max_corrections=3), dispatcher=dispatcher)
    assert len(seen_calls_per_attempt) == 2
    assert seen_calls_per_attempt[0]                    # attempt 1: replay list present
    assert not seen_calls_per_attempt[1]                # attempt 2: consumed — never replayed
    assert ctx.superego_result.response == "final reply"   # recovered and voiced


async def test_correction_feeds_critique_into_metadata(stub_embedder, stub_backend, dispatcher):
    ego = FakeEgo()
    sup = FakeSuperego(approve_after=2, critique="missing the amount")
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=ego, superego=sup)
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=3), dispatcher=dispatcher)
    # the rejected attempt's critique was wired into the correction metadata
    assert ctx.metadata["ego_correction"]["reason"] == "missing the amount"
    assert ctx.metadata["ego_correction"]["attempt"] == 2


async def test_ego_prompt_is_passed_through(stub_embedder, stub_backend, dispatcher):
    ego = FakeEgo()
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=ego)
    await pipe.run_turn(_ctx(), _cfg(stub_backend, ego_prompt="you are the executor"),
                        dispatcher=dispatcher)
    assert ego.last_system_prompt == "you are the executor"


async def test_retry_metrics_accumulate(stub_embedder, stub_backend, dispatcher):
    """Scope + each judge attempt land in ctx.retry_metrics."""
    sup = FakeSuperego(approve_after=2)
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), superego=sup)
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, scope_prompt="s", max_corrections=3),
                              dispatcher=dispatcher)
    stages = [m.stage for m in ctx.retry_metrics]
    assert "superego_scope" in stages
    assert stages.count("superego_judge") == 2


async def test_token_accounting_loses_nothing(stub_embedder, stub_backend, dispatcher):
    """Every LLM call's tokens must reach ctx for host billing — nothing dropped.

    Turn: scope on, EGO route, judge rejects attempt 1 then approves attempt 2.
    The nine LLM calls — NOUMENO, NER, ID, scope, ego#1 (rejected), judge#1,
    ego#2 (approved), judge#2, voice — each contribute their (1+1) tokens to
    ``ctx.total_tokens``. The rejected EGO attempt and every judge attempt ride in
    ``retry_metrics``; the final EGO + the voice ride in the per-stage metrics.
    """
    sup = FakeSuperego(approve_after=2)
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), superego=sup)
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, scope_prompt="s", max_corrections=3),
                              dispatcher=dispatcher)
    # 5 per-stage (noumeno/ner/id/ego-final/voice) + 4 retry (scope/judge#1/ego#1/judge#2)
    assert len(ctx.stage_metrics) == 9
    assert ctx.total_llm_tokens == 18          # 9 calls × (1 in + 1 out)
    assert ctx.total_tokens == 18              # no embeddings in the fakes
    stage_names = [m.stage for m in ctx.stage_metrics]
    for expected in ("noumeno", "ner", "id", "ego", "superego_scope",
                     "superego_judge", "superego_voice"):
        assert expected in stage_names, f"{expected} tokens dropped from accounting"


async def test_tokens_counted_on_handoff(stub_embedder, stub_backend, dispatcher):
    """Even when the turn ends in handoff, the spent tokens are still accounted."""
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=_SideEffectEgo(),
                     superego=FakeSuperego(approve=False))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=2), dispatcher=dispatcher)
    assert ctx.needs_handoff is True
    assert ctx.total_llm_tokens > 0            # the EGO + judge attempts are billed


# ── per-stage model routing: each stage runs on its own backend ─────────────────────────
class _RecNoumeno(FakeNoumeno):
    def __init__(self, log):
        super().__init__()
        self.log = log

    async def process(self, ctx, backend):
        self.log["noumeno"] = backend
        return await super().process(ctx, backend)


class _RecNER(FakeNER):
    def __init__(self, log):
        super().__init__()
        self.log = log

    async def process(self, ctx, backend):
        self.log["ner"] = backend
        return await super().process(ctx, backend)


class _RecEgo(FakeEgo):
    def __init__(self, log):
        super().__init__()
        self.log = log

    async def process(self, ctx, backend, dispatcher, *, system_prompt):
        self.log["ego"] = backend
        return await super().process(ctx, backend, dispatcher, system_prompt=system_prompt)


class _RecSuperego(FakeSuperego):
    def __init__(self, log):
        super().__init__(approve=True)
        self.log = log

    async def check_input_scope(self, ctx, backend, *, scope_prompt):
        self.log["scope"] = backend
        return await super().check_input_scope(ctx, backend, scope_prompt=scope_prompt)
    async def evaluate(self, ctx, backend, *, limits_prompt):
        self.log["judge"] = backend
        return await super().evaluate(ctx, backend, limits_prompt=limits_prompt)
    async def voice(self, ctx, backend, *, voice_prompt):
        self.log["voice"] = backend
        return await super().voice(ctx, backend, voice_prompt=voice_prompt)


async def test_per_stage_backends_route_independently(stub_embedder, dispatcher):
    """Each JSON stage runs on its OWN backend when pinned (NOUMENO/NER/scope/judge distinct)."""
    log: dict = {}
    b = {k: StubBackend() for k in ("noumeno", "ner", "ego", "scope", "judge", "voice", "gen")}
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"),
                     noumeno=_RecNoumeno(log), ner=_RecNER(log), ego=_RecEgo(log),
                     superego=_RecSuperego(log))
    cfg = TurnConfig(gen_backend=b["gen"], ego_backend=b["ego"], ego_prompt="exec",
                     scope_prompt="guard", noumeno_backend=b["noumeno"], ner_backend=b["ner"],
                     scope_backend=b["scope"], judge_backend=b["judge"], voice_backend=b["voice"])
    await pipe.run_turn(_ctx(), cfg, dispatcher=dispatcher)
    assert log["noumeno"] is b["noumeno"]
    assert log["ner"] is b["ner"]
    assert log["scope"] is b["scope"]
    assert log["judge"] is b["judge"]
    assert log["voice"] is b["voice"]
    assert log["ego"] is b["ego"]


async def test_escalate_bumps_the_ego_backend(stub_embedder, stub_backend, dispatcher):
    """The pipeline consults ``cfg.escalate`` (the host's complexity ladder) for the EGO backend."""
    log: dict = {}
    strong = StubBackend()
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=_RecEgo(log))
    # escalate returns a stronger backend for the EGO → the loop runs on it
    cfg = _cfg(stub_backend, escalate=lambda ctx, stage: strong if stage == "ego" else None)
    await pipe.run_turn(_ctx(), cfg, dispatcher=dispatcher)
    assert log["ego"] is strong
    # None (easy turn / not the ego stage) keeps the configured backend
    cfg2 = _cfg(stub_backend, escalate=lambda ctx, stage: None)
    await pipe.run_turn(_ctx(), cfg2, dispatcher=dispatcher)
    assert log["ego"] is stub_backend


async def test_unpinned_json_stage_falls_back_to_gen(stub_embedder, dispatcher):
    """A JSON stage left unset uses gen_backend (backward compatible); a pinned one overrides."""
    log: dict = {}
    gen, ner_b = StubBackend(), StubBackend()
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="SUPEREGO"),
                     noumeno=_RecNoumeno(log), ner=_RecNER(log), superego=_RecSuperego(log))
    cfg = TurnConfig(gen_backend=gen, ego_backend=StubBackend(), ego_prompt="exec",
                     ner_backend=ner_b)           # only NER pinned; NOUMENO unset → gen
    await pipe.run_turn(_ctx(), cfg, dispatcher=dispatcher)
    assert log["noumeno"] is gen                  # fell back to gen_backend
    assert log["ner"] is ner_b                    # its own model
