"""Control-flow tests for ``Pipeline.run_turn`` using fake stages."""


import cogno_anima.metakeys as mk
from cogno_anima.types import committed_this_turn

from cogno_soma import Pipeline, TurnConfig

from cogno_anima.types import SuperegoResult

from tests.conftest import (
    FakeEgo,
    FakeID,
    FakeNER,
    FakeNoumeno,
    FakeSuperego,
    RecordingDispatcher,
    StubBackend,
    metrics,
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


class _FailedMutationEgo(FakeEgo):
    """A FakeEgo whose mutating call FAILED (ok=False) — nothing was committed even though
    the tool is side-effecting (e.g. the confirmed slot was taken between propose and commit)."""

    async def process(self, ctx, backend, dispatcher, *, system_prompt):
        from cogno_anima.types import EgoResult, EgoStep, ToolExecution
        ctx = await super().process(ctx, backend, dispatcher, system_prompt=system_prompt)
        ctx.ego_result = EgoResult(
            steps=[EgoStep(index=0, path="native",
                           tool_calls=[ToolExecution(tool="book_appointment", ok=False,
                                                     side_effect=True, error="slot taken")])],
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
    # the judge's verdict reaches the voice — otherwise it narrates the goal as done
    assert ctx.metadata["voice_correction"]["reason"] == "fix it"


async def test_plan_max_self_corrections_caps_the_loop(stub_embedder, stub_backend, dispatcher):
    """The per-plan correction budget (plan_limits.max_self_corrections, injected by the host's
    BudgetGuard) OVERRIDES cfg.max_corrections — a Free tenant (1) does ONE EGO attempt even when
    the config would allow 3. Without plan_limits it falls back to cfg (see the test above)."""
    ego = FakeEgo()
    sup = FakeSuperego(approve=False)     # judge rejects every attempt → would retry up to the budget
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=ego, superego=sup)
    ctx = _ctx()
    ctx.metadata["plan_limits"] = {"max_self_corrections": 1}   # Free tier: 1 attempt, no retry
    ctx = await pipe.run_turn(ctx, _cfg(stub_backend, max_corrections=3), dispatcher=dispatcher)
    assert ego.invocations == 1           # plan cap (1) wins over cfg.max_corrections (3)


class _PendingConfirmEgo(FakeEgo):
    """A FakeEgo that HELD a destructive call for confirmation (Gate B) — nothing executed."""

    async def process(self, ctx, backend, dispatcher, *, system_prompt):
        from cogno_anima.types import EgoResult, EgoStep, ToolExecution
        ctx = await super().process(ctx, backend, dispatcher, system_prompt=system_prompt)
        held = ToolExecution(tool="book_appointment", ok=False, error="needs_confirmation",
                             result="[PENDING CONFIRMATION] held")
        ctx.ego_result = EgoResult(steps=[EgoStep(index=0, path="native", tool_calls=[held])],
                                   pending_confirmation=[held], metrics=ctx.ego_result.metrics)
        return ctx


async def test_gate_b_hold_skips_judge_and_retry(stub_embedder, stub_backend, dispatcher):
    """A Gate-B confirmation hold is a PROPOSE turn — the judge is skipped entirely (it would
    false-reject 'booking not completed' and burn a wasteful retry on every confirmation)."""
    ego = _PendingConfirmEgo()
    sup = FakeSuperego(approve=False)      # the judge WOULD reject — but must not be consulted
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=ego, superego=sup)
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=3), dispatcher=dispatcher)
    assert ego.invocations == 1                          # no retry
    assert sup._evals == 0                               # judge NEVER consulted for a Gate-B hold
    assert ctx.stop_reason != "needs_clarification"      # not a false rejection
    assert ctx.superego_result.response == "final reply"  # voiced the proposal


async def test_reject_after_side_effect_hands_off(stub_embedder, stub_backend, dispatcher):
    """Judge rejects AND the EGO already committed a mutating call (side_effect) → hand off.
    Fail-closed: never voice an unverified action as done."""
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=_SideEffectEgo(),
                     superego=FakeSuperego(approve=False))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=3), dispatcher=dispatcher)
    assert ctx.needs_handoff is True
    assert ctx.stop_reason == "human_handoff"


async def test_reject_after_failed_mutation_stays_alive(stub_embedder, stub_backend, dispatcher):
    """Judge rejects AND the only mutating call FAILED (nothing committed — e.g. the confirmed
    slot was taken) → needs_clarification, NOT handoff: voice a truthful continuation."""
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=_FailedMutationEgo(),
                     superego=FakeSuperego(approve=False))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=2), dispatcher=dispatcher)
    assert ctx.needs_handoff is False
    assert ctx.stop_reason == "needs_clarification"
    assert ctx.superego_result.response == "final reply"   # voiced → conversation stays alive
    assert ctx.metadata["voice_correction"]["reason"] == "fix it"


async def test_correction_loop_recovers_and_voices(stub_embedder, stub_backend, dispatcher):
    """Judge rejects the first attempt, approves the second → voices, no handoff."""
    ego = FakeEgo()
    sup = FakeSuperego(approve_after=2)
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=ego, superego=sup)
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=3), dispatcher=dispatcher)
    assert ego.invocations == 2
    assert ctx.needs_handoff is False
    assert ctx.superego_result.response == "final reply"
    assert "voice_correction" not in ctx.metadata          # approved → no rejection reaches the voice


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

# ── two-tier judge (judge_fast_backend) ─────────────────────────────────────

class _BackendAwareSuperego(FakeSuperego):
    """A FakeSuperego whose judge verdict depends on WHICH backend evaluates:
    the backend's ``model`` is looked up in ``verdicts`` (default approve)."""

    def __init__(self, verdicts: dict, **kw) -> None:
        super().__init__(**kw)
        self._verdicts = verdicts
        self.judged_with: list = []

    async def evaluate(self, ctx, backend, *, limits_prompt):
        model = getattr(backend, "model", "unknown")
        self.judged_with.append(model)
        approved = self._verdicts.get(model, True)
        return SuperegoResult(response="", approved=approved,
                              critique=None if approved else self._critique,
                              metrics=metrics("superego_judge"))


def _two_backends():
    fast, strong = StubBackend(), StubBackend()
    fast.model, strong.model = "fast-judge", "strong-judge"
    return fast, strong


async def test_two_tier_fast_approve_is_final(stub_embedder, stub_backend, dispatcher):
    """Fast judge approves → the strong judge is never consulted (the cost bet)."""
    fast, strong = _two_backends()
    sup = _BackendAwareSuperego({"fast-judge": True})
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), superego=sup)
    ctx = await pipe.run_turn(
        _ctx(), _cfg(stub_backend, judge_backend=strong, judge_fast_backend=fast),
        dispatcher=dispatcher)
    assert sup.judged_with == ["fast-judge"]
    assert ctx.superego_result.response == "final reply"
    assert "voice_correction" not in ctx.metadata


async def test_two_tier_fast_reject_escalates_and_strong_overrides(
        stub_embedder, stub_backend, dispatcher):
    """Fast rejects (the over-block case) → strong re-judges and its APPROVE wins:
    no correction retry is spent on a fast false-reject."""
    fast, strong = _two_backends()
    ego = FakeEgo()
    sup = _BackendAwareSuperego({"fast-judge": False, "strong-judge": True})
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=ego, superego=sup)
    ctx = await pipe.run_turn(
        _ctx(), _cfg(stub_backend, judge_backend=strong, judge_fast_backend=fast,
                     max_corrections=3),
        dispatcher=dispatcher)
    assert sup.judged_with == ["fast-judge", "strong-judge"]
    assert ego.invocations == 1                       # no retry was triggered
    assert "voice_correction" not in ctx.metadata


async def test_two_tier_both_reject_drives_correction(stub_embedder, stub_backend, dispatcher):
    """Both tiers reject → the correction loop runs (strong critique feeds the retry),
    and every judge call of every attempt is billed in retry_metrics."""
    fast, strong = _two_backends()
    ego = FakeEgo()
    sup = _BackendAwareSuperego({"fast-judge": False, "strong-judge": False},
                                critique="wrong tool")
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=ego, superego=sup)
    ctx = await pipe.run_turn(
        _ctx(), _cfg(stub_backend, judge_backend=strong, judge_fast_backend=fast,
                     max_corrections=2),
        dispatcher=dispatcher)
    # 2 attempts × (fast + strong)
    assert sup.judged_with == ["fast-judge", "strong-judge"] * 2
    assert ego.invocations == 2
    assert ctx.metadata["voice_correction"]["reason"] == "wrong tool"
    assert [m.stage for m in ctx.retry_metrics].count("superego_judge") == 4


async def test_two_tier_same_backend_judges_once(stub_embedder, stub_backend, dispatcher):
    """judge_fast_backend IS the strong judge → degrade to single-tier (no double call)."""
    fast, _ = _two_backends()
    sup = _BackendAwareSuperego({"fast-judge": False}, critique="nope")
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), superego=sup)
    await pipe.run_turn(
        _ctx(), _cfg(stub_backend, judge_backend=fast, judge_fast_backend=fast,
                     max_corrections=1),
        dispatcher=dispatcher)
    assert sup.judged_with == ["fast-judge"]


async def test_rejection_kind_separates_an_unverified_claim_from_a_failed_action(
        stub_embedder, stub_backend, dispatcher):
    """Two different rejections used to wear one signal, and the voice prompt only ever
    covered the ACTION one ("nothing was committed"). A persona with no tools executes
    nothing, so the verdict is about what the draft CLAIMS — and the refused claim was
    re-voiced verbatim (live: "Sim, o Cogno integra com o Bling", rejected twice, delivered).
    The kind is what lets the voice — and the host's escalation policy — tell them apart."""
    # FakeEgo has no tools_executed → nothing ran at all
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=FakeEgo(),
                     superego=FakeSuperego(approve=False))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=2),
                              dispatcher=dispatcher)
    assert ctx.metadata["voice_correction"]["kind"] == "unverified_claim"

    # a mutation that RAN and failed is a different animal: there is a trace to ground in
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=_FailedMutationEgo(),
                     superego=FakeSuperego(approve=False))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=2),
                              dispatcher=dispatcher)
    assert ctx.metadata["voice_correction"]["kind"] == "not_executed"


async def test_the_judge_verdict_is_recorded_so_it_can_be_counted(
        stub_embedder, stub_backend, dispatcher):
    """Only REJECTIONS were observable: they log at WARNING while approvals log at INFO, which a
    deployment's handlers may drop. "No approvals in the log" then reads identically to "the
    judge approves nothing" — and a full day went into chasing the wrong one. A rate needs a
    denominator, so the outcome rides on the context."""
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=FakeEgo(),
                     superego=FakeSuperego(approve=True))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=2),
                              dispatcher=dispatcher)
    assert ctx.metadata["judge_verdict"] == {"approved": True, "attempts": 1}

    # and a turn that burned the whole budget reports the attempts it took
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=FakeEgo(),
                     superego=FakeSuperego(approve=False))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=3),
                              dispatcher=dispatcher)
    assert ctx.metadata["judge_verdict"]["approved"] is False
    assert ctx.metadata["judge_verdict"]["attempts"] == 3


async def test_every_judge_verdict_is_recorded_with_its_critique(
        stub_embedder, stub_backend, dispatcher):
    """The count says how many; this says WHY — and the why is the half that was missing.

    A rejected attempt's critique went into ``EGO_CORRECTION.reason``, was overwritten by the
    next one, and then dropped. After the turn nothing survived but the number, so a red bench
    check could not be explained without attaching a debugger to the judge — done twice in one
    day, after three wrong hypotheses. Worse, contradictory critiques across attempts (rejecting
    a turn for listing rows, then for not listing them) were invisible: exactly the shape that
    tells you the criteria, not the execution, are what is broken.

    Approvals are recorded too. A list holding only rejections would make "approved on the
    second try" indistinguishable from "rejected once and gave up"."""
    sup = FakeSuperego(approve=False, critique="the execution did not do X")
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=FakeEgo(), superego=sup)
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=3),
                              dispatcher=dispatcher)
    attempts = ctx.metadata["judge_attempts"]
    assert [a["attempt"] for a in attempts] == [1, 2, 3]
    assert all(a["approved"] is False for a in attempts)
    assert all(a["critique"] == "the execution did not do X" for a in attempts)

    # …and a turn that recovers records the rejection AND the approval that followed it
    sup = FakeSuperego(approve=False, approve_after=2, critique="wrong row")
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=FakeEgo(), superego=sup)
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=3),
                              dispatcher=dispatcher)
    attempts = ctx.metadata["judge_attempts"]
    assert [a["approved"] for a in attempts] == [False, True]
    assert attempts[0]["critique"] == "wrong row"


async def test_a_rejected_attempts_writes_survive_into_the_record(
        stub_embedder, stub_backend, dispatcher):
    """The dangerous half of a rejected attempt is not the critique — it is what it WROTE.

    `ctx = await self._ego.process(...)` replaces `ego_result` on every retry, so a write made
    by attempt 1 vanished from everything downstream while the write itself had COMMITTED
    (nothing rolls back). Measured on doctor-notify (2026-08): attempt 1 wrongly confirmed all
    three rows, the judge rightly rejected it, attempt 2 truthfully said "no pendings", and the
    turn shipped hiding three commits. The critiques became data in #30; this pins the other
    eye: every JUDGE_ATTEMPTS entry carries that attempt's executions, side effects marked."""
    from cogno_anima.types import ToolExecution

    # DIFFERENT tools per attempt, and that is the test's whole discriminating power: the
    # review MUTATION-PROVED that with identical per-attempt tools, an implementation that
    # backfills every entry with the LAST attempt's tools — the exact stale-ego_result bug
    # class this fix targets, re-manifested inside the fix — stayed green.
    wrong = ToolExecution(tool="confirm_appointment", arguments={"appointment_id": "e4d1a201"},
                          result="Appointment e4d1a201 is now CONFIRMED.", ok=True,
                          side_effect=True)
    right = ToolExecution(tool="list_appointments", arguments={},
                          result="No PENDING appointments found.", ok=True, side_effect=False)
    sup = FakeSuperego(approve=False, approve_after=2, critique="wrong rows")
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"),
                     ego=FakeEgo(tool_calls=[[wrong], [right]]), superego=sup)
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=3),
                              dispatcher=dispatcher)
    attempts = ctx.metadata["judge_attempts"]
    assert [a["approved"] for a in attempts] == [False, True]
    rejected = attempts[0]["tools"]
    assert [t["tool"] for t in rejected] == ["confirm_appointment"], \
        "attempt 1 must carry ITS OWN write — not a backfill of the final attempt"
    assert rejected[0]["side_effect"] is True and rejected[0]["ok"] is True
    assert "e4d1a201" in rejected[0]["args"]          # WHICH row — the point of keeping args
    assert [t["tool"] for t in attempts[1]["tools"]] == ["list_appointments"], \
        "attempt 2 must carry attempt 2's tools — the entries are a per-attempt diff"


async def test_each_attempt_records_the_surface_it_was_OFFERED(
        stub_embedder, stub_backend, dispatcher):
    """The other half of the same eye: not what the attempt DID, but what it COULD have done.

    `tools_offered` lives on `EgoResult`, which the correction loop replaces every retry, so
    only the surviving attempt's surface reached a reader. The turn this pins is the real one:
    attempt 1 is offered `book_appointment` and books; the judge rejects; attempt 2 runs
    read-only masked and can only list. Reading the survivor alone, the record says the write
    tool "was never on the table" for a turn that booked.

    Why `tools` alone cannot answer it: an empty `tools` is "the model declined" and "the model
    was never given the option" at the same time, and those have OPPOSITE fixes — one a prompt
    problem, one a wiring problem. This project has spent whole rounds on the wrong one.

    The surfaces DIFFER per attempt, and that is this test's whole discriminating power: with
    identical ones, "per attempt" and "last, copied" are indistinguishable — the same mutation
    that already proved the executions test insufficient (see `_per_attempt` in conftest).
    """
    from cogno_anima.types import ToolExecution

    booked = ToolExecution(tool="book_appointment", arguments={"slot": "14:00"},
                           result="Booked 14:00.", ok=True, side_effect=True)
    listed = ToolExecution(tool="list_appointments", arguments={},
                           result="1 appointment.", ok=True, side_effect=False)
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"),
                     ego=FakeEgo(tool_calls=[[booked], [listed]],
                                 tools_offered=[["book_appointment", "list_appointments"],
                                                ["list_appointments"]]),
                     superego=FakeSuperego(approve=False, approve_after=2, critique="wrong slot"))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=3),
                              dispatcher=dispatcher)

    attempts = ctx.metadata["judge_attempts"]
    assert [a["approved"] for a in attempts] == [False, True]
    assert attempts[0]["tools_offered"] == ["book_appointment", "list_appointments"], \
        "attempt 1 must carry ITS OWN surface — not a backfill of the final attempt"
    assert attempts[1]["tools_offered"] == ["list_appointments"], \
        "attempt 2 ran masked: the write tool was NOT on its table, and the record must say so"
    assert "book_appointment" not in attempts[1]["tools_offered"], \
        "the masked attempt inherited the earlier surface — the entries are a per-attempt diff"


async def test_a_tools_result_and_args_are_truncated_in_the_record(
        stub_embedder, stub_backend, dispatcher):
    """Same rule as the critique: this rides in metadata the host persists, and a tool result
    is unbounded model-facing prose while arguments can embed the user's own text."""
    from cogno_anima.types import ToolExecution

    from cogno_soma.pipeline import _TOOL_ARGS_CHARS, _TOOL_RESULT_CHARS

    huge = ToolExecution(tool="notify_user",
                         arguments={"message": "x" * (_TOOL_ARGS_CHARS * 3)},
                         result="y" * (_TOOL_RESULT_CHARS * 3), ok=True, side_effect=True)
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"),
                     ego=FakeEgo(tool_calls=[huge]),
                     superego=FakeSuperego(approve=False))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=1),
                              dispatcher=dispatcher)
    rec = ctx.metadata["judge_attempts"][0]["tools"][0]
    # o corte carrega o MARCADOR: JSON cortado sem sinal se apresenta como inteiro-porém-
    # quebrado, e o leitor não distingue truncamento de corrupção
    assert rec["result"].endswith("…") and len(rec["result"]) == _TOOL_RESULT_CHARS + 1
    assert rec["args"].endswith("…") and len(rec["args"]) == _TOOL_ARGS_CHARS + 1


async def test_a_blocked_call_keeps_BOTH_its_error_and_its_prose(
        stub_embedder, stub_backend, dispatcher):
    """`result or error` derrubava a taxonomia do erro quando os dois existem — e as execuções
    que a anima SINTETIZA (blocked_retry / duplicate) preenchem os dois: o rótulo curto no
    `error` e a prosa que vai ao modelo no `result`. Quem lê a evidência quer os dois."""
    from cogno_anima.types import ToolExecution

    blocked = ToolExecution(tool="confirm_appointment", arguments={}, ok=False,
                            error="blocked_retry",
                            result="[BLOCKED] 'confirm_appointment' already FAILED.")
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"),
                     ego=FakeEgo(tool_calls=[blocked]),
                     superego=FakeSuperego(approve=False))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=1),
                              dispatcher=dispatcher)
    rec = ctx.metadata["judge_attempts"][0]["tools"][0]
    assert "blocked_retry" in rec["result"] and "[BLOCKED]" in rec["result"]


async def test_the_offered_cap_is_reported_on_BOTH_paths(
        stub_embedder, stub_backend, dispatcher):
    """`tools_offered_dropped` não tinha teste nenhum, e a assimetria entre os dois retornos
    passou despercebida por isso: o caminho saudável emitia o contador e o DEGRADADO cortava em
    silêncio. Um nome ausente lê-se como "nunca foi ofertado" — a leitura que este campo existe
    para acabar —, portanto o ramo degradado a reintroduzia.

    Achado do `/code-review` pós-merge do #33, que eu tinha mergeado sem revisão.

    Mutação: emitir o contador só no `entry`, ou trocar `offered_cut` por `if False` — e este
    morre nos dois casos."""
    from cogno_anima.types import ToolExecution

    from cogno_soma.pipeline import _TOOLS_PER_ATTEMPT

    muitos = [f"tool_{i:02d}" for i in range(_TOOLS_PER_ATTEMPT + 5)]

    class _Cursed:
        def __str__(self):
            raise RuntimeError("str() explode")

    ok = ToolExecution(tool="t", arguments={}, result="ok", ok=True, side_effect=False)
    odd = ToolExecution(tool="t", arguments={"x": _Cursed()}, result="ok", ok=True,
                        side_effect=True)

    for nome, exec_, degradado in (("saudável", ok, False), ("degradado", odd, True)):
        pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"),
                         ego=FakeEgo(tool_calls=[exec_], tools_offered=muitos),
                         superego=FakeSuperego(approve=False))
        ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=1),
                                  dispatcher=dispatcher)
        rec = ctx.metadata["judge_attempts"][0]
        assert ("tools_error" in rec) is degradado, f"o caminho {nome} não é o que se pensa"
        assert len(rec["tools_offered"]) == _TOOLS_PER_ATTEMPT, nome
        assert rec.get("tools_offered_dropped") == 5, (
            f"o caminho {nome} cortou {5} nomes e não disse — ausência lê-se como "
            f"'nunca foi ofertado'")


async def test_a_catalog_that_FITS_carries_no_dropped_marker(
        stub_embedder, stub_backend, dispatcher):
    """O gémeo: sem ele, emitir o contador sempre passaria no teste acima."""
    from cogno_anima.types import ToolExecution

    ok = ToolExecution(tool="t", arguments={}, result="ok", ok=True, side_effect=False)
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"),
                     ego=FakeEgo(tool_calls=[ok], tools_offered=["a", "b"]),
                     superego=FakeSuperego(approve=False))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=1),
                              dispatcher=dispatcher)
    rec = ctx.metadata["judge_attempts"][0]
    assert rec["tools_offered"] == ["a", "b"] and "tools_offered_dropped" not in rec


async def test_a_value_whose_str_raises_does_not_kill_the_turn(
        stub_embedder, stub_backend, dispatcher):
    """`default=str` fecha só um dos caminhos de raise do json.dumps. O do review (chave de
    dict não-string) foi REFUTADO medindo: o pydantic do ToolExecution recusa a chave na
    CONSTRUÇÃO, então ela nunca chega ao dumps. O caminho residual real é pelo VALOR — o
    `arguments` é dict[str, Any], um Any aceita objeto arbitrário, e `default=str` chama
    str(obj), que pode levantar. O registro inteiro é embrulhado: entrada degradada com
    `tools_error` em vez de turno morto."""
    from cogno_anima.types import ToolExecution

    class Cursed:
        def __str__(self):
            raise RuntimeError("str() explode")

    odd = ToolExecution(tool="book_appointment",
                        arguments={"when": Cursed()},
                        result="ok", ok=True, side_effect=True)
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"),
                     ego=FakeEgo(tool_calls=[odd],
                                 tools_offered=["book_appointment", "list_appointments"]),
                     superego=FakeSuperego(approve=False))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=1),
                              dispatcher=dispatcher)
    rec = ctx.metadata["judge_attempts"][0]
    assert rec["tools"] == [] and rec["tools_error"] == "RuntimeError"
    # …e a superfície OFERTADA sobrevive à degradação, que é o único momento em que ela é
    # indispensável: `tools: []` degradado é indistinguível de "não chamou nada", e é
    # exactamente essa ambiguidade que o campo existe para desfazer. Por isso ele é calculado
    # FORA do try, ao lado do `committed`.
    assert rec["tools_offered"] == ["book_appointment", "list_appointments"], \
        "a entrada degradada perdeu a superfície ofertada — resta o lado ambíguo e nenhuma resposta"


async def test_a_write_from_a_REJECTED_attempt_STILL_HANDS_OFF(
        stub_embedder, stub_backend, dispatcher):
    """A forma do doctor-notify, do lado da POLÍTICA do turno.

    Tentativa 1 confirma três linhas erradas (`side_effect` + `ok`), o juiz reprova; tentativa 2
    só LÊ e é reprovada também. O `ctx.ego_result` é SUBSTITUÍDO a cada retry, então o turno
    saía do laço parecendo que não executou nada — e ia para `needs_clarification`, com a voz
    recebendo "NOTHING was committed" como REGRA DURA. O banco tinha três linhas mudadas que
    nenhum juiz aprovou, e o usuário era informado do contrário.

    Fail-CLOSED: escrita não aprovada → humano."""
    from cogno_anima.types import ToolExecution

    wrote = ToolExecution(tool="confirm_appointment", arguments={"appointment_id": "a1"},
                          result="Appointment a1 is now CONFIRMED.", ok=True, side_effect=True)
    only_read = ToolExecution(tool="list_appointments", arguments={},
                              result="No PENDING appointments.", ok=True, side_effect=False)
    sup = FakeSuperego(approve=False, critique="confirmed rows that were not pending")
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"),
                     ego=FakeEgo(tool_calls=[[wrote], [only_read]]), superego=sup)
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=2),
                              dispatcher=dispatcher)

    assert ctx.stop_reason == "human_handoff"
    assert ctx.needs_handoff is True
    # …e a voz NÃO roda: não há continuação verdadeira a escrever sobre uma escrita que
    # ninguém aprovou (é o `handoff_message` do host que sai).
    assert ctx.superego_result is None
    assert mk.VOICE_CORRECTION not in ctx.metadata, \
        "a voz não pode receber 'nada foi executado' quando ALGO foi"


async def test_a_host_seeded_garbage_ledger_does_not_kill_the_turn(
        stub_embedder, stub_backend, dispatcher):
    """`ctx.metadata` é do HOST. `setdefault(chave, []).append(...)` sobre um valor que não é
    lista levanta AttributeError DE DENTRO do laço de correção de produção, onde só
    `StopPipeline` é capturada — um turno inteiro perdido para escrever telemetria."""
    ctx0 = _ctx()
    ctx0.metadata[mk.JUDGE_ATTEMPTS] = 3          # o host semeou lixo
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=FakeEgo(),
                     superego=FakeSuperego(approve=True))
    ctx = await pipe.run_turn(ctx0, _cfg(stub_backend, max_corrections=1),
                              dispatcher=dispatcher)
    assert isinstance(ctx.metadata[mk.JUDGE_ATTEMPTS], list)
    assert ctx.superego_result is not None        # o turno chegou ao fim


async def test_a_gate_B_hold_does_not_swallow_the_writes_that_preceded_it(
        stub_embedder, stub_backend, dispatcher):
    """O `break` do gate-B pulava o juiz E o registro. Uma tentativa que EXECUTOU escritas e
    só então segurou uma chamada destrutiva não deixava rastro nenhum — buraco no ledger
    exatamente onde havia escrita. As execuções entram no turno ANTES de qualquer saída."""
    from cogno_anima.types import EgoResult, EgoStep, StageMetrics, ToolExecution

    wrote = ToolExecution(tool="cancel_appointment", arguments={"appointment_id": "a1"},
                          result="canceled", ok=True, side_effect=True)
    held = ToolExecution(tool="delete_patient", arguments={"id": "p9"})

    class _HoldingEgo:
        async def process(self, ctx, backend, dispatcher, *, system_prompt):
            ctx.ego_result = EgoResult(
                steps=[EgoStep(index=0, path="native", assistant_text="",
                               tool_calls=[wrote])],
                pending_confirmation=[held],
                metrics=StageMetrics(stage="ego", elapsed_ms=0.0, tokens_in=0, tokens_out=0,
                                     model="fake"))
            return ctx

    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=_HoldingEgo(),
                     superego=FakeSuperego(approve=True))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=2),
                              dispatcher=dispatcher)
    assert [t.tool for t in ctx.turn_executions] == ["cancel_appointment"]
    assert committed_this_turn(ctx) is True


async def test_a_turn_that_only_READ_still_keeps_the_conversation_alive(
        stub_embedder, stub_backend, dispatcher):
    """Braço-controle do teste acima — sem ele, "sempre handoff" passaria nos dois.

    Nada foi commitado em tentativa NENHUMA: encerrar numa pessoa seria um beco sem saída onde
    a continuação verdadeira ("achei seu horário — mudo para as 11h?") resolve."""
    from cogno_anima.types import ToolExecution

    read = ToolExecution(tool="list_appointments", arguments={}, result="3 rows", ok=True,
                         side_effect=False)
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"),
                     ego=FakeEgo(tool_calls=[[read], [read]]),
                     superego=FakeSuperego(approve=False, critique="did not answer"))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=2),
                              dispatcher=dispatcher)

    assert ctx.stop_reason == "needs_clarification"
    assert not getattr(ctx, "needs_handoff", False)
    assert ctx.metadata[mk.VOICE_CORRECTION]["kind"] == "not_executed"
    assert ctx.superego_result is not None      # a voz escreve a continuação


async def test_a_FAILED_write_from_a_rejected_attempt_is_not_a_commit(
        stub_embedder, stub_backend, dispatcher):
    """`side_effect` viaja mesmo quando a chamada falhou (é por NOME da tool). Uma mutação que
    FALHOU não mudou nada — encerrar num humano por causa dela é o falso-positivo simétrico."""
    from cogno_anima.types import ToolExecution

    failed = ToolExecution(tool="book_appointment", arguments={"when": "9h"}, result="",
                           error="slot taken", ok=False, side_effect=True)
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"),
                     ego=FakeEgo(tool_calls=[[failed], [failed]]),
                     superego=FakeSuperego(approve=False, critique="not booked"))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=2),
                              dispatcher=dispatcher)

    assert ctx.stop_reason == "needs_clarification"
    assert ctx.superego_result is not None


async def test_the_committed_bit_does_not_ride_the_DISPLAY_path(
        stub_embedder, stub_backend, dispatcher):
    """O bit que roteia o turno é calculado sobre a lista INTEIRA e fora do try.

    Se ele viesse da lista exibida, duas coisas o apagariam sem ninguém notar: o teto por
    tentativa (a escrita cai fora do corte) e um `arguments` que não serializa (a entrada
    degrada para `tools: []`). Nos dois casos o turno deixaria de escalar — falhar para o lado
    perigoso por causa do caminho de EXIBIÇÃO."""
    from cogno_anima.types import ToolExecution

    from cogno_soma.pipeline import _TOOLS_PER_ATTEMPT, _attempt_tools

    class Cursed:
        def __str__(self):
            raise RuntimeError("str() explode")

    reads = [ToolExecution(tool=f"read{i}", arguments={}, result="ok", ok=True,
                           side_effect=False) for i in range(_TOOLS_PER_ATTEMPT)]
    write = ToolExecution(tool="confirm_appointment", arguments={"appointment_id": "a1"},
                          result="now CONFIRMED", ok=True, side_effect=True)
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"),
                     ego=FakeEgo(tool_calls=[[*reads, write], [reads[0]]]),
                     superego=FakeSuperego(approve=False, critique="x"))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=2),
                              dispatcher=dispatcher)
    entry = ctx.metadata[mk.JUDGE_ATTEMPTS][0]
    assert entry["tools_dropped"] == 1                      # a escrita ficou fora da exibição
    assert not any(t["side_effect"] for t in entry["tools"])
    assert entry["committed"] is True                       # …e mesmo assim o turno escala
    assert ctx.stop_reason == "human_handoff"

    # …e a entrada DEGRADADA também mantém o bit
    cursed = ToolExecution(tool="confirm_appointment", arguments={"when": Cursed()},
                           result="now CONFIRMED", ok=True, side_effect=True)

    class _Ego:
        tools_executed = [cursed]

    degraded = _attempt_tools(_Ego())
    assert degraded["tools"] == [] and degraded["tools_error"] == "RuntimeError"
    assert degraded["committed"] is True


async def test_the_tools_list_is_capped_and_the_overflow_counted(
        stub_embedder, stub_backend, dispatcher):
    """Os cortes por campo limitavam cada entrada; nada limitava a LISTA — e o teto real não é
    o max_steps default: plano premium injeta ego_max_steps=25, chamadas por passo são
    ilimitadas, e bloqueadas/duplicadas também são registradas (pior caso medido no review:
    ~375 entradas num turno). O excedente é CONTADO, nunca silencioso."""
    from cogno_anima.types import ToolExecution

    from cogno_soma.pipeline import _TOOLS_PER_ATTEMPT

    many = [ToolExecution(tool=f"t{i}", arguments={}, result="ok", ok=True, side_effect=False)
            for i in range(_TOOLS_PER_ATTEMPT + 5)]
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"),
                     ego=FakeEgo(tool_calls=[many]),
                     superego=FakeSuperego(approve=False))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=1),
                              dispatcher=dispatcher)
    rec = ctx.metadata["judge_attempts"][0]
    assert len(rec["tools"]) == _TOOLS_PER_ATTEMPT
    assert rec["tools_dropped"] == 5


async def test_a_non_json_argument_does_not_kill_the_turn_to_record_telemetry(
        stub_embedder, stub_backend, dispatcher):
    """The record runs inside the PRODUCTION correction loop. The args dict usually comes from
    model JSON, but a host-injected value (RBAC pinning, a resolved date) is not guaranteed
    serializable — and telemetry must never be the reason a turn dies. Found by probing the
    diff itself: `json.dumps({"when": date(...)})` raises TypeError."""
    import datetime

    from cogno_anima.types import ToolExecution

    odd = ToolExecution(tool="book_appointment",
                        arguments={"when": datetime.date(2026, 8, 20)},
                        result="ok", ok=True, side_effect=True)
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"),
                     ego=FakeEgo(tool_calls=[odd]),
                     superego=FakeSuperego(approve=False))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=1),
                              dispatcher=dispatcher)
    rec = ctx.metadata["judge_attempts"][0]["tools"][0]
    assert "2026-08-20" in rec["args"]           # gravado como string, turno vivo


async def test_a_long_critique_is_truncated_before_it_rides_on_the_context(
        stub_embedder, stub_backend, dispatcher):
    """This metadata is persisted by the host. A judge critique is model prose with no length
    contract, and one per attempt, so unbounded it grows a stored record nobody bounded."""
    from cogno_soma.pipeline import _CRITIQUE_CHARS

    sup = FakeSuperego(approve=False, critique="x" * (_CRITIQUE_CHARS * 3))
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=FakeEgo(), superego=sup)
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=1),
                              dispatcher=dispatcher)
    assert len(ctx.metadata["judge_attempts"][0]["critique"]) == _CRITIQUE_CHARS


# ── per-call identity: seq / attempt / prompt_sha ────────────────────────────────────────
#
# `stage_metrics` is "the populated canonical slots, then retry_metrics", and each canonical
# slot holds the value that SURVIVED — so it is not call order, and reading it as one blames
# the wrong call on a retried turn (measured on a real run: two `ego` entries at 3874 and 3747
# prompt tokens, and the write belonged to the SECOND-listed one). The orchestrator is the
# only layer that sequences the stages, so it is the only one that can say the true order.

def _by_seq(ctx):
    return [(m.seq, m.stage, m.attempt) for m in sorted(ctx.stage_metrics, key=lambda m: m.seq)]


async def test_seq_recovers_the_true_call_order_that_stage_metrics_loses(
        stub_embedder, stub_backend):
    """The judge rejects once, so the EGO runs twice — and the two `ego` entries land in
    OPPOSITE halves of `stage_metrics`. Sorting by `seq` puts them back in the order they ran.

    Mutation: stop stamping (or stamp a constant) and the ordering assertion dies.
    """
    pipe = Pipeline(
        embedder=stub_embedder,
        noumeno=FakeNoumeno(), ner=FakeNER(), id_stage=FakeID(route="EGO"),
        ego=FakeEgo(), superego=FakeSuperego(approve_after=2))
    cfg = TurnConfig(gen_backend=stub_backend, ego_backend=stub_backend, ego_prompt="x",
                     limits_prompt="limits", voice_prompt="voice", max_corrections=3)
    ctx = await pipe.run_turn(_ctx("registra 150"), cfg,
                              dispatcher=RecordingDispatcher())

    ordered = _by_seq(ctx)
    seqs = [s for s, _, _ in ordered]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), "um seq por chamada, sem furos"
    stages = [st for _, st, _ in ordered]
    assert stages.index("noumeno") < stages.index("ner") < stages.index("id")
    # the two EGO runs, in the order they happened, each carrying its attempt
    egos = [(s, a) for s, st, a in ordered if st == "ego"]
    assert [a for _, a in egos] == [1, 2], "a 1ª tentativa roda ANTES da 2ª"
    judges = [a for _, st, a in ordered if st == "superego_judge"]
    assert judges == [1, 2]
    # …and the raw list really does disagree, which is why seq had to exist
    raw = [m.stage for m in ctx.stage_metrics]
    assert raw != stages, "se a lista crua já estivesse em ordem, o seq seria supérfluo"


async def test_the_label_comes_from_the_HOST_not_from_hashing_the_rendered_prompt(
        stub_embedder, stub_backend):
    """`prompt_sha` labels a configuration, and the host is the only layer that can name one.

    The first cut digested `cfg.ego_prompt` here and called the result PII-free by
    construction. It was PII BY construction: the host renders `{identity_label}` and
    `{identity_email}` into the system/scope/limits/voice slots before handing them over
    (`cogno_host.persona.render_slot`). Hashing that gives a different sha PER CONTACT — one
    row per conversation in a content-addressed store — and the text behind it carries a name
    and an e-mail. Refuted by a peer before the store existed, which is why it cost a revert.

    So the host digests its TEMPLATES and passes the map down; this layer only says which slot
    each call used. Mutation: hash `cfg.*_prompt` here again and the per-contact assertion
    below dies.
    """
    async def run(context, ego_prompt):
        pipe = Pipeline(embedder=stub_embedder, noumeno=FakeNoumeno(), ner=FakeNER(),
                        id_stage=FakeID(route="EGO"), ego=FakeEgo(), superego=FakeSuperego())
        cfg = TurnConfig(gen_backend=stub_backend, ego_backend=stub_backend,
                         ego_prompt=ego_prompt, voice_prompt="voice")
        c = _ctx("registra 150")
        c.metadata[mk.PROMPT_SHAS] = {"ego": "9f3c1a", "voice": "b18d04"}
        c.metadata[mk.EGO_CONTEXT] = context
        return await pipe.run_turn(c, cfg, dispatcher=RecordingDispatcher())

    # SAME template, rendered for two different contacts — the rendered text differs, the
    # label must not, or the store degenerates into one row per conversation.
    a = await run("[MEMORIES] x", "Você atende Marina (marina@exemplo.com).")
    b = await run("[MEMORIES] y", "Você atende João (joao@exemplo.com).")
    sha_a = next(m.prompt_sha for m in a.stage_metrics if m.stage == "ego")
    sha_b = next(m.prompt_sha for m in b.stage_metrics if m.stage == "ego")
    assert sha_a == sha_b == "9f3c1a", "o rótulo é do TEMPLATE, não do texto renderizado"
    assert next(m.prompt_sha for m in a.stage_metrics if m.stage == "superego_voice") == "b18d04"


async def test_the_two_ego_attempts_SHARE_a_configuration_and_differ_by_attempt(
        stub_embedder, stub_backend):
    """The two attempts of a retried turn ran the same configuration; `attempt` is the axis
    that carries the difference between them."""
    pipe = Pipeline(
        embedder=stub_embedder,
        noumeno=FakeNoumeno(), ner=FakeNER(), id_stage=FakeID(route="EGO"),
        ego=FakeEgo(), superego=FakeSuperego(approve_after=2))
    cfg = TurnConfig(gen_backend=stub_backend, ego_backend=stub_backend,
                     ego_prompt="x", limits_prompt="limits", voice_prompt="voice",
                     max_corrections=3)
    ctx = _ctx("registra 150")
    ctx.metadata[mk.PROMPT_SHAS] = {"ego": "9f3c1a", "judge": "77e0f1"}
    ctx = await pipe.run_turn(ctx, cfg, dispatcher=RecordingDispatcher())

    egos = [(m.attempt, m.prompt_sha) for m in ctx.stage_metrics if m.stage == "ego"]
    assert {a for a, _ in egos} == {1, 2}
    assert egos[0][1] == egos[1][1] == "9f3c1a"


async def test_the_judge_label_carries_the_criteria_flag(stub_embedder, stub_backend):
    """`JUDGE_CONVERSATIONAL` swaps the entire criteria set, so the same limits template under
    different criteria is a different configuration and must not share a label. The host cannot
    know the flag at template time, so it is appended here."""
    async def run(conversational):
        pipe = Pipeline(embedder=stub_embedder, noumeno=FakeNoumeno(), ner=FakeNER(),
                        id_stage=FakeID(route="EGO"), ego=FakeEgo(), superego=FakeSuperego())
        cfg = TurnConfig(gen_backend=stub_backend, ego_backend=stub_backend, ego_prompt="x",
                         limits_prompt="limits", voice_prompt="voice")
        c = _ctx("oi")
        c.metadata[mk.PROMPT_SHAS] = {"judge": "77e0f1"}
        if conversational:
            c.metadata[mk.JUDGE_CONVERSATIONAL] = True
        return await pipe.run_turn(c, cfg, dispatcher=RecordingDispatcher())

    plain = await run(False)
    conv = await run(True)
    assert next(m.prompt_sha for m in plain.stage_metrics if m.stage == "superego_judge") == "77e0f1"
    assert next(m.prompt_sha for m in conv.stage_metrics if m.stage == "superego_judge") == "77e0f1+conv"


async def test_no_host_map_means_UNLABELLED_not_broken(stub_embedder, stub_backend):
    """A host that supplies nothing gets `seq`/`attempt` and an empty sha — "unlabelled",
    never "no prompt", and never an exception. A malformed map degrades the same way."""
    for supplied in (None, "not a dict", {"ego": 42}):
        pipe = Pipeline(embedder=stub_embedder, noumeno=FakeNoumeno(), ner=FakeNER(),
                        id_stage=FakeID(route="EGO"), ego=FakeEgo(), superego=FakeSuperego())
        cfg = TurnConfig(gen_backend=stub_backend, ego_backend=stub_backend, ego_prompt="x",
                         voice_prompt="voice")
        c = _ctx("oi")
        if supplied is not None:
            c.metadata[mk.PROMPT_SHAS] = supplied
        ctx = await pipe.run_turn(c, cfg, dispatcher=RecordingDispatcher())
        ego = next(m for m in ctx.stage_metrics if m.stage == "ego")
        assert ego.prompt_sha == "" and ego.seq > 0, f"supplied={supplied!r}"


async def test_a_stage_with_no_host_supplied_prompt_has_an_EMPTY_sha(
        stub_embedder, stub_backend):
    """NOUMENO/NER render their own templates and the ID calls no model, so nothing a
    deployment sets reaches them this turn. An empty sha says "nothing a deployment set" —
    which is a different claim from "not recorded", and `seq` proves it was recorded."""
    pipe = Pipeline(embedder=stub_embedder, noumeno=FakeNoumeno(), ner=FakeNER(),
                    id_stage=FakeID(route="SUPEREGO"), ego=FakeEgo(), superego=FakeSuperego())
    cfg = TurnConfig(gen_backend=stub_backend, ego_backend=stub_backend, ego_prompt="x",
                     voice_prompt="voice")
    ctx = _ctx("oi")
    # The host labels the slots it authors — and there is no slot for NOUMENO/NER, because
    # there is no deployment-authored text for them to run.
    ctx.metadata[mk.PROMPT_SHAS] = {"ego": "9f3c1a", "voice": "b18d04"}
    ctx = await pipe.run_turn(ctx, cfg, dispatcher=RecordingDispatcher())

    by_stage = {m.stage: m for m in ctx.stage_metrics}
    assert by_stage["noumeno"].prompt_sha == "" and by_stage["noumeno"].seq > 0
    assert by_stage["ner"].prompt_sha == ""
    assert by_stage["superego_voice"].prompt_sha == "b18d04", "a voz roda texto da persona"


async def test_the_stamp_counter_does_not_leak_into_persisted_metadata(
        stub_embedder, stub_backend):
    """It is bookkeeping, not a contract, and `ctx.metadata` is what the host persists."""
    pipe = Pipeline(embedder=stub_embedder, noumeno=FakeNoumeno(), ner=FakeNER(),
                    id_stage=FakeID(route="SUPEREGO"), ego=FakeEgo(), superego=FakeSuperego())
    ctx = await pipe.run_turn(_ctx("oi"),
                              TurnConfig(gen_backend=stub_backend, ego_backend=stub_backend,
                                         ego_prompt="x", voice_prompt="v"),
                              dispatcher=RecordingDispatcher())
    assert "_call_seq" not in ctx.metadata
