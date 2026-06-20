"""Control-flow tests for ``Pipeline.run_turn`` using fake stages."""


from cogno_soma import Pipeline, TurnConfig

from tests.conftest import (
    FakeEgo,
    FakeID,
    FakeNER,
    FakeNoumeno,
    FakeSuperego,
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


async def test_non_task_path_voices_response(stub_embedder, stub_backend, dispatcher):
    """SUPEREGO route (no EGO): goes straight to voice, never runs the EGO."""
    ego = FakeEgo()
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="SUPEREGO"), ego=ego)
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend), dispatcher=dispatcher)
    assert ctx.superego_result.response == "final reply"
    assert ctx.stop_reason == "completed"
    assert ego.invocations == 0


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
    """Judge rejects every attempt → loop runs max_corrections times → handoff."""
    ego = FakeEgo()
    sup = FakeSuperego(approve=False)
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), ego=ego, superego=sup)
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=3), dispatcher=dispatcher)
    assert ego.invocations == 3
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
    pipe = _pipeline(stub_embedder, id_stage=FakeID(route="EGO"), superego=FakeSuperego(approve=False))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, max_corrections=2), dispatcher=dispatcher)
    assert ctx.needs_handoff is True
    assert ctx.total_llm_tokens > 0            # the EGO + judge attempts are billed
