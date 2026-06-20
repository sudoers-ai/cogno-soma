"""Tests for the interception seam: Hooks fire points + StopPipeline + atomicity."""


from cogno_soma import Hooks, Pipeline, StopPipeline, TurnConfig
from cogno_anima.types import PipelineContext

from tests.conftest import FakeEgo, FakeID, FakeNER, FakeNoumeno, FakeSuperego


def _ctx(text="hi"):
    return PipelineContext(user_input=text)


def _pipe(embedder, calls, *, route="EGO", **over):
    return Pipeline(
        embedder=embedder,
        noumeno=over.get("noumeno", FakeNoumeno(calls=calls)),
        ner=FakeNER(calls=calls),
        id_stage=FakeID(route=route, calls=calls),
        ego=over.get("ego", FakeEgo(calls=calls)),
        superego=over.get("superego", FakeSuperego(calls=calls)),
    )


def _cfg(backend, hooks, **kw):
    return TurnConfig(gen_backend=backend, ego_backend=backend, ego_prompt="x", hooks=hooks, **kw)


async def test_hooks_fire_in_order(stub_embedder, stub_backend, dispatcher):
    seen: list[str] = []
    hooks = Hooks(
        before_turn=lambda c: seen.append("before_turn"),
        after_noumeno=lambda c: seen.append("after_noumeno"),
        after_ner=lambda c: seen.append("after_ner"),
        after_id=lambda c: seen.append("after_id"),
        after_ego=lambda c: seen.append("after_ego"),
        after_superego=lambda c: seen.append("after_superego"),
        after_turn=lambda c: seen.append("after_turn"),
        on_commit=lambda c: seen.append("on_commit"),
    )
    pipe = _pipe(stub_embedder, [], route="EGO")
    await pipe.run_turn(_ctx(), _cfg(stub_backend, hooks), dispatcher=dispatcher)
    assert seen == [
        "before_turn", "after_noumeno", "after_ner", "after_id",
        "after_ego", "on_commit", "after_superego", "after_turn",
    ]


async def test_async_hook_is_awaited(stub_embedder, stub_backend, dispatcher):
    seen: list[str] = []

    async def amem(ctx):
        seen.append("async_ran")
        ctx.metadata["mem"] = "injected"

    hooks = Hooks(after_ner=amem)
    pipe = _pipe(stub_embedder, [], route="SUPEREGO")
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, hooks), dispatcher=dispatcher)
    assert seen == ["async_ran"]
    assert ctx.metadata["mem"] == "injected"


async def test_hook_can_mutate_context_for_downstream(stub_embedder, stub_backend, dispatcher):
    """after_ner injects memory that the (fake) downstream can read from metadata."""
    def inject(ctx):
        ctx.metadata["ego_context"] = "[MEMORIES] balance is 100"

    pipe = _pipe(stub_embedder, [], route="EGO")
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, Hooks(after_ner=inject)), dispatcher=dispatcher)
    assert ctx.metadata["ego_context"].startswith("[MEMORIES]")


async def test_stop_pipeline_halts_with_response(stub_embedder, stub_backend, dispatcher):
    """A safety hook raises StopPipeline → terminal response, no EGO, no voice."""
    def crisis(ctx):
        raise StopPipeline(reason="human_handoff", response="Please call 988.", blocked=True)

    ego = FakeEgo()
    pipe = _pipe(stub_embedder, [], route="EGO", ego=ego)
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, Hooks(after_ner=crisis)), dispatcher=dispatcher)
    assert ctx.stop_reason == "human_handoff"
    assert ctx.superego_result.response == "Please call 988."
    assert ctx.superego_result.blocked is True
    assert ego.invocations == 0


async def test_stop_pipeline_without_response(stub_embedder, stub_backend, dispatcher):
    def stop(ctx):
        raise StopPipeline(reason="semantic_cache")

    pipe = _pipe(stub_embedder, [], route="SUPEREGO")
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, Hooks(after_id=stop)), dispatcher=dispatcher)
    assert ctx.stop_reason == "semantic_cache"
    assert ctx.superego_result is None


async def test_on_rollback_fires_per_retry(stub_embedder, stub_backend, dispatcher):
    rolls: list[int] = []
    hooks = Hooks(on_rollback=lambda c: rolls.append(1), on_commit=lambda c: rolls.append(99))
    sup = FakeSuperego(approve_after=3)  # rejects twice, approves on 3rd
    pipe = _pipe(stub_embedder, [], route="EGO", superego=sup)
    await pipe.run_turn(_ctx(), _cfg(stub_backend, hooks, max_corrections=3), dispatcher=dispatcher)
    assert rolls == [1, 1, 99]  # two rollbacks (before each retry), then a commit


async def test_no_commit_on_handoff(stub_embedder, stub_backend, dispatcher):
    committed: list[int] = []
    hooks = Hooks(on_commit=lambda c: committed.append(1))
    pipe = _pipe(stub_embedder, [], route="EGO", superego=FakeSuperego(approve=False))
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, hooks, max_corrections=2), dispatcher=dispatcher)
    assert ctx.needs_handoff is True
    assert committed == []  # never commit an unapproved execution


async def test_after_turn_fires_on_blocked_path(stub_embedder, stub_backend, dispatcher):
    seen: list[str] = []
    hooks = Hooks(after_turn=lambda c: seen.append("after_turn"))
    # PII-CRITICAL block path still reaches the after_turn bookend
    pipe = Pipeline(embedder=stub_embedder, noumeno=FakeNoumeno(), ner=FakeNER(),
                    id_stage=FakeID(route="SUPEREGO", blocked=True), ego=FakeEgo(),
                    superego=FakeSuperego())
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, hooks), dispatcher=dispatcher)
    assert ctx.stop_reason == "pii_blocked"
    assert seen == ["after_turn"]
