"""``Hooks.after_ego_attempt`` — the seam between the EXECUTOR and the JUDGE, once per attempt.

Why it exists. ``after_ego`` fires AFTER the EGO⇄SUPEREGO loop, so a host that has to correct
what the executor produced BEFORE the judge reads it had no hook: the reference host aligns a
held message's recipient to the name the contact wrote (a deterministic rule, its own), and with
the alignment in ``after_ego`` the judge read the executor's rendering of the name, rejected it,
and the turn exhausted. The host now WRAPS the EGO stage it hands the pipeline to run the
alignment after every executor pass — a wrapper doing a hook's job.

The contract pinned here:

* it fires after EVERY executor pass that returned — the first and each correction — and
  BEFORE that attempt is judged;
* what it changes on ``ctx.ego_result`` is what the judge reads;
* it fires also on a pass whose proposal the judge does not read (a held call that sends no
  text), because a held call is exactly what a host aligns;
* it may raise ``StopPipeline`` like any hook: the turn ends with the host's answer, the judge
  never reads that attempt, and what the pass executed is already on the turn;
* absent, nothing changes; ``after_ego`` still fires once, after the loop.
"""

from __future__ import annotations

from cogno_anima.types import PipelineContext, ToolExecution

from cogno_soma import Hooks, Pipeline, StopPipeline, TurnConfig
from tests.conftest import FakeEgo, FakeID, FakeNER, FakeNoumeno, FakeSuperego


def _ctx(text="hi"):
    return PipelineContext(user_input=text)


def _pipe(embedder, calls, *, ego=None, superego=None):
    return Pipeline(embedder=embedder, noumeno=FakeNoumeno(calls=calls),
                    ner=FakeNER(calls=calls), id_stage=FakeID(route="EGO", calls=calls),
                    ego=ego or FakeEgo(calls=calls), superego=superego or FakeSuperego(calls=calls))


def _cfg(backend, hooks, **kw):
    return TurnConfig(gen_backend=backend, ego_backend=backend, ego_prompt="x", hooks=hooks, **kw)


class _ReadingJudge(FakeSuperego):
    """Records the draft each evaluate call READ."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.read: "list[str]" = []

    async def evaluate(self, ctx, backend, *, limits_prompt):
        self.read.append(ctx.ego_result.draft if ctx.ego_result else "")
        return await super().evaluate(ctx, backend, limits_prompt=limits_prompt)


async def test_TWIN_it_fires_after_every_executor_pass_and_before_its_judgement(
        stub_embedder, stub_backend, dispatcher):
    """Red before: ``Hooks`` had no such field."""
    calls: "list[str]" = []
    hooks = Hooks(after_ego_attempt=lambda c: calls.append("after_ego_attempt"),
                  after_ego=lambda c: calls.append("after_ego"))
    pipe = _pipe(stub_embedder, calls, superego=FakeSuperego(calls=calls, approve_after=2))
    await pipe.run_turn(_ctx(), _cfg(stub_backend, hooks, max_corrections=2), dispatcher=dispatcher)
    loop = [c for c in calls if c in ("ego", "after_ego_attempt", "judge", "after_ego")]
    assert loop == ["ego", "after_ego_attempt", "judge",
                    "ego", "after_ego_attempt", "judge", "after_ego"]


async def test_TWIN_what_it_changes_is_what_the_judge_reads(stub_embedder, stub_backend,
                                                           dispatcher):
    def align(ctx):
        steps = ctx.ego_result.steps
        steps[-1] = steps[-1].model_copy(update={"assistant_text": "aligned by the host"})

    judge = _ReadingJudge(approve=True)
    pipe = _pipe(stub_embedder, [], ego=FakeEgo(drafts=["the executor's draft"]), superego=judge)
    await pipe.run_turn(_ctx(), _cfg(stub_backend, Hooks(after_ego_attempt=align)),
                        dispatcher=dispatcher)
    assert judge.read == ["aligned by the host"]


async def test_TWIN_it_fires_on_a_held_proposal_the_judge_does_not_read(
        stub_embedder, stub_backend, dispatcher):
    """A held call that sends no text is a PROPOSE turn: the judge is skipped — and that held
    call is exactly what a host aligns, so the seam fires before the skip."""
    class _HoldingEgo(FakeEgo):
        async def process(self, ctx, backend, dispatcher, *, system_prompt):
            ctx = await super().process(ctx, backend, dispatcher, system_prompt=system_prompt)
            ctx.ego_result.pending_confirmation = [ToolExecution(tool="cancel_appointment")]
            return ctx

    calls: "list[str]" = []
    hooks = Hooks(after_ego_attempt=lambda c: calls.append(
        f"seen:{c.ego_result.pending_confirmation[0].tool}"))
    pipe = _pipe(stub_embedder, calls, ego=_HoldingEgo(calls=calls))
    await pipe.run_turn(_ctx(), _cfg(stub_backend, hooks), dispatcher=dispatcher)
    assert "seen:cancel_appointment" in calls and "judge" not in calls


async def test_TWIN_a_stop_from_it_ends_the_turn_unjudged_and_the_pass_is_recorded(
        stub_embedder, stub_backend, dispatcher):
    def ask(ctx):
        raise StopPipeline(reason="needs_clarification", response="Qual das duas pessoas?")

    calls: "list[str]" = []
    ego = FakeEgo(calls=calls, tool_calls=[[ToolExecution(tool="lookup_people", ok=True)]])
    pipe = _pipe(stub_embedder, calls, ego=ego)
    ctx = await pipe.run_turn(_ctx(), _cfg(stub_backend, Hooks(after_ego_attempt=ask)),
                              dispatcher=dispatcher)
    assert ctx.stop_reason == "needs_clarification"
    assert ctx.superego_result.response == "Qual das duas pessoas?"
    assert "judge" not in calls
    assert [t.tool for t in ctx.turn_executions] == ["lookup_people"]


async def test_an_async_hook_is_awaited(stub_embedder, stub_backend, dispatcher):
    seen: "list[int]" = []

    async def count(ctx):
        seen.append(1)

    pipe = _pipe(stub_embedder, [])
    await pipe.run_turn(_ctx(), _cfg(stub_backend, Hooks(after_ego_attempt=count)),
                        dispatcher=dispatcher)
    assert seen == [1]


async def test_CONTROL_without_it_the_loop_is_the_one_of_before(stub_embedder, stub_backend,
                                                                dispatcher):
    calls: "list[str]" = []
    hooks = Hooks(after_ego=lambda c: calls.append("after_ego"))
    pipe = _pipe(stub_embedder, calls, superego=FakeSuperego(calls=calls, approve_after=2))
    await pipe.run_turn(_ctx(), _cfg(stub_backend, hooks, max_corrections=2), dispatcher=dispatcher)
    loop = [c for c in calls if c in ("ego", "judge", "after_ego")]
    assert loop == ["ego", "judge", "ego", "judge", "after_ego"]
