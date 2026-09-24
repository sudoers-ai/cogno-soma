"""A proposal turn whose held call SENDS TEXT TO A PERSON is judged — every other one is not.

The judge is skipped on a gate-B hold, deliberately (`test_gate_b_hold_skips_judge_and_retry`):
the action is intentionally incomplete and the judge would reject the hold itself. For a held
MESSAGE the skip was the defect: the text is final at the hold — the confirmed replay sends
those exact bytes — so its only review ran after delivery. Measured on a downstream host: 3 of
4 messages delivered to staff were wrong, and the critique that named the defect ran one turn
too late. The host declares which held calls deliver text (`mk.HELD_DELIVERED_TEXT`); names and
texts here are invented.
"""

import cogno_anima.metakeys as mk
from cogno_anima.types import EgoResult, EgoStep, PipelineContext, ToolExecution

from cogno_soma import Pipeline, STOP_JUDGE_EXHAUSTED, TurnConfig
from cogno_soma.hooks import Hooks

from tests.conftest import FakeEgo, FakeID, FakeNER, FakeNoumeno, FakeSuperego, metrics

ANNOUNCES = "Olá Joana, estou a contactar para partilhar o resumo da sua aula de hoje."


class _HeldMessageEgo(FakeEgo):
    """Holds a message-sending call on every attempt (a rewrite is a NEW held message)."""

    async def process(self, ctx, backend, dispatcher, *, system_prompt):
        ctx = await super().process(ctx, backend, dispatcher, system_prompt=system_prompt)
        held = ToolExecution(tool="notify_user", ok=False, error="needs_confirmation",
                             arguments={"target": "Joana",
                                        "message": f"{ANNOUNCES} (tentativa {self.invocations})"},
                             result="", tool_mutating=True)
        ctx.ego_result = EgoResult(steps=[EgoStep(index=0, path="native", tool_calls=[held])],
                                   pending_confirmation=[held], metrics=metrics("ego"))
        return ctx


def _run(sup, *, declared=True, max_corrections=2, after_ego=None):
    ego = _HeldMessageEgo()
    pipe = Pipeline(embedder=None, noumeno=FakeNoumeno(), ner=FakeNER(),
                    id_stage=FakeID(route="EGO"), ego=ego, superego=sup)
    ctx = PipelineContext(user_input="envia à Joana o resumo da aula de hoje")
    if declared:
        ctx.metadata[mk.HELD_DELIVERED_TEXT] = {"notify_user": "message"}
    return ego, pipe, ctx, TurnConfig(gen_backend=None, ego_backend=None, ego_prompt="exec",
                                      max_corrections=max_corrections,
                                      hooks=Hooks(after_ego=after_ego))


async def test_a_held_message_is_judged_before_it_can_be_proposed(stub_backend, dispatcher):
    sup = FakeSuperego(approve=True)
    ego, pipe, ctx, cfg = _run(sup)
    ctx = await pipe.run_turn(ctx, cfg, dispatcher=dispatcher)
    assert sup._evals == 1, "the held message reached the host without the judge reading it"
    assert ego.invocations == 1
    assert ctx.metadata[mk.JUDGE_VERDICT] == {"approved": True, "attempts": 1}
    assert [a["attempt"] for a in ctx.metadata[mk.JUDGE_ATTEMPTS]] == [1]


async def test_a_rejected_held_message_is_rewritten_once_then_left_unapproved(stub_backend,
                                                                           dispatcher):
    """Rejected → the EGO rewrites within the SAME budget; still rejected → the loop ends
    unapproved, and the host's gate (after_ego) sees a verdict it must not propose over."""
    seen = []
    sup = FakeSuperego(approve=False, critique="the message does not carry the summary")
    ego, pipe, ctx, cfg = _run(sup, after_ego=lambda c: seen.append(
        dict(c.metadata[mk.JUDGE_VERDICT])))
    ctx = await pipe.run_turn(ctx, cfg, dispatcher=dispatcher)
    assert ego.invocations == 2 and sup._evals == 2
    assert seen == [{"approved": False, "attempts": 2}]
    assert ctx.metadata[mk.EGO_CORRECTION]["reason"] == "the message does not carry the summary"
    # no host gate here: the orchestrator's own exhaustion path takes it — nothing proposed
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED


async def test_a_rewrite_the_judge_approves_goes_ahead(stub_backend, dispatcher):
    sup = FakeSuperego(approve_after=2)
    ego, pipe, ctx, cfg = _run(sup, max_corrections=3)
    ctx = await pipe.run_turn(ctx, cfg, dispatcher=dispatcher)
    assert ego.invocations == 2 and sup._evals == 2
    assert ctx.metadata[mk.JUDGE_VERDICT] == {"approved": True, "attempts": 2}


async def test_the_control_an_undeclared_hold_is_still_not_judged(stub_backend, dispatcher):
    """Same held call, no declaration → today's skip, byte for byte (and the first test above
    proves this ego DOES reach the judge when declared)."""
    sup = FakeSuperego(approve=False)
    ego, pipe, ctx, cfg = _run(sup, declared=False)
    ctx = await pipe.run_turn(ctx, cfg, dispatcher=dispatcher)
    assert sup._evals == 0 and ego.invocations == 1
    assert ctx.stop_reason != STOP_JUDGE_EXHAUSTED
    assert mk.JUDGE_ATTEMPTS not in ctx.metadata
