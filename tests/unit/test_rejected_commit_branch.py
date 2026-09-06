"""Where a JUDGE REJECTION goes, under the production correction budget of ONE attempt.

The host cut ``max_self_corrections`` to 1 on every plan (``cogno_host/plans.py``): over 586
real execution turns the retries were 46.2% of the branch's tokens, and of the 88 turns they
approved only 3 wrote anything. The cut rests on that COST; what the extra attempts CATCH is
unmeasured and is not claimed anywhere in this file.

That makes THIS fork hot. It used to be reached after two or three attempts; now almost every
rejection reaches it on the first, so which side a turn falls on stops being an edge case:

    nothing committed  →  voice the continuation, ``judge_exhausted``, NO handoff
    something committed →  hand off; the world changed and review would not sign it

``tests/unit/test_pipeline.py`` already exercises both sides at ``max_corrections=3``. What was
NOT pinned is the pair at the budget production actually runs, and the pair is the point: the
same rejection, the same judge, the same fake EGO — one wrote and one did not — and the ONLY
difference in the outcome must be that fact. A test that exercises one side proves nothing
about the fork; a mutation that collapses the fork has to fail here.

MUTATION (one per property):

  * drop ``not committed`` from the condition (blind voice)    → the handoff twin dies;
  * invert it to ``committed`` (blind handoff)                 → the voiced twin dies.

**What deliberately does NOT change, and it is the finding rather than an omission.** The
loosening on the table was "a rejection about the TEXT voices; only a rejection naming a WRONG
WRITE hands off". It is not built, for two reasons and neither of them is a claim about the
judge:

* **there is no population to validate a classifier against.** Over the 750 persisted turns
  carrying a judge verdict, rejected-AND-committed is **1**. A classifier validated against n=1
  produces false confidence, not a decision — on the path that calls a human;
* **the trace-side alternative degenerates.** Call arguments against ``noumeno.preserved_terms``
  / ``intent.constraints`` / ``intent.negation``: the trace says what was written, never whether
  writing it was right, and the comparison is vacuous on the turns it would have to decide. A
  vacuous guard green-lights everything.

How faithfully ``judge.critique`` describes the execution it judged is **unmeasured, in either
direction** — and cannot be measured from what is persisted today, because critiques are kept
PER ATTEMPT and the draft only for the LAST one, so the two are not in the same record. (A
reading that claimed otherwise was withdrawn by its author on 2026-09-06 for exactly that
reason; nothing replaces it here.) So ambiguity keeps handing off, the saving stays in the
branch where the budget was cut, and the twin for "commit + text-only rejection → voice" is
deliberately absent rather than guessed at.
"""

from cogno_anima.types import ToolExecution

from cogno_soma import STOP_JUDGE_EXHAUSTED

from tests.conftest import FakeEgo, FakeID, FakeNER, FakeNoumeno, FakeSuperego
from cogno_soma import Pipeline, TurnConfig


# The production budget, as `cogno_host.plans` now sets it on every tier. Written as the
# metadata the host injects rather than as `cfg.max_corrections`, because that IS the channel:
# `_run_ego_loop` reads `plan_limits.max_self_corrections` and the config is only its fallback.
PLAN_LIMITS_ONE_ATTEMPT = {"max_self_corrections": 1}

BOOKED = ToolExecution(tool="book_appointment", ok=True, side_effect=True)
LOOKED = ToolExecution(tool="list_appointments", ok=True, side_effect=False)


def _pipeline(embedder, ego):
    return Pipeline(embedder=embedder, noumeno=FakeNoumeno(), ner=FakeNER(),
                    id_stage=FakeID(route="EGO"), ego=ego,
                    superego=FakeSuperego(approve=False, critique="not what was asked"))


async def _run(embedder, backend, dispatcher, ego):
    from cogno_anima.types import PipelineContext
    ctx = PipelineContext(user_input="marca para amanhã às 18h30")
    ctx.metadata["plan_limits"] = dict(PLAN_LIMITS_ONE_ATTEMPT)
    return await _pipeline(embedder, ego).run_turn(
        ctx, TurnConfig(gen_backend=backend, ego_backend=backend, ego_prompt="exec",
                        max_corrections=3),          # the plan cap must beat this
        dispatcher=dispatcher)


async def test_a_rejected_read_only_turn_voices_on_one_attempt(stub_embedder, stub_backend,
                                                               dispatcher):
    """TWIN (c) — nothing committed: the contact keeps the conversation, not a handoff.

    Asserted together and not in pieces, because each half alone is satisfiable by the wrong
    implementation: `needs_handoff is False` is also true of a turn that raised before deciding,
    and a `response` is also produced on paths that never reached this fork."""
    ego = FakeEgo(tool_calls=[LOOKED])
    ctx = await _run(stub_embedder, stub_backend, dispatcher, ego)
    assert ego.invocations == 1                       # the plan cap (1) beat cfg.max_corrections (3)
    assert ctx.needs_handoff is False
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED
    assert ctx.superego_result.response == "final reply"
    # the critique reaches the voice — without it the voice narrates the goal as done
    assert ctx.metadata["voice_correction"]["reason"] == "not what was asked"
    # ran tools → the draft is grounded, not an unverified claim (the voice keeps it)
    assert ctx.metadata["voice_correction"]["kind"] == "not_executed"


async def test_a_rejected_commit_hands_off_on_one_attempt(stub_embedder, stub_backend,
                                                          dispatcher):
    """TWIN (b) — the same rejection over a turn that WROTE: hand off. Fail-closed; nothing
    voices an unverified commit as done, and no retry is left to make it right."""
    ego = FakeEgo(tool_calls=[LOOKED, BOOKED])
    ctx = await _run(stub_embedder, stub_backend, dispatcher, ego)
    assert ego.invocations == 1
    assert ctx.needs_handoff is True
    assert ctx.stop_reason == "human_handoff"
    assert ctx.superego_result is None                # no voice ran: nothing was said as done
    assert "voice_correction" not in ctx.metadata


async def test_the_write_is_the_only_difference_between_the_twins(stub_embedder, stub_backend,
                                                                  dispatcher):
    """The fork itself, as one assertion. Same judge, same critique, same budget, same EGO
    shape — the ONLY input that differs is whether a mutating call succeeded, so an
    implementation that decides on anything else (the critique's wording, the attempt number,
    whether tools ran at all) cannot satisfy both rows."""
    read_only = await _run(stub_embedder, stub_backend, dispatcher, FakeEgo(tool_calls=[LOOKED]))
    wrote = await _run(stub_embedder, stub_backend, dispatcher,
                       FakeEgo(tool_calls=[LOOKED, BOOKED]))
    assert (read_only.needs_handoff, wrote.needs_handoff) == (False, True)
    assert (read_only.stop_reason, wrote.stop_reason) == (STOP_JUDGE_EXHAUSTED, "human_handoff")


async def test_a_failed_write_is_not_a_write(stub_embedder, stub_backend, dispatcher):
    """The mutating call RAN and failed (the slot was taken between propose and commit). The
    contact's world did not change, so this belongs on the voiced side — pinned at the budget
    of 1 because that is where a turn now lands on its first and only attempt."""
    failed = ToolExecution(tool="book_appointment", ok=False, side_effect=True,
                           error="slot taken")
    ctx = await _run(stub_embedder, stub_backend, dispatcher, FakeEgo(tool_calls=[failed]))
    assert ctx.needs_handoff is False
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED
    assert ctx.superego_result.response == "final reply"


async def test_routing_between_our_own_personas_is_not_a_write(stub_embedder, stub_backend,
                                                               dispatcher):
    """A turn whose only side-effecting call moved the conversation between OUR personas did
    nothing for the CONTACT, so a rejection over it voices instead of handing off — the host
    declares which tools are routing (``mk.ROUTING_ONLY_TOOLS``) and ``wrote_for_the_contact``
    filters by it. This is why the measured population of the handoff twin is 1 turn in 750 and
    not 7: six of the seven rejected-and-side-effecting turns were transfers."""
    import cogno_anima.metakeys as mk
    from cogno_anima.types import PipelineContext
    transfer = ToolExecution(tool="transfer_persona", ok=True, side_effect=True)
    ctx = PipelineContext(user_input="quero falar com o financeiro")
    ctx.metadata["plan_limits"] = dict(PLAN_LIMITS_ONE_ATTEMPT)
    ctx.metadata[mk.ROUTING_ONLY_TOOLS] = ["transfer_persona"]
    ctx = await _pipeline(stub_embedder, FakeEgo(tool_calls=[transfer])).run_turn(
        ctx, TurnConfig(gen_backend=stub_backend, ego_backend=stub_backend, ego_prompt="exec"),
        dispatcher=dispatcher)
    assert ctx.needs_handoff is False
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED


async def test_without_the_host_declaration_a_transfer_still_hands_off(stub_embedder,
                                                                      stub_backend, dispatcher):
    """The other half of the rule above, and the direction that must stay STRICT: with no
    ``ROUTING_ONLY_TOOLS`` declaration ``wrote_for_the_contact`` degrades to
    ``committed_this_turn``, so an undeclared transfer counts as a write and the turn hands off.
    Absence of the host's declaration makes this branch stricter, never looser."""
    transfer = ToolExecution(tool="transfer_persona", ok=True, side_effect=True)
    ctx = await _run(stub_embedder, stub_backend, dispatcher, FakeEgo(tool_calls=[transfer]))
    assert ctx.needs_handoff is True
    assert ctx.stop_reason == "human_handoff"
