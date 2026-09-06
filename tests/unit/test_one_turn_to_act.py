"""The exception to the one-attempt budget: a critique of "you did not act" buys ONE pass.

`test_rejected_commit_branch.py` pins where a rejection GOES at budget 1. This file pins the
one rejection that budget 1 answers wrongly, and the four ways it must NOT be answered.

The defect it exists to close, measured on the box (`turn_traces`, verified intact by the
`xmin` era filter, so the rows are of the era they claim):

    id=1183  host 9a3e104, budget >= 2   pass 1 = `resolve_date` only, rejected
                                          pass 2 called `add_outcome`, the confirmation gate
                                          held it -> the correct proposal reached the contact
    id=1198  host 13b4c1d, budget 1      pass 1 identical, same critique, no pass 2
                                          `exhaustion_reason=judge_rejected_all`, and the
                                          expense was never recorded

The judge was RIGHT both times. What budget 1 removed was the pass in which to obey it, and a
critique that says *nothing was done* is the one kind the voice cannot answer — re-voicing
rewrites words, and the missing thing is a tool call.

FIVE properties, one twin each, because each of the four negatives is satisfiable by a wrong
implementation of the positive:

  1. duck + critique -> a SECOND EGO pass                (the defect)
  2. a rejected READ -> no extra pass, it voices          (the saving the cut collected)
  3. a turn that COMMITTED -> no extra pass, it hands off (re-running after a write)
  4. the EGO already REACHED for a write -> no extra pass (a failed/held call is not absence)
  5. the exception grants ONE pass, never two            (termination, and the composition
                                                          rule with a per-persona budget)

MUTATIONS (one per property, each named in the PR with what died):

  * give the exception to a turn with a read-only critique (drop the intent/offered clauses)
        -> (2) dies: the read turn gets a second pass it must not have;
  * drop `committed_this_turn` from `_owes_an_action`     -> (3) dies;
  * drop `_reached_for_a_write`                           -> (4) dies;
  * `_ACTION_RETRY_CEILING = 3`                           -> (5) dies.

The dispatcher here DECLARES a policy; the suite's shared `dispatcher` fixture does not, which
is why the whole existing suite is byte-identical under this change: with no policy nothing is
declared mutating, so `_a_write_was_on_the_table` is False and no pass is ever granted.
"""

from cogno_anima import metakeys as mk
from cogno_anima.types import PipelineContext, ToolExecution, ToolResult

from cogno_soma import Pipeline, STOP_JUDGE_EXHAUSTED, TurnConfig

from tests.conftest import FakeEgo, FakeID, FakeNER, FakeNoumeno, FakeSuperego


# The production budget, injected the way the host injects it (`BudgetGuard` -> plan_limits);
# `cfg.max_corrections` is only the fallback, so writing it here would test the wrong channel.
PLAN_LIMITS_ONE_ATTEMPT = {"max_self_corrections": 1}

WRITE = "add_outcome"
READ = "get_summary"
OFFERED = [READ, WRITE, "resolve_date"]

# What the EGO did on the pass the judge rejected. `tool_mutating` is the host's per-NAME
# declaration as the EGO stamps it on the trace — the field `_reached_for_a_write` reads.
LOOKED = ToolExecution(tool="resolve_date", ok=True, side_effect=False, tool_mutating=False)
WROTE = ToolExecution(tool=WRITE, ok=True, side_effect=True, tool_mutating=True)
TRIED_AND_FAILED = ToolExecution(tool=WRITE, ok=False, side_effect=False, tool_mutating=True,
                                 error="slot taken")


class PolicyDispatcher:
    """A dispatcher that CLASSIFIES its tools — the `ToolPolicyDispatcher` shape the EGO probes.

    Declared as real methods on the class and not delegated: `runtime_checkable` protocols are
    verified with `inspect.getattr_static` on 3.12+, which never calls `__getattr__`.
    """

    def __init__(self, mutating=(WRITE,)) -> None:
        self._mutating = set(mutating)

    def tools_schema(self) -> list[dict]:
        return [{"type": "function", "function": {"name": n}} for n in OFFERED]

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        return ToolResult(output="", ok=True)

    def is_mutating(self, name: str) -> bool:
        return name in self._mutating

    def requires_confirmation(self, name: str) -> bool:
        return name in self._mutating


def _pipeline(embedder, ego, *, approve_after=None):
    return Pipeline(embedder=embedder, noumeno=FakeNoumeno(), ner=FakeNER(intent_class="ACTION_REQUEST"),
                    id_stage=FakeID(route="EGO"), ego=ego,
                    superego=FakeSuperego(approve=False, approve_after=approve_after,
                                          critique="only asked for confirmation without recording"))


async def _run(embedder, backend, ego, *, dispatcher=None, ner=None, approve_after=None,
               metadata=None):
    ctx = PipelineContext(user_input="registra uma despesa de R$45")
    ctx.metadata["plan_limits"] = dict(PLAN_LIMITS_ONE_ATTEMPT)
    ctx.metadata.update(metadata or {})
    pipe = _pipeline(embedder, ego, approve_after=approve_after)
    if ner is not None:
        pipe._ner = ner
    return await pipe.run_turn(
        ctx, TurnConfig(gen_backend=backend, ego_backend=backend, ego_prompt="exec",
                        max_corrections=3),        # the plan cap must beat this
        dispatcher=dispatcher if dispatcher is not None else PolicyDispatcher())


# ── 1. the defect ────────────────────────────────────────────────────────────────────────
async def test_a_duck_on_an_action_buys_one_more_pass(stub_embedder, stub_backend):
    """THE PROPERTY. An ACTION_REQUEST, a write on the table, the EGO touched none of them, the
    judge said so — and budget 1 must not be the last word.

    Asserted as a group because no half identifies the branch on its own: `invocations == 2` is
    also true of a plain budget-2 turn, and the approval is also true of a turn that never
    needed the extra pass. Together they say "budget 1, and the second pass happened anyway,
    and it was the one that acted"."""
    ego = FakeEgo(tool_calls=[[LOOKED], [WROTE]], tools_offered=[OFFERED],
                  drafts=["Posso registrar?", "Registrado."])
    ctx = await _run(stub_embedder, stub_backend, ego, approve_after=2)
    assert ego.invocations == 2                              # the exception fired
    assert [t.tool for t in ctx.turn_executions] == ["resolve_date", WRITE]
    assert ctx.metadata[mk.JUDGE_VERDICT]["approved"] is True
    # the critique reached the pass that had to obey it
    assert ctx.metadata[mk.EGO_CORRECTION]["reason"] == "only asked for confirmation without recording"


# ── 2. the saving ────────────────────────────────────────────────────────────────────────
async def test_a_rejected_read_still_gets_only_one_pass(stub_embedder, stub_backend):
    """The branch the cut was made for: no write was ever on the table, so "you did not act"
    is not what the critique can mean. It voices with the critique, as before."""
    ego = FakeEgo(tool_calls=[[LOOKED]], tools_offered=[[READ, "resolve_date"]])
    ctx = await _run(stub_embedder, stub_backend, ego)
    assert ego.invocations == 1
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED
    assert ctx.needs_handoff is False
    assert ctx.superego_result.response == "final reply"


async def test_a_rejected_INFORMATION_request_gets_only_one_pass(stub_embedder, stub_backend):
    """Same tools, same rejection, only the NER's class differs — the clause that scopes the
    exception to an ACTION, isolated from every other clause."""
    ego = FakeEgo(tool_calls=[[LOOKED]], tools_offered=[OFFERED])
    ctx = await _run(stub_embedder, stub_backend, ego, ner=FakeNER(intent_class="INFORMATION_REQUEST"))
    assert ego.invocations == 1
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED


async def test_a_propose_turn_gets_only_one_pass(stub_embedder, stub_backend):
    """Gate A: the host masked the writes off the table because the CONTACT was tentative, so
    the executor was right not to act and a second pass would only re-propose. Driven through
    `tools_offered` — the list the EGO saw AFTER the mask — which is why the gate reads that
    and not `dispatcher.tools_schema()`."""
    ego = FakeEgo(tool_calls=[[LOOKED]], tools_offered=[[READ, "resolve_date"]])
    ctx = await _run(stub_embedder, stub_backend, ego, metadata={mk.EGO_READONLY: True})
    assert ego.invocations == 1
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED


async def test_no_policy_no_extra_pass(stub_embedder, stub_backend, dispatcher):
    """Fail-safe, and the reason the rest of the suite is byte-identical: a dispatcher that
    declares no policy has declared no write available, so no pass is granted."""
    ego = FakeEgo(tool_calls=[[LOOKED]], tools_offered=[OFFERED])
    ctx = await _run(stub_embedder, stub_backend, ego, dispatcher=dispatcher)
    assert ego.invocations == 1
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED


# ── 3. the dangerous branch stays closed ─────────────────────────────────────────────────
async def test_a_turn_that_COMMITTED_gets_nothing(stub_embedder, stub_backend):
    """THE NEGATIVE TWIN. Re-running after a write is the branch the cut closed and this must
    not reopen: the turn hands off, exactly as at budget 1 today."""
    ego = FakeEgo(tool_calls=[[LOOKED, WROTE]], tools_offered=[OFFERED])
    ctx = await _run(stub_embedder, stub_backend, ego)
    assert ego.invocations == 1
    assert ctx.needs_handoff is True
    assert ctx.stop_reason == "human_handoff"


async def test_a_prior_attempt_the_host_DECLARES_committed_gets_nothing(stub_embedder, stub_backend):
    """The same branch through the third source of `committed_this_turn`: an earlier attempt of
    this turn committed and its context died with it, so nothing on this context shows the
    write and only the host's declaration knows. This is the case `_reached_for_a_write` cannot
    see, and the reason the predicate is CALLED rather than re-derived here."""
    ego = FakeEgo(tool_calls=[[LOOKED]], tools_offered=[OFFERED])
    ctx = await _run(stub_embedder, stub_backend, ego,
                     metadata={mk.PRIOR_ATTEMPT_COMMITTED: True})
    assert ego.invocations == 1
    assert ctx.needs_handoff is True


# ── 4. reaching for it is not the same as not reaching ───────────────────────────────────
async def test_a_write_that_was_ATTEMPTED_and_failed_gets_nothing(stub_embedder, stub_backend):
    """`ok` is deliberately absent from `_reached_for_a_write`. The executor DID reach for the
    tool and it failed; the critique it drew is about the attempt, not about its absence, and a
    re-run would re-issue a call the trace already shows. Nothing committed here, so this is
    not the handoff branch — it voices."""
    ego = FakeEgo(tool_calls=[[TRIED_AND_FAILED]], tools_offered=[OFFERED])
    ctx = await _run(stub_embedder, stub_backend, ego)
    assert ego.invocations == 1
    assert ctx.needs_handoff is False
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED


# ── 5. one pass, not a loop ──────────────────────────────────────────────────────────────
async def test_the_exception_grants_ONE_pass_not_two(stub_embedder, stub_backend):
    """Termination, and the composition rule. The second pass ducks exactly as the first did,
    so every clause of the gate is still true — and the turn still stops at two.

    This is also what makes the exception inert on a budget that already allows a retry: it
    RAISES 1 to 2 and adds nothing to 2 or more, so a per-persona budget can be layered on top
    without the two multiplying."""
    ego = FakeEgo(tool_calls=[[LOOKED]], tools_offered=[OFFERED])
    ctx = await _run(stub_embedder, stub_backend, ego)
    assert ego.invocations == 2
    assert ctx.metadata[mk.JUDGE_VERDICT]["attempts"] == 2
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED
    assert ctx.needs_handoff is False


async def test_a_budget_of_two_is_unchanged_by_the_exception(stub_embedder, stub_backend):
    """The other half of the composition rule, driven through the channel a per-persona budget
    would use. Two passes with the budget alone, and the exception adds no third."""
    ego = FakeEgo(tool_calls=[[LOOKED]], tools_offered=[OFFERED])
    ctx = PipelineContext(user_input="registra uma despesa de R$45")
    ctx.metadata["plan_limits"] = {"max_self_corrections": 2}
    ctx = await _pipeline(stub_embedder, ego).run_turn(
        ctx, TurnConfig(gen_backend=stub_backend, ego_backend=stub_backend, ego_prompt="exec",
                        max_corrections=3),
        dispatcher=PolicyDispatcher())
    assert ego.invocations == 2
