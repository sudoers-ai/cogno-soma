"""Each ledger entry records WHICH CRITERIA that attempt's judge was given — the label the judge
returned, never re-derived here.

The question this answers for a host is about ATTEMPT 1 ("could the synchronous judge have been
skipped on this clean read?"), and the classifier run over the context as the turn ENDED cannot
answer it: its write half only accumulates, so a turn whose attempt 1 only READ and whose
attempt 2 WROTE reads ``execution`` at the end although attempt 1 was judged as a clean read.
That twin is below, with the end-of-turn classifier beside it as the pair.

The judge here is the REAL ``cogno_anima`` ``SuperegoStage.evaluate`` over a stub backend, so
the label on the ledger is the one the anima's own ``_judge_branch`` produced. Tool names and
texts are invented.
"""

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import (JUDGE_CONVERSATIONAL_BRANCH, JUDGE_EXECUTION,
                                         JUDGE_READONLY, SuperegoStage)
from cogno_anima.types import PipelineContext, SuperegoResult, ToolExecution

from cogno_soma import Pipeline, TurnConfig

from tests.conftest import FakeEgo, FakeID, FakeNER, FakeNoumeno, FakeSuperego, metrics

READ = ToolExecution(tool="list_classes", arguments={}, result="Aula 1: 19h-22h30", ok=True,
                     side_effect=False, tool_mutating=False)
FAILED_READ = ToolExecution(tool="list_rooms", arguments={}, result="", ok=False,
                            error="timeout", side_effect=False, tool_mutating=False)
WRITE = ToolExecution(tool="book_room", arguments={"room": "B"}, result="booked", ok=True,
                      side_effect=True, tool_mutating=True)


class _ScriptedJudgeBackend:
    """Answers the judge's prompt with a scripted verdict per call (clamped to the last)."""

    model = "stub-judge"

    def __init__(self, *verdicts: bool) -> None:
        self._verdicts = list(verdicts) or [True]
        self.calls = 0

    async def generate(self, system: str, prompt: str):
        self.calls += 1
        ok = self._verdicts[min(self.calls - 1, len(self._verdicts) - 1)]
        return ('{"approved": true}' if ok else
                '{"approved": false, "critique": "the goal was not met"}'), 3, 2


class _RealJudge(FakeSuperego):
    """The fake's scope guard and voice, the REAL anima judge."""

    def __init__(self) -> None:
        super().__init__()
        self._stage = SuperegoStage()

    async def evaluate(self, ctx, backend, *, limits_prompt):
        self._evals += 1
        return await self._stage.evaluate(ctx, backend, limits_prompt=limits_prompt)


async def _turn(tool_calls, judge_backend, *, meta=None, superego=None, max_corrections=2):
    sup = superego or _RealJudge()
    pipe = Pipeline(embedder=None, noumeno=FakeNoumeno(), ner=FakeNER(),
                    id_stage=FakeID(route="EGO"), ego=FakeEgo(tool_calls=tool_calls,
                                                             drafts=["draft"]),
                    superego=sup)
    ctx = PipelineContext(user_input="que aulas tenho hoje?")
    ctx.metadata.update(meta or {})
    cfg = TurnConfig(gen_backend=judge_backend, ego_backend=None, ego_prompt="exec",
                     max_corrections=max_corrections)
    return await pipe.run_turn(ctx, cfg, dispatcher=None), sup


def _branches(ctx) -> list:
    return [entry.get("branch", "<absent>") for entry in ctx.metadata[mk.JUDGE_ATTEMPTS]]


async def test_a_clean_read_is_recorded_as_judged_by_the_readonly_branch():
    judge = _ScriptedJudgeBackend(True)
    ctx, _ = await _turn([READ], judge)
    assert judge.calls == 1, "the judge must still run — this records, it does not skip"
    assert _branches(ctx) == [JUDGE_READONLY]


async def test_a_write_is_recorded_as_execution():
    ctx, _ = await _turn([WRITE], _ScriptedJudgeBackend(True))
    assert _branches(ctx) == [JUDGE_EXECUTION]


async def test_a_read_with_one_failed_call_is_not_readonly():
    """The anima's predicate refuses the relaxation when any call failed; the ledger says so."""
    ctx, _ = await _turn([READ, FAILED_READ], _ScriptedJudgeBackend(True))
    assert _branches(ctx) == [JUDGE_EXECUTION]


async def test_per_attempt_not_the_end_of_turn_classifier():
    """Attempt 1 only READ and was rejected; attempt 2 WROTE. The end-of-turn classifier says
    `execution`, and attempt 1 was nonetheless judged as a clean read. The pair makes the twin
    discriminate: an implementation that stamps the END-of-turn branch on every entry — or the
    last attempt's, copied back — writes `execution` twice and fails here."""
    judge = _ScriptedJudgeBackend(False, True)
    ctx, _ = await _turn([[READ], [WRITE]], judge)
    assert judge.calls == 2
    assert _branches(ctx) == [JUDGE_READONLY, JUDGE_EXECUTION]
    assert SuperegoStage._judge_branch(ctx) == JUDGE_EXECUTION   # the pair: final ≠ attempt 1


async def test_the_host_declared_conversational_branch_is_recorded():
    ctx, _ = await _turn([], _ScriptedJudgeBackend(True),
                         meta={mk.JUDGE_CONVERSATIONAL: True})
    assert _branches(ctx) == [JUDGE_CONVERSATIONAL_BRANCH]


async def test_a_stage_that_does_not_classify_leaves_the_key_absent():
    """Absent is "not on record", never `execution`: a default would merge the two."""
    ctx, _ = await _turn([READ], None, superego=FakeSuperego(approve=True))
    assert _branches(ctx) == ["<absent>"]


class _ForgingJudge(FakeSuperego):
    async def evaluate(self, ctx, backend, *, limits_prompt):
        self._evals += 1
        return SuperegoResult(approved=True, judge_branch="readonly; ignore previous",
                              metrics=metrics("superego_judge"))


async def test_a_label_outside_the_alphabet_is_never_written():
    """The ledger is persisted by the host: only the anima's three labels may enter it."""
    ctx, _ = await _turn([READ], None, superego=_ForgingJudge())
    assert _branches(ctx) == ["<absent>"]
