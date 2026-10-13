"""The per-attempt ledger records WHAT the judge's reply needed repaired.

``SuperegoStage.evaluate`` (cogno-anima 0.1.9) says what it had to repair in the judge's reply
before it could use it — a chain-of-thought block or a code fence stripped, the object dug out
of prose, a critique that was missing on a rejection — on ``SuperegoResult.repairs``, a list of
marks from the closed ``cogno_anima.repairs.VALID_JUDGE_REPAIRS``. This loop kept the result and
dropped that field, exactly as it once dropped ``verdict_read``, so a judge that answers in a
fence on every turn and a judge that answers a bare object left the same ledger.

The ledger now carries it, per attempt: ``judge_attempts[i]["repairs"]`` is the list of the
verdict that DECIDED the attempt, and ``["fast_repairs"]`` appears only when a fast judge ran
first and the attempt escalated past it. Only marks of the judge's alphabet are written; never
a byte of what the model replied. It does not repeat ``verdict_read`` — the two sit side by side.
"""

from __future__ import annotations

import cogno_anima.metakeys as mk
import pytest
from cogno_anima import repairs as anima_repairs
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import PipelineContext, SuperegoResult, ToolExecution

import cogno_soma
from cogno_soma import (JUDGE_SKIP_PREJUDGED_REPLAY, LEDGER_FAST_REPAIRS,
                        LEDGER_FAST_VERDICT_READ, LEDGER_REPAIRS, LEDGER_VERDICT_READ, Pipeline,
                        TurnConfig, pipeline as pipeline_module)
from tests.conftest import (FakeEgo, FakeID, FakeNER, FakeNoumeno, FakeSuperego,
                            RecordingDispatcher, StubBackend, StubEmbedder, metrics)

JUDGE_MARKS = anima_repairs.VALID_JUDGE_REPAIRS
APPROVE = '{"approved": true, "critique": ""}'
REJECT = '{"approved": false, "critique": "booked the wrong day"}'
SENTINEL = "SENTINEL-REPLY-TEXT"
FENCED = "```json\n" + APPROVE + "\n```"
THOUGHT = "<think>" + SENTINEL + "</think>\n" + APPROVE
IN_PROSE = "My verdict, " + SENTINEL + ": " + APPROVE
NO_CRITIQUE = '{"approved": false}'
NULL_CRITIQUE = '{"approved": false, "critique": null}'
LIST_CRITIQUE = '{"approved": false, "critique": ["' + SENTINEL + '"]}'
STRING_FALSE = '{"approved": "false", "critique": "' + SENTINEL + '"}'

READ_ONLY = [ToolExecution(tool="read_plans", arguments={}, result="plan: 10", ok=True,
                           side_effect=False)]


class _Scripted:
    def __init__(self, replies, model="judge") -> None:
        self.replies, self.model, self.calls = list(replies), model, 0

    async def generate(self, system: str, prompt: str):
        self.calls += 1
        return (self.replies.pop(0) if self.replies else APPROVE), 30, 6


class _Raising:
    model = "down"

    async def generate(self, system: str, prompt: str):
        raise ConnectionError("backend down")


async def _turn(judge, *, fast=None, budget=1, superego=None):
    """One EGO-routed turn with the REAL judge (the list is anima's to report; a double has
    none), doubles for the rest. Returns ``(ctx, ledger)``."""
    pipe = Pipeline(embedder=StubEmbedder(), noumeno=FakeNoumeno("How much is the plan?"),
                    ner=FakeNER(), id_stage=FakeID(route="EGO"),
                    ego=FakeEgo(tool_calls=READ_ONLY, drafts=["The plan costs 10."]),
                    superego=superego or SuperegoStage())
    cfg = TurnConfig(gen_backend=StubBackend(), ego_backend=StubBackend(), ego_prompt="x",
                     voice_backend=_Scripted(["O plano custa 10."] * 4, "voice"),
                     judge_backend=judge, judge_fast_backend=fast, max_corrections=budget)
    ctx = PipelineContext(user_input="quanto custa o plano?")
    ctx = await pipe.run_turn(ctx, cfg, dispatcher=RecordingDispatcher())
    return ctx, ctx.metadata.get(mk.JUDGE_ATTEMPTS) or []


# ── the twin: a repaired reply and a bare one are different rows ───────────────────────────

async def test_twin_a_fenced_verdict_is_recorded_as_such_on_the_attempt():
    """The same approval, once bare and once in a fence. Before this, the two rows were equal."""
    _, bare = await _turn(_Scripted([APPROVE]))
    _, fenced = await _turn(_Scripted([FENCED]))
    assert bare[0]["approved"] is fenced[0]["approved"] is True, "the premise: same verdict"
    assert bare[0][LEDGER_VERDICT_READ] == fenced[0][LEDGER_VERDICT_READ] == "boolean"
    assert bare[0][LEDGER_REPAIRS] == []
    assert fenced[0][LEDGER_REPAIRS] == ["fence_or_think_stripped"]


@pytest.mark.parametrize("reply, expected", [
    (APPROVE, []),
    (REJECT, []),
    (FENCED, ["fence_or_think_stripped"]),
    (THOUGHT, ["fence_or_think_stripped"]),
    (IN_PROSE, ["salvaged_object"]),
    (NO_CRITIQUE, ["default:critique"]),
    (NULL_CRITIQUE, ["null_field:critique"]),
    (LIST_CRITIQUE, ["coerced:critique"]),
])
async def test_every_mark_of_the_judges_alphabet_reaches_the_ledger_as_itself(reply, expected):
    _, ledger = await _turn(_Scripted([reply]))
    assert ledger[0][LEDGER_REPAIRS] == expected


async def test_sync_the_table_above_holds_the_whole_alphabet():
    seen = set()
    for reply in (FENCED, IN_PROSE, NO_CRITIQUE, NULL_CRITIQUE, LIST_CRITIQUE):
        _, ledger = await _turn(_Scripted([reply]))
        seen |= set(ledger[0][LEDGER_REPAIRS])
    assert seen == set(JUDGE_MARKS)


async def test_each_attempt_carries_its_own_list():
    """Attempt 1 is rejected with no critique; attempt 2 is approved in a fence. A list per
    attempt, not the survivor's."""
    _, ledger = await _turn(_Scripted([NO_CRITIQUE, FENCED]), budget=2)
    assert [(e["attempt"], e["approved"], e[LEDGER_REPAIRS]) for e in ledger] == [
        (1, False, ["default:critique"]), (2, True, ["fence_or_think_stripped"])]


# ── it does not repeat the read ────────────────────────────────────────────────────────────

async def test_an_unread_verdict_is_the_reads_to_say_and_the_list_stays_beside_it():
    """A verdict spelled in a string: ``verdict_read`` names it and ``repairs`` does not repeat
    it. Both keys are present — the judge was asked."""
    _, ledger = await _turn(_Scripted([STRING_FALSE]))
    assert ledger[0][LEDGER_VERDICT_READ] == "string_bool"
    assert ledger[0][LEDGER_REPAIRS] == []
    _, ledger = await _turn(_Raising())
    assert (ledger[0][LEDGER_VERDICT_READ], ledger[0][LEDGER_REPAIRS]) == ("call_failed", [])


# ── never a byte of the reply ──────────────────────────────────────────────────────────────

async def test_no_reply_text_reaches_the_list():
    for reply in (THOUGHT, IN_PROSE, LIST_CRITIQUE, STRING_FALSE):
        _, ledger = await _turn(_Scripted([reply]))
        assert SENTINEL in reply, "the premise"
        assert set(ledger[0][LEDGER_REPAIRS]) <= set(JUDGE_MARKS)
        assert SENTINEL not in " ".join(ledger[0][LEDGER_REPAIRS])


class _ReportsItsOwn(FakeSuperego):
    """A stand-in judge that reports whatever it is given — to prove the ledger closes it."""

    def __init__(self, repairs, read="boolean") -> None:
        super().__init__(approve=True)
        self._repairs, self._read = repairs, read

    async def evaluate(self, ctx, backend, *, limits_prompt):
        result = SuperegoResult(response="", approved=True, metrics=metrics("superego_judge"))
        object.__setattr__(result, "verdict_read", self._read)
        object.__setattr__(result, "repairs", self._repairs)
        return result


@pytest.mark.parametrize("reported, written", [
    (["salvaged_object", "the model said " + SENTINEL, "cut:goal", 7, None], ["salvaged_object"]),
    (["default:critique", "default:critique"], ["default:critique"]),
    (("fence_or_think_stripped",), ["fence_or_think_stripped"]),
    (["unparseable", "truncated_retry", "default:goal"], []),     # other stages' marks
    ([], []),
])
async def test_only_a_mark_of_the_judges_alphabet_is_ever_written(reported, written):
    _, ledger = await _turn(StubBackend(), superego=_ReportsItsOwn(reported))
    assert ledger[0][LEDGER_REPAIRS] == written


# ── absent means "not on record" — never an empty list ────────────────────────────────────

@pytest.mark.parametrize("reported", ["salvaged_object", {"salvaged_object": 1}, None, 3])
async def test_a_carrier_that_is_not_a_list_leaves_the_key_absent(reported):
    _, ledger = await _turn(StubBackend(), superego=_ReportsItsOwn(reported))
    assert ledger and LEDGER_REPAIRS not in ledger[0]


async def test_a_judge_that_reports_no_read_leaves_the_key_absent():
    """A stand-in judge returns no ``verdict_read``: it classified nothing, so there is no reply
    whose repairs could be on record — even if the result model defaults the list to ``[]``."""
    _, ledger = await _turn(StubBackend(), superego=FakeSuperego(approve=True))
    assert ledger and LEDGER_REPAIRS not in ledger[0] and LEDGER_VERDICT_READ not in ledger[0]
    _, ledger = await _turn(StubBackend(),
                            superego=_ReportsItsOwn(["salvaged_object"], read=""))
    assert LEDGER_REPAIRS not in ledger[0]


class _NoListAtAll(FakeSuperego):
    """The shape of a result from before the field existed."""

    async def evaluate(self, ctx, backend, *, limits_prompt):
        class _Old:
            approved, critique, verdict_read = True, None, "boolean"
            judge_branch = ""
            metrics = None
        old = _Old()
        old.metrics = metrics("superego_judge")
        return old


async def test_a_result_with_no_list_leaves_the_key_absent():
    _, ledger = await _turn(StubBackend(), superego=_NoListAtAll())
    assert ledger[0][LEDGER_VERDICT_READ] == "boolean", "the premise: the read IS on record"
    assert LEDGER_REPAIRS not in ledger[0]


async def test_an_attempt_the_judge_did_not_read_has_no_list():
    from cogno_soma import PREJUDGED_TEXT_SHA, prejudged_digest

    text = "Hello, tomorrow's class is confirmed."
    held = [ToolExecution(tool="notify_staff", arguments={"message": text}, result="sent",
                          ok=True, side_effect=True)]
    pipe = Pipeline(embedder=StubEmbedder(), noumeno=FakeNoumeno(), ner=FakeNER(),
                    id_stage=FakeID(route="EGO"), ego=FakeEgo(tool_calls=held),
                    superego=SuperegoStage())
    judge = _Scripted([FENCED])
    cfg = TurnConfig(gen_backend=StubBackend(), ego_backend=StubBackend(), ego_prompt="x",
                     voice_backend=_Scripted(["ok"], "voice"), judge_backend=judge)
    ctx = PipelineContext(user_input="sim")
    ctx.metadata[mk.EGO_CONFIRMED] = True
    ctx.metadata[mk.HELD_DELIVERED_TEXT] = {"notify_staff": "message"}
    ctx.metadata[mk.EGO_CONFIRMED_CALLS] = [{"tool": "notify_staff",
                                              "arguments": {"message": text},
                                              PREJUDGED_TEXT_SHA: prejudged_digest(text)}]
    ctx = await pipe.run_turn(ctx, cfg, dispatcher=RecordingDispatcher())
    ledger = ctx.metadata[mk.JUDGE_ATTEMPTS]
    assert ledger[0].get("skipped") == JUDGE_SKIP_PREJUDGED_REPLAY and judge.calls == 0, (
        "the premise: this attempt was not read by the judge")
    assert LEDGER_REPAIRS not in ledger[0] and LEDGER_FAST_REPAIRS not in ledger[0]


# ── the two-tier judge ─────────────────────────────────────────────────────────────────────

async def test_twin_what_the_fast_reply_needed_is_recorded_when_the_strong_decides():
    """The fast judge rejects in a fence with no critique; the strong judge approves bare. The
    deciding list is the strong one's, and without the second key the fast model's repairs —
    the model that answers on EVERY attempt — would be on record nowhere."""
    fast = _Scripted(["```json\n" + NO_CRITIQUE + "\n```"], "fast")
    strong = _Scripted([APPROVE], "strong")
    _, ledger = await _turn(strong, fast=fast)
    assert (fast.calls, strong.calls) == (1, 1), "the premise: it escalated"
    assert ledger[0][LEDGER_REPAIRS] == []
    assert ledger[0][LEDGER_FAST_REPAIRS] == ["fence_or_think_stripped", "default:critique"]
    assert ledger[0][LEDGER_FAST_VERDICT_READ] == "boolean"


async def test_an_escalated_fast_reply_that_needed_nothing_says_so():
    """The key marks an ESCALATION, not a repair: ``[]`` is what makes the rate a rate."""
    fast, strong = _Scripted([REJECT], "fast"), _Scripted([FENCED], "strong")
    _, ledger = await _turn(strong, fast=fast)
    assert ledger[0][LEDGER_FAST_REPAIRS] == []
    assert ledger[0][LEDGER_REPAIRS] == ["fence_or_think_stripped"]


async def test_control_a_fast_approval_is_the_deciding_verdict_and_has_no_second_key():
    fast, strong = _Scripted([FENCED], "fast"), _Scripted([APPROVE], "strong")
    _, ledger = await _turn(strong, fast=fast)
    assert (fast.calls, strong.calls) == (1, 0), "the premise: the fast approval was final"
    assert ledger[0][LEDGER_REPAIRS] == ["fence_or_think_stripped"]
    assert LEDGER_FAST_REPAIRS not in ledger[0]


async def test_control_a_single_tier_judge_never_writes_the_fast_key():
    for reply in (APPROVE, REJECT, FENCED, NO_CRITIQUE, STRING_FALSE):
        _, ledger = await _turn(_Scripted([reply]))
        assert LEDGER_FAST_REPAIRS not in ledger[0]


# ── nothing else on the row moved ──────────────────────────────────────────────────────────

async def test_the_rest_of_the_row_is_what_it_was(monkeypatch):
    """The control: with the two fields switched off, every row is the same minus the keys."""
    replies = [NO_CRITIQUE, FENCED]
    _, with_fields = await _turn(_Scripted(list(replies)), budget=2)
    monkeypatch.setattr(pipeline_module, "_attempt_repairs", lambda judge, fast=None: {})
    _, without = await _turn(_Scripted(list(replies)), budget=2)
    assert all(LEDGER_REPAIRS in row for row in with_fields)
    assert not any(LEDGER_REPAIRS in row for row in without)
    stripped = [{k: v for k, v in row.items() if k not in (LEDGER_REPAIRS, LEDGER_FAST_REPAIRS)}
                for row in with_fields]
    assert stripped == without


# ── the names a host reads, and the alphabet it closes them with ───────────────────────────

def test_the_ledger_keys_are_exported_and_the_values_are_animas_alphabet():
    assert cogno_soma.LEDGER_REPAIRS == "repairs"
    assert cogno_soma.LEDGER_FAST_REPAIRS == "fast_repairs"
    assert {"LEDGER_REPAIRS", "LEDGER_FAST_REPAIRS"} <= set(cogno_soma.__all__)
    # ONE alphabet and ONE closing function, anima's: this module declares neither
    assert pipeline_module.closed_repairs is anima_repairs.closed_repairs
    assert pipeline_module._JUDGE_STAGE in anima_repairs.REPAIRS_BY_STAGE
    assert anima_repairs.REPAIRS_BY_STAGE[pipeline_module._JUDGE_STAGE] is JUDGE_MARKS
