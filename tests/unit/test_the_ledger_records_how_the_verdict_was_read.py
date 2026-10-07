"""The per-attempt ledger records HOW the judge's verdict was read.

``SuperegoStage.evaluate`` (cogno-anima 0.1.3) reads its verdict strictly — only a JSON boolean
counts — and reports the read on ``SuperegoResult.verdict_read``, from the closed
``cogno_anima.VALID_VERDICT_READS``. ``boolean`` means ``approved`` is what the judge SAID; any
other value is the fail-CLOSED fallback. This loop kept the result and dropped that field, so a
rejection the judge gave and a rejection we fell back to were the same ledger row, and the rate
of non-boolean verdicts per model could not be counted from anything a host persists.

The ledger now carries it, per attempt: ``judge_attempts[i]["verdict_read"]`` is the read of
the verdict that DECIDED the attempt, and ``["fast_verdict_read"]`` appears only when a fast
judge ran first and the attempt escalated past it — the one place where the repair (the strong
judge reads it) would otherwise erase the record (the fast model did not answer in a boolean).
Only the VALUE of the alphabet is written; never a byte of what the model replied.
"""

from __future__ import annotations

import json

import cogno_anima
import cogno_anima.metakeys as mk
import pytest
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import PipelineContext, SuperegoResult, ToolExecution

import cogno_soma
from cogno_soma import (JUDGE_SKIP_PREJUDGED_REPLAY, LEDGER_FAST_VERDICT_READ,
                        LEDGER_VERDICT_READ, Pipeline, TurnConfig, pipeline as pipeline_module)
from tests.conftest import (FakeEgo, FakeID, FakeNER, FakeNoumeno, FakeSuperego,
                            RecordingDispatcher, StubBackend, StubEmbedder, metrics)

READS = cogno_anima.VALID_VERDICT_READS
BOOLEAN_TRUE = '{"approved": true, "critique": ""}'
BOOLEAN_FALSE = '{"approved": false, "critique": "booked the wrong day"}'
STRING_FALSE = '{"approved": "false", "critique": "SENTINEL-REPLY-TEXT"}'
DUPLICATED = '{"approved": false, "approved": true}'
PROSE = "It looks fine to me. SENTINEL-REPLY-TEXT"

READ_ONLY = [ToolExecution(tool="read_plans", arguments={}, result="plan: 10", ok=True,
                           side_effect=False)]


class _Scripted:
    def __init__(self, replies, model="judge") -> None:
        self.replies, self.model, self.calls = list(replies), model, 0

    async def generate(self, system: str, prompt: str):
        self.calls += 1
        return (self.replies.pop(0) if self.replies else BOOLEAN_TRUE), 30, 6


class _Raising:
    model = "down"

    async def generate(self, system: str, prompt: str):
        raise ConnectionError("backend down")


async def _turn(judge, *, fast=None, budget=1, superego=None, metadata=None, scope_prompt="",
                id_stage=None):
    """One EGO-routed turn with the REAL judge (the read is anima's to report; a double has
    none), doubles for the rest. Returns ``(ctx, ledger)``."""
    pipe = Pipeline(embedder=StubEmbedder(), noumeno=FakeNoumeno("How much is the plan?"),
                    ner=FakeNER(), id_stage=id_stage or FakeID(route="EGO"),
                    ego=FakeEgo(tool_calls=READ_ONLY, drafts=["The plan costs 10."]),
                    superego=superego or SuperegoStage())
    cfg = TurnConfig(gen_backend=StubBackend(), ego_backend=StubBackend(), ego_prompt="x",
                     voice_backend=_Scripted(["O plano custa 10."] * 4, "voice"),
                     judge_backend=judge, judge_fast_backend=fast, max_corrections=budget,
                     scope_prompt=scope_prompt)
    ctx = PipelineContext(user_input="quanto custa o plano?")
    ctx.metadata.update(metadata or {})
    ctx = await pipe.run_turn(ctx, cfg, dispatcher=RecordingDispatcher())
    return ctx, ctx.metadata.get(mk.JUDGE_ATTEMPTS) or []


# ── the twin: an unread verdict and a rejection the judge gave are different rows ──────────

async def test_twin_a_verdict_in_a_string_is_recorded_as_such_on_the_attempt():
    """The judge answered ``"approved": "false"`` — a string. Anima rejects (fail-closed); the
    ledger row says the rejection is OURS, not the judge's. Before this field the two rows
    below differed only in the critique's text."""
    _, unread = await _turn(_Scripted([STRING_FALSE]))
    _, given = await _turn(_Scripted([BOOLEAN_FALSE]))
    assert unread[0]["approved"] is False and given[0]["approved"] is False
    assert unread[0][LEDGER_VERDICT_READ] == "string_bool"
    assert given[0][LEDGER_VERDICT_READ] == "boolean"
    assert LEDGER_FAST_VERDICT_READ not in unread[0] and LEDGER_FAST_VERDICT_READ not in given[0]


async def test_control_an_approval_read_as_a_boolean_says_so():
    ctx, ledger = await _turn(_Scripted([BOOLEAN_TRUE]))
    assert ctx.stop_reason == "completed"
    assert [(e["approved"], e[LEDGER_VERDICT_READ]) for e in ledger] == [(True, "boolean")]


@pytest.mark.parametrize("reply, expected", [
    (BOOLEAN_TRUE, "boolean"), (BOOLEAN_FALSE, "boolean"), (STRING_FALSE, "string_bool"),
    ('{"approved": 1}', "not_boolean"), ('{"critique": "c"}', "missing"),
    (DUPLICATED, "duplicated"), (PROSE, "unparseable"),
])
async def test_every_read_the_judge_can_report_reaches_the_ledger(reply, expected):
    _, ledger = await _turn(_Scripted([reply]))
    assert ledger[0][LEDGER_VERDICT_READ] == expected


async def test_the_call_that_failed_is_its_own_read():
    _, ledger = await _turn(_Raising())
    assert (ledger[0]["approved"], ledger[0][LEDGER_VERDICT_READ]) == (False, "call_failed")


async def test_per_attempt_each_row_carries_its_own_read():
    """Two attempts, two reads — per ATTEMPT, never backfilled from the last one (the mutation
    that already proved two earlier ledger tests insufficient)."""
    _, ledger = await _turn(_Scripted([STRING_FALSE, BOOLEAN_TRUE]), budget=2)
    assert [(e["attempt"], e["approved"], e[LEDGER_VERDICT_READ]) for e in ledger] == [
        (1, False, "string_bool"), (2, True, "boolean")]


# ── only the VALUE: nothing of what the model replied ──────────────────────────────────────

async def test_only_the_value_of_the_alphabet_is_written_never_the_reply():
    for reply in (STRING_FALSE, PROSE):
        ctx, ledger = await _turn(_Scripted([reply]))
        assert ledger[0][LEDGER_VERDICT_READ] in READS
        assert "SENTINEL" not in json.dumps(ctx.metadata, default=str)


# ── the two-tier judge: the fast read survives the escalation that repairs it ──────────────

async def test_twin_a_fast_verdict_that_could_not_be_read_is_recorded_when_the_strong_decides():
    """The fast judge answers in a string; anima rejects, so the attempt escalates and the
    strong judge decides — the right repair, and without the second key it erases the only
    record that the FAST model did not answer in a JSON boolean."""
    fast, strong = _Scripted([STRING_FALSE], "fast"), _Scripted([BOOLEAN_TRUE], "strong")
    ctx, ledger = await _turn(strong, fast=fast)
    assert (fast.calls, strong.calls) == (1, 1), "the premise: it escalated"
    assert ctx.stop_reason == "completed"
    assert ledger[0][LEDGER_VERDICT_READ] == "boolean", "the STRONG judge's, which decided"
    assert ledger[0][LEDGER_FAST_VERDICT_READ] == "string_bool"


async def test_a_fast_rejection_the_judge_GAVE_is_recorded_too_as_a_boolean():
    """The key marks an ESCALATION, not an error: a fast judge that said ``false`` properly
    and was overruled reads ``boolean`` — which is what makes the rate a rate (non-boolean
    fast reads over ALL escalated fast reads)."""
    fast, strong = _Scripted([BOOLEAN_FALSE], "fast"), _Scripted([BOOLEAN_TRUE], "strong")
    _, ledger = await _turn(strong, fast=fast)
    assert (ledger[0][LEDGER_VERDICT_READ], ledger[0][LEDGER_FAST_VERDICT_READ]) == (
        "boolean", "boolean")


async def test_control_a_fast_approval_is_the_deciding_verdict_and_has_no_second_key():
    fast, strong = _Scripted([BOOLEAN_TRUE], "fast"), _Scripted([BOOLEAN_TRUE], "strong")
    _, ledger = await _turn(strong, fast=fast)
    assert (fast.calls, strong.calls) == (1, 0), "the premise: the fast approval was final"
    assert ledger[0][LEDGER_VERDICT_READ] == "boolean"
    assert LEDGER_FAST_VERDICT_READ not in ledger[0]


async def test_control_a_single_tier_judge_never_writes_the_fast_key():
    for reply in (BOOLEAN_TRUE, BOOLEAN_FALSE, STRING_FALSE):
        _, ledger = await _turn(_Scripted([reply]))
        assert LEDGER_FAST_VERDICT_READ not in ledger[0]


# ── absent means "not on record" — never a value ───────────────────────────────────────────

async def test_a_judge_that_reports_no_read_leaves_the_key_absent():
    """A stand-in judge (a host's own, a test double) returns no ``verdict_read``. The row has
    NO key: absent is "not on record", and a default would read as a read."""
    _, ledger = await _turn(StubBackend(), superego=FakeSuperego(approve=True))
    assert ledger and LEDGER_VERDICT_READ not in ledger[0]


class _LabelsItsOwn(FakeSuperego):
    def __init__(self, label) -> None:
        super().__init__(approve=True)
        self._label = label

    async def evaluate(self, ctx, backend, *, limits_prompt):
        result = SuperegoResult(response="", approved=True, metrics=metrics("superego_judge"))
        object.__setattr__(result, "verdict_read", self._label)
        return result


@pytest.mark.parametrize("label", ["", "BOOLEAN", "ok", "approved", 7, None, ["boolean"]])
async def test_a_label_outside_the_alphabet_is_never_written(label):
    _, ledger = await _turn(StubBackend(), superego=_LabelsItsOwn(label))
    assert ledger and LEDGER_VERDICT_READ not in ledger[0]


async def test_an_attempt_the_judge_did_not_read_has_no_read():
    """The pre-approved replay is recorded as ``skipped``: no judge was called, so there is no
    verdict to have been read — the row carries no key, not ``boolean``."""
    from cogno_soma import PREJUDGED_TEXT_SHA, prejudged_digest

    text = "Olá, a aula de amanhã foi confirmada."
    held = [ToolExecution(tool="notify_staff", arguments={"message": text}, result="sent",
                          ok=True, side_effect=True)]
    pipe = Pipeline(embedder=StubEmbedder(), noumeno=FakeNoumeno(), ner=FakeNER(),
                    id_stage=FakeID(route="EGO"), ego=FakeEgo(tool_calls=held),
                    superego=SuperegoStage())
    judge = _Scripted([BOOLEAN_TRUE])
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
    assert LEDGER_VERDICT_READ not in ledger[0]


# ── the guard's mark: a turn that never calls the guard says nothing ───────────────────────

@pytest.mark.parametrize("scope_prompt, id_stage, why", [
    ("", None, "no scope prompt: the guard is never called"),
    ("only plans", FakeID(route="EGO", blocked=True), "the PII gate ends the turn before it"),
])
async def test_a_turn_that_never_calls_the_guard_carries_no_stale_guard_read(
        scope_prompt, id_stage, why):
    """``ctx.metadata[mk.SCOPE_VERDICT_READ]`` is the guard's, per turn, and the guard removes
    it itself on the paths where it asks nothing — but only when it is CALLED. On the two exits
    that never call it, a value a carrier left there would read as this turn's."""
    ctx, _ = await _turn(_Scripted([BOOLEAN_TRUE]), scope_prompt=scope_prompt,
                         id_stage=id_stage, metadata={mk.SCOPE_VERDICT_READ: "string_bool"})
    assert mk.SCOPE_VERDICT_READ not in ctx.metadata, why


async def test_control_a_consulted_guard_leaves_its_own_read():
    """The control of the row above: the pop is BEFORE the guard, never after — a guard that
    asked the classifier leaves the read anima stamped."""
    ctx, _ = await _turn(_Scripted([BOOLEAN_TRUE]), scope_prompt="only plans",
                         metadata={mk.SCOPE_VERDICT_READ: "string_bool"})
    # gen_backend (the StubBackend) answers "{}" to the guard: an object with no `blocked`
    assert ctx.metadata[mk.SCOPE_VERDICT_READ] == "missing"


# ── the names a host reads, and the alphabet it closes them with ───────────────────────────

def test_the_ledger_keys_are_exported_and_the_values_are_animas_alphabet():
    assert cogno_soma.LEDGER_VERDICT_READ == "verdict_read"
    assert cogno_soma.LEDGER_FAST_VERDICT_READ == "fast_verdict_read"
    assert {"LEDGER_VERDICT_READ", "LEDGER_FAST_VERDICT_READ"} <= set(cogno_soma.__all__)
    # ONE alphabet, anima's: this module imports it and declares none of its own
    assert pipeline_module.VALID_VERDICT_READS is cogno_anima.VALID_VERDICT_READS


async def test_sync_every_value_of_the_alphabet_can_reach_the_ledger_and_no_other_does():
    """Both directions, like the block tables: every value anima's judge can report lands in
    the ledger as itself, and the set the ledger can hold is exactly the alphabet."""
    replies = {"boolean": BOOLEAN_TRUE, "string_bool": STRING_FALSE,
               "not_boolean": '{"approved": null}', "missing": "{}", "duplicated": DUPLICATED,
               "unparseable": PROSE}
    seen = set()
    for expected, reply in replies.items():
        _, ledger = await _turn(_Scripted([reply]))
        assert ledger[0][LEDGER_VERDICT_READ] == expected
        seen.add(ledger[0][LEDGER_VERDICT_READ])
    _, ledger = await _turn(_Raising())
    seen.add(ledger[0][LEDGER_VERDICT_READ])
    assert seen == set(READS)
    for value in READS:
        assert pipeline_module._attempt_read(SuperegoResult(
            approved=False, verdict_read=value, metrics=metrics("j"))) == {
                LEDGER_VERDICT_READ: value}
