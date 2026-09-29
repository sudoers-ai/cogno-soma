"""A held MESSAGE the judge already approved is not judged again once the «sim» sends it.

The shape, measured on a downstream host (the data stays there; names and texts here are
invented): a staff reader confirms a proposed `notify_user`, the replay DELIVERS it (`ok=True`,
`side_effect=True`), and the post-send judge rejects the execution for want of a read THIS turn —
the read was made on the proposal turn, where the same judge had read and approved those exact
bytes. Budget 1: the turn ended in `human_handoff` before the voice, over a message that went out
right. The skip is `_prejudged_replay`'s four conditions; every control below is one of them
failing, and each goes to today's judge with today's ledger, byte for byte.
"""

import hashlib
import json

import pytest

import cogno_anima.metakeys as mk
from cogno_anima.stages.ego import EgoStage
from cogno_anima.types import PipelineContext, ToolExecution, ToolResult

import cogno_soma.pipeline as pipeline_mod
from cogno_soma import (JUDGE_SKIP_PREJUDGED_REPLAY, JUDGE_SKIPS, PREJUDGED_TEXT_SHA, Pipeline,
                        STOP_JUDGE_EXHAUSTED, TurnConfig, prejudged_digest)

from tests.conftest import (FakeEgo, FakeID, FakeNER, FakeNoumeno, FakeSuperego,
                            RecordingDispatcher, metrics)

MESSAGE = "Olá Marisol, a aula de Química de quinta, turma 2B, passa para as 10h."
ARGS = {"target": "Marisol Quaresma", "message": MESSAGE}
DECLARED = {"notify_user": "message"}
CRITIQUE = "the message states a date, a class and a subject with no schedule read this turn"
VOICE = "Pronto, a Marisol já foi avisada."


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sent(args=None, *, ok=True, side_effect=True, tool="notify_user") -> ToolExecution:
    return ToolExecution(tool=tool, arguments=dict(args or ARGS), ok=ok, side_effect=side_effect,
                         tool_mutating=True,
                         result="Message delivered." if ok else "Delivery failed.",
                         error=None if ok else "upstream_unavailable")


def _row(args=None, *, stamp="ok", tool="notify_user") -> dict:
    row = {"tool": tool, "arguments": dict(args or ARGS)}
    if stamp == "ok":
        row[PREJUDGED_TEXT_SHA] = prejudged_digest(row["arguments"]["message"])
    elif stamp is not None:
        row[PREJUDGED_TEXT_SHA] = stamp
    return row


class _HoldsANewOne(FakeEgo):
    """The replay, plus a NEW message the executor held on the same attempt."""

    async def process(self, ctx, backend, dispatcher, *, system_prompt):
        ctx = await super().process(ctx, backend, dispatcher, system_prompt=system_prompt)
        ctx.ego_result.pending_confirmation = [ToolExecution(
            tool="notify_user", arguments={"target": "Rufino", "message": "outra coisa"},
            ok=False, error="needs_confirmation", tool_mutating=True)]
        return ctx


async def _run(executed, rows, *, declared=DECLARED, confirmed=True, ego_cls=FakeEgo,
               consult=None, flag=True):
    ego = ego_cls(tool_calls=[executed], tools_offered=[["notify_user"]], drafts=["Feito."])
    sup = FakeSuperego(approve=False, critique=CRITIQUE, voice=VOICE)
    pipe = Pipeline(embedder=None, noumeno=FakeNoumeno(), ner=FakeNER(intent_class="ACTION_REQUEST"),
                    id_stage=FakeID(route="EGO"), ego=ego, superego=sup)
    ctx = PipelineContext(user_input="sim")
    ctx.consult_result = consult
    if declared is not None:
        ctx.metadata[mk.HELD_DELIVERED_TEXT] = dict(declared)
    if confirmed:
        if flag:
            ctx.metadata[mk.EGO_CONFIRMED] = sorted({r["tool"] for r in rows})
        ctx.metadata[mk.EGO_CONFIRMED_CALLS] = [dict(r) for r in rows]
    ctx = await pipe.run_turn(ctx, TurnConfig(gen_backend=None, ego_backend=None,
                                              ego_prompt="exec", max_corrections=1),
                              dispatcher=RecordingDispatcher())
    return ctx, sup, ego


def _ledger_digest(ctx) -> str:
    return _sha(json.dumps(ctx.metadata.get(mk.JUDGE_ATTEMPTS), sort_keys=True,
                           ensure_ascii=False, default=str))


# ── the twin: the measured shape ────────────────────────────────────────────────────────────

async def test_the_measured_shape_is_not_judged_again_and_the_voice_speaks():
    ctx, sup, ego = await _run([_sent()], [_row()])
    assert sup._evals == 0, "the judge read again a message it had approved before it was sent"
    assert ego.invocations == 1
    assert ctx.metadata[mk.JUDGE_VERDICT] == {"approved": True, "attempts": 1}
    assert ctx.stop_reason == "completed" and ctx.needs_handoff is False
    assert ctx.superego_result.response == VOICE
    (row,) = ctx.metadata[mk.JUDGE_ATTEMPTS]
    assert row["skipped"] == JUDGE_SKIP_PREJUDGED_REPLAY and row["skipped"] in JUDGE_SKIPS
    assert row["attempt"] == 1 and row["approved"] is True and row["critique"] == ""
    assert row["committed"] is True and [t["tool"] for t in row["tools"]] == ["notify_user"]


async def test_a_refused_re_issue_of_the_same_call_ran_nothing():
    """The anima blocks a model that calls the confirmed tool again after the replay; that row
    executed nothing, so it is not an extra call."""
    blocked = ToolExecution(tool="notify_user", arguments=dict(ARGS), ok=False,
                            error="duplicate", tool_mutating=True, result="[DUPLICATE] …")
    ctx, sup, _ego = await _run([_sent(), blocked], [_row()])
    assert sup._evals == 0
    assert ctx.metadata[mk.JUDGE_ATTEMPTS][0]["skipped"] == JUDGE_SKIP_PREJUDGED_REPLAY


# ── the controls: each goes to today's judge, with today's ledger byte for byte ──────────────

OTHER = {"target": "Marisol Quaresma", "message": MESSAGE + "!"}  # one byte more

CONTROLS = {
    # (a) something beside the replay ran
    "an_extra_call": ([_sent(), ToolExecution(tool="resolve_date", arguments={"text": "quinta"},
                                              ok=True, side_effect=False, tool_mutating=False,
                                              result="2026-10-01")], [_row()], DECLARED, True),
    # (b) the tool does not declare a text sent to a person
    "an_undeclared_tool": ([_sent()], [_row()], {}, True),
    # (c) the replay failed
    "a_failed_replay": ([_sent(ok=False, side_effect=False)], [_row()], DECLARED, True),
    # no «sim» at all
    "no_confirmation": ([_sent()], [_row()], DECLARED, False),
    # (d) the stamp is ABSENT
    "no_stamp": ([_sent()], [_row(stamp=None)], DECLARED, True),
    # (d) the stamp is of a text one byte different from the one sent
    "a_different_stamp": ([_sent()], [_row(stamp=_sha(OTHER["message"]))], DECLARED, True),
    # (a) the other direction: a confirmed call that did NOT run (only one of two went out)
    "a_confirmed_call_not_executed": ([_sent()], [_row(), _row(OTHER)], DECLARED, True),
    # rows with no «sim» flag (the host's onboarding injection is this shape): not a confirmation
    "rows_without_the_confirmation_flag": ([_sent()], [_row()], DECLARED, "rows_only"),
    # (a) as a MULTISET: one confirmed call that went out TWICE, both ok, no duplicate block — a
    # DOUBLE DELIVERY, the class the hosts' idempotency guard (`cogno_anima.tools.
    # IdempotentDispatcher`) exists to prevent. A host's third-party gate may let it out once,
    # but a lib cannot assume every host has one: a double send is judged, never pre-approved.
    "a_double_delivery": ([_sent(), _sent()], [_row()], DECLARED, True),
    # (a) a FAILED extra call beside the ok replay, and not a duplicate: it ran, it is not the
    # replay, so the attempt is not only the replay. Only a blocked re-issue OF A CONFIRMED CALL
    # is excluded — a filter on `ok is False` alone would wave this one through.
    "a_failed_extra_call": ([_sent(), ToolExecution(
        tool="list_classes", arguments={}, ok=False, side_effect=False, tool_mutating=False,
        error="upstream_unavailable", result="")], [_row()], DECLARED, True),
}

# The ledger each control produces on soma `main` f4aed352, BEFORE this change (sha256 of the
# canonical JSON). A landing-time proof that "judged as today" is today's bytes, not merely a
# judge call; regenerated only when the ledger's shape changes on purpose.
LEDGER_ON_MAIN = {
    "an_extra_call": "a70fe3cd97467288bda7bc31782bcc87c32bc4f2ebb9b01b43b6885f784468c0",
    "an_undeclared_tool": "e748c10a058c34aae2c9c68169d009124fc82d75fca66a1815dbfe46e3c0d7ef",
    "a_failed_replay": "d10eda02fd63b32c73e520ba3f5a7011521d6ecfc64c15231745f450d3948f81",
    "no_confirmation": "e748c10a058c34aae2c9c68169d009124fc82d75fca66a1815dbfe46e3c0d7ef",
    "no_stamp": "e748c10a058c34aae2c9c68169d009124fc82d75fca66a1815dbfe46e3c0d7ef",
    "a_different_stamp": "e748c10a058c34aae2c9c68169d009124fc82d75fca66a1815dbfe46e3c0d7ef",
    "a_confirmed_call_not_executed": "e748c10a058c34aae2c9c68169d009124fc82d75fca66a1815dbfe46e3c0d7ef",
    "rows_without_the_confirmation_flag": "e748c10a058c34aae2c9c68169d009124fc82d75fca66a1815dbfe46e3c0d7ef",
    "a_double_delivery": "2ee195730f6676edcd04cbb85a769bb1adb26c19de531a4d6dc546cd5b89a255",
    "a_failed_extra_call": "ede980ec4245af156f494818ef927582e7b4cf8481d50e89c97b99daf5ea44ce",
}


@pytest.mark.parametrize("name", sorted(CONTROLS))
async def test_control_goes_to_todays_judge_with_todays_ledger(name):
    """Each of the four conditions failing, and the neighbours of each. ``a_double_delivery`` is
    the one a SET comparison would let through: one confirmed call executed twice, both ``ok``,
    no duplicate block — a double send, the class the idempotency guard exists to prevent
    (``cogno_anima.tools.IdempotentDispatcher``). It is judged, never pre-approved."""
    executed, rows, declared, confirmed = CONTROLS[name]
    ctx, sup, _ego = await _run(executed, rows, declared=declared, confirmed=bool(confirmed),
                                flag=confirmed is True)
    assert sup._evals == 1, f"{name}: the judge must read this attempt"
    assert all("skipped" not in e for e in ctx.metadata[mk.JUDGE_ATTEMPTS])
    assert ctx.metadata[mk.JUDGE_VERDICT] == {"approved": False, "attempts": 1}
    expected = STOP_JUDGE_EXHAUSTED if name == "a_failed_replay" else "human_handoff"
    assert ctx.stop_reason == expected
    assert _ledger_digest(ctx) == LEDGER_ON_MAIN[name]


@pytest.mark.parametrize("name", sorted(CONTROLS))
async def test_control_is_the_same_turn_the_predicate_off_would_give(name, monkeypatch):
    """The same controls against the loop with the skip switched off — the in-process half of
    "byte for byte": the predicate answering False changes nothing else in the turn."""
    executed, rows, declared, confirmed = CONTROLS[name]
    kw = {"declared": declared, "confirmed": bool(confirmed), "flag": confirmed is True}
    on, _s, _e = await _run(executed, rows, **kw)
    monkeypatch.setattr(pipeline_mod, "_prejudged_replay", lambda ctx: False)
    off, _s, _e = await _run(executed, rows, **kw)
    assert on.metadata[mk.JUDGE_ATTEMPTS] == off.metadata[mk.JUDGE_ATTEMPTS]
    assert on.metadata[mk.JUDGE_VERDICT] == off.metadata[mk.JUDGE_VERDICT]
    assert on.stop_reason == off.stop_reason


# ── two more of the author's: another executor, and another held message, beside the replay ──

def _consulted():
    from cogno_anima.types import EgoResult, EgoStep
    read = ToolExecution(tool="list_classes", arguments={}, ok=True, side_effect=False,
                         tool_mutating=False, result="2B: quinta 10h")
    return EgoResult(steps=[EgoStep(index=0, path="native", tool_calls=[read])],
                     metrics=metrics("ego"))


@pytest.mark.parametrize("extra", ["a_consult_ran", "a_new_hold_beside_it"])
async def test_control_another_executor_or_another_hold_is_judged(extra, monkeypatch):
    kw = ({"consult": _consulted()} if extra == "a_consult_ran"
          else {"ego_cls": _HoldsANewOne})
    on, sup, _e = await _run([_sent()], [_row()], **kw)
    assert sup._evals == 1, f"{extra}: the judge must read this attempt"
    assert all("skipped" not in e for e in on.metadata[mk.JUDGE_ATTEMPTS])
    monkeypatch.setattr(pipeline_mod, "_prejudged_replay", lambda ctx: False)
    off, _s, _e = await _run([_sent()], [_row()], **kw)
    assert on.metadata[mk.JUDGE_ATTEMPTS] == off.metadata[mk.JUDGE_ATTEMPTS]
    assert on.stop_reason == off.stop_reason


# ── the stamp's one digest ──────────────────────────────────────────────────────────────────

def test_the_digest_is_of_the_bytes_as_they_are():
    assert prejudged_digest(MESSAGE) == _sha(MESSAGE)
    assert prejudged_digest(MESSAGE + " ") != prejudged_digest(MESSAGE)   # no strip
    assert prejudged_digest(None) is None and prejudged_digest(3) is None


# ── the composition, with the anima's own EgoStage ─────────────────────────────────────────

class _Notifier:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def tools_schema(self) -> list[dict]:
        return [{"type": "function", "function": {
            "name": "notify_user", "description": "send a message to a colleague",
            "parameters": {"type": "object", "properties": {"target": {"type": "string"},
                                                            "message": {"type": "string"}}}}}]

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        self.sent.append({"tool": name, "arguments": dict(arguments)})
        return ToolResult(output="Message delivered.", ok=True, side_effect=True)


class _DraftsOnly:
    """The executor after the replay: it reads «[ALREADY EXECUTED]» and drafts, no tool."""

    model = "stub-ego"

    async def generate(self, system: str, prompt: str):
        return "Avisei a Marisol.", 3, 3


async def test_the_REAL_replay_sends_the_row_without_the_stamp_and_skips_the_judge():
    """The stamp rides in the confirmed row, and the anima's replay reads `tool` and `arguments`
    only: the tool receives the arguments the user saw, never the stamp — and the skip holds."""
    notifier = _Notifier()
    sup = FakeSuperego(approve=False, critique=CRITIQUE, voice=VOICE)
    pipe = Pipeline(embedder=None, noumeno=FakeNoumeno(), ner=FakeNER(intent_class="ACTION_REQUEST"),
                    id_stage=FakeID(route="EGO"), ego=EgoStage(), superego=sup)
    ctx = PipelineContext(user_input="sim")
    ctx.metadata[mk.HELD_DELIVERED_TEXT] = dict(DECLARED)
    ctx.metadata[mk.EGO_CONFIRMED] = ["notify_user"]
    ctx.metadata[mk.EGO_CONFIRMED_CALLS] = [_row()]
    backend = _DraftsOnly()
    ctx = await pipe.run_turn(ctx, TurnConfig(gen_backend=backend, ego_backend=backend,
                                              ego_prompt="exec", max_corrections=1),
                              dispatcher=notifier)
    assert notifier.sent == [{"tool": "notify_user", "arguments": ARGS}]
    assert sup._evals == 0
    assert ctx.metadata[mk.JUDGE_ATTEMPTS][0]["skipped"] == JUDGE_SKIP_PREJUDGED_REPLAY
    assert ctx.stop_reason == "completed" and ctx.superego_result.response == VOICE
