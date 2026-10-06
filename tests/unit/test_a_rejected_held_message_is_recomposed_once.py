"""A held MESSAGE the judge rejected buys ONE recomposition by the EGO, with the critique.

The shape, measured live on a downstream host (the shape only; the names here are invented):

    the contact asked for a held message to be recomposed
    the EGO composed a NEW message and the confirmation gate held it, as it must
    the judge read the held text (`_HELD_MESSAGE_RULE`, criterion (a)) and REJECTED it, rightly:
      "the message must mention today's class"
    budget 1 → no pass left in which to obey the critique, so nothing was proposed and the
    contact read the host's neutral "the review did not approve the send"

The critique says what to write and nobody used it. Re-voicing cannot answer it: the voice does
not write the held text, the EGO does (it is a tool ARGUMENT). So the rejection buys one more EGO
pass, under the same ceiling as its two siblings (`test_one_turn_to_act.py`,
`test_one_turn_to_read.py`).

THE TWIN (the live shape): pass 1 rejected with the critique, pass 2 approved → the turn leaves
with a proposal. The EGO ran twice, the critique reached pass 2, the ledger entry that bought the
pass says `held_message_rejected`, and the rejected pass's cost is in `retry_metrics`.

FAB: the message says something else and the critique does not save it → pass 2 is rejected
too → the same outcome as today, and no third pass.

THE CONTROLS: approved at once → zero extra passes and a byte-identical context; a rejected
turn WITHOUT a held message does not take this path; a turn that COMMITTED gets nothing; a budget
that already retries gets no label.

GATES B/C INTACT: with the anima's REAL `EgoStage` and a policy dispatcher, the recomposed
message is HELD again and nothing is executed.

MUTATIONS (each named in the PR with what died):

  * no extra pass (the predicate always False)     -> the twin and the real-executor test die;
  * an unbounded pass (asked before the ceiling)   -> FAB dies (the double stops the loop);
  * the critique not handed to the EGO             -> the twin and the real-executor test die;
  * the pass granted without a held message        -> the no-held-message control dies.
"""

from typing import Optional

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.ego import EgoStage
from cogno_anima.types import (EgoResult, EgoStep, PipelineContext, SuperegoResult, ToolExecution,
                               ToolResult, held_delivered_texts)

from cogno_soma import (EXTRA_PASS_ACTION, EXTRA_PASS_HELD_MESSAGE, EXTRA_PASS_READ,
                        EXTRA_PASSES, Pipeline, STOP_JUDGE_EXHAUSTED, TurnConfig)
from cogno_soma import pipeline as soma_pipeline

from tests.conftest import FakeEgo, FakeID, FakeNER, FakeNoumeno, FakeSuperego, metrics

PLAN_LIMITS_ONE_ATTEMPT = {"max_self_corrections": 1}
NOTIFY = "notify_user"
DECLARED = {mk.HELD_DELIVERED_TEXT: {NOTIFY: "message"}}

# What the judge demands, and the critique that names it (invented, the live shape).
MUST_SAY = "aula de hoje"
CRITIQUE = "a mensagem deve mencionar a aula de hoje, que é o que o contacto pediu para avisar"
NEUTRAL = "Olá Rui, tudo bem? Queria falar consigo sobre a turma."
RIGHT = "Olá Rui, a aula de hoje passou para as 19h, na sala 4."
OFF_TOPIC = "Olá Rui, lembrete: a fatura de outubro já está disponível."
# A loop that never ends is the failure an "unbounded pass" mutation produces; the double stops it
# loudly instead of hanging the suite.
_RUNAWAY = 6


class _ComposingEgo(FakeEgo):
    """Holds a `notify_user` on every pass, and composes its text from what it was HANDED.

    ``obeys``: when the correction it is handed carries the critique, the recomposed message says
    what the critique asks (the live shape). ``obeys=False`` is FAB: it never does. Records the
    correction each pass saw, so "the critique reached pass 2" is read off the executor, not off
    the metadata after the turn."""

    def __init__(self, *, obeys: bool = True, first: str = NEUTRAL,
                 commit_first: bool = False) -> None:
        super().__init__()
        self._obeys = obeys
        self._first = first
        self._commit_first = commit_first
        self.seen_corrections: list[Optional[dict]] = []

    async def process(self, ctx, backend, dispatcher, *, system_prompt):
        ctx = await super().process(ctx, backend, dispatcher, system_prompt=system_prompt)
        if self.invocations > _RUNAWAY:
            raise AssertionError("the EGO ran past any ceiling — an unbounded extra pass")
        correction = ctx.metadata.get(mk.EGO_CORRECTION)
        self.seen_corrections.append(dict(correction) if isinstance(correction, dict) else None)
        told = isinstance(correction, dict) and CRITIQUE in str(correction.get("reason") or "")
        text = RIGHT if (told and self._obeys) else (self._first if self.invocations == 1
                                                     else OFF_TOPIC)
        held = ToolExecution(tool=NOTIFY, ok=False, error="needs_confirmation",
                             arguments={"target": "Rui", "message": text}, result="",
                             tool_mutating=True)
        calls = [held]
        if self._commit_first and self.invocations == 1:
            calls = [ToolExecution(tool="add_note", ok=True, side_effect=True,
                                   tool_mutating=True), held]
        ctx.ego_result = EgoResult(steps=[EgoStep(index=0, path="native", tool_calls=calls)],
                                   pending_confirmation=[held], metrics=metrics("ego"),
                                   tools_offered=[NOTIFY])
        return ctx


class _MessageJudge(FakeSuperego):
    """Criterion (a) of the held-message rule, as a double: approve the held text iff it says
    what was asked. A turn with no held message is judged by the plain programmable verdict."""

    def __init__(self, *, approve_plain: bool = False) -> None:
        super().__init__(approve=approve_plain, critique=CRITIQUE)
        self.read_texts: list[list[str]] = []

    async def evaluate(self, ctx, backend, *, limits_prompt):
        texts = [t for _tool, t in held_delivered_texts(ctx)]
        if not texts:
            return await super().evaluate(ctx, backend, limits_prompt=limits_prompt)
        self._evals += 1
        self.read_texts.append(texts)
        approved = all(MUST_SAY in t for t in texts)
        return SuperegoResult(response="", approved=approved,
                              critique=None if approved else CRITIQUE,
                              metrics=metrics("superego_judge"))


async def _run(ego, judge, *, backend, dispatcher, metadata=DECLARED,
               plan=PLAN_LIMITS_ONE_ATTEMPT, intent="ACTION_REQUEST"):
    ctx = PipelineContext(user_input="refaz a mensagem para o Rui, fala da aula de hoje")
    if plan is not None:
        ctx.metadata["plan_limits"] = dict(plan)
    ctx.metadata.update(metadata or {})
    pipe = Pipeline(embedder=None, noumeno=FakeNoumeno(), ner=FakeNER(intent_class=intent),
                    id_stage=FakeID(route="EGO"), ego=ego, superego=judge)
    return await pipe.run_turn(
        ctx, TurnConfig(gen_backend=backend, ego_backend=backend, ego_prompt="exec",
                        max_corrections=3),          # the plan cap must beat this
        dispatcher=dispatcher)


# ── the twin ────────────────────────────────────────────────────────────────────────────────
async def test_a_rejected_held_message_is_recomposed_ONCE_with_the_critique(stub_backend,
                                                                           dispatcher):
    """THE PROPERTY, asserted as a group: two passes alone are also true of a budget-2 turn; the
    critique alone is also true of a reason built and never used; the label alone of a stamp on
    a pass that never ran."""
    ego, judge = _ComposingEgo(), _MessageJudge()
    ctx = await _run(ego, judge, backend=stub_backend, dispatcher=dispatcher)
    assert ego.invocations == 2 and judge._evals == 2
    # the critique reached the SECOND pass, and only the second
    assert ego.seen_corrections[0] is None
    assert ego.seen_corrections[1] == {"reason": CRITIQUE, "attempt": 2}
    assert judge.read_texts == [[NEUTRAL], [RIGHT]]
    # the turn leaves with a PROPOSAL, approved, and nothing executed
    assert ctx.metadata[mk.JUDGE_VERDICT] == {"approved": True, "attempts": 2}
    assert [c.arguments["message"] for c in ctx.ego_result.pending_confirmation] == [RIGHT]
    assert ctx.stop_reason != STOP_JUDGE_EXHAUSTED
    assert mk.VOICE_CORRECTION not in ctx.metadata
    assert dispatcher.executed == []
    # countable: the rejected entry that bought the pass says which exception did
    ledger = ctx.metadata[mk.JUDGE_ATTEMPTS]
    assert [e.get("extra_pass") for e in ledger] == [EXTRA_PASS_HELD_MESSAGE, None]
    # and it is PAID for on the ledger: the rejected EGO pass + both judge reads are retries
    stages = [m.stage for m in ctx.retry_metrics]
    assert stages.count("ego") == 1 and stages.count("superego_judge") == 2
    assert ctx.ego_metrics is not None and ctx.ego_metrics.attempt == 2


async def test_FAB_a_recomposition_the_critique_does_not_save_ends_as_today(stub_backend,
                                                                            dispatcher,
                                                                            monkeypatch):
    """Pass 2 says something else and is rejected again → no third pass, and the OUTCOME the
    contact reads is today's: the same stop reason, the same voice correction, the same reply.
    'Today' is the same turn with the predicate forced off, run beside it."""
    ego, judge = _ComposingEgo(obeys=False), _MessageJudge()
    branch = await _run(ego, judge, backend=stub_backend, dispatcher=dispatcher)
    assert ego.invocations == 2 and judge._evals == 2
    assert judge.read_texts == [[NEUTRAL], [OFF_TOPIC]]
    assert [e.get("extra_pass") for e in branch.metadata[mk.JUDGE_ATTEMPTS]] == \
        [EXTRA_PASS_HELD_MESSAGE, None]
    assert branch.metadata[mk.JUDGE_VERDICT] == {"approved": False, "attempts": 2}

    monkeypatch.setattr(soma_pipeline, "_owes_a_held_rewrite", lambda ctx: False)
    ego0, judge0 = _ComposingEgo(obeys=False), _MessageJudge()
    today = await _run(ego0, judge0, backend=stub_backend, dispatcher=dispatcher)
    assert ego0.invocations == 1                       # the pair: today is ONE pass

    assert branch.stop_reason == today.stop_reason == STOP_JUDGE_EXHAUSTED
    assert branch.metadata[mk.VOICE_CORRECTION] == today.metadata[mk.VOICE_CORRECTION]
    assert branch.superego_result.response == today.superego_result.response
    assert dispatcher.executed == []


# ── the controls ────────────────────────────────────────────────────────────────────────────
async def test_control_approved_at_once_is_byte_identical(stub_backend, dispatcher,
                                                          monkeypatch):
    """Approved on pass 1 → zero extra passes, and the whole context is the one the predicate
    forced off produces, byte for byte (the feature is inert where it does not fire)."""
    ego, judge = _ComposingEgo(first=RIGHT), _MessageJudge()
    on = await _run(ego, judge, backend=stub_backend, dispatcher=dispatcher)
    assert ego.invocations == 1 and judge.read_texts == [[RIGHT]]
    assert all("extra_pass" not in e for e in on.metadata[mk.JUDGE_ATTEMPTS])

    monkeypatch.setattr(soma_pipeline, "_owes_a_held_rewrite", lambda ctx: False)
    ego0 = _ComposingEgo(first=RIGHT)
    off = await _run(ego0, _MessageJudge(), backend=stub_backend, dispatcher=dispatcher)
    assert on.model_dump_json() == off.model_dump_json()


async def test_control_a_rejection_WITHOUT_a_held_message_does_not_take_this_path(
        stub_backend, dispatcher):
    """The same declaration, the same rejection, a turn that only READ: no held message, so no
    recomposition (its siblings' predicates are false here too: no write on the table, no
    declared source read)."""
    ego = FakeEgo(tool_calls=[[ToolExecution(tool="get_schedule", ok=True, side_effect=False,
                                             tool_mutating=False, result="19h")]],
                  tools_offered=[["get_schedule"]], drafts=["A aula é às 19h."])
    judge = _MessageJudge(approve_plain=False)
    ctx = await _run(ego, judge, backend=stub_backend, dispatcher=dispatcher)
    assert judge._evals == 1, "the turn was judged (the rejection exists)"
    assert ego.invocations == 1
    assert all("extra_pass" not in e for e in ctx.metadata[mk.JUDGE_ATTEMPTS])
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED


async def test_control_an_UNDECLARED_hold_is_not_judged_and_not_recomposed(stub_backend,
                                                                           dispatcher):
    """No `HELD_DELIVERED_TEXT` → the proposal turn is not judged at all (today's skip), so
    there is no rejection to answer."""
    ego, judge = _ComposingEgo(), _MessageJudge()
    ctx = await _run(ego, judge, backend=stub_backend, dispatcher=dispatcher, metadata={})
    assert ego.invocations == 1 and judge._evals == 0
    assert mk.JUDGE_ATTEMPTS not in ctx.metadata


async def test_control_a_turn_that_COMMITTED_gets_nothing(stub_backend, dispatcher):
    """Pass 1 wrote something, then held the message. Re-running would write a second time,
    so the siblings' guard holds here too: one pass, and the rejected commit goes the way it
    always went."""
    ego, judge = _ComposingEgo(commit_first=True), _MessageJudge()
    ctx = await _run(ego, judge, backend=stub_backend, dispatcher=dispatcher)
    assert ego.invocations == 1
    assert all("extra_pass" not in e for e in ctx.metadata[mk.JUDGE_ATTEMPTS])


async def test_control_a_budget_that_already_retries_gets_no_label(stub_backend, dispatcher):
    """Budget 2: the ordinary retry recomposes (as before this change), and the ceiling makes the
    exception inert — two passes, no `extra_pass` anywhere."""
    ego, judge = _ComposingEgo(obeys=False), _MessageJudge()
    ctx = await _run(ego, judge, backend=stub_backend, dispatcher=dispatcher,
                     plan={"max_self_corrections": 2})
    assert ego.invocations == 2
    assert all("extra_pass" not in e for e in ctx.metadata[mk.JUDGE_ATTEMPTS])


async def test_the_three_exceptions_SHARE_one_ceiling(stub_backend, dispatcher, monkeypatch):
    """All three predicates forced TRUE: still ONE extra pass. The held message is asked FIRST
    (the trace shows the judge read a held text — the critique is about that text, and the
    read sibling's sentence must not ride on it)."""
    monkeypatch.setattr(soma_pipeline, "_owes_an_action", lambda ctx, dispatcher: True)
    monkeypatch.setattr(soma_pipeline, "_owes_a_read", lambda ctx: ["consult_documents"])
    ego, judge = _ComposingEgo(obeys=False), _MessageJudge()
    ctx = await _run(ego, judge, backend=stub_backend, dispatcher=dispatcher)
    assert ego.invocations == 2
    assert [e.get("extra_pass") for e in ctx.metadata[mk.JUDGE_ATTEMPTS]] == \
        [EXTRA_PASS_HELD_MESSAGE, None]
    assert ego.seen_corrections[1] == {"reason": CRITIQUE, "attempt": 2}


def test_the_extra_pass_alphabet_is_CLOSED_and_exported():
    assert EXTRA_PASS_HELD_MESSAGE == "held_message_rejected"
    assert EXTRA_PASSES == {EXTRA_PASS_ACTION, EXTRA_PASS_READ, EXTRA_PASS_HELD_MESSAGE}


# ── gates B/C intact: the anima's REAL executor ─────────────────────────────────────────────
class _ScriptedEgoBackend:
    """A text-only backend for the REAL anima EGO (the `<TOOL_CALL>` fallback path).

    Every pass calls `notify_user`; the message carries what the critique asks only when the
    prompt it was handed carries the critique, so the recomposition is CAUSED by the critique."""

    model = "stub-ego"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, system: str, prompt: str):
        handed = f"{system}\n{prompt}"
        self.prompts.append(handed)
        text = RIGHT if CRITIQUE in handed else NEUTRAL
        return ('<TOOL_CALL>{"tool": "notify_user", "args": {"target": "Rui", '
                f'"message": "{text}"}}}}</TOOL_CALL>'), 5, 5


class _NotifyPolicyDispatcher:
    """`notify_user` is a WRITE that requires confirmation (gate B). Records every execute."""

    def __init__(self) -> None:
        self.executed: list[str] = []

    def tools_schema(self) -> list[dict]:
        return [{"type": "function", "function": {
            "name": NOTIFY, "description": "send a message to a person",
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string"}, "message": {"type": "string"}}}}}]

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        self.executed.append(name)
        return ToolResult(output="sent", ok=True, side_effect=True)

    def is_mutating(self, name: str) -> bool:
        return name == NOTIFY

    def requires_confirmation(self, name: str) -> bool:
        return name == NOTIFY


@pytest.mark.parametrize("plan", [PLAN_LIMITS_ONE_ATTEMPT])
async def test_the_REAL_executor_recomposes_and_the_gate_HOLDS_it_again(plan):
    backend = _ScriptedEgoBackend()
    disp = _NotifyPolicyDispatcher()
    judge = _MessageJudge()
    ctx = await _run(EgoStage(), judge, backend=backend, dispatcher=disp, plan=plan)
    assert disp.executed == [], "a recomposition executed a send nobody confirmed"
    assert judge.read_texts == [[NEUTRAL], [RIGHT]]
    assert [c.arguments["message"] for c in ctx.ego_result.pending_confirmation] == [RIGHT]
    assert ctx.metadata[mk.JUDGE_VERDICT] == {"approved": True, "attempts": 2}
    assert "# Correction requested" not in backend.prompts[0]
    second = next(p for p in backend.prompts if "# Correction requested" in p)
    assert CRITIQUE in second
