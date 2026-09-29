"""The sibling exception: "there is no X" over a source nobody read buys ONE executor pass.

`test_one_turn_to_act.py` pins the exception for a critique of *"you did not act"*. This file
pins its sibling on the READ side, and the ways it must NOT fire.

The defect, measured on a rehearsal tenant (the shape only, never its data):

    an INFORMATION_REQUEST to a bookkeeping persona, the document-reading tool on the table
    pass 1 called only the ledger summary, which came back empty
    the draft asserted ABSENCE ("Não há entradas no livro… R$ 0,00 em aluguéis")
    the judge REJECTED it, rightly, and its critique proposed repeating the negative
    budget 1 → no pass left, and the voice shipped the same negative, which was FALSE: the
    documents held the four values

Re-voicing rewrites words over the same empty evidence. The missing thing is a tool call, and
only the EGO calls tools.

THE TWIN: that shape re-runs the executor, and the second pass receives the critique PLUS the
closed sentence that names `consult_documents`.

THE CONTROLS (none of them fires):

  (a) the negative with the read MADE and empty ("nada relevante") passes as today;
  (b) the four production matches: no document read on the table;
  (c) a persona with no tools at all;
  (d) an APPROVED turn never enters;
  (e) on an ACTION_REQUEST `_owes_an_action` keeps its precedence and the read sentence never
      rides; an ACTION_REQUEST without a write on the table gets no pass from this sibling;
  (f) no declaration, or a garbled one, is OFF.

Plus the conditions and guards the controls do not isolate: a draft that asserts nothing
absent, a turn that COMMITTED, the ceiling, and a budget of two.

MUTATIONS (each named in the PR with what died):

  * drop condition 2 (`owed = ["consult_documents"]`: any rejection with a negative)
        -> (a) and (b) die;
  * drop condition 1 (the intent clause)   -> the ACTION_REQUEST-without-a-write control dies;
  * drop condition 3 (`_asserts_absence` always True) -> the no-negative control dies;
  * drop `committed_this_turn`             -> the committed control dies;
  * drop the sentence from the reason      -> the twin dies;
  * give the sentence to every rejected retry -> the budget-of-two control dies;
  * a DEFAULT declaration when the key is absent -> the switch twin and (f) absent die;
  * give the read exception a ceiling of its own -> the shared-ceiling test dies;
  * ask the read exception FIRST               -> the shared-ceiling test dies (the ledger
                                                  field is what makes the order observable);
  * drop either `extra_pass` stamp             -> the twin / (e) and the shared-ceiling test die.
"""

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.ego import EgoStage
from cogno_anima.types import PipelineContext, ToolExecution, ToolResult

from cogno_soma import (EXTRA_PASS_ACTION, EXTRA_PASS_READ, EXTRA_PASSES, Pipeline,
                        STOP_JUDGE_EXHAUSTED, TurnConfig)
from cogno_soma import pipeline as soma_pipeline
from cogno_soma.pipeline import _READ_OWED_NOTE, _asserts_absence

from tests.conftest import FakeEgo, FakeID, FakeNER, FakeNoumeno, FakeSuperego

PLAN_LIMITS_ONE_ATTEMPT = {"max_self_corrections": 1}

DOCS = "consult_documents"
SUMMARY = "get_summary"
OFFERED = [SUMMARY, DOCS, "resolve_date"]
DECLARED = {mk.SOURCE_READS: [DOCS]}

NEGATIVE = "Não há entradas no livro financeiro para a Vila Azul: R$ 0,00 em aluguéis."
ANSWER = "Os documentos registram 4 aluguéis da Vila Azul."
# The measured critique proposes repeating the negative, which is why the sentence is needed.
CRITIQUE = "a resposta deveria dizer que não há dados específicos da Vila Azul disponíveis"

EMPTY_SUMMARY = ToolExecution(tool=SUMMARY, result="Income R$0,00 (0 entries)", ok=True,
                              side_effect=False, tool_mutating=False)
READ_DOCS = ToolExecution(tool=DOCS, arguments={"query": "aluguel Vila Azul"},
                          result="4 aluguéis: R$ 900,00 …", ok=True, side_effect=False,
                          tool_mutating=False)
DOCS_EMPTY = ToolExecution(tool=DOCS, arguments={"query": "aluguel Vila Azul"},
                           result="nada relevante", ok=True, side_effect=False,
                           tool_mutating=False)
WROTE = ToolExecution(tool="add_outcome", ok=True, side_effect=True, tool_mutating=True)


def _pipeline(embedder, ego, *, intent="INFORMATION_REQUEST", approve=False, approve_after=None):
    return Pipeline(embedder=embedder, noumeno=FakeNoumeno(), ner=FakeNER(intent_class=intent),
                    id_stage=FakeID(route="EGO"), ego=ego,
                    superego=FakeSuperego(approve=approve, approve_after=approve_after,
                                          critique=CRITIQUE))


async def _run(embedder, backend, ego, *, intent="INFORMATION_REQUEST", metadata=DECLARED,
               approve=False, approve_after=None, plan=PLAN_LIMITS_ONE_ATTEMPT, dispatcher=None):
    ctx = PipelineContext(user_input="quanto recebi de aluguel da Vila Azul?")
    ctx.metadata["plan_limits"] = dict(plan)
    ctx.metadata.update(metadata or {})
    pipe = _pipeline(embedder, ego, intent=intent, approve=approve, approve_after=approve_after)
    return await pipe.run_turn(
        ctx, TurnConfig(gen_backend=backend, ego_backend=backend, ego_prompt="exec",
                        max_corrections=3),        # the plan cap must beat this
        dispatcher=dispatcher)


def _note(*tools: str) -> str:
    return _READ_OWED_NOTE.format(tools=" or ".join(f"`{t}`" for t in tools),
                                  verb="it was" if len(tools) == 1 else "they were")


# ── the twin ────────────────────────────────────────────────────────────────────────────────
async def test_a_negative_over_an_UNREAD_source_buys_one_more_pass(stub_embedder, stub_backend,
                                                                  dispatcher):
    """THE PROPERTY. Budget 1, the source offered and not called, the draft says "there is
    none", the judge rejected: the executor runs again, and it is told which tool to call.

    Asserted as a group. `invocations == 2` alone is also true of a plain budget-2 turn, and
    the reason alone is also true of a sentence built and never used."""
    ego = FakeEgo(tool_calls=[[EMPTY_SUMMARY], [READ_DOCS]], tools_offered=[OFFERED],
                  drafts=[NEGATIVE, ANSWER])
    ctx = await _run(stub_embedder, stub_backend, ego, approve_after=2, dispatcher=dispatcher)
    assert ego.invocations == 2                               # the exception fired
    assert [t.tool for t in ctx.turn_executions] == [SUMMARY, DOCS]
    assert ctx.metadata[mk.JUDGE_VERDICT] == {"approved": True, "attempts": 2}
    reason = ctx.metadata[mk.EGO_CORRECTION]["reason"]
    assert reason == f"{CRITIQUE}\n\n{_note(DOCS)}"
    assert "`consult_documents`" in reason and "not called" in reason
    # …and it is COUNTABLE: the rejected entry that bought the pass says which exception did.
    ledger = ctx.metadata[mk.JUDGE_ATTEMPTS]
    assert ledger[0]["extra_pass"] == EXTRA_PASS_READ
    assert "extra_pass" not in ledger[1], "the approved attempt bought nothing"


async def test_the_same_turn_with_the_source_READ_is_one_pass(stub_embedder, stub_backend,
                                                             dispatcher):
    """The twin's pair: the source READ on pass 1, the world where the executor did its job.
    Same negative, same rejection, one pass. Without this half the twin could be a constant: a
    loop that always grants a second pass passes the test above."""
    ego = FakeEgo(tool_calls=[[EMPTY_SUMMARY, READ_DOCS]], tools_offered=[OFFERED],
                  drafts=[NEGATIVE])
    ctx = await _run(stub_embedder, stub_backend, ego, dispatcher=dispatcher)
    assert ego.invocations == 1
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED


# ── the controls ────────────────────────────────────────────────────────────────────────────
async def test_a_the_read_was_MADE_and_found_nothing(stub_embedder, stub_backend, dispatcher):
    """(a) "nada relevante" out of the documents is a TRUE negative. It voices, as today."""
    ego = FakeEgo(tool_calls=[[DOCS_EMPTY]], tools_offered=[OFFERED], drafts=[NEGATIVE])
    ctx = await _run(stub_embedder, stub_backend, ego, dispatcher=dispatcher)
    assert ego.invocations == 1
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED
    assert ctx.needs_handoff is False
    assert ctx.superego_result.response == "final reply"
    assert "extra_pass" not in ctx.metadata[mk.JUDGE_ATTEMPTS][0]


@pytest.mark.parametrize("offered", [
    [SUMMARY, "knowledge_search", "search_history"],   # an empty ledger before any document
    ["consult_material", "resolve_date"],               # a pay estimate, another knowledge read
], ids=["ledger_without_documents", "another_knowledge_read"])
async def test_b_the_production_matches_have_no_document_read_on_the_table(
        stub_embedder, stub_backend, dispatcher, offered):
    """(b) The four production turns the negative matched. None of them had the declared
    source on the table, so none of them is owed a read."""
    ego = FakeEgo(tool_calls=[[EMPTY_SUMMARY]], tools_offered=[offered], drafts=[NEGATIVE])
    ctx = await _run(stub_embedder, stub_backend, ego, dispatcher=dispatcher)
    assert ego.invocations == 1
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED


async def test_c_a_persona_with_no_tools(stub_embedder, stub_backend, dispatcher):
    """(c) Nothing was offered, so nothing could have been read."""
    ego = FakeEgo(tool_calls=[[]], tools_offered=[[]], drafts=[NEGATIVE])
    ctx = await _run(stub_embedder, stub_backend, ego, dispatcher=dispatcher)
    assert ego.invocations == 1
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED


async def test_d_an_APPROVED_turn_never_enters(stub_embedder, stub_backend, dispatcher):
    """(d) The exception lives on the rejected branch. An approved negative is the judge's
    call, and it ships."""
    ego = FakeEgo(tool_calls=[[EMPTY_SUMMARY]], tools_offered=[OFFERED], drafts=[NEGATIVE])
    ctx = await _run(stub_embedder, stub_backend, ego, approve=True, dispatcher=dispatcher)
    assert ego.invocations == 1
    assert mk.EGO_CORRECTION not in ctx.metadata
    assert ctx.metadata[mk.JUDGE_VERDICT] == {"approved": True, "attempts": 1}


class _PolicyDispatcher:
    """Classifies `add_outcome` as mutating, like `test_one_turn_to_act.py`'s."""

    def tools_schema(self) -> list[dict]:
        return []

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        return ToolResult(output="", ok=True)

    def is_mutating(self, name: str) -> bool:
        return name == "add_outcome"

    def requires_confirmation(self, name: str) -> bool:
        return name == "add_outcome"


async def test_e_on_an_ACTION_the_action_exception_keeps_its_precedence(stub_embedder,
                                                                        stub_backend):
    """(e) An ACTION_REQUEST that ducked a write, with the read-owed shape around it. The pass
    that is granted is the ACTION one, and it carries the critique ALONE: the read sentence is
    for a turn whose debt is a read."""
    ego = FakeEgo(tool_calls=[[EMPTY_SUMMARY]], tools_offered=[OFFERED + ["add_outcome"]],
                  drafts=[NEGATIVE])
    ctx = await _run(stub_embedder, stub_backend, ego, intent="ACTION_REQUEST",
                     dispatcher=_PolicyDispatcher())
    assert ego.invocations == 2                               # `_owes_an_action` fired
    assert ctx.metadata[mk.EGO_CORRECTION]["reason"] == CRITIQUE
    assert ctx.metadata[mk.JUDGE_ATTEMPTS][0]["extra_pass"] == EXTRA_PASS_ACTION


async def test_e_an_ACTION_without_a_write_gets_nothing_from_the_read_sibling(
        stub_embedder, stub_backend, dispatcher):
    """(e) Condition 1 isolated: same table, same negative, same unread source; only the
    intent class differs from the twin."""
    ego = FakeEgo(tool_calls=[[EMPTY_SUMMARY]], tools_offered=[OFFERED], drafts=[NEGATIVE])
    ctx = await _run(stub_embedder, stub_backend, ego, intent="ACTION_REQUEST",
                     dispatcher=dispatcher)
    assert ego.invocations == 1
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED


@pytest.mark.parametrize("metadata", [
    {}, {mk.SOURCE_READS: []}, {mk.SOURCE_READS: {DOCS: True}}, {mk.SOURCE_READS: 7},
], ids=["absent", "empty", "mapping", "int"])
async def test_f_no_declaration_is_OFF(stub_embedder, stub_backend, dispatcher, metadata):
    """(f) A host reverts the feature by not stamping the key. A garbled key reads the same."""
    ego = FakeEgo(tool_calls=[[EMPTY_SUMMARY]], tools_offered=[OFFERED], drafts=[NEGATIVE])
    ctx = await _run(stub_embedder, stub_backend, ego, metadata=metadata, dispatcher=dispatcher)
    assert ego.invocations == 1
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED


# ── the conditions and guards the controls do not isolate ───────────────────────────────────
async def test_a_draft_that_asserts_NOTHING_absent_voices(stub_embedder, stub_backend,
                                                          dispatcher):
    """Condition 3 isolated: the twin's turn with a draft that claims no absence. The judge
    rejected it for some other reason, and that is a text rejection: it voices."""
    ego = FakeEgo(tool_calls=[[EMPTY_SUMMARY]], tools_offered=[OFFERED],
                  drafts=["O resumo do livro mostra R$ 0,00 de receita neste mês."])
    ctx = await _run(stub_embedder, stub_backend, ego, dispatcher=dispatcher)
    assert ego.invocations == 1
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED


async def test_a_turn_that_COMMITTED_gets_nothing(stub_embedder, stub_backend, dispatcher):
    """Re-running after a write commits a second time, whatever the intent class said. It
    takes the rejected-commit branch, as at budget 1 today."""
    ego = FakeEgo(tool_calls=[[EMPTY_SUMMARY, WROTE]], tools_offered=[OFFERED],
                  drafts=[NEGATIVE])
    ctx = await _run(stub_embedder, stub_backend, ego, dispatcher=dispatcher)
    assert ego.invocations == 1
    assert ctx.needs_handoff is True


async def test_the_exception_grants_ONE_pass_not_two(stub_embedder, stub_backend, dispatcher):
    """The second pass ducks exactly like the first, so every condition still holds, and the
    turn still stops at two: the sibling shares `_ACTION_RETRY_CEILING`."""
    ego = FakeEgo(tool_calls=[[EMPTY_SUMMARY]], tools_offered=[OFFERED], drafts=[NEGATIVE])
    ctx = await _run(stub_embedder, stub_backend, ego, dispatcher=dispatcher)
    assert ego.invocations == 2
    assert ctx.metadata[mk.JUDGE_VERDICT]["attempts"] == 2
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED


async def test_a_budget_of_two_keeps_the_critique_alone(stub_embedder, stub_backend,
                                                        dispatcher):
    """A budget that already retries is left byte for byte: its retry carries the critique,
    the sentence rides only on the GRANTED extra pass, and no third pass is added."""
    ego = FakeEgo(tool_calls=[[EMPTY_SUMMARY]], tools_offered=[OFFERED], drafts=[NEGATIVE])
    ctx = await _run(stub_embedder, stub_backend, ego, plan={"max_self_corrections": 2},
                     dispatcher=dispatcher)
    assert ego.invocations == 2
    assert ctx.metadata[mk.EGO_CORRECTION]["reason"] == CRITIQUE


# ── the closed absence list, with its own twin ──────────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "Não há entradas no livro.", "não existe registro disso", "Não existem registros.",
    "Não encontrei esse valor.", "Não consta no sistema.", "Não temos essa informação.",
    "O relatório não está disponível.", "Nenhum aluguel registrado.", "Nenhuma entrada.",
    "Sem registros para o período.", "sem dados", "Sem informações sobre isso.",
    "There is no record of it.", "There are no entries.", "No records for that month.",
    "no data", "The file was not found.", "That figure is not available.",
])
def test_the_absence_list_matches_every_listed_form(text):
    assert _asserts_absence(text) is True


@pytest.mark.parametrize("text", [
    ANSWER, "Há 4 registros de aluguel.", "Encontrei os valores nos documentos.",
    "O total é R$ 900,00.", "There are four entries.", "Records found: 4.",
    "Não havia pendências?",   # a question about the past, no listed form
    "", None,
])
def test_the_absence_list_does_not_match_an_answer(text):
    assert _asserts_absence(text) is False



# ── the real executor reads the sentence ────────────────────────────────────────────────────
class _ScriptedEgoBackend:
    """A text-only backend for the REAL anima EGO (the `<TOOL_CALL>` fallback path).

    Pass 1 looks at the empty summary and drafts the negative. Pass 2 reads the documents only
    if the prompt it was handed names the tool, so the second read is caused by the sentence and
    not by the script."""

    model = "stub-ego"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, system: str, prompt: str):
        # The EGO renders `# Correction requested` in its SYSTEM message and the tool results in
        # the prompt, so the backend reads what it was handed as a whole.
        handed = f"{system}\n{prompt}"
        self.prompts.append(handed)
        told = _note(DOCS) in handed
        has_result = "4 aluguéis" in prompt or "Income R$0,00" in prompt
        if not has_result:
            tool = DOCS if told else SUMMARY
            return f'<TOOL_CALL>{{"tool": "{tool}", "args": {{}}}}</TOOL_CALL>', 5, 5
        return (ANSWER if told else NEGATIVE), 5, 5


class _SourceDispatcher:
    """The two reads, for real: the summary is empty, the documents hold the values."""

    def __init__(self) -> None:
        self.executed: list[str] = []

    def tools_schema(self) -> list[dict]:
        return [{"type": "function", "function": {"name": n, "description": n,
                                                  "parameters": {"type": "object",
                                                                 "properties": {}}}}
                for n in (SUMMARY, DOCS)]

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        self.executed.append(name)
        if name == DOCS:
            return ToolResult(output="4 aluguéis: R$ 900,00 …", ok=True)
        return ToolResult(output="Income R$0,00 (0 entries)", ok=True)


async def test_the_REAL_executor_is_handed_the_sentence_and_reads(stub_embedder):
    """The composition, with the anima's own EgoStage. The sentence is rendered into what the
    second pass is handed, under `# Correction requested` in the EGO's system message; the
    executor calls the documents because of it; and the pair is the first pass, which carries
    no correction at all."""
    backend = _ScriptedEgoBackend()
    disp = _SourceDispatcher()
    pipe = Pipeline(embedder=stub_embedder, noumeno=FakeNoumeno(),
                    ner=FakeNER(intent_class="INFORMATION_REQUEST"),
                    id_stage=FakeID(route="EGO"), ego=EgoStage(),
                    superego=FakeSuperego(approve=False, approve_after=2, critique=CRITIQUE))
    ctx = PipelineContext(user_input="quanto recebi de aluguel da Vila Azul?")
    ctx.metadata["plan_limits"] = dict(PLAN_LIMITS_ONE_ATTEMPT)
    ctx.metadata.update(DECLARED)
    ctx = await pipe.run_turn(ctx, TurnConfig(gen_backend=backend, ego_backend=backend,
                                              ego_prompt="exec", max_corrections=3),
                              dispatcher=disp)
    assert disp.executed == [SUMMARY, DOCS]
    assert ctx.metadata[mk.JUDGE_VERDICT] == {"approved": True, "attempts": 2}
    assert "# Correction requested" not in backend.prompts[0]
    second_pass = next(p for p in backend.prompts if "# Correction requested" in p)
    assert _note(DOCS) in second_pass
    assert CRITIQUE in second_pass


# ── the three landing conditions, each with its twin ────────────────────────────────────────
async def test_the_switch_is_the_DECLARATION(stub_embedder, stub_backend, dispatcher):
    """ONE turn, both worlds: declared → one extra pass; not declared → none at all.

    This is the host's reversal (`COGNO_READ_OWED_RETRY=0` stops the stamp), so it is pinned as
    a pair rather than inferred from the controls: a mutant that falls back to a DEFAULT
    declaration when the key is absent passes every control that declares something."""
    ego = FakeEgo(tool_calls=[[EMPTY_SUMMARY]], tools_offered=[OFFERED], drafts=[NEGATIVE])
    on = await _run(stub_embedder, stub_backend, ego, metadata=DECLARED, dispatcher=dispatcher)
    assert ego.invocations == 2
    assert on.metadata[mk.JUDGE_ATTEMPTS][0]["extra_pass"] == EXTRA_PASS_READ

    ego = FakeEgo(tool_calls=[[EMPTY_SUMMARY]], tools_offered=[OFFERED], drafts=[NEGATIVE])
    off = await _run(stub_embedder, stub_backend, ego, metadata={}, dispatcher=dispatcher)
    assert ego.invocations == 1
    assert mk.SOURCE_READS not in off.metadata
    assert all("extra_pass" not in e for e in off.metadata[mk.JUDGE_ATTEMPTS])


async def test_the_two_exceptions_SHARE_one_ceiling(stub_embedder, stub_backend, dispatcher,
                                                    monkeypatch):
    """Both predicates TRUE on every rejection, and the turn still gets ONE extra pass.

    The two cannot be true together through a real turn: their intent classes are disjoint.
    So both are forced, which is the only way to ask the question the ceiling answers: a turn
    never collects the action pass AND the read pass. The pass that is granted is the action
    one, because it is asked first; the second rejection meets the ceiling and buys nothing."""
    monkeypatch.setattr(soma_pipeline, "_owes_an_action", lambda ctx, dispatcher: True)
    monkeypatch.setattr(soma_pipeline, "_owes_a_read", lambda ctx: [DOCS])
    ego = FakeEgo(tool_calls=[[EMPTY_SUMMARY]], tools_offered=[OFFERED], drafts=[NEGATIVE])
    ctx = await _run(stub_embedder, stub_backend, ego, dispatcher=dispatcher)
    assert ego.invocations == 2
    ledger = ctx.metadata[mk.JUDGE_ATTEMPTS]
    assert [e.get("extra_pass") for e in ledger] == [EXTRA_PASS_ACTION, None]
    assert ctx.stop_reason == STOP_JUDGE_EXHAUSTED


def test_the_extra_pass_alphabet_is_CLOSED_and_exported():
    """The host persists this ledger and closes the field from these constants."""
    assert EXTRA_PASSES == {EXTRA_PASS_ACTION, EXTRA_PASS_READ} == {"action_owed", "read_owed"}
