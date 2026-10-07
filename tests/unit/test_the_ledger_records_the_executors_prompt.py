"""The per-attempt ledger records what each attempt's EXECUTOR PROMPT carried.

``EgoStage.process`` (cogno-anima 0.1.5) records its own prompt on the ``EgoResult``: which
parts it carried and how long each was (``prompt_blocks``), a digest of what the attempt started
from (``prompt_sha``) and the path the catalogue took (``prompt_path``). The correction loop
REPLACES ``ctx.ego_result`` on every retry, so only the surviving attempt's record reached a
reader — and a retry's prompt is not the first attempt's: it gains ``correction``, and
``actions_done`` when the rejected attempt wrote.

The ledger now carries it, per attempt, on BOTH places a row is written (a judged attempt and a
pre-judged replay). Only slugs, lengths, a digest and a path; never a byte of the prompt.

Pinned here, each with the check that would catch its opposite:
- two attempts leave two DIFFERENT records, and the first is not the survivor's (the mutation
  that backfills every row from the last attempt fails);
- one twin per path (text and native);
- a stand-in executor, or a result with nothing on record, leaves the keys ABSENT — never ``[]``;
- a value from outside anima's closed alphabets is dropped and counted, never written;
- no text of the prompt reaches the row;
- the hole this does NOT close: a hold with no delivered text has no ledger row at all.
"""

from __future__ import annotations

import json

import cogno_anima
import cogno_anima.metakeys as mk
import pytest
from cogno_anima.stages.ego import EgoStage
from cogno_anima.types import EgoResult, PipelineContext, ToolExecution, ToolResult

import cogno_soma
from cogno_soma import (JUDGE_SKIP_PREJUDGED_REPLAY, LEDGER_EGO_PROMPT_BLOCKS,
                        LEDGER_EGO_PROMPT_BLOCKS_DROPPED, LEDGER_EGO_PROMPT_PATH,
                        LEDGER_EGO_PROMPT_SHA, PREJUDGED_TEXT_SHA, Pipeline, TurnConfig,
                        prejudged_digest)
from cogno_soma import pipeline as pipeline_module
from cogno_soma.pipeline import _attempt_prompt
from tests.conftest import (FakeEgo, FakeID, FakeNER, FakeNoumeno, FakeSuperego,
                            RecordingDispatcher, StubEmbedder, metrics)

NEW_KEYS = (LEDGER_EGO_PROMPT_BLOCKS, LEDGER_EGO_PROMPT_BLOCKS_DROPPED, LEDGER_EGO_PROMPT_SHA,
            LEDGER_EGO_PROMPT_PATH)
PERSONA = "You run the front desk. persona-canary-51c0"
NOTES = "[TODAY] 2026-10-07\nnotes-canary-8e2d"
CRITIQUE = "The draft names a price no tool returned. critique-canary-77aa"


class _Text:
    """A plain text backend: the executor takes its text path and drafts a final answer."""
    model = "text"

    async def generate(self, system: str, prompt: str):
        return "The plan costs 10.", 5, 3


class _Native(_Text):
    """The same answer through the function-calling API: the executor takes its native path."""
    model = "native"

    def supports_native_tools(self) -> bool:
        return True

    async def chat_with_tools(self, messages, tools, tool_choice=None):
        return {"content": "The plan costs 10."}, 5, 3


class _Catalogue(RecordingDispatcher):
    def tools_schema(self) -> list[dict]:
        return [{"type": "function", "function": {
            "name": "read_plans", "description": "Reads the plans.",
            "parameters": {"type": "object", "properties": {"kind": {}}}}}]


def _slugs(row: dict) -> list[str]:
    return [b["block"] for b in row[LEDGER_EGO_PROMPT_BLOCKS]]


async def _turn(*, backend=None, ego=None, superego=None, budget=2, metadata=None,
                dispatcher=None, intent="INFORMATION_REQUEST"):
    """One EGO-routed turn with the REAL executor (the record is anima's to produce; a double
    has none) and a scripted judge. Returns ``(ctx, ledger)``."""
    pipe = Pipeline(embedder=StubEmbedder(), noumeno=FakeNoumeno("How much is the plan?"),
                    ner=FakeNER(intent_class=intent), id_stage=FakeID(route="EGO"),
                    ego=ego or EgoStage(),
                    superego=superego or FakeSuperego(approve_after=2, critique=CRITIQUE))
    cfg = TurnConfig(gen_backend=_Text(), ego_backend=backend or _Text(), ego_prompt=PERSONA,
                     max_corrections=budget)
    ctx = PipelineContext(user_input="quanto custa o plano?")
    ctx.metadata[mk.EGO_CONTEXT] = NOTES
    ctx.metadata.update(metadata or {})
    ctx = await pipe.run_turn(ctx, cfg, dispatcher=dispatcher or _Catalogue())
    return ctx, ctx.metadata.get(mk.JUDGE_ATTEMPTS) or []


# ── the twin: two attempts, two records ──────────────────────────────────────────────────────

async def test_twin_each_attempt_keeps_its_own_prompt_record():
    """Attempt 1 is rejected, attempt 2 approved. The retry's prompt gained ``correction``; the
    ledger shows both, and the first row is NOT the survivor's record."""
    ctx, ledger = await _turn()
    assert [(e["attempt"], e["approved"]) for e in ledger] == [(1, False), (2, True)]
    first, second = ledger
    assert _slugs(first) == ["persona", "task_context", "context", "available_tools", "tool_calls"]
    assert _slugs(second) == ["persona", "task_context", "context", "correction",
                              "available_tools", "tool_calls"]
    assert first[LEDGER_EGO_PROMPT_SHA] != second[LEDGER_EGO_PROMPT_SHA]
    assert first[LEDGER_EGO_PROMPT_PATH] == second[LEDGER_EGO_PROMPT_PATH] == "fallback"
    # The survivor on the context is attempt 2's — which is why attempt 1's had to be copied.
    survivor = ctx.ego_result
    assert second[LEDGER_EGO_PROMPT_BLOCKS] == survivor.prompt_blocks
    assert second[LEDGER_EGO_PROMPT_SHA] == survivor.prompt_sha
    assert first[LEDGER_EGO_PROMPT_BLOCKS] != survivor.prompt_blocks
    # What the two attempts share has the same length on both rows; the pair still adds up.
    shared = {b["block"]: b["chars"] for b in first[LEDGER_EGO_PROMPT_BLOCKS]}
    assert all(shared[b["block"]] == b["chars"] for b in second[LEDGER_EGO_PROMPT_BLOCKS]
               if b["block"] in shared)
    rows = second[LEDGER_EGO_PROMPT_BLOCKS]
    assert sum(b["chars"] for b in rows) + 2 * (len(rows) - 1) == len(survivor.prompt_text)
    assert LEDGER_EGO_PROMPT_BLOCKS_DROPPED not in first and LEDGER_EGO_PROMPT_BLOCKS_DROPPED not in second


async def test_control_one_approved_attempt_is_one_row_with_one_record():
    ctx, ledger = await _turn(superego=FakeSuperego(approve=True))
    (row,) = ledger
    assert row["approved"] is True and "correction" not in _slugs(row)
    assert row[LEDGER_EGO_PROMPT_BLOCKS] == ctx.ego_result.prompt_blocks


async def test_the_native_path_is_recorded_as_native_and_renders_no_catalogue():
    _, ledger = await _turn(backend=_Native())
    assert [e[LEDGER_EGO_PROMPT_PATH] for e in ledger] == ["native", "native"]
    assert _slugs(ledger[0]) == ["persona", "task_context", "context"]
    assert _slugs(ledger[1]) == ["persona", "task_context", "context", "correction"]
    # …and the two paths over the same turn do not share a digest.
    _, text = await _turn()
    assert ledger[0][LEDGER_EGO_PROMPT_SHA] != text[0][LEDGER_EGO_PROMPT_SHA]


async def test_a_rejected_attempt_that_wrote_shows_the_block_its_retry_was_given():
    """The retry after a COMMITTED write is told what was already done (``actions_done``); the
    ledger row of that retry says so. (The loop hands off after a rejected commit, so the retry
    is run here the way a host's correction would: the same stage, twice.)"""
    class _Wrote(_Catalogue):
        async def execute(self, name, arguments):
            return ToolResult(output="recorded", ok=True, side_effect=True)

    class _CallsOnce(_Text):
        def __init__(self) -> None:
            self.n = 0

        async def generate(self, system: str, prompt: str):
            self.n += 1
            if self.n == 1:
                return '<TOOL_CALL>{"tool": "read_plans", "args": {"kind": "a"}}</TOOL_CALL>', 5, 3
            return "Done.", 5, 3

    stage, ctx = EgoStage(), PipelineContext(user_input="regista")
    await FakeNoumeno("Record it.").process(ctx, None)
    await FakeNER(intent_class="ACTION_REQUEST").process(ctx, None)
    ctx = await stage.process(ctx, _CallsOnce(), _Wrote(), system_prompt=PERSONA)
    first = _attempt_prompt(ctx.ego_result)
    ctx.metadata[mk.EGO_CORRECTION] = {"reason": CRITIQUE, "attempt": 2}
    ctx = await stage.process(ctx, _Text(), _Wrote(), system_prompt=PERSONA)
    second = _attempt_prompt(ctx.ego_result)
    assert "actions_done" not in _slugs(first)
    assert _slugs(second)[2:4] == ["actions_done", "correction"]


# ── both places a row is written ─────────────────────────────────────────────────────────────

ARGS = {"target": "Aldina", "message": "A reunião passou para as 15h."}


class _Recorded(FakeEgo):
    """A double that EXECUTES like `FakeEgo` and carries a prompt record like the real stage."""

    async def process(self, ctx, backend, dispatcher, *, system_prompt):
        ctx = await super().process(ctx, backend, dispatcher, system_prompt=system_prompt)
        ctx.ego_result = ctx.ego_result.model_copy(update={
            "prompt_blocks": [{"block": "persona", "chars": 4}, {"block": "task_context", "chars": 9}],
            "prompt_sha": "0123456789ab", "prompt_path": "native", "prompt_text": "exec\n\n# Task c."})
        return ctx


async def _replay(ego_cls):
    sent = ToolExecution(tool="notify_user", arguments=dict(ARGS), ok=True, side_effect=True,
                         tool_mutating=True, result="Message delivered.")
    row = {"tool": "notify_user", "arguments": dict(ARGS),
           PREJUDGED_TEXT_SHA: prejudged_digest(ARGS["message"])}
    sup = FakeSuperego(approve=False, critique=CRITIQUE)
    pipe = Pipeline(embedder=None, noumeno=FakeNoumeno(), ner=FakeNER(intent_class="ACTION_REQUEST"),
                    id_stage=FakeID(route="EGO"), superego=sup,
                    ego=ego_cls(tool_calls=[sent], tools_offered=[["notify_user"]], drafts=["Feito."]))
    ctx = PipelineContext(user_input="sim")
    ctx.metadata.update({mk.HELD_DELIVERED_TEXT: {"notify_user": "message"},
                         mk.EGO_CONFIRMED: ["notify_user"], mk.EGO_CONFIRMED_CALLS: [row]})
    ctx = await pipe.run_turn(ctx, TurnConfig(gen_backend=None, ego_backend=None,
                                              ego_prompt="exec", max_corrections=1),
                              dispatcher=RecordingDispatcher())
    return sup, ctx.metadata[mk.JUDGE_ATTEMPTS]


async def test_the_row_of_a_prejudged_replay_carries_the_record_too():
    sup, (row,) = await _replay(_Recorded)
    assert sup._evals == 0 and row["skipped"] == JUDGE_SKIP_PREJUDGED_REPLAY     # the other site
    assert _slugs(row) == ["persona", "task_context"]
    assert (row[LEDGER_EGO_PROMPT_SHA], row[LEDGER_EGO_PROMPT_PATH]) == ("0123456789ab", "native")


# ── ABSENT, never defaulted ──────────────────────────────────────────────────────────────────

async def test_a_stand_in_executor_leaves_the_keys_absent_on_both_rows():
    """`FakeEgo` builds its own `EgoResult` and records no prompt: anima's "not on record"
    (an empty inventory, a `None` digest, a blank path). The row says nothing rather than
    ``[]``, which would read as "a prompt with no parts"."""
    _, judged = await _turn(ego=FakeEgo(drafts=["The plan costs 10."]))
    assert len(judged) == 2 and not [k for row in judged for k in NEW_KEYS if k in row]
    _, (replayed,) = await _replay(FakeEgo)
    assert not [k for k in NEW_KEYS if k in replayed]
    # The control: the same two shapes with a recording executor DO carry the keys.
    _, with_record = await _turn()
    assert all(LEDGER_EGO_PROMPT_BLOCKS in row for row in with_record)


def test_a_result_that_predates_the_fields_leaves_the_keys_absent():
    class _Old:                                   # an EgoResult of a cogno-anima before 0.1.5
        tools_executed: list = []
        draft = "x"

    assert _attempt_prompt(_Old()) == {}
    assert _attempt_prompt(None) == {}
    assert _attempt_prompt(EgoResult(metrics=metrics("ego"))) == {}


# ── closed alphabets: nothing from outside them is written ───────────────────────────────────

class _Raw:
    def __init__(self, blocks=None, sha=None, path=None) -> None:
        self.prompt_blocks, self.prompt_sha, self.prompt_path = blocks, sha, path


def test_a_slug_outside_the_alphabet_is_dropped_and_counted():
    out = _attempt_prompt(_Raw(blocks=[
        {"block": "persona", "chars": 12}, {"block": "somebody's sentence", "chars": 5},
        {"block": "context", "chars": "9"}, {"block": "context", "chars": True},
        {"block": "context", "chars": -1}, "context", {"chars": 3},
        {"block": "correction", "chars": 0, "text": "must not travel"}]))
    assert out == {LEDGER_EGO_PROMPT_BLOCKS: [{"block": "persona", "chars": 12},
                                              {"block": "correction", "chars": 0}],
                   LEDGER_EGO_PROMPT_BLOCKS_DROPPED: 6}
    assert set(cogno_anima.EGO_PROMPT_BLOCKS) >= {b["block"] for b in out[LEDGER_EGO_PROMPT_BLOCKS]}


def test_a_record_with_nothing_writable_says_so_and_is_not_an_absence():
    out = _attempt_prompt(_Raw(blocks=[{"block": "nobody", "chars": 3}]))
    assert out == {LEDGER_EGO_PROMPT_BLOCKS: [], LEDGER_EGO_PROMPT_BLOCKS_DROPPED: 1}


def test_the_rows_are_bounded_and_the_overflow_is_counted():
    many = [{"block": "persona", "chars": n} for n in range(pipeline_module._PROMPT_BLOCKS_PER_ATTEMPT + 3)]
    out = _attempt_prompt(_Raw(blocks=many))
    assert len(out[LEDGER_EGO_PROMPT_BLOCKS]) == pipeline_module._PROMPT_BLOCKS_PER_ATTEMPT
    assert out[LEDGER_EGO_PROMPT_BLOCKS_DROPPED] == 3


@pytest.mark.parametrize("sha", ["", "not-a-digest", "ABCDEF012345", "0123", "a" * 65, 12, None,
                                 "0123456789ab; the contact said"])
def test_a_digest_that_is_not_a_digest_is_not_written(sha):
    assert LEDGER_EGO_PROMPT_SHA not in _attempt_prompt(_Raw(sha=sha))


@pytest.mark.parametrize("path", ["", "Native", "text", "native ", 1, None])
def test_a_path_outside_the_alphabet_is_not_written(path):
    assert LEDGER_EGO_PROMPT_PATH not in _attempt_prompt(_Raw(path=path))


def test_control_a_digest_and_a_path_of_the_alphabets_are_written():
    out = _attempt_prompt(_Raw(sha="0123456789ab", path="fallback"))
    assert out == {LEDGER_EGO_PROMPT_SHA: "0123456789ab", LEDGER_EGO_PROMPT_PATH: "fallback"}
    assert set(cogno_anima.VALID_EGO_PROMPT_PATHS) == {"native", "fallback"}


def test_an_unreadable_result_costs_the_record_and_never_the_turn():
    class _Explodes:
        @property
        def prompt_blocks(self):
            raise RuntimeError("unreadable")

    assert _attempt_prompt(_Explodes()) == {}


# ── never the text ───────────────────────────────────────────────────────────────────────────

async def test_no_text_of_the_prompt_reaches_the_row():
    ctx, ledger = await _turn()
    # The control first: the canaries WERE in the prompt of the attempt the survivor is.
    text = ctx.ego_result.prompt_text
    assert all(c in text for c in ("persona-canary-51c0", "notes-canary-8e2d", "critique-canary-77aa"))
    new = json.dumps([{k: row[k] for k in NEW_KEYS if k in row} for row in ledger])
    assert "canary" not in new and "prompt_text" not in json.dumps(ledger)
    assert text not in json.dumps(ledger, ensure_ascii=False)


# ── the hole this does NOT close ─────────────────────────────────────────────────────────────

async def test_a_hold_with_no_delivered_text_has_no_row_only_the_survivor():
    """Gate B holds a destructive call and the loop leaves BEFORE any ledger row is written.
    Stated as a fact, so the day a row is written there this test says the note is stale."""
    class _Destructive(_Catalogue):
        def is_mutating(self, name: str) -> bool:
            return True

        def requires_confirmation(self, name: str) -> bool:
            return True

    class _Calls(_Text):
        async def generate(self, system: str, prompt: str):
            return '<TOOL_CALL>{"tool": "read_plans", "args": {"kind": "all"}}</TOOL_CALL>', 5, 3

    sup = FakeSuperego(approve=False)
    ctx, ledger = await _turn(backend=_Calls(), dispatcher=_Destructive(), superego=sup,
                              intent="ACTION_REQUEST")
    assert ctx.ego_result.pending_confirmation and sup._evals == 0
    assert ledger == []
    assert [b["block"] for b in ctx.ego_result.prompt_blocks][:2] == ["persona", "task_context"]
    assert ctx.ego_result.prompt_sha and ctx.ego_result.prompt_path == "fallback"


# ── the contract a host reads ────────────────────────────────────────────────────────────────

def test_the_keys_are_exported_and_spelled_once():
    assert (LEDGER_EGO_PROMPT_BLOCKS, LEDGER_EGO_PROMPT_BLOCKS_DROPPED, LEDGER_EGO_PROMPT_SHA,
            LEDGER_EGO_PROMPT_PATH) == ("ego_prompt_blocks", "ego_prompt_blocks_dropped",
                                        "ego_prompt_sha", "ego_prompt_path")
    for name in ("LEDGER_EGO_PROMPT_BLOCKS", "LEDGER_EGO_PROMPT_BLOCKS_DROPPED",
                 "LEDGER_EGO_PROMPT_SHA", "LEDGER_EGO_PROMPT_PATH"):
        assert name in cogno_soma.__all__
        assert getattr(cogno_soma, name) is getattr(pipeline_module, name)
