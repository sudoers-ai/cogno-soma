"""The ledger's tool result is cut WITH a mark, at the ceiling the HOST declares.

The writer cut at 200 and appended a bare ``…``; the reader (``cogno_host/trace.py``) cuts at
240 — 2000 for the schedule reads — and marks with ``…[cortado, faltam N chars]``. Below the
reader's base the reader's ceiling never bit on the ledger, so its mark never appeared there:
``trace["ego"]["tools"]`` got the marked cut and ``trace["judge"]["ledger"]`` stayed mute, which
is how a report about one half reads as a statement about the pair.

Each test below names the mutation it kills. Two of them carry an explicit CONTROL, because the
easiest way to pass "it always marks" is to mark everything, and the easiest way to pass "the
ceiling is honoured" is to have one ceiling.
"""

from __future__ import annotations

import pytest
from cogno_anima.types import ToolExecution

from cogno_anima import metakeys as mk
from cogno_anima.types import PipelineContext

from cogno_soma import Pipeline, TurnConfig
from cogno_soma.pipeline import _attempt_tools
from cogno_soma.trace_cuts import (CUT_MARK, TOOL_RESULT_CHARS, cut_dropped,
                                   cut_with_mark, resolve_limit, was_cut)
from tests.conftest import FakeEgo, FakeID, FakeNER, FakeNoumeno, FakeSuperego

# A host family and its ceiling, as a POLICY this test injects — which is the whole point of the
# parameter, and the reason these two names are made up: no tool name from anybody's catalog
# belongs in cogno-soma, its tests included. The shape is what is under test (a listing read gets
# the bigger ceiling, the tool beside it does not), never the names.
_BIG = 2000
_LISTING_READ = "list_the_whole_month"
_OTHER_READ = "search_the_corpus"


def _host_policy(tool: str) -> int:
    return _BIG if tool == _LISTING_READ else TOOL_RESULT_CHARS


def _exec(tool: str, result: str) -> ToolExecution:
    return ToolExecution(tool=tool, arguments={}, result=result, ok=True, side_effect=False)


class _Ego:
    def __init__(self, execs):
        self.tools_executed = execs
        self.tools_offered = []


def _rows(execs, limit_fn=None):
    entry = _attempt_tools(_Ego(execs), limit_fn)
    assert "tools_error" not in entry, "the display path degraded — read the error, not the rows"
    return entry["tools"]


# ── the mark, and the number in it ───────────────────────────────────────────────────────

def test_a_cut_result_says_so_and_says_how_much_is_missing():
    """Mutation: ``cut_with_mark`` returning ``text[:limit]`` (the silent cut this replaces), or
    the mark rendering the KEPT count / the TOTAL instead of the missing one.

    The assertion is arithmetic over the row alone — prefix + N == the original length — so a
    mark carrying the wrong number fails even though it is present and well-formed. The old
    behaviour (``text[:200] + "…"``) fails on the first clause."""
    original = "y" * 900
    row = _rows([_exec("read_ledger", original)])[0]
    assert was_cut(row["result"])
    kept = len(row["result"]) - len(CUT_MARK.format(n=cut_dropped(row["result"])))
    assert kept + cut_dropped(row["result"]) == len(original), (
        "N must be what is MISSING: prefix + N has to reconstruct the original length")


def test_a_short_result_is_returned_BYTE_FOR_BYTE_and_carries_no_mark():
    """The CONTROL for the test above. "Always mark" passes it trivially; this is what makes
    the pair mean something — and it is also the property the stored corpus depends on, since
    a reader comparing old rows with new must see an ADDITION on the cut ones and nothing at
    all on the rest.

    Mutation: marking unconditionally (dropping the ``len(text) <= limit`` early return)."""
    short = "ok, 3 aulas"
    row = _rows([_exec("read_ledger", short)])[0]
    assert row["result"] == short
    assert not was_cut(row["result"]) and cut_dropped(row["result"]) == 0


@pytest.mark.parametrize("limit", [TOOL_RESULT_CHARS, _BIG])
@pytest.mark.parametrize("length", [0, 1, 239, 240, 241, 1999, 2000, 2001, 9000])
def test_the_mark_fits_INSIDE_the_ceiling(limit: int, length: int):
    """Mutation: ``text[:limit] + mark`` (pasting the mark on the outside), which is what the
    old ``_cut`` did — it returned ``limit + 1`` characters for every cut field.

    A marker that overflows turns the ceiling into an approximation, and the ceiling exists
    because of the size of the row. The contract is ``<= limit``, not ``== limit``: the
    reservation is computed from the length of the WHOLE text, which majorates N's digit
    count, so a shorter N lands a character or two under."""
    assert len(cut_with_mark("y" * length, limit)) <= limit


def test_the_mark_is_read_at_the_END_so_prose_cannot_forge_it():
    """Mutation: dropping the ``$`` anchor from ``CUT_MARK_RE``.

    A tool whose output QUOTES the mark mid-text (a trace-reading tool describing its own
    corpus is the realistic case) would otherwise be counted as truncated, and its missing
    chars added to a total nobody could explain."""
    quoted = f"o resultado terminava em {CUT_MARK.format(n=12)} e por isso foi excluído"
    assert not was_cut(quoted) and cut_dropped(quoted) == 0


def test_a_whole_result_exactly_AT_the_ceiling_is_not_read_as_cut():
    """The length test was the only detector there was, and with two ceilings it is not a guess
    but an inversion: a 900-char schedule listing that arrived WHOLE is ``>= 240``, so the
    length test calls truncated precisely the family the bigger ceiling was raised for.

    Mutation: any ``len(result) >= TOOL_RESULT_CHARS`` detector. This row is exactly at the
    base ceiling and exactly whole."""
    whole = "y" * TOOL_RESULT_CHARS
    row = _rows([_exec("read_ledger", whole)])[0]
    assert row["result"] == whole and not was_cut(row["result"])


# ── the ceiling is the HOST's, per tool ──────────────────────────────────────────────────

def test_the_host_policy_decides_per_tool_and_the_sibling_call_is_the_control():
    """The schedule read gets the big ceiling; the tool BESIDE IT in the same turn does not.

    Mutation A: ignoring ``limit_fn`` (the ledger back at one ceiling) — the schedule row is cut.
    Mutation B: applying the big ceiling to everything — the control row stops being cut, and
    with it the argument that the raise is "maior para horário" and not "maior para tudo".

    The two rows travel in ONE call because that is the shape the defect had: a per-turn
    ceiling cannot tell them apart, and only a per-call one can."""
    body = "y" * 1500
    rows = _rows([_exec(_LISTING_READ, body), _exec(_OTHER_READ, body)],
                 _host_policy)
    assert rows[0]["result"] == body and not was_cut(rows[0]["result"])   # whole, at 2000
    assert was_cut(rows[1]["result"]) and len(rows[1]["result"]) <= TOOL_RESULT_CHARS


def test_a_result_over_the_BIG_ceiling_is_still_cut_and_still_marked():
    """Mutation: treating a declared family as "never cut" instead of "cut later". 2000 is a
    ceiling, not an exemption, and the marked row is what tells the next person sizing it how
    far over the traffic actually goes."""
    row = _rows([_exec(_LISTING_READ, "y" * (_BIG * 2))], _host_policy)[0]
    assert was_cut(row["result"]) and len(row["result"]) <= _BIG
    assert cut_dropped(row["result"]) == _BIG * 2 - (
        len(row["result"]) - len(CUT_MARK.format(n=cut_dropped(row["result"]))))


# ── the policy is a host's, so every way a host can break it degrades ─────────────────────

@pytest.mark.parametrize("bad", [
    pytest.param(lambda tool: 1 / 0, id="raises"),
    pytest.param(lambda tool: "muitos", id="not-a-number"),
    pytest.param(lambda tool: None, id="None"),
])
def test_a_broken_host_policy_degrades_to_the_default_and_never_kills_the_turn(bad):
    """This runs inside the PRODUCTION correction loop. The caller's ``except`` degrades the
    whole entry to ``tools: []`` — which reads as "called nothing", the exact ambiguity the
    ledger exists to end — so a policy that raises must be caught HERE, not there.

    Mutation: calling ``limit_fn`` bare. ``_rows`` asserts no ``tools_error``, so an escaping
    exception fails the test rather than silently emptying the list."""
    row = _rows([_exec(_LISTING_READ, "y" * 900)], bad)[0]
    assert was_cut(row["result"]) and len(row["result"]) <= TOOL_RESULT_CHARS


def test_a_policy_BELOW_the_default_is_floored_to_it():
    """The direction the host's pair test pins for the other shared constants, held in code
    here: below the reader's base the reader's ceiling never bites, the ledger half of the
    trace is cut shorter than the turn half, and the pair differs by the CUT and not by the
    turn. That is the 200 defect, and a config must not be able to ask for it back.

    Mutation: ``return limit`` — honouring the smaller number."""
    assert resolve_limit("read_ledger", lambda tool: 50) == TOOL_RESULT_CHARS
    row = _rows([_exec("read_ledger", "y" * 900)], lambda tool: 50)[0]
    assert 50 < len(row["result"]) <= TOOL_RESULT_CHARS and was_cut(row["result"])


def test_the_default_ceiling_can_hold_the_mark_it_promises():
    """A ceiling under the mark's own rendered length would keep the mark and drop the text —
    correct, and useless. The floor guarantees the default is far above it; this pins the gap
    rather than the current number, so raising or lowering ``TOOL_RESULT_CHARS`` is free and
    collapsing it is not."""
    assert TOOL_RESULT_CHARS > 4 * len(CUT_MARK.format(n=10 ** 6))


# ── and the policy actually TRAVELS from the config to the row ───────────────────────────

async def test_the_turn_config_policy_reaches_the_ledger(stub_embedder, stub_backend,
                                                         dispatcher):
    """Everything above calls ``_attempt_tools`` directly, so every one of those tests stays
    green if the pipeline stops passing ``cfg.tool_result_limit`` — the parameter would exist,
    be correct, and be consulted by nobody. That is the shape of the defect this whole change
    is about (a mechanism landed on one side of a pair), so it gets its own test through the
    real ``run_turn``.

    Mutation: ``_attempt_tools(ctx.ego_result)`` — dropping the argument at the call site. The
    schedule row then comes back cut at the base ceiling."""
    body = "y" * 1500
    ego = FakeEgo(tool_calls=[[_exec(_LISTING_READ, body),
                               _exec(_OTHER_READ, body)]])
    pipe = Pipeline(embedder=stub_embedder, noumeno=FakeNoumeno(), ner=FakeNER(),
                    id_stage=FakeID(route="EGO"), ego=ego,
                    superego=FakeSuperego(approve=False, critique="x"))
    cfg = TurnConfig(gen_backend=stub_backend, ego_backend=stub_backend, ego_prompt="exec",
                     max_corrections=1, tool_result_limit=_host_policy)
    ctx = await pipe.run_turn(PipelineContext(user_input="hi"), cfg, dispatcher=dispatcher)

    rows = ctx.metadata[mk.JUDGE_ATTEMPTS][0]["tools"]
    assert rows[0]["result"] == body and not was_cut(rows[0]["result"])
    assert was_cut(rows[1]["result"]) and len(rows[1]["result"]) <= TOOL_RESULT_CHARS
