"""The cut that leaves a MARK — one spelling, one ceiling, for the tool results soma persists.

``_attempt_tools`` writes a per-attempt ledger into ``ctx.metadata``; the host reads it, bounds
it again and persists it. Both sides cut the same text, and until now they cut it *differently*
and *silently*:

* the writer (here) cut a tool result at 200 characters and appended a bare ``…`` — no number,
  no shape a reader could match, and one character OVER the stated limit;
* the reader (``cogno_host/trace.py``) cuts at 240, or at 2000 for the schedule reads, and since
  its own #927 it appends ``…[cortado, faltam N chars]`` with the real N.

200 is BELOW even the reader's base ceiling, so the reader's ceiling never bit on the ledger and
its mark never appeared there: the half of the trace that answers "what did the rejected attempt
read" was the one half still cut in the dark, and a report saying "the ceiling was raised" was
true of ``trace["ego"]["tools"]`` and false of ``trace["judge"]["ledger"]``. Half a landed pair
looks exactly like a whole one.

**Three properties, and each is here because its absence was the defect:**

1. *A cut says so, and says how much.* ``…[cortado, faltam N chars]`` — N is what is MISSING,
   not what was kept and not the total. Ambiguity in the mark is a number the next reader has
   to guess, which is where this started.
2. *The mark fits INSIDE the limit.* Room is reserved before cutting, so the field never exceeds
   ``limit``. A marker pasted on the outside turns the ceiling into an approximation — and the
   old ``text[:limit] + "…"`` did exactly that.
3. *A reader detects a cut by the MARK, never by the length.* With two ceilings in play, the
   length test is not merely a guess (a result that happens to be exactly 240 long reads as cut)
   — it is WRONG in the one direction that matters: a 900-character schedule listing that
   arrived WHOLE is ``>= 240``, and the length test calls truncated exactly the family the
   bigger ceiling was raised for.

**Why the spelling lives HERE and not in both repos.** The host depends on cogno-soma, never the
other way round, so a constant the two must agree on can have exactly one home and this is it.
Two hand-kept copies of one mark would be a second duplicated contract written to repair the
first: the copies drift, and the day they do, ``was_cut`` stops recognising the writer's own
output and the corpus goes quiet again. Same argument as ``STOP_JUDGE_EXHAUSTED``, exported for
the same reason — "a string literal copied there would be a second definition of a contract that
has already been wrong once".

**Why the CEILING is a parameter and the mark is not.** Which tools deserve a bigger ceiling is a
catalog of tool NAMES — "the reads whose output is a listing", whatever those are called in a
given deployment — and a catalog of tool names is the host's business, not an infra-agnostic
orchestrator's. Nothing here may name one. This module ships the MECHANISM and takes the
declaration as a parameter (``TurnConfig.tool_result_limit``, in the shape ``EscalateFn`` already
established: the host owns the policy, soma consults it). What is left here is the DEFAULT for a
host that declares nothing.

**Why the default is 240 — equality, not a direction.** The writer/reader pairs this repo already
has (``_DRAFT_CHARS``, ``_CRITIQUE_CHARS``) are pinned by a DIRECTION: the writer must not cut
below the reader, because the reader re-cuts everything at its own number anyway. That argument
does not carry over to a MARKED field, and the difference is measurable in both directions:

* ``writer < reader`` — the reader's ceiling never bites on the ledger and its own mark never
  appears. This is the 200 defect, byte for byte.
* ``writer > reader`` — the reader re-cuts a string that already ENDS in a mark, so the
  writer's mark is sliced off the end and the surviving N counts only what the READER dropped.
  Reconstructed exactly (both cutters are pure, so this needs no corpus): a 700-character result
  cut here at 300 and then re-cut at 240 keeps 212 characters and reports ``faltam 88`` for a
  field that is missing 488 — the mark's one job, answering "would a bigger ceiling have held
  this row", wrong by 5.5x.

Equality is the only relation correct in both directions, and one number is a stronger way to
hold an equality than two numbers and a test. Half of it is held in code rather than in a test:
:func:`resolve_limit` floors a declared ceiling at :data:`TOOL_RESULT_CHARS`, so ``writer <
reader`` cannot be configured back into existence.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

#: The base ceiling for ONE tool result in the per-attempt ledger, used when the host declares
#: no policy. It is the host reader's own base (``cogno_host/trace.py::_FIELD_CHARS``) because
#: equality is the only safe relation — see the module docstring. A host that wants more (the
#: schedule reads get 2000 there) declares it through ``TurnConfig.tool_result_limit``.
TOOL_RESULT_CHARS = 240

#: The mark itself. ``…`` opens it (a human sees the cut without reading the rest), ``[cortado,``
#: is a closed, greppable literal, and ``faltam N chars`` states the DIRECTION. It goes at the
#: END so the stored prefix stays byte for byte what it was: whoever compares old rows with new
#: sees an addition, never a shift.
CUT_MARK = "…[cortado, faltam {n} chars]"

#: Anchored at the END, with ``\d+`` mandatory: a tool result that happens to contain the phrase
#: mid-text is not read as cut. The mark is not unforgeable — nothing in a free-text field is —
#: but the anchor makes forging it deliberate.
CUT_MARK_RE = re.compile(r"…\[cortado, faltam (\d+) chars\]$")

#: Host policy: given a tool NAME, the ceiling for that tool's result in the ledger. The host
#: passes its own per-family function (the same one it applies when it re-cuts what arrives),
#: so the number and the family have ONE definition in the deployment instead of two.
ToolResultLimitFn = Callable[[str], int]


def cut_with_mark(text: str, limit: int) -> str:
    """``text`` bounded to ``limit`` characters — and if it had to cut, it SAYS SO.

    The reservation is computed from the length of the WHOLE text, which majorates the digit
    count of ``N``; so the rendered mark never exceeds the room kept for it and the result is
    always ``<= limit``. When ``N`` has fewer digits the field lands one character short, and
    that is the contract: ``<= limit``, not ``== limit``.

    A ``limit`` under the mark's own rendered length (26-31 characters) keeps the MARK and
    drops the text — the same choice the host reader makes, deliberately: a bound that cannot
    say "this was cut" is a misconfiguration, and silently returning an unmarked prefix is the
    defect this module exists to end. It is unreachable through :func:`resolve_limit`, whose
    floor is an order of magnitude above it.
    """
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(CUT_MARK.format(n=len(text))))
    return text[:keep] + CUT_MARK.format(n=len(text) - keep)


def was_cut(text: Any) -> bool:
    """Did this field get cut? Reads the MARK, never the length — see property 3 above."""
    return CUT_MARK_RE.search(str(text or "")) is not None


def cut_dropped(text: Any) -> int:
    """How many characters the cut removed; 0 when the field was not cut.

    This is the half that makes the mark worth writing: "it was cut" tells a reader to exclude
    the row, "and 3412 chars are missing" tells them whether the next ceiling would have held
    it. The number the censored corpus could not answer is exactly this one.
    """
    m = CUT_MARK_RE.search(str(text or ""))
    return int(m.group(1)) if m else 0


def resolve_limit(tool: Any, limit_fn: Optional[ToolResultLimitFn]) -> int:
    """The ceiling for ONE tool's result: the host's policy, or :data:`TOOL_RESULT_CHARS`.

    Everything a host can get wrong here degrades to the default instead of raising. This runs
    inside the PRODUCTION correction loop, on the telemetry path, and the rule that path has
    obeyed since ``_attempt_tools`` was written is that a display detail must never be the
    reason a turn dies — a policy function that raises, returns a string, or returns a number
    too small to hold the mark would otherwise take the whole ledger entry down with it (the
    caller's ``except`` degrades to ``tools: []``, which reads as "called nothing": the exact
    ambiguity the ledger exists to end).
    """
    if limit_fn is None:
        return TOOL_RESULT_CHARS
    try:
        limit = int(limit_fn(str(tool or "")))
    except Exception:  # noqa: BLE001 — telemetry policy, never a dead turn
        return TOOL_RESULT_CHARS
    # The floor is the writer/reader DIRECTION, enforced here instead of only in the host's
    # pair test: below the reader's base the reader's own ceiling never bites, the ledger half
    # of the trace is cut shorter than the turn half, and a reader comparing "what the judge
    # read on attempt N" against "what the turn kept" is reading a difference made by the CUT
    # and not by the turn. A host asking for a smaller trace is asking for the 200 defect back.
    return limit if limit >= TOOL_RESULT_CHARS else TOOL_RESULT_CHARS
