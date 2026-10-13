"""``LOGGING.md`` lists the log events this package emits — the code's list, both ways.

The document said the pipeline logs three DEBUG lines and nothing at INFO. Measured on
``b2bf3c3`` (this file, run against that tree): the package makes 15 log calls, one of them at
INFO, and the document named 3 of the 15 events. A host that sets a level from that page turned
on twelve lines it was never told about.

So the table in ``LOGGING.md`` is held to the code, in both directions:

* every ``logger.<level>("<event> …")`` call of the package has a row — logger, event, level and
  the ``key=`` fields of its message;
* every row is a call the package still makes.

The events are read from the source by AST, never listed here, so a new call fails this file
until the page says what it is. The same goes for the modules that log nothing, and for what the
"Level policy" section says about each level.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import cogno_soma

PACKAGE = Path(cogno_soma.__file__).parent
DOC = PACKAGE.parent / "LOGGING.md"
LEVELS = {"debug": "DEBUG", "info": "INFO", "warning": "WARNING", "error": "ERROR",
          "exception": "ERROR", "critical": "CRITICAL"}


def _module(path: Path) -> str:
    rel = path.relative_to(PACKAGE.parent).with_suffix("")
    parts = [p for p in rel.parts if p != "__init__"]
    return ".".join(parts)


def emitted(source: str, module: str) -> "list[tuple[str, str, str, frozenset[str], bool]]":
    """(logger, event, level, fields, traceback) of every ``logger.<level>(...)`` call in
    ``source``. The event is the first word of the message and the fields are its ``key=``
    names. A message that is not a string literal has no name a page could list, so it raises:
    the call must be given one."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in LEVELS and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("logger", "logging")):
            continue
        first = node.args[0] if node.args else None
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            raise AssertionError(f"{module}:{node.lineno}: a log message that is not a literal")
        message = first.value
        traceback = node.func.attr == "exception" or any(
            kw.arg == "exc_info" and not (isinstance(kw.value, ast.Constant)
                                          and kw.value.value in (False, None))
            for kw in node.keywords)
        out.append((module, message.split()[0], LEVELS[node.func.attr],
                    frozenset(re.findall(r"(\w+)=", message)), traceback))
    return out


def _code() -> "list[tuple[str, str, str, frozenset[str], bool]]":
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        found.extend(emitted(path.read_text(encoding="utf-8"), _module(path)))
    return found


def documented(text: str) -> "list[tuple[str, str, str, frozenset[str], bool]]":
    """The same five facts, read off the table of ``LOGGING.md``. A row is
    ``| `logger` | `event` (fields) | LEVEL | when |``; a pipe inside a cell is written ``\\|``.
    The fields are the backticked names in the event cell after the event (``key`` or
    ``key=…``); "traceback" in the level cell is the mark of a line that carries one."""
    out = []
    for line in text.splitlines():
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip())]
        if len(cells) != 6 or cells[0] or cells[-1]:
            continue
        logger = re.fullmatch(r"`(cogno_soma[\w.]*)`", cells[1])
        event = re.match(r"`(\w+)`", cells[2])
        level = re.match(r"([A-Z]+)\b", cells[3])
        if not (logger and event and level):
            continue
        fields = frozenset(re.findall(r"`(\w+)(?:=[^`]*)?`", cells[2][event.end():]))
        out.append((logger.group(1), event.group(1), level.group(1), fields,
                    "traceback" in cells[3]))
    return out


CODE = _code()
PAGE = documented(DOC.read_text(encoding="utf-8"))


# ── the two directions ─────────────────────────────────────────────────────────────────────

def test_every_event_the_code_emits_has_a_row():
    missing = sorted(set(CODE) - set(PAGE))
    assert not missing, f"emitted and not on the page (or not as the page says): {missing}"


def test_every_row_is_an_event_the_code_still_emits():
    stale = sorted(set(PAGE) - set(CODE))
    assert not stale, f"on the page and not emitted (or not as the page says): {stale}"


def test_no_event_is_listed_or_emitted_twice_under_one_name():
    """One row per call. Two calls sharing an event name are told apart on the page by their
    fields (``turn_blocked`` is emitted from two gates, with the same fields: one row, twice
    in the code)."""
    assert len(PAGE) == len(set(PAGE)), "a row appears twice"
    assert {(row[0], row[1]) for row in PAGE} == {(row[0], row[1]) for row in CODE}


# ── the controls: the readers can see what they look for ───────────────────────────────────

def test_control_the_code_reader_finds_the_calls_known_today():
    """A reader that found nothing would agree with an empty page. These are on ``b2bf3c3``
    already; the count is derived, not pinned."""
    events = {(row[1], row[2]) for row in CODE}
    assert {("turn_stopped", "DEBUG"), ("turn_blocked", "DEBUG"),
            ("scope_block_lifted", "INFO"), ("judge_escalated", "DEBUG")} <= events
    assert len(CODE) >= 15


def test_control_a_planted_call_is_read_and_a_planted_row_is_read():
    planted = ('logger = None\n'
               'def f():\n'
               '    logger.warning("made_up_event reason=%s attempt=%s", 1, 2, exc_info=True)\n')
    assert emitted(planted, "cogno_soma.x") == [
        ("cogno_soma.x", "made_up_event", "WARNING", frozenset({"reason", "attempt"}), True)]
    row = ("| `cogno_soma.x` | `made_up_event` (`reason=a\\|b`, `attempt`) "
           "| WARNING, with a traceback | never |")
    assert documented(row) == emitted(planted, "cogno_soma.x")
    with pytest.raises(AssertionError):
        emitted("logger = None\nlogger.info(some_variable)\n", "cogno_soma.x")


def test_control_the_comparison_fails_on_a_page_that_names_three_events():
    """The page as it was on ``b2bf3c3``, read by the same reader: nothing in it is a table
    row, so every event the code emits is missing."""
    old = ("- **DEBUG** — one line per terminal outcome of a turn: `turn_blocked`\n"
           "  (`stop_reason=pii_blocked|scope_blocked`), `turn_handoff`, `turn_stopped`.\n"
           "- `cogno_soma.pipeline` — DEBUG `turn_blocked` / `turn_handoff` / `turn_stopped`.\n")
    assert documented(old) == []
    assert len(set(CODE) - set(documented(old))) == len(set(CODE)) >= 14


# ── the two lists beside the table ─────────────────────────────────────────────────────────

def _section(title: str) -> str:
    text = DOC.read_text(encoding="utf-8")
    start = text.index(f"## {title}")
    end = text.find("\n## ", start + 1)
    return text[start:end if end > 0 else len(text)]


def test_the_modules_that_log_nothing_are_the_modules_that_log_nothing():
    modules = {_module(p) for p in PACKAGE.rglob("*.py")}
    loud = {row[0] for row in CODE}
    listed = set(re.findall(r"`(cogno_soma[\w.]*)`", _section("Modules that log nothing")))
    assert listed == modules - loud
    assert {row[0] for row in PAGE} == loud


@pytest.mark.parametrize("level", ["ERROR", "WARNING", "INFO", "DEBUG"])
def test_the_level_policy_says_which_levels_are_used(level):
    """Each level has one bullet. It says "never emitted" exactly when no call of the package
    is at that level, and otherwise names every event that is."""
    bullets = [ln for ln in _section("Level policy").splitlines()
               if ln.startswith(f"- **{level}**")]
    assert len(bullets) == 1, level
    at_level = {row[1] for row in CODE if row[2] == level}
    assert ("never emitted" in bullets[0]) == (not at_level), bullets[0]
    start = _section("Level policy").index(bullets[0])
    rest = _section("Level policy")[start:]
    end = rest.find("\n- **", 1)
    assert set(re.findall(r"`(\w+)`", rest[:end if end > 0 else len(rest)])) >= at_level
