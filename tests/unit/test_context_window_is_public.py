"""The runner's verbatim window is a PUBLIC contract — the numbers and the behaviour agree.

The host writes rows into the runner's transcript itself (a delivered message, a proactive
turn) and has to know whether its row is still inside the window the model reads. Until
2026-09-30 it imported the private ``_DEFAULT_BURST_GAP_SECONDS`` across its pinned boundary
and kept its OWN copy of the exchange count (6), falling back to its own 4 h in silence if the
private name ever went away. ``CONTEXT_WINDOW_GAP_SECONDS`` / ``CONTEXT_WINDOW_EXCHANGES`` are
the names it can hold the soma to.

Three things are pinned here, and each would pass on its own while the contract rotted:
the names exist at the front door (the package root, in ``__all__``); they ARE the defaults of
``SessionRunner`` (a default changed as a literal would leave the constant describing a window
nobody runs); and the window a default runner actually applies is the one they describe — the
twin just inside each edge is kept, the one just outside is not.
"""

from __future__ import annotations

import inspect

import cogno_soma
from cogno_soma import session as session_mod
from cogno_soma.session import SessionRunner


def _runner(rows: list[tuple[str, str, float]]) -> SessionRunner:
    """A default-window runner over a persisted transcript. The window never touches the
    pipeline or the config, so neither is built."""
    return SessionRunner(None, None, state={"transcript": [list(r) for r in rows]})  # type: ignore[arg-type]


def test_the_names_are_at_the_front_door():
    assert "CONTEXT_WINDOW_GAP_SECONDS" in cogno_soma.__all__
    assert "CONTEXT_WINDOW_EXCHANGES" in cogno_soma.__all__
    assert cogno_soma.CONTEXT_WINDOW_GAP_SECONDS == 4 * 60 * 60
    assert cogno_soma.CONTEXT_WINDOW_EXCHANGES == 6
    assert cogno_soma.CONTEXT_WINDOW_GAP_SECONDS is session_mod.CONTEXT_WINDOW_GAP_SECONDS


def test_the_private_spelling_is_an_alias_of_the_public_one():
    """Nothing that imported the old name breaks, and it cannot drift: it is the same object."""
    assert session_mod._DEFAULT_BURST_GAP_SECONDS is session_mod.CONTEXT_WINDOW_GAP_SECONDS


def test_the_constants_are_the_runner_defaults():
    params = inspect.signature(SessionRunner.__init__).parameters
    assert params["burst_gap_seconds"].default is session_mod.CONTEXT_WINDOW_GAP_SECONDS
    # `==`, not `is`: a small int is cached, so `is` would also pass for a literal 6. The count
    # twin below is what catches a default that drifted from the constant.
    assert params["max_history"].default == session_mod.CONTEXT_WINDOW_EXCHANGES


def test_the_gap_edge_is_where_the_constant_says():
    """Twin across the gap: one second inside is verbatim, one second outside is not."""
    now = 1_000_000.0
    gap = cogno_soma.CONTEXT_WINDOW_GAP_SECONDS
    inside = _runner([("dentro", "ok", now - gap + 1)])._verbatim_transcript(now)
    outside = _runner([("fora", "ok", now - gap - 1)])._verbatim_transcript(now)
    assert "User: dentro" in inside
    assert outside == ""


def test_the_count_edge_is_where_the_constant_says():
    """Twin across the count: of N+1 fresh exchanges, the newest N are verbatim and the oldest
    is not."""
    now = 1_000_000.0
    n = cogno_soma.CONTEXT_WINDOW_EXCHANGES
    rows = [(f"troca {i}", "ok", now - 10 * (n - i)) for i in range(n + 1)]
    text = _runner(rows)._verbatim_transcript(now)
    assert text.count("User: ") == n
    assert "User: troca 0" not in text
    assert f"User: troca {n}" in text
