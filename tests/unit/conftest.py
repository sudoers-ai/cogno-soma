"""Unit-suite fixtures. The one that matters: no unit test reaches Ollama (see ``_ollama_gate``)."""

from __future__ import annotations

import pytest

from tests.unit._ollama_gate import Gate, fail_if_attempted, install_ollama_gate

_GATE = Gate()


@pytest.fixture(scope="session", autouse=True)
def _ollama_gate_installed():
    """Install the refusal for the WHOLE unit session — session scope, so it is in place before
    any other fixture (a session-scoped one included) can open a connection."""
    with pytest.MonkeyPatch.context() as mp:
        install_ollama_gate(mp, _GATE)
        yield _GATE


@pytest.fixture(autouse=True)
def ollama_gate(_ollama_gate_installed, request):
    """Per test: start with an empty record, and FAIL the test if it tried to reach Ollama —
    even when the code under test swallowed the refusal. A test that expects the refusal
    clears ``ollama_gate.attempts`` after asserting on it."""
    _GATE.attempts.clear()
    yield _GATE
    fail_if_attempted(_GATE, request.node.nodeid)
