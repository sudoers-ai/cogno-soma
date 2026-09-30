"""The gate that keeps the unit suite off the GPU (``tests/unit/_ollama_gate.py``).

Every test here that EXPECTS a refusal points the real client at a loopback port nothing
listens on, and declares that port as the Ollama through ``OLLAMA_BASE_URL``. That is what makes
the mutation safe to run: with the gate removed, the same call gets ``ConnectionRefused`` from an
empty port — the twin goes red because it stopped RAISING THE GATE'S ERROR, and nothing ever
reaches the Ollama that serves traffic. None of them touches port 11434.
"""

from __future__ import annotations

import socket
import urllib.request

import pytest
from cogno_synapse import OllamaBackend

from tests.conftest import StubBackend
from tests.unit._ollama_gate import OllamaGateError, fail_if_attempted, is_ollama


def _closed_port() -> int:
    """A loopback port with nothing listening (bound, read, released)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_ollama_url(monkeypatch) -> str:
    """Declare an EMPTY loopback port as this box's Ollama — the gate must refuse it."""
    url = f"http://127.0.0.1:{_closed_port()}"
    monkeypatch.setenv("OLLAMA_BASE_URL", url)
    return url


async def test_TWIN_a_real_backend_calling_generate_is_refused(fake_ollama_url, ollama_gate):
    """The incident's shape: a REAL ``OllamaBackend`` (what a ``None`` backend defaults to)
    calling ``generate``. It must raise the gate's error, cleanly, at the call site."""
    backend = OllamaBackend(model="qwen3:8b", base_url=fake_ollama_url)
    with pytest.raises(OllamaGateError, match="UNIT test tried to open a connection to Ollama"):
        await backend.generate("system", "prompt")
    assert len(ollama_gate.attempts) == 1
    ollama_gate.attempts.clear()   # expected — the teardown check must not fail THIS test


def test_TWIN_the_socket_layer_refuses_a_client_that_is_not_httpx(fake_ollama_url, ollama_gate):
    """``urllib`` never touches httpcore: only the socket layer can refuse it."""
    with pytest.raises(OllamaGateError):
        urllib.request.urlopen(f"{fake_ollama_url}/api/tags", timeout=2)   # noqa: S310
    assert [a for a in ollama_gate.attempts if "socket.connect" in a]
    ollama_gate.attempts.clear()


async def test_a_SWALLOWED_refusal_is_still_recorded(fake_ollama_url, ollama_gate):
    """``is_available`` catches every exception and answers False — a green test that tried
    to call the GPU, unless the attempt is recorded for the teardown check."""
    assert await OllamaBackend(model="qwen3:8b", base_url=fake_ollama_url).is_available() is False
    assert len(ollama_gate.attempts) == 1
    with pytest.raises(pytest.fail.Exception, match="tried to reach Ollama"):
        fail_if_attempted(ollama_gate, "the swallowing test")
    assert ollama_gate.attempts == []   # the check consumes the record it reports


def test_the_DEFAULT_endpoint_is_ollama_without_connecting():
    """The default a ``None`` backend falls back to — asserted on the predicate, never dialled."""
    for host in ("localhost", "127.0.0.1", "::1", "host.docker.internal"):
        assert is_ollama(host, 11434)


async def test_CONTROL_a_stub_passes(ollama_gate):
    assert await StubBackend().generate("system", "prompt") == ("{}", 0, 0)
    assert ollama_gate.attempts == []


def test_CONTROL_a_loopback_server_on_another_port_is_reachable(fake_ollama_url, ollama_gate):
    """The gate is NARROW: a test's own server on loopback is not Ollama."""
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        with socket.create_connection(srv.getsockname(), timeout=2):
            conn, _ = srv.accept()
            conn.close()
    assert not is_ollama("127.0.0.1", 8080)
    assert ollama_gate.attempts == []
