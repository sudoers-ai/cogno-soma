"""The unit suite never reaches Ollama — ENFORCED, not promised.

The local Ollama is the GPU that serves live traffic. A unit test that reaches it queues real
contact turns behind the suite, and it does so SILENTLY: the test passes on a box where the
server is up, and nothing in the output says a model was called. The incident that made this a
gate (2026-09-30, in the host): a unit test built its object with ``gen_backend=None``, the code
did what production code should do — built the REAL default, an ``OllamaBackend`` on
``localhost:11434`` — and the test called the GPU. A convention ("unit tests use stubs") had
nothing to enforce it; this module is the enforcement, installed for the whole unit session by
``tests/unit/conftest.py``.

## Where it cuts, and why there (two layers, one predicate, one record)

1. ``socket.socket.connect`` / ``connect_ex`` — the LOWEST point every client ends in (httpx,
   the OpenAI SDK, ``urllib``, ``requests``, ``aiohttp``). This is what makes "no attempt gets
   through" true whatever library the code under test picked. It sees the RESOLVED address,
   which is why the predicate also accepts IPs.
2. ``httpcore``'s network backends (``AnyIOBackend.connect_tcp`` / ``SyncBackend.connect_tcp``)
   — one level ABOVE the socket, for httpx only, and only for the MESSAGE. Measured before
   writing it: a refusal raised at the socket reaches an ``httpx.AsyncClient`` caller wrapped in
   anyio's ``ExceptionGroup`` (its happy-eyeballs task group), and the OpenAI SDK re-wraps it as
   ``APIConnectionError``. Raised here, where the host name has not been resolved yet, the
   cogno-synapse path (``OllamaBackend.generate``) raises :class:`OllamaGateError` itself.

Neither layer is enough ALONE to make the test FAIL, because code under test is allowed to
swallow a connection error — ``OllamaBackend.is_available`` returns ``False``, the NOUMENO
degrades an embedder failure to ``UNKNOWN`` — and a swallowed refusal is a green test that tried
to call the GPU. So every refusal is also RECORDED, and the per-test fixture fails the test at
teardown when the record is not empty (:func:`fail_if_attempted`). A test that EXPECTS the
refusal (this gate's own tests) clears the record after asserting on it.

## What counts as "Ollama"

* port ``11434`` on ANY host — Ollama's port; no unit test has a reason to open it, and a
  container's ``host.docker.internal:11434`` is covered without knowing its name;
* the host:port of ``OLLAMA_BASE_URL`` / ``COGNO_OLLAMA_URL`` / ``OLLAMA_HOST``, read at CONNECT
  time (never at import), so a box that relocates its Ollama is still covered.

A loopback server a test binds for itself on another port is NOT touched (pinned by the
control in ``test_ollama_gate.py``). What this does NOT see: a child PROCESS the test spawns
(it has its own sockets and does not load this conftest unless it runs pytest here).
"""

from __future__ import annotations

import functools
import ipaddress
import os
import socket
from urllib.parse import urlsplit

import httpcore
import pytest

OLLAMA_PORT = 11434

#: Where a deployment says its Ollama lives. Read at connect time.
URL_ENV = ("OLLAMA_BASE_URL", "COGNO_OLLAMA_URL", "OLLAMA_HOST")


class OllamaGateError(RuntimeError):
    """A unit test tried to open a connection to Ollama."""


def configured_endpoints() -> set[tuple[str, int]]:
    """``(host, port)`` of every Ollama the environment names; unparseable values are skipped."""
    out: set[tuple[str, int]] = set()
    for var in URL_ENV:
        raw = (os.environ.get(var) or "").strip()
        if not raw:
            continue
        parts = urlsplit(raw if "://" in raw else f"http://{raw}")
        try:
            port = parts.port or OLLAMA_PORT
        except ValueError:
            continue
        if parts.hostname:
            out.add((parts.hostname.lower(), port))
    return out


def _is_local(host: str) -> bool:
    """Loopback, or the unspecified bind address (an Ollama on 0.0.0.0 answers on loopback)."""
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_unspecified


@functools.lru_cache(maxsize=64)
def _addresses(host: str) -> frozenset[str]:
    try:
        return frozenset(str(info[4][0]) for info in socket.getaddrinfo(host, None))
    except OSError:
        return frozenset()


def is_ollama(host: str, port: int) -> bool:
    """Does ``host:port`` (a name OR a resolved IP) point at an Ollama?"""
    if port == OLLAMA_PORT:
        return True
    host = str(host).lower().strip("[]")
    for c_host, c_port in configured_endpoints():
        if port != c_port:
            continue
        if host == c_host or (_is_local(host) and _is_local(c_host)) or host in _addresses(c_host):
            return True
    return False


class Gate:
    """The refusal and its record — one instance for the whole unit session."""

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def check(self, host: object, port: object, via: str) -> None:
        try:
            port_n = int(port)  # type: ignore[call-overload]
        except (TypeError, ValueError):
            return
        if not is_ollama(str(host), port_n):
            return
        where = f"{host}:{port_n}"
        self.attempts.append(f"{where} (via {via})")
        raise OllamaGateError(
            f"a UNIT test tried to open a connection to Ollama at {where} (via {via}). "
            "The local Ollama is the GPU that serves live traffic, so the unit suite refuses it. "
            "Inject a stub backend/embedder (tests/conftest.py) instead of letting the code "
            "build its real default (e.g. a None backend → OllamaBackend on localhost:11434); "
            "a test that needs a real model belongs in tests/integration.")


def install_ollama_gate(mp: pytest.MonkeyPatch, gate: Gate) -> None:
    """Patch both layers (see the module docstring) onto ``mp``; undone when ``mp`` is."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _check_sock(sock: socket.socket, address: object, via: str) -> None:
        if (sock.family in (socket.AF_INET, socket.AF_INET6)
                and isinstance(address, tuple) and len(address) >= 2):
            gate.check(address[0], address[1], via)

    def connect(self: socket.socket, address: object) -> None:
        _check_sock(self, address, "socket.connect")
        return real_connect(self, address)  # type: ignore[arg-type]

    def connect_ex(self: socket.socket, address: object) -> int:
        _check_sock(self, address, "socket.connect_ex")
        return real_connect_ex(self, address)  # type: ignore[arg-type]

    mp.setattr(socket.socket, "connect", connect)
    mp.setattr(socket.socket, "connect_ex", connect_ex)

    real_async = httpcore.AnyIOBackend.connect_tcp
    real_sync = httpcore.SyncBackend.connect_tcp

    async def async_connect_tcp(self, host, port, *args, **kwargs):  # type: ignore[no-untyped-def]
        gate.check(host, port, "httpcore.AnyIOBackend.connect_tcp")
        return await real_async(self, host, port, *args, **kwargs)

    def sync_connect_tcp(self, host, port, *args, **kwargs):  # type: ignore[no-untyped-def]
        gate.check(host, port, "httpcore.SyncBackend.connect_tcp")
        return real_sync(self, host, port, *args, **kwargs)

    mp.setattr(httpcore.AnyIOBackend, "connect_tcp", async_connect_tcp)
    mp.setattr(httpcore.SyncBackend, "connect_tcp", sync_connect_tcp)


def fail_if_attempted(gate: Gate, nodeid: str = "") -> None:
    """Fail the test when it tried to reach Ollama — even if the code swallowed the refusal."""
    if not gate.attempts:
        return
    seen = "; ".join(gate.attempts)
    gate.attempts.clear()
    pytest.fail(
        f"{nodeid or 'this test'} tried to reach Ollama ({seen}) and the unit suite refused it. "
        "The refusal may have been swallowed by the code under test (a probe returning False, a "
        "degraded embedder) — the attempt is the defect either way: inject a stub.",
        pytrace=False)
