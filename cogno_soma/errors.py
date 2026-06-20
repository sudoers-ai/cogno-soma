"""Errors and control-flow signals for the soma orchestrator."""

from __future__ import annotations

from typing import Optional


class SomaError(RuntimeError):
    """Base class for every error raised by ``cogno-soma``."""


class StopPipeline(SomaError):  # noqa: N818 — a control signal, not a failure
    """Raised by a host hook to halt the turn early and return the partial context.

    This is the orchestrator's interception seam (the parent's ``StopPipeline``):
    any hook — a memory persister, a crisis/safety screen, an audit gate — can
    raise it to stop the pipeline at that point. ``run_turn`` catches it, records
    ``ctx.stop_reason = reason`` and, when a ``response`` is supplied, writes a
    terminal ``SuperegoResult`` so the host still has a reply to send.

    ``reason`` should be a value from ``cogno_anima.vocab.VALID_STOP_REASONS``
    (the core's closed terminal vocabulary), but the orchestrator does not enforce
    it — the host owns the meaning of its own stop reasons.
    """

    def __init__(
        self,
        *,
        reason: str = "completed",
        response: Optional[str] = None,
        blocked: bool = False,
    ) -> None:
        super().__init__(f"pipeline stopped: {reason}")
        self.reason = reason
        self.response = response
        self.blocked = blocked
