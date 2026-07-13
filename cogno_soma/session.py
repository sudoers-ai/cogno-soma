"""``SessionRunner`` — threads cross-turn state for a multi-turn session.

``Pipeline.run_turn`` is single-turn and stateless: every cross-turn signal rides
in ``ctx.metadata`` (``id_state`` for goal continuity, ``last_rewritten`` for the
subject-change check, NER carry-over like ``last_goal``/``active_domains``). In a
real conversation something has to carry that forward turn-to-turn — in the
parent that was the stateful ``PipelineRunner`` (``self._turn_number`` /
``self._last_pii_risk``); in the cognobench it was hand-rolled per case.

``SessionRunner`` encapsulates exactly that threading. It is itself a thin,
**serializable** holder: ``.state`` exports a plain dict the host persists (to its
DB / Redis), and the constructor's ``state=`` restores it — so a multi-worker HTTP
host reconstructs the runner per request instead of pinning a live instance. The
host still owns persistence; soma owns the carry logic.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

from cogno_anima import metakeys as mk
from cogno_anima.tools import ToolDispatcher
from cogno_anima.types import PipelineContext

from cogno_soma.config import TurnConfig
from cogno_soma.errors import SomaError
from cogno_soma.pipeline import Pipeline

# The voicer draws facts from THIS turn's tool results and the user's message; everything
# else is SUPPORT. Layered by authority + freshness so a stale listing is never re-voiced
# as current data (the 2026-07 doctor's-agenda fabrication: a 2-day-old agenda listing sat
# in the verbatim window and the voicer copied it over an empty fresh read).
_SOURCES_INSTRUCTION = (
    "Facts you state — dates, times, amounts, names, statuses — must come from THIS turn's "
    "tool results or the user's message. The sections below are SUPPORTING context only: "
    "RECENT CONVERSATION for reference and continuity, EARLIER CONTEXT / MEMORIES / KNOWLEDGE "
    "GRAPH for background. Do NOT restate their data as if it were current — when a fact is "
    "not backed by a fresh tool result this turn, do not assert it."
)
# Verbatim conversation is kept only for the current time-burst: an exchange older than this
# many seconds from the current turn drops out of the verbatim window (it may still be
# represented, summarised and payload-free, by the host's EARLIER CONTEXT). Coarse on
# purpose — the axis that matters is minutes-ago vs days-ago, not exact spacing.
_DEFAULT_BURST_GAP_SECONDS = 4 * 60 * 60


class SessionRunner:
    """Drive a multi-turn session, threading ``id_state`` + history + NER carry-over.

    Args:
        pipeline:           the shared :class:`Pipeline` (stages are stateless).
        config:             the per-session :class:`TurnConfig` (backends + prompts).
        dispatcher:         a dispatcher reused for every turn, OR ...
        dispatcher_factory: a callable building a fresh dispatcher per turn (the host
                            typically rebuilds it with the request's tenant/auth scope).
                            ``run(dispatcher=...)`` overrides both per-call.
        persona_id:         optional ``active_persona_id`` stamped on each turn.
        mcp_module:         optional ``active_mcp_module`` stamped on each turn.
        force_language:     optional per-session language (``ctx.force_language``).
        state:              a dict from a prior ``.state`` to resume the session.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        config: TurnConfig,
        *,
        dispatcher: Optional[ToolDispatcher] = None,
        dispatcher_factory: Optional[Callable[[], ToolDispatcher]] = None,
        persona_id: Optional[str] = None,
        mcp_module: Optional[str] = None,
        force_language: Optional[str] = None,
        max_history: int = 6,
        burst_gap_seconds: float = _DEFAULT_BURST_GAP_SECONDS,
        state: Optional[dict] = None,
    ) -> None:
        self._pipeline = pipeline
        self._config = config
        self._dispatcher = dispatcher
        self._dispatcher_factory = dispatcher_factory
        self._persona_id = persona_id
        self._mcp_module = mcp_module
        self._force_language = force_language
        # How many recent (user, assistant) exchanges to feed back as conversation context.
        self._max_history = max_history
        # An exchange older than this (seconds, from the current turn) leaves the verbatim window.
        self._burst_gap = burst_gap_seconds

        state = state or {}
        self._carry: dict = dict(state.get("carry", {}))
        self._history: list[str] = list(state.get("history", []))
        # The rolling transcript: (user text, voiced reply, epoch-ts) — what makes a follow-up
        # like "the 9am one" / "Vinicius Vale" legible. The timestamp lets the verbatim window
        # track the real wall-clock gap across persisted state. Older 2-tuple rows (pre-ts state)
        # load with ts=0.0 → treated as ancient → never verbatim (safe: they only leave the window).
        self._transcript: list[tuple[str, str, float]] = [
            (row[0], row[1], float(row[2]) if len(row) > 2 else 0.0)
            for row in state.get("transcript", [])]
        self._turn: int = int(state.get("turn_number", 0))

    async def run(
        self,
        user_input: str,
        *,
        memories: Optional[Sequence[str]] = None,
        prior_summary: Optional[str] = None,
        graph_context: Optional[str] = None,
        dispatcher: Optional[ToolDispatcher] = None,
        metadata: Optional[dict] = None,
        now: Optional[float] = None,
    ) -> PipelineContext:
        """Run the next turn.

        The EGO/voicer context is layered by authority + freshness (see
        ``_SOURCES_INSTRUCTION``): RECENT CONVERSATION (verbatim, current time-burst only) →
        EARLIER CONTEXT (``prior_summary``, host-summarised older turns, payload-free) →
        MEMORIES (``memories``, durable user facts) → KNOWLEDGE GRAPH (``graph_context``,
        tenant relations). All support layers are optional; ``metadata`` is merged last
        (host overrides win). ``now`` overrides the burst clock (default wall-clock) for
        deterministic tests/replay."""
        self._turn += 1
        now = time.time() if now is None else now
        ctx = PipelineContext(user_input=user_input, force_language=self._force_language)
        ctx.metadata.update(self._carry)
        ctx.metadata[mk.TURN_NUMBER] = self._turn
        if self._persona_id:
            ctx.metadata[mk.ACTIVE_PERSONA_ID] = self._persona_id
        if self._mcp_module:
            ctx.metadata[mk.ACTIVE_MCP_MODULE] = self._mcp_module
        if self._history:
            ctx.metadata[mk.LAST_REWRITTEN] = self._history[-1]

        # RECENT CONVERSATION — only the current time-burst is fed verbatim, so a listing from
        # hours/days ago never re-enters as if it were fresh data (it may reappear, summarised
        # and payload-free, via prior_summary). A bare "sim" after a long gap is genuinely a new
        # turn and starts clean rather than inheriting a stale antecedent.
        transcript = self._verbatim_transcript(now)
        blocks: list[str] = [f"[SOURCES]\n{_SOURCES_INSTRUCTION}"]
        if transcript:
            blocks.append("[RECENT CONVERSATION]\n" + transcript)
            # The perception stages (NOUMENO/NER) read this to resolve a bare follow-up
            # ("com o Vinicius Vale") against the assistant's last question instead of
            # classifying it UNKNOWN and scope-blocking — same burst-scoped view.
            ctx.metadata[mk.CONVERSATION_HISTORY] = transcript
        if prior_summary:
            blocks.append("[EARLIER CONTEXT]\n" + prior_summary)
        if memories:
            blocks.append("[MEMORIES]\n" + "\n".join(memories))
        if graph_context:
            blocks.append("[KNOWLEDGE GRAPH]\n" + graph_context)
        # Always set EGO_CONTEXT (the SOURCES instruction alone is worth carrying) so the host's
        # own stamps (which prepend to EGO_CONTEXT) still land on a purely-social first turn.
        ctx.metadata[mk.EGO_CONTEXT] = "\n\n".join(blocks)
        if metadata:
            ctx.metadata.update(metadata)

        disp = self._resolve_dispatcher(dispatcher)
        ctx = await self._pipeline.run_turn(ctx, self._config, dispatcher=disp)
        self._thread_forward(ctx, user_input, now)
        return ctx

    def _verbatim_transcript(self, now: float) -> str:
        """The recent-burst transcript as ``User:``/``Assistant:`` lines. An exchange is kept
        only when it is within ``burst_gap`` of ``now`` AND within the ``max_history`` count,
        taken contiguously from the newest — so the window closes at the first stale/old turn."""
        kept: list[tuple[str, str]] = []
        for user_turn, assistant_turn, ts in reversed(self._transcript[-self._max_history:]):
            if now - ts > self._burst_gap:
                break                        # older than the burst → and everything before it too
            kept.append((user_turn, assistant_turn))
        lines: list[str] = []
        for user_turn, assistant_turn in reversed(kept):
            lines.append(f"User: {user_turn}")
            if assistant_turn:
                lines.append(f"Assistant: {assistant_turn}")
        return "\n".join(lines)

    def _resolve_dispatcher(self, override: Optional[ToolDispatcher]) -> ToolDispatcher:
        disp = override or self._dispatcher
        if disp is None and self._dispatcher_factory is not None:
            disp = self._dispatcher_factory()
        if disp is None:
            raise SomaError(
                "no dispatcher: pass run(dispatcher=...), or set dispatcher / "
                "dispatcher_factory on the SessionRunner")
        return disp

    def _thread_forward(self, ctx: PipelineContext, user_input: str, now: float) -> None:
        carry: dict = {mk.ID_STATE: ctx.metadata.get(mk.ID_STATE, {})}
        if ctx.intent and ctx.intent.goal:
            carry[mk.LAST_GOAL] = ctx.intent.goal
        if ctx.intent and ctx.intent.domains:
            carry[mk.ACTIVE_DOMAINS] = ctx.intent.domains
        self._carry = carry
        if ctx.noumeno:
            self._history.append(ctx.noumeno.rewritten)
        # Record the exchange (user + voiced reply + this turn's timestamp) for the next turn's
        # conversation context, keeping only the most recent window so state stays bounded.
        reply = ctx.superego_result.response if ctx.superego_result else ""
        self._transcript.append((user_input, reply, now))
        self._transcript = self._transcript[-self._max_history:]

    @property
    def state(self) -> dict:
        """A serializable snapshot the host persists and feeds back via ``state=``."""
        return {
            "carry": dict(self._carry),
            "history": list(self._history),
            "transcript": [[u, a, ts] for u, a, ts in self._transcript],
            "turn_number": self._turn,
        }

    @property
    def turn_number(self) -> int:
        return self._turn
