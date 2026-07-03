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

from typing import Callable, Optional, Sequence

from cogno_anima.tools import ToolDispatcher
from cogno_anima.types import PipelineContext

from cogno_soma.config import TurnConfig
from cogno_soma.errors import SomaError
from cogno_soma.pipeline import Pipeline


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

        state = state or {}
        self._carry: dict = dict(state.get("carry", {}))
        self._history: list[str] = list(state.get("history", []))
        # The rolling transcript (original user text + the voiced assistant reply) — what makes a
        # follow-up like "the 9am one" / "Vinicius Vale" legible. Persisted so it survives workers.
        self._transcript: list[tuple[str, str]] = [
            (u, a) for u, a in state.get("transcript", [])]
        self._turn: int = int(state.get("turn_number", 0))

    async def run(
        self,
        user_input: str,
        *,
        memories: Optional[Sequence[str]] = None,
        dispatcher: Optional[ToolDispatcher] = None,
        metadata: Optional[dict] = None,
    ) -> PipelineContext:
        """Run the next turn. ``memories`` are host-retrieved facts injected as
        EGO context; ``metadata`` is merged last (host overrides win)."""
        self._turn += 1
        ctx = PipelineContext(user_input=user_input, force_language=self._force_language)
        ctx.metadata.update(self._carry)
        ctx.metadata["turn_number"] = self._turn
        if self._persona_id:
            ctx.metadata["active_persona_id"] = self._persona_id
        if self._mcp_module:
            ctx.metadata["active_mcp_module"] = self._mcp_module
        if self._history:
            ctx.metadata["last_rewritten"] = self._history[-1]
        # Conversation context: the recent transcript (so a follow-up resolves against what was
        # actually said — "Vinicius Vale" answers the assistant's "com quem?") + host memories.
        blocks: list[str] = []
        if self._transcript:
            lines: list[str] = []
            for user_turn, assistant_turn in self._transcript[-self._max_history:]:
                lines.append(f"User: {user_turn}")
                if assistant_turn:
                    lines.append(f"Assistant: {assistant_turn}")
            blocks.append("[CONVERSATION HISTORY]\n" + "\n".join(lines))
        if memories:
            blocks.append("[MEMORIES]\n" + "\n".join(memories))
        if blocks:
            ctx.metadata["ego_context"] = "\n\n".join(blocks)
        if metadata:
            ctx.metadata.update(metadata)

        disp = self._resolve_dispatcher(dispatcher)
        ctx = await self._pipeline.run_turn(ctx, self._config, dispatcher=disp)
        self._thread_forward(ctx, user_input)
        return ctx

    def _resolve_dispatcher(self, override: Optional[ToolDispatcher]) -> ToolDispatcher:
        disp = override or self._dispatcher
        if disp is None and self._dispatcher_factory is not None:
            disp = self._dispatcher_factory()
        if disp is None:
            raise SomaError(
                "no dispatcher: pass run(dispatcher=...), or set dispatcher / "
                "dispatcher_factory on the SessionRunner")
        return disp

    def _thread_forward(self, ctx: PipelineContext, user_input: str) -> None:
        carry: dict = {"id_state": ctx.metadata.get("id_state", {})}
        if ctx.intent and ctx.intent.goal:
            carry["last_goal"] = ctx.intent.goal
        if ctx.intent and ctx.intent.domains:
            carry["active_domains"] = ctx.intent.domains
        self._carry = carry
        if ctx.noumeno:
            self._history.append(ctx.noumeno.rewritten)
        # Record the exchange (user + the voiced reply) for the next turn's conversation context,
        # keeping only the most recent window so state stays bounded.
        reply = ctx.superego_result.response if ctx.superego_result else ""
        self._transcript.append((user_input, reply))
        self._transcript = self._transcript[-self._max_history:]

    @property
    def state(self) -> dict:
        """A serializable snapshot the host persists and feeds back via ``state=``."""
        return {
            "carry": dict(self._carry),
            "history": list(self._history),
            "transcript": [list(pair) for pair in self._transcript],
            "turn_number": self._turn,
        }

    @property
    def turn_number(self) -> int:
        return self._turn
