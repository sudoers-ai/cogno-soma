"""The interception seam: optional per-stage hooks + atomicity callbacks.

These mirror the parent ``PipelineRunner``'s ``InterceptorChain`` (``after_noumeno``
/ ``after_ner`` / ``after_id``) and ``StopPipeline`` mechanism, but infra-agnostic:
a hook is just a callable handed the live ``PipelineContext``. It may inspect or
mutate the context (e.g. inject retrieved memories into ``ctx.metadata``), or raise
``StopPipeline`` to halt the turn. Hooks may be sync **or** async (a memory store
lookup is naturally async) — the orchestrator awaits the result when it is awaitable.

Typical wiring at the host:
  * ``after_ner``    → persist the turn + inject episodic memory (cogno-engram),
                       run a crisis/safety screen (cogno-aegis) that may StopPipeline.
  * ``after_id``     → audit the route / attach tracing spans.
  * ``on_rollback``  → open/rollback the DB tx or write-behind buffer before a retry.
  * ``on_commit``    → flush the outbox once SUPEREGO approves the EGO execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Union

from cogno_anima.types import PipelineContext

# A hook receives the live context; returns nothing (sync) or an awaitable (async).
HookFn = Callable[[PipelineContext], Union[None, Awaitable[None]]]


@dataclass
class Hooks:
    """Optional callbacks fired at fixed points around the stages.

    Every field defaults to ``None`` (no-op). Fire order within a turn:
    ``before_turn`` → NOUMENO → ``after_noumeno`` → NER → ``after_ner`` → ID →
    ``after_id`` → [gates] → EGO⇄SUPEREGO loop → ``after_ego`` → SUPEREGO.voice →
    ``after_superego`` → ``after_turn``. ``on_rollback`` fires before each EGO retry;
    ``on_commit`` fires once the judge approves.
    """

    before_turn: Optional[HookFn] = None
    after_noumeno: Optional[HookFn] = None
    after_ner: Optional[HookFn] = None
    after_id: Optional[HookFn] = None
    after_ego: Optional[HookFn] = None
    after_superego: Optional[HookFn] = None
    after_turn: Optional[HookFn] = None
    # atomicity (the host opens/commits/rolls back its own tx / outbox here)
    on_commit: Optional[HookFn] = None
    on_rollback: Optional[HookFn] = None
