"""``TurnConfig`` — the static, per-session wiring a turn needs.

Everything here is fixed for the life of a session (the backends, the persona's
four prompt slots, the correction budget, the hooks). The one thing that varies
per turn is the ``ToolDispatcher`` (the host builds it fresh with the request's
auth/tenant scope), so it is passed to ``run_turn`` separately, not held here.

The four prompts are plain strings: soma is **prompt-agnostic**. The host resolves
which persona answers (e.g. via ``cogno-persona``'s ``PersonaSelector``) and hands
the resolved ``system``/``scope``/``limits``/``voice`` text in — soma never depends
on a persona store. ``gen_backend`` drives the JSON stages (NOUMENO/NER/scope/judge);
``ego_backend`` drives the EGO executor (native FC or text-fallback); ``voice_backend``
defaults to ``ego_backend`` but may differ (e.g. a smaller voicer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cogno_synapse import LLMBackend

from cogno_soma.hooks import Hooks


@dataclass
class TurnConfig:
    gen_backend: LLMBackend           # NOUMENO / NER / scope / judge (JSON-capable)
    ego_backend: LLMBackend           # EGO executor (FC or text fallback)
    ego_prompt: str                   # persona execution prompt (SUPEREGO is the voicer)
    scope_prompt: str = ""            # empty → skip the cheap pre-EGO scope guard
    limits_prompt: str = ""           # judge criterion: persona limits
    voice_prompt: str = ""            # voicer: persona voice + limits
    voice_backend: Optional[LLMBackend] = None  # None → reuse ego_backend
    max_corrections: int = 2          # EGO⇄SUPEREGO retry budget
    hooks: Optional[Hooks] = None     # None → no interception
