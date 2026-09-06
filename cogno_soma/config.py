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

**Per-stage model routing.** Each JSON stage can override ``gen_backend`` individually via
``noumeno_backend`` / ``ner_backend`` / ``scope_backend`` / ``judge_backend`` (all default None →
``gen_backend``). This lets a host give every step its own model — e.g. a cheap NOUMENO/NER, a
strong judge, a mid-tier voicer — instead of one shared JSON backend. Unset → identical to before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from cogno_synapse import LLMBackend

from cogno_soma.hooks import Hooks

if TYPE_CHECKING:
    from cogno_anima.types import IntentResult, NoumenoResult, PipelineContext

# Complexity-based escalation: given the running turn (its ``id_result.complexity``) and a stage
# name, return a stronger backend to run that stage on, or None to keep the configured one. The
# host owns the policy (the model ladder + the plan gate); soma just consults it after the ID.
EscalateFn = Callable[["PipelineContext", str], Optional[LLMBackend]]


@dataclass
class TurnConfig:
    gen_backend: LLMBackend           # NOUMENO / NER / scope / judge (JSON-capable)
    ego_backend: LLMBackend           # EGO executor (FC or text fallback)
    ego_prompt: str                   # persona execution prompt (SUPEREGO is the voicer)
    scope_prompt: str = ""            # empty → skip the cheap pre-EGO scope guard
    limits_prompt: str = ""           # judge criterion: persona limits
    voice_prompt: str = ""            # voicer: persona voice + limits
    voice_backend: Optional[LLMBackend] = None  # None → reuse ego_backend
    # PII-CRITICAL block reply in the persona/tenant language. "" → the core's English
    # _BLOCKED_FALLBACK — which is what a contact receives when the host forgets this field,
    # so a deployment whose contacts are not English speakers should always set it.
    #
    # SAFETY INVARIANT — read before filling this with anything dynamic: the blocked branch
    # returns BEFORE voice(), and the outgoing-PII provenance net runs only inside voice(),
    # so this is the ONLY outgoing message that never passes the PII detector. It is safe
    # while it stays a static per-tenant constant. The day a host interpolates turn data
    # into it ("we can't process your CPF 123..."), it becomes the one leak path with no
    # net underneath. Keep it constant.
    block_message: str = ""
    max_corrections: int = 2          # EGO⇄SUPEREGO retry budget
    hooks: Optional[Hooks] = None     # None → no interception
    # Per-stage overrides for the JSON stages (None → gen_backend). Let each step run its own model.
    noumeno_backend: Optional[LLMBackend] = None  # None → gen_backend
    ner_backend: Optional[LLMBackend] = None      # None → gen_backend
    scope_backend: Optional[LLMBackend] = None    # None → gen_backend
    judge_backend: Optional[LLMBackend] = None    # None → gen_backend
    # Two-tier judge: a cheap screening judge runs first on every EGO attempt; only its
    # REJECTIONS are re-judged by the strong judge (judge_backend), whose verdict is
    # authoritative. A fast approve is final — that is the cost bet: the happy path
    # (most turns) costs the cheap model, and the strong model is paid only on rejects.
    # None → single-tier (judge_backend or gen_backend), identical to before.
    judge_fast_backend: Optional[LLMBackend] = None
    # ── Precomputed perception (a turn nobody spoke) ─────────────────────────────────────
    # Set either of these and the pipeline SEATS it instead of running that stage: no backend
    # is consulted, no embedder is called, no tokens are spent. It exists for the turn the
    # AGENT opens (see ``cogno_soma.opening``), where the text the host holds is its own
    # directive and running the two stages that exist to understand a PERSON over it is not
    # wasteful but wrong — the NER reads OUR words for PII and blocks the turn, the ID files
    # them as the CONTACT's goal, and the NOUMENO measures drift and subject continuity
    # against a sentence nobody wrote.
    #
    # Unlike everything else on this dataclass these are PER TURN, and that is deliberate: the
    # host already builds each turn's config with ``dataclasses.replace``, which is where the
    # opening's ``scope_prompt=""`` and voice section are set too, so the opening arrives as
    # one substitution instead of a second channel into ``run_turn``.
    #
    # The fields that are FACTS about the turn are not read from here: ``original``,
    # ``rewritten`` and ``language`` come from ``ctx.user_input``/``ctx.force_language``, and
    # ``langue`` from the seated NOUMENO, exactly as the real stages take them. A stand-in can
    # state decisions; it cannot claim the turn carried text it did not.
    noumeno_result: Optional["NoumenoResult"] = None   # None → run the NOUMENO stage
    intent_result: Optional["IntentResult"] = None     # None → run the NER stage
    # Host escalation policy consulted AFTER the ID computes complexity: a hard task can bump the
    # EGO onto a stronger model for this turn. None → no escalation (the configured backends run).
    escalate: Optional[EscalateFn] = None
