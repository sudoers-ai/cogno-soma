"""The stage contract :class:`~cogno_soma.Pipeline` calls — structural, so a host can inject its own.

``Pipeline(noumeno=…, ner=…, id_stage=…, ego=…, superego=…)`` has always promised that any stage
may be swapped for a custom implementation (a cheaper NER, a cached NOUMENO, a test double) "as long
as it matches the cogno-anima stage signature" (``docs/HOST_INTEGRATION.md`` §6). The five
parameters were nevertheless typed with the CONCRETE cogno-anima classes, so the promise held at
runtime and was refused by the type checker: a double written to exactly the documented shape was
an ``arg-type`` error at every construction site, one per stage, and a host type-checking its own
harness carried those errors as a permanent floor that no correct double could clear.

These are the shapes the Pipeline actually CALLS, member by member:

* **NOUMENO and NER** — :class:`cogno_anima.BaseStage`, REUSED and not copied. It already exists for
  exactly these two stages (``process(ctx, llm)``; cogno-anima's own suite pins that ``Noumeno`` and
  ``IntentAnalyzer`` satisfy it), and a second protocol here would be a second definition of a
  contract the anima owns.
* **ID, EGO, SUPEREGO** — their call shapes are not ``BaseStage``'s (the ID takes the embedder, the
  EGO the dispatcher and the execution prompt, the SUPEREGO four operations), so they are declared
  here, beside their one caller.

Every protocol carries ``name: str`` because ``BaseStage`` does: the anima's rule is that every stage
of the pipeline has one, and every anima stage does. The Pipeline itself never reads it.

``SuperegoStageProtocol._blocked_response`` is private-named in cogno-anima and called by the
Pipeline anyway (the PII-CRITICAL terminal reply, with the host's ``block_message``), so it is part
of the contract. Leaving it out would let a double satisfy the type and then fail on the one path a
PII-CRITICAL turn takes — which is also the path a scripted double is least likely to exercise.

All three are ``runtime_checkable``, like ``BaseStage``, so a host can probe its doubles with
``isinstance`` in a test. That probe checks that the members EXIST, not their signatures: the
signature check is the type checker's. The Pipeline annotates its own defaults with these
protocols, so ``mypy cogno_soma`` fails the day a cogno-anima stage stops matching them — the
contract is checked from both ends, the stage the anima ships and the double a host writes.

Types only: nothing here changes what the Pipeline does at runtime.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from cogno_anima.tools import ToolDispatcher
from cogno_anima.types import PipelineContext, ScopeCheckResult, SuperegoResult
from cogno_synapse import Embedder, LLMBackend


@runtime_checkable
class IDStageProtocol(Protocol):
    """The ID as the Pipeline calls it: routing and continuity over the embedder, no LLM.

    Satisfied by ``cogno_anima.stages.id.IDStage``.
    """

    name: str

    async def process(self, ctx: PipelineContext, embedder: Embedder) -> PipelineContext:
        """Populate ``ctx.id_result`` and return the same context."""


@runtime_checkable
class EgoStageProtocol(Protocol):
    """The EGO as the Pipeline calls it: the executor, once per attempt of the correction loop.

    Satisfied by ``cogno_anima.stages.ego.EgoStage``.
    """

    name: str

    async def process(
        self,
        ctx: PipelineContext,
        backend: LLMBackend,
        dispatcher: ToolDispatcher,
        *,
        system_prompt: str,
    ) -> PipelineContext:
        """Populate ``ctx.ego_result`` and return the same context."""


@runtime_checkable
class SuperegoStageProtocol(Protocol):
    """The SUPEREGO as the Pipeline calls it: scope guard, judge, voice and the PII-CRITICAL reply.

    Satisfied by ``cogno_anima.stages.superego.SuperegoStage``.
    """

    name: str

    async def check_input_scope(
        self, ctx: PipelineContext, backend: LLMBackend, *, scope_prompt: str,
    ) -> ScopeCheckResult:
        """The cheap ALLOW/BLOCK relevance guard, before the EGO."""

    async def evaluate(
        self, ctx: PipelineContext, backend: LLMBackend, *, limits_prompt: str,
    ) -> SuperegoResult:
        """The judge: approve the EGO's execution or return a critique."""

    async def voice(
        self, ctx: PipelineContext, backend: LLMBackend, *, voice_prompt: str,
    ) -> SuperegoResult:
        """Write the reply the contact reads."""

    def _blocked_response(
        self, ctx: PipelineContext, *, block_message: Optional[str] = None,
    ) -> SuperegoResult:
        """The terminal reply of a PII-CRITICAL turn (``block_message`` is the host's text)."""
