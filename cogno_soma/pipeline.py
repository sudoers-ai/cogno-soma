"""``Pipeline`` — the reference orchestrator, promoted to a shippable lib.

This is the "host glue" the ``cogno-anima`` library deliberately does NOT ship:
control flow (stage sequence + routing + the EGO⇄SUPEREGO correction loop + the
PII/scope/handoff gates + the atomicity and interception seams). It is the keystone
that imports the cognitive stages and wires them end-to-end — infra-agnostic: the
host injects backends, the dispatcher, and the prompt strings; soma never touches a
DB, an MCP client, billing, or a persona store.

    NOUMENO → NER → ID → [PII gate] → [scope gate] → EGO ⇄ SUPEREGO(judge) →
    SUPEREGO(voice) → (drift is computed inside the ID/SUPEREGO stages)

Compared to the parent ``PipelineRunner`` (1.6k lines soldered to CoreDB / MCP /
OpenTelemetry / RBAC / BudgetGuard), this keeps only the orchestration: memory,
safety, audit and atomicity become host ``Hooks``; persona selection and metering
stay at the host. The single-turn ``run_turn`` is stateless across turns (all
cross-turn state rides in ``ctx.metadata["id_state"]``); ``SessionRunner`` threads
that state for a multi-turn session.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Optional

from cogno_anima.stages.ego import EgoStage
from cogno_anima.stages.id import IDStage
from cogno_anima.stages.ner import IntentAnalyzer
from cogno_anima.stages.noumeno import Noumeno
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.tools import ToolDispatcher
from cogno_anima.types import PipelineContext, StageMetrics, SuperegoResult
from cogno_synapse import Embedder

from cogno_soma.config import TurnConfig
from cogno_soma.errors import StopPipeline
from cogno_soma.hooks import Hooks, HookFn

logger = logging.getLogger(__name__)


def _zero_metrics(stage: str = "superego_voice") -> StageMetrics:
    """A no-cost metrics row for a synthesized (non-LLM) terminal response."""
    return StageMetrics(stage=stage, elapsed_ms=0.0, tokens_in=0, tokens_out=0, model="none")


class Pipeline:
    """NOUMENO → NER → ID → [guard] → EGO ⇄ SUPEREGO(judge) → SUPEREGO(voice)."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        prompts_dir: Optional[Path] = None,
        slangs=None,
        noumeno: Optional[Noumeno] = None,
        ner: Optional[IntentAnalyzer] = None,
        id_stage: Optional[IDStage] = None,
        ego: Optional[EgoStage] = None,
        superego: Optional[SuperegoStage] = None,
    ) -> None:
        # Stages default to the cogno-anima implementations; a host (or a test) may
        # inject its own — e.g. a cheaper NER, or a fake stage with no LLM.
        self._embedder = embedder
        self._noumeno = noumeno or Noumeno(embedder=embedder, prompts_dir=prompts_dir, slangs=slangs or {})
        self._ner = ner or IntentAnalyzer(prompts_dir=prompts_dir)
        self._id = id_stage or IDStage()
        self._ego = ego or EgoStage()
        self._superego = superego or SuperegoStage()

    async def run_turn(
        self,
        ctx: PipelineContext,
        cfg: TurnConfig,
        *,
        dispatcher: ToolDispatcher,
    ) -> PipelineContext:
        """Run one full turn. Returns the same (mutated) ``ctx``.

        A host hook may raise :class:`StopPipeline` at any fire point to halt early;
        the partial ``ctx`` is returned with ``stop_reason`` set (and a terminal
        ``superego_result`` when the hook supplied a ``response``).
        """
        hooks = cfg.hooks or Hooks()
        voice_backend = cfg.voice_backend or cfg.ego_backend
        try:
            await self._fire(hooks.before_turn, ctx)

            # ── perception + routing ──────────────────────────────────
            ctx = await self._noumeno.process(ctx, cfg.noumeno_backend or cfg.gen_backend)
            await self._fire(hooks.after_noumeno, ctx)
            ctx = await self._ner.process(ctx, cfg.ner_backend or cfg.gen_backend)
            await self._fire(hooks.after_ner, ctx)
            ctx = await self._id.process(ctx, self._embedder)
            await self._fire(hooks.after_id, ctx)

            # ── PII-CRITICAL gate (from ID) ───────────────────────────
            if ctx.id_result and ctx.id_result.blocked:
                ctx.superego_result = self._superego._blocked_response(ctx)
                ctx.stop_reason = "pii_blocked"
                logger.debug("turn_blocked stop_reason=pii_blocked")
                return await self._finish(ctx, hooks)

            # ── early scope guard (optional, cheap ALLOW/BLOCK) ───────
            if cfg.scope_prompt:
                scope = await self._superego.check_input_scope(
                    ctx, cfg.scope_backend or cfg.gen_backend, scope_prompt=cfg.scope_prompt)
                ctx.retry_metrics.append(scope.metrics)
                if scope.blocked:
                    ctx.superego_result = SuperegoResult(
                        response=scope.refusal_message, blocked=True, metrics=_zero_metrics())
                    ctx.stop_reason = "scope_blocked"
                    logger.debug("turn_blocked stop_reason=scope_blocked")
                    return await self._finish(ctx, hooks)

            # ── EGO route: execute + correction loop ──────────────────
            # A confirmed action (gate-B completion) MUST run through the EGO to be executed,
            # even when the user's bare "sim" would otherwise route to the SUPEREGO — otherwise
            # the approved call is never dispatched (and its side effects never fire).
            force_ego = bool(ctx.metadata.get("ego_confirmed") and ctx.metadata.get("ego_confirmed_calls"))
            if force_ego or (ctx.id_result and ctx.id_result.triad_route == "EGO"):
                judge = await self._run_ego_loop(ctx, cfg, dispatcher, hooks)
                await self._fire(hooks.after_ego, ctx)
                if judge is not None and not judge.approved:
                    ego = ctx.ego_result
                    # A COMMITTED side effect requires the mutating call to have SUCCEEDED
                    # (ok) — a mutation that FAILED (e.g. the confirmed slot was taken between
                    # propose and commit) changed nothing, so it is not a commit to misreport.
                    committed = ego is not None and any(
                        t.side_effect and t.ok for t in ego.tools_executed)
                    if ego is not None and not committed:
                        # The judge rejected, but nothing was actually COMMITTED — the EGO only
                        # READ, or every mutating call failed. Rather than dead-end in a human
                        # handoff, keep the conversation alive: signal needs_clarification and
                        # fall through to voice() below, which grounds a truthful continuation
                        # in the trace (e.g. "I found your appointment — change it to 11:00?"
                        # or "that slot was just taken — want another?"). Fail-closed is
                        # preserved: a SUCCESSFUL-but-rejected mutation still hands off (never
                        # voice an unverified commit as done). The HOST owns the escalation
                        # policy on this signal (force a real handoff after N).
                        ctx.stop_reason = "needs_clarification"
                        # The voice must know the execution was rejected — otherwise it only
                        # sees the successful reads and the upbeat draft and happily narrates
                        # the goal as done ("Prontinho! confirmados") while the DB never
                        # changed. Hand it the judge's critique so it grounds a truthful
                        # continuation instead of claiming completion (anima renders this as
                        # a hard rule in the voice prompt).
                        ctx.metadata["voice_correction"] = {
                            "reason": (judge.critique or "").strip() or
                                      "the execution did not accomplish the user's goal",
                        }
                        logger.debug("turn_clarify stop_reason=needs_clarification")
                        # no on_commit: nothing was committed
                    else:
                        ctx.needs_handoff = True
                        ctx.stop_reason = "human_handoff"
                        logger.debug("turn_handoff stop_reason=human_handoff")
                        return await self._finish(ctx, hooks)
                else:
                    await self._fire(hooks.on_commit, ctx)

            # ── voice (writes the final response; EGO and non-task paths) ──
            ctx.superego_result = await self._superego.voice(
                ctx, voice_backend, voice_prompt=cfg.voice_prompt)
            await self._fire(hooks.after_superego, ctx)
            return await self._finish(ctx, hooks)

        except StopPipeline as stop:
            ctx.stop_reason = stop.reason
            if stop.response is not None:
                ctx.superego_result = SuperegoResult(
                    response=stop.response, blocked=stop.blocked, metrics=_zero_metrics())
            logger.debug("turn_stopped stop_reason=%s", stop.reason)
            return ctx

    # ── internals ─────────────────────────────────────────────────────
    async def _run_ego_loop(self, ctx, cfg: TurnConfig, dispatcher, hooks: Hooks):
        """The EGO⇄SUPEREGO correction loop; returns the last judge result."""
        attempt = 1
        judge = None
        while True:
            # Complexity escalation: after the ID, a hard task can bump the EGO onto a stronger
            # model for this turn (the host's ladder policy); None keeps the configured backend.
            ego_backend = (cfg.escalate(ctx, "ego") if cfg.escalate else None) or cfg.ego_backend
            ctx = await self._ego.process(ctx, ego_backend, dispatcher, system_prompt=cfg.ego_prompt)
            judge = await self._judge(ctx, cfg)
            if judge.approved or attempt >= cfg.max_corrections:
                break
            # rejected → this EGO attempt becomes retry history; feed the critique back
            if ctx.ego_result:
                ctx.retry_metrics.append(ctx.ego_result.metrics)
            await self._fire(hooks.on_rollback, ctx)
            ctx.metadata["ego_correction"] = {"reason": judge.critique, "attempt": attempt + 1}
            # Gate-B replay is once-only: the confirmed calls were already executed on this
            # attempt (their outcome is in the trace) — a correction re-run must NOT replay
            # them, or a rejected-but-successful call would execute twice (double booking).
            ctx.metadata.pop("ego_confirmed_calls", None)
            attempt += 1
        return judge

    async def _judge(self, ctx, cfg: TurnConfig):
        """One judge verdict, optionally two-tier (``judge_fast_backend``).

        The fast judge screens every attempt; only its REJECTIONS escalate to the
        strong judge, whose verdict is authoritative. A fast approve is final (the
        cost bet), and fail-CLOSED is preserved: nothing gets approved that neither
        judge approved. Every call's metrics land in ``retry_metrics`` (the judge is
        never the "main" superego — voice is), distinguished by the model field.
        """
        strong = cfg.judge_backend or cfg.gen_backend
        fast = cfg.judge_fast_backend
        if fast is not None and fast is not strong:
            verdict = await self._superego.evaluate(ctx, fast, limits_prompt=cfg.limits_prompt)
            ctx.retry_metrics.append(verdict.metrics)
            if verdict.approved:
                return verdict
            logger.debug("judge_escalated fast_model=%s", getattr(fast, "model", "unknown"))
        verdict = await self._superego.evaluate(ctx, strong, limits_prompt=cfg.limits_prompt)
        ctx.retry_metrics.append(verdict.metrics)
        return verdict

    async def _finish(self, ctx, hooks: Hooks) -> PipelineContext:
        await self._fire(hooks.after_turn, ctx)
        return ctx

    @staticmethod
    async def _fire(fn: Optional[HookFn], ctx: PipelineContext) -> None:
        """Invoke a hook (if set); await it when it returns an awaitable."""
        if fn is None:
            return
        result = fn(ctx)
        if inspect.isawaitable(result):
            await result
