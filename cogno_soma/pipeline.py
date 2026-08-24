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
import json
import logging
from pathlib import Path
from typing import Optional

from cogno_anima import metakeys as mk
from cogno_anima.types import committed_this_turn
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
# Judge critiques are model prose; the whole point is to make a red check explainable, and a
# couple of sentences does that. Unbounded, they ride into whatever the host persists.
_CRITIQUE_CHARS = 400

# Same bound, same reason, for what each attempt EXECUTED: a tool result is model-facing prose
# and the arguments can embed user text, so both are stringified and cut before they ride in
# persisted metadata. Small on purpose — the question this answers is "which rows did the
# rejected attempt touch", not "what did the tool say in full".
_TOOL_RESULT_CHARS = 200
_TOOL_ARGS_CHARS = 160
# Entries per attempt. The per-field cuts bound each entry; nothing bounded the LIST — and the
# real ceiling is not the default max_steps: premium plans inject ego_max_steps=25, calls per
# step are uncapped, and blocked/duplicate calls are recorded too. Review measured the worst
# case at ~375 entries (~160 KB) for ONE turn. Enough survives to answer "which rows"; the
# overflow is counted, never silent.
_TOOLS_PER_ATTEMPT = 30


def _cut(text: str, limit: int) -> str:
    """Truncate WITH a marker. A cut JSON string without one renders as broken-but-plausible
    JSON presented as if complete — the reader cannot tell truncation from corruption."""
    return text if len(text) <= limit else text[:limit] + "…"


def _outcome(exec_) -> str:
    """What the call came back with. `result or error` alone drops the error TAXONOMY whenever
    both are set — anima's synthesized blocked/duplicate executions carry error="blocked_retry"
    /"duplicate" PLUS model-facing prose — so both travel when they say different things."""
    result = (exec_.result or "").strip()
    error = (exec_.error or "").strip()
    if error and error not in result:
        return f"{error}: {result}" if result else error
    return result


# ── per-call identity: the orchestrator is the only layer that SEQUENCES the stages ──
#
# `PipelineContext.stage_metrics` is composed as "the populated canonical slots, then
# `retry_metrics`", and each canonical slot holds the value that SURVIVED — so the list is not
# call order, and reading it as one attributes a retried turn backwards (the `ego` among the
# canonical slots is the LAST attempt; the one further down is the first). Nobody downstream
# can recover the order by inference, because the information is not there. Here it is.
#
# `prompt_sha` labels the CONFIGURATION a call ran under. The digest is NOT computed here: by
# the time a prompt reaches this layer it has been RENDERED, and the host substitutes the
# contact's name and e-mail into the system/scope/limits/voice slots before handing them over
# (`cogno_host.persona.render_slot` → `identity_context`). Digesting that would produce a
# different sha per CONTACT — one row per conversation in a content-addressed store, which is
# the opposite of the point — and any text stored behind it would carry a name and an e-mail
# into a table the identity purge does not know about.
#
# So the host digests its TEMPLATES (where `{identity_label}` is literal and stable) and hands
# the map down as `mk.PROMPT_SHAS`; this layer only says which slot each call used. The layer
# that AUTHORS a text owns its identity; this one sequences.
#
# First cut hashed the rendered text right here and claimed it was PII-free by construction.
# It was PII by construction — found by asking a peer to refute it before the store existed,
# which is the only reason it cost a revert instead of a migration.
_SEQ_KEY = "_call_seq"          # private to this module; popped in `_finish`


def _sha_for(ctx, kind: str) -> str:
    """The host's digest for this prompt slot, or "" when the host supplied none — an empty
    sha says "unlabelled", never "no prompt". Total: a malformed map degrades to unlabelled
    rather than killing a turn."""
    shas = ctx.metadata.get(mk.PROMPT_SHAS)
    if not isinstance(shas, dict):
        return ""
    value = shas.get(kind)
    return value if isinstance(value, str) else ""


def _judge_sha(ctx) -> str:
    """The judge's slot, plus the flag that swaps its whole criteria set
    (`JUDGE_CONVERSATIONAL`: a persona with no tool executes nothing by design, so the
    execution criteria are unsatisfiable and the judge rejected 100% of turns). Same limits
    text under different criteria is a different configuration, so the flag rides in the
    label — appended, since the host cannot know it at template time."""
    base = _sha_for(ctx, "judge")
    if not base:
        return ""
    return f"{base}+conv" if ctx.metadata.get(mk.JUDGE_CONVERSATIONAL) else base


def _stamp(ctx, metrics, *, attempt: int = 0, prompt: str = "") -> None:
    """Mark one executed stage with its place in the turn. Never raises: telemetry must not be
    the reason a turn dies, and an unstamped metric degrades to the old behaviour exactly."""
    try:
        seq = int(ctx.metadata.get(_SEQ_KEY, 0)) + 1
        ctx.metadata[_SEQ_KEY] = seq
        if metrics is None:
            return
        metrics.seq = seq
        metrics.attempt = attempt
        metrics.prompt_sha = prompt
    except Exception:  # noqa: BLE001 — see above
        logger.debug("stage_stamp_failed", exc_info=True)


def _attempt_tools(ego_result) -> dict:
    """One attempt's executions: the POLICY bit, then the bounded display list.

    `committed` is deliberately computed FIRST, outside the try, over the FULL list: it is the
    only field this turn's routing depends on (see `committed_this_turn`), so it must not inherit
    the display path's failure modes — not the per-attempt cap, not the truncation, not a
    `json.dumps` that raises. Reading `t.side_effect`/`t.ok` cannot raise; serializing the
    arguments can.

    The rest is wrapped whole because this runs inside the PRODUCTION correction loop and
    telemetry must never be the reason a turn dies. `default=str` closes only one of
    json.dumps's raise paths — a `__str__` that itself raises unwinds too. A degraded entry
    beats a dead turn."""
    execs = list(ego_result.tools_executed) if ego_result else []
    # Per-attempt, for the reader of the evidence. The TURN's answer comes from
    # `committed_this_turn(ctx)` over `turn_executions` — one definition, in anima.
    committed = any(t.side_effect and t.ok for t in execs)
    try:
        entry: dict = {"committed": committed, "tools": [
            {"tool": t.tool,
             "args": _cut(json.dumps(t.arguments or {}, ensure_ascii=False, default=str),
                          _TOOL_ARGS_CHARS),
             "ok": bool(t.ok), "side_effect": bool(t.side_effect),
             "result": _cut(_outcome(t), _TOOL_RESULT_CHARS)}
            for t in execs[:_TOOLS_PER_ATTEMPT]
        ]}
        if len(execs) > _TOOLS_PER_ATTEMPT:
            entry["tools_dropped"] = len(execs) - _TOOLS_PER_ATTEMPT
        return entry
    except Exception as exc:  # noqa: BLE001 — degraded telemetry, never a dead turn
        # The display list is what degrades. `committed` survives — a turn must never route
        # on a missing bit because a tool argument would not serialize.
        return {"committed": committed, "tools": [], "tools_error": type(exc).__name__}


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
            _stamp(ctx, ctx.noumeno.metrics if ctx.noumeno else None)
            await self._fire(hooks.after_noumeno, ctx)
            ctx = await self._ner.process(ctx, cfg.ner_backend or cfg.gen_backend)
            _stamp(ctx, ctx.intent.metrics if ctx.intent else None)
            await self._fire(hooks.after_ner, ctx)
            ctx = await self._id.process(ctx, self._embedder)
            _stamp(ctx, ctx.id_result.metrics if ctx.id_result else None)
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
                _stamp(ctx, scope.metrics, prompt=_sha_for(ctx, "scope"))
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
            force_ego = bool(ctx.metadata.get(mk.EGO_CONFIRMED) and ctx.metadata.get(mk.EGO_CONFIRMED_CALLS))
            if force_ego or (ctx.id_result and ctx.id_result.triad_route == "EGO"):
                judge = await self._run_ego_loop(ctx, cfg, dispatcher, hooks)
                await self._fire(hooks.after_ego, ctx)
                if judge is not None and not judge.approved:
                    ego = ctx.ego_result
                    # ACROSS EVERY ATTEMPT, not just the last one — `ctx.ego_result` is
                    # replaced on each retry, and a write the judge refused does not un-happen
                    # because a later attempt only read. The definition of a commit lives in
                    # anima, beside the type it reads: three repos must agree on it, and this
                    # orchestrator is only one of six places that were getting it wrong.
                    committed = committed_this_turn(ctx)
                    if ego is not None and not committed:
                        # The judge rejected, but nothing was actually COMMITTED — the EGO only
                        # READ, or every mutating call failed. Rather than dead-end in a human
                        # handoff, keep the conversation alive: signal needs_clarification and
                        # fall through to voice() below, which grounds a truthful continuation
                        # in the trace (e.g. "I found your appointment — change it to 11:00?"
                        # or "that slot was just taken — want another?"). Fail-closed is
                        # preserved: a SUCCESSFUL-but-rejected mutation still hands off (never
                        # voice an unverified commit as done) — and since `committed_this_turn`
                        # reads every attempt, the "NOTHING was committed" the voice renders
                        # below as a HARD RULE is now TRUE of the whole turn, not just of the
                        # attempt that happened to be last. The HOST owns the escalation
                        # policy on this signal (force a real handoff after N).
                        ctx.stop_reason = "needs_clarification"
                        # The voice must know the execution was rejected — otherwise it only
                        # sees the successful reads and the upbeat draft and happily narrates
                        # the goal as done ("Prontinho! confirmados") while the DB never
                        # changed. Hand it the judge's critique so it grounds a truthful
                        # continuation instead of claiming completion (anima renders this as
                        # a hard rule in the voice prompt).
                        # Two different rejections wear the same signal. When the EGO ran
                        # tools, the verdict is about an ACTION that fell short and the voice
                        # grounds a continuation in the trace. When NOTHING ran (a persona
                        # with no tools — a seller, an SDR), there is no trace to ground in:
                        # the draft is an unverified CLAIM, and re-voicing it ships exactly
                        # what review just refused. Name the kind so the voice can tell them
                        # apart — and so the HOST can act on an unverified turn (it could
                        # previously only infer `needs_clarification`, which a legitimate
                        # mid-flow question raises too).
                        # …e POR TURNO pela mesma razão do `committed` acima: a tentativa
                        # sobrevivente pode voltar com trace VAZIO (a re-chamada dela é
                        # bloqueada por duplicata — rotineiro), e ler isso como "nada executou"
                        # marcaria como AFIRMAÇÃO NÃO VERIFICADA um turno que reuniu dados de
                        # verdade: a voz DESCARTA o rascunho, o host mede como unverified_claim
                        # e, com a flag ligada, encerra num humano.
                        ran = ctx.turn_executions or ego.tools_executed
                        kind = "unverified_claim" if not ran else "not_executed"
                        ctx.metadata[mk.VOICE_CORRECTION] = {
                            "reason": (judge.critique or "").strip() or
                                      "the execution did not accomplish the user's goal",
                            "kind": kind,
                        }
                        logger.debug("turn_clarify stop_reason=needs_clarification kind=%s", kind)
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
            _stamp(ctx, ctx.superego_result.metrics if ctx.superego_result else None,
                   prompt=_sha_for(ctx, "voice"))
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
        # The correction budget is per-plan: the host's BudgetGuard injects ``plan_limits`` into
        # ctx.metadata, so a Free tenant caps at fewer EGO attempts than a paid one (was ignored —
        # the loop always used cfg.max_corrections). Falls back to cfg.max_corrections when no plan
        # is present (host without a subscription store); clamped to >= 1 (always one attempt).
        max_corrections = max(1, int(
            (ctx.metadata.get("plan_limits") or {}).get("max_self_corrections") or cfg.max_corrections
        ))
        while True:
            # Complexity escalation: after the ID, a hard task can bump the EGO onto a stronger
            # model for this turn (the host's ladder policy); None keeps the configured backend.
            ego_backend = (cfg.escalate(ctx, "ego") if cfg.escalate else None) or cfg.ego_backend
            ctx = await self._ego.process(ctx, ego_backend, dispatcher, system_prompt=cfg.ego_prompt)
            _stamp(ctx, ctx.ego_result.metrics if ctx.ego_result else None, attempt=attempt,
                   prompt=_sha_for(ctx, "ego"))
            # Gate B held a destructive call → a PROPOSE turn: the action was deliberately NOT
            # executed (the host runs the confirmation UX next; ``_awaiting_confirmation`` is set
            # independently in the assembler). There is nothing to judge-and-retry — the booking is
            # intentionally incomplete — so skip the judge: it would false-reject "booking not
            # completed" and burn a wasteful EGO retry on EVERY confirmation turn (measured:
            # the gpt-4o-mini judge rejects the [PENDING CONFIRMATION] hold 3/3). Voice grounds it.
            # O que ESTA tentativa executou entra no turno agora — antes de qualquer saída do
            # laço. O `break` do gate-B abaixo pulava o registro junto com o juiz, e uma
            # tentativa que EXECUTOU escritas e só então segurou uma chamada destrutiva não
            # deixava rastro nenhum: o ledger que este changeset promoveu a "a autoridade"
            # tinha um buraco exatamente onde havia escrita.
            if ctx.ego_result:
                ctx.turn_executions.extend(ctx.ego_result.tools_executed)
            if ctx.ego_result and ctx.ego_result.pending_confirmation:
                judge = SuperegoResult(approved=True, metrics=_zero_metrics())
                break
            judge = await self._judge(ctx, cfg, attempt=attempt)
            # Record WHY, not just how many. The rejected attempts' critiques were fed into
            # EGO_CORRECTION.reason (overwritten each attempt) and then dropped, so after the
            # turn the only surviving fact was the count — and a red bench check could not be
            # explained without attaching a debugger to the judge. Truncated: this rides in
            # metadata the host persists, and a full critique per attempt is unbounded text.
            # …and record what this attempt EXECUTED, because the verdict alone hides the
            # dangerous half. `ctx = await self._ego.process(...)` REPLACES ctx.ego_result on
            # every retry, so a write made by a rejected attempt vanished from everything
            # downstream — the voice, the after-turn hooks, the bench evidence — while the
            # write itself had COMMITTED (nothing here rolls back). Measured long before this
            # fix (doctor-notify, 2026-08): attempt 1 wrongly confirmed all three rows, the
            # judge rightly rejected it, attempt 2 truthfully said "no pendings", and the turn
            # shipped hiding three commits. The critiques became data in #30; this closes the
            # other eye. Rollback / judge-aware voicing remain deliberately OUT of scope: this
            # makes the writes VISIBLE, it does not undo them.
            # `setdefault(...).append(...)` mata o turno se o host tiver semeado a chave com
            # algo que não é lista (AttributeError sai do laço de correção de PRODUÇÃO, e só
            # StopPipeline é capturada). Telemetria nunca pode ser o motivo de perder um turno.
            ledger = ctx.metadata.get(mk.JUDGE_ATTEMPTS)
            if not isinstance(ledger, list):
                ledger = []
                ctx.metadata[mk.JUDGE_ATTEMPTS] = ledger
            ledger.append({
                "attempt": attempt,
                "approved": bool(judge.approved),
                "critique": (judge.critique or "")[:_CRITIQUE_CHARS],
                **_attempt_tools(ctx.ego_result),
            })
            if judge.approved or attempt >= max_corrections:
                break
            # rejected → this EGO attempt becomes retry history; feed the critique back
            if ctx.ego_result:
                ctx.retry_metrics.append(ctx.ego_result.metrics)
            await self._fire(hooks.on_rollback, ctx)
            ctx.metadata[mk.EGO_CORRECTION] = {"reason": judge.critique, "attempt": attempt + 1}
            # Gate-B replay is once-only: the confirmed calls were already executed on this
            # attempt (their outcome is in the trace) — a correction re-run must NOT replay
            # them, or a rejected-but-successful call would execute twice (double booking).
            ctx.metadata.pop(mk.EGO_CONFIRMED_CALLS, None)
            attempt += 1
        # Record the outcome so it can be COUNTED. Only rejections were ever observable (logged
        # at WARNING; approvals go to INFO, which a deployment's handlers may drop), so "no
        # approvals in the log" read identically to "the judge approves nothing" — the host had
        # no denominator and no way to tell a healthy judge from one rejecting 100% of turns.
        if judge is not None:
            ctx.metadata[mk.JUDGE_VERDICT] = {"approved": bool(judge.approved),
                                              "attempts": attempt}
        return judge

    async def _judge(self, ctx, cfg: TurnConfig, *, attempt: int = 0):
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
            _stamp(ctx, verdict.metrics, attempt=attempt, prompt=_judge_sha(ctx))
            ctx.retry_metrics.append(verdict.metrics)
            if verdict.approved:
                return verdict
            logger.debug("judge_escalated fast_model=%s", getattr(fast, "model", "unknown"))
        verdict = await self._superego.evaluate(ctx, strong, limits_prompt=cfg.limits_prompt)
        _stamp(ctx, verdict.metrics, attempt=attempt, prompt=_judge_sha(ctx))
        ctx.retry_metrics.append(verdict.metrics)
        return verdict

    async def _finish(self, ctx, hooks: Hooks) -> PipelineContext:
        # The stamp counter is bookkeeping, not a contract: every exit goes through here, so
        # it never reaches the metadata a host persists (the seq itself lives on the metrics).
        ctx.metadata.pop(_SEQ_KEY, None)
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
