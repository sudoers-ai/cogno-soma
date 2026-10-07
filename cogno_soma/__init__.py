"""cogno-soma — the reference orchestrator for the Cogno cognitive pipeline.

Promotes the stages of ``cogno-anima`` (NOUMENO → NER → ID → EGO ⇄ SUPEREGO →
Drift) into an end-to-end, infra-agnostic runner: control flow, the correction
loop, the PII/scope/handoff gates, an interception seam (:class:`Hooks` /
:class:`StopPipeline`) and atomicity callbacks. The host injects backends, the
dispatcher and the persona's prompt strings; soma never touches a DB, an MCP
client, billing or a persona store.
"""

from cogno_soma.config import TurnConfig
from cogno_soma.errors import SomaError, StopPipeline
from cogno_soma.hooks import HookFn, Hooks
from cogno_soma.opening import (OPENING_INTENT, OPENING_MODEL, opening_intent,
                                opening_noumeno, opening_perception)
from cogno_soma.pipeline import (EXTRA_PASS_ACTION, EXTRA_PASS_HELD_MESSAGE, EXTRA_PASS_READ,
                                 EXTRA_PASSES, JUDGE_SKIP_PREJUDGED_REPLAY, JUDGE_SKIPS,
                                 LEDGER_EGO_PROMPT_BLOCKS, LEDGER_EGO_PROMPT_BLOCKS_DROPPED,
                                 LEDGER_EGO_PROMPT_PATH, LEDGER_EGO_PROMPT_SHA,
                                 LEDGER_FAST_VERDICT_READ, LEDGER_VERDICT_READ,
                                 PREJUDGED_TEXT_SHA, Pipeline, STOP_JUDGE_EXHAUSTED, prejudged_digest)
from cogno_soma.session import (CARRIER_CONTEXT, CARRIER_CONTEXT_DATA, CONTEXT_CARRIERS,
                                CONTEXT_LAYER_SLUGS, CONTEXT_LAYERS_KEY,
                                CONTEXT_WINDOW_EXCHANGES, CONTEXT_WINDOW_GAP_SECONDS,
                                SessionRunner)
from cogno_soma.stages import EgoStageProtocol, IDStageProtocol, SuperegoStageProtocol
from cogno_soma.trace_cuts import (CUT_MARK, CUT_MARK_RE, TOOL_RESULT_CHARS,
                                   ToolResultLimitFn, cut_dropped, cut_with_mark,
                                   resolve_limit, was_cut)

__all__ = [
    "Pipeline",
    "SessionRunner",
    # The runner's verbatim window — how many exchanges, and how old the oldest may be. The
    # HOST writes rows into that transcript itself (a delivered message) and must know whether
    # its row is still in the window the model reads; a copy of the numbers on its side would
    # be a second definition of a contract it does not own.
    "CONTEXT_WINDOW_GAP_SECONDS",
    "CONTEXT_WINDOW_EXCHANGES",
    "TurnConfig",
    "Hooks",
    "HookFn",
    "StopPipeline",
    # The terminal signal for "the correction loop ran out and nothing was committed".
    # Exported because the HOST has to match on it (its proactive-delivery whitelist and
    # its escalation caps both key on `stop_reason`), and a string literal copied there
    # would be a second definition of a contract that has already been wrong once.
    "STOP_JUDGE_EXHAUSTED",
    # Which exception granted the one extra EGO pass (`judge_attempts[i]["extra_pass"]`).
    # Exported for the same reason: the HOST persists the ledger and closes the alphabet.
    "EXTRA_PASS_ACTION",
    "EXTRA_PASS_READ",
    "EXTRA_PASS_HELD_MESSAGE",
    "EXTRA_PASSES",
    # Why a ledger row records an attempt the judge did NOT read (`judge_attempts[i]["skipped"]`),
    # and the stamp that buys it: the HOST writes `PREJUDGED_TEXT_SHA` (with `prejudged_digest`,
    # the one digest) where it checks that the judge approved a held message, and persists the
    # ledger closing its alphabet from `JUDGE_SKIPS` — the same contract as `EXTRA_PASSES`.
    "JUDGE_SKIP_PREJUDGED_REPLAY",
    "JUDGE_SKIPS",
    # The ledger keys for HOW a judge verdict was read (`judge_attempts[i]["verdict_read"]`, and
    # `["fast_verdict_read"]` when a fast judge was escalated past). The VALUES are
    # `cogno_anima.VALID_VERDICT_READS`; a host persisting the ledger reads the keys from here.
    "LEDGER_VERDICT_READ",
    "LEDGER_FAST_VERDICT_READ",
    # The ledger keys for what each attempt's EXECUTOR PROMPT carried
    # (`judge_attempts[i]["ego_prompt_blocks"]`, `["ego_prompt_sha"]`, `["ego_prompt_path"]`).
    # The values are closed by `cogno_anima.EGO_PROMPT_BLOCKS` / `VALID_EGO_PROMPT_PATHS`; a
    # key is ABSENT when the attempt has nothing on record.
    "LEDGER_EGO_PROMPT_BLOCKS",
    "LEDGER_EGO_PROMPT_BLOCKS_DROPPED",
    "LEDGER_EGO_PROMPT_SHA",
    "LEDGER_EGO_PROMPT_PATH",
    # The layers `SessionRunner.run` composed into the turn's context, recorded by the
    # composer on `ctx.metadata[CONTEXT_LAYERS_KEY]` (per turn): slugs from
    # `CONTEXT_LAYER_SLUGS`, lengths, and which of `CONTEXT_CARRIERS` each travelled on.
    "CONTEXT_LAYERS_KEY",
    "CONTEXT_LAYER_SLUGS",
    "CONTEXT_CARRIERS",
    "CARRIER_CONTEXT",
    "CARRIER_CONTEXT_DATA",
    "PREJUDGED_TEXT_SHA",
    "prejudged_digest",
    "SomaError",
    # ── the stage contract ───────────────────────────────────────────────────────────────
    # What `Pipeline(id_stage=…, ego=…, superego=…)` accepts, structurally. Exported so a host
    # can hold its own doubles to it; NOUMENO and NER take `cogno_anima.BaseStage`, which the
    # anima owns and this package does not re-export (one definition, one import path).
    "IDStageProtocol",
    "EgoStageProtocol",
    "SuperegoStageProtocol",
    # ── the cut that leaves a mark ───────────────────────────────────────────────────────
    # Exported for the same reason as STOP_JUDGE_EXHAUSTED above: the HOST reads this ledger,
    # re-cuts it at its own ceiling and counts the cuts, so the mark's spelling and the ceiling
    # are a contract BETWEEN the two repos. A copy on the reader's side would be a second
    # definition — and the first one has already been wrong once, at 200 against 240, with the
    # reader's ceiling never biting and the writer's cut never saying it happened.
    "CUT_MARK",
    "CUT_MARK_RE",
    "TOOL_RESULT_CHARS",
    "ToolResultLimitFn",
    "cut_with_mark",
    "cut_dropped",
    "was_cut",
    "resolve_limit",
    # The turn the AGENT opens: perception stand-ins for a turn nobody spoke.
    "opening_perception",
    "opening_noumeno",
    "opening_intent",
    "OPENING_MODEL",
    "OPENING_INTENT",
]

__version__ = "0.1.0"
