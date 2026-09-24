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
from cogno_soma.pipeline import Pipeline, STOP_JUDGE_EXHAUSTED
from cogno_soma.session import SessionRunner
from cogno_soma.stages import EgoStageProtocol, IDStageProtocol, SuperegoStageProtocol
from cogno_soma.trace_cuts import (CUT_MARK, CUT_MARK_RE, TOOL_RESULT_CHARS,
                                   ToolResultLimitFn, cut_dropped, cut_with_mark,
                                   resolve_limit, was_cut)

__all__ = [
    "Pipeline",
    "SessionRunner",
    "TurnConfig",
    "Hooks",
    "HookFn",
    "StopPipeline",
    # The terminal signal for "the correction loop ran out and nothing was committed".
    # Exported because the HOST has to match on it (its proactive-delivery whitelist and
    # its escalation caps both key on `stop_reason`), and a string literal copied there
    # would be a second definition of a contract that has already been wrong once.
    "STOP_JUDGE_EXHAUSTED",
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
