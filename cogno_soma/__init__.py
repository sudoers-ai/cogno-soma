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
from cogno_soma.pipeline import Pipeline, STOP_JUDGE_EXHAUSTED
from cogno_soma.session import SessionRunner

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
]

__version__ = "0.1.0"
