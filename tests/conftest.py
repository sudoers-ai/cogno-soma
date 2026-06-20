"""Test doubles for orchestrator tests.

An orchestrator is best tested with **fake stages**: deterministic doubles that set
the context fields directly (no LLM, no network), so a test can drive any control-flow
path — a given route, a PII block, a scope block, a judge rejection → handoff — and
assert the wiring (gates, the correction loop, hooks, session threading). The real
stages are exercised by the Ollama-gated integration test instead.

The fakes match the cogno-anima stage method signatures the Pipeline calls.
"""

from __future__ import annotations

from typing import Optional

import pytest

from cogno_anima.types import (
    EgoResult,
    IdResult,
    IntentResult,
    NoumenoResult,
    ScopeCheckResult,
    StageMetrics,
    SuperegoResult,
    ToolResult,
)


def metrics(stage: str = "fake") -> StageMetrics:
    return StageMetrics(stage=stage, elapsed_ms=0.0, tokens_in=1, tokens_out=1, model="fake")


# ── stubbed transport (never actually called by the fake stages) ───────────
class StubBackend:
    model = "stub"

    async def generate(self, system: str, prompt: str):
        return "{}", 0, 0


class StubEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    async def similarity(self, a: str, b: str) -> float:
        return 1.0 if a == b else 0.0


class RecordingDispatcher:
    """A minimal ToolDispatcher that records nothing is executed by the fakes."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, dict]] = []

    def tools_schema(self) -> list[dict]:
        return []

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        self.executed.append((name, arguments))
        return ToolResult(output="", ok=True)


# ── fake stages ─────────────────────────────────────────────────────────
class FakeNoumeno:
    def __init__(self, rewritten: str = "hello", calls: Optional[list] = None) -> None:
        self._rewritten = rewritten
        self._calls = calls

    async def process(self, ctx, backend):
        if self._calls is not None:
            self._calls.append("noumeno")
        ctx.noumeno = NoumenoResult(
            original=ctx.user_input, rewritten=self._rewritten, context_turn="",
            language="en", drift_score=0.0, drift_tag="PASS_THROUGH", changed=False,
            confidence=0.9, change_subject=False, subject_similarity=1.0,
            context_used=False, preserved_terms=[], rewrite_warnings=[], metrics=metrics("noumeno"))
        return ctx


class FakeNER:
    def __init__(self, *, intent_class="INFORMATION_REQUEST", triad_signal="BALANCED",
                 goal="help the user", domains=None, calls: Optional[list] = None) -> None:
        self._kw = dict(intent_class=intent_class, triad_signal=triad_signal,
                        goal=goal, domains=domains or [])
        self._calls = calls

    async def process(self, ctx, backend):
        if self._calls is not None:
            self._calls.append("ner")
        ctx.intent = IntentResult(
            intent_class=self._kw["intent_class"], sentiment="NEUTRAL", confidence=0.9,
            temporal_class="PRESENT", triad_signal=self._kw["triad_signal"],
            goal=self._kw["goal"], domains=self._kw["domains"], metrics=metrics("ner"))
        return ctx


class FakeID:
    def __init__(self, *, route="EGO", blocked=False, goal_status="ONGOING",
                 calls: Optional[list] = None) -> None:
        self._route, self._blocked, self._goal_status = route, blocked, goal_status
        self._calls = calls

    async def process(self, ctx, embedder):
        if self._calls is not None:
            self._calls.append("id")
        ctx.metadata.setdefault("id_state", {})["seen"] = True
        ctx.id_result = IdResult(
            triad_route=self._route, blocked=self._blocked, goal_status=self._goal_status,
            metrics=metrics("id"))
        return ctx


class FakeEgo:
    def __init__(self, *, calls: Optional[list] = None) -> None:
        self._calls = calls
        self.invocations = 0
        self.last_system_prompt: Optional[str] = None

    async def process(self, ctx, backend, dispatcher, *, system_prompt):
        self.invocations += 1
        self.last_system_prompt = system_prompt
        if self._calls is not None:
            self._calls.append("ego")
        ctx.ego_result = EgoResult(metrics=metrics("ego"))
        return ctx


class FakeSuperego:
    """Programmable judge + voicer + scope guard + block."""

    def __init__(self, *, approve=True, critique="fix it", voice="final reply",
                 scope_blocked=False, refusal="out of scope", block_response="blocked",
                 approve_after: Optional[int] = None, calls: Optional[list] = None) -> None:
        self._approve = approve
        self._critique = critique
        self._voice = voice
        self._scope_blocked = scope_blocked
        self._refusal = refusal
        self._block_response = block_response
        self._approve_after = approve_after  # approve only on the Nth evaluate call
        self._evals = 0
        self._calls = calls

    async def check_input_scope(self, ctx, backend, *, scope_prompt):
        if self._calls is not None:
            self._calls.append("scope")
        return ScopeCheckResult(blocked=self._scope_blocked, refusal_message=self._refusal,
                                metrics=metrics("superego_scope"))

    async def evaluate(self, ctx, backend, *, limits_prompt):
        self._evals += 1
        if self._calls is not None:
            self._calls.append("judge")
        approved = self._approve
        if self._approve_after is not None:
            approved = self._evals >= self._approve_after
        return SuperegoResult(response="", approved=approved,
                              critique=None if approved else self._critique,
                              metrics=metrics("superego_judge"))

    async def voice(self, ctx, backend, *, voice_prompt):
        if self._calls is not None:
            self._calls.append("voice")
        return SuperegoResult(response=self._voice, approved=True, metrics=metrics("superego_voice"))

    def _blocked_response(self, ctx):
        return SuperegoResult(response=self._block_response, blocked=True, metrics=metrics("superego_voice"))


@pytest.fixture
def stub_embedder():
    return StubEmbedder()


@pytest.fixture
def stub_backend():
    return StubBackend()


@pytest.fixture
def dispatcher():
    return RecordingDispatcher()
