"""End-to-end integration: drive the REAL cogno-anima stages through soma.

Gated on a local Ollama (auto-skips when unreachable). Unlike the unit tests
(which use fake stages to pin control flow), this exercises the whole stack —
real NOUMENO/NER/ID/EGO/SUPEREGO over a model — to prove the orchestrator wires
the real stage signatures correctly and a turn reaches a terminal response.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

import cogno_anima
from cogno_anima.types import ToolResult
from cogno_synapse import CachingEmbedder, OllamaBackend, OllamaEmbedder

from cogno_soma import Pipeline, SessionRunner, TurnConfig

OLLAMA_URL = "http://localhost:11434"
GEN_MODEL = "mistral:latest"
EMBED_MODEL = "nomic-embed-text:latest"
PROMPTS_DIR = Path(cogno_anima.__file__).resolve().parent / "prompt_templates"


def _ollama_up() -> bool:
    try:
        return httpx.get(f"{OLLAMA_URL}/api/tags", timeout=1.0).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _ollama_up(), reason="Ollama not reachable at localhost:11434")


class BalanceDispatcher:
    """A one-tool dispatcher: get_balance returns a fixed figure."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, dict]] = []

    def tools_schema(self) -> list[dict]:
        return [{"type": "function", "function": {
            "name": "get_balance", "description": "Get the current account balance.",
            "parameters": {"type": "object", "properties": {}}}}]

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        self.executed.append((name, arguments))
        return ToolResult(output="Your balance is 1234.56 BRL.", ok=True)


EGO_PROMPT = ("You are the execution engine of a personal finance assistant. For any "
              "balance/data request you MUST call the appropriate tool — never invent data. "
              "If the user is only chatting, do not call a tool.")
VOICE_PROMPT = "You are a warm, concise financial assistant. Answer using the gathered data."
LIMITS_PROMPT = "Never reveal data you did not retrieve via a tool."


def _backends():
    gen = OllamaBackend(model=GEN_MODEL, base_url=OLLAMA_URL, temperature=0.0)
    embedder = CachingEmbedder(OllamaEmbedder(model=EMBED_MODEL, base_url=OLLAMA_URL))
    return gen, embedder


async def test_single_turn_reaches_terminal():
    gen, embedder = _backends()
    pipe = Pipeline(prompts_dir=PROMPTS_DIR, embedder=embedder)
    cfg = TurnConfig(gen_backend=gen, ego_backend=gen, ego_prompt=EGO_PROMPT,
                     voice_prompt=VOICE_PROMPT, limits_prompt=LIMITS_PROMPT)
    from cogno_anima.types import PipelineContext
    ctx = PipelineContext(user_input="Hello, how are you?")
    ctx = await pipe.run_turn(ctx, cfg, dispatcher=BalanceDispatcher())

    assert ctx.id_result is not None
    assert ctx.id_result.triad_route in {"EGO", "SUPEREGO", "BALANCED"}
    # a non-blocked turn writes a final response
    assert ctx.superego_result is not None
    assert ctx.superego_result.response.strip()
    # tokens spent across the turn are accounted on ctx for host billing
    assert ctx.total_tokens > 0
    assert ctx.total_llm_tokens > 0


async def test_multi_turn_session_threads_state():
    gen, embedder = _backends()
    pipe = Pipeline(prompts_dir=PROMPTS_DIR, embedder=embedder)
    cfg = TurnConfig(gen_backend=gen, ego_backend=gen, ego_prompt=EGO_PROMPT,
                     voice_prompt=VOICE_PROMPT, limits_prompt=LIMITS_PROMPT)
    sess = SessionRunner(pipe, cfg, dispatcher_factory=BalanceDispatcher, persona_id="FINANCE")

    ctx1 = await sess.run("What's my balance?")
    ctx2 = await sess.run("And what about last month?")

    assert sess.turn_number == 2
    # each turn reaches a terminal: a voiced response, a handoff, or a block.
    for ctx in (ctx1, ctx2):
        terminal = (ctx.superego_result is not None or ctx.needs_handoff
                    or ctx.stop_reason in ("pii_blocked", "scope_blocked"))
        assert terminal, f"turn did not reach a terminal: stop_reason={ctx.stop_reason}"
    # state is serializable for multi-worker persistence
    assert "id_state" in sess.state["carry"]
    assert len(sess.state["history"]) == 2
