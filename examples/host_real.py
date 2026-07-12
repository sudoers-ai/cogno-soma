"""A **real** minimal host: the full cognitive pipeline over local Ollama.

Where ``host_min.py`` uses fake stages to show the *orchestration shape* with no
network, this wires the actual cogno-anima stages over a local model and a real
(in-memory) tool — so you see genuine cognition: NOUMENO rewrites the input, the
NER reads intent + PII, the ID routes, the EGO calls the tool, the SUPEREGO judges
and voices the reply.

It is deliberately ~90 lines: a host is just *backends + a dispatcher + four
persona prompts + persistence you own*. Everything below the prompts is the
public seam; swap the tool for your MCP client, the prompts for your persona
store, and the in-process call for your HTTP handler.

    ollama pull qwen3:8b && ollama pull nomic-embed-text
    pip install cogno-soma            # pulls cogno-anima + cogno-synapse
    python examples/host_real.py
"""

import asyncio

from cogno_anima.types import PipelineContext, ToolResult
from cogno_synapse import OllamaBackend, OllamaEmbedder, CachingEmbedder

from cogno_soma import Pipeline, TurnConfig


# ── 1. The tool layer (your "hands"): a ToolDispatcher the host injects. ──────
# A real host routes this to an MCP client, in-process skills, or native funcs
# (merge several with cogno_anima.tools.CompositeDispatcher). Here: one fake tool.
class BankDispatcher:
    _BALANCES = {"default": "1234.56 BRL"}

    def tools_schema(self):
        # OpenAI-format tool specs — the same shape native FC and the text
        # fallback both consume.
        return [{
            "type": "function",
            "function": {
                "name": "get_balance",
                "description": "Return the current account balance for the user.",
                "parameters": {"type": "object", "properties": {}},
            },
        }]

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        if name == "get_balance":
            return ToolResult(output=self._BALANCES["default"], ok=True)
        return ToolResult(output="", ok=False, error=f"unknown tool: {name}")


# ── 2. The persona (your prompts): four strings the host owns. ───────────────
# A real host resolves these per persona (e.g. via cogno-persona). soma is
# prompt-agnostic — it never reads a persona store.
EGO_PROMPT = "You are a bank assistant's executor. Use tools to gather the facts the user asked for."
SCOPE_PROMPT = "You handle personal banking questions. Allow those; block anything unrelated."
LIMITS_PROMPT = "Never invent figures. Only state balances returned by a tool. Do not give financial advice."
VOICE_PROMPT = "Reply in one warm, concise sentence. State the exact figure the tool returned."


async def main() -> None:
    # ── 3. The backends (your models): local + free here; swap for cloud via
    #        cogno_synapse.create_backend("openai:gpt-4o-mini"), etc. ──────────
    gen = OllamaBackend(model="qwen3:8b", temperature=0.0, format="json")  # JSON stages
    ego = OllamaBackend(model="qwen3:8b", temperature=0.0)                 # executor (text FC fallback)
    embedder = CachingEmbedder(OllamaEmbedder(model="nomic-embed-text"))

    pipe = Pipeline(embedder=embedder)  # stages default to the real cogno-anima ones
    cfg = TurnConfig(
        gen_backend=gen, ego_backend=ego,
        ego_prompt=EGO_PROMPT, scope_prompt=SCOPE_PROMPT,
        limits_prompt=LIMITS_PROMPT, voice_prompt=VOICE_PROMPT,
    )

    # ── 4. One turn. In an HTTP host this is your request handler body. ───────
    ctx = PipelineContext(user_input="quanto tá o meu saldo?")   # any language in
    ctx = await pipe.run_turn(ctx, cfg, dispatcher=BankDispatcher())

    print("user:   ", ctx.user_input)
    print("rewrite:", ctx.noumeno.rewritten)          # NOUMENO → canonical English
    print("intent: ", ctx.intent.intent_class, "| route:", ctx.id_result.triad_route)
    print("tools:  ", [t.tool for t in ctx.ego_result.tools_executed])  # what the EGO called
    print("reply:  ", ctx.superego_result.response)   # SUPEREGO voices the final answer
    print("tokens: ", ctx.total_tokens, "(feed your meter — soma counts, never prices)")


if __name__ == "__main__":
    asyncio.run(main())
