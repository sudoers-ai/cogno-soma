# Build your own host

Cogno ships the **brain** — perception, routing, tool execution, guardrails,
voicing — as composable libraries. It does **not** ship the **body**: the process
that holds a database, serves requests, binds personas to tools, meters cost and
talks to a channel. That is the *host*, and you build it. Our own product host
(`cogno-host`) holds no special powers — it rides the same public seams this guide
uses. This page takes you from zero to a running agent using **only** the
open-source libs.

> A host is just four things: **backends + a tool dispatcher + four persona
> prompts + the persistence you own.** Everything else is a library.

## 0. The shape, offline (no model, 20 seconds)

Start with [`examples/host_min.py`](../examples/host_min.py). It injects *fake*
stages so you can watch the **orchestration** — routing, the EGO⇄SUPEREGO
correction loop, hooks, `StopPipeline`, a multi-turn session — with no network:

```bash
pip install cogno-soma
python examples/host_min.py
```

Read it once. The takeaway: `soma.Pipeline.run_turn(ctx, cfg, dispatcher=...)` is
the whole control flow, and every stage is something you *inject*.

## 1. A real host over a free local model (~5 min)

Now swap the fakes for real cognition. [`examples/host_real.py`](../examples/host_real.py)
is a complete ~90-line host over local Ollama with one real tool:

```bash
ollama pull qwen3:8b && ollama pull nomic-embed-text
python examples/host_real.py
```

```text
user:    quanto tá o meu saldo?
rewrite: How much is my balance?
intent:  INFORMATION_REQUEST | route: EGO
tools:   ['get_balance']
reply:   Seu saldo atual é de 1234,56 BRL.
tokens:  6175
```

Any language in; the NOUMENO rewrites to canonical English, the ID routes to the
EGO, the EGO calls your tool, and the SUPEREGO voices the answer back **in the
user's language**, grounded in the exact figure the tool returned. That is the
entire value proposition in one run.

## 2. The four seams you own

Everything a host provides plugs into one of these. In `host_real.py`:

| Seam | What you inject | Swap it for… |
|------|-----------------|--------------|
| **Backends** | `OllamaBackend` / `OllamaEmbedder` | any cloud model via `cogno_synapse.create_backend("openai:gpt-4o-mini")`; a `FallbackBackend` chain |
| **Dispatcher** | `BankDispatcher` (one in-memory tool) | your MCP client (`cogno-mcp`), in-process skills (`cogno-cortex`), or native funcs — merge several with `cogno_anima.tools.CompositeDispatcher` |
| **Persona prompts** | four hard-coded strings | a persona store / selector (`cogno-persona`) resolving them per tenant |
| **Persistence** | none (single turn) | your DB: persist `ctx.metadata["id_state"]` + history between turns (see §3) |

soma never reaches past these seams — no DB, no MCP, no billing, no persona store.
It **signals** (`stop_reason`, `needs_handoff`, `blocked`, `drift_action`); your
host **decides**.

## 3. Multi-turn: persistence is yours

`run_turn` is stateless — every cross-turn signal rides in
`ctx.metadata["id_state"]`, a serializable dict. `SessionRunner` threads it (plus
history and NER carry-over) for you, and is itself serializable so a multi-worker
host reconstructs it per request:

```python
from cogno_soma import SessionRunner

sess = SessionRunner(pipe, cfg, dispatcher_factory=BankDispatcher,
                     persona_id="FINANCE", state=load_state(session_id))
reply = (await sess.run("quanto tá o meu saldo?")).superego_result.response
save_state(session_id, sess.state)      # you persist the snapshot; any KV store works
```

`cogno-engram` is the optional persistence/memory/graph substrate if you want one
off the shelf; a dict in Redis works just as well.

## 4. Hooks: inject memory, screen for safety, keep atomicity

Retrieval, safety screens, auditing and transaction boundaries plug in as `Hooks`
fired around the stages — no fork of soma required:

```python
from cogno_soma import Hooks, StopPipeline

async def inject_memory(ctx):
    ctx.metadata["ego_context"] = await store.retrieve(ctx.user_input)

def crisis_screen(ctx):
    if is_crisis(ctx.noumeno.rewritten):
        raise StopPipeline(reason="human_handoff", response=CRISIS_MSG, blocked=True)

cfg = TurnConfig(..., hooks=Hooks(
    after_ner=inject_memory,
    after_id=crisis_screen,
    on_rollback=lambda ctx: tx.rollback(),   # before each EGO retry
    on_commit=lambda ctx: outbox.flush(),    # once the judge approves
))
```

## 5. Cost accounting

Every LLM call the turn makes lands on the returned `ctx`. soma counts, never
prices — feed the totals to your meter (`cogno-meter` prices them, or your own):

```python
ctx.total_tokens            # LLM + embeddings for the whole turn
ctx.total_llm_tokens        # prompt + completion only
```

## 6. From here to a product

You now have a turn. A shipping host adds, in order of how often people need them:

1. **A channel** — put `sess.run(text)` behind an HTTP handler, or use
   `cogno-gateway` for Telegram / WhatsApp / web.
2. **Personas + tools** — resolve prompts per persona (`cogno-persona`) and back
   the dispatcher with real capabilities (`cogno-mcp` servers like `cogno-praxis`,
   or `cogno-cortex` skills).
3. **Persistence** — sessions, history, memories (`cogno-engram` or your DB).
4. **Metering, RBAC, model ladders, human handoff** — all host policy; soma gives
   you the signals to drive them.

Each lib's `docs/HOST_INTEGRATION.md` documents its seam. The demo host you can
run today — the SECRETARY in `cogno-praxis` — is exactly this pattern with a real
scheduling vertical behind the dispatcher.
