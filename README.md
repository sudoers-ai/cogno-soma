# cogno-soma

**The reference orchestrator for the Cogno cognitive pipeline.**

`cogno-anima` ships the cognitive *stages* (NOUMENO → NER → ID → EGO ⇄ SUPEREGO →
Drift) as decoupled pieces — but deliberately **not** the glue that runs them
end-to-end. `cogno-soma` is that glue, promoted to a shippable, infra-agnostic lib:
the control flow, the EGO⇄SUPEREGO correction loop, the PII/scope/handoff gates, an
**interception seam** (`Hooks` / `StopPipeline`) and **atomicity** callbacks.

It is the keystone that imports the stages and wires them — and *only* that. The
host still injects the backends, the tool dispatcher and the persona's prompts;
soma never touches a DB, an MCP client, billing, or a persona store.

```
NOUMENO → NER → ID → [PII gate] → [scope gate] → EGO ⇄ SUPEREGO(judge) → SUPEREGO(voice)
```

## See it run — a complete agent in 15 minutes

The SECRETARY demo in
[`cogno-praxis`](https://github.com/sudoers-ai/cogno-praxis#quickstart--a-complete-agent-in-15-minutes)
is soma doing its job end-to-end: the full cognitive loop over local Ollama,
with a real scheduling vertical served as an MCP subprocess and the cognition
visible stage by stage (`--trace`). Free, offline, any language in:

```bash
git clone https://github.com/sudoers-ai/cogno-praxis && cd cogno-praxis
python examples/secretary_demo.py --trace     # full install chain in the praxis README
```

## Install

```bash
pip install cogno-soma          # pulls cogno-anima + cogno-synapse
```

> The sibling libs are not on PyPI yet — install them from git first
> (`cogno-homeo`, then `cogno-synapse`, then `cogno-anima`); see `.github/workflows/ci.yml`.

## Quick start — one turn

```python
from cogno_soma import Pipeline, TurnConfig
from cogno_anima.types import PipelineContext
from cogno_synapse import OllamaBackend, OllamaEmbedder

gen = OllamaBackend(model="mistral:latest")        # NOUMENO/NER/scope/judge (JSON)
embedder = OllamaEmbedder(model="nomic-embed-text:latest")

pipe = Pipeline(prompts_dir=PROMPTS_DIR, embedder=embedder)
cfg = TurnConfig(
    gen_backend=gen, ego_backend=gen,
    ego_prompt=persona_system, scope_prompt=persona_scope,
    limits_prompt=persona_limits, voice_prompt=persona_voice,
)

ctx = PipelineContext(user_input="What's my balance?")
ctx = await pipe.run_turn(ctx, cfg, dispatcher=my_dispatcher)
print(ctx.superego_result.response)     # the final, voiced reply
```

The four prompt strings come from the host (e.g. resolved via
[`cogno-persona`](https://github.com/sudoers-ai/cogno-persona)'s `PersonaSelector`).
soma is **prompt-agnostic** — it does not depend on a persona store.

## Multi-turn sessions

`run_turn` is single-turn and **stateless** (every cross-turn signal rides in
`ctx.metadata["id_state"]`). `SessionRunner` threads that state — `id_state`,
conversation history, NER carry-over — across a session, and is itself a thin,
**serializable** holder so a multi-worker host reconstructs it per request:

```python
from cogno_soma import SessionRunner

sess = SessionRunner(pipe, cfg, dispatcher_factory=build_dispatcher,
                     persona_id="FINANCE", state=load_state(session_id))

ctx = await sess.run("What's my balance?", memories=retrieved_facts)
save_state(session_id, sess.state)      # persist the snapshot for the next request
```

## Hooks — the interception seam

Memory injection, safety screens, auditing and atomicity plug in as optional
`Hooks` fired around the stages (mirroring the parent's `InterceptorChain`). A
hook may be sync or async, may mutate the context, or raise `StopPipeline` to halt
the turn with a terminal response:

```python
from cogno_soma import Hooks, StopPipeline

async def inject_memory(ctx):
    ctx.metadata["ego_context"] = await store.retrieve(ctx.user_input)

def crisis_screen(ctx):
    if is_crisis(ctx.noumeno.rewritten):
        raise StopPipeline(reason="human_handoff", response=CRISIS_MESSAGE, blocked=True)

cfg = TurnConfig(..., hooks=Hooks(
    after_ner=inject_memory,
    after_id=crisis_screen,
    on_rollback=lambda ctx: tx.rollback(),   # before each EGO retry
    on_commit=lambda ctx: outbox.flush(),    # once the judge approves
))
```

Fire order: `before_turn` → NOUMENO → `after_noumeno` → NER → `after_ner` → ID →
`after_id` → gates → EGO⇄SUPEREGO (`on_rollback` per retry, `on_commit` on
approval) → `after_ego` → voice → `after_superego` → `after_turn`.

## The turn the agent opens

A proactive turn has no utterance. What you hold is your OWN directive ("open the conversation
with Ana, mention the checkup"), and running the two stages that exist to understand a *person*
over it is not wasteful but wrong — five ways, each one a stage doing its job correctly on the
wrong input:

| stage | what it does to your directive |
| --- | --- |
| NER (PII) | classifies YOUR words; "checkup" reads as `HEALTH_DATA` → CRITICAL → the ID blocks the turn and the agent never opens |
| ID (goal) | files the directive as the CONTACT's goal, so their first reply reads as a topic change |
| SUPEREGO (scope) | judges whether your own sentence is on-topic for your deployment |
| NOUMENO (drift) | measures every drift number for the turn against your prompt |
| NOUMENO (continuity) | embeds your internal marker against the contact's last message to decide `change_subject` |

So hand the two results in and the pipeline seats them — no LLM, no embedder, no tokens:

```python
import dataclasses
from cogno_soma import opening_perception

cfg = dataclasses.replace(cfg, scope_prompt="", **opening_perception())
ctx.user_input = "[OPENING]"          # a marker, not words you put in the contact's mouth
ctx = await pipe.run_turn(ctx, cfg, dispatcher=dispatcher)
```

The EGO, the judge and the voice still run for real — the judge is what stops an opening from
inventing the history it greets someone with — and the scope guard goes off because a guard
whose input is your own directive can only ever block you.

`TurnConfig.noumeno_result` / `intent_result` are the general seam; `opening_perception()` is
the one set of values this library ships for it. **What is a decision comes from the result;
what is a fact about the turn comes from the context** — `original` is `ctx.user_input`,
`language` is `ctx.force_language`, `langue` is inherited from the NOUMENO — so a stand-in can
never claim the turn carried text nobody put in it.

## Token accounting

Every LLM call a turn makes — NOUMENO, NER, the scope guard, **each** judge
attempt, **each** EGO attempt of the correction loop, and the voice — lands on the
returned `ctx`. soma drops nothing; read the totals and feed your meter:

```python
ctx.total_tokens            # LLM + embedding tokens for the whole turn
ctx.total_llm_tokens        # prompt + completion (no embeddings)
ctx.total_embedding_tokens  # NOUMENO continuity + ID goal-similarity
```

soma preserves the counts but does not price them — metering is the host's
(`cogno-meter`). A blocked / handoff turn still reports what it spent.

## What stays at the host

Persona selection, metering/billing, the DB/MCP execution behind the dispatcher,
RBAC, retrieval, the real human handoff and semantic cache — soma **signals**
(`stop_reason`, `needs_handoff`), the host **decides**. See `docs/HOST_INTEGRATION.md`.

## The Cogno ecosystem

`cogno-soma` is one organ of **[Cogno](https://github.com/sudoers-ai)** — a family of
small, composable, Apache-2.0 libraries that together form a complete
conversational-agent platform. Each library owns a single concern and stays
infra-agnostic; a **host** assembles them into a running agent:

![The Cogno ecosystem](docs/assets/cogno-ecosystem.svg)

The open-source libraries are the organs; the **host is the body** that joins
them. Our reference host — `cogno-host`, with its `cogno-ui` dashboard — is the
private product layer, but it holds no special powers: everything it does rides
on the public seams documented in each library's `docs/HOST_INTEGRATION.md`, so
you can assemble a body of your own.

## Development

```bash
pip install -e ".[dev]"
pytest tests/unit -q            # fast, no network (fake stages)
pytest tests/integration -q     # real stages over Ollama, auto-skips if absent
ruff check cogno_soma tests && mypy cogno_soma
```

Apache-2.0.
