# Host integration — cogno-soma

soma is the **orchestrator**: it sequences the `cogno-anima` stages and runs the
correction loop, but every side-effecting decision is the host's. This guide maps
the seams a host wires up.

## 1. What you inject

| Inject | Where | Notes |
|---|---|---|
| `gen_backend` | `TurnConfig` | JSON-capable `LLMBackend` for NOUMENO/NER/scope/judge. |
| `ego_backend` | `TurnConfig` | EGO executor; native FC (`ToolCallingBackend`) or text-fallback. |
| `voice_backend` | `TurnConfig` | optional; defaults to `ego_backend`. May be a smaller voicer. |
| `embedder` | `Pipeline(...)` | `Embedder` for NOUMENO subject-continuity + ID goal similarity. |
| `dispatcher` | `run_turn(...)` / `SessionRunner` | the host's `ToolDispatcher` — your DB/MCP/API hands. Build it per turn with the request's auth/tenant scope. |
| `ego_prompt` / `scope_prompt` / `limits_prompt` / `voice_prompt` | `TurnConfig` | the persona's four prompt slots, as plain strings. |
| `block_message` | `TurnConfig` | PII-CRITICAL block reply, in your tenant's language; unset → the core's English fallback reaches the contact. Keep it a static constant — it is the only outgoing message that never passes the outgoing-PII detector (the blocked branch returns before `voice()`). |
| `tool_result_limit` | `TurnConfig` | optional `Callable[[str], int]` — how many characters of ONE tool's result survive into the per-attempt judge ledger, per tool NAME. See **§4.1** below: if you persist that ledger and re-cut it, pass the SAME function you re-cut it with. |

soma does **not** select the persona. Resolve it at the host (e.g. with
[`cogno-persona`](https://github.com/sudoers-ai/cogno-persona)'s `PersonaSelector`)
and pass the four resolved strings in. Empty `scope_prompt` skips the pre-EGO scope
guard.

## 2. Single turn vs session

`Pipeline.run_turn(ctx, cfg, *, dispatcher)` runs one turn and is **stateless** —
all cross-turn state lives in `ctx.metadata["id_state"]`. For a conversation, use
`SessionRunner`, which threads `id_state` + history + NER carry-over and exposes a
serializable `.state`:

```python
sess = SessionRunner(pipe, cfg, dispatcher_factory=build_dispatcher,
                     persona_id=p.id, mcp_module=p.primary_module,
                     force_language=tenant.lang, state=load(session_id))
ctx = await sess.run(user_text, memories=retrieved_facts)
save(session_id, sess.state)
```

Persist `sess.state` (a plain dict) keyed by session id; reconstruct per request so
a multi-worker deployment stays correct (no pinned in-memory instance).

## 3. Hooks — interception + atomicity

`Hooks` are optional callbacks (sync or async) fired at fixed points. Use them for:

- **memory** — `after_ner` to inject retrieved facts into `ctx.metadata["ego_context"]`
  (or pass `memories=` to `SessionRunner.run`), and to persist the turn (cogno-engram).
- **safety** — `after_id` (or `after_ner`) to screen input; raise `StopPipeline` to
  halt with a terminal response.
- **audit / tracing** — `before_turn` / `after_turn` bookends; `after_id` to record
  the route.
- **atomicity** — `on_rollback` (fires before each EGO retry) and `on_commit` (fires
  once the judge approves) to manage your DB tx / write-behind buffer / outbox.

```python
raise StopPipeline(reason="pii_blocked", response=refusal, blocked=True)
```

`reason` should come from `cogno_anima.vocab.VALID_STOP_REASONS`.

## 4. Reading the outcome

After a turn, inspect `ctx`:

- `ctx.superego_result.response` — the final reply to send (None on a pure handoff).
- `ctx.stop_reason` — `completed | human_handoff | scope_blocked | pii_blocked` (+
  any reason your hooks set).
- `ctx.needs_handoff` — escalate to a human (you own the actual handoff: queue,
  notify, etc.).
- `ctx.id_result.triad_route` / `ctx.id_result.blocked` — routing + PII gate.
- `ctx.ego_result.tools_executed` — what the EGO actually ran.
- metrics: see **§5 Token accounting** below.

### 4.1 The per-attempt ledger, and the cut that leaves a mark

`ctx.metadata["judge_attempts"]` carries one entry per EGO⇄SUPEREGO attempt: the verdict, the
critique, the draft that was judged, and the calls that attempt executed. Tool results there are
**cut**, because the entry rides in metadata a host persists and a tool result is unbounded prose.

A cut ends in `…[cortado, faltam N chars]`, where **N is what is MISSING** — not what was kept,
not the total. Three properties come with it, and `cogno_soma.trace_cuts` exports what you need
to rely on them:

- `was_cut(text)` / `cut_dropped(text)` — **detect a cut by the MARK, never by the length.** With
  a per-family ceiling the length test is not a guess but an inversion: a long result that
  arrived WHOLE is `>= 240` and a length test calls it truncated.
- the mark fits **inside** the ceiling (room is reserved before cutting), so the field is always
  `<= limit` and the ceiling stays a number rather than an approximation;
- `CUT_MARK` / `CUT_MARK_RE` are exported so a reader never re-spells them. Two copies of one
  mark drift, and the day they do the reader stops recognising the writer's own output.

**If your host re-cuts this ledger at its own ceiling, pass that same ceiling in as
`tool_result_limit`.** Which tools deserve a bigger ceiling is a catalog of tool NAMES, which is
your business, not an infra-agnostic orchestrator's — so soma ships the mechanism and consults
your policy, the same shape as `escalate`. Leave it unset and every result is cut at
`trace_cuts.TOOL_RESULT_CHARS`; a declared ceiling BELOW that is floored to it, because a writer
cutting under its reader's ceiling is how one half of the trace ends up shorter than the other
and the difference reads as a fact about the turn.

## 5. Token accounting (for billing / control)

Every LLM call a turn makes lands on `ctx` — soma drops nothing:

- per-stage: `ctx.noumeno_metrics`, `ner_metrics`, `id_metrics`, `ego_metrics`
  (the final EGO attempt), `superego_metrics` (the voice).
- `ctx.retry_metrics` — the scope guard, **every** judge attempt, and each
  **rejected** EGO attempt of the correction loop.
- `ctx.stage_metrics` is the union of both; the totals sum over it:

```python
ctx.total_tokens            # LLM + embedding tokens for the whole turn
ctx.total_llm_tokens        # prompt + completion only (no embeddings)
ctx.total_embedding_tokens  # NOUMENO subject-continuity + ID goal-similarity
ctx.total_elapsed_ms
```

Read these after `run_turn` (or `SessionRunner.run`) and feed them to your meter
(e.g. `cogno-meter`). soma itself does **not** price or meter — it only preserves
the counts. A turn that ends in `pii_blocked` / `scope_blocked` / `human_handoff`
still reports the tokens it spent up to that point.

## 6. Stage override (advanced)

`Pipeline(embedder=..., noumeno=..., ner=..., id_stage=..., ego=..., superego=...)`
lets you swap any stage for a custom implementation (a cheaper NER, a cached
NOUMENO, a test double) as long as it matches the cogno-anima stage signature.

## 7. What stays yours

Persona selection, model-ladder/escalation, RBAC, metering/billing, the real
DB/MCP execution behind the dispatcher, retrieval, the human handoff, semantic
cache, session splitting. soma **signals**, you **decide**.
