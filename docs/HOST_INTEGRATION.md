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
critique, the draft that was judged, what the attempt was offered, and the calls it executed.
The keys, and where each is written (`cogno_soma/pipeline.py`, in the correction loop):

| key | what it is | written by |
|---|---|---|
| `attempt`, `approved` | the attempt number and the judge's verdict | the loop |
| `critique` | the judge's critique, cut at `_CRITIQUE_CHARS` | the loop |
| `draft`, `draft_len` | the text THIS attempt's judge read, cut at `_DRAFT_CHARS`, and its full length — both on every attempt, `""`/`0` when the executor wrote nothing | `_attempt_draft` |
| `committed` | this attempt ran a write that succeeded (`ok ∧ side_effect`), over its FULL list — the turn's own answer is `wrote_for_the_contact`, in the anima | `_attempt_tools` |
| `tools_offered` | the tools this attempt was OFFERED, after every mask — an empty `tools` then tells "declined" from "never on the table" | `_attempt_tools` |
| `tools` | the calls: `tool`, `args` (cut at `_TOOL_ARGS_CHARS`), `ok`, `side_effect`, `result` (cut with a mark, below) | `_attempt_tools` |
| `tools_dropped`, `tools_offered_dropped` | how many past `_TOOLS_PER_ATTEMPT` were left out — present only when some were | `_attempt_tools` |
| `tools_error` | the display list could not be built (the exception's TYPE); `committed` and `tools_offered` survive it | `_attempt_tools` |
| `branch` | the criteria block this attempt's judge was given (below) | `_attempt_branch` |

The tests that pin them: `test_pipeline.py` (`test_each_attempt_records_the_surface_it_was_OFFERED`,
`test_each_attempt_records_the_draft_the_judge_ACTUALLY_read`,
`test_the_offered_cap_is_reported_on_BOTH_paths`) and
`test_the_ledger_records_the_judges_branch.py`.

Each entry also carries `branch` — which criteria block THAT attempt's judge was given
(`execution` | `conversational` | `readonly`), copied from the `SuperegoResult.judge_branch` the
judge returned, never re-derived. It is per attempt because the branch can differ between
attempts: the classifier run over the context as the turn ENDS reads `execution` on a turn whose
attempt 1 only read (and was rejected) and whose attempt 2 wrote, so a question about attempt 1 —
"could the synchronous judge have been skipped on this clean read?" — is answered by
`judge_attempts[0]["branch"]` and by nothing computed afterwards. Closed alphabet: a label outside
the three is dropped. The key is ABSENT when the judge named no branch (a stand-in stage that does
not classify, or the anima's `evaluate` path that returns before choosing because nothing
executed), which is "not on record", not `execution`. Tool results there are
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

### 4.2 A held proposal, and when the judge reads it

When the EGO's gate B holds a call for the contact's "yes" (`ctx.ego_result.pending_confirmation`
is non-empty), the turn is a **proposal**: the action is incomplete on purpose. By default the
loop does **not** judge it, because a judge would reject the hold itself and spend a retry on
every confirmation turn. The loop records a stand-in approval instead and goes to the voice:
`judge_verdict` reads `{"approved": true, "attempts": 1}`, no `judge_attempts` entry is written,
and `on_commit` fires although nothing was committed.

**The exception is a held call that SENDS TEXT TO A PERSON** (a message, a note to staff). That
text is final at the hold: the confirmed replay sends those exact bytes. Skipping the judge there
meant the message's only review ran after it was delivered, where a correct critique un-sends
nothing (measured on a downstream host: 3 of 4 delivered messages were wrong).

So declare, per turn, which tools deliver which argument's text, in
`ctx.metadata[mk.HELD_DELIVERED_TEXT]` as `{tool_name: argument_name}`. Read it from each tool's
own manifest; the core never guesses it from a name. When
`cogno_anima.types.held_delivered_texts(ctx)` returns anything, the proposal turn is judged like
any other turn, and the judge reads the held text itself (cogno-anima's
`# Messages HELD for the user's confirmation` block):

- **approved**: the loop ends approved, and you propose the call as usual;
- **rejected**: the critique goes back to the EGO (`ego_correction`), which rewrites the message
  within the same correction budget;
- **still rejected when the budget is spent**: the loop ends unapproved. `judge_verdict` is
  already set when `after_ego` fires, so a proposal gate there must not arm a confirmation over
  `approved: false`. Without such a gate, soma's own exhaustion path takes the turn
  (`stop_reason = "judge_exhausted"`, nothing is proposed).

Every other hold keeps the skip, byte for byte: no declaration, a declaration that is not a
mapping, or held calls whose tools the declaration does not name. A declared call whose text
argument is missing or empty is still judged: an empty message is still a message about to be
proposed. Requires a cogno-anima with `held_delivered_texts` (sudoers-ai/cogno-anima#183).

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

The parameters are typed **structurally**, so a double that matches the shape is
accepted by the type checker too — no subclassing, no `cast`:

| parameter  | protocol                                    | what the Pipeline calls |
|------------|---------------------------------------------|-------------------------|
| `noumeno`  | `cogno_anima.BaseStage`                     | `process(ctx, llm)` |
| `ner`      | `cogno_anima.BaseStage`                     | `process(ctx, llm)` |
| `id_stage` | `cogno_soma.IDStageProtocol`                | `process(ctx, embedder)` |
| `ego`      | `cogno_soma.EgoStageProtocol`               | `process(ctx, backend, dispatcher, *, system_prompt)` |
| `superego` | `cogno_soma.SuperegoStageProtocol`          | `check_input_scope`, `evaluate`, `voice`, `_blocked_response(ctx, *, block_message=None)` |

Every one also carries `name: str`, because `BaseStage` does (every anima stage
has one; the Pipeline never reads it). `_blocked_response` is private-named in
cogno-anima but the Pipeline calls it on a PII-CRITICAL turn, so it is part of the
contract — a SUPEREGO double without it fails on exactly the path it least often
runs. `BaseStage` is reused from cogno-anima, not copied here. All four protocols are
`runtime_checkable`: `isinstance(double, IDStageProtocol)` in a test says the
members EXIST; the signatures are the type checker's to hold.

## 7. What stays yours

Persona selection, model-ladder/escalation, RBAC, metering/billing, the real
DB/MCP execution behind the dispatcher, retrieval, the human handoff, semantic
cache, session splitting. soma **signals**, you **decide**.
