# Changelog

## Unreleased

Entries here start on 2026-09-24. Earlier changes since 0.1.0 are in the git history only.

### Added

- **`SessionRunner.run(split_context=True)` hands the context over by provenance, on two
  carriers** (2026-10-07). The context the runner composes has two authors: the `[SOURCES]`
  instruction is this library's, and the conversation, the earlier-session summary, the memories
  and the graph facts were written by other people. In one block the two are the same unfenced
  text, and a line typed inside a memory in the instruction's own style cannot be told from the
  instruction. On, the instruction stays in `mk.EGO_CONTEXT` and the layers go to
  `mk.EGO_CONTEXT_UNTRUSTED`, which `cogno-anima` renders inside a fence.
  - **Off is the default and is byte for byte**: the test pins the whole block as a literal, and
    it passes unchanged on the parent tree `ed81f9a`.
  - On, the two carriers joined by a blank line ARE that block; with no layer the second key is
    not set; `metadata=` still wins on both; `mk.CONVERSATION_HISTORY` does not move.
  - Needs a `cogno-anima` with `mk.EGO_CONTEXT_UNTRUSTED` (anima #213). **This repo's CI now PINS
    `cogno-anima` at that merge, `8f3970b8`**, in both install steps of `ci.yml`: until now every
    sibling was installed from its `main` with no SHA, so a run was not reproducible and a
    sibling's landing could turn this suite red with nobody having touched it. Only
    `cogno-anima` is pinned here; `cogno-homeo`, `cogno-synapse` and `cogno-engram` still float.
    Moving the pin is now a deliberate edit of `ci.yml`.
  - `tests/unit/test_context_is_handed_over_by_provenance.py`: 11 tests, 10 red on `ed81f9a`
    (the argument does not exist there) and the byte-for-byte control green on both.

- **No UNIT test reaches the local Ollama — enforced, not promised** (2026-09-30, #58).

- **A held message the judge rejected is recomposed ONCE by the EGO, with the critique
  (`_owes_a_held_rewrite`, `extra_pass: held_message_rejected`).** Measured live downstream: the
  contact asked for a held message to be recomposed, the EGO composed a new one, the judge read it
  (the anima's held-message rule, criterion (a)) and rejected it with a correct critique ("it must
  mention today's class"); with a budget of one there was no pass left, so nothing was proposed and
  the contact read the host's neutral sentence. The voice cannot answer it: the held text is a tool
  argument only the EGO writes.
  - The third sibling of `_owes_an_action`/`_owes_a_read`, under the same `_ACTION_RETRY_CEILING`
    (two EGO passes in all) and asked FIRST of the three (precedence decides only the label).
  - Fires when the rejected attempt is a held-message proposal (`pending_confirmation` and
    `held_delivered_texts` both non-empty — exactly the proposal turns the loop judges) and
    nothing was committed (`committed_this_turn`). The pass carries the critique alone in
    `ego_correction`; its cost lands in `retry_metrics` like every rejected pass.
  - Never executes anything unconfirmed: the recomposed message is held again by gates B/C (pinned
    with the anima's REAL `EgoStage` and a policy dispatcher) and judged again — approved → the
    host proposes it; rejected → today's outcome (same stop reason, voice correction and reply,
    pinned against the predicate forced off), no third pass. Inert on a budget of two or more.
  - `EXTRA_PASS_HELD_MESSAGE` joins the exported closed alphabet `EXTRA_PASSES`.
  - `tests/unit/test_a_rejected_held_message_is_recomposed_once.py` (twin red at the base,
    measured; FAB; controls: approved at once byte-identical, no held message, undeclared hold, a
    commit, a budget of two; the shared ceiling; the alphabet). `docs/HOST_INTEGRATION.md` §4.2,
    §4.3; `README.md`.

- **No UNIT test reaches the local Ollama — enforced, not promised** (2026-09-30).
  `tests/unit/conftest.py` (new) installs `tests/unit/_ollama_gate.py` as a session-scoped autouse
  fixture: `socket.socket.connect`/`connect_ex` (the lowest point, where every client ends) and
  httpcore's `connect_tcp` (so the httpx path raises the gate's error unwrapped, not inside anyio's
  `ExceptionGroup` — measured) refuse port 11434 on any host and the host:port of
  `OLLAMA_BASE_URL`/`COGNO_OLLAMA_URL`/`OLLAMA_HOST` (read at connect time) with `OllamaGateError`;
  a per-test fixture fails the test at teardown when an attempt was made, even one the code under
  test swallowed. Why: the local Ollama is the GPU serving live traffic, and a host unit test that
  left a backend `None` got the real `OllamaBackend` default and called `localhost:11434`.
  `tests/integration` does not load it. `tests/unit/test_ollama_gate.py` pins the twins (on an
  EMPTY loopback port declared as the Ollama, never 11434), the controls and the swallowed case.

- **"Did you mean…?" on a refused turn: `TurnConfig.scope_selector_backend` + `scope_options` (#57).**
  Both or nothing. On a turn the scope guard BLOCKED, `cogno_anima.stages.scope_options` asks the
  host-injected backend which of the host's CLOSED options the message is about. A `covered` pick
  (the very thing asked) means the refusal was FALSE and the turn goes on to the EGO/voice — a
  false refusal is corrected by letting it through, never by answering it with a question. Any
  other outcome keeps the refusal byte for byte and leaves the record on
  `mk.SCOPE_OPTIONS_SELECTION` for the host to render its closed question. The call has its own
  ledger line (`superego_select`, in `retry_metrics`). The record is PER TURN: popped before the
  PII gate on every turn. Unset → the pipeline of before (`tests/unit/test_the_scope_selector.py`).
  A `covered` pick lifts only with the anima's code-side evidence (a shared non-generic term);
  `cogno-engram` joins this repo's CI chain because the anima reads it for that (fail-closed without).

- **The runner's verbatim window is PUBLIC: `CONTEXT_WINDOW_GAP_SECONDS` (4 h) and
  `CONTEXT_WINDOW_EXCHANGES` (6) (#56)**, exported from `cogno_soma` and from `cogno_soma.session`, and
  the defaults of `SessionRunner(burst_gap_seconds=, max_history=)`. The host writes rows into the
  transcript itself (a delivered message) and needs to know whether its row is still in the window;
  it used to import the private `_DEFAULT_BURST_GAP_SECONDS` across its pinned boundary and keep its
  own copy of the 6. The private name stays as an ALIAS (the same object), so nothing breaks.
  `docs/HOST_INTEGRATION.md` § 2, `tests/unit/test_context_window_is_public.py` (the names at the
  front door, the defaults, and a twin across each edge of the window).

- **The «sim» that sends a held message the judge already approved is not judged again
  (`_prejudged_replay`, #55).** Measured downstream: a staff reader confirmed a proposed `notify_user`,
  the replay delivered it, and the post-send judge rejected it for want of a read "this turn" —
  the read, and the judge's own approval of those exact bytes, were on the proposal turn. Budget 1:
  `human_handoff` before the voice, over a message that went out right.
  - Attempt 1 of a confirmed turn skips the judge when ALL four hold: (a) it executed exactly the
    multiset of `mk.EGO_CONFIRMED_CALLS`; (b) each tool is in `mk.HELD_DELIVERED_TEXT`; (c) each
    came back `ok ∧ side_effect`; (d) each confirmed row carries the host's stamp
    `PREJUDGED_TEXT_SHA` (`prejudged_digest` of the approved text) and it matches the bytes sent.
  - Recorded, never silent: the ledger row carries `skipped: "prejudged_replay"`, exported as
    `JUDGE_SKIPS` for the host's persisted alphabet (the `extra_pass` lesson).
  - **No stamp, no skip**: a host that does not stamp keeps today's judge, byte for byte; each
    control (an extra call, an undeclared tool, a failed replay, no confirmation, no stamp, a stamp
    one byte off, a confirmed call that did not run, rows without the «sim» flag, a DOUBLE
    delivery — one confirmed call executed twice, which (a) counts as a multiset — and a FAILED
    extra call beside the ok replay, which only a blocked re-issue of a confirmed call is
    excused from) is pinned
    against the ledger `main` produced, and two more (a consult ran, another message was held
    beside the replay) against the loop with the skip switched off.
  - Not this change: carrying the PROPOSAL turn's reads to the post-send judge for any confirmed
    write (a judge-prompt rule and tool results in session state — a separate, larger item).
  - `tests/unit/test_a_prejudged_replay_is_not_judged_again.py`.

### Changed (docs)

- **The README catches up with #54–#58** (docs sweep, 2026-10-06). A new section, "When the guard
  refuses, and when the budget is spent", states the "did you mean…?" selector letting a false
  refusal through (#57) and the two extra EGO passes under one ceiling (#54). "Multi-turn
  sessions" names the public verbatim window (#56). The install note adds `cogno-engram` to the
  git chain, as `.github/workflows/ci.yml` installs it since #57. The prejudged-replay paragraph
  and the Ollama gate note cite their PRs (#55, #58), and so do the five `Added` entries above.
  Each points at `docs/HOST_INTEGRATION.md` rather than repeating it. Docs only.

- **`docs/HOST_INTEGRATION.md` §4.1 lists every key of a judge-ledger entry** (Phase 2 docs
  sweep B). The prose named four of them — verdict, critique, draft, calls — while an entry also
  carries `draft_len`, `committed`, `tools_offered`, the two `*_dropped` counts, `tools_error` and
  (#52) `branch`. The section is now a table: each key, what it is, and the function in
  `cogno_soma/pipeline.py` that writes it, with the tests that pin them. The absence of `branch`
  is also stated in full: a stand-in judge that does not classify, OR the anima's `evaluate`
  returning before it chooses because nothing executed. Docs only.

### Added

- **One more EGO pass for a negative over a source nobody read (`_owes_a_read`, item (i), #54).**
  The sibling of `_owes_an_action`, asked after it, under the same `_ACTION_RETRY_CEILING` (two
  EGO passes in all). The shape, measured on a rehearsal tenant: an INFORMATION_REQUEST with the
  document-reading tool on the table; the EGO called only an empty ledger summary; the draft said
  "there are no records"; the judge rejected it; and with a budget of one the voice shipped the
  same false negative, while the documents held the values. The voice cannot read.
  - It fires only when three conditions hold together, all deterministic:
    1. an INFORMATION_REQUEST that the judge rejected;
    2. a host-declared source read (`mk.SOURCE_READS`) was offered and no pass called any of
       them, read by `cogno_anima.types.source_reads_not_called` over the anima's shared walk
       and never re-derived here;
    3. the draft asserts absence, by one closed PT/EN list (`_ABSENCE`).
  - It also carries the sibling's guard: a turn that committed gets nothing.
  - The extra pass's `ego_correction.reason` is the critique plus one closed sentence
    (`_READ_OWED_NOTE`) that names the tool that was not called.
  - Reach, measured over 244 production turns (01/09 to 29/09, with a judge block and a table):
    the wide shape matches 20 turns; with the negative, 4, none of them this defect; with the set
    of one tool the host starts with, 0. On the rehearsal tenant it matches 4 of 4.
  - Off without the declaration, so a host reverts the feature by not stamping the key.
  - The two exceptions SHARE the ceiling: with both predicates forced true, a turn still gets
    one extra pass.
  - **Countable:** the rejected ledger entry that bought the pass carries `extra_pass`
    (`action_owed` | `read_owed`, exported as `cogno_soma.EXTRA_PASSES`), absent otherwise. Until
    now a grant, the action one included, left only a DEBUG log line.
  - `tests/unit/test_one_turn_to_read.py` covers:
    - the twin (red at the base, measured) and its pair with the source read on pass 1;
    - the switch as ONE twin (declared → one extra pass; absent → none) and the shared ceiling;
    - controls (a) to (f);
    - condition 3, the committed guard, the ceiling and a budget of two;
    - the absence list with its own twin;
    - the REAL anima `EgoStage` over a scripted backend, which is handed the sentence and reads.
  - `docs/HOST_INTEGRATION.md` §4.3. Requires sudoers-ai/cogno-anima#196.

- **Each judge-ledger entry records which criteria its judge was given (`branch`).**
  `ctx.metadata["judge_attempts"][i]["branch"]` is the `SuperegoResult.judge_branch` that
  attempt's `evaluate` returned (`execution` | `conversational` | `readonly`) — the anima's own
  `_judge_branch`, reused, never re-derived here. The loop kept the verdict and dropped the label,
  so a host could only re-run the classifier over the context as the turn ENDED, and that is a
  different fact: its write half only accumulates, so a turn whose attempt 1 only READ (rejected)
  and whose attempt 2 WROTE reads `execution` at the end although attempt 1 was judged as a clean
  read. That is exactly the attempt a "could the synchronous judge have been skipped?" shadow
  measures. Closed alphabet (anything else is dropped); absent when the judge named no branch.
  No behaviour changes — the judge runs exactly as before.
  `tests/unit/test_the_ledger_records_the_judges_branch.py` runs the REAL anima judge over a stub
  backend: clean read → `readonly`, write → `execution`, a read with one failed call →
  `execution`, the per-attempt twin with the end-of-turn classifier as its pair, and the two
  absence controls. See `docs/HOST_INTEGRATION.md` §4.1.

### Changed

- **The stage parameters of `Pipeline` are typed structurally.** `noumeno`/`ner` take
  `cogno_anima.BaseStage` (reused — the anima owns it), `id_stage`/`ego`/`superego` take the new
  `IDStageProtocol`/`EgoStageProtocol`/`SuperegoStageProtocol` (`cogno_soma.stages`, exported at
  the package root). They were typed with the CONCRETE cogno-anima classes, so the documented
  "swap any stage for anything matching its signature" held at runtime and was refused by the type
  checker: a host type-checking its own harness carried one `arg-type` error per injected stage,
  with no correct double able to clear them. The Pipeline annotates its own defaults with the same
  protocols, so `mypy cogno_soma` now fails if a cogno-anima stage stops matching the contract.
  Types only — no runtime behaviour changes. `tests/unit/test_stage_protocols.py` pins that the
  parameters stay structural, that the anima defaults and this repo's doubles satisfy them, and
  that the probe refuses a double missing a called member. See `docs/HOST_INTEGRATION.md` §6.

- **A proposal turn whose held call sends text to a person is judged (#49).** The correction
  loop skips the judge on a gate-B hold, deliberately: the action is incomplete and the judge
  would reject the hold itself. For a held MESSAGE that skip was the defect. The text is final
  at the hold (the confirmed replay sends those exact bytes), so its only review ran after
  delivery. Measured on a downstream host: 3 of 4 messages delivered to staff were wrong.
  When `cogno_anima.types.held_delivered_texts(ctx)` is non-empty (the host declares
  `mk.HELD_DELIVERED_TEXT`), the proposal turn is judged like any other: approved → the host
  proposes it; rejected → the EGO rewrites it within the same budget; still rejected → the loop
  ends unapproved and nothing is proposed. Every other hold keeps today's skip, byte for byte.
  Requires cogno-anima with `held_delivered_texts` (sudoers-ai/cogno-anima#183). See
  `docs/HOST_INTEGRATION.md` §4.2.

## 0.1.0 — 2026-07-25

First public release on PyPI.

The reference orchestrator for the Cogno cognitive pipeline — wires cogno-anima's stages (NOUMENO→NER→ID→EGO⇄SUPEREGO→Drift) end-to-end with the correction loop, PII/scope/handoff gates, an interception seam and atomicity hooks. Infra-agnostic: the host injects backends, the dispatcher and prompts.
