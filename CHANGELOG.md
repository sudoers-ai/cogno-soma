# Changelog

## 0.1.4 — 2026-10-13 — the ledger records what the judge's reply needed repaired; the guard's repairs key is cleared with the others

Needs cogno-anima 0.1.9, which counts what each JSON stage had to repair in a model's reply
(`cogno_anima.repairs`). This release carries two of those lists to where a host reads them.
Nothing is decided from them: the loop, the budget and every verdict are what they were.

### Added

- **`judge_attempts[i]["repairs"]`: what the deciding judge reply needed repaired, per attempt.**
  `SuperegoStage.evaluate` reports it on `SuperegoResult.repairs` — a chain-of-thought block or a
  code fence stripped, the object dug out of prose, a critique that was missing on a rejection.
  The loop kept the result and dropped the field, as it once dropped `verdict_read`, so a judge
  that answers in a fence on every turn and one that answers a bare object left the same ledger.
  - **`fast_repairs`** is present only when a fast judge ran first and the attempt escalated
    past it. It is written on every escalation, `[]` included, so the rate has its denominator.
    When the fast judge approved, its list is `repairs`.
  - **Closed.** The values go through anima's `closed_repairs` with the judge's stage. A mark of
    another stage, or any other string a stand-in reports, is dropped.
  - **Absent is "not on record".** The key is absent when the judge reported no read, when its
    result carries no list, and on a `skipped` row. A present `[]` means the judge was asked
    and its reply was read as it came.
  - **It does not repeat `verdict_read`.** A verdict spelled in a string is `string_bool` there
    and `[]` here. Count a judge's repairs as its marks plus its non-`boolean` reads.
  - The keys are exported: `cogno_soma.LEDGER_REPAIRS`, `cogno_soma.LEDGER_FAST_REPAIRS`.

### Fixed

- **`ctx.metadata["scope_repairs"]` is cleared with the guard's other per-turn keys.**
  cogno-anima 0.1.9 writes a fourth fact about the guard on the metadata. The guard removes it
  on the paths where it asks nothing, but only when it is called. `_clear_the_guards_record` now
  pops it too, at both of its calls. That function clears FIVE keys: the guard's four and the
  selector's record, which 0.1.3 moved into it.
  - **Found by the derivation, not by reading.** `test_the_guards_per_turn_keys_are_cleared.py`
    reads the keys the guard and the selector write from anima's own functions and compares
    them with the keys the pipeline clears. Against cogno-anima 0.1.9 and `4256cb7` it fails:
    1 failed, 669 passed.
  - **Two tests of 0.1.3 move with it, and why.** `test_the_clearings_live_in_one_function`
    counted four cleared keys and now counts five. The derivation expected at least three keys
    from the guard and now expects four. Both state the number of keys the guard writes, and
    that number is one higher.
  - The twins of that file now run over five keys: one per key on the two exits that skip the
    guard, one per key and per hook for a stop before the gates, and one per key, per hook and
    per exit for a carrier that restores its metadata from inside an early hook.

### Changed

- `cogno-anima>=0.1.9`. `Pipeline._judge` (private) returns a third element, the escalated fast
  reply's list.
- CI pins cogno-anima at the commit that carries 0.1.9.

### Tests

- `tests/unit/test_the_ledger_records_what_the_judge_repaired.py`, 31 tests: the twin (a fenced
  approval and a bare one are different rows), every mark of the judge's alphabet reaching the
  ledger as itself, a list per attempt, the two-tier judge, the stand-ins (a free string, a
  mark of another stage, a carrier that is not a list, a result with no list), the skipped row,
  and a control: with the two fields switched off every row is the same minus the keys.
- No model was run. The suite is 715 tests; it was 670.

## 0.1.3 — 2026-10-13 — a stop before the gates carries no selector record either

One small correction next to 0.1.2. A turn nobody stops is byte for byte what it was.

### Fixed

- **A hook that stops the turn BEFORE the gates left the selector's record in place.**
  `scope_options_selection` is what anima's `select_scope_options` picked on a turn the scope
  guard blocked, and the host renders its closed question from it. It is per-turn like the
  guard's three keys, and it was cleared before the gates only. `docs/HOST_INTEGRATION.md`
  said so.
  - Measured on `92c06bb`: a `StopPipeline` from `before_turn`, `after_noumeno`, `after_ner` or
    `after_id` left a carried record on the context, 4 of 4.
  - Now it is the fourth key of `pipeline._clear_the_guards_record`, which `run_turn` calls at
    the top, before any hook fires, and again before the gates. The separate removal that sat
    before the gates is gone: one function holds the list.
  - A turn that runs the selector keeps what the selector wrote, also when a later hook stops
    it. Both clearings are before the selector.
  - **Still not covered**, as in 0.1.2: a hook that restores the keys and then stops the turn
    itself.
  - **A defence, not a served defect**: `SessionRunner` builds a fresh context every turn and
    never carried this key.
  - `tests/unit/test_the_guards_per_turn_keys_are_cleared.py`: the selector's key joins the
    twins by hook and the restored-carrier twins; two controls on a turn that runs the selector;
    the premise that a stop before the gates never reaches it. The derivation by AST now reads
    the keys anima's guard writes AND the one its selector writes, and requires the pipeline to
    clear exactly those. A last test holds that `run_turn` removes none of them itself.

## 0.1.2 — 2026-10-13 — a stop before the gates carries none of the guard's record; a stopped turn leaves no stamp counter; LOGGING.md is the code's list

Three small corrections next to 0.1.1. A turn nobody stops is byte for byte what it was.

### Fixed

- **A hook that stops the turn BEFORE the gates left the guard's three per-turn keys in place.**
  `scope_verdict_read`, `scope_prompt_sha` and `scope_prompt_blocks` were cleared after the ID,
  right before the gates. A `StopPipeline` from `before_turn`, `after_noumeno`, `after_ner` or
  `after_id` ended the turn first, so the context left with whatever it arrived with. 0.1.1
  named this as not covered.
  - Now one function clears them (`pipeline._clear_the_guards_record`) and `run_turn` calls it
    twice: at the top, before any hook fires, and again where the clearing always was.
  - **Why twice, and not moved.** The call at the top cannot see what a hook puts back. A host
    that restores a session's metadata from inside one of those four hooks was covered by the
    clearing before the gates, and a move would have dropped that. Measured: with the second
    call removed, 25 tests fail. 24 are that host's shape (4 hooks × 3 keys × the 2 exits that
    skip the guard), and they pass on `b2bf3c3`; the other reads where the two calls sit.
  - A turn that calls the guard keeps what the guard wrote, also when a later hook stops it.
  - **Still not covered:** a hook that restores the keys and then stops the turn itself.
  - **A defence, not a served defect**, as in 0.1.1: `SessionRunner` builds a fresh context
    every turn and never carried these keys.
  - `tests/unit/test_the_guards_per_turn_keys_are_cleared.py`: one twin per hook and per key,
    the control, the adverse half, the restored-carrier twin, and where the two calls sit in
    `run_turn`, read from its source.
- **A turn a `StopPipeline` stopped left the stamp counter on `ctx.metadata`.** `_stamp` keeps
  its count under `_call_seq` and `_finish` removes it, but a stopped turn does not go through
  `_finish`. The one test there was covered the ordinary exit.
  - Measured on `b2bf3c3`: a stop from `after_noumeno` left `_call_seq == 1`, from `after_id`
    `3`, from `after_ego`, `on_commit` and `on_rollback` `5`, from `after_superego` `4`, and a
    stop from inside the executor left it too.
  - The consequence: a carrier that hands the metadata on made the next turn's first call
    `seq == 4` instead of `1`.
  - Now the `except StopPipeline` of `run_turn` removes it. The `seq` already stamped on each
    metrics row stays. `after_turn` is still not fired on a stop.
  - **Not covered:** an exception that is not `StopPipeline` leaves `run_turn` with the
    counter on the context, as before.
  - `tests/unit/test_the_stamp_counter_never_outlives_the_turn.py`: one twin per stop point,
    each reading the counter at the instant of the stop first, and the two controls
    (`before_turn`, where nothing was stamped; `after_turn`, where `_finish` already ran).

### Docs

- **`LOGGING.md` is the code's list.** It said the pipeline logs three DEBUG lines and nothing
  at INFO. The package makes 15 log calls for 14 events, one of them at INFO
  (`scope_block_lifted`), and three with a traceback.
  - The page now has one row per event: logger, event, level, the `key=` fields and when.
  - `tests/unit/test_logging_md_lists_what_the_code_emits.py` reads the calls from the source
    by AST and compares them with the rows, in both directions. A new call fails it until the
    page says what it is. The modules that log nothing and the level policy are held the same
    way.
  - It also corrects a sentence: the page said only the control-flow outcome is logged. Tool
    NAMES (`read_retry_granted`), a model name (`judge_escalated`) and three tracebacks are
    logged too, at DEBUG.
- `docs/HOST_INTEGRATION.md` §4: the paragraph on the guard's keys says what a stop before the
  gates now does, and what it does not.

## 0.1.1 — 2026-10-13 — a stop from inside the executor leaves that pass on record; a turn that skips the guard carries none of its prompt record

The first version number since 0.1.0. Everything under "Unreleased" below is in this tree too
and was never given a number of its own.

### Fixed

- **A `StopPipeline` raised from INSIDE the executor's `process` left that pass unrecorded**
  (2026-10-13). `Pipeline(ego=…)` takes any stage of the documented shape, and a host wraps the
  executor: the inner stage runs to its end, then the wrapper stops the turn. The loop recorded
  a pass on the line after the `await`, which a stop never reaches.
  - Before: the pass's `StageMetrics` row kept `seq == 0` and no host label (`prompt_sha == ""`;
    `attempt == 0` too, for an executor that does not stamp its own), and what the pass
    executed never joined `ctx.turn_executions`.
  - Now: the row has its `seq`, its `attempt` and the host's `ego` label, and the executions
    are on the turn, once, after those of any earlier attempt.
  - **`committed_this_turn` did not change its answer.** It reads the surviving
    `ctx.ego_result` as a floor, so it was already `True` on such a turn. A reader of
    `turn_executions` alone, and a reader of the row, are who see the difference.
  - One function records a pass (`_record_ego_pass`), on both roads: after `process` returned,
    and in the `except StopPipeline` around it.
  - Only a result THAT call produced is recorded. A stage that stops before replacing
    `ctx.ego_result` leaves the previous attempt's row and executions as they were.
  - Only `StopPipeline` is handled. `asyncio.CancelledError` and every other exception leave
    `run_turn` as before, and that pass is not recorded.
  - A record that cannot be made never turns the stop into a crash.
  - No ledger row is written for that pass: the judge did not read it.
  - **Out of reach:** a `StopPipeline` raised by a TOOL. cogno-anima's executor wraps whatever
    escapes `dispatcher.execute` in `ToolExecutionError`, so it leaves `run_turn` as an error
    with no result built. Pinned as a boundary, not repaired.
  - A turn nobody stops is byte for byte what it was.
    `tests/unit/test_a_stop_inside_the_executor_is_recorded.py`.
- **A turn that never calls the scope guard could carry a stale record of the guard's prompt**
  (2026-10-13). cogno-anima's guard leaves three per-turn facts on `ctx.metadata`:
  `scope_prompt_blocks`, `scope_prompt_sha` and `scope_verdict_read`. It keeps them honest
  itself, but only when it is called, and two exits of `run_turn` never call it: a turn with no
  scope prompt, and a turn the PII gate ends first. The third key was cleared before those
  exits already; the other two were not.
  - Now `scope_prompt_sha` and `scope_prompt_blocks` are cleared at the same place, so on those
    two exits both are ABSENT whatever the context arrived with.
  - A turn that calls the guard keeps what the guard wrote, as before. `[]` (the guard was
    called and built no prompt) stays apart from absent (the guard was not called).
  - **A defence, not a served defect.** `SessionRunner` builds a fresh context every turn and
    carries a whitelist these keys are not on, so no carrier shipped here could deliver a stale
    one. A host that reuses a context, or carries metadata its own way, is who this protects.
  - A derivation pins the rule: every metakey the guard's function assigns or removes must be
    one `run_turn` removes. A fourth key fails the test until it is cleared too.
  - **Not covered:** a hook that raises `StopPipeline` before the gates (`before_turn` to
    `after_id`) ends the turn before this clearing, for all three keys.
  - `tests/unit/test_the_guards_per_turn_keys_are_cleared.py`.

## Unreleased

Entries here start on 2026-09-24. Earlier changes since 0.1.0 are in the git history only.

### Added

- **The per-attempt ledger records what each attempt's EXECUTOR PROMPT carried** (2026-10-07;
  needs cogno-anima ≥ 0.1.5, and the dependency floor says so). `EgoStage.process` now records
  its own prompt on the `EgoResult`: the parts and their lengths, a digest, and the path the
  catalogue took. This loop replaces `ctx.ego_result` on every retry, so only the surviving
  attempt's record reached a reader. A retry's prompt is not the first attempt's: it gains
  `correction`, and `actions_done` when the rejected attempt wrote.
  - `judge_attempts[i]["ego_prompt_blocks"]` — `[{"block", "chars"}]`, the slugs from
    `cogno_anima.EGO_PROMPT_BLOCKS`. At most 16 rows.
  - `judge_attempts[i]["ego_prompt_sha"]` — anima's digest of what the attempt started from.
  - `judge_attempts[i]["ego_prompt_path"]` — `native` or `fallback`
    (`cogno_anima.VALID_EGO_PROMPT_PATHS`).
  - `judge_attempts[i]["ego_prompt_blocks_dropped"]` — only when a row was left out: a slug
    outside the alphabet, a length that is not a non-negative integer, or a row past the 16th.
  - Written on BOTH places a row is written: a judged attempt and a pre-judged replay.
  - Every key is ABSENT when there is nothing on record: a stand-in executor, an `EgoResult`
    without the fields, or anima's own "not on record". Never `[]` by default.
  - Never the prompt's text. `EgoResult.prompt_text` stays in memory; nothing here reads it.
  - The keys are exported: `LEDGER_EGO_PROMPT_BLOCKS`, `LEDGER_EGO_PROMPT_BLOCKS_DROPPED`,
    `LEDGER_EGO_PROMPT_SHA`, `LEDGER_EGO_PROMPT_PATH`.
  - **The hole this does not close:** a gate-B or gate-C hold with no delivered text leaves the
    loop before any ledger row is written. That turn has no row, and the surviving
    `ctx.ego_result` is the only record of its executor prompt.
  - Nothing a turn decides changes. `tests/unit/test_the_ledger_records_the_executors_prompt.py`.
  - **CI pin:** `.github/workflows/ci.yml` installs cogno-anima at `55788c8` (0.1.5, the release
    that carries the record) in both install steps; the other three sibling pins do not move.
- **`SessionRunner.run` records the layers it composed into the context** (2026-10-07), on
  `ctx.metadata["context_layers"]` (`cogno_soma.CONTEXT_LAYERS_KEY`), per turn:
  `[{"block", "chars", "carrier"}]`, in the order composed.
  - `block` is one of `cogno_soma.CONTEXT_LAYER_SLUGS`: `sources`, `recent_conversation`,
    `earlier_context`, `memories`, `knowledge_graph`.
  - `carrier` is one of `cogno_soma.CONTEXT_CARRIERS`: `context` (`ego_context`) or
    `context_data` (`ego_context_untrusted`). Without `split_context` every layer is on the
    first. The two names are the slugs cogno-anima gives those two parts of the executor's
    prompt.
  - `chars` is the layer as it sits on the carrier, label included. On each carrier,
    `sum(chars) + 2 * (rows - 1)` is the length of what the runner put there. What a host
    finds on that carrier beyond that is text somebody else prepended.
  - The record is the composer's: a `[KNOWLEDGE GRAPH]` line typed inside a memory adds no row.
  - Written after `metadata=` is merged. A carrier the host replaced through `metadata=` has no
    layers on record, and a `context_layers` key a caller hands in is overwritten.
  - ABSENT when the context was not composed here (`Pipeline.run_turn` driven directly). Not
    carried to the next turn and not in `sess.state`.
  - **The composed context did not change by a byte**: 32 configurations (transcript ×
    earlier context × memories × graph × split) have the digest they had on `f56a09d`.
    `tests/unit/test_the_composer_records_its_layers.py`.
- **The per-attempt ledger records HOW the judge's verdict was read** (2026-10-07; needs
  cogno-anima ≥ 0.1.3, and the dependency floor says so). anima now reads the judge's verdict
  strictly — only a JSON boolean counts — and reports the read on `SuperegoResult.verdict_read`.
  This loop kept the result and dropped that field, so a rejection the judge GAVE and a rejection
  anima fell back to (a verdict in a string, a duplicated key, no JSON) were the same ledger row,
  and the rate of non-boolean verdicts per model could not be counted from anything a host
  persists.
  - `judge_attempts[i]["verdict_read"]` — the read of the verdict that DECIDED the attempt, a
    value of `cogno_anima.VALID_VERDICT_READS`. Only the value; never a byte of the reply.
  - `judge_attempts[i]["fast_verdict_read"]` — only when a fast judge ran first and the attempt
    escalated past it. A fast verdict that cannot be read is a fast rejection, so it escalates:
    the right repair, which without this key would erase the only record of it.
  - Both keys ABSENT when there is nothing to say: a stand-in judge that reports no read, a
    `skipped` row (no judge was called), a label outside the alphabet.
  - The keys are exported — `cogno_soma.LEDGER_VERDICT_READ`, `LEDGER_FAST_VERDICT_READ` — so a
    host persisting the ledger reads them from here and the values from anima.
  - **The guard's read** (`ctx.metadata["scope_verdict_read"]`, anima's) is cleared before the two
    exits that never call the guard (no scope prompt; the PII gate), so ABSENT means "no verdict
    was asked this turn" on those turns too.
  - Nothing a turn DECIDES changes: `_judge` returns the same verdict (and, beside it, the fast
    read). `tests/unit/test_the_ledger_records_how_the_verdict_was_read.py`.
  - **CI pin:** `.github/workflows/ci.yml` installs cogno-anima at `1923c85` (0.1.3, the release
    that carries `verdict_read`) in both install steps; the other three sibling pins do not move.

### Fixed

- **The stage stamp ERASED the `prompt_sha` of NOUMENO and NER** (2026-10-07). `_stamp` wrote
  `metrics.prompt_sha = prompt` unconditionally, from a keyword that defaults to `""`. NOUMENO and
  NER author their own templates and stamp the digest of them on their own metrics
  (`cogno_anima.prompts.prompt_digest` — the layer that authors a text owns its identity); the
  pipeline stamps those two with no label, because the host has no slot for a text it did not
  write, so the default overwrote the digest. Measured downstream: `stages[noumeno|ner].prompt_sha`
  empty on 141 of 141 traces over three days — no prompt A/B and no reversal could be read off a
  trace for those two stages.
  - **The rule is about the class, not the field: the stamp writes a field only when the
    orchestrator HAS a value for it.** `_stamp` writes three fields. `seq` — always; only this
    layer knows the call order. `attempt` — only when the caller passes one (`None`, the keyword
    left out, writes nothing; any value passed is written, an explicit `0` included).
    `prompt_sha` — only when the host labelled the slot (an empty label erases nothing). `attempt` had the
    same shape (assigned from a default of `0`) and no victim: the EGO is the only stage that
    sets its own, and its one call site always passes the loop's count.
  - **Red → green** with the REAL anima stages through the real pipeline (a double has no digest
    to lose): on the tree before, `'' == 'e8b9344aed58'`; after, each stage's `prompt_sha` equals
    `prompt_digest` over the template it used, and two different contacts get the same label (a
    digest of the template, not of what was rendered).
  - **Unchanged:** the four slots the HOST labels (`scope`, `ego`, `judge`, `voice`, from
    `mk.PROMPT_SHAS`) arrive as before. A stage with no template (the ID, the scope selector)
    stays `""` — the type's own "nothing a deployment set". A stage that did not run has no
    metrics row at all.
  - **A chosen behaviour, declared: on `attempt`, the LOOP's count is what the row carries.** The
    real EGO reads its attempt from `mk.EGO_CORRECTION`, which this loop writes, so inside the
    loop the two are the same number on every pass. They can differ on one path only: a host that
    seeds that metakey with an attempt other than 1 BEFORE the turn. There the executor reports
    the host's number on `EgoResult.attempt` and the stamped row says the loop's — the axis the
    judge ledger and every other row of the turn are on. Pinned by a test over seeds 1, 2 and 5.
  - **Replay, with its ruler.** The erased digest is not recoverable from a trace that stored
    `""` — the erased value is the missing one. What IS determinable without re-running a turn
    is on how many of those rows the stage actually RAN: anima stamps the digest on every call
    that returns, so each of those carries one after this fix. Counted on a downstream host's
    exported stage rows (the traces of three days that hold at least one `noumeno` row): **141
    of 141 `noumeno` rows and 141 of 141 `ner` rows — in 117 traces, 19 of them with more than
    one pass — ran the stage** (a model answered, input tokens above zero) with an empty
    `prompt_sha`, **and would carry it. 0 rows are a seated result.** The unit is the stage
    ROW, not the trace. The other shape — a turn whose perception the host SEATED
    (`TurnConfig.noumeno_result` / `intent_result`, no model call), which keeps `""` because no
    template ran — does not occur in that window: it is proven by the test, not by the corpus.
    The same export confirms the controls: the ID's row is empty on 141 of 141 (by design),
    and the executor, the judge and the scope guard carry the host's label.
  - **Not fixed here, declared:** the pre-judge (`judge_pre`) rows carry no digest either, for
    a different reason — `cogno_anima.stages.ProposalJudge` never stamps one on its metrics,
    and those rows do not pass through this stamp at all (a host files them). That is a change
    in cogno-anima, queued.
  `tests/unit/test_the_stamp_fills_and_never_erases.py`: the twin, the controls, each field one
  by one, and the nine `_stamp` call sites read off the source.

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
  - Needs a `cogno-anima` with `mk.EGO_CONTEXT_UNTRUSTED` (anima #213).

- **This repo's CI pins every sibling library by SHA** (2026-10-07), in both install steps of
  `ci.yml`. Until now each was installed from its `main` with no SHA, so a run was not
  reproducible and a sibling's landing could turn this suite red with nobody having touched it.
  - `cogno-anima` `8f3970b8` (the #213 merge: `run(split_context=True)` needs the metakey);
  - `cogno-synapse` `d2766784`, `cogno-homeo` `9ff238c7`, `cogno-engram` `c3201fe1`: the SHAs
    `cogno-host`'s own `ci.yml` pins on its `main` `ffbf920e`.
  - Moving a pin is now a deliberate edit of `ci.yml`. Nothing moves them automatically, so they
    age until someone does.
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
