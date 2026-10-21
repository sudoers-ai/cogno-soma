# Logging in cogno-soma

This library follows the Cogno house rule: **libraries emit, the host configures.**

- The one module that logs, `cogno_soma.pipeline`, takes its logger from
  `logging.getLogger(__name__)` and emits lazy `key=value` messages. The library installs
  **no** handlers/formatters and never calls `basicConfig`.
- The host attaches its handler and sets the level per package, e.g.
  `logging.getLogger("cogno_soma").setLevel(logging.INFO)`.

## Level policy

- **ERROR** — never emitted. soma propagates stage/backend exceptions to the host;
  `SessionRunner` raises `SomaError` for a missing dispatcher. The host decides how
  to surface failures.
- **WARNING** — never emitted by soma itself. The stages emit their own warnings (e.g. the
  SUPEREGO judge rejection); soma does not duplicate them.
- **INFO** — one line, and it is rare: `scope_block_lifted`. The scope guard refused
  the turn and the turn went on anyway, which a host wants to count.
- **DEBUG** — everything else, in three families:
  - how a turn ENDED when it did not simply complete: `turn_blocked`, `turn_exhausted`,
    `turn_handoff`, `turn_clarify`, `turn_stopped`;
  - what the correction loop DECIDED: `judge_skipped`, `judge_escalated`,
    `held_message_rewrite_granted`, `action_retry_granted`, `read_retry_granted`;
  - bookkeeping that FAILED and was survived: `stage_stamp_failed`,
    `attempt_prompt_failed`, `ego_pass_record_failed`.

An ordinary turn logs nothing: no gate fired, nobody stopped it, the judge that was asked
approved at once, and the reply did not end the turn as a clarification.

## Events

This table is the code's list. `tests/unit/test_logging_md_lists_what_the_code_emits.py`
reads every `logger.<level>(…)` call of the package and compares it with the rows below,
in both directions: the logger, the event, the level, the `key=` fields and whether the
line carries a traceback.

| Logger | Event | Level | When |
|---|---|---|---|
| `cogno_soma.pipeline` | `turn_blocked` (`stop_reason=pii_blocked\|scope_blocked`) | DEBUG | the PII gate or the scope guard ended the turn before the executor ran |
| `cogno_soma.pipeline` | `scope_block_lifted` (`reason=selector_covered`) | INFO | the scope guard blocked, the options selector answered that a listed option covers the request, and the turn goes on |
| `cogno_soma.pipeline` | `turn_exhausted` (`stop_reason=judge_exhausted`, `kind=unverified_claim\|not_executed`) | DEBUG | the correction budget ran out with nothing committed for the contact; the voice still writes a reply |
| `cogno_soma.pipeline` | `turn_handoff` (`stop_reason=human_handoff`) | DEBUG | the judge rejected an execution that committed something for the contact; the turn ends without a voiced reply |
| `cogno_soma.pipeline` | `turn_clarify` (`stop_reason=needs_clarification`) | DEBUG | a completed ACTION turn that wrote nothing for the contact, held nothing for confirmation, and whose reply ends as a question |
| `cogno_soma.pipeline` | `turn_stopped` (`stop_reason`) | DEBUG | a hook or a stage raised `StopPipeline`; `stop_reason` is the reason it gave |
| `cogno_soma.pipeline` | `judge_skipped` (`reason=prejudged_replay`) | DEBUG | the judge did not read this attempt: it replays a held message the judge already approved |
| `cogno_soma.pipeline` | `judge_escalated` (`fast_model`) | DEBUG | the fast judge rejected (or could not be read) and the strong judge is asked; `fast_model` is the backend's model name |
| `cogno_soma.pipeline` | `held_message_rewrite_granted` (`attempt`) | DEBUG | the budget was spent and ONE more executor pass was granted to rewrite a held message the judge rejected |
| `cogno_soma.pipeline` | `action_retry_granted` (`attempt`) | DEBUG | the budget was spent and ONE more executor pass was granted because the rejected pass owed an action |
| `cogno_soma.pipeline` | `read_retry_granted` (`attempt`, `tools`) | DEBUG | the budget was spent and ONE more executor pass was granted because a source read was offered and never called; `tools` is the NAMES of those tools |
| `cogno_soma.pipeline` | `stage_stamp_failed` | DEBUG, with a traceback | a call could not be given its place in the turn (`seq`, `attempt`, the host's label); the turn goes on with that row unstamped |
| `cogno_soma.pipeline` | `attempt_prompt_failed` | DEBUG, with a traceback | the record of the executor's prompt could not be copied to one ledger row; the row is written without it |
| `cogno_soma.pipeline` | `ego_pass_record_failed` | DEBUG, with a traceback | a `StopPipeline` came from inside the executor and that pass could not be recorded; the stop goes on |

`turn_blocked` is emitted from two places, the PII gate and the scope guard, with the same
fields. It has one row.

## Modules that log nothing

- `cogno_soma`
- `cogno_soma.config`
- `cogno_soma.errors`
- `cogno_soma.hooks`
- `cogno_soma.opening`
- `cogno_soma.session`
- `cogno_soma.stages`
- `cogno_soma.trace_cuts`

## What a line carries

No message formats user text, a prompt, a tool argument or gathered data.

- The values are closed tokens (`stop_reason`, `kind`, `reason`), an attempt number, tool
  NAMES (`tools`) and a model name (`fast_model`).
- `stop_reason` on `turn_stopped` is whatever the hook that stopped the turn passed as its
  reason. A host that puts free text there is logging it.
- Three DEBUG lines carry a traceback. It is the exception of soma's own bookkeeping, and
  it says what that exception says.

Per-turn telemetry (tokens, latency) lives on `ctx`'s `StageMetrics`, not in logs.
