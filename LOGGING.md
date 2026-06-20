# Logging in cogno-soma

This library follows the Cogno house rule: **libraries emit, the host configures.**

- Each module does `logger = logging.getLogger(__name__)` and emits lazy
  `key=value` messages. The library installs **no** handlers/formatters and never
  calls `basicConfig`.
- The host attaches its handler and sets the level per package, e.g.
  `logging.getLogger("cogno_soma").setLevel(logging.INFO)`.

## Level policy
- **ERROR** — never emitted. soma propagates stage/backend exceptions to the host;
  `SessionRunner` raises `SomaError` for a missing dispatcher. The host decides how
  to surface failures.
- **WARNING** — none from soma itself (the stages emit their own warnings, e.g. the
  SUPEREGO judge rejection; soma does not duplicate them).
- **INFO** — none.
- **DEBUG** — one line per terminal outcome of a turn: `turn_blocked`
  (`stop_reason=pii_blocked|scope_blocked`), `turn_handoff`
  (`stop_reason=human_handoff`), `turn_stopped` (a host hook raised `StopPipeline`,
  with its `stop_reason`). The happy path is silent.

## What gets logged
- `cogno_soma.pipeline` — DEBUG `turn_blocked` / `turn_handoff` / `turn_stopped`.
- `cogno_soma.{session,config,hooks,errors}` — nothing.

User text, prompts, tool arguments and gathered data are **never** logged — only
the control-flow outcome (`stop_reason`). Per-turn telemetry (tokens, latency) lives
on `ctx`'s `StageMetrics`, not in logs.
