# Changelog

## Unreleased

Entries here start on 2026-09-24. Earlier changes since 0.1.0 are in the git history only.

### Changed (docs)

- **`docs/HOST_INTEGRATION.md` §4.1 lists every key of a judge-ledger entry** (Phase 2 docs
  sweep B). The prose named four of them — verdict, critique, draft, calls — while an entry also
  carries `draft_len`, `committed`, `tools_offered`, the two `*_dropped` counts, `tools_error` and
  (#52) `branch`. The section is now a table: each key, what it is, and the function in
  `cogno_soma/pipeline.py` that writes it, with the tests that pin them. The absence of `branch`
  is also stated in full: a stand-in judge that does not classify, OR the anima's `evaluate`
  returning before it chooses because nothing executed. Docs only.

### Added

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
