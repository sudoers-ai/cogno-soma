# Changelog

## Unreleased

Entries here start on 2026-09-24. Earlier changes since 0.1.0 are in the git history only.

### Changed

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
