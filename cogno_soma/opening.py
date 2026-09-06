"""The turn the AGENT opens — perception stand-ins for a turn nobody spoke.

A proactive turn has no user utterance. What the caller holds is its OWN directive ("open the
conversation with Ana, mention the checkup"), and feeding that through NOUMENO/NER — the two
stages that exist to understand what a *person* said — is not merely wasteful, it is actively
wrong. Five failures, each of them a stage doing its job correctly on the wrong input:

* **PII.** The NER classifies our directive. A word like "checkup" reads as ``HEALTH_DATA``,
  which is CRITICAL, which makes the ID set ``blocked`` — the turn dies as ``pii_blocked`` and
  the agent never opens. Our own privacy guard censoring our own instruction.
* **Goal.** The ID records the directive as the *contact's* goal, so when the human finally
  replies, goal continuity compares their words against our command and reads it as a topic
  change — right at the moment continuity matters most.
* **Scope.** The pre-EGO guard judges whether our own sentence is on-topic for the deployment.
  (That one the caller turns off with ``scope_prompt=""``; it is named here because it belongs
  to the same list.)
* **Drift.** Every drift number for the turn is measured against our prompt.
* **Continuity.** With a session behind it, the NOUMENO compares the user slot against
  ``metadata[LAST_REWRITTEN]`` — i.e. our internal marker against the contact's last real
  message — and decides ``change_subject`` from that. Two embedding calls to answer a question
  about a sentence nobody wrote.

So the caller hands the two results in (``TurnConfig.noumeno_result`` / ``intent_result``) and
the pipeline seats them instead of running the stages. Nothing is called: no LLM, no embedder,
no tokens.

**These are typed results, deliberately, and not a canned model reply.** The shape this
replaces stood a fixed JSON document behind a fake backend and let the stages parse it — which
looks equivalent and is not, in three ways that were each measured before this module existed:

1. every field is validated at CONSTRUCTION here, whereas a JSON key the sanitizer does not
   read is silently dropped. The host version this was rebuilt from carried ``raw_goal``, a key
   that exists only in the NER *prompt* and in no result — it had been inert since the day it
   was written, next to a comment saying the fields had been checked;
2. the values the sanitizer DERIVES (``mandatory_tags`` falling back to ``NER.UNKNOWN``, an
   empty goal becoming ``None``, ``pii_risk`` computed from ``pii``) are stated here instead of
   being implied by an input that produces them. What a reader sees is what the turn gets;
3. the stages ran anyway. The NOUMENO still embedded, still measured drift against itself,
   still compared our marker to the contact's history — so "no model call, no tokens" was not
   true of the path that claimed it.

**What is a DECISION lives here; what is a FACT about the turn comes from the context.**
``original`` is whatever the caller put in the user slot, ``language`` is the language the
caller forced, ``langue`` is inherited from the NOUMENO — exactly as the real stages take them
— so a stand-in can never disagree with the turn it is standing in for. The pipeline fills
those in when it seats the result; everything else is stated below and is a choice.
"""

from __future__ import annotations

from typing import Any

from cogno_anima.types import IntentResult, NoumenoResult, StageMetrics

# Stamped as the ``model`` of both stand-ins. A synthetic result that looked model-authored
# would send the next person debugging a turn hunting a hallucination that never happened.
OPENING_MODEL = "opening:synthetic"

# The NER's ``intent_class`` for an opening. INFORMATION_REQUEST routes to the EGO — an opening
# may legitimately need to look something up ("your appointment is Thursday") — while
# ACTION_REQUEST would ALSO route there but forces ``tool_choice="required"`` on the first
# iteration, demanding a tool call from an opening that usually needs none. SOCIAL would route
# to the SUPEREGO and take the executor off the table entirely.
OPENING_INTENT = "INFORMATION_REQUEST"


def _metrics(stage: str) -> StageMetrics:
    """A no-cost row. Zero everywhere is the point: nothing ran."""
    return StageMetrics(stage=stage, elapsed_ms=0.0, tokens_in=0, tokens_out=0,
                        model=OPENING_MODEL)


def opening_noumeno() -> NoumenoResult:
    """The NOUMENO stand-in for an opening turn.

    ``original``/``rewritten``/``language`` are left EMPTY on purpose — the pipeline fills them
    from ``ctx.user_input`` and ``ctx.force_language`` when it seats this result, the same way
    the real stage takes them. Hardcoding them here would let a stand-in claim the turn carried
    text it did not.

    Everything else is a decision: nothing was rewritten, so drift is 0 and the tag is
    ``PASS_THROUGH``; there is no history comparison to make, so continuity is a clean 1.0;
    and ``rewrite_warnings`` stays empty because a warning there raises the ID's
    ``clarification_suggested`` signal on a turn where there is nothing to clarify.
    """
    return NoumenoResult(
        original="", rewritten="", context_turn="",
        language="", canonical_language="en",
        drift_score=0.0, drift_tag="PASS_THROUGH", changed=False, confidence=1.0,
        change_subject=False, subject_similarity=1.0, context_used=False,
        preserved_terms=[], rewrite_warnings=[],
        metrics=_metrics("noumeno"),
    )


def opening_intent() -> IntentResult:
    """The NER stand-in for an opening turn.

    ``langue`` is left EMPTY and inherited from the seated NOUMENO, exactly as the real stage
    inherits it.

    The two fields that decide whether the turn lives at all:

    * ``pii``/``pii_risk`` — empty and ``NONE``. Any PII read off our own directive becomes the
      ID's ``blocked``, and the opening never happens.
    * ``goal`` — ``None``. The human's reply is what establishes a goal; seeding it with our
      directive is what makes their first answer look like a topic change.

    ``mandatory_tags`` is ``["NER.UNKNOWN"]`` and ``domains`` is empty, together. A tag here
    would be mapped to a knowledge domain (SYSTEM → TECH) and that invented domain would then
    ride ``active_domains`` into the contact's first real reply; ``NER.UNKNOWN`` is the honest
    answer for a turn nobody spoke, and it is written out rather than left to a fallback.
    """
    return IntentResult(
        intent_class=OPENING_INTENT,
        sentiment="NEUTRAL",
        confidence=1.0,
        temporal_class="TIMELESS",          # an opening is not anchored in time
        triad_signal="EGO",
        entities_people=[], entities_objects=[], entities_concepts=[], location=None,
        mandatory_tags=["NER.UNKNOWN"],
        aristotelian={},
        domains=[],
        goal=None,
        causal_chain=[],
        parole="FORMAL",
        langue="",
        negation=[], constraints=[],
        modality="CERTAIN",
        speech_act="CONSTATIVE",
        verbs=[],
        context_dependent=False,
        is_composite=False,
        is_sequential=False,
        pii=[], pii_risk="NONE",
        metrics=_metrics("ner"),
        raw_response="",                    # nothing answered; a canned document is not a reply
    )


def opening_perception() -> "dict[str, Any]":
    """The two stand-ins in the shape ``dataclasses.replace(cfg, **…)`` takes.

    ``cfg = replace(cfg, scope_prompt="", **opening_perception())`` is the whole opening
    wiring on the perception side: the two results, and the scope guard turned off because a
    guard whose input is our own directive can only ever block us.

    ``Any``, not ``object``: this is a kwargs bag unpacked into ``dataclasses.replace`` on a
    typed dataclass, and ``object`` is not assignable to ``NoumenoResult`` — nor to any OTHER
    field of that dataclass, which is what a type checker has to assume about ``**bag``. The
    host this was rebuilt from had already learned it once, on the function this replaces, and
    only in CI: a local editable install collapses the sibling lib to ``Any`` and hides it.
    """
    return {"noumeno_result": opening_noumeno(), "intent_result": opening_intent()}
