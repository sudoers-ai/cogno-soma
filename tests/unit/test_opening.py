"""The turn the AGENT opens: the stand-in VALUES, and the seat that puts them on a turn.

Two properties, and they fail in different directions.

The **values** are a safety contract: each one is there to stop a specific stage from doing
its job correctly on the wrong input (our own directive), and every one of them fails
SILENTLY — a PII risk read off our instruction blocks the turn, a goal seeded from it makes
the contact's first reply look like a topic change, a fabricated domain rides into the next
turn. So they are asserted one by one, with the failure each prevents named.

The **seat** is a truthfulness contract: what the stand-in may state is a decision; what the
turn actually carried is a fact and comes from the context. A stand-in that could claim the
user slot held text nobody put there would be believed by everything downstream, the stored
transcript included.
"""

from __future__ import annotations

from cogno_anima import vocab
from cogno_anima.security.pii import PII_RISK_LEVELS
from cogno_anima.types import PipelineContext

from cogno_soma import (
    OPENING_INTENT,
    OPENING_MODEL,
    Pipeline,
    TurnConfig,
    opening_intent,
    opening_noumeno,
    opening_perception,
)
from cogno_soma.pipeline import _seat_intent, _seat_noumeno

from tests.conftest import (
    FakeEgo,
    FakeID,
    FakeNER,
    FakeNoumeno,
    FakeSuperego,
)


# ── the values, each with the failure it prevents ─────────────────────────────────────

def test_no_pii_so_the_turn_is_not_blocked_by_our_own_words():
    """The NER classifying OUR directive is how an opening dies: a word like "checkup" reads
    as HEALTH_DATA, which is CRITICAL, which makes the ID set ``blocked``."""
    intent = opening_intent()
    assert intent.pii == []
    assert intent.pii_risk == "NONE"


def test_no_goal_so_the_contact_s_first_reply_is_not_a_topic_change():
    """Seeding the goal with our command makes goal continuity compare the human's answer
    against it — a topic change at the exact moment continuity matters most."""
    assert opening_intent().goal is None


def test_unknown_tags_and_no_domains_travel_together():
    """A tag here is mapped to a knowledge domain by the sanitizer's fallback (SYSTEM → TECH),
    and that invented domain then rides ``active_domains`` into the contact's first real
    reply. NER.UNKNOWN is the honest answer for a turn nobody spoke."""
    intent = opening_intent()
    assert intent.mandatory_tags == ["NER.UNKNOWN"]
    assert intent.domains == []


def test_the_intent_class_reaches_the_executor_without_demanding_a_tool():
    """INFORMATION_REQUEST routes to the EGO — an opening may need to look something up —
    while ACTION_REQUEST would also route there and force ``tool_choice="required"`` on the
    first iteration, demanding a call from a turn that usually needs none. SOCIAL would take
    the executor off the table entirely."""
    assert OPENING_INTENT == "INFORMATION_REQUEST"
    assert opening_intent().intent_class == OPENING_INTENT
    assert opening_intent().triad_signal == "EGO"


def test_nothing_was_rewritten_so_there_is_no_drift():
    n = opening_noumeno()
    assert n.changed is False
    assert n.drift_score == 0.0
    assert n.drift_tag == "PASS_THROUGH"


def test_no_rewrite_warning_so_the_id_asks_for_no_clarification():
    """A warning here raises the ID's ``clarification_suggested`` signal on a turn where
    there is nothing to clarify."""
    assert opening_noumeno().rewrite_warnings == []


def test_both_stand_ins_are_stamped_as_synthetic_and_cost_nothing():
    """A synthetic result that looked model-authored sends the next person debugging a turn
    hunting a hallucination that never happened. And the cost is zero on every axis,
    EMBEDDINGS INCLUDED — which is the half the shape this replaced could not deliver."""
    for result in (opening_noumeno(), opening_intent()):
        m = result.metrics
        assert m.model == OPENING_MODEL
        assert (m.tokens_in, m.tokens_out, m.tokens_total) == (0, 0, 0)
        assert (m.embedding_tokens, m.embedding_calls) == (0, 0)
        assert m.prompt_sha == ""          # no prompt ran; an unlabelled call is the truth
    assert opening_intent().raw_response == ""   # nothing answered


# ── every value is IN the closed vocabulary ───────────────────────────────────────────

def test_every_stated_value_is_in_the_closed_vocabulary():
    """The reason these are constructed and not parsed. A JSON document handed to the NER's
    sanitizer has its off-vocab values silently coerced away, and the stand-in this replaced
    shipped a key (``raw_goal``) that no result has ever carried — inert since the day it was
    written, beside a comment claiming the fields had been checked. Checked here, against the
    vocabulary module rather than against a memory of it."""
    intent = opening_intent()
    assert intent.intent_class in vocab.VALID_INTENTS
    assert intent.sentiment in vocab.VALID_SENTIMENTS
    assert intent.temporal_class in vocab.VALID_TEMPORAL
    assert intent.triad_signal in vocab.VALID_TRIAD
    assert intent.modality in vocab.VALID_MODALITY
    assert intent.speech_act in vocab.VALID_SPEECH_ACTS
    assert intent.parole in vocab.VALID_PAROLE
    assert intent.pii_risk in PII_RISK_LEVELS
    assert set(intent.domains) <= set(vocab.NER_KNOWLEDGE_DOMAINS)
    assert {t.split(".", 1)[-1] for t in intent.mandatory_tags} <= vocab.VALID_MANDATORY
    # The drift tag has no vocabulary CONSTANT to check against — it is enumerated in a
    # comment on `NoumenoResult.drift_tag` and produced by `Noumeno._classify_drift`. So the
    # check is against the producer: whatever that function can return for a 0.0 drift on an
    # unchanged text is what an unrewritten stand-in must say.
    from cogno_anima.stages.noumeno import classify_drift
    assert opening_noumeno().drift_tag == classify_drift(opening_noumeno().drift_score)


# ── the seat: decisions from the stand-in, facts from the turn ────────────────────────

def test_the_user_slot_comes_from_the_TURN_not_from_the_stand_in():
    """``original`` IS ``ctx.user_input`` — that is what the real stage sets it to. A
    stand-in able to hardcode it could claim the turn carried text nobody put there, and the
    stored transcript would keep that text."""
    ctx = PipelineContext(user_input="[OPENING]")
    seated = _seat_noumeno(opening_noumeno(), ctx)
    assert seated.original == "[OPENING]"
    assert seated.rewritten == "[OPENING]"      # the stage's own empty-rewrite fallback

    # …and the marker is the CALLER's word, not ours: this library ships no text at all, so
    # any string in the user slot must come back out of the result unchanged.
    other = _seat_noumeno(opening_noumeno(), PipelineContext(user_input="<<agent speaks first>>"))
    assert other.original == other.rewritten == "<<agent speaks first>>"


def test_the_language_comes_from_the_TURN_and_the_ner_inherits_it():
    ctx = PipelineContext(user_input="[OPENING]", force_language="pt-BR")
    ctx.noumeno = _seat_noumeno(opening_noumeno(), ctx)
    assert ctx.noumeno.language == "pt-BR"
    assert _seat_intent(opening_intent(), ctx).langue == "pt-BR"


def test_a_value_the_caller_DID_state_is_kept():
    """The fallback fills a gap; it does not overrule. A caller with a real rewrite or a
    language of its own keeps them."""
    n = opening_noumeno()
    n.rewritten, n.language = "good morning", "es"
    seated = _seat_noumeno(n, PipelineContext(user_input="<<agent speaks first>>", force_language="pt-BR"))
    assert seated.rewritten == "good morning"
    assert seated.language == "es"


def test_original_is_NEVER_a_choice_even_when_the_stand_in_states_one():
    """The twin of the test above, and the one that has teeth. Everything else on the result
    is a decision the caller may state; ``original`` is a fact about the turn, so a stand-in
    that names one must still lose to ``ctx.user_input``. Written as an OVERRIDE rather than
    a fallback because a fallback here reads identically until the day someone fills the
    field — and then the stored transcript keeps a sentence the contact never wrote."""
    n = opening_noumeno()
    n.original = "please open the conversation and mention the checkup"
    seated = _seat_noumeno(n, PipelineContext(user_input="<<agent speaks first>>"))
    assert seated.original == "<<agent speaks first>>"


def test_langue_is_never_a_choice_either_when_the_noumeno_has_spoken():
    """Same shape on the NER half: the two stand-ins describe ONE turn and must not be able
    to disagree about its language."""
    ctx = PipelineContext(user_input="[OPENING]", force_language="pt-BR")
    ctx.noumeno = _seat_noumeno(opening_noumeno(), ctx)
    assert _seat_intent(opening_intent(), ctx).langue == "pt-BR"


def test_the_seat_COPIES_so_two_turns_never_share_one_object():
    """A config is built once and reused. Seating the caller's own object would let one
    turn's stamped ``seq`` — and one turn's text — leak into the next."""
    perception = opening_perception()
    a = _seat_noumeno(perception["noumeno_result"], PipelineContext(user_input="A"))
    b = _seat_noumeno(perception["noumeno_result"], PipelineContext(user_input="B"))
    assert (a.original, b.original) == ("A", "B")
    assert a is not b and a is not perception["noumeno_result"]
    assert a.metrics is not b.metrics
    assert perception["noumeno_result"].original == ""   # the template is untouched


def test_opening_perception_is_the_shape_replace_takes(stub_backend):
    """`dataclasses.replace(cfg, scope_prompt="", **opening_perception())` is the whole
    perception side of an opening. Pinned as the keys, because a rename here breaks a caller
    with a `TypeError` at turn time and nothing earlier."""
    import dataclasses

    cfg = TurnConfig(gen_backend=stub_backend, ego_backend=stub_backend, ego_prompt="x")
    cfg = dataclasses.replace(cfg, scope_prompt="", **opening_perception())
    assert cfg.noumeno_result is not None and cfg.intent_result is not None
    assert cfg.scope_prompt == ""


# ── the pipeline seam ─────────────────────────────────────────────────────────────────

def _pipeline(embedder, calls):
    return Pipeline(embedder=embedder,
                    noumeno=FakeNoumeno(calls=calls), ner=FakeNER(calls=calls),
                    id_stage=FakeID(route="SUPEREGO", calls=calls),
                    ego=FakeEgo(), superego=FakeSuperego())


async def test_a_precomputed_perception_replaces_the_two_stages(
        stub_embedder, stub_backend, dispatcher):
    """Not "the stages run over values we chose" — they do not run. Asserted on the stages'
    own call log, because a result that merely gets overwritten afterwards looks identical
    from the outside and still costs the model and the embedder calls."""
    import dataclasses

    calls: list = []
    pipe = _pipeline(stub_embedder, calls)
    cfg = dataclasses.replace(
        TurnConfig(gen_backend=stub_backend, ego_backend=stub_backend, ego_prompt="x"),
        **opening_perception())
    ctx = await pipe.run_turn(PipelineContext(user_input="[OPENING]", force_language="pt-BR"),
                              cfg, dispatcher=dispatcher)
    assert "noumeno" not in calls and "ner" not in calls
    assert "id" in calls                                  # the ID still routes the turn
    assert ctx.noumeno.metrics.model == OPENING_MODEL
    assert ctx.intent.metrics.model == OPENING_MODEL
    assert ctx.intent.pii_risk == "NONE" and ctx.intent.goal is None


async def test_without_it_both_stages_run_exactly_as_before(
        stub_embedder, stub_backend, dispatcher):
    """The control. Unset → byte-identical to the pipeline that had no such field."""
    calls: list = []
    pipe = _pipeline(stub_embedder, calls)
    cfg = TurnConfig(gen_backend=stub_backend, ego_backend=stub_backend, ego_prompt="x")
    ctx = await pipe.run_turn(PipelineContext(user_input="oi"), cfg, dispatcher=dispatcher)
    assert calls[:3] == ["noumeno", "ner", "id"]
    assert ctx.noumeno.metrics.model == "fake"


async def test_either_half_can_be_supplied_alone(stub_embedder, stub_backend, dispatcher):
    """Two independent fields, not one switch: a caller with a real utterance but a
    precomputed NER (or the reverse) gets exactly the stage it did not supply."""
    calls: list = []
    pipe = _pipeline(stub_embedder, calls)
    cfg = TurnConfig(gen_backend=stub_backend, ego_backend=stub_backend, ego_prompt="x",
                     intent_result=opening_intent())
    await pipe.run_turn(PipelineContext(user_input="oi"), cfg, dispatcher=dispatcher)
    assert "noumeno" in calls and "ner" not in calls
