"""The orchestrator's stamp FILLS a stage's metrics; it never ERASES what the stage wrote.

``_stamp`` marks every executed stage with its place in the turn. It wrote three fields, all
three unconditionally, two of them from keyword DEFAULTS::

    metrics.seq = seq
    metrics.attempt = attempt          # default 0
    metrics.prompt_sha = prompt        # default ""

NOUMENO and NER author their own templates, so they stamp their own digest on their metrics
(``cogno_anima.prompts.prompt_digest`` over the template — "the layer that AUTHORS a text owns
its identity"). The pipeline then called ``_stamp`` for them with no ``prompt``, because the
host has no slot for a text it does not author — and the default ``""`` overwrote the digest.
Measured downstream: ``stages[noumeno|ner].prompt_sha`` empty in 141 of 141 traces over three
days. Without it no prompt A/B and no reversal can be read off a trace for those two stages.

The fix is a rule about the CLASS, not about one field: the stamp writes a field only when
the orchestrator HAS a value for it — a keyword left at its default erases nothing. This file is the red twin (the REAL anima stages through
the REAL pipeline — a fake stage has no digest to lose), then every field ``_stamp`` writes,
one by one, and the call sites enumerated from the source so a tenth cannot be added unseen.
"""

from __future__ import annotations

import ast
import inspect
import json

import cogno_anima.metakeys as mk
import pytest
from cogno_anima.prompts import prompt_digest
from cogno_anima.stages.ego import EgoStage
from cogno_anima.stages.ner import IntentAnalyzer
from cogno_anima.stages.noumeno import Noumeno
from cogno_anima.types import PipelineContext, StageMetrics

from cogno_soma import Pipeline, TurnConfig, pipeline as pipeline_module
from tests.conftest import (FakeEgo, FakeID, FakeNER, FakeNoumeno, FakeSuperego,
                            RecordingDispatcher, StubBackend, StubEmbedder)

_NOUMENO_REPLY = json.dumps({
    "rewritten": "How much is the monthly plan?", "context_turn": "", "confidence": 0.9,
    "changed": True, "preserved_terms": [], "rewrite_warnings": [],
})
_NER_REPLY = json.dumps({
    "intent_class": "INFORMATION_REQUEST", "sentiment": "NEUTRAL", "confidence": 0.9,
    "temporal_class": "TIMELESS", "triad_signal": "EGO",
    "entities": {"people": [], "objects": ["plan"], "concepts": []}, "location": None,
    "mandatory_tags": [], "aristotelian": {}, "goal": "know the plan price", "causal_chain": [],
    "parole": "COLOQUIAL", "negation": [], "constraints": [], "domains": [],
    "modality": "CERTAIN", "speech_act": "INTERROGATIVE", "is_composite": False,
    "is_sequential": False, "verbs": [], "context_dependent": False, "pii": [],
    "pii_risk": "NONE",
})


class _Scripted:
    """One fixed reply per call — the transport of a REAL stage, with no network."""

    def __init__(self, reply: str, model: str = "scripted") -> None:
        self.reply, self.model, self.calls = reply, model, 0

    async def generate(self, system: str, prompt: str):
        self.calls += 1
        return self.reply, 40, 12


def _real_perception():
    embedder = StubEmbedder()
    return Noumeno(embedder=embedder, slangs={}), IntentAnalyzer(), embedder


async def _turn(*, noumeno, ner, embedder, ego=None, superego=None, route="SUPEREGO",
                metadata=None, **cfg_kw):
    pipe = Pipeline(embedder=embedder, noumeno=noumeno, ner=ner,
                    id_stage=FakeID(route=route), ego=ego or FakeEgo(),
                    superego=superego or FakeSuperego())
    cfg = TurnConfig(gen_backend=StubBackend(), ego_backend=StubBackend(), ego_prompt="x",
                     noumeno_backend=_Scripted(_NOUMENO_REPLY), ner_backend=_Scripted(_NER_REPLY),
                     voice_prompt="voice", **cfg_kw)
    ctx = PipelineContext(user_input="quanto custa o plano mensal?", force_language="pt")
    ctx.metadata.update(metadata or {})
    return await pipe.run_turn(ctx, cfg, dispatcher=RecordingDispatcher())


# ── RED TWIN — the digest the stage wrote survives the stamp ───────────────────────────────

async def test_red_twin_the_real_noumeno_and_ner_keep_the_digest_of_their_own_template():
    """One turn through the real pipeline with the real perception stages. Their
    ``prompt_sha`` after the turn is the digest of the TEMPLATE each one used — non-empty, and
    equal to what ``prompt_digest`` gives over that template. On the tree before the fix both
    came back ``""``."""
    noumeno, ner, embedder = _real_perception()
    ctx = await _turn(noumeno=noumeno, ner=ner, embedder=embedder)

    assert ctx.noumeno is not None and ctx.intent is not None, "the premise: both stages ran"
    expected_noumeno = prompt_digest(noumeno._system, noumeno._user_tpl)
    expected_ner = prompt_digest(ner._system, ner._user_tpl)
    assert expected_noumeno and expected_ner and expected_noumeno != expected_ner, (
        "the premise: each stage has a template of its own, and a digest of it")
    assert ctx.noumeno.metrics.prompt_sha == expected_noumeno
    assert ctx.intent.metrics.prompt_sha == expected_ner
    # …and both WERE stamped: the digest survived a stamp that ran, not a stamp that skipped
    assert ctx.noumeno.metrics.seq == 1 and ctx.intent.metrics.seq == 2


async def test_the_digest_is_of_the_TEMPLATE_and_not_of_what_was_rendered():
    """Two different contacts, two different sentences: the same template, the same label. A
    digest of the RENDERED prompt would differ per turn — one row per conversation in a
    content-addressed store, and the contact's words behind it."""
    async def run(text: str) -> "tuple[str, str]":
        noumeno, ner, embedder = _real_perception()
        pipe = Pipeline(embedder=embedder, noumeno=noumeno, ner=ner,
                        id_stage=FakeID(route="SUPEREGO"), ego=FakeEgo(),
                        superego=FakeSuperego())
        cfg = TurnConfig(gen_backend=StubBackend(), ego_backend=StubBackend(), ego_prompt="x",
                         noumeno_backend=_Scripted(_NOUMENO_REPLY),
                         ner_backend=_Scripted(_NER_REPLY), voice_prompt="voice")
        ctx = await pipe.run_turn(PipelineContext(user_input=text, force_language="pt"), cfg,
                                  dispatcher=RecordingDispatcher())
        return ctx.noumeno.metrics.prompt_sha, ctx.intent.metrics.prompt_sha

    first = await run("quanto custa o plano mensal?")
    second = await run("preciso remarcar a aula de quinta para sexta às dez")
    assert first == second and all(first)


# ── the controls: what already arrived, still arrives; what has none, has none ─────────────

async def test_control_the_host_labelled_slots_are_stamped_as_before():
    """Scope, EGO, judge and voice run text the HOST authors, so their label comes from the
    host's map — and it did reach them before the fix. Unchanged."""
    noumeno, ner, embedder = _real_perception()
    ctx = await _turn(
        noumeno=noumeno, ner=ner, embedder=embedder, route="EGO",
        superego=FakeSuperego(approve_after=2), max_corrections=3,
        scope_prompt="only plans", limits_prompt="limits",
        metadata={mk.PROMPT_SHAS: {"scope": "5c0pe1", "ego": "9f3c1a", "judge": "77e0f1",
                                   "voice": "b18d04"}})
    by_stage: "dict[str, list[StageMetrics]]" = {}
    for m in ctx.stage_metrics:
        by_stage.setdefault(m.stage, []).append(m)
    assert [m.prompt_sha for m in by_stage["superego_scope"]] == ["5c0pe1"]
    assert [m.prompt_sha for m in by_stage["ego"]] == ["9f3c1a", "9f3c1a"]
    assert [m.prompt_sha for m in by_stage["superego_judge"]] == ["77e0f1", "77e0f1"]
    assert [m.prompt_sha for m in by_stage["superego_voice"]] == ["b18d04"]


async def test_control_a_stage_with_no_template_stays_unlabelled_and_a_stage_that_did_not_run_is_absent():
    """The ID calls no model and the doubles author nothing: ``""`` — "nothing a deployment
    set", the type's own unlabelled value — and never a digest invented for them. A stage that
    did not run has no metrics at all: absent, not empty."""
    ctx = await _turn(noumeno=FakeNoumeno(), ner=FakeNER(), embedder=StubEmbedder())
    assert ctx.noumeno.metrics.prompt_sha == "" and ctx.noumeno.metrics.seq == 1
    assert ctx.intent.metrics.prompt_sha == ""
    assert ctx.id_result.metrics.prompt_sha == "" and ctx.id_result.metrics.seq == 3
    assert ctx.ego_result is None, "the route was SUPEREGO: the executor did not run"
    assert "ego" not in {m.stage for m in ctx.stage_metrics}


async def test_control_a_SEATED_perception_runs_no_template_and_keeps_an_empty_sha():
    """The turn the AGENT opens has no utterance to perceive: the host seats a precomputed
    NOUMENO and NER result and no model is asked. No template ran, so there is no digest — the
    row is stamped (it has its place in the turn) and its sha stays ``""``. The fix must not
    invent one for it."""
    from cogno_soma import opening_perception

    noumeno_backend, ner_backend = _Scripted(_NOUMENO_REPLY), _Scripted(_NER_REPLY)
    noumeno, ner, embedder = _real_perception()
    pipe = Pipeline(embedder=embedder, noumeno=noumeno, ner=ner,
                    id_stage=FakeID(route="SUPEREGO"), ego=FakeEgo(), superego=FakeSuperego())
    cfg = TurnConfig(gen_backend=StubBackend(), ego_backend=StubBackend(), ego_prompt="x",
                     noumeno_backend=noumeno_backend, ner_backend=ner_backend,
                     voice_prompt="voice", **opening_perception())
    ctx = await pipe.run_turn(PipelineContext(user_input="", force_language="pt"), cfg,
                              dispatcher=RecordingDispatcher())
    assert (noumeno_backend.calls, ner_backend.calls) == (0, 0), "the premise: nothing was asked"
    assert ctx.noumeno.metrics.prompt_sha == "" and ctx.noumeno.metrics.seq == 1
    assert ctx.intent.metrics.prompt_sha == "" and ctx.intent.metrics.seq == 2


# ── the class, field by field ──────────────────────────────────────────────────────────────

def _stamped() -> "tuple[PipelineContext, StageMetrics]":
    """A metrics row a stage ALREADY filled in — the thing a stamp must not undo."""
    ctx = PipelineContext(user_input="x")
    row = StageMetrics(stage="s", elapsed_ms=1.0, tokens_in=1, tokens_out=1, model="m",
                       attempt=2, prompt_sha="own-digest")
    return ctx, row


def test_the_fields_the_stamp_writes_are_exactly_these_three():
    """Enumerated from the SOURCE: every ``metrics.<field> = …`` in ``_stamp``. A fourth
    written tomorrow fails here, and has to come with its own proof below."""
    tree = ast.parse(inspect.getsource(pipeline_module._stamp))
    written = {t.attr for node in ast.walk(tree) if isinstance(node, ast.Assign)
               for t in node.targets
               if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
               and t.value.id == "metrics"}
    assert written == {"seq", "attempt", "prompt_sha"}


def test_field_seq_is_always_written_and_only_the_orchestrator_has_it():
    """``seq`` is the one field the stamp OWNS: only the orchestrator knows the call order, so
    there is nothing of a stage's to erase — and a stage that pre-filled it would be wrong."""
    ctx, row = _stamped()
    row.seq = 99
    pipeline_module._stamp(ctx, row)
    assert row.seq == 1
    pipeline_module._stamp(ctx, row)
    assert row.seq == 2


def test_field_prompt_sha_is_not_erased_by_an_absent_or_empty_label():
    """The measured defect, at the function: no label, ``None`` and ``""`` all leave the
    stage's own digest where it was."""
    for kwargs in ({}, {"prompt": None}, {"prompt": ""}):
        ctx, row = _stamped()
        pipeline_module._stamp(ctx, row, **kwargs)
        assert row.prompt_sha == "own-digest", kwargs


def test_field_prompt_sha_takes_the_hosts_label_when_there_is_one():
    """The control of the rule above: a label the host supplied is written — the fix did not
    turn the stamp into a no-op for this field."""
    ctx, row = _stamped()
    row.prompt_sha = ""
    pipeline_module._stamp(ctx, row, prompt="9f3c1a")
    assert row.prompt_sha == "9f3c1a"


def test_field_attempt_is_not_erased_by_an_absent_or_zero_attempt():
    """The SAME shape as the defect, one field over: ``attempt`` was written from a default of
    ``0``. No call site reached it with a stage that had set its own (the EGO, the only stage
    that does, is always stamped with the loop's count) — so nothing was being lost. The rule
    is closed anyway: a call that has no attempt to give leaves the stage's own."""
    for kwargs in ({}, {"attempt": None}, {"attempt": 0}):
        ctx, row = _stamped()
        pipeline_module._stamp(ctx, row, **kwargs)
        assert row.attempt == 2, kwargs


def test_field_attempt_takes_the_loops_count_when_there_is_one():
    ctx, row = _stamped()
    row.attempt = 0
    pipeline_module._stamp(ctx, row, attempt=3)
    assert row.attempt == 3


def test_a_stamp_never_raises_and_a_missing_row_still_advances_the_order():
    ctx = PipelineContext(user_input="x")
    pipeline_module._stamp(ctx, None)
    pipeline_module._stamp(ctx, object())          # not a metrics row at all
    row = StageMetrics(stage="s", elapsed_ms=0.0, tokens_in=0, tokens_out=0, model="m")
    pipeline_module._stamp(ctx, row)
    assert row.seq == 3


# ── `attempt` through the real executor: is it ever SWAPPED? ───────────────────────────────

class _NoTools(RecordingDispatcher):
    pass


async def _ego_turn(seeded_attempt=None, *, approve_after=1, budget=3):
    ego = EgoStage()
    pipe = Pipeline(embedder=StubEmbedder(), noumeno=FakeNoumeno(), ner=FakeNER(),
                    id_stage=FakeID(route="EGO"), ego=ego,
                    superego=FakeSuperego(approve_after=approve_after))
    cfg = TurnConfig(gen_backend=StubBackend(), ego_backend=_Scripted("The plan costs 10."),
                     ego_prompt="x", limits_prompt="limits", voice_prompt="voice",
                     max_corrections=budget)
    ctx = PipelineContext(user_input="how much is the plan?")
    if seeded_attempt is not None:
        ctx.metadata[mk.EGO_CORRECTION] = {"reason": "seeded by the host", "attempt": seeded_attempt}
    ctx = await pipe.run_turn(ctx, cfg, dispatcher=_NoTools())
    return ctx, sorted((m for m in ctx.stage_metrics if m.stage == "ego"), key=lambda m: m.seq)


async def test_attempt_of_the_real_executor_agrees_with_the_loop_on_every_pass():
    """The real EGO stamps its OWN attempt (from ``mk.EGO_CORRECTION``), and the loop stamps
    its count over it. Inside the loop the two are the same number by construction — the loop
    writes the metakey the executor reads — so the overwrite replaces a value with itself."""
    ctx, egos = await _ego_turn(approve_after=3)
    assert [m.attempt for m in egos] == [1, 2, 3]
    assert ctx.ego_result.attempt == ctx.ego_result.metrics.attempt == 3


@pytest.mark.parametrize("seeded", [1, 2, 5])
async def test_attempt_when_a_host_seeds_the_correction_the_loops_count_is_what_is_stamped(seeded):
    """The ONE path where the two can differ: a host that seeds ``mk.EGO_CORRECTION`` before
    the turn (a repair pass). The executor then reads the host's number and the loop still
    counts from 1. The stamp writes the LOOP's count — the orchestrator is the layer that
    sequences, and its count is the axis the judge ledger and every other row of the turn are
    on. Declared and pinned, not accidental: with a seed of 1, which is what a host sends
    today, the two agree; with any other seed the row says 1 and ``EgoResult.attempt`` keeps
    the host's number."""
    ctx, egos = await _ego_turn(seeded_attempt=seeded)
    assert [m.attempt for m in egos] == [1]
    assert ctx.ego_result.attempt == seeded
    assert ctx.metadata[mk.JUDGE_ATTEMPTS][0]["attempt"] == 1


# ── the call sites, enumerated ─────────────────────────────────────────────────────────────

def test_every_stamp_call_site_is_one_this_file_accounts_for():
    """Read off the source: which calls pass a ``prompt`` label and which pass an ``attempt``.
    A new call site changes one of these sets and has to say which kind of stage it stamps —
    one that authors its own text (no label: its digest must survive) or one the host labels."""
    tree = ast.parse(inspect.getsource(pipeline_module))
    sites: "list[tuple[bool, bool]]" = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_stamp"):
            names = {kw.arg for kw in node.keywords}
            sites.append(("prompt" in names, "attempt" in names))
    assert len(sites) == 9
    # (labelled by the host?, on the loop's attempt axis?)
    assert sorted(sites) == sorted([
        (False, False),      # noumeno — authors its own template
        (False, False),      # ner — authors its own template
        (False, False),      # id — no model, no text
        (False, False),      # scope selector — a fixed prompt of its own, unlabelled
        (True, False),       # scope guard — the host's slot
        (True, False),       # voice — the host's slot
        (True, True),        # ego — the host's slot, per attempt
        (True, True),        # judge, fast tier — the host's slot, per attempt
        (True, True),        # judge, strong tier
    ])
