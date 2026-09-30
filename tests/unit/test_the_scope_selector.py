"""The scope guard's "did you mean…?" selector, as the ORCHESTRATOR runs it.

A refusal the selector proves FALSE (``covered``: one of the host's closed options is the very
thing asked) is LET THROUGH — never answered with a question. Any other outcome keeps the refusal
byte for byte; the record left on ``mk.SCOPE_OPTIONS_SELECTION`` is what the host reads to render
its closed question. With no selector backend (or no list) nothing is called and nothing is
written. Invented names throughout.
"""

import json

import cogno_anima.metakeys as mk
from cogno_anima.types import PipelineContext

from cogno_soma import Pipeline, TurnConfig

from tests.conftest import FakeEgo, FakeID, FakeNER, FakeNoumeno, FakeSuperego

OPTIONS = ("Calendário acadêmico", "Oficina de Fotografia", "consult_material")


class Selector:
    model = "stub-selector"

    def __init__(self, reply, *, raises=None):
        self.reply = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        self.raises = raises
        self.calls = 0

    async def generate(self, system, prompt):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.reply, 50, 8


def _pipe(embedder, *, blocked=True, route="SUPEREGO", ego=None):
    return Pipeline(embedder=embedder, noumeno=FakeNoumeno(), ner=FakeNER(),
                    id_stage=FakeID(route=route), ego=ego or FakeEgo(),
                    superego=FakeSuperego(scope_blocked=blocked, refusal="Não posso ajudar."))


def _cfg(backend, **kw):
    base = dict(gen_backend=backend, ego_backend=backend, ego_prompt="exec", scope_prompt="scope")
    base.update(kw)
    return TurnConfig(**base)


def _ctx(text="Quando começa o curso de Fotografia Digital?"):
    return PipelineContext(user_input=text)


# ── no selector → the refusal of today, byte for byte ────────────────────────────────────

async def test_without_a_selector_backend_the_refusal_is_todays(stub_embedder, stub_backend,
                                                                dispatcher):
    ctx = await _pipe(stub_embedder).run_turn(_ctx(), _cfg(stub_backend, scope_options=OPTIONS),
                                              dispatcher=dispatcher)
    assert ctx.stop_reason == "scope_blocked"
    assert ctx.superego_result.response == "Não posso ajudar."
    assert mk.SCOPE_OPTIONS_SELECTION not in ctx.metadata
    assert [m.stage for m in ctx.retry_metrics] == ["superego_scope"]


async def test_without_a_list_the_selector_is_never_called(stub_embedder, stub_backend, dispatcher):
    sel = Selector({"answers": ["Oficina de Fotografia"], "maybe": []})
    ctx = await _pipe(stub_embedder).run_turn(_ctx(), _cfg(stub_backend, scope_selector_backend=sel),
                                              dispatcher=dispatcher)
    assert sel.calls == 0
    assert ctx.stop_reason == "scope_blocked" and ctx.superego_result.response == "Não posso ajudar."
    assert mk.SCOPE_OPTIONS_SELECTION not in ctx.metadata


async def test_an_allowed_turn_never_calls_the_selector(stub_embedder, stub_backend, dispatcher):
    sel = Selector({"answers": [], "maybe": ["Oficina de Fotografia"]})
    ctx = await _pipe(stub_embedder, blocked=False).run_turn(
        _ctx(), _cfg(stub_backend, scope_selector_backend=sel, scope_options=OPTIONS),
        dispatcher=dispatcher)
    assert sel.calls == 0 and ctx.stop_reason == "completed"
    assert mk.SCOPE_OPTIONS_SELECTION not in ctx.metadata


# ── covered → the refusal was FALSE → the turn goes on (1a), no question ───────────────────

async def test_a_covered_pick_lets_the_refused_turn_through(stub_embedder, stub_backend, dispatcher):
    ego = FakeEgo()
    sel = Selector({"answers": ["Calendário acadêmico"], "maybe": ["Oficina de Fotografia"]})
    ctx = await _pipe(stub_embedder, route="EGO", ego=ego).run_turn(
        _ctx("Onde vejo o calendário acadêmico do semestre?"),
        _cfg(stub_backend, scope_selector_backend=sel, scope_options=OPTIONS),
        dispatcher=dispatcher)
    assert sel.calls == 1
    assert ctx.stop_reason != "scope_blocked"
    assert ego.invocations == 1                       # the executor ran
    assert ctx.superego_result.response == "final reply"
    assert ctx.superego_result.blocked is False
    rec = ctx.metadata[mk.SCOPE_OPTIONS_SELECTION]
    assert rec["outcome"] == "covered" and rec["suggested"] == []   # no question rides a pass


# ── suggested / none / error → blocked, the refusal text untouched ───────────────────────

async def test_a_suggested_pick_keeps_the_refusal_and_leaves_the_record(stub_embedder, stub_backend,
                                                                        dispatcher):
    sel = Selector({"answers": [], "maybe": ["Oficina de Fotografia", "Inventada"],
                    "asked": "curso de Fotografia Digital"})
    ctx = await _pipe(stub_embedder).run_turn(
        _ctx(), _cfg(stub_backend, scope_selector_backend=sel, scope_options=OPTIONS),
        dispatcher=dispatcher)
    assert ctx.stop_reason == "scope_blocked"
    assert ctx.superego_result.response == "Não posso ajudar."   # the host renders the question
    rec = ctx.metadata[mk.SCOPE_OPTIONS_SELECTION]
    assert rec["outcome"] == "suggested"
    assert rec["suggested"] == ["Oficina de Fotografia"] and rec["discarded"] == 1


async def test_the_selector_gets_its_own_ledger_line(stub_embedder, stub_backend, dispatcher):
    sel = Selector({"answers": [], "maybe": []})
    ctx = await _pipe(stub_embedder).run_turn(
        _ctx(), _cfg(stub_backend, scope_selector_backend=sel, scope_options=OPTIONS),
        dispatcher=dispatcher)
    stages = [m.stage for m in ctx.retry_metrics]
    assert stages == ["superego_scope", "superego_select"]
    assert ctx.retry_metrics[1].tokens_in == 50 and ctx.retry_metrics[1].seq > 0


async def test_a_selector_that_raises_is_the_refusal_of_today(stub_embedder, stub_backend,
                                                              dispatcher):
    sel = Selector({}, raises=RuntimeError("provider down"))
    ctx = await _pipe(stub_embedder).run_turn(
        _ctx(), _cfg(stub_backend, scope_selector_backend=sel, scope_options=OPTIONS),
        dispatcher=dispatcher)
    assert ctx.stop_reason == "scope_blocked"
    assert ctx.superego_result.response == "Não posso ajudar."
    assert ctx.metadata[mk.SCOPE_OPTIONS_SELECTION]["outcome"] == "error"


async def test_the_wifi_password_stays_refused(stub_embedder, stub_backend, dispatcher):
    sel = Selector({"answers": [], "maybe": [], "asked": "senha do Wi-Fi"})
    ctx = await _pipe(stub_embedder).run_turn(
        _ctx("Qual é a senha do Wi-Fi da escola?"),
        _cfg(stub_backend, scope_selector_backend=sel, scope_options=OPTIONS),
        dispatcher=dispatcher)
    assert ctx.stop_reason == "scope_blocked"
    assert ctx.superego_result.response == "Não posso ajudar."
    assert ctx.metadata[mk.SCOPE_OPTIONS_SELECTION]["outcome"] == "none"


async def test_an_out_of_list_covered_pick_does_not_lift(stub_embedder, stub_backend, dispatcher):
    # The closed alphabet decides the PASS too: an invented "answer" lifts nothing.
    sel = Selector({"answers": ["Calendário de provas"], "maybe": []})
    ctx = await _pipe(stub_embedder).run_turn(
        _ctx("Quando são as provas?"),
        _cfg(stub_backend, scope_selector_backend=sel, scope_options=OPTIONS),
        dispatcher=dispatcher)
    assert ctx.stop_reason == "scope_blocked"


# ── the record is PER TURN ───────────────────────────────────────────────────────────────

async def test_a_carried_record_is_dropped_on_a_turn_without_a_selection(stub_embedder,
                                                                         stub_backend, dispatcher):
    ctx = _ctx()
    ctx.metadata[mk.SCOPE_OPTIONS_SELECTION] = {"outcome": "suggested",
                                                "suggested": ["Oficina de Fotografia"]}
    ctx = await _pipe(stub_embedder).run_turn(ctx, _cfg(stub_backend), dispatcher=dispatcher)
    assert mk.SCOPE_OPTIONS_SELECTION not in ctx.metadata


async def test_a_covered_pick_without_evidence_does_not_lift(stub_embedder, stub_backend,
                                                            dispatcher):
    """The shape the anima's nightly measured (qwen3 covered the Wi-Fi with a capability): the
    pick shares no term with the question, so the refusal stands and the count says why."""
    sel = Selector({"answers": ["consult_material"], "maybe": []})
    ctx = await _pipe(stub_embedder).run_turn(
        _ctx("Qual é a senha do Wi-Fi da escola?"),
        _cfg(stub_backend, scope_selector_backend=sel, scope_options=OPTIONS),
        dispatcher=dispatcher)
    assert ctx.stop_reason == "scope_blocked"
    assert ctx.superego_result.response == "Não posso ajudar."
    rec = ctx.metadata[mk.SCOPE_OPTIONS_SELECTION]
    assert rec["outcome"] == "none" and rec["covered_unsupported"] == 1
