"""The stage contract the Pipeline calls is STRUCTURAL — and both of its ends satisfy it.

``Pipeline(noumeno=…, ner=…, id_stage=…, ego=…, superego=…)`` is typed with
``cogno_anima.BaseStage`` (NOUMENO, NER — reused, the anima owns it) and the three protocols of
``cogno_soma.stages`` (ID, EGO, SUPEREGO). Until they existed the five parameters named the
CONCRETE anima classes, so the documented promise — swap any stage for anything that matches the
stage signature — held at runtime and was refused by the type checker at every host construction
site, one ``arg-type`` error per stage.

The SIGNATURE check belongs to mypy: the Pipeline annotates its own defaults with the protocols,
so a cogno-anima stage that stops matching fails ``mypy cogno_soma``. What a runtime test can hold
is the rest:

* the parameters stay structural — reverting one to its concrete class keeps THIS repo green and
  silently turns every host double back into a type error, which is the state this ends;
* the anima defaults and this repo's own doubles satisfy the contract;
* the probe can REFUSE — a double missing a member the Pipeline calls is not a stage. Without
  that half, an ``isinstance`` that answers True for everything would pass the other two.
"""

from __future__ import annotations

import typing
from typing import Optional

import pytest

from cogno_anima import BaseStage
from cogno_soma import EgoStageProtocol, IDStageProtocol, Pipeline, SuperegoStageProtocol

from tests.conftest import FakeEgo, FakeID, FakeNER, FakeNoumeno, FakeSuperego, StubEmbedder

# (Pipeline parameter, the attribute it lands on, the protocol, this repo's double)
_CONTRACT = [
    ("noumeno", "_noumeno", BaseStage, FakeNoumeno),
    ("ner", "_ner", BaseStage, FakeNER),
    ("id_stage", "_id", IDStageProtocol, FakeID),
    ("ego", "_ego", EgoStageProtocol, FakeEgo),
    ("superego", "_superego", SuperegoStageProtocol, FakeSuperego),
]


@pytest.mark.parametrize("param,_attr,proto,_fake", _CONTRACT, ids=[c[0] for c in _CONTRACT])
def test_the_stage_parameters_are_structural(param, _attr, proto, _fake):
    hints = typing.get_type_hints(Pipeline.__init__)
    assert hints[param] == Optional[proto]


@pytest.mark.parametrize("_param,attr,proto,_fake", _CONTRACT, ids=[c[0] for c in _CONTRACT])
def test_the_anima_defaults_satisfy_the_contract(_param, attr, proto, _fake):
    pipe = Pipeline(embedder=StubEmbedder())
    assert isinstance(getattr(pipe, attr), proto)


@pytest.mark.parametrize("_param,_attr,proto,fake", _CONTRACT, ids=[c[0] for c in _CONTRACT])
def test_this_repos_doubles_satisfy_the_contract(_param, _attr, proto, fake):
    assert isinstance(fake(), proto)


def test_the_probe_refuses_a_superego_without_the_pii_critical_reply():
    """``_blocked_response`` is private-named in the anima and CALLED by the Pipeline. A protocol
    that left it out would let this double type-check and then fail on the one path a
    PII-CRITICAL turn takes."""

    class _NoBlockedResponse:
        name = "superego"

        async def check_input_scope(self, ctx, backend, *, scope_prompt):
            raise NotImplementedError

        async def evaluate(self, ctx, backend, *, limits_prompt):
            raise NotImplementedError

        async def voice(self, ctx, backend, *, voice_prompt):
            raise NotImplementedError

    assert not isinstance(_NoBlockedResponse(), SuperegoStageProtocol)
    # The control: the same double WITH the member is a SUPEREGO — so the refusal above is about
    # that member and not about something else the class lacks.
    setattr(_NoBlockedResponse, "_blocked_response", FakeSuperego._blocked_response)
    assert isinstance(_NoBlockedResponse(), SuperegoStageProtocol)


@pytest.mark.parametrize("proto,fake", [(IDStageProtocol, FakeID), (EgoStageProtocol, FakeEgo)],
                         ids=["id_stage", "ego"])
def test_the_probe_refuses_a_stage_without_a_name(proto, fake):
    """``name`` is in every protocol because ``BaseStage`` carries it (every anima stage has one)."""
    nameless = type("Nameless", (), {"process": fake.process})
    assert not isinstance(nameless(), proto)
    assert isinstance(type("Named", (nameless,), {"name": "x"})(), proto)
