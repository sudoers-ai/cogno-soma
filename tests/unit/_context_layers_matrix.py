"""The configurations `SessionRunner.run` composes a turn's context in — shared by the inertia
proof and the layer-record tests, so the contexts whose BYTES are pinned are the contexts whose
LAYERS are counted.

Thirty-two cells: a verbatim transcript or none × an earlier-session summary × memories × graph
facts × the context whole or handed over by provenance (`split_context`). Invented text only.
"""

from __future__ import annotations

import hashlib
from itertools import product

from cogno_anima import metakeys as mk

from cogno_soma import Pipeline, SessionRunner, TurnConfig
from tests.conftest import (FakeEgo, FakeID, FakeNER, FakeNoumeno, FakeSuperego,
                            RecordingDispatcher, StubBackend, StubEmbedder)

NOW = 1_000_000.0
TRANSCRIPT = [["which day is free?", "Thursday at ten is free.", NOW - 60]]
SUMMARY = "Last session the contact asked about the autumn timetable."
MEMORIES = ["Prefers morning slots.", "Asked for a receipt in August."]
GRAPH = "# About this contact\n- works at the Example Bakery"
DATA_KEY = "ego_context_untrusted"


def cells():
    """``(key, transcript, summary, memories, graph, split)`` in a fixed order. Key letters:
    T transcript · E earlier context · M memories · G graph · S split; ``-`` = absent."""
    out = []
    for t, e, m, g, s in product((True, False), repeat=5):
        key = "".join(flag if on else "-" for flag, on in zip("TEMGS", (t, e, m, g, s)))
        out.append((key, t, e, m, g, s))
    return out


async def compose(cell, *, metadata=None) -> dict:
    """Run one turn of ``cell`` and return the metadata the STAGES saw (before any hook)."""
    _key, t, e, m, g, split = cell
    seen: dict = {}
    pipe = Pipeline(embedder=StubEmbedder(), noumeno=FakeNoumeno(rewritten="x"),
                    ner=FakeNER(goal="g"), id_stage=FakeID(route="SUPEREGO"), ego=FakeEgo(),
                    superego=FakeSuperego())
    orig = pipe._id.process

    async def spy(ctx, emb):
        seen.update(ctx.metadata)
        return await orig(ctx, emb)

    pipe._id.process = spy  # type: ignore[method-assign]
    cfg = TurnConfig(gen_backend=StubBackend(), ego_backend=StubBackend(), ego_prompt="x")
    sess = SessionRunner(pipe, cfg, dispatcher=RecordingDispatcher(),
                         state={"transcript": TRANSCRIPT} if t else None)
    kwargs: dict = {"split_context": True} if split else {}
    await sess.run("and the price?", now=NOW, prior_summary=SUMMARY if e else None,
                   memories=MEMORIES if m else None, graph_context=GRAPH if g else None,
                   metadata=metadata, **kwargs)
    return seen


def digest(seen: dict) -> str:
    """One digest of BOTH carriers, with the absence of the second one kept apart from ``""``."""
    notes = seen.get(mk.EGO_CONTEXT)
    data = seen.get(DATA_KEY, "<<absent>>")
    return hashlib.sha256(f"{notes!r}\x00{data!r}".encode("utf-8")).hexdigest()[:16]
