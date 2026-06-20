"""Minimal, **offline** host wiring for cogno-soma.

Runs with no network: it injects tiny fake stages (the same trick the unit tests
use) so you can see the orchestration — routing, the correction loop, hooks,
StopPipeline, and a multi-turn session — without an LLM. A real host injects the
cogno-anima stages over a backend instead (see docs/HOST_INTEGRATION.md).

    python examples/host_min.py
"""

import asyncio

from cogno_anima.types import (
    EgoResult,
    IdResult,
    IntentResult,
    NoumenoResult,
    PipelineContext,
    StageMetrics,
    SuperegoResult,
    ToolResult,
)

from cogno_soma import Hooks, Pipeline, SessionRunner, StopPipeline, TurnConfig


def _m(stage):
    # non-zero so the per-turn token total below is meaningful in the demo
    return StageMetrics(stage=stage, elapsed_ms=0.0, tokens_in=12, tokens_out=8, model="fake")


# ── fake stages (a real host injects the cogno-anima stages instead) ───────
class FakeNoumeno:
    async def process(self, ctx, backend):
        ctx.noumeno = NoumenoResult(
            original=ctx.user_input, rewritten=ctx.user_input, context_turn="",
            language="en", drift_score=0.0, drift_tag="PASS_THROUGH", changed=False,
            confidence=0.9, change_subject=False, subject_similarity=1.0,
            context_used=False, preserved_terms=[], rewrite_warnings=[], metrics=_m("noumeno"))
        return ctx


class FakeNER:
    async def process(self, ctx, backend):
        ctx.intent = IntentResult(
            intent_class="INFORMATION_REQUEST", sentiment="NEUTRAL", confidence=0.9,
            temporal_class="PRESENT", triad_signal="BALANCED", goal="check balance",
            domains=["finance"], metrics=_m("ner"))
        return ctx


class FakeID:
    async def process(self, ctx, embedder):
        ctx.metadata.setdefault("id_state", {})["turn"] = ctx.metadata.get("turn_number")
        ctx.id_result = IdResult(triad_route="EGO", goal_status="ONGOING", metrics=_m("id"))
        return ctx


class FakeEgo:
    async def process(self, ctx, backend, dispatcher, *, system_prompt):
        await dispatcher.execute("get_balance", {})
        ctx.ego_result = EgoResult(metrics=_m("ego"))
        return ctx


class FakeSuperego:
    async def check_input_scope(self, ctx, backend, *, scope_prompt):
        from cogno_anima.types import ScopeCheckResult
        return ScopeCheckResult(blocked=False, refusal_message="", metrics=_m("superego_scope"))

    async def evaluate(self, ctx, backend, *, limits_prompt):
        return SuperegoResult(response="", approved=True, metrics=_m("superego_judge"))

    async def voice(self, ctx, backend, *, voice_prompt):
        return SuperegoResult(response="Your balance is 1234.56 BRL.",
                              approved=True, metrics=_m("superego_voice"))

    def _blocked_response(self, ctx):
        return SuperegoResult(response="(blocked)", blocked=True, metrics=_m("superego_voice"))


class Dispatcher:
    def __init__(self):
        self.executed = []

    def tools_schema(self):
        return []

    async def execute(self, name, arguments):
        self.executed.append(name)
        return ToolResult(output="1234.56", ok=True)


async def main():
    pipe = Pipeline(
        embedder=None,  # the fake ID does not use it
        noumeno=FakeNoumeno(), ner=FakeNER(), id_stage=FakeID(),
        ego=FakeEgo(), superego=FakeSuperego(),
    )

    # hooks: inject "memory" after NER, and an audit bookend after the turn.
    def inject_memory(ctx):
        ctx.metadata["ego_context"] = "[MEMORIES] currency=BRL"

    def audit(ctx):
        print(f"  [audit] route={ctx.id_result.triad_route} stop={ctx.stop_reason}")

    cfg = TurnConfig(
        gen_backend=None, ego_backend=None,
        ego_prompt="You are the executor.", voice_prompt="Be concise.",
        hooks=Hooks(after_ner=inject_memory, after_turn=audit),
    )

    print("── single turn ──")
    ctx = PipelineContext(user_input="What's my balance?")
    ctx = await pipe.run_turn(ctx, cfg, dispatcher=Dispatcher())
    print("  reply:", ctx.superego_result.response)
    print(f"  tokens: total={ctx.total_tokens} llm={ctx.total_llm_tokens}  (feed your meter)")

    print("── StopPipeline (a safety hook halts the turn) ──")
    def crisis(ctx):
        raise StopPipeline(reason="human_handoff", response="Please call 988.", blocked=True)
    stop_cfg = TurnConfig(gen_backend=None, ego_backend=None, ego_prompt="x",
                          hooks=Hooks(after_id=crisis))
    ctx = await pipe.run_turn(PipelineContext(user_input="..."), stop_cfg, dispatcher=Dispatcher())
    print("  reply:", ctx.superego_result.response, "| stop_reason:", ctx.stop_reason)

    print("── multi-turn session ──")
    sess = SessionRunner(pipe, cfg, dispatcher_factory=Dispatcher, persona_id="FINANCE")
    await sess.run("What's my balance?")
    await sess.run("And last month?")
    print("  turns:", sess.turn_number, "| state keys:", sorted(sess.state["carry"]))


if __name__ == "__main__":
    asyncio.run(main())
