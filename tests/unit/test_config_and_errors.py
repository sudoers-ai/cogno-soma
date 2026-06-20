"""Small surface tests: TurnConfig defaults, StopPipeline attrs, exports."""

import cogno_soma
from cogno_soma import Hooks, SomaError, StopPipeline, TurnConfig


def test_turn_config_defaults(stub_backend):
    cfg = TurnConfig(gen_backend=stub_backend, ego_backend=stub_backend, ego_prompt="x")
    assert cfg.scope_prompt == ""
    assert cfg.limits_prompt == ""
    assert cfg.voice_prompt == ""
    assert cfg.voice_backend is None
    assert cfg.max_corrections == 2
    assert cfg.hooks is None


def test_voice_backend_defaults_to_ego(stub_backend):
    # The Pipeline resolves voice_backend or ego_backend; here just assert the field.
    cfg = TurnConfig(gen_backend=stub_backend, ego_backend=stub_backend, ego_prompt="x")
    assert (cfg.voice_backend or cfg.ego_backend) is stub_backend


def test_stop_pipeline_is_soma_error():
    err = StopPipeline(reason="pii_blocked", response="no", blocked=True)
    assert isinstance(err, SomaError)
    assert err.reason == "pii_blocked"
    assert err.response == "no"
    assert err.blocked is True


def test_stop_pipeline_defaults():
    err = StopPipeline()
    assert err.reason == "completed"
    assert err.response is None
    assert err.blocked is False


def test_hooks_default_all_none():
    h = Hooks()
    assert h.before_turn is None and h.after_turn is None and h.on_commit is None


def test_public_exports():
    for name in ("Pipeline", "SessionRunner", "TurnConfig", "Hooks", "HookFn",
                 "StopPipeline", "SomaError"):
        assert hasattr(cogno_soma, name)
    assert isinstance(cogno_soma.__version__, str)
