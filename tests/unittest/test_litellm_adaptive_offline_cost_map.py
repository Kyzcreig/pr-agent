"""Adaptive thinking must survive a stale litellm model-cost map.

litellm fetches its cost map from GitHub at import and falls back to the copy bundled with the
wheel when the fetch fails. A bundled map older than the model has no supports_reasoning entry,
and litellm then rejects thinking={"type": "adaptive"} with UnsupportedParamsError.
"""
from types import SimpleNamespace

import litellm
import pytest

import pr_agent.algo.ai_handlers.litellm_ai_handler as litellm_handler
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler

# Not in any cost map: stands in for a model newer than the bundled map.
UNKNOWN_ADAPTIVE = "anthropic/claude-sonnet-5-99"


@pytest.fixture
def stale_map(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", {k: dict(v) for k, v in litellm.model_cost.items()})
    litellm.utils._invalidate_model_cost_lowercase_map()
    yield
    litellm.utils._invalidate_model_cost_lowercase_map()


def settings():
    config = SimpleNamespace(reasoning_effort="high", get=lambda key, default=None: default)
    return SimpleNamespace(config=config, get=lambda key, default=None: default)


def optional_params(model, kwargs):
    return litellm.utils.get_optional_params(
        model=model.split("/", 1)[1], custom_llm_provider="anthropic", thinking=kwargs["thinking"]
    )


def test_unknown_adaptive_model_is_rejected_without_registration(stale_map):
    assert not litellm.supports_reasoning(model=UNKNOWN_ADAPTIVE)
    with pytest.raises(litellm.UnsupportedParamsError):
        optional_params(UNKNOWN_ADAPTIVE, {"thinking": {"type": "adaptive"}})


def test_adaptive_config_registers_reasoning_for_unknown_model(stale_map, monkeypatch):
    monkeypatch.setattr(litellm_handler, "get_settings", settings)
    kwargs = LiteLLMAIHandler._configure_claude_adaptive_thinking(
        LiteLLMAIHandler.__new__(LiteLLMAIHandler), UNKNOWN_ADAPTIVE, {}
    )
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert litellm.supports_reasoning(model=UNKNOWN_ADAPTIVE)
    assert optional_params(UNKNOWN_ADAPTIVE, kwargs)["thinking"] == {"type": "adaptive"}


def test_non_anthropic_route_is_not_registered(stale_map):
    LiteLLMAIHandler._ensure_litellm_reasoning_support("openai/claude-sonnet-5-99")
    assert "openai/claude-sonnet-5-99" not in litellm.model_cost


def test_known_model_entry_is_left_untouched(stale_map):
    known = next(k for k, v in litellm.model_cost.items()
                 if v.get("litellm_provider") == "anthropic" and v.get("supports_reasoning"))
    before = dict(litellm.model_cost[known])
    LiteLLMAIHandler._ensure_litellm_reasoning_support(f"anthropic/{known}")
    assert litellm.model_cost[known] == before
    assert f"anthropic/{known}" not in litellm.model_cost
