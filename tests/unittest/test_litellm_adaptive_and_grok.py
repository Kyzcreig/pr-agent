from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import pr_agent.algo.ai_handlers.litellm_ai_handler as litellm_handler
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler


def make_settings(reasoning_effort="medium", config_flags=None):
    """Fake settings: config attributes plus a .get() that serves feature flags."""
    flags = dict(config_flags or {})

    def config_get(key, default=None):
        return flags.get(key, default)

    config = SimpleNamespace(
        reasoning_effort=reasoning_effort,
        ai_timeout=120,
        custom_reasoning_model=False,
        max_model_tokens=32000,
        verbosity_level=0,
        get=config_get,
    )
    return SimpleNamespace(
        config=config,
        litellm=SimpleNamespace(get=lambda key, default=None: default),
        get=lambda key, default=None: default,
    )


def mock_response():
    response = MagicMock()
    response.__getitem__ = lambda self, key: {
        "choices": [{"message": {"content": "test"}, "finish_reason": "stop"}]
    }[key]
    response.dict.return_value = {"choices": [{"message": {"content": "test"}, "finish_reason": "stop"}]}
    return response


async def run_completion(monkeypatch, model, settings):
    monkeypatch.setattr(litellm_handler, "get_settings", lambda: settings)
    with patch(
        "pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock
    ) as completion:
        completion.return_value = mock_response()
        handler = LiteLLMAIHandler()
        await handler.chat_completion(model=model, system="sys", user="usr")
        return completion.call_args[1]


# ---------- Grok reasoning-effort gating ----------

@pytest.mark.asyncio
async def test_grok_without_flag_sends_no_reasoning_effort(monkeypatch):
    """Default behavior must stay byte-identical: no flag => no reasoning_effort for grok."""
    kwargs = await run_completion(monkeypatch, "grok-4.5", make_settings("high"))
    assert "reasoning_effort" not in kwargs
    assert "allowed_openai_params" not in kwargs
    assert "temperature" in kwargs  # unchanged legacy path


@pytest.mark.asyncio
async def test_grok_with_flag_sends_reasoning_effort_like_gpt5(monkeypatch):
    settings = make_settings("high", {"enable_grok_reasoning_effort": True})
    kwargs = await run_completion(monkeypatch, "grok-4.5", settings)
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["allowed_openai_params"] == ["reasoning_effort"]
    assert "temperature" not in kwargs  # dropped exactly like the GPT-5 path


@pytest.mark.asyncio
async def test_grok_with_flag_invalid_effort_falls_back_to_medium(monkeypatch):
    settings = make_settings("bogus", {"enable_grok_reasoning_effort": True})
    kwargs = await run_completion(monkeypatch, "grok-4.5", settings)
    assert kwargs["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_gpt5_path_unchanged_by_grok_flag(monkeypatch):
    settings = make_settings("high", {"enable_grok_reasoning_effort": True})
    kwargs = await run_completion(monkeypatch, "gpt-5-2025-08-07", settings)
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["allowed_openai_params"] == ["reasoning_effort"]
    assert kwargs["model"] == "openai/gpt-5-2025-08-07"


# ---------- Claude adaptive thinking gating ----------

@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-opus-4-8", True),
        ("anthropic/claude-opus-4-8", True),
        ("claude-opus-4-7", True),
        ("claude-sonnet-5", True),
        ("bedrock/us.anthropic.claude-opus-4-7-v1:0", True),
        ("claude-opus-4-5-20251101", False),
        ("claude-sonnet-4-6", False),
        ("gpt-5", False),
    ],
)
def test_adaptive_model_detection(model, expected):
    assert LiteLLMAIHandler._is_claude_adaptive_thinking_model(model) is expected


@pytest.mark.asyncio
async def test_adaptive_claude_without_flag_sends_no_thinking(monkeypatch):
    """Default behavior must stay byte-identical: no flag => no thinking param."""
    kwargs = await run_completion(monkeypatch, "anthropic/claude-opus-4-8", make_settings("high"))
    assert "thinking" not in kwargs
    assert "output_config" not in kwargs


@pytest.mark.asyncio
async def test_adaptive_claude_with_flag_sends_adaptive_thinking(monkeypatch):
    settings = make_settings("high", {"enable_claude_adaptive_thinking": True})
    kwargs = await run_completion(monkeypatch, "anthropic/claude-opus-4-8", settings)
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high"}
    assert kwargs["temperature"] == 1


@pytest.mark.asyncio
async def test_adaptive_claude_skips_output_config_for_openai_only_effort(monkeypatch):
    settings = make_settings("minimal", {"enable_claude_adaptive_thinking": True})
    kwargs = await run_completion(monkeypatch, "anthropic/claude-opus-4-8", settings)
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert "output_config" not in kwargs


@pytest.mark.asyncio
async def test_adaptive_flag_does_not_touch_budget_token_models(monkeypatch):
    """Non-adaptive Claude models must not get the adaptive payload even when the flag is on."""
    settings = make_settings("high", {"enable_claude_adaptive_thinking": True})
    kwargs = await run_completion(monkeypatch, "claude-sonnet-4-6", settings)
    assert "thinking" not in kwargs
    assert "output_config" not in kwargs


# ---------- Dynaconf env override plumbing ----------

def test_config_reasoning_effort_env_override_reaches_settings(monkeypatch):
    """CONFIG__REASONING_EFFORT must override configuration.toml via Dynaconf's env loader."""
    monkeypatch.setenv("CONFIG__REASONING_EFFORT", "high")
    monkeypatch.setenv("CONFIG__ENABLE_CLAUDE_ADAPTIVE_THINKING", "true")
    monkeypatch.setenv("CONFIG__ENABLE_GROK_REASONING_EFFORT", "true")
    from dynaconf import Dynaconf

    from pr_agent.config_loader import dynconf_kwargs, global_settings
    fresh = Dynaconf(
        envvar_prefix=False,
        load_dotenv=False,
        settings_files=list(global_settings.settings_module),
        **dynconf_kwargs,
    )
    assert fresh.config.reasoning_effort == "high"
    assert fresh.config.get("enable_claude_adaptive_thinking") is True
    assert fresh.config.get("enable_grok_reasoning_effort") is True
    # defaults with the env vars absent stay false (guard for the concurrent replay)
    monkeypatch.delenv("CONFIG__REASONING_EFFORT")
    monkeypatch.delenv("CONFIG__ENABLE_CLAUDE_ADAPTIVE_THINKING")
    monkeypatch.delenv("CONFIG__ENABLE_GROK_REASONING_EFFORT")
    clean = Dynaconf(
        envvar_prefix=False,
        load_dotenv=False,
        settings_files=list(global_settings.settings_module),
        **dynconf_kwargs,
    )
    assert clean.config.reasoning_effort == "medium"
    assert clean.config.get("enable_claude_adaptive_thinking") is False
    assert clean.config.get("enable_grok_reasoning_effort") is False
