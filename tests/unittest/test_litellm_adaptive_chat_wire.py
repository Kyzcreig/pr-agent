"""Adaptive Claude on an OpenAI chat route must emit chat wire only.

An ``openai/<claude>`` model rides a chat-completions bridge. The Anthropic-only adaptive
fields would make the body mixed-wire: litellm drops ``thinking`` under drop_params but
forwards ``output_config``, and a strict chat relay rejects the body (wire_body_mismatch).
On that route the effort travels as ``reasoning_effort``; the Anthropic route is unchanged.
"""
import pytest

from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
from tests.unittest.test_litellm_adaptive_and_grok import make_settings, run_completion

FLAG = {"enable_claude_adaptive_thinking": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh"])
async def test_chat_route_sends_reasoning_effort_and_no_anthropic_fields(monkeypatch, effort):
    kwargs = await run_completion(monkeypatch, "openai/claude-sonnet-5-5", make_settings(effort, FLAG))
    assert "thinking" not in kwargs
    assert "output_config" not in kwargs
    assert kwargs["reasoning_effort"] == effort
    assert kwargs["allowed_openai_params"] == ["reasoning_effort"]
    assert kwargs["temperature"] == 1
    assert kwargs["model"] == "openai/claude-sonnet-5-5"


@pytest.mark.asyncio
async def test_chat_route_skips_openai_only_effort(monkeypatch):
    kwargs = await run_completion(monkeypatch, "openai/claude-sonnet-5-5", make_settings("minimal", FLAG))
    assert "thinking" not in kwargs
    assert "output_config" not in kwargs
    assert "reasoning_effort" not in kwargs


@pytest.mark.asyncio
async def test_anthropic_route_keeps_native_adaptive_fields(monkeypatch):
    kwargs = await run_completion(monkeypatch, "anthropic/claude-sonnet-5-5", make_settings("high", FLAG))
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high"}
    assert "reasoning_effort" not in kwargs


@pytest.mark.asyncio
async def test_chat_route_without_flag_is_unchanged(monkeypatch):
    kwargs = await run_completion(monkeypatch, "openai/claude-sonnet-5-5", make_settings("high"))
    for key in ("thinking", "output_config", "reasoning_effort", "allowed_openai_params"):
        assert key not in kwargs


@pytest.mark.parametrize(
    "model,expected",
    [
        ("openai/claude-sonnet-5-5", True),
        ("anthropic/claude-sonnet-5-5", False),
        ("claude-opus-4-8", False),
        ("bedrock/us.anthropic.claude-opus-4-7-v1:0", False),
    ],
)
def test_routes_chat_wire(model, expected):
    assert LiteLLMAIHandler._routes_chat_wire(model) is expected
