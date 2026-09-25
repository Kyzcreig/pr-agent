"""Unit tests for the retry-configuration knobs in litellm_ai_handler.

Covers config.retry_same_model_on_timeout and config.num_retries (ported from upstream
qodo-ai/pr-agent c69c2f4d) plus the fork-only config.same_model_attempts: defaults preserve
the previous behavior, env-style string values parse the way an operator expects, and
malformed values are ignored rather than raised.

The fault-injection test at the bottom stands up a local HTTP server that answers every
request with 503 and counts hits, proving how many provider calls one chat_completion makes
under each configuration.
"""

import http.server
import threading

import httpx
import openai
import pytest

from pr_agent.algo.ai_handlers.litellm_ai_handler import (
    MODEL_RETRIES,
    LiteLLMAIHandler,
    _as_bool,
    _configured_client_retries,
    _configured_same_model_attempts,
    _should_retry_same_model,
)
from pr_agent.config_loader import get_settings
from tests.unittest._settings_helpers import restore_settings, snapshot_settings

KEYS = (
    "config.retry_same_model_on_timeout",
    "config.num_retries",
    "config.same_model_attempts",
    "config.ai_timeout",
    "config.model",
    "config.fallback_models",
    "openai.key",
    "openai.api_base",
)


@pytest.fixture(autouse=True)
def _isolate_settings():
    snap = snapshot_settings(KEYS)
    yield
    restore_settings(snap)


def _timeout_error():
    return openai.APITimeoutError(request=httpx.Request("POST", "http://model.invalid"))


def _api_error():
    return openai.APIError("boom", request=httpx.Request("POST", "http://model.invalid"), body=None)


class TestShouldRetrySameModel:
    def test_default_retries_timeouts(self):
        assert _should_retry_same_model(_timeout_error()) is True

    @pytest.mark.parametrize("value", [False, "false", "False", "0", "no", "off"])
    def test_disabled_hands_timeout_to_fallback(self, value):
        get_settings().set("config.retry_same_model_on_timeout", value)
        assert _should_retry_same_model(_timeout_error()) is False

    @pytest.mark.parametrize("value", [True, "true", "TRUE", "1", "yes", "on"])
    def test_enabled_keeps_retrying(self, value):
        get_settings().set("config.retry_same_model_on_timeout", value)
        assert _should_retry_same_model(_timeout_error()) is True

    def test_rate_limit_never_retries_same_model(self):
        err = openai.RateLimitError(
            "slow down", response=httpx.Response(429, request=httpx.Request("POST", "http://model.invalid")), body=None
        )
        assert _should_retry_same_model(err) is False

    def test_other_api_errors_still_retry(self):
        assert _should_retry_same_model(_api_error()) is True

    def test_non_api_errors_never_retry(self):
        assert _should_retry_same_model(ValueError("not an API error")) is False


class TestAsBool:
    def test_non_string_non_bool_falls_back_to_default(self):
        assert _as_bool(object(), default=True) is True
        assert _as_bool(None, default=False) is False


class TestConfiguredClientRetries:
    def test_unset_means_client_defaults(self):
        assert _configured_client_retries() is None

    @pytest.mark.parametrize("value,expected", [(0, 0), (2, 2), ("0", 0), (" 3 ", 3)])
    def test_valid_values_parse(self, value, expected):
        get_settings().set("config.num_retries", value)
        assert _configured_client_retries() == expected

    @pytest.mark.parametrize("value", ["abc", "1.5", "", -1, "-2"])
    def test_invalid_values_are_ignored_not_raised(self, value):
        get_settings().set("config.num_retries", value)
        assert _configured_client_retries() is None


class TestConfiguredSameModelAttempts:
    def test_unset_keeps_model_retries(self):
        assert _configured_same_model_attempts() == MODEL_RETRIES

    @pytest.mark.parametrize("value,expected", [(1, 1), ("1", 1), (" 3 ", 3)])
    def test_valid_values_parse(self, value, expected):
        get_settings().set("config.same_model_attempts", value)
        assert _configured_same_model_attempts() == expected

    @pytest.mark.parametrize("value", ["abc", "", 0, "-1"])
    def test_invalid_values_keep_default(self, value):
        get_settings().set("config.same_model_attempts", value)
        assert _configured_same_model_attempts() == MODEL_RETRIES


class _Always503(http.server.BaseHTTPRequestHandler):
    hits = 0

    def do_POST(self):  # noqa: N802 (http.server API)
        type(self).hits += 1
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = b'{"error": {"message": "injected 503", "type": "server_error"}}'
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def always_503_server():
    _Always503.hits = 0
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Always503)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def _provider_calls_for_one_chat_completion(api_base: str) -> int:
    import asyncio

    get_settings().set("config.ai_timeout", 10)
    get_settings().set("openai.key", "sk-fault-injection")
    get_settings().set("openai.api_base", api_base)
    handler = LiteLLMAIHandler()
    handler.api_base = api_base
    with pytest.raises(Exception):
        asyncio.run(handler.chat_completion(model="openai/fault-injected", system="s", user="u"))
    return _Always503.hits


class TestFaultInjected503:
    def test_all_knobs_pinned_gives_exactly_one_provider_call(self, always_503_server):
        get_settings().set("config.num_retries", 0)
        get_settings().set("config.retry_same_model_on_timeout", False)
        get_settings().set("config.same_model_attempts", 1)
        assert _provider_calls_for_one_chat_completion(always_503_server) == 1

    def test_defaults_stack_retries(self, always_503_server):
        # Baseline: unpinned, the handler's tenacity layer times the client's own retries.
        assert _provider_calls_for_one_chat_completion(always_503_server) > 1
