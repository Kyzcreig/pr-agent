"""LITELLM.EXTRA_BODY may carry an OpenAI `prompt_cache_key` (FleetReview v2 spec §5.7, D11)."""
import json

import pytest

from pr_agent.algo.ai_handlers.litellm_helpers import _process_litellm_extra_body
from pr_agent.config_loader import get_settings


@pytest.fixture
def extra_body():
    previous = get_settings().get("litellm.extra_body", None)

    def _set(value):
        get_settings().set("litellm.extra_body", value)

    yield _set
    get_settings().set("litellm.extra_body", previous)


def test_prompt_cache_key_is_forwarded(extra_body):
    extra_body(json.dumps({"prompt_cache_key": "fr-abc123"}))
    kwargs = _process_litellm_extra_body({"model": "gpt-6-sol"})
    assert kwargs == {"model": "gpt-6-sol", "prompt_cache_key": "fr-abc123"}


def test_unknown_key_is_still_refused(extra_body):
    extra_body(json.dumps({"prompt_cache_key": "k", "temperature": 0}))
    with pytest.raises(ValueError, match="unsupported keys: temperature"):
        _process_litellm_extra_body({"model": "gpt-6-sol"})


def test_prompt_cache_key_cannot_override_existing_kwarg(extra_body):
    extra_body(json.dumps({"prompt_cache_key": "k"}))
    with pytest.raises(ValueError, match="cannot override"):
        _process_litellm_extra_body({"model": "m", "prompt_cache_key": "other"})
