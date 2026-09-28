"""Wire test: LITELLM.EXTRA_BODY `prompt_cache_key` must reach the HTTP request body.

litellm (1.84.0) silently drops a top-level `prompt_cache_key` kwarg for `openai/<custom>`
models, with drop_params True or False. Only `extra_body={"prompt_cache_key": ...}` puts it
on the wire. This test posts through litellm to a fake local OpenAI-compatible endpoint and
asserts on the captured request body, not on the kwargs dict.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import litellm
import pytest

from pr_agent.algo.ai_handlers.litellm_helpers import _process_litellm_extra_body
from pr_agent.config_loader import get_settings

_COMPLETION = {
    "id": "chatcmpl-wire",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-6-sol",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.fixture
def fake_endpoint():
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            bodies.append(json.loads(self.rfile.read(length)))
            payload = json.dumps(_COMPLETION).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1", bodies
    server.shutdown()
    server.server_close()


@pytest.fixture
def extra_body():
    previous = get_settings().get("litellm.extra_body", None)

    def _set(value):
        get_settings().set("litellm.extra_body", value)

    yield _set
    get_settings().set("litellm.extra_body", previous)


def _send(api_base, **extra):
    return litellm.completion(
        model="openai/gpt-6-sol",
        messages=[{"role": "user", "content": "hi"}],
        api_base=api_base,
        api_key="sk-wire-test",
        num_retries=0,
        max_retries=0,
        **extra,
    )


@pytest.mark.parametrize("drop_params", [False, True])
def test_prompt_cache_key_reaches_the_wire(fake_endpoint, extra_body, monkeypatch, drop_params):
    api_base, bodies = fake_endpoint
    monkeypatch.setattr(litellm, "drop_params", drop_params)
    extra_body(json.dumps({"prompt_cache_key": "fr-wire-123"}))

    kwargs = _process_litellm_extra_body({})
    _send(api_base, **kwargs)

    assert len(bodies) == 1
    assert bodies[0].get("prompt_cache_key") == "fr-wire-123"


@pytest.mark.parametrize("drop_params", [False, True])
def test_top_level_prompt_cache_key_is_dropped_by_litellm(fake_endpoint, monkeypatch, drop_params):
    """Control: documents the litellm behaviour the fix routes around. If this starts
    failing after a litellm bump, top-level forwarding works again and extra_body is optional."""
    api_base, bodies = fake_endpoint
    monkeypatch.setattr(litellm, "drop_params", drop_params)

    _send(api_base, prompt_cache_key="fr-top-level")

    assert len(bodies) == 1
    assert "prompt_cache_key" not in bodies[0]
