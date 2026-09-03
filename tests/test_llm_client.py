"""OpenAI-compatible client against a local fake server: retry on 429, caching, model-id capture."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from zengziagent.llm.base import ChatMessage
from zengziagent.llm.cache import ResponseCache
from zengziagent.llm.openai_compat import OpenAICompatibleBackend
from zengziagent.schema import SamplingParams


class _Handler(BaseHTTPRequestHandler):
    calls: list = []
    fail_next = False

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/v1/models"):
            self._send(200, {"data": [{"id": "fake-model"}, {"id": "other"}]})
        else:
            self._send(404, {"error": "nope"})

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        _Handler.calls.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        if _Handler.fail_next:
            _Handler.fail_next = False
            self._send(429, {"error": {"message": "rate limited"}})
            return
        self._send(200, {"id": "req-1", "model": "fake-model-2024-01-01", "choices": [{"message": {"role": "assistant", "content": "<annotations></annotations>"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 12, "completion_tokens": 3}})

    def log_message(self, *args):  # silence
        pass


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_client_retries_caches_and_records_model(server, tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    cache = ResponseCache(tmp_path / "cache.sqlite")
    be = OpenAICompatibleBackend(provider="custom", model="fake-model", api_key="secret", base_url=server, cache=cache, max_retries=2)
    assert be.list_models() == ["fake-model", "other"] and be.check_model_available() is True
    _Handler.calls.clear()
    _Handler.fail_next = True
    msgs = [ChatMessage("system", "sys"), ChatMessage("user", "hello")]
    resp = be.complete(msgs, SamplingParams(temperature=0.0, max_tokens=64, seed=7), run_index=0)
    assert resp.text == "<annotations></annotations>"
    assert resp.model_requested == "fake-model" and resp.model_returned == "fake-model-2024-01-01"
    assert resp.prompt_tokens == 12 and not resp.cached
    assert len(_Handler.calls) == 2  # one 429 + one success
    assert _Handler.calls[-1]["auth"] == "Bearer secret"
    assert _Handler.calls[-1]["body"]["seed"] == 7 and _Handler.calls[-1]["body"]["temperature"] == 0.0
    assert _Handler.calls[-1]["path"].endswith("/v1/chat/completions")
    again = be.complete(msgs, SamplingParams(temperature=0.0, max_tokens=64, seed=7), run_index=0)
    assert again.cached and len(_Handler.calls) == 2
    other_run = be.complete(msgs, SamplingParams(temperature=0.0, max_tokens=64, seed=7), run_index=1)
    assert not other_run.cached and len(_Handler.calls) == 3 and _Handler.calls[-1]["body"]["seed"] == 8
    assert cache.count() == 2
