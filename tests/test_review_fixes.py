"""Regression tests for defects found by the adversarial code review."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd
import pytest

from zengziagent.alignment import TextAligner, normalize_for_match
from zengziagent.analyzer import Analyzer, parse_xml_annotations
from zengziagent.evaluation.agreement import inter_run_agreement
from zengziagent.experiments.analyze_ablation import _unit_correct
from zengziagent.llm.base import ChatMessage
from zengziagent.llm.mock import MockBackend
from zengziagent.llm.openai_compat import LLMRequestError, OpenAICompatibleBackend
from zengziagent.tokenization import RegexTokenizer


def test_alignment_cache_is_keyed_by_value_not_identity():
    al = TextAligner(tokenizer=RegexTokenizer())
    a = "The paper is interesting, but the experimental validation is insufficient."
    assert al._source_norm(a) == normalize_for_match(a)
    b = "Completely different reviewer comment with the same length as the other."
    assert al._source_norm(b) == normalize_for_match(b)
    # simulate object-id reuse: a new string object equal in content still maps correctly
    c = "".join(list(a))
    assert c is not a and al._source_norm(c)[0] == normalize_for_match(a)[0]


def test_numeric_character_references_are_decoded():
    raw = "<annotations><annotation><text>don&#39;t agree &#x2019;s &amp;lt;</text><label>Eval_neg</label></annotation></annotations>"
    p = parse_xml_annotations(raw)
    assert p.annotations[0].text == "don't agree ’s &lt;"


def test_refinement_skips_empty_assistant_turn():
    backend = MockBackend()
    captured = {}

    def fake_complete(messages, params=None, run_index=0, tag=""):
        captured["messages"] = messages
        return backend.__class__.complete(backend, messages, params, run_index, tag)

    backend.complete = fake_complete
    an = Analyzer(backend)
    from zengziagent.actor import PromptBundle

    bundle = PromptBundle("sys", '<review id="{review_id}">\n{text}\n</review>', "fixed", "none", None, "h")
    an.annotate(bundle, "r1", "The paper is good.", validation_feedback="- EMPTY_OUTPUT", previous_output="")
    roles = [m.role for m in captured["messages"]]
    assert "assistant" not in roles and "previous response was empty" in captured["messages"][-1].content
    an.annotate(bundle, "r1", "The paper is good.", validation_feedback="- x", previous_output="<annotations></annotations>")
    assert [m.role for m in captured["messages"]] == ["system", "user", "assistant", "user"]


class _Unauthorized(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):  # noqa: N802
        _Unauthorized.calls += 1
        body = b'{"error": {"message": "invalid api key"}}'
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_non_retryable_http_errors_fail_fast(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    srv = HTTPServer(("127.0.0.1", 0), _Unauthorized)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        be = OpenAICompatibleBackend(provider="custom", model="m", api_key="bad", base_url=f"http://127.0.0.1:{srv.server_port}/v1", max_retries=5)
        with pytest.raises(LLMRequestError):
            be.complete([ChatMessage("user", "hi")])
        assert _Unauthorized.calls == 1
    finally:
        srv.shutdown()


def test_inter_run_agreement_reports_dispersion():
    r = inter_run_agreement([["a", "b", "c", "d"], ["a", "b", "c", "x"], ["a", "y", "c", "d"]])
    assert "pairwise_agreement_sd" in r and r["pairwise_agreement_sd"] >= 0


def test_mcnemar_pooled_policy_collapses_to_distinct_units():
    units = pd.DataFrame(
        {"unit_id": ["u1", "u1", "u2", "u2", "u3", "u3"], "run": [0, 1, 0, 1, 0, 1], "correct": [1, 1, 1, 0, 0, 0]}
    )
    pooled = _unit_correct(units, "pooled")
    assert list(pooled.index) == ["u1", "u2", "u3"] and pooled.tolist() == [1, 1, 0]
    run0 = _unit_correct(units, "run0")
    assert run0.tolist() == [1, 1, 0]


def test_agreement_pivot_keeps_units_never_predicted():
    u = pd.DataFrame({"unit_id": ["a", "a", "b", "b"], "run": [0, 1, 0, 1], "pred_label": [None, None, "Eval_pos", "Eval_pos"]})
    piv = u.assign(pl=u["pred_label"].fillna("None")).pivot_table(index="unit_id", columns="run", values="pl", aggfunc="first").fillna("None")
    assert len(piv) == 2
    r = inter_run_agreement([piv[k].tolist() for k in piv.columns])
    assert r["pairwise_agreement"] == 1.0


def test_reported_view_and_manuscript_cross_check():
    from zengziagent.evaluation.audit import compare_with_manuscript, reported_view

    rows = []
    for tau in (0.5, 1.0):
        for run in (0, 1):
            rows.append({"dataset": "d", "backend": "b", "config_id": "F", "run": run, "round": 1, "tau": tau, "n_units": 10, "n_correct": 8, "tp": 7 if tau == 0.5 else 3, "fp": 2, "fn": 3 if tau == 0.5 else 7, "accuracy": 0.8, "precision": 0.0, "recall": 0.0, "f1": 0.0})
    master = pd.DataFrame(rows)
    view = reported_view(master, 0.5)
    assert len(view) == 1 and view.iloc[0]["tp"] == 14 and abs(view.iloc[0]["precision"] - 14 / 18) < 1e-9
    manuscript = pd.DataFrame([{"dataset": "d", "backend": "b", "config_id": "F", "precision": round(14 / 18, 4), "recall": round(14 / 20, 4)}])
    cmp = compare_with_manuscript(master, manuscript, tol=1e-4)
    assert cmp["ok"].all()


class _Flaky(BaseHTTPRequestHandler):
    """First two calls: HTTP 200 with an empty completion; third: truncated; fourth: fine."""

    calls = 0

    def do_POST(self):  # noqa: N802
        _Flaky.calls += 1
        n = _Flaky.calls
        if n <= 2:
            payload = {"id": "x", "model": "m", "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}]}
        elif n == 3:
            payload = {"id": "x", "model": "m", "choices": [{"message": {"role": "assistant", "content": "<annotations><annotation><text>a</text>"}, "finish_reason": "length"}]}
        else:
            payload = {"id": "x", "model": "m", "choices": [{"message": {"role": "assistant", "content": "<annotations></annotations>"}, "finish_reason": "stop"}]}
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_empty_completions_are_retried_and_truncations_not_cached(monkeypatch, tmp_path):
    from zengziagent.llm.cache import ResponseCache

    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setattr("zengziagent.llm.openai_compat.time.sleep", lambda s: None)
    srv = HTTPServer(("127.0.0.1", 0), _Flaky)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cache = ResponseCache(tmp_path / "c.sqlite")
        be = OpenAICompatibleBackend(provider="custom", model="m", api_key="k", base_url=f"http://127.0.0.1:{srv.server_port}/v1", cache=cache, max_retries=5)
        r = be.complete([ChatMessage("user", "one")])  # two empty payloads are retried, third is truncated
        assert _Flaky.calls == 3 and r.finish_reason == "length" and r.text.startswith("<annotations>")
        assert cache.count() == 0  # truncated completion not cached
        r2 = be.complete([ChatMessage("user", "one")])
        assert _Flaky.calls == 4 and r2.finish_reason == "stop" and not r2.cached and cache.count() == 1
        r3 = be.complete([ChatMessage("user", "one")])
        assert r3.cached and _Flaky.calls == 4
    finally:
        srv.shutdown()
