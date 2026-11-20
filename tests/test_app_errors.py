"""Phase 6: request ids, logging and JSON error responses."""

import logging

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.llm_service import LLMUnavailableError


def _boom():
    raise RuntimeError("secret internal detail")


def _llm_down():
    raise LLMUnavailableError("Ollama is not running at http://127.0.0.1:11434")


# test-only routes that fail on purpose
app.add_api_route("/_test/boom", _boom)
app.add_api_route("/_test/llm-down", _llm_down)

# raise_server_exceptions=False: behave like a real client, receive the 500
client = TestClient(app, raise_server_exceptions=False)


def test_every_response_has_a_request_id():
    response = client.get("/health")
    assert len(response.headers["X-Request-ID"]) == 12


def test_client_request_id_is_reused():
    response = client.get("/health", headers={"X-Request-ID": "trace-abc-123"})
    assert response.headers["X-Request-ID"] == "trace-abc-123"


def test_requests_are_logged(caplog):
    with caplog.at_level(logging.INFO, logger="app"):
        client.get("/health", headers={"X-Request-ID": "log-check"})
    assert any("GET /health -> 200" in r.message and "log-check" in r.message
               for r in caplog.records)


def test_unexpected_error_returns_json_without_internals(caplog):
    with caplog.at_level(logging.ERROR, logger="app"):
        response = client.get("/_test/boom", headers={"X-Request-ID": "bug-42"})

    assert response.status_code == 500
    body = response.json()
    assert body["request_id"] == "bug-42"
    assert "secret internal detail" not in response.text          # nothing leaked
    assert any("bug-42" in r.message for r in caplog.records)      # but logged


def test_llm_errors_become_503():
    response = client.get("/_test/llm-down")
    assert response.status_code == 503
    assert "not running" in response.json()["detail"]
    assert response.json()["request_id"]


@pytest.mark.parametrize("status", ["200", "503"])
def test_health_llm_is_documented(status):
    schema = app.openapi()["paths"]["/health/llm"]["get"]["responses"][status]
    ref = schema["content"]["application/json"]["schema"]["$ref"]
    assert ref.endswith("/LLMHealth")