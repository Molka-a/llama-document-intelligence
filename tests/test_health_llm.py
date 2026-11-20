"""/health/llm tests: Ollama is replaced, so they run without it."""

from fastapi.testclient import TestClient

import app.main as main
from app.services.llm_service import DEFAULT_MODEL, LLMUnavailableError

client = TestClient(main.app)


def test_basic_health_does_not_need_ollama():
    assert client.get("/health").json() == {"status": "ok"}


def test_llm_ready(monkeypatch):
    monkeypatch.setattr(main, "list_local_models", lambda: [DEFAULT_MODEL, "llama3.2:1b"])

    response = client.get("/health/llm")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_installed"] is True


def test_model_not_installed(monkeypatch):
    monkeypatch.setattr(main, "list_local_models", lambda: ["some-other-model"])

    response = client.get("/health/llm")

    assert response.status_code == 503
    assert response.json()["model_installed"] is False
    assert "ollama pull" in response.json()["detail"]


def test_ollama_not_running(monkeypatch):
    def down():
        raise LLMUnavailableError("Ollama is not running at http://127.0.0.1:11434")

    monkeypatch.setattr(main, "list_local_models", down)

    response = client.get("/health/llm")

    assert response.status_code == 503
    assert response.json()["ollama"] == "unreachable"
