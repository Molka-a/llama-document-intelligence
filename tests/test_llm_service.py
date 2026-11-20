"""LLM service tests.

Ollama is replaced by a fake client, so these tests run in a second,
without Ollama running and without downloading a model.
"""

from types import SimpleNamespace

import httpx
import ollama
import pytest
from pydantic import BaseModel, field_validator

from app.services import llm_service
from app.services.llm_service import (
    LLMModelNotFoundError,
    LLMOutputError,
    LLMUnavailableError,
    generate_structured,
)


class Person(BaseModel):
    name: str
    city: str

    @field_validator("city")
    @classmethod
    def city_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("city must not be empty")
        return value


def fake_stream(content: str, done_reason: str = "stop"):
    """Mimic Ollama streaming: several chunks, stats on the last one."""
    middle = len(content) // 2
    return [
        SimpleNamespace(message=SimpleNamespace(content=content[:middle]),
                        prompt_eval_count=None, eval_count=None, done_reason=None),
        SimpleNamespace(message=SimpleNamespace(content=content[middle:]),
                        prompt_eval_count=50, eval_count=20, done_reason=done_reason),
    ]


class FakeClient:
    """Returns the given answers in order and records every call."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, tuple):          # (content, done_reason)
            return fake_stream(*answer)
        return fake_stream(answer)


@pytest.fixture
def use_fake(monkeypatch):
    def _install(*answers):
        client = FakeClient(*answers)
        monkeypatch.setattr(llm_service, "_client", client)
        return client
    return _install


def test_valid_json_on_first_try(use_fake):
    use_fake('{"name": "Molka Amara", "city": "Rabat"}')

    result = generate_structured("Extract.", "Molka Amara lives in Rabat.", Person)

    assert result.data == Person(name="Molka Amara", city="Rabat")
    assert result.attempts == 1
    assert result.output_tokens == 20


def test_schema_and_options_are_sent(use_fake):
    client = use_fake('{"name": "A", "city": "B"}')

    generate_structured("Extract.", "text", Person)

    call = client.calls[0]
    assert call["format"]["properties"] == Person.model_json_schema()["properties"]
    assert call["options"]["temperature"] == 0
    assert call["options"]["num_ctx"] >= 4096
    assert call["options"]["num_predict"] > 0
    assert call["stream"] is True


def test_invalid_json_is_corrected_on_retry(use_fake):
    client = use_fake(
        '{"name": "Molka Amara", "city": ""}',          # fails the validator
        '{"name": "Molka Amara", "city": "Rabat"}',     # corrected
    )

    result = generate_structured("Extract.", "text", Person)

    assert result.attempts == 2
    assert result.data.city == "Rabat"
    correction = client.calls[1]["messages"][-1]["content"]
    assert "city" in correction and "did not pass validation" in correction


def test_gives_up_after_max_retries(use_fake):
    use_fake('not json', '{"name": "only a name"}')

    with pytest.raises(LLMOutputError, match="2 attempts"):
        generate_structured("Extract.", "text", Person, max_retries=1)


def test_ollama_not_running(use_fake):
    use_fake(ConnectionError("Failed to connect"))

    with pytest.raises(LLMUnavailableError, match="not running"):
        generate_structured("Extract.", "text", Person)


def test_timeout(use_fake):
    use_fake(httpx.ReadTimeout("too slow"))

    with pytest.raises(LLMUnavailableError, match="sent nothing"):
        generate_structured("Extract.", "text", Person)


def test_model_not_pulled(use_fake):
    use_fake(ollama.ResponseError("model 'x' not found", status_code=404))

    with pytest.raises(LLMModelNotFoundError, match="ollama pull"):
        generate_structured("Extract.", "text", Person, model="x")


def test_schema_is_written_into_system_prompt(use_fake):
    client = use_fake('{"name": "A", "city": "B"}')

    generate_structured("Extract the person.", "text", Person)

    system = client.calls[0]["messages"][0]["content"]
    assert system.startswith("Extract the person.")
    assert '"city"' in system and "JSON schema" in system


def test_streamed_tokens_are_passed_to_callback(use_fake):
    use_fake('{"name": "Molka Amara", "city": "Rabat"}')
    received = []

    result = generate_structured("Extract.", "text", Person, on_token=received.append)

    assert "".join(received) == '{"name": "Molka Amara", "city": "Rabat"}'
    assert result.prompt_tokens == 50 and result.output_tokens == 20


def test_runaway_output_stops_without_retry(use_fake):
    client = use_fake(('{"name": "A", "city": "B B B B', "length"))

    with pytest.raises(LLMOutputError, match="token output limit"):
        generate_structured("Extract.", "text", Person, max_retries=1)
    assert len(client.calls) == 1     # no second attempt


def test_every_field_is_required_for_the_model():
    """Optional fields must still be written by the model (at least as null)."""
    from typing import Optional

    class Job(BaseModel):
        title: str
        start_date: Optional[str] = None

    class Cv(BaseModel):
        name: Optional[str] = None
        jobs: list[Job] = []

    strict = llm_service._require_all_fields(Cv.model_json_schema())

    assert strict["required"] == ["name", "jobs"]
    assert strict["$defs"]["Job"]["required"] == ["title", "start_date"]
    assert "required" not in Cv.model_json_schema()      # original untouched