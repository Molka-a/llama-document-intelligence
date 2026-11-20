"""Local LLM access through Ollama, with schema-constrained JSON output.

The model is forced to answer in the JSON structure of a Pydantic schema
(Ollama "structured outputs"). The answer is then validated by Pydantic.
If validation fails, the errors are sent back to the model once so it can
correct its JSON.
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Generic, TypeVar

import httpx
import ollama
from pydantic import BaseModel, ValidationError

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))      # context window in tokens
TIMEOUT_S = float(os.getenv("LLM_TIMEOUT", "600"))   # a full CV can take minutes on CPU

T = TypeVar("T", bound=BaseModel)

_client = ollama.Client(host=OLLAMA_HOST, timeout=TIMEOUT_S)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base class for LLM errors."""


class LLMUnavailableError(LLMError):
    """Ollama is not running, or the request timed out."""


class LLMModelNotFoundError(LLMError):
    """The requested model has not been pulled."""


class LLMOutputError(LLMError):
    """The model kept returning JSON that does not match the schema."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class LLMResult(Generic[T]):
    data: T                  # the validated Pydantic object
    model: str
    attempts: int            # 1 = valid on first try, 2 = needed one correction
    duration_s: float        # wall-clock time for all attempts
    prompt_tokens: int       # tokens read (last attempt)
    output_tokens: int       # tokens generated (last attempt)

    @property
    def tokens_per_second(self) -> float:
        return self.output_tokens / self.duration_s if self.duration_s else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_local_models() -> list[str]:
    """Names of the models pulled in Ollama, e.g. ['llama3.2:3b']."""
    try:
        response = _client.list()
    except (ConnectionError, httpx.ConnectError) as exc:
        raise LLMUnavailableError(f"Ollama is not running at {OLLAMA_HOST}") from exc
    return [m.model for m in response.models]


def generate_structured(
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    model: str = DEFAULT_MODEL,
    max_retries: int = 1,
) -> LLMResult[T]:
    """Ask the model for JSON matching `schema` and return it validated.

    Raises:
        LLMUnavailableError: Ollama is not reachable or timed out.
        LLMModelNotFoundError: the model is not pulled.
        LLMOutputError: still invalid after `max_retries` corrections.
    """
    json_schema = schema.model_json_schema()
    messages = [
        {"role": "system", "content": _with_schema(system_prompt, json_schema)},
        {"role": "user", "content": user_prompt},
    ]
    start = time.perf_counter()
    last_error: ValidationError | None = None

    for attempt in range(1, max_retries + 2):
        response = _chat(model, messages, json_schema)
        content = response.message.content or ""

        try:
            data = schema.model_validate_json(content)
        except ValidationError as exc:
            last_error = exc
            # Show the model its own answer and what was wrong with it
            messages += [
                {"role": "assistant", "content": content},
                {"role": "user", "content": _correction_prompt(exc)},
            ]
            continue

        return LLMResult(
            data=data,
            model=model,
            attempts=attempt,
            duration_s=round(time.perf_counter() - start, 2),
            prompt_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
        )

    raise LLMOutputError(
        f"Model '{model}' returned invalid JSON after {max_retries + 1} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _chat(model: str, messages: list[dict], json_schema: dict):
    try:
        return _client.chat(
            model=model,
            messages=messages,
            format=json_schema,                  # constrain output to the schema
            options={
                "temperature": 0,                # deterministic: same input, same output
                "num_ctx": NUM_CTX,              # room for a full CV + instructions
            },
        )
    except (ConnectionError, httpx.ConnectError) as exc:
        raise LLMUnavailableError(f"Ollama is not running at {OLLAMA_HOST}") from exc
    except httpx.TimeoutException as exc:
        raise LLMUnavailableError(f"Ollama did not answer within {TIMEOUT_S:.0f} s") from exc
    except ollama.ResponseError as exc:
        if exc.status_code == 404:
            raise LLMModelNotFoundError(
                f"Model '{model}' is not installed. Run: ollama pull {model}"
            ) from exc
        raise LLMError(f"Ollama error: {exc.error}") from exc


def _with_schema(system_prompt: str, json_schema: dict) -> str:
    """Append the schema as text: `format` constrains the output, but the
    model never reads it, so it cannot see the field descriptions otherwise."""
    return (
        f"{system_prompt}\n\n"
        "Answer with JSON only, matching this JSON schema "
        "(read each field's description):\n"
        f"{json.dumps(json_schema, ensure_ascii=False)}"
    )


def _correction_prompt(error: ValidationError, max_errors: int = 5) -> str:
    lines = [
        f"- {'.'.join(str(p) for p in e['loc']) or 'root'}: {e['msg']}"
        for e in error.errors()[:max_errors]
    ]
    return (
        "Your JSON did not pass validation:\n"
        + "\n".join(lines)
        + "\nReturn the corrected JSON only, with the same structure."
    )