"""FastAPI application: routes, health checks, logging and error handling."""

import logging
import time
import uuid
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.api.routes import jobs_router
from app.api.routes import router as document_router
from app.core.logging import setup_logging
from app.services.llm_service import DEFAULT_MODEL, LLMError, list_local_models

setup_logging()
logger = logging.getLogger("app")

app = FastAPI(
    title="Llama Document Intelligence",
    version="1.0.0",
    description="Extract, clean and understand PDF and Word documents with a local Llama model.",
)

app.include_router(document_router)
app.include_router(jobs_router)


# ---------------------------------------------------------------------------
# Request logging
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Give every request an id (returned in X-Request-ID) and log its duration."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.perf_counter()

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info("%s %s -> %s in %.0f ms [%s]", request.method, request.url.path,
                response.status_code, elapsed_ms, request_id)
    return response


# ---------------------------------------------------------------------------
# Error handling: always JSON, always with the request id
# ---------------------------------------------------------------------------

def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    logger.warning("LLM unavailable: %s [%s]", exc, _request_id(request))
    return JSONResponse(status_code=503,
                        content={"detail": str(exc), "request_id": _request_id(request)})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    """Unexpected bug: full traceback in the logs, no internals leaked to the client."""
    logger.exception("Unhandled error on %s %s [%s]", request.method, request.url.path,
                     _request_id(request))
    return JSONResponse(status_code=500, content={
        "detail": "Internal server error. The request id helps find it in the server logs.",
        "request_id": _request_id(request),
    })


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class LLMHealth(BaseModel):
    status: Literal["ok", "unavailable"]
    ollama: Literal["running", "unreachable"]
    model: str
    model_installed: bool | None = None
    installed_models: list[str] = []
    detail: str | None = None


@app.get("/")
def root():
    return {"message": "Llama Document Intelligence API is running", "docs": "/docs"}


@app.get("/health")
def health():
    """Liveness: the API process is up. Fast, no dependencies."""
    return {"status": "ok"}


@app.get("/health/llm", response_model=LLMHealth, responses={503: {"model": LLMHealth}})
def health_llm():
    """Readiness of the LLM layer: is Ollama running and is the model installed?

    Returns 200 when extraction can run, 503 otherwise, with the reason.
    """
    try:
        models = list_local_models()
    except LLMError as exc:
        body = LLMHealth(status="unavailable", ollama="unreachable", model=DEFAULT_MODEL,
                         detail=str(exc))
        return JSONResponse(status_code=503, content=body.model_dump())

    installed = DEFAULT_MODEL in models
    body = LLMHealth(
        status="ok" if installed else "unavailable",
        ollama="running",
        model=DEFAULT_MODEL,
        model_installed=installed,
        installed_models=models,
        detail=None if installed else f"Model not installed. Run: ollama pull {DEFAULT_MODEL}",
    )
    return JSONResponse(status_code=200 if installed else 503, content=body.model_dump())