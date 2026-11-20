from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.routes import jobs_router
from app.api.routes import router as document_router
from app.services.llm_service import DEFAULT_MODEL, LLMError, list_local_models

app = FastAPI(
    title="Llama Document Intelligence",
    version="0.4.0",
    description="Extract, clean and understand PDF and Word documents with a local Llama model.",
)

app.include_router(document_router)
app.include_router(jobs_router)


@app.get("/")
def root():
    return {"message": "Llama Document Intelligence API is running"}


@app.get("/health")
def health():
    """Liveness: the API process is up. Fast, no dependencies."""
    return {"status": "ok"}


@app.get("/health/llm")
def health_llm():
    """Readiness of the LLM layer: is Ollama running and is the model installed?

    Returns 200 when extraction can run, 503 otherwise, with the reason.
    """
    try:
        models = list_local_models()
    except LLMError as exc:
        return JSONResponse(status_code=503, content={
            "status": "unavailable",
            "ollama": "unreachable",
            "model": DEFAULT_MODEL,
            "detail": str(exc),
        })

    installed = DEFAULT_MODEL in models
    body = {
        "status": "ok" if installed else "unavailable",
        "ollama": "running",
        "model": DEFAULT_MODEL,
        "model_installed": installed,
        "installed_models": models,
    }
    if not installed:
        body["detail"] = f"Model not installed. Run: ollama pull {DEFAULT_MODEL}"
    return JSONResponse(status_code=200 if installed else 503, content=body)