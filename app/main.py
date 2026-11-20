from fastapi import FastAPI

from app.api.routes import router as document_router

app = FastAPI(
    title="Llama Document Intelligence",
    version="0.3.0",
    description="Extract, clean and understand PDF and Word documents with a local Llama model.",
)

app.include_router(document_router)


@app.get("/")
def root():
    return {"message": "Llama Document Intelligence API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}
