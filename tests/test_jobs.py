"""Background jobs: the manager on its own, then through the API.

The LLM processor is replaced by fast fakes, so a job finishes in
milliseconds instead of minutes.
"""

import threading
import time

import pytest
from fastapi.testclient import TestClient

import app.api.routes as routes
from app.main import app
from app.services.job_service import JobManager, JobStatus
from app.services.llm_service import DEFAULT_MODEL, LLMUnavailableError


def wait_for(manager_or_client, job_id, timeout=3.0):
    """Poll until the job is finished (completed or failed)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if isinstance(manager_or_client, JobManager):
            job = manager_or_client.get(job_id).model_dump()
        else:
            job = manager_or_client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish in {timeout} s")


# --- JobManager ----------------------------------------------------------

def test_job_completes_with_outcome():
    manager = JobManager(processor=lambda text, kind: {"document_type": "cv", "echo": text})

    job = manager.submit("JANE DOE", "cv.pdf", "auto")
    assert job.status == JobStatus.QUEUED

    done = wait_for(manager, job.job_id)
    assert done["status"] == "completed"
    assert done["outcome"] == {"document_type": "cv", "echo": "JANE DOE"}
    assert done["duration_s"] is not None and done["error"] is None


def test_failure_is_recorded_in_the_job():
    def broken(text, kind):
        raise LLMUnavailableError("Ollama is not running")

    manager = JobManager(processor=broken)
    done = wait_for(manager, manager.submit("text", "cv.pdf").job_id)

    assert done["status"] == "failed"
    assert "Ollama is not running" in done["error"]


def test_jobs_run_one_at_a_time_in_order():
    release = threading.Event()
    order = []

    def slow(text, kind):
        order.append(text)
        release.wait(2)
        return {}

    manager = JobManager(processor=slow)
    first = manager.submit("first", "a.pdf")
    second = manager.submit("second", "b.pdf")
    time.sleep(0.1)

    assert manager.get(first.job_id).status == JobStatus.PROCESSING
    assert manager.get(second.job_id).status == JobStatus.QUEUED
    assert manager.get(second.job_id).queue_position == 1

    release.set()
    wait_for(manager, second.job_id)
    assert order == ["first", "second"]


def test_unknown_job_returns_none():
    assert JobManager(processor=lambda t, k: {}).get("nope") is None


def test_old_finished_jobs_are_forgotten():
    manager = JobManager(processor=lambda t, k: {}, max_jobs=2)
    ids = []
    for i in range(3):
        ids.append(manager.submit(f"t{i}", "a.pdf").job_id)
        wait_for(manager, ids[-1])
    assert manager.get(ids[0]) is None and manager.get(ids[2]) is not None


# --- API -----------------------------------------------------------------

@pytest.fixture
def api(monkeypatch):
    """API client with Ollama 'running' and a fake, instant extraction."""
    monkeypatch.setattr(routes, "list_local_models", lambda: [DEFAULT_MODEL])
    manager = JobManager(processor=lambda text, kind: {
        "document_type": "cv" if kind == "auto" else kind,
        "result": {"chars": len(text)},
    })
    monkeypatch.setattr(routes, "job_manager", manager)
    return TestClient(app)


def test_extract_returns_202_then_result(api, make_pdf):
    response = api.post("/documents/extract",
                        files={"file": ("cv.pdf", make_pdf("JANE DOE\nPROFILE"), "application/pdf")})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["status_url"] == f"/jobs/{body['job_id']}"

    done = wait_for(api, body["job_id"])
    assert done["status"] == "completed"
    assert done["filename"] == "cv.pdf"
    assert done["outcome"]["document_type"] == "cv"


def test_extract_with_explicit_type(api, make_pdf):
    response = api.post("/documents/extract?document_type=cover_letter",
                        files={"file": ("l.pdf", make_pdf("Dear team"), "application/pdf")})
    done = wait_for(api, response.json()["job_id"])
    assert done["requested_type"] == "cover_letter"
    assert done["outcome"]["document_type"] == "cover_letter"


def test_extract_rejects_invalid_type(api, make_pdf):
    response = api.post("/documents/extract?document_type=invoice",
                        files={"file": ("a.pdf", make_pdf(), "application/pdf")})
    assert response.status_code == 422


def test_extract_rejects_document_without_text(api):
    import pymupdf
    doc = pymupdf.open()
    doc.new_page()
    blank = doc.tobytes()

    response = api.post("/documents/extract", files={"file": ("scan.pdf", blank, "application/pdf")})

    assert response.status_code == 422
    assert "OCR" in response.json()["detail"]


def test_extract_keeps_upload_checks(api):
    response = api.post("/documents/extract", files={"file": ("a.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_extract_fails_fast_when_ollama_is_down(api, monkeypatch, make_pdf):
    def down():
        raise LLMUnavailableError("Ollama is not running at http://127.0.0.1:11434")

    monkeypatch.setattr(routes, "list_local_models", down)

    response = api.post("/documents/extract", files={"file": ("a.pdf", make_pdf(), "application/pdf")})

    assert response.status_code == 503
    assert "not running" in response.json()["detail"]


def test_extract_fails_fast_when_model_missing(api, monkeypatch, make_pdf):
    monkeypatch.setattr(routes, "list_local_models", lambda: ["other-model"])

    response = api.post("/documents/extract", files={"file": ("a.pdf", make_pdf(), "application/pdf")})

    assert response.status_code == 503
    assert "ollama pull" in response.json()["detail"]


def test_unknown_job_is_404(api):
    assert api.get("/jobs/does-not-exist").status_code == 404


def test_upload_endpoint_still_works(api, make_pdf):
    response = api.post("/documents/upload", files={"file": ("a.pdf", make_pdf(), "application/pdf")})
    assert response.status_code == 200
    assert response.json()["message"] == "PDF processed successfully"
