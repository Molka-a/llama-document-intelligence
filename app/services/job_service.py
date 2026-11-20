"""Background jobs for long-running LLM extractions.

An extraction takes minutes on CPU, far too long for one HTTP request.
The API stores a job, returns its id immediately, and a single worker
thread processes jobs one at a time in arrival order. One at a time is
deliberate: on an 8 GB CPU-only machine, two parallel generations would
each run at half speed and compete for memory.

Jobs live in memory: they are lost when the server restarts. That is
enough for a local tool; a database or Redis queue would replace this
in production.
"""

import queue
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel

from app.services.extraction_service import RequestedType, process_document


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(BaseModel):
    job_id: str
    status: JobStatus
    filename: str
    requested_type: str
    queue_position: int | None = None       # jobs waiting ahead of this one (0 = next)
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_s: float | None = None          # processing time, without the wait in the queue
    outcome: dict[str, Any] | None = None    # classification, result, LLM stats
    error: str | None = None


Processor = Callable[[str, RequestedType], dict[str, Any]]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobManager:
    def __init__(self, processor: Processor, max_jobs: int = 100):
        self._processor = processor
        self._max_jobs = max_jobs
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._texts: dict[str, str] = {}      # kept apart: never returned by the API
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    # -- public -------------------------------------------------------------

    def submit(self, text: str, filename: str, requested_type: RequestedType = "auto") -> Job:
        job = Job(
            job_id=uuid.uuid4().hex,
            status=JobStatus.QUEUED,
            filename=filename,
            requested_type=requested_type,
            created_at=_now(),
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._texts[job.job_id] = text
            self._forget_old_jobs()
        self._queue.put(job.job_id)
        self._ensure_worker()
        return self.get(job.job_id)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job = job.model_copy()
            if job.status == JobStatus.QUEUED:
                job.queue_position = sum(
                    1 for other in self._jobs.values()
                    if other.status in (JobStatus.QUEUED, JobStatus.PROCESSING)
                    and other.created_at < job.created_at
                )
            return job

    # -- internals ----------------------------------------------------------

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, name="llm-worker", daemon=True)
                self._worker.start()

    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run(job_id)
            finally:
                self._queue.task_done()

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            text = self._texts.pop(job_id, None)
            if job is None or text is None:
                return
            job.status, job.started_at = JobStatus.PROCESSING, _now()

        try:
            outcome = self._processor(text, job.requested_type)
            status, error = JobStatus.COMPLETED, None
        except Exception as exc:                       # any failure ends up in the job
            outcome, status, error = None, JobStatus.FAILED, f"{type(exc).__name__}: {exc}"

        with self._lock:
            job.status, job.outcome, job.error = status, outcome, error
            job.finished_at = _now()
            job.duration_s = round((job.finished_at - job.started_at).total_seconds(), 2)

    def _forget_old_jobs(self) -> None:
        """Keep memory bounded: drop the oldest finished jobs beyond max_jobs."""
        finished = [jid for jid, j in self._jobs.items()
                    if j.status in (JobStatus.COMPLETED, JobStatus.FAILED)]
        while len(self._jobs) > self._max_jobs and finished:
            self._jobs.pop(finished.pop(0), None)


job_manager = JobManager(processor=process_document)