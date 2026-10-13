from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

from pinpoint.server.service import PinpointInferenceService
from pinpoint.server.storage import JobStore


@dataclass(frozen=True)
class QueuedJob:
    job_id: str
    image_bytes: bytes
    sha256: str


class JobRunner:
    def __init__(
        self,
        *,
        store: JobStore,
        service: PinpointInferenceService,
    ) -> None:
        self.store = store
        self.service = service
        self._queue: queue.Queue[QueuedJob | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="pinpoint-job-runner", daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        self._started = False

    def submit(self, job: QueuedJob) -> None:
        self._queue.put(job)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                break
            try:
                self.store.mark_running(job.job_id)
                started_at = time.perf_counter()
                result = self.service.predict_from_image_bytes(job.image_bytes, job.sha256)
                inference_time_ms = max(0, round((time.perf_counter() - started_at) * 1000))
                self.store.mark_completed(
                    job.job_id,
                    latitude=result.latitude,
                    longitude=result.longitude,
                    inference_time_ms=inference_time_ms,
                )
            except Exception as exc:
                self.store.mark_failed(job.job_id, str(exc))
            finally:
                self._queue.task_done()
