from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    status: str
    created_at: str
    updated_at: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    uploader_login: str | None
    uploader_name: str | None
    error: str | None
    latitude: float | None
    longitude: float | None
    actual_latitude: float | None
    actual_longitude: float | None
    inference_time_ms: int | None
    model_spec: str
    backbone_model_name: str

    def to_public_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "error": self.error,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "actual_latitude": self.actual_latitude,
            "actual_longitude": self.actual_longitude,
            "inference_time_ms": self.inference_time_ms,
        }


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}

    def create_job(
        self,
        *,
        job_id: str,
        filename: str,
        content_type: str,
        size_bytes: int,
        sha256: str,
        uploader_login: str | None,
        uploader_name: str | None,
        model_spec: str,
        backbone_model_name: str,
        actual_latitude: float | None,
        actual_longitude: float | None,
    ) -> None:
        timestamp = utc_now_iso()
        with self._lock:
            self._jobs[job_id] = JobRecord(
                job_id=job_id,
                status="queued",
                created_at=timestamp,
                updated_at=timestamp,
                filename=filename,
                content_type=content_type,
                size_bytes=int(size_bytes),
                sha256=sha256,
                uploader_login=uploader_login,
                uploader_name=uploader_name,
                error=None,
                latitude=None,
                longitude=None,
                actual_latitude=actual_latitude,
                actual_longitude=actual_longitude,
                inference_time_ms=None,
                model_spec=model_spec,
                backbone_model_name=backbone_model_name,
            )

    def mark_running(self, job_id: str) -> None:
        self._update_status(job_id, status="running")

    def mark_failed(self, job_id: str, error: str) -> None:
        self._update_status(job_id, status="failed", error=error)

    def mark_completed(
        self,
        job_id: str,
        *,
        latitude: float,
        longitude: float,
        inference_time_ms: int,
    ) -> None:
        self._update_record(
            job_id,
            status="completed",
            error=None,
            latitude=float(latitude),
            longitude=float(longitude),
            inference_time_ms=int(inference_time_ms),
        )

    def _update_status(
        self,
        job_id: str,
        *,
        status: str,
        error: str | None = None,
    ) -> None:
        self._update_record(job_id, status=status, error=error)

    def _update_record(self, job_id: str, **updates: object) -> None:
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return
            values = current.__dict__.copy()
            values.update(updates)
            values["updated_at"] = utc_now_iso()
            self._jobs[job_id] = JobRecord(**values)

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)
