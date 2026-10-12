from __future__ import annotations

from pydantic import BaseModel


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    error: str | None
    latitude: float | None
    longitude: float | None
    actual_latitude: float | None
    actual_longitude: float | None
    inference_time_ms: int | None
