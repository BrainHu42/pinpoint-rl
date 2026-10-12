from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import ExifTags, Image

from pinpoint.server.inference_runtime import (
    ImageEmbeddingRuntime,
    load_geo_model,
    load_rgb_image_from_bytes,
    sha256_digest,
)
from pinpoint.server.config import ServerConfig


@dataclass(frozen=True)
class StoredUpload:
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    image_bytes: bytes
    actual_latitude: float | None
    actual_longitude: float | None


@dataclass(frozen=True)
class PredictionResult:
    latitude: float
    longitude: float


class PinpointInferenceService:
    def __init__(
        self,
        config: ServerConfig,
        *,
        runtime_dir: str | Path,
    ) -> None:
        self.config = config
        self.runtime_dir = Path(runtime_dir).expanduser().resolve()

        self.embedder = ImageEmbeddingRuntime(
            backbone_model_name=config.backbone_model_name,
            device=config.device,
            inference_dtype=config.inference_dtype,
            max_image_pixels=config.max_image_pixels,
        )
        self.geo_model = load_geo_model(
            config.model_spec,
            checkpoint_path=config.checkpoint_path,
            retrieval_db_source=config.retrieval_db_source,
            backbone_model_name=config.backbone_model_name,
            device=config.device,
            inference_dtype=config.inference_dtype,
            batch_size=config.batch_size,
            checkpoint_strict=config.checkpoint_strict,
        )
        self.backbone_model_name = self.embedder.backbone_model_name

    def warmup(self) -> None:
        image_bytes = self._warmup_image_bytes()
        embedding = self.embedder.embed_bytes(image_bytes)
        self.geo_model.predict(embedding)

    def validate_upload(
        self,
        *,
        filename: str,
        content_type: str,
        image_bytes: bytes,
    ) -> None:
        if content_type not in set(self.config.allowed_mime_types):
            allowed = ", ".join(self.config.allowed_mime_types)
            raise ValueError(f"Unsupported content type '{content_type}'. Allowed: {allowed}")
        if not filename:
            raise ValueError("Filename is required")
        if len(image_bytes) > self.config.max_upload_bytes:
            raise ValueError(
                f"Upload exceeds the maximum size of {self.config.max_upload_mb} MiB"
            )
        image = load_rgb_image_from_bytes(
            image_bytes,
            max_image_pixels=self.config.max_image_pixels,
        )
        image.close()

    def store_upload(
        self,
        *,
        job_id: str,
        filename: str,
        content_type: str,
        image_bytes: bytes,
    ) -> StoredUpload:
        self.validate_upload(
            filename=filename,
            content_type=content_type,
            image_bytes=image_bytes,
        )
        digest = sha256_digest(image_bytes)
        safe_filename = Path(filename).name or "upload"
        actual_latitude, actual_longitude = extract_gps_coordinates(image_bytes)
        return StoredUpload(
            filename=safe_filename,
            content_type=content_type,
            size_bytes=len(image_bytes),
            sha256=digest,
            image_bytes=bytes(image_bytes),
            actual_latitude=actual_latitude,
            actual_longitude=actual_longitude,
        )

    def predict_from_image_bytes(self, image_bytes: bytes, sha256: str) -> PredictionResult:
        embedding = self.embedder.embed_bytes(image_bytes)
        latitude, longitude = self.geo_model.predict(embedding)
        return PredictionResult(
            latitude=float(latitude),
            longitude=float(longitude),
        )

    @staticmethod
    def _warmup_image_bytes() -> bytes:
        with io.BytesIO() as buffer:
            image = Image.new("RGB", (32, 32), color=(140, 148, 156))
            image.save(buffer, format="PNG")
            return buffer.getvalue()


def extract_gps_coordinates(image_bytes: bytes) -> tuple[float | None, float | None]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        exif = image.getexif()
        if not exif:
            return None, None
        gps_ifd = exif.get_ifd(ExifTags.Base.GPSInfo)
        if not gps_ifd:
            return None, None

    latitude = _decimal_degrees_from_gps(
        gps_ifd.get(ExifTags.GPS.GPSLatitude),
        gps_ifd.get(ExifTags.GPS.GPSLatitudeRef),
    )
    longitude = _decimal_degrees_from_gps(
        gps_ifd.get(ExifTags.GPS.GPSLongitude),
        gps_ifd.get(ExifTags.GPS.GPSLongitudeRef),
    )
    return latitude, longitude


def _decimal_degrees_from_gps(
    values: tuple[object, object, object] | None,
    reference: object | None,
) -> float | None:
    if not values or reference is None or len(values) != 3:
        return None

    degrees = _coerce_gps_number(values[0])
    minutes = _coerce_gps_number(values[1])
    seconds = _coerce_gps_number(values[2])
    if degrees is None or minutes is None or seconds is None:
        return None

    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    reference_text = str(reference).strip().upper()
    if reference_text in {"S", "W"}:
        decimal *= -1.0
    return decimal


def _coerce_gps_number(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
