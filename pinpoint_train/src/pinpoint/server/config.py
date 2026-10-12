from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "")
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(values)


def _load_env_file() -> None:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)
        return

    repo_dotenv = Path(__file__).resolve().parents[3] / ".env"
    if repo_dotenv.exists():
        load_dotenv(dotenv_path=repo_dotenv, override=False)


@dataclass(frozen=True)
class ServerConfig:
    bind_host: str
    port: int
    model_spec: str
    checkpoint_path: str | None
    retrieval_db_source: str | None
    backbone_model_name: str | None
    device: str
    inference_dtype: str
    batch_size: int
    checkpoint_strict: bool | None
    warmup_on_startup: bool
    max_upload_mb: int
    max_image_pixels: int
    allowed_mime_types: tuple[str, ...]
    secret_token: str | None

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> ServerConfig:
        _load_env_file()
        allowed_mime_types = _env_csv("PINPOINT_SERVER_ALLOWED_MIME_TYPES") or (
            "image/jpeg",
            "image/png",
            "image/webp",
        )

        return cls(
            bind_host=os.environ.get("PINPOINT_SERVER_BIND_HOST", "127.0.0.1").strip() or "127.0.0.1",
            port=int(os.environ.get("PINPOINT_SERVER_PORT", "8080")),
            model_spec=os.environ.get(
                "PINPOINT_SERVER_MODEL_SPEC",
                "pinpoint.attention_reranker.inference:build_model",
            ).strip(),
            checkpoint_path=os.environ.get("PINPOINT_SERVER_CHECKPOINT_PATH", "").strip() or None,
            retrieval_db_source=os.environ.get("PINPOINT_SERVER_RETRIEVAL_DB_SOURCE", "").strip() or None,
            backbone_model_name=os.environ.get("PINPOINT_SERVER_BACKBONE_MODEL_NAME", "").strip() or None,
            device=os.environ.get("PINPOINT_SERVER_DEVICE", "auto").strip() or "auto",
            inference_dtype=os.environ.get("PINPOINT_SERVER_INFERENCE_DTYPE", "auto").strip() or "auto",
            batch_size=int(os.environ.get("PINPOINT_SERVER_BATCH_SIZE", "1")),
            checkpoint_strict=None,
            warmup_on_startup=_env_bool("PINPOINT_SERVER_WARMUP_ON_STARTUP", True),
            max_upload_mb=int(os.environ.get("PINPOINT_SERVER_MAX_UPLOAD_MB", "20")),
            max_image_pixels=int(os.environ.get("PINPOINT_SERVER_MAX_IMAGE_PIXELS", "80000000")),
            allowed_mime_types=allowed_mime_types,
            secret_token=os.environ.get("PINPOINT_SERVER_SECRET_TOKEN", "").strip() or None,
        )
