from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import secrets
import tempfile
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse

from pinpoint.server.auth import AuthenticatedRequester, authenticate_request
from pinpoint.server.config import ServerConfig
from pinpoint.server.jobs import JobRunner, QueuedJob
from pinpoint.server.schemas import JobCreateResponse, JobStatusResponse
from pinpoint.server.service import PinpointInferenceService
from pinpoint.server.storage import JobStore


@lru_cache(maxsize=1)
def _load_index_html() -> str:
    static_path = Path(__file__).with_name("static") / "index.html"
    return static_path.read_text(encoding="utf-8")


def _resolved_startup_config(config: ServerConfig) -> ServerConfig:
    if config.secret_token:
        return config
    return replace(config, secret_token=secrets.token_hex(16))


def _print_startup_summary(config: ServerConfig) -> None:
    print("Pinpoint demo server")
    print(f"Local URL: http://{config.bind_host}:{config.port}")
    print(f"Secret token: {config.secret_token}")
    print("State: ephemeral (submitted images and job state stay in memory only)")
    print("")
    print("To expose this briefly with Tailscale Funnel, run in another terminal:")
    print(f"tailscale funnel --bg {config.port}")
    print("")
    print("Then share the Funnel URL and the secret token with the demo participant.")
    print(flush=True)


def create_app(config: ServerConfig | None = None) -> FastAPI:
    resolved_config = config or ServerConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with tempfile.TemporaryDirectory(prefix="pinpoint-upload-") as runtime_dir:
            service = PinpointInferenceService(resolved_config, runtime_dir=runtime_dir)
            if resolved_config.warmup_on_startup:
                print("Warming SigLIP embedder and geolocation model...", flush=True)
                service.warmup()
                print("Warm-up complete.", flush=True)
            store = JobStore()
            runner = JobRunner(store=store, service=service)
            runner.start()
            app.state.config = resolved_config
            app.state.service = service
            app.state.store = store
            app.state.runner = runner
            app.state.runtime_dir = runtime_dir
            try:
                yield
            finally:
                runner.stop()

    app = FastAPI(
        title="Pinpoint Demo Server",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.config = resolved_config

    def get_config(request: Request) -> ServerConfig:
        return request.app.state.config

    def require_requester(
        request: Request,
        app_config: ServerConfig = Depends(get_config),
    ) -> AuthenticatedRequester:
        return authenticate_request(request, app_config)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _load_index_html()

    @app.post("/api/jobs", response_model=JobCreateResponse, status_code=status.HTTP_202_ACCEPTED)
    async def create_job(
        request: Request,
        image: UploadFile = File(...),
        requester: AuthenticatedRequester = Depends(require_requester),
    ) -> JobCreateResponse:
        app_config: ServerConfig = app.state.config
        if image.content_type not in set(app_config.allowed_mime_types):
            allowed = ", ".join(app_config.allowed_mime_types)
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported content type '{image.content_type}'. Allowed: {allowed}",
            )

        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > app_config.max_upload_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Upload exceeds the maximum size of {app_config.max_upload_mb} MiB",
                    )
            except ValueError:
                pass

        image_bytes = await image.read()
        if len(image_bytes) > app_config.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Upload exceeds the maximum size of {app_config.max_upload_mb} MiB",
            )

        try:
            job_id = uuid4().hex
            stored = app.state.service.store_upload(
                job_id=job_id,
                filename=image.filename or "upload",
                content_type=image.content_type or "application/octet-stream",
                image_bytes=image_bytes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        finally:
            await image.close()

        try:
            app.state.store.create_job(
                job_id=job_id,
                filename=stored.filename,
                content_type=stored.content_type,
                size_bytes=stored.size_bytes,
                sha256=stored.sha256,
                uploader_login=requester.login,
                uploader_name=requester.name,
                model_spec=app.state.config.model_spec,
                backbone_model_name=app.state.service.backbone_model_name,
                actual_latitude=stored.actual_latitude,
                actual_longitude=stored.actual_longitude,
            )
            app.state.runner.submit(
                QueuedJob(
                    job_id=job_id,
                    image_bytes=stored.image_bytes,
                    sha256=stored.sha256,
                )
            )
        except Exception:
            raise
        return JobCreateResponse(job_id=job_id, status="queued")

    @app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job(
        job_id: str,
        _: AuthenticatedRequester = Depends(require_requester),
    ) -> JobStatusResponse:
        record = app.state.store.get_job(job_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        return JobStatusResponse(**record.to_public_dict())

    return app


def main() -> None:
    config = _resolved_startup_config(ServerConfig.from_env())
    _print_startup_summary(config)
    uvicorn.run(
        create_app(config),
        host=config.bind_host,
        port=config.port,
    )


app = create_app()
