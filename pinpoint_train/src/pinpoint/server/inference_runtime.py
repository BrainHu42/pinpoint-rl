from __future__ import annotations

import hashlib
import io
import warnings
from contextlib import nullcontext
from typing import Any

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from benchmarks.eval import load_model_with_kwargs
from pinpoint.model_api import BaseGeoModel
from pinpoint.utils import (
    _resolve_device,
    _resolve_inference_dtype,
    _to_feature_tensor,
    resolve_backbone_model_name,
)


def load_rgb_image_from_bytes(
    image_bytes: bytes,
    *,
    max_image_pixels: int | None = None,
) -> Image.Image:
    if not image_bytes:
        raise ValueError("Uploaded image is empty")

    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            with Image.open(io.BytesIO(image_bytes)) as probe:
                probe.verify()
            with Image.open(io.BytesIO(image_bytes)) as decoded:
                width, height = decoded.size
                if width <= 0 or height <= 0:
                    raise ValueError("Image dimensions must be positive")
                if max_image_pixels is not None and width * height > max_image_pixels:
                    raise ValueError(
                        f"Image is too large: {width}x{height} exceeds {max_image_pixels} pixels"
                    )
                return decoded.convert("RGB")
        except Exception as exc:
            raise ValueError("Uploaded file is not a valid supported image") from exc


def sha256_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_geo_model(
    model_spec: str,
    *,
    checkpoint_path: str | None = None,
    retrieval_db_source: str | None = None,
    backbone_model_name: str | None = None,
    device: str | None = None,
    inference_dtype: str | None = None,
    batch_size: int | None = None,
    checkpoint_strict: bool | None = None,
) -> BaseGeoModel:
    model_kwargs: dict[str, Any] = {}
    if checkpoint_path:
        model_kwargs["checkpoint_path"] = checkpoint_path
    if retrieval_db_source:
        model_kwargs["retrieval_db_source"] = retrieval_db_source
    if backbone_model_name:
        model_kwargs["backbone_model_name"] = backbone_model_name
    if device:
        model_kwargs["device"] = device
    if inference_dtype:
        model_kwargs["inference_dtype"] = inference_dtype
    if batch_size is not None:
        model_kwargs["batch_size"] = int(batch_size)
    if checkpoint_strict is not None:
        model_kwargs["checkpoint_strict"] = bool(checkpoint_strict)
    return load_model_with_kwargs(model_spec, model_kwargs=model_kwargs)


class ImageEmbeddingRuntime:
    def __init__(
        self,
        *,
        backbone_model_name: str | None = None,
        device: str = "auto",
        inference_dtype: str = "auto",
        max_image_pixels: int | None = None,
    ) -> None:
        self.backbone_model_name = backbone_model_name or resolve_backbone_model_name()
        self.device = _resolve_device(device)
        self.inference_dtype = _resolve_inference_dtype(inference_dtype, self.device)
        self.max_image_pixels = max_image_pixels

        model_dtype = self.inference_dtype
        if self.device.type == "cpu" and model_dtype == torch.float16:
            model_dtype = torch.float32

        self.processor = AutoImageProcessor.from_pretrained(self.backbone_model_name, use_fast=True)
        self.backbone = AutoModel.from_pretrained(
            self.backbone_model_name,
            torch_dtype=model_dtype,
        ).to(self.device)
        self.backbone.eval()

    def embed_bytes(
        self,
        image_bytes: bytes,
    ) -> torch.Tensor:
        rgb_image = load_rgb_image_from_bytes(
            image_bytes,
            max_image_pixels=self.max_image_pixels,
        )
        try:
            model_inputs = self.processor(images=rgb_image, return_tensors="pt")
        finally:
            rgb_image.close()

        model_inputs = {
            name: value.to(
                self.device,
                non_blocking=(self.device.type == "cuda"),
            )
            for name, value in model_inputs.items()
        }

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.inference_dtype)
            if self.device.type == "cuda" and self.inference_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )

        with torch.inference_mode():
            with autocast_ctx:
                if hasattr(self.backbone, "get_image_features"):
                    raw = self.backbone.get_image_features(**model_inputs)
                else:
                    raw = self.backbone(**model_inputs)
                features = _to_feature_tensor(raw)

        embedding = features.squeeze(0).detach().cpu().to(torch.float32)
        if embedding.ndim != 1:
            raise ValueError(f"Expected 1D embedding, got shape {tuple(embedding.shape)}")
        return embedding
