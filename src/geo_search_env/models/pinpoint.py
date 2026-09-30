"""Frozen Pinpoint retrieval baseline and query-image embedding adapter."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any

from ..core.contracts import Coordinate, InitialCandidate


# Checkout of the Pinpoint submission (contrastive checkpoint + retrieval index); override with PINPOINT_ROOT.
DEFAULT_PINPOINT_ROOT = Path(os.environ.get("PINPOINT_ROOT", "/home/brian/workspace/pinpoint-submission/submission"))
DEFAULT_PINPOINT_CHECKPOINT = (
    DEFAULT_PINPOINT_ROOT / "exp/contrastive_retrieval/checkpoints/ckpt_best.pt"
)
DEFAULT_PINPOINT_INDEX_ROOT = (
    DEFAULT_PINPOINT_ROOT / "exp/contrastive_retrieval/checkpoints/retrieval_index"
)
DEFAULT_BACKBONE = "google/siglip2-giant-opt-patch16-384"


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    rank: int
    coordinate: Coordinate
    score: float
    retrieval_index: int

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("retrieval rank must be a positive integer")
        if not isinstance(self.coordinate, Coordinate):
            raise ValueError("retrieval candidate must contain a coordinate")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)) or not math.isfinite(self.score):
            raise ValueError("retrieval score must be finite")
        if type(self.retrieval_index) is not int or self.retrieval_index < 0:
            raise ValueError("retrieval index must be a nonnegative integer")
        object.__setattr__(self, "score", float(self.score))


def _optional_retrieval_imports():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Pinpoint retrieval requires installation with the 'retrieval' extra") from error
    return torch


def _discover_index(index_root: Path, checkpoint_path: Path, source: str) -> Path:
    candidates = []
    for manifest_path in sorted(index_root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        manifest_checkpoint = manifest.get("checkpoint_path")
        if (
            manifest.get("version") == 3
            and manifest.get("retrieval_db_source") == source
            and isinstance(manifest_checkpoint, str)
            and Path(manifest_checkpoint).resolve() == checkpoint_path
        ):
            candidates.append(manifest_path.parent)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one version-3 {source} retrieval index under {index_root}, found {len(candidates)}"
        )
    return candidates[0]


class PinpointRetrievalBaseline:
    """Exact, frozen retrieval-only baseline ported from Pinpoint submission."""

    def __init__(
        self,
        checkpoint_path: Path | str = DEFAULT_PINPOINT_CHECKPOINT,
        index_dir: Path | str | None = None,
        *,
        retrieval_source: str = "mp16",
        device: str = "auto",
        inference_dtype: str = "auto",
        chunk_size: int = 131_072,
    ) -> None:
        _optional_retrieval_imports()
        from ._pinpoint_torch import TorchRetrievalRuntime

        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"missing Pinpoint checkpoint: {self.checkpoint_path}")
        if not isinstance(retrieval_source, str) or not retrieval_source.strip():
            raise ValueError("retrieval_source must be a nonempty string")
        self.retrieval_source = retrieval_source.strip().lower()
        resolved_index = (
            Path(index_dir).resolve()
            if index_dir is not None
            else _discover_index(DEFAULT_PINPOINT_INDEX_ROOT, self.checkpoint_path, self.retrieval_source)
        )
        self.index_dir = resolved_index
        self._runtime = TorchRetrievalRuntime(
            self.checkpoint_path,
            resolved_index,
            source=self.retrieval_source,
            device=device,
            inference_dtype=inference_dtype,
            chunk_size=chunk_size,
        )
        checkpoint_stat = self.checkpoint_path.stat()
        fingerprint = {
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_size": checkpoint_stat.st_size,
            "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
            "index": str(self.index_dir),
            "source": self.retrieval_source,
        }
        self.version = "pinpoint-contrastive-retrieval-v1:" + hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True).encode()
        ).hexdigest()[:16]

    def predict_candidates(self, image_embedding: Any, top_k: int) -> tuple[RetrievalCandidate, ...]:
        rows = self._runtime.predict_candidates(image_embedding, top_k)
        return tuple(
            RetrievalCandidate(rank, Coordinate(latitude, longitude), score, index)
            for rank, (index, latitude, longitude, score) in enumerate(rows, start=1)
        )

    def predict(self, image_embedding: Any) -> Coordinate:
        return self.predict_candidates(image_embedding, 1)[0].coordinate

    def project_image_embeddings(self, image_embeddings: Any, *, source: str):
        """Return normalized outputs from a frozen checkpoint image adapter/tower."""

        return self._runtime.encode_embeddings(image_embeddings, source=source)

    def initial_candidates(self, image_embedding: Any, top_k: int) -> tuple[InitialCandidate, ...]:
        return tuple(
            InitialCandidate(
                f"retrieval:{candidate.rank}",
                candidate.coordinate,
                candidate.rank,
                (max(-1.0, min(1.0, candidate.score)) + 1.0) / 2.0,
            )
            for candidate in self.predict_candidates(image_embedding, top_k)
        )


class PinpointImageEmbedder:
    """The submission's frozen SigLIP2 preprocessing and image encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_BACKBONE,
        *,
        device: str = "auto",
        inference_dtype: str = "auto",
        max_image_pixels: int | None = None,
    ) -> None:
        torch = _optional_retrieval_imports()
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:
            raise RuntimeError("Pinpoint image embedding requires installation with the 'retrieval' extra") from error
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        dtype_names = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
        dtype_name = "float16" if inference_dtype == "auto" and self.device.type == "cuda" else inference_dtype
        dtype_name = "float32" if dtype_name == "auto" else dtype_name
        if dtype_name not in dtype_names:
            raise ValueError("inference_dtype must be auto, float16, float32, or bfloat16")
        model_dtype = dtype_names[dtype_name]
        if self.device.type == "cpu" and model_dtype is torch.float16:
            model_dtype = torch.float32
        self._torch = torch
        self.model_dtype = model_dtype
        self.model_name = model_name
        self.max_image_pixels = max_image_pixels
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=model_dtype).to(self.device)
        self.model.eval()

    @staticmethod
    def _feature_tensor(output: Any):
        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state[:, 0]
        if isinstance(output, dict):
            for name in ("image_embeds", "pooler_output", "last_hidden_state"):
                value = output.get(name)
                if value is not None:
                    return value if name != "last_hidden_state" else value[:, 0]
        return output

    def embed_bytes(self, image_bytes: bytes):
        if not image_bytes:
            raise ValueError("query image is empty")
        try:
            from PIL import Image
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
                width, height = image.size
                if self.max_image_pixels is not None and width * height > self.max_image_pixels:
                    raise ValueError("query image exceeds the configured pixel limit")
                rgb_image = image.convert("RGB")
                try:
                    inputs = self.processor(images=rgb_image, return_tensors="pt")
                finally:
                    rgb_image.close()
        except ValueError:
            raise
        except Exception as error:
            raise ValueError("query is not a valid supported image") from error
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        autocast = (
            self._torch.autocast(device_type="cuda", dtype=self.model_dtype)
            if self.device.type == "cuda"
            and self.model_dtype in {self._torch.float16, self._torch.bfloat16}
            else nullcontext()
        )
        with self._torch.inference_mode(), autocast:
            output = self.model.get_image_features(**inputs) if hasattr(
                self.model, "get_image_features"
            ) else self.model(**inputs)
            embedding = self._feature_tensor(output).squeeze(0).detach().cpu().float()
        if embedding.ndim != 1:
            raise ValueError("SigLIP image encoder did not return one feature vector")
        return embedding

    def embed_path(self, path: Path | str):
        return self.embed_bytes(Path(path).read_bytes())
