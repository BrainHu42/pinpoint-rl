"""Torch implementation details for the frozen Pinpoint retrieval baseline."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualMLPBlock(nn.Module):
    def __init__(self, dimension: int, expansion: int) -> None:
        super().__init__()
        hidden = dimension * expansion
        self.norm = nn.LayerNorm(dimension)
        self.fc1 = nn.Linear(dimension, hidden)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(0.0)
        self.fc2 = nn.Linear(hidden, dimension)
        self.dropout2 = nn.Dropout(0.0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.norm(value)
        residual = self.fc1(residual)
        residual = self.act(residual)
        residual = self.dropout1(residual)
        residual = self.fc2(residual)
        residual = self.dropout2(residual)
        return value + residual


class _ImageTower(nn.Module):
    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        adapter_depth: int,
        adapter_expansion: int,
        trunk_depth: int,
        trunk_expansion: int,
        sources: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.adapters = nn.ModuleDict(
            {
                source: nn.Sequential(
                    nn.LayerNorm(input_dimension),
                    nn.Linear(input_dimension, hidden_dimension),
                    nn.GELU(),
                    nn.Dropout(0.0),
                    *(
                        _ResidualMLPBlock(hidden_dimension, adapter_expansion)
                        for _ in range(adapter_depth - 1)
                    ),
                )
                for source in sources
            }
        )
        self.blocks = nn.ModuleList(
            _ResidualMLPBlock(hidden_dimension, trunk_expansion)
            for _ in range(trunk_depth)
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dimension),
            nn.Linear(hidden_dimension, output_dimension),
        )

    def forward(self, value: torch.Tensor, source: str) -> torch.Tensor:
        if source not in self.adapters:
            raise ValueError(f"checkpoint has no image adapter for source {source!r}")
        value = self.adapters[source](value)
        for block in self.blocks:
            value = block(value)
        return self.out_proj(value)


def _numbered_depth(state: dict[str, torch.Tensor], prefix: str) -> int:
    numbers = {
        int(key[len(prefix) :].split(".", 1)[0])
        for key in state
        if key.startswith(prefix) and key[len(prefix) :].split(".", 1)[0].isdigit()
    }
    return max(numbers) + 1 if numbers else 0


def _residual_depth(state: dict[str, torch.Tensor], prefix: str) -> int:
    return sum(
        key.startswith(prefix) and key.endswith(".norm.weight")
        for key in state
    )


def _load_image_tower(checkpoint_path: Path) -> tuple[_ImageTower, int, int, tuple[str, ...]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    raw_state = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(raw_state, dict):
        raise ValueError("Pinpoint checkpoint has no state_dict")
    state = {
        key.removeprefix("image_tower."): value
        for key, value in raw_state.items()
        if key.startswith("image_tower.")
    }
    sources = tuple(
        sorted(
            {
                key.split(".", 2)[1]
                for key in state
                if key.startswith("adapters.") and len(key.split(".", 2)) == 3
            }
        )
    )
    if not sources:
        raise ValueError("Pinpoint checkpoint contains no image adapters")
    first = sources[0]
    input_weight = state.get(f"adapters.{first}.1.weight")
    output_weight = state.get("out_proj.1.weight")
    if input_weight is None or output_weight is None:
        raise ValueError("Pinpoint checkpoint image tower is incomplete")
    hidden_dimension, input_dimension = map(int, input_weight.shape)
    output_dimension = int(output_weight.shape[0])
    adapter_depth = 1 + _residual_depth(state, f"adapters.{first}.")
    trunk_depth = _numbered_depth(state, "blocks.")
    adapter_fc = state.get(f"adapters.{first}.4.fc1.weight")
    trunk_fc = state.get("blocks.0.fc1.weight")
    adapter_expansion = int(adapter_fc.shape[0] // hidden_dimension) if adapter_fc is not None else 1
    trunk_expansion = int(trunk_fc.shape[0] // hidden_dimension) if trunk_fc is not None else 1
    if adapter_depth < 1 or trunk_depth < 1:
        raise ValueError("Pinpoint checkpoint image tower has invalid depth")
    tower = _ImageTower(
        input_dimension=input_dimension,
        hidden_dimension=hidden_dimension,
        output_dimension=output_dimension,
        adapter_depth=adapter_depth,
        adapter_expansion=adapter_expansion,
        trunk_depth=trunk_depth,
        trunk_expansion=trunk_expansion,
        sources=sources,
    )
    tower.load_state_dict(state, strict=True)
    tower.eval()
    return tower, input_dimension, output_dimension, sources


class TorchRetrievalRuntime:
    """Inference-only port of Pinpoint's exact contrastive retriever."""

    def __init__(
        self,
        checkpoint_path: Path,
        index_dir: Path,
        *,
        source: str,
        device: str,
        inference_dtype: str,
        chunk_size: int,
    ) -> None:
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("retrieval device must be cpu, cuda, or auto")
        dtype_name = "float16" if inference_dtype == "auto" and self.device.type == "cuda" else inference_dtype
        dtype_name = "float32" if dtype_name == "auto" else dtype_name
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        if dtype_name not in dtype_map:
            raise ValueError("inference_dtype must be auto, float16, float32, or bfloat16")
        self.inference_dtype = dtype_map[dtype_name]
        if self.device.type == "cpu" and self.inference_dtype is torch.float16:
            raise ValueError("float16 model inference is unsupported on CPU")

        self.tower, self.input_dimension, self.output_dimension, self.sources = (
            _load_image_tower(checkpoint_path)
        )
        if source not in self.sources:
            raise ValueError(f"checkpoint does not support retrieval source {source!r}")
        self.source = source
        self.tower.to(self.device)
        self.chunk_size = chunk_size

        manifest_path = index_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid retrieval index manifest: {manifest_path}") from error
        if manifest.get("version") != 3:
            raise ValueError("unsupported Pinpoint retrieval index version")
        manifest_checkpoint = manifest.get("checkpoint_path")
        if not isinstance(manifest_checkpoint, str) or Path(manifest_checkpoint).resolve() != checkpoint_path:
            raise ValueError("retrieval index was built for a different checkpoint")
        if manifest.get("retrieval_db_source") != source:
            raise ValueError("retrieval index source does not match the requested image tower")
        self.num_locations = int(manifest.get("num_samples", 0))
        if self.num_locations < 1 or int(manifest.get("embedding_dim", 0)) != self.output_dimension:
            raise ValueError("retrieval index dimensions do not match the checkpoint")
        files = manifest.get("files", {})
        shapes = manifest.get("shapes", {})
        if shapes.get("gps_embeddings") != [self.num_locations, self.output_dimension]:
            raise ValueError("retrieval GPS embedding shape is invalid")
        if shapes.get("latlon_deg") != [self.num_locations, 2]:
            raise ValueError("retrieval coordinate shape is invalid")
        self.embeddings_path = (index_dir / str(files.get("gps_embeddings", ""))).resolve()
        self.coordinates_path = (index_dir / str(files.get("latlon_deg", ""))).resolve()
        try:
            self.embeddings_path.relative_to(index_dir)
            self.coordinates_path.relative_to(index_dir)
        except ValueError as error:
            raise ValueError("retrieval index file escapes its index directory") from error
        if not self.embeddings_path.is_file() or not self.coordinates_path.is_file():
            raise FileNotFoundError("retrieval index binary files are missing")
        expected_embeddings = self.num_locations * self.output_dimension * np.dtype("float16").itemsize
        expected_coordinates = self.num_locations * 2 * np.dtype("float32").itemsize
        if self.embeddings_path.stat().st_size != expected_embeddings:
            raise ValueError("retrieval GPS embedding file size is invalid")
        if self.coordinates_path.stat().st_size != expected_coordinates:
            raise ValueError("retrieval coordinate file size is invalid")
        self._embeddings = np.memmap(
            self.embeddings_path,
            dtype=np.float16,
            mode="r",
            shape=(self.num_locations, self.output_dimension),
        )
        self._coordinates = np.memmap(
            self.coordinates_path,
            dtype=np.float32,
            mode="r",
            shape=(self.num_locations, 2),
        )

    def encode_embeddings(self, image_embeddings: Any, *, source: str) -> np.ndarray:
        """Project one or more backbone features through a frozen source adapter."""

        if source not in self.sources:
            raise ValueError(f"checkpoint does not support image source {source!r}")
        value = torch.as_tensor(image_embeddings)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[1] != self.input_dimension:
            raise ValueError(
                f"image embeddings must have shape [{self.input_dimension}] or [N, {self.input_dimension}]"
            )
        value = torch.nan_to_num(value.to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
        autocast = (
            torch.autocast(device_type="cuda", dtype=self.inference_dtype)
            if self.device.type == "cuda" and self.inference_dtype in {torch.float16, torch.bfloat16}
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            encoded = self.tower(value.to(torch.float32), source)
            encoded = F.normalize(encoded, dim=-1, eps=1e-8)
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0)
        return encoded.to(torch.float32).cpu().numpy()

    def _encode(self, image_embedding: Any) -> torch.Tensor:
        encoded = self.encode_embeddings(image_embedding, source=self.source)
        return torch.from_numpy(encoded[0]).to(self.device)

    def predict_candidates(self, image_embedding: Any, top_k: int) -> list[tuple[int, float, float, float]]:
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        top_k = min(top_k, self.num_locations)
        query = self._encode(image_embedding).to(self.device, dtype=torch.float16)
        best_scores: torch.Tensor | None = None
        best_indices: torch.Tensor | None = None
        with torch.inference_mode():
            for start in range(0, self.num_locations, self.chunk_size):
                end = min(start + self.chunk_size, self.num_locations)
                block = torch.from_numpy(
                    np.array(self._embeddings[start:end], dtype=np.float16, copy=True, order="C")
                ).to(self.device, non_blocking=self.device.type == "cuda")
                scores = torch.matmul(block, query)
                local_k = min(top_k, end - start)
                local_scores, local_indices = torch.topk(scores, local_k, sorted=True)
                local_indices = local_indices + start
                if best_scores is not None and best_indices is not None:
                    local_scores = torch.cat((best_scores, local_scores))
                    local_indices = torch.cat((best_indices, local_indices))
                    keep = torch.topk(local_scores, top_k, sorted=True).indices
                    local_scores = local_scores[keep]
                    local_indices = local_indices[keep]
                best_scores, best_indices = local_scores, local_indices
        assert best_scores is not None and best_indices is not None
        result = []
        for index, score in zip(best_indices.cpu().tolist(), best_scores.float().cpu().tolist()):
            latitude = float(np.clip(self._coordinates[index, 0], -90.0, 90.0))
            longitude = float(((float(self._coordinates[index, 1]) + 180.0) % 360.0) - 180.0)
            result.append((int(index), latitude, longitude, float(score)))
        return result
