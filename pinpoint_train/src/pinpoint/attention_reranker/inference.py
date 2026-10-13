"""Benchmark inference model for reranker checkpoints."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pinpoint.attention_reranker.lightning_module import AttentionInterpolateLightningModule
from pinpoint.attention_reranker.retrieval_index import _default_retrieval_cache_roots
from pinpoint.model_api import BaseGeoModel
from pinpoint.utils import (
    _resolve_default_checkpoint_path,
    _resolve_device,
    _resolve_inference_dtype,
    resolve_backbone_model_name,
    sanitize_latlon_deg_torch,
)


BACKBONE_MODEL_NAME = resolve_backbone_model_name()
_DEFAULT_MODEL: AttentionInterpolateGeoModel | None = None
_OSV_SUPPORT_CACHE_DIRNAME = "osv_support_cache"


def _local_exp_root(checkpoint_path: Path) -> Path:
    for parent in checkpoint_path.parents:
        if parent.name == "exp":
            return parent
    return Path.cwd().resolve() / "exp"


def _osv_support_files(cache_root: Path) -> tuple[dict[str, Any], Path, Path]:
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing OSV support manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    filenames = (files.get("embeddings"), files.get("latlon_deg"))
    if not all(filenames):
        raise ValueError(f"Invalid OSV support manifest: {manifest_path}")
    paths = tuple(cache_root / str(filename) for filename in filenames)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing OSV support file: {missing[0]}")
    return manifest, paths[0], paths[1]


def _bundle_osv_support_cache(
    source_root: Path,
    retrieval_index_dir: Path,
    geo_module: AttentionInterpolateLightningModule,
) -> Path:
    target_root = retrieval_index_dir / _OSV_SUPPORT_CACHE_DIRNAME
    if source_root == target_root:
        return target_root
    if target_root.exists():
        raise FileExistsError(f"Refusing to overwrite OSV support cache: {target_root}")

    manifest, embeddings_path, latlon_path = _osv_support_files(source_root)
    with tempfile.TemporaryDirectory(
        prefix=f".{_OSV_SUPPORT_CACHE_DIRNAME}.",
        dir=retrieval_index_dir,
    ) as temporary_dir:
        temporary_root = Path(temporary_dir)
        subprocess.run(
            ["cp", "--reflink=auto", "--", str(latlon_path), str(temporary_root / latlon_path.name)],
            check=True,
        )

        if str(manifest.get("embedding_space", "backbone")) == "contrastive_osv":
            target_embeddings_path = temporary_root / embeddings_path.name
            subprocess.run(
                ["cp", "--reflink=auto", "--", str(embeddings_path), str(target_embeddings_path)],
                check=True,
            )
        else:
            source_shape = tuple(int(value) for value in manifest["shapes"]["embeddings"])
            source_embeddings = np.memmap(
                embeddings_path,
                dtype=np.float16,
                mode="r",
                shape=source_shape,
            )
            target_embeddings_path = temporary_root / "embeddings.contrastive_osv.f16.bin"
            target_embeddings = np.memmap(
                target_embeddings_path,
                dtype=np.float16,
                mode="w+",
                shape=(source_shape[0], geo_module.output_dim),
            )
            print(f"Projecting OSV support cache into {geo_module.output_dim} dimensions...")
            try:
                for start in range(0, source_shape[0], 8192):
                    end = min(start + 8192, source_shape[0])
                    inputs = torch.from_numpy(
                        np.array(
                            source_embeddings[start:end],
                            dtype=np.float16,
                            copy=True,
                            order="C",
                        )
                    )
                    encoded = geo_module._encode_osv_support_images(inputs)
                    target_embeddings[start:end] = encoded.cpu().numpy().astype(np.float16)
            finally:
                target_embeddings.flush()
                del source_embeddings
                del target_embeddings
            manifest["embedding_space"] = "contrastive_osv"
            manifest["embedding_dim"] = int(geo_module.output_dim)
            manifest["shapes"]["embeddings"] = [source_shape[0], int(geo_module.output_dim)]
            manifest["files"]["embeddings"] = target_embeddings_path.name

        with (temporary_root / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_root.replace(target_root)
    return target_root


def _resolve_local_inference_assets(
    checkpoint_path: Path,
    backbone_model_name: str,
) -> tuple[Path, Path, Path]:
    exp_root = _local_exp_root(checkpoint_path)
    contrastive_checkpoint_path = (
        exp_root / "contrastive_retrieval" / "checkpoints" / "ckpt_best.pt"
    ).resolve()
    if not contrastive_checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing local contrastive checkpoint: {contrastive_checkpoint_path}")

    index_root = contrastive_checkpoint_path.parent / "attention_retrieval_index"
    manifests = sorted(index_root.glob("*/manifest.json"))
    if len(manifests) > 1:
        choices = "\n".join(f"  - {manifest.parent}" for manifest in manifests)
        raise RuntimeError(
            f"Multiple local retrieval indexes found under {index_root}:\n{choices}"
        )

    if manifests:
        retrieval_index_dir = manifests[0].parent.resolve()
        osv_support_cache_root = retrieval_index_dir / _OSV_SUPPORT_CACHE_DIRNAME
        _osv_support_files(osv_support_cache_root)
        return contrastive_checkpoint_path, retrieval_index_dir, osv_support_cache_root

    retrieval_index_dir = index_root.resolve()
    osv_support_cache_root = Path(
        _default_retrieval_cache_roots(
            backbone_model_name,
            retrieval_db_source="osv5m",
        )[0]
    ).expanduser().resolve()
    _osv_support_files(osv_support_cache_root)
    return contrastive_checkpoint_path, retrieval_index_dir, osv_support_cache_root


class AttentionInterpolateGeoModel(BaseGeoModel):
    """Inference model for reranker checkpoints from `pinpoint.attention_reranker.train`."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        retrieval_db_source: str | None = None,
        backbone_model_name: str | None = None,
        device: str | None = None,
        inference_dtype: str | None = None,
        batch_size: int | None = None,
        checkpoint_strict: bool = True,
    ):
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        checkpoint_candidate = (
            Path(checkpoint_path).expanduser()
            if checkpoint_path
            else _resolve_default_checkpoint_path(__file__)
        )
        if checkpoint_candidate is None or not checkpoint_candidate.exists():
            raise FileNotFoundError("No checkpoint found. Pass checkpoint_path.")
        self.checkpoint_path = checkpoint_candidate.resolve()

        self.backbone_model_name = backbone_model_name or BACKBONE_MODEL_NAME
        self.device = _resolve_device(device or "auto")
        self.inference_dtype = _resolve_inference_dtype(inference_dtype or "auto", self.device)
        self.batch_size = int(batch_size) if batch_size is not None else 1

        contrastive_path, retrieval_path, osv_support_path = _resolve_local_inference_assets(
            self.checkpoint_path,
            self.backbone_model_name,
        )
        load_overrides: dict[str, Any] = {
            "contrastive_checkpoint_path": str(contrastive_path),
            "retrieval_index_dir": str(retrieval_path),
            "osv_support_cache_root": str(osv_support_path),
        }
        if retrieval_db_source is not None:
            load_overrides["retrieval_db_source"] = str(retrieval_db_source)

        self.geo_module = AttentionInterpolateLightningModule.load_from_checkpoint(
            str(self.checkpoint_path),
            map_location="cpu",
            strict=checkpoint_strict,
            **load_overrides,
        ).to(self.device)
        self.geo_module.eval()

        self.osv_support_cache_root = _bundle_osv_support_cache(
            osv_support_path,
            self.geo_module.retrieval_index.index_dir,
            self.geo_module,
        )

        self.embed_dim = int(getattr(self.geo_module, "embed_dim", 0))
        self.image_embed_dim = int(getattr(self.geo_module, "image_embed_dim", 0))
        self.contrastive_embed_dim = int(getattr(self.geo_module, "contrastive_embed_dim", 0))

    def _prepare_embeddings(self, image_embedding: Any) -> torch.Tensor:
        if torch.is_tensor(image_embedding):
            embedding = image_embedding.detach()
        elif isinstance(image_embedding, np.ndarray):
            embedding = torch.from_numpy(image_embedding)
        else:
            embedding = torch.as_tensor(image_embedding)

        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        elif embedding.ndim != 2:
            raise ValueError(
                f"predict expects embeddings of shape [D] or [B, D], got {tuple(embedding.shape)}"
            )

        dim = int(embedding.shape[1])
        if dim not in {self.image_embed_dim, self.contrastive_embed_dim}:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.image_embed_dim} or {self.contrastive_embed_dim}, got {dim}"
            )

        if embedding.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            embedding = embedding.to(torch.float32)
        embedding = torch.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

        return embedding.to(
            self.device,
            non_blocking=(self.device.type == "cuda"),
        )

    def _prepare_embedding(self, image_embedding: Any) -> torch.Tensor:
        embedding = self._prepare_embeddings(image_embedding)
        if embedding.ndim != 2 or embedding.shape[0] != 1:
            raise ValueError(
                f"predict expects one embedding of shape [D] or [1, D], got {tuple(embedding.shape)}"
            )
        return embedding

    def predict(self, image_embedding: Any) -> tuple[float, float]:
        embedding = self._prepare_embedding(image_embedding)
        with torch.inference_mode():
            outputs = self.geo_module(
                embedding,
                query_indices=None,
                apply_leave_one_out=False,
            )
            top_pos = torch.argmax(outputs["rerank_scores"], dim=-1, keepdim=True)
            latlon_deg = self.geo_module._gather_candidates(outputs["candidate_latlon_deg"], top_pos)
            latlon_deg = sanitize_latlon_deg_torch(latlon_deg.to(torch.float32))

        pred = latlon_deg[0, 0].detach().cpu()
        return (float(pred[0].item()), float(pred[1].item()))


def build_model(**kwargs: Any) -> AttentionInterpolateGeoModel:
    """Factory for benchmarks.load_model()."""
    return AttentionInterpolateGeoModel(**kwargs)


def predict(image_embedding: Any) -> tuple[float, float]:
    """Module-level BaseGeoModel-compatible predict helper."""
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = build_model()
    return _DEFAULT_MODEL.predict(image_embedding)
