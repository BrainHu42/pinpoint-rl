"""Inference-time retrieval model built from contrastive retrieval checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from pinpoint.contrastive_retrieval.lightning_module import GeoLightningModule
from pinpoint.model_api import BaseGeoModel
from pinpoint.utils import (
    _resolve_default_checkpoint_path,
    _resolve_device,
    _resolve_inference_dtype,
    model_name_tail,
    resolve_backbone_model_name,
)


BACKBONE_MODEL_NAME = resolve_backbone_model_name()
_INDEX_MANIFEST_VERSION = 3
_INDEX_EMBED_FILENAME = "gps_embeddings.bin"
_INDEX_LATLON_FILENAME = "gps_latlon_deg.bin"
_INDEX_MANIFEST_FILENAME = "manifest.json"
_SUPPORTED_HALF_DTYPES = (torch.float16, torch.bfloat16)


def _normalize_retrieval_db_source(raw_source: str | None) -> str:
    source_name = str(
        raw_source or os.environ.get("PINPOINT_CONTRASTIVE_RETRIEVAL_DB_SOURCE", "mp16")
    ).strip().lower()
    if source_name not in {"mp16", "osv5m"}:
        raise ValueError(f"Unsupported retrieval_db_source '{source_name}'. Expected one of: mp16, osv5m")
    return source_name


def _default_retrieval_cache_roots(backbone_model_name: str, *, retrieval_db_source: str) -> tuple[str, ...]:
    """Resolve the default cache roots used to build the retrieval index."""
    backbone_tail = model_name_tail(backbone_model_name)
    data_root = os.environ.get("PINPOINT_DATA_ROOT", "/data/pinpoint")
    if retrieval_db_source == "mp16":
        mp16_embed_root = os.environ.get("PINPOINT_MP16_EMBED_ROOT", f"{data_root}/mp16-embed")
        mp16_cache_root = os.environ.get(
            "MP16_CACHE_ROOT",
            os.environ.get("PINPOINT_MP16_CACHE_ROOT", f"{mp16_embed_root}/{backbone_tail}"),
        )
        return (mp16_cache_root,)
    osv5m_embed_root = os.environ.get("PINPOINT_OSV5M_EMBED_ROOT", f"{data_root}/osv5m-embed")
    osv5m_cache_root = os.environ.get(
        "OSV5M_CACHE_ROOT",
        os.environ.get("PINPOINT_OSV5M_CACHE_ROOT", f"{osv5m_embed_root}/{backbone_tail}"),
    )
    return (osv5m_cache_root,)


class LinearGeoModel(BaseGeoModel):
    """
    Inference model for checkpoints produced by `pinpoint.contrastive_retrieval.train`.

    This model performs retrieval over precomputed location embeddings:
    image embedding -> source-matched image tower -> image/GPS similarity -> (lat, lon).
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        backbone_model_name: str | None = None,
        device: str | None = None,
        inference_dtype: str | None = None,
        batch_size: int | None = None,
        checkpoint_strict: bool = True,
        cache_roots: str | Path | Sequence[str | Path] | None = None,
        index_cache_dir: str | Path | None = None,
        build_batch_size: int = 8192,
        retrieval_db_source: str | None = None,
    ):
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if build_batch_size <= 0:
            raise ValueError("build_batch_size must be > 0")

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
        # Retained for API compatibility with evaluators that pass model-level batching knobs.
        self.batch_size = int(batch_size) if batch_size is not None else 1
        self.build_batch_size = int(build_batch_size)

        self.geo_module = GeoLightningModule.load_from_checkpoint(
            str(self.checkpoint_path),
            map_location="cpu",
            strict=checkpoint_strict,
        ).to(self.device)
        self.geo_module.eval()
        self.embed_dim = int(getattr(self.geo_module, "embed_dim", 0))
        self.image_embed_dim = int(getattr(self.geo_module, "image_embed_dim", self.embed_dim))
        self.output_dim = int(getattr(self.geo_module, "output_dim", 0))
        self.retrieval_db_source = _normalize_retrieval_db_source(retrieval_db_source)
        if self.output_dim <= 0:
            raise RuntimeError("Checkpoint did not expose output_dim for contrastive retrieval.")
        if self.image_embed_dim <= 0:
            raise RuntimeError("Checkpoint did not expose a valid image_embed_dim for inference.")

        self.cache_roots = self._resolve_cache_roots(
            cache_roots,
            self.backbone_model_name,
            retrieval_db_source=self.retrieval_db_source,
        )
        self._source_specs = self._load_latlon_sources(self.cache_roots)
        self.num_locations = int(sum(spec["num_samples"] for spec in self._source_specs))
        if self.num_locations <= 0:
            raise RuntimeError("No location samples found in cache roots.")

        base_index_dir = (
            Path(index_cache_dir).expanduser()
            if index_cache_dir is not None
            else self.checkpoint_path.parent / "retrieval_index"
        )
        base_index_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir = self._resolve_index_dir(base_index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index_manifest_path = self.index_dir / _INDEX_MANIFEST_FILENAME
        self._index_embeddings_path = self.index_dir / _INDEX_EMBED_FILENAME
        self._index_latlon_path = self.index_dir / _INDEX_LATLON_FILENAME

        self._exact_gps_embeddings: torch.Tensor | None = None
        self._ensure_index_cache()
        self._load_index_memmaps()

    @staticmethod
    def _resolve_cache_roots(
        cache_roots: str | Path | Sequence[str | Path] | None,
        backbone_model_name: str,
        *,
        retrieval_db_source: str,
    ) -> list[Path]:
        if cache_roots is None:
            env_value = os.environ.get("PINPOINT_CONTRASTIVE_CACHE_ROOTS", "").strip()
            if env_value:
                raw_roots = [part.strip() for part in env_value.split(",") if part.strip()]
            else:
                raw_roots = list(
                    _default_retrieval_cache_roots(
                        backbone_model_name,
                        retrieval_db_source=retrieval_db_source,
                    )
                )
        elif isinstance(cache_roots, (str, Path)):
            raw_roots = [str(cache_roots)]
        else:
            raw_roots = [str(part) for part in cache_roots]

        resolved: list[Path] = []
        seen: set[Path] = set()
        for root in raw_roots:
            path = Path(root).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"cache root does not exist: {path}")
            if path in seen:
                continue
            seen.add(path)
            resolved.append(path)
        if not resolved:
            raise ValueError("No cache roots provided for retrieval index build.")
        return resolved

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Missing cache manifest: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _resolve_optional_file(cache_root: Path, filename: str | None) -> Path | None:
        if not filename:
            return None
        path = cache_root / str(filename)
        if not path.exists():
            raise FileNotFoundError(f"Missing cache file: {path}")
        return path

    def _load_latlon_sources(self, cache_roots: Sequence[Path]) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for cache_root in cache_roots:
            manifest = self._load_manifest(cache_root / "manifest.json")
            files = manifest.get("files", {})
            shapes = manifest.get("shapes", {})
            num_samples = int(manifest.get("num_samples", 0))
            if num_samples <= 0:
                raise ValueError(f"Invalid num_samples in manifest: {cache_root / 'manifest.json'}")
            latlon_name = files.get("latlon_deg") or files.get("latlon_norm")
            if not latlon_name:
                raise ValueError(
                    f"Manifest missing files.latlon_deg/files.latlon_norm: {cache_root / 'manifest.json'}"
                )
            latlon_path = cache_root / str(latlon_name)
            if not latlon_path.exists():
                raise FileNotFoundError(f"Missing latlon file: {latlon_path}")

            has_deg_coords = files.get("latlon_deg") is not None
            latlon_shape = tuple(int(v) for v in shapes.get("latlon_deg", shapes.get("latlon_norm", [])))
            if latlon_shape != (num_samples, 2):
                raise ValueError(
                    f"Unexpected latlon shape in {cache_root / 'manifest.json'}: "
                    f"{latlon_shape}, expected ({num_samples}, 2)"
                )
            row_index_path = self._resolve_optional_file(cache_root, files.get("row_index"))
            image_ids_path = self._resolve_optional_file(cache_root, files.get("image_ids"))
            specs.append(
                {
                    "cache_root": cache_root,
                    "manifest_path": cache_root / "manifest.json",
                    "latlon_path": latlon_path,
                    "num_samples": num_samples,
                    "is_normalized": not has_deg_coords,
                    "row_index_path": row_index_path,
                    "image_ids_path": image_ids_path,
                }
            )
        return specs

    def _resolve_index_dir(self, base_index_dir: Path) -> Path:
        ckpt_stat = self.checkpoint_path.stat()
        fingerprint_payload = {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_mtime_ns": int(ckpt_stat.st_mtime_ns),
            "checkpoint_size": int(ckpt_stat.st_size),
            "cache_roots": [str(p) for p in self.cache_roots],
            "num_locations": int(self.num_locations),
            "output_dim": int(self.output_dim),
            "retrieval_db_source": self.retrieval_db_source,
        }
        digest = hashlib.sha1(
            json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        return base_index_dir / digest

    def _expected_index_manifest(self) -> dict[str, Any]:
        manifest = {
            "version": _INDEX_MANIFEST_VERSION,
            "checkpoint_path": str(self.checkpoint_path),
            "cache_roots": [str(p) for p in self.cache_roots],
            "num_samples": int(self.num_locations),
            "embedding_dim": int(self.output_dim),
            "retrieval_db_source": self.retrieval_db_source,
            "files": {
                "gps_embeddings": _INDEX_EMBED_FILENAME,
                "latlon_deg": _INDEX_LATLON_FILENAME,
            },
            "dtypes": {
                "gps_embeddings": "float16",
                "latlon_deg": "float32",
            },
            "shapes": {
                "gps_embeddings": [int(self.num_locations), int(self.output_dim)],
                "latlon_deg": [int(self.num_locations), 2],
            },
        }
        return manifest

    def _is_index_cache_valid(self) -> bool:
        if not self._index_manifest_path.exists():
            return False
        if not self._index_embeddings_path.exists():
            return False
        if not self._index_latlon_path.exists():
            return False

        try:
            manifest = self._load_manifest(self._index_manifest_path)
        except Exception:
            return False

        expected = self._expected_index_manifest()
        keys_to_match = (
            "version",
            "checkpoint_path",
            "cache_roots",
            "num_samples",
            "embedding_dim",
            "retrieval_db_source",
            "files",
            "dtypes",
            "shapes",
        )
        for key in keys_to_match:
            if manifest.get(key) != expected.get(key):
                return False
        return True

    def _ensure_index_cache(self) -> None:
        if self._is_index_cache_valid():
            return
        self._build_index_cache()

    def _encode_images(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            if self.device.type == "cuda" and self.inference_dtype in _SUPPORTED_HALF_DTYPES:
                with torch.autocast(device_type="cuda", dtype=self.inference_dtype):
                    return self.geo_module.encode_image_for_source(
                        image_embeddings,
                        source_name=self.retrieval_db_source,
                    )
            return self.geo_module.encode_image_for_source(
                image_embeddings.to(torch.float32),
                source_name=self.retrieval_db_source,
            )

    def _build_index_cache(self) -> None:
        use_half_autocast = self.device.type == "cuda" and self.inference_dtype in _SUPPORTED_HALF_DTYPES

        def _encoded_to_numpy(tensor: torch.Tensor) -> np.ndarray:
            tensor = torch.nan_to_num(tensor.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            tensor = tensor.to(dtype=torch.float16)
            return np.ascontiguousarray(tensor.cpu().numpy(), dtype=np.float16)

        emb_tmp = self._index_embeddings_path.with_suffix(self._index_embeddings_path.suffix + ".tmp")
        lat_tmp = self._index_latlon_path.with_suffix(self._index_latlon_path.suffix + ".tmp")
        manifest_tmp = self._index_manifest_path.with_suffix(self._index_manifest_path.suffix + ".tmp")

        emb_mm = np.memmap(
            emb_tmp,
            dtype=np.float16,
            mode="w+",
            shape=(self.num_locations, self.output_dim),
        )
        lat_mm = np.memmap(
            lat_tmp,
            dtype=np.float32,
            mode="w+",
            shape=(self.num_locations, 2),
        )
        write_offset = 0
        progress_bar = tqdm(
            total=self.num_locations,
            desc="Building retrieval cache",
            unit="rows",
            dynamic_ncols=True,
        )
        try:
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=self.inference_dtype)
                if use_half_autocast
                else nullcontext()
            )
            with torch.inference_mode():
                with autocast_context:
                    for source in self._source_specs:
                        src_latlon = np.memmap(
                            source["latlon_path"],
                            dtype=np.float32,
                            mode="r",
                            shape=(source["num_samples"], 2),
                        )
                        num_rows = int(source["num_samples"])
                        for start in range(0, num_rows, self.build_batch_size):
                            end = min(start + self.build_batch_size, num_rows)
                            batch_np = np.array(src_latlon[start:end], dtype=np.float32, copy=True, order="C")
                            if source.get("is_normalized", False):
                                batch_np = batch_np * np.asarray([90.0, 180.0], dtype=np.float32)
                            batch_t = torch.from_numpy(batch_np).to(
                                self.device,
                                non_blocking=(self.device.type == "cuda"),
                            )
                            loc_input = batch_t if use_half_autocast else batch_t.to(torch.float32)
                            loc_np = _encoded_to_numpy(self.geo_module.encode_location(loc_input))

                            rows = end - start
                            dst_slice = slice(write_offset, write_offset + rows)
                            emb_mm[dst_slice] = loc_np
                            lat_mm[dst_slice] = batch_np
                            write_offset += rows
                            progress_bar.update(rows)
                        del src_latlon
            if write_offset != self.num_locations:
                raise RuntimeError(
                    f"Index build wrote {write_offset} rows, expected {self.num_locations}"
                )
        finally:
            progress_bar.close()
            emb_mm.flush()
            lat_mm.flush()
            del emb_mm
            del lat_mm

        manifest = self._expected_index_manifest()
        with manifest_tmp.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

        emb_tmp.replace(self._index_embeddings_path)
        lat_tmp.replace(self._index_latlon_path)
        manifest_tmp.replace(self._index_manifest_path)

    def _load_index_memmaps(self) -> None:
        self._gps_embeddings_mm = np.memmap(
            self._index_embeddings_path,
            dtype=np.float16,
            mode="r",
            shape=(self.num_locations, self.output_dim),
        )
        self._gps_latlon_mm = np.memmap(
            self._index_latlon_path,
            dtype=np.float32,
            mode="r",
            shape=(self.num_locations, 2),
        )
        self._load_exact_embeddings()

    def _embeddings_mm_to_device(self, embeddings_mm: np.memmap) -> torch.Tensor:
        embeddings_np = np.array(embeddings_mm, dtype=np.float16, copy=True, order="C")
        embeddings_t = torch.from_numpy(embeddings_np)
        embeddings_t = embeddings_t.to(
            self.device,
            dtype=torch.float16,
            non_blocking=(self.device.type == "cuda"),
        )
        return embeddings_t.contiguous()

    def _load_exact_embeddings(self) -> None:
        self._exact_gps_embeddings = self._embeddings_mm_to_device(self._gps_embeddings_mm)

    def _prepare_embedding(self, image_embedding: Any) -> torch.Tensor:
        if torch.is_tensor(image_embedding):
            embedding = image_embedding.detach()
        elif isinstance(image_embedding, np.ndarray):
            embedding = torch.from_numpy(image_embedding)
        else:
            embedding = torch.as_tensor(image_embedding)

        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        elif embedding.ndim != 2 or embedding.shape[0] != 1:
            raise ValueError(
                f"predict expects one embedding of shape [D] or [1, D], got {tuple(embedding.shape)}"
            )

        incoming_dim = int(embedding.shape[1])
        if incoming_dim == self.image_embed_dim:
            image_only = embedding
        else:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.image_embed_dim}, got {incoming_dim}"
            )

        if image_only.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            image_only = image_only.to(torch.float32)
        image_only = torch.nan_to_num(image_only, nan=0.0, posinf=0.0, neginf=0.0)

        return image_only.to(
            self.device,
            non_blocking=(self.device.type == "cuda"),
        )

    def _encode_image(self, embedding: torch.Tensor) -> torch.Tensor:
        query = self._encode_images(embedding)
        query = torch.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)
        return query[0].contiguous()

    def _search_exact(self, query_vec: torch.Tensor) -> tuple[int, float]:
        if self._exact_gps_embeddings is None:
            raise RuntimeError("Exact GPS embedding matrix was not loaded.")
        with torch.inference_mode():
            query_half = query_vec.to(
                self._exact_gps_embeddings.device,
                dtype=torch.float16,
                non_blocking=(self._exact_gps_embeddings.device.type == "cuda"),
            )
            gps_scores = torch.matmul(self._exact_gps_embeddings, query_half)
            best_idx = int(torch.argmax(gps_scores).item())
            best_score = float(gps_scores[best_idx].to(torch.float32).item())
        return best_idx, best_score

    def _search(self, query_vec: torch.Tensor) -> tuple[int, float]:
        return self._search_exact(query_vec)

    @staticmethod
    def _normalize_lon(lon: float) -> float:
        wrapped = ((lon + 180.0) % 360.0) - 180.0
        return wrapped

    def predict(self, image_embedding: Any) -> tuple[float, float]:
        embedding = self._prepare_embedding(image_embedding)
        query_vec = self._encode_image(embedding)
        best_idx, _best_score = self._search(query_vec)
        lat = float(self._gps_latlon_mm[best_idx, 0])
        lon = float(self._gps_latlon_mm[best_idx, 1])
        lat = float(np.clip(lat, -90.0, 90.0))
        lon = self._normalize_lon(float(lon))
        return (lat, lon)


def build_model(
    checkpoint_path: str | Path | None = None,
    backbone_model_name: str | None = None,
    device: str | None = None,
    inference_dtype: str | None = None,
    batch_size: int | None = None,
    checkpoint_strict: bool = True,
    retrieval_db_source: str | None = None,
) -> LinearGeoModel:
    """Factory for benchmarks.load_model()."""
    return LinearGeoModel(
        checkpoint_path=checkpoint_path,
        backbone_model_name=backbone_model_name,
        device=device,
        inference_dtype=inference_dtype,
        batch_size=batch_size,
        checkpoint_strict=checkpoint_strict,
        retrieval_db_source=retrieval_db_source,
    )
