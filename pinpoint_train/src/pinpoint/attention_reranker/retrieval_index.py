from __future__ import annotations

import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from pinpoint.utils import (
    _resolve_device,
    _resolve_inference_dtype,
    model_name_tail,
    resolve_backbone_model_name,
)


_INDEX_MANIFEST_VERSION = 6
_INDEX_MANIFEST_FILENAME = "manifest.json"
_INDEX_IMAGE_EMBED_FILENAME = "image_embeddings.bin"
_INDEX_RAW_IMAGE_EMBED_FILENAME = "raw_image_embeddings.bin"
_INDEX_RAW_IMAGE_SCALE_FILENAME = "raw_image_scales.bin"
_INDEX_GPS_EMBED_FILENAME = "gps_embeddings.bin"
_INDEX_LATLON_FILENAME = "latlon_deg.bin"
_INDEX_SOURCE_ID_FILENAME = "source_ids.bin"
_INDEX_CACHE_IDX_FILENAME = "cache_indices.bin"
_SUPPORTED_HALF_DTYPES = (torch.float16, torch.bfloat16)
_RAW_IMAGE_LOAD_CHUNK_BYTES = 1024 * 1024 * 1024
_CONTRASTIVE_MODEL_MODULE = "pinpoint.contrastive_retrieval.lightning_module"


def _resolve_contrastive_geo_module_class():
    module = import_module(_CONTRASTIVE_MODEL_MODULE)
    return getattr(module, "GeoLightningModule")


def _l2_normalize_numpy_rows(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    x = x / np.clip(norms, 1e-8, None)
    return np.ascontiguousarray(x, dtype=np.float32)


def _rows_per_chunk(embedding_dim: int, *, target_bytes: int) -> int:
    bytes_per_row = max(int(embedding_dim), 1) * np.dtype(np.float32).itemsize
    return max(1, int(target_bytes) // bytes_per_row)


def _normalize_retrieval_db_source(raw_source: str | None) -> str:
    source_name = str(
        raw_source or os.environ.get("PINPOINT_CONTRASTIVE_RETRIEVAL_DB_SOURCE", "mp16")
    ).strip().lower()
    if source_name not in {"mp16", "osv5m"}:
        raise ValueError(f"Unsupported retrieval_db_source '{source_name}'. Expected one of: mp16, osv5m")
    return source_name


def _default_retrieval_cache_roots(backbone_model_name: str, *, retrieval_db_source: str) -> tuple[str, ...]:
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


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing cache manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _split_bucket(image_id: str) -> int:
    digest = hashlib.md5(image_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


def _keep_for_split(image_id: str, split: str, val_pct: int) -> bool:
    if split == "all":
        return True
    bucket = _split_bucket(image_id)
    val_start = 100 - int(val_pct)
    if split == "train":
        return bucket < val_start
    if split == "val":
        return bucket >= val_start
    raise ValueError(f"Unknown split '{split}'. Expected one of: all, train, val")


def _manifest_val_pct(manifest: dict[str, Any]) -> int:
    manifest_val_pct = manifest.get("val_pct")
    if manifest_val_pct is None:
        manifest_val_pct = manifest.get("val_bucket_pct")
    if manifest_val_pct is None and "train_bucket_pct" in manifest:
        manifest_val_pct = 100 - int(manifest["train_bucket_pct"])
    if manifest_val_pct is None:
        manifest_val_pct = 2
    return int(manifest_val_pct)


def _normalize_named_roots(
    cache_roots: str | Path | Sequence[str | Path] | Mapping[str, str | Path] | None,
    *,
    backbone_model_name: str,
    retrieval_db_source: str,
) -> list[tuple[str, Path]]:
    if cache_roots is None:
        env_value = os.environ.get("PINPOINT_CONTRASTIVE_CACHE_ROOTS", "").strip()
        if env_value:
            raw_items: list[tuple[str, str]] = []
            for index, part in enumerate(env_value.split(",")):
                part = part.strip()
                if not part:
                    continue
                raw_items.append((f"source_{index}", part))
        else:
            defaults = _default_retrieval_cache_roots(
                backbone_model_name,
                retrieval_db_source=retrieval_db_source,
            )
            raw_items = [(f"source_{index}", root) for index, root in enumerate(defaults)]
    elif isinstance(cache_roots, Mapping):
        raw_items = [(str(key), str(value)) for key, value in cache_roots.items()]
    elif isinstance(cache_roots, (str, Path)):
        raw_items = [("source_0", str(cache_roots))]
    else:
        raw_items = [(f"source_{index}", str(value)) for index, value in enumerate(cache_roots)]

    normalized: list[tuple[str, Path]] = []
    used_names: set[str] = set()
    for fallback_name, raw_path in raw_items:
        root = Path(raw_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"cache root does not exist: {root}")
        suggested = fallback_name.strip() or root.name or "source"
        name = suggested
        suffix = 1
        while name in used_names:
            name = f"{suggested}_{suffix}"
            suffix += 1
        used_names.add(name)
        normalized.append((name, root))

    if not normalized:
        raise ValueError("No cache roots provided for retrieval index build.")
    return normalized


def _resolve_optional_file(cache_root: Path, filename: str | None) -> Path | None:
    if not filename:
        return None
    path = cache_root / str(filename)
    if not path.exists():
        raise FileNotFoundError(f"Missing cache file: {path}")
    return path


def _load_image_ids(path: Path, num_samples: int) -> list[str]:
    image_ids = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    if len(image_ids) != num_samples:
        raise ValueError(
            f"image_ids length mismatch in {path}: got {len(image_ids)}, expected {num_samples}"
        )
    return image_ids


def _select_source_rows(
    source: dict[str, Any],
    *,
    split: str,
    val_pct_override: int | None,
) -> np.ndarray:
    num_samples = int(source["num_samples"])
    if split == "all":
        return np.arange(num_samples, dtype=np.int64)

    image_ids_path = source.get("image_ids_path")
    if image_ids_path is None:
        raise FileNotFoundError(
            f"split={split} requires image_ids metadata, but none was found for {source['manifest_path']}"
        )

    image_ids = _load_image_ids(image_ids_path, num_samples)
    source_val_pct = int(source.get("val_pct", 2))
    val_pct = int(source_val_pct if val_pct_override is None else val_pct_override)
    if not (0 <= val_pct < 100):
        raise ValueError("val_pct must be in [0, 99]")

    keep_rows: list[int] = []
    for row_idx, image_id in enumerate(image_ids):
        if _keep_for_split(str(image_id), split=split, val_pct=val_pct):
            keep_rows.append(row_idx)
    if not keep_rows:
        raise ValueError(f"No rows kept for source={source['source_name']} split={split}")
    return np.asarray(keep_rows, dtype=np.int64)


def _encode_locations(
    geo_module,
    latlon_deg: torch.Tensor,
    *,
    device: torch.device,
    inference_dtype: torch.dtype,
) -> torch.Tensor:
    with torch.inference_mode():
        if device.type == "cuda" and inference_dtype in _SUPPORTED_HALF_DTYPES:
            with torch.autocast(device_type="cuda", dtype=inference_dtype):
                return geo_module.encode_location(latlon_deg)
        return geo_module.encode_location(latlon_deg.to(torch.float32))


def _encode_image(
    geo_module,
    image_embeddings: torch.Tensor,
    *,
    device: torch.device,
    inference_dtype: torch.dtype,
    retrieval_db_source: str,
) -> torch.Tensor:
    with torch.inference_mode():
        if device.type == "cuda" and inference_dtype in _SUPPORTED_HALF_DTYPES:
            with torch.autocast(device_type="cuda", dtype=inference_dtype):
                if hasattr(geo_module, "encode_image_for_source"):
                    return geo_module.encode_image_for_source(
                        image_embeddings,
                        source_name=retrieval_db_source,
                    )
                return geo_module.encode_image(image_embeddings)
        image_embeddings = image_embeddings.to(torch.float32)
        if hasattr(geo_module, "encode_image_for_source"):
            return geo_module.encode_image_for_source(
                image_embeddings,
                source_name=retrieval_db_source,
            )
        return geo_module.encode_image(image_embeddings)


def _resolve_index_dir(
    *,
    checkpoint_path: Path,
    named_cache_roots: Sequence[tuple[str, Path]],
    split: str,
    val_pct: int | None,
    retrieval_db_source: str,
    base_index_dir: Path,
) -> Path:
    ckpt_stat = checkpoint_path.stat()
    fingerprint_payload = {
        "version": _INDEX_MANIFEST_VERSION,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_mtime_ns": int(ckpt_stat.st_mtime_ns),
        "checkpoint_size": int(ckpt_stat.st_size),
        "cache_roots": [(name, str(path)) for name, path in named_cache_roots],
        "split": split,
        "val_pct": val_pct,
        "retrieval_db_source": retrieval_db_source,
    }
    digest = hashlib.sha1(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return base_index_dir / digest


def _load_source_specs(
    *,
    named_cache_roots: Sequence[tuple[str, Path]],
    image_embed_dim: int,
    split: str,
    val_pct_override: int | None,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for source_id, (source_name, cache_root) in enumerate(named_cache_roots):
        manifest_path = cache_root / "manifest.json"
        manifest = _load_manifest(manifest_path)
        files = manifest.get("files", {})
        shapes = manifest.get("shapes", {})
        dtypes = manifest.get("dtypes", {})

        num_samples = int(manifest.get("num_samples", 0))
        embedding_dim = int(manifest.get("embedding_dim", 0))
        if num_samples <= 0:
            raise ValueError(f"Invalid num_samples in manifest: {manifest_path}")
        if embedding_dim != int(image_embed_dim):
            raise ValueError(
                f"Image embedding dim mismatch for {manifest_path}: cache={embedding_dim}, "
                f"contrastive_image_embed_dim={image_embed_dim}"
            )

        image_emb_name = files.get("embeddings")
        if not image_emb_name:
            raise ValueError(f"Manifest missing files.embeddings: {manifest_path}")
        image_emb_path = cache_root / str(image_emb_name)
        if not image_emb_path.exists():
            raise FileNotFoundError(f"Missing embeddings file: {image_emb_path}")
        image_emb_shape = tuple(int(v) for v in shapes.get("embeddings", []))
        if image_emb_shape != (num_samples, embedding_dim):
            raise ValueError(
                f"Unexpected embeddings shape in {manifest_path}: {image_emb_shape}, "
                f"expected ({num_samples}, {embedding_dim})"
            )

        latlon_name = files.get("latlon_deg") or files.get("latlon_norm")
        if not latlon_name:
            raise ValueError(
                f"Manifest missing files.latlon_deg/files.latlon_norm: {manifest_path}"
            )
        latlon_path = cache_root / str(latlon_name)
        if not latlon_path.exists():
            raise FileNotFoundError(f"Missing latlon file: {latlon_path}")

        latlon_shape = tuple(int(v) for v in shapes.get("latlon_deg", shapes.get("latlon_norm", [])))
        if latlon_shape != (num_samples, 2):
            raise ValueError(
                f"Unexpected latlon shape in {manifest_path}: {latlon_shape}, expected ({num_samples}, 2)"
            )

        source = {
            "source_id": int(source_id),
            "source_name": source_name,
            "cache_root": cache_root,
            "manifest_path": manifest_path,
            "num_samples": num_samples,
            "val_pct": _manifest_val_pct(manifest),
            "image_embeddings_path": image_emb_path,
            "image_embeddings_np_dtype": np.dtype(dtypes.get("embeddings", "float32")),
            "latlon_path": latlon_path,
            "is_normalized": files.get("latlon_deg") is None,
            "row_index_path": _resolve_optional_file(cache_root, files.get("row_index")),
            "image_ids_path": _resolve_optional_file(cache_root, files.get("image_ids")),
            "row_index_dtype": np.dtype(dtypes.get("row_index", "int64")),
        }

        selected_rows = _select_source_rows(source, split=split, val_pct_override=val_pct_override)
        source["selected_rows"] = selected_rows
        source["selected_count"] = int(selected_rows.shape[0])
        specs.append(source)

    return specs


def ensure_attention_retrieval_index(
    *,
    contrastive_checkpoint_path: str | Path,
    cache_roots: str | Path | Sequence[str | Path] | Mapping[str, str | Path] | None,
    index_cache_dir: str | Path | None = None,
    split: str = "train",
    val_pct: int | None = None,
    build_batch_size: int = 8192,
    build_device: str | None = None,
    build_dtype: str | None = None,
    checkpoint_strict: bool = True,
    retrieval_db_source: str | None = None,
) -> Path:
    if split not in {"all", "train", "val"}:
        raise ValueError("split must be one of: all, train, val")
    if build_batch_size <= 0:
        raise ValueError("build_batch_size must be > 0")

    checkpoint_path = Path(contrastive_checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"contrastive checkpoint not found: {checkpoint_path}")

    build_device_obj = _resolve_device(build_device or "auto")
    build_dtype_obj = _resolve_inference_dtype(build_dtype or "auto", build_device_obj)
    retrieval_db_source = _normalize_retrieval_db_source(retrieval_db_source)

    contrastive_geo_module_cls = _resolve_contrastive_geo_module_class()
    geo_module = contrastive_geo_module_cls.load_from_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
        strict=checkpoint_strict,
    ).to(build_device_obj)
    geo_module.eval()
    for parameter in geo_module.parameters():
        parameter.requires_grad = False

    backbone_model_name = resolve_backbone_model_name()
    named_cache_roots = _normalize_named_roots(
        cache_roots,
        backbone_model_name=backbone_model_name,
        retrieval_db_source=retrieval_db_source,
    )

    if index_cache_dir is not None:
        direct_index_dir = Path(index_cache_dir).expanduser().resolve()
        direct_manifest_path = direct_index_dir / _INDEX_MANIFEST_FILENAME
        if direct_manifest_path.exists():
            direct_manifest = _load_manifest(direct_manifest_path)
            direct_files = direct_manifest.get("files", {})
            direct_required = [
                direct_index_dir / str(direct_files.get("image_embeddings", "")),
                direct_index_dir / str(direct_files.get("gps_embeddings", "")),
                direct_index_dir / str(direct_files.get("latlon_deg", "")),
                direct_index_dir / str(direct_files.get("source_ids", "")),
                direct_index_dir / str(direct_files.get("cache_indices", "")),
            ]
            raw_image_file = direct_files.get("raw_image_embeddings")
            if raw_image_file:
                direct_required.append(direct_index_dir / str(raw_image_file))
            raw_image_scale_file = direct_files.get("raw_image_scales")
            if raw_image_scale_file:
                direct_required.append(direct_index_dir / str(raw_image_scale_file))
            if all(path.exists() for path in direct_required):
                if (
                    int(direct_manifest.get("version", -1)) == _INDEX_MANIFEST_VERSION
                    and int(direct_manifest.get("embedding_dim", -1)) == int(getattr(geo_module, "output_dim", 0))
                    and str(direct_manifest.get("split", "")) == split
                    # the released index records the authors' checkpoint path; a directly given index is trusted
                    and str(direct_manifest.get("retrieval_db_source", "mp16")) == retrieval_db_source
                ):
                    return direct_index_dir

    base_index_dir = (
        Path(index_cache_dir).expanduser().resolve()
        if index_cache_dir is not None
        else checkpoint_path.parent / "attention_retrieval_index"
    )
    base_index_dir.mkdir(parents=True, exist_ok=True)
    index_dir = _resolve_index_dir(
        checkpoint_path=checkpoint_path,
        named_cache_roots=named_cache_roots,
        split=split,
        val_pct=val_pct,
        retrieval_db_source=retrieval_db_source,
        base_index_dir=base_index_dir,
    )
    index_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = index_dir / _INDEX_MANIFEST_FILENAME
    image_emb_path = index_dir / _INDEX_IMAGE_EMBED_FILENAME
    raw_image_emb_path = index_dir / _INDEX_RAW_IMAGE_EMBED_FILENAME
    raw_image_scale_path = index_dir / _INDEX_RAW_IMAGE_SCALE_FILENAME
    gps_emb_path = index_dir / _INDEX_GPS_EMBED_FILENAME
    latlon_path = index_dir / _INDEX_LATLON_FILENAME
    source_ids_path = index_dir / _INDEX_SOURCE_ID_FILENAME
    cache_idx_path = index_dir / _INDEX_CACHE_IDX_FILENAME

    manifest = _load_manifest(manifest_path) if manifest_path.exists() else None
    manifest_files = manifest.get("files", {}) if manifest is not None else {}
    required_manifest_keys = (
        "image_embeddings",
        "raw_image_embeddings",
        "raw_image_scales",
        "gps_embeddings",
        "latlon_deg",
        "source_ids",
        "cache_indices",
    )
    has_required = manifest is not None and all(
        manifest_files.get(key) and (index_dir / str(manifest_files[key])).exists()
        for key in required_manifest_keys
    )
    if has_required:
        assert manifest is not None
        if (
            int(manifest.get("version", -1)) == _INDEX_MANIFEST_VERSION
            and int(manifest.get("embedding_dim", -1)) == int(getattr(geo_module, "output_dim", 0))
            and str(manifest.get("split", "")) == split
            and manifest.get("checkpoint_path") == str(checkpoint_path)
            and str(manifest.get("retrieval_db_source", "mp16")) == retrieval_db_source
        ):
            return index_dir

    specs = _load_source_specs(
        named_cache_roots=named_cache_roots,
        image_embed_dim=int(getattr(geo_module, "image_embed_dim", 0)),
        split=split,
        val_pct_override=val_pct,
    )

    output_dim = int(getattr(geo_module, "output_dim", 0))
    if output_dim <= 0:
        raise RuntimeError("Contrastive checkpoint does not expose a valid output_dim")

    num_rows = int(sum(spec["selected_count"] for spec in specs))
    if num_rows <= 0:
        raise RuntimeError("No rows selected for retrieval index build")

    image_tmp = image_emb_path.with_suffix(image_emb_path.suffix + ".tmp")
    raw_image_tmp = raw_image_emb_path.with_suffix(raw_image_emb_path.suffix + ".tmp")
    raw_image_scale_tmp = raw_image_scale_path.with_suffix(raw_image_scale_path.suffix + ".tmp")
    gps_tmp = gps_emb_path.with_suffix(gps_emb_path.suffix + ".tmp")
    latlon_tmp = latlon_path.with_suffix(latlon_path.suffix + ".tmp")
    source_ids_tmp = source_ids_path.with_suffix(source_ids_path.suffix + ".tmp")
    cache_idx_tmp = cache_idx_path.with_suffix(cache_idx_path.suffix + ".tmp")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")

    image_mm = np.memmap(image_tmp, dtype=np.float16, mode="w+", shape=(num_rows, output_dim))
    raw_image_mm = np.memmap(
        raw_image_tmp,
        dtype=np.int8,
        mode="w+",
        shape=(num_rows, int(getattr(geo_module, "image_embed_dim", 0))),
    )
    raw_image_scale_mm = np.memmap(
        raw_image_scale_tmp,
        dtype=np.float16,
        mode="w+",
        shape=(num_rows,),
    )
    gps_mm = np.memmap(gps_tmp, dtype=np.float16, mode="w+", shape=(num_rows, output_dim))
    latlon_mm = np.memmap(latlon_tmp, dtype=np.float32, mode="w+", shape=(num_rows, 2))
    source_ids_mm = np.memmap(source_ids_tmp, dtype=np.int16, mode="w+", shape=(num_rows,))
    cache_idx_mm = np.memmap(cache_idx_tmp, dtype=np.int64, mode="w+", shape=(num_rows,))

    write_offset = 0
    try:
        for spec in specs:
            src_image_mm = np.memmap(
                spec["image_embeddings_path"],
                dtype=spec["image_embeddings_np_dtype"],
                mode="r",
                shape=(spec["num_samples"], int(getattr(geo_module, "image_embed_dim", 0))),
            )
            src_latlon_mm = np.memmap(
                spec["latlon_path"],
                dtype=np.float32,
                mode="r",
                shape=(spec["num_samples"], 2),
            )

            selected_rows = spec["selected_rows"]
            for start in range(0, selected_rows.shape[0], build_batch_size):
                end = min(start + build_batch_size, selected_rows.shape[0])
                rows = selected_rows[start:end]

                latlon_np = np.asarray(src_latlon_mm[rows], dtype=np.float32)
                if spec.get("is_normalized", False):
                    latlon_np = latlon_np * np.asarray([90.0, 180.0], dtype=np.float32)
                latlon_np = np.ascontiguousarray(latlon_np, dtype=np.float32)
                latlon_t = torch.from_numpy(latlon_np).to(
                    build_device_obj,
                    non_blocking=(build_device_obj.type == "cuda"),
                )
                gps_features = _encode_locations(
                    geo_module,
                    latlon_t,
                    device=build_device_obj,
                    inference_dtype=build_dtype_obj,
                )
                gps_np = gps_features.to(torch.float32).detach().cpu().numpy()
                gps_np = np.nan_to_num(gps_np, nan=0.0, posinf=0.0, neginf=0.0)
                gps_np = np.ascontiguousarray(gps_np, dtype=np.float32)

                image_raw_np = np.asarray(src_image_mm[rows], dtype=np.float32)
                image_raw_np = np.nan_to_num(image_raw_np, nan=0.0, posinf=0.0, neginf=0.0)
                image_raw_np = np.ascontiguousarray(image_raw_np, dtype=np.float32)
                image_raw_t = torch.from_numpy(image_raw_np).to(
                    build_device_obj,
                    non_blocking=(build_device_obj.type == "cuda"),
                )
                image_features = _encode_image(
                    geo_module,
                    image_raw_t,
                    device=build_device_obj,
                    inference_dtype=build_dtype_obj,
                    retrieval_db_source=retrieval_db_source,
                )
                image_np = image_features.to(torch.float32).detach().cpu().numpy()
                image_np = np.nan_to_num(image_np, nan=0.0, posinf=0.0, neginf=0.0)
                image_np = np.ascontiguousarray(image_np, dtype=np.float32)
                raw_image_np = _l2_normalize_numpy_rows(image_raw_np)
                raw_image_scale_np = np.maximum(
                    np.max(np.abs(raw_image_np), axis=1),
                    1e-8,
                ) / 127.0
                raw_image_quantized_np = np.clip(
                    np.rint(raw_image_np / raw_image_scale_np[:, None]),
                    -127,
                    127,
                ).astype(np.int8)

                rows_written = end - start
                dst = slice(write_offset, write_offset + rows_written)
                image_mm[dst] = image_np.astype(np.float16, copy=False)
                raw_image_mm[dst] = raw_image_quantized_np
                raw_image_scale_mm[dst] = raw_image_scale_np.astype(np.float16, copy=False)
                gps_mm[dst] = gps_np.astype(np.float16, copy=False)
                latlon_mm[dst] = latlon_np
                source_ids_mm[dst] = int(spec["source_id"])
                cache_idx_mm[dst] = rows.astype(np.int64, copy=False)
                write_offset += rows_written

            del src_image_mm
            del src_latlon_mm

        if write_offset != num_rows:
            raise RuntimeError(f"Index build wrote {write_offset} rows, expected {num_rows}")
    finally:
        image_mm.flush()
        raw_image_mm.flush()
        raw_image_scale_mm.flush()
        gps_mm.flush()
        latlon_mm.flush()
        source_ids_mm.flush()
        cache_idx_mm.flush()
        del image_mm
        del raw_image_mm
        del raw_image_scale_mm
        del gps_mm
        del latlon_mm
        del source_ids_mm
        del cache_idx_mm

    sources_manifest = [
        {
            "id": int(spec["source_id"]),
            "name": str(spec["source_name"]),
            "cache_root": str(spec["cache_root"]),
            "num_samples": int(spec["num_samples"]),
            "selected_count": int(spec["selected_count"]),
        }
        for spec in specs
    ]
    manifest = {
        "version": _INDEX_MANIFEST_VERSION,
        "checkpoint_path": str(checkpoint_path),
        "split": split,
        "val_pct": val_pct,
        "retrieval_db_source": retrieval_db_source,
        "embedding_dim": int(output_dim),
        "num_samples": int(num_rows),
        "sources": sources_manifest,
        "files": {
            "image_embeddings": _INDEX_IMAGE_EMBED_FILENAME,
            "raw_image_embeddings": _INDEX_RAW_IMAGE_EMBED_FILENAME,
            "raw_image_scales": _INDEX_RAW_IMAGE_SCALE_FILENAME,
            "gps_embeddings": _INDEX_GPS_EMBED_FILENAME,
            "latlon_deg": _INDEX_LATLON_FILENAME,
            "source_ids": _INDEX_SOURCE_ID_FILENAME,
            "cache_indices": _INDEX_CACHE_IDX_FILENAME,
        },
        "dtypes": {
            "image_embeddings": "float16",
            "raw_image_embeddings": "int8",
            "raw_image_scales": "float16",
            "gps_embeddings": "float16",
            "latlon_deg": "float32",
            "source_ids": "int16",
            "cache_indices": "int64",
        },
        "shapes": {
            "image_embeddings": [int(num_rows), int(output_dim)],
            "raw_image_embeddings": [int(num_rows), int(getattr(geo_module, "image_embed_dim", 0))],
            "raw_image_scales": [int(num_rows)],
            "gps_embeddings": [int(num_rows), int(output_dim)],
            "latlon_deg": [int(num_rows), 2],
            "source_ids": [int(num_rows)],
            "cache_indices": [int(num_rows)],
        },
    }

    with manifest_tmp.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    image_tmp.replace(image_emb_path)
    raw_image_tmp.replace(raw_image_emb_path)
    raw_image_scale_tmp.replace(raw_image_scale_path)
    gps_tmp.replace(gps_emb_path)
    latlon_tmp.replace(latlon_path)
    source_ids_tmp.replace(source_ids_path)
    cache_idx_tmp.replace(cache_idx_path)
    manifest_tmp.replace(manifest_path)
    return index_dir


class AttentionRetrievalIndex:
    """Load retrieval index files and provide tensors plus query-id lookup utilities."""

    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir).expanduser().resolve()
        self.manifest_path = self.index_dir / _INDEX_MANIFEST_FILENAME
        self.manifest = _load_manifest(self.manifest_path)

        self.num_samples = int(self.manifest.get("num_samples", 0))
        self.embedding_dim = int(self.manifest.get("embedding_dim", 0))
        if self.num_samples <= 0:
            raise ValueError(f"Invalid num_samples in {self.manifest_path}")
        if self.embedding_dim <= 0:
            raise ValueError(f"Invalid embedding_dim in {self.manifest_path}")

        files = self.manifest.get("files", {})

        self.image_embeddings_path = self.index_dir / str(files.get("image_embeddings", ""))
        raw_image_name = files.get("raw_image_embeddings")
        raw_image_scale_name = files.get("raw_image_scales")
        self.raw_image_embeddings_path = (
            self.index_dir / str(raw_image_name)
            if raw_image_name
            else None
        )
        self.raw_image_scales_path = (
            self.index_dir / str(raw_image_scale_name)
            if raw_image_scale_name
            else None
        )
        self.gps_embeddings_path = self.index_dir / str(files.get("gps_embeddings", ""))
        self.latlon_path = self.index_dir / str(files.get("latlon_deg", ""))
        self.source_ids_path = self.index_dir / str(files.get("source_ids", ""))
        self.cache_idx_path = self.index_dir / str(files.get("cache_indices", ""))

        required = [
            self.image_embeddings_path,
            self.gps_embeddings_path,
            self.latlon_path,
            self.source_ids_path,
            self.cache_idx_path,
        ]
        if self.raw_image_embeddings_path is not None:
            required.append(self.raw_image_embeddings_path)
        if self.raw_image_scales_path is not None:
            required.append(self.raw_image_scales_path)
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Missing retrieval index file: {path}")

        raw_image_shape = tuple(int(v) for v in self.manifest.get("shapes", {}).get("raw_image_embeddings", []))
        self.raw_image_embedding_dim = int(raw_image_shape[1]) if len(raw_image_shape) == 2 else 0
        self.raw_image_storage_dtype = np.dtype(
            self.manifest.get("dtypes", {}).get("raw_image_embeddings", "float16")
        )
        if self.raw_image_storage_dtype == np.dtype(np.int8) and self.raw_image_scales_path is None:
            raise ValueError(
                f"Quantized raw image embeddings require files.raw_image_scales: {self.manifest_path}"
            )

        self._image_embeddings_mm = np.memmap(
            self.image_embeddings_path,
            dtype=np.float16,
            mode="r",
            shape=(self.num_samples, self.embedding_dim),
        )
        self._gps_embeddings_mm = np.memmap(
            self.gps_embeddings_path,
            dtype=np.float16,
            mode="r",
            shape=(self.num_samples, self.embedding_dim),
        )
        self._raw_image_embeddings_mm = (
            np.memmap(
                self.raw_image_embeddings_path,
                dtype=self.raw_image_storage_dtype,
                mode="r",
                shape=(self.num_samples, self.raw_image_embedding_dim),
            )
            if self.raw_image_embeddings_path is not None and self.raw_image_embedding_dim > 0
            else None
        )
        self._raw_image_scales_mm = (
            np.memmap(
                self.raw_image_scales_path,
                dtype=np.float16,
                mode="r",
                shape=(self.num_samples,),
            )
            if self.raw_image_scales_path is not None
            else None
        )
        self._latlon_mm = np.memmap(
            self.latlon_path,
            dtype=np.float32,
            mode="r",
            shape=(self.num_samples, 2),
        )
        self._source_ids_mm = np.memmap(
            self.source_ids_path,
            dtype=np.int16,
            mode="r",
            shape=(self.num_samples,),
        )
        self._cache_idx_mm = np.memmap(
            self.cache_idx_path,
            dtype=np.int64,
            mode="r",
            shape=(self.num_samples,),
        )

        self.image_embeddings: torch.Tensor | None = None
        self.raw_image_embeddings: torch.Tensor | None = None
        self.gps_embeddings: torch.Tensor | None = None
        self.latlon_deg: torch.Tensor | None = None
        self._device: torch.device | None = None

        self._source_lookup = self._build_source_lookup()

    def _build_source_lookup(self) -> dict[str, np.ndarray]:
        source_ids = np.asarray(self._source_ids_mm, dtype=np.int16)
        cache_indices = np.asarray(self._cache_idx_mm, dtype=np.int64)
        lookup: dict[str, np.ndarray] = {}
        for source in self.manifest.get("sources", []):
            source_name = str(source.get("name", ""))
            source_id = int(source.get("id", -1))
            source_num_samples = int(source.get("num_samples", 0))
            if not source_name or source_id < 0 or source_num_samples <= 0:
                continue
            mapping = np.full(source_num_samples, -1, dtype=np.int64)
            mask = source_ids == source_id
            positions = np.nonzero(mask)[0].astype(np.int64, copy=False)
            source_cache_indices = cache_indices[mask]
            valid = (source_cache_indices >= 0) & (source_cache_indices < source_num_samples)
            mapping[source_cache_indices[valid]] = positions[valid]
            lookup[source_name] = mapping
        # The released index names its one source "source_0" while the data module tags samples with the dataset name ("mp16"):
        # without this alias no training query maps to its own index row and leave-one-out never excludes the query itself.
        if len(lookup) == 1:
            lookup.setdefault(str(self.manifest.get("retrieval_db_source", "mp16")), next(iter(lookup.values())))
        return lookup

    def to_device(self, device: torch.device | str) -> None:
        torch_device = torch.device(device)
        if self._device is not None and self._device == torch_device and self.image_embeddings is not None:
            return

        image_np = np.array(self._image_embeddings_mm, dtype=np.float16, copy=True, order="C")
        gps_np = np.array(self._gps_embeddings_mm, dtype=np.float16, copy=True, order="C")
        latlon_np = np.array(self._latlon_mm, dtype=np.float32, copy=True, order="C")

        self.image_embeddings = torch.from_numpy(image_np).to(
            torch_device,
            dtype=torch.float16,
            non_blocking=(torch_device.type == "cuda"),
        ).contiguous()
        self.gps_embeddings = torch.from_numpy(gps_np).to(
            torch_device,
            dtype=torch.float16,
            non_blocking=(torch_device.type == "cuda"),
        ).contiguous()
        if self._raw_image_embeddings_mm is not None:
            self.raw_image_embeddings = torch.empty(
                (self.num_samples, self.raw_image_embedding_dim),
                device=torch_device,
                dtype=torch.float16,
            )
            rows_per_chunk = _rows_per_chunk(
                self.raw_image_embedding_dim,
                target_bytes=_RAW_IMAGE_LOAD_CHUNK_BYTES,
            )
            for start in range(0, self.num_samples, rows_per_chunk):
                end = min(start + rows_per_chunk, self.num_samples)
                raw_chunk_np = np.array(
                    self._raw_image_embeddings_mm[start:end],
                    dtype=np.float32,
                    copy=True,
                    order="C",
                )
                if self._raw_image_scales_mm is not None:
                    raw_chunk_np *= np.asarray(
                        self._raw_image_scales_mm[start:end],
                        dtype=np.float32,
                    )[:, None]
                raw_chunk_np = _l2_normalize_numpy_rows(raw_chunk_np)
                raw_chunk_t = torch.from_numpy(raw_chunk_np).to(
                    torch_device,
                    dtype=torch.float16,
                    non_blocking=(torch_device.type == "cuda"),
                )
                self.raw_image_embeddings[start:end].copy_(
                    raw_chunk_t,
                    non_blocking=(torch_device.type == "cuda"),
                )
        else:
            self.raw_image_embeddings = None
        self.latlon_deg = torch.from_numpy(latlon_np).to(
            torch_device,
            dtype=torch.float32,
            non_blocking=(torch_device.type == "cuda"),
        ).contiguous()

        self._device = torch_device

    def gather_raw_image_embeddings(
        self,
        top_indices: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor | None:
        if self.raw_image_embeddings is None:
            return None

        index_device = self.raw_image_embeddings.device
        top_indices_local = top_indices.detach().to(device=index_device, dtype=torch.long)
        gathered = self.raw_image_embeddings[top_indices_local].to(dtype=dtype)
        return gathered.to(
            device,
            dtype=dtype,
            non_blocking=(device.type == "cuda"),
        )

    def query_indices_from_metadata(
        self,
        metadata_batch: Sequence[dict[str, Any]] | None,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if metadata_batch is None:
            return torch.full((0,), -1, dtype=torch.long, device=device)

        indices: list[int] = []
        for metadata in metadata_batch:
            if str(metadata.get("source_split", "")).lower() == "test":
                indices.append(-1)
                continue
            source_name = str(metadata.get("source", ""))
            cache_idx_raw = metadata.get("cache_idx", metadata.get("image_cache_idx"))
            if cache_idx_raw is None:
                indices.append(-1)
                continue
            try:
                cache_idx = int(cache_idx_raw)
            except (TypeError, ValueError):
                indices.append(-1)
                continue
            source_map = self._source_lookup.get(source_name)
            if source_map is None or cache_idx < 0 or cache_idx >= source_map.shape[0]:
                indices.append(-1)
                continue
            indices.append(int(source_map[cache_idx]))

        return torch.as_tensor(indices, dtype=torch.long, device=device)
