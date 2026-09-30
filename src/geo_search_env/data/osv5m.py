"""Lazy, joined access to OSV-5M images and cached SigLIP embeddings."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OSV5M_ROOT = Path("/data/hf/datasets/osv5m")
DEFAULT_OSV5M_EMBEDDING_ROOT = Path(
    "/data/pinpoint/osv5m-embed/siglip2-giant-opt-patch16-384"
)
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


@dataclass(frozen=True, slots=True, eq=False)
class OSV5MSample:
    """One cache-aligned OSV-5M record with both visual representations."""

    image_id: str
    cache_index: int
    source_row_index: int
    latitude: float
    longitude: float
    image_path: Path
    embedding: np.ndarray
    image_bytes: bytes | None = None


class _IndexedTextLines:
    """Bounded-memory random access to a large newline-delimited ID file."""

    def __init__(self, path: Path, expected_lines: int) -> None:
        self.path = path
        self.expected_lines = expected_lines
        self._offsets: array[int] | None = None

    def _build(self) -> array[int]:
        offsets = array("Q")
        offset = 0
        with self.path.open("rb") as stream:
            for line in stream:
                offsets.append(offset)
                offset += len(line)
        if len(offsets) != self.expected_lines:
            raise ValueError(
                f"image ID count mismatch: got {len(offsets)}, expected {self.expected_lines}"
            )
        self._offsets = offsets
        return offsets

    def get(self, index: int) -> str:
        offsets = self._offsets if self._offsets is not None else self._build()
        with self.path.open("rb") as stream:
            stream.seek(offsets[index])
            value = stream.readline().rstrip(b"\r\n")
        try:
            image_id = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("image_ids contains invalid UTF-8") from error
        if not image_id:
            raise ValueError(f"image_ids contains an empty ID at cache row {index}")
        if Path(image_id).name != image_id or "/" in image_id or "\\" in image_id:
            raise ValueError(f"image_ids contains an unsafe ID at cache row {index}")
        return image_id


class OSV5MDataset:
    """Map-style OSV-5M access backed by read-only NumPy memory maps.

    The cache row is the canonical index. ``image_ids.txt``, coordinates,
    source CSV row indices, and embeddings are required to have identical row
    order. Raw image paths are resolved lazily across the extracted source
    split's shard directories.
    """

    def __init__(
        self,
        raw_root: Path | str = DEFAULT_OSV5M_ROOT,
        embedding_root: Path | str = DEFAULT_OSV5M_EMBEDDING_ROOT,
        *,
        load_image_bytes: bool = False,
    ) -> None:
        self.raw_root = Path(raw_root).resolve()
        self.embedding_root = Path(embedding_root).resolve()
        self.load_image_bytes = bool(load_image_bytes)
        self.manifest_path = self.embedding_root / "manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"missing embedding manifest: {self.manifest_path}")
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid embedding manifest: {self.manifest_path}") from error
        if not isinstance(manifest, dict):
            raise ValueError("embedding manifest must be a JSON object")
        self.manifest: dict[str, Any] = manifest
        if manifest.get("format") != "flat_binary_memmap_v1":
            raise ValueError("unsupported embedding cache format")

        self.num_samples = self._positive_integer(manifest.get("num_samples"), "num_samples")
        self.embedding_dim = self._positive_integer(
            manifest.get("embedding_dim"), "embedding_dim"
        )
        self.source_split = manifest.get("source_split")
        if self.source_split not in {"train", "test"}:
            raise ValueError("embedding manifest source_split must be train or test")

        files = manifest.get("files")
        dtypes = manifest.get("dtypes")
        shapes = manifest.get("shapes")
        if not all(isinstance(value, dict) for value in (files, dtypes, shapes)):
            raise ValueError("embedding manifest files, dtypes, and shapes must be objects")
        try:
            self._embedding_dtype = np.dtype(dtypes.get("embeddings", ""))
            self._row_index_dtype = np.dtype(dtypes.get("row_index", ""))
        except TypeError as error:
            raise ValueError("embedding manifest contains an invalid dtype") from error
        if self._embedding_dtype not in (np.dtype("float16"), np.dtype("float32")):
            raise ValueError("embedding cache must use float16 or float32")
        if self._row_index_dtype != np.dtype("int64"):
            raise ValueError("row index cache must use int64")

        self._embedding_shape = self._shape(shapes.get("embeddings"), "embeddings")
        self._coordinate_shape = self._shape(shapes.get("latlon_deg"), "latlon_deg")
        self._row_index_shape = self._shape(shapes.get("row_index"), "row_index")
        if self._embedding_shape != (self.num_samples, self.embedding_dim):
            raise ValueError("embedding shape does not match manifest dimensions")
        if self._coordinate_shape != (self.num_samples, 2):
            raise ValueError("latlon_deg shape must be (num_samples, 2)")
        if self._row_index_shape != (self.num_samples,):
            raise ValueError("row_index shape must be (num_samples,)")

        self.embeddings_path = self._cache_file(files.get("embeddings"), "embeddings")
        self.coordinates_path = self._cache_file(files.get("latlon_deg"), "latlon_deg")
        self.row_indices_path = self._cache_file(files.get("row_index"), "row_index")
        self.image_ids_path = self._cache_file(files.get("image_ids"), "image_ids")
        self._validate_binary_size(
            self.embeddings_path, self._embedding_shape, self._embedding_dtype
        )
        self._validate_binary_size(
            self.coordinates_path, self._coordinate_shape, np.dtype("float32")
        )
        self._validate_binary_size(
            self.row_indices_path, self._row_index_shape, self._row_index_dtype
        )

        self.images_root = self.raw_root / "images" / self.source_split
        if not self.images_root.is_dir():
            raise FileNotFoundError(f"missing extracted image directory: {self.images_root}")
        self._shards = tuple(sorted(path for path in self.images_root.iterdir() if path.is_dir()))
        if not self._shards:
            raise FileNotFoundError(f"no image shards under: {self.images_root}")

        self._image_ids = _IndexedTextLines(self.image_ids_path, self.num_samples)
        self._image_path_cache: dict[str, Path] = {}
        self._embeddings: np.memmap | None = None
        self._coordinates: np.memmap | None = None
        self._row_indices: np.memmap | None = None

    @staticmethod
    def _positive_integer(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"embedding manifest {name} must be a positive integer")
        return value

    @staticmethod
    def _shape(value: Any, name: str) -> tuple[int, ...]:
        if not isinstance(value, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in value
        ):
            raise ValueError(f"embedding manifest {name} shape is invalid")
        return tuple(value)

    def _cache_file(self, value: Any, name: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"embedding manifest has no {name} filename")
        path = (self.embedding_root / value).resolve()
        try:
            path.relative_to(self.embedding_root)
        except ValueError as error:
            raise ValueError(f"embedding manifest {name} escapes the cache root") from error
        if not path.is_file():
            raise FileNotFoundError(f"missing embedding cache file: {path}")
        return path

    @staticmethod
    def _validate_binary_size(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> None:
        expected = int(np.prod(shape)) * dtype.itemsize
        actual = path.stat().st_size
        if actual != expected:
            raise ValueError(
                f"cache file size mismatch for {path.name}: got {actual}, expected {expected}"
            )

    def _normalize_index(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer")
        normalized = index + self.num_samples if index < 0 else index
        if not 0 <= normalized < self.num_samples:
            raise IndexError("dataset index out of range")
        return normalized

    def _ensure_memmaps(self) -> None:
        if self._embeddings is not None:
            return
        self._embeddings = np.memmap(
            self.embeddings_path,
            dtype=self._embedding_dtype,
            mode="r",
            shape=self._embedding_shape,
        )
        self._coordinates = np.memmap(
            self.coordinates_path,
            dtype=np.float32,
            mode="r",
            shape=self._coordinate_shape,
        )
        self._row_indices = np.memmap(
            self.row_indices_path,
            dtype=self._row_index_dtype,
            mode="r",
            shape=self._row_index_shape,
        )

    def __len__(self) -> int:
        return self.num_samples

    def image_id_at(self, index: int) -> str:
        return self._image_ids.get(self._normalize_index(index))

    def image_path_for_id(self, image_id: str) -> Path:
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError("image_id must be a nonempty string")
        cached = self._image_path_cache.get(image_id)
        if cached is not None:
            return cached
        for shard in self._shards:
            for suffix in _IMAGE_SUFFIXES:
                candidate = shard / f"{image_id}{suffix}"
                if candidate.is_file():
                    self._image_path_cache[image_id] = candidate
                    return candidate
        raise FileNotFoundError(f"raw OSV-5M image is missing for ID {image_id!r}")

    def embedding_at(self, index: int) -> np.ndarray:
        normalized = self._normalize_index(index)
        self._ensure_memmaps()
        assert self._embeddings is not None
        return self._embeddings[normalized]

    def embeddings_at(self, indices: list[int] | tuple[int, ...] | np.ndarray) -> np.ndarray:
        """Read a batch of cache rows while preserving the requested order."""

        normalized = [self._normalize_index(int(index)) for index in indices]
        self._ensure_memmaps()
        assert self._embeddings is not None
        return np.asarray(self._embeddings[normalized])

    def __getitem__(self, index: int) -> OSV5MSample:
        normalized = self._normalize_index(index)
        self._ensure_memmaps()
        assert self._embeddings is not None
        assert self._coordinates is not None
        assert self._row_indices is not None
        image_id = self._image_ids.get(normalized)
        image_path = self.image_path_for_id(image_id)
        image_bytes = image_path.read_bytes() if self.load_image_bytes else None
        coordinate = self._coordinates[normalized]
        return OSV5MSample(
            image_id=image_id,
            cache_index=normalized,
            source_row_index=int(self._row_indices[normalized]),
            latitude=float(coordinate[0]),
            longitude=float(coordinate[1]),
            image_path=image_path,
            embedding=self._embeddings[normalized],
            image_bytes=image_bytes,
        )
