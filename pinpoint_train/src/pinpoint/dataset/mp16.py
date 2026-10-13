"""MP16-Pro dataset utilities."""

import csv
import hashlib
import io
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info

_TAR_INDEX_CACHE = {}


def load_tar_index(index_path):
    """Cache the large tar index so train/val splits share one copy."""
    key = str(index_path)
    if key not in _TAR_INDEX_CACHE:
        with index_path.open("rb") as f:
            _TAR_INDEX_CACHE[key] = pickle.load(f)
    return _TAR_INDEX_CACHE[key]


def split_bucket(image_id):
    digest = hashlib.md5(image_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


def keep_for_split(image_id, split, val_pct):
    if split == "all":
        return True

    bucket = split_bucket(image_id)
    val_start = 100 - val_pct

    if split == "train":
        return bucket < val_start
    if split == "val":
        return bucket >= val_start
    raise ValueError(f"Unknown split '{split}'. Expected one of: all, train, val")


def read_image_bytes(image_id, tar_index, part_handles, chunk_size):
    tar_info = tar_index.get(image_id)
    if tar_info is None:
        return None

    offset = tar_info.offset_data
    size = tar_info.size
    part_idx = offset // chunk_size
    local_offset = offset % chunk_size

    if part_idx >= len(part_handles):
        return None

    if local_offset + size <= chunk_size:
        part_handles[part_idx].seek(local_offset)
        return part_handles[part_idx].read(size)

    if part_idx + 1 >= len(part_handles):
        return None

    bytes_from_first = chunk_size - local_offset
    part_handles[part_idx].seek(local_offset)
    first = part_handles[part_idx].read(bytes_from_first)
    part_handles[part_idx + 1].seek(0)
    second = part_handles[part_idx + 1].read(size - bytes_from_first)
    return first + second


class MP16ImageDataset(IterableDataset):
    """MP16-Pro image samples using tar shard offsets from tar_index.pkl."""

    def __init__(
        self,
        root,
        processor,
        split="train",
        seed=21,
        val_pct=2,
        shuffle_buffer=10_000,
    ):
        super().__init__()
        self.root = Path(root)
        self.processor = processor
        self.split = split
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.val_pct = val_pct

        if split not in {"all", "train", "val"}:
            raise ValueError(f"Unknown split '{split}'. Expected one of: all, train, val")
        if not (0 <= val_pct < 100):
            raise ValueError("--val-pct must be in [0, 99]")

        self.metadata_csv = self.root / "metadata" / "MP16_Pro_filtered.csv"
        self.index_pkl = self.root / "metadata" / "tar_index.pkl"
        self.parts = sorted(self.root.glob("mp-16-images[0-9][0-9]"))

        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Missing metadata CSV: {self.metadata_csv}")
        if not self.index_pkl.exists():
            raise FileNotFoundError(f"Missing tar index: {self.index_pkl}")
        if not self.parts:
            raise FileNotFoundError(
                f"No shard files found under {self.root} matching mp-16-images[0-9][0-9]"
            )

        self.chunk_size = self.parts[0].stat().st_size
        self.tar_index = load_tar_index(self.index_pkl)

    def _split_bucket(self, image_id):
        return split_bucket(image_id)

    def _keep_for_split(self, image_id):
        return keep_for_split(
            image_id,
            self.split,
            val_pct=self.val_pct,
        )

    def _read_image_bytes(self, image_id, part_handles):
        return read_image_bytes(
            image_id,
            tar_index=self.tar_index,
            part_handles=part_handles,
            chunk_size=self.chunk_size,
        )

    def _to_tensors(self, row_idx, image_id, lat, lon, part_handles):
        image_bytes = self._read_image_bytes(image_id, part_handles)
        if image_bytes is None:
            return None

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
        except Exception:
            return None

        target = torch.tensor([lat / 90.0, lon / 180.0], dtype=torch.float32)
        return (
            pixel_values,
            target,
            {
                "image_id": image_id,
                "row_idx": row_idx,
                "lat": lat,
                "lon": lon,
            },
        )

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        rng = np.random.default_rng(self.seed + worker_id)
        part_handles = [open(path, "rb") for path in self.parts]

        try:
            shuffle_enabled = self.split == "train" and self.shuffle_buffer > 0
            buffer = []

            with self.metadata_csv.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row_idx, row in enumerate(reader):
                    if row_idx % num_workers != worker_id:
                        continue

                    image_id = row.get("IMG_ID")
                    if image_id is None:
                        continue
                    if image_id not in self.tar_index:
                        continue
                    if not self._keep_for_split(image_id):
                        continue

                    try:
                        lat = float(row["LAT"])
                        lon = float(row["LON"])
                    except (KeyError, TypeError, ValueError):
                        continue

                    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                        continue

                    sample = (row_idx, image_id, lat, lon)
                    if not shuffle_enabled:
                        item = self._to_tensors(*sample, part_handles)
                        if item is not None:
                            yield item
                        continue

                    if len(buffer) < self.shuffle_buffer:
                        buffer.append(sample)
                        continue

                    chosen = int(rng.integers(0, len(buffer)))
                    out_sample = buffer[chosen]
                    buffer[chosen] = sample
                    item = self._to_tensors(*out_sample, part_handles)
                    if item is not None:
                        yield item

            if shuffle_enabled and buffer:
                rng.shuffle(buffer)
                for sample in buffer:
                    item = self._to_tensors(*sample, part_handles)
                    if item is not None:
                        yield item
        finally:
            for handle in part_handles:
                handle.close()


class MP16EmbedDataset(Dataset):
    """Map-style MP16 dataset backed by memory-mapped cached embeddings."""

    def __init__(
        self,
        cache_root,
        split="all",
        val_pct=None,
        embedding_dtype=torch.float32,
        return_metadata=True,
    ):
        super().__init__()
        self.cache_root = Path(cache_root)
        self.split = split
        self.embedding_dtype = embedding_dtype
        self.return_metadata = bool(return_metadata)

        if split not in {"all", "train", "val"}:
            raise ValueError(f"Unknown split '{split}'. Expected one of: all, train, val")

        self.manifest_name = "manifest.json"
        self.manifest_path = self.cache_root / self.manifest_name
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing cache manifest: {self.manifest_path}")
        with self.manifest_path.open("r") as f:
            self.manifest = json.load(f)

        self.num_samples = int(self.manifest.get("num_samples", 0))
        self.embedding_dim = int(self.manifest.get("embedding_dim", 0))
        if self.num_samples <= 0:
            raise ValueError(f"Invalid num_samples={self.num_samples} in {self.manifest_path}")
        if self.embedding_dim <= 0:
            raise ValueError(f"Invalid embedding_dim={self.embedding_dim} in {self.manifest_path}")

        manifest_val_pct = self.manifest.get("val_pct")
        if manifest_val_pct is None:
            manifest_val_pct = self.manifest.get("val_bucket_pct")
        if manifest_val_pct is None and "train_bucket_pct" in self.manifest:
            manifest_val_pct = 100 - int(self.manifest["train_bucket_pct"])
        if manifest_val_pct is None:
            manifest_val_pct = 2

        self.val_pct = int(manifest_val_pct) if val_pct is None else int(val_pct)
        if not (0 <= self.val_pct < 100):
            raise ValueError("--val-pct must be in [0, 99]")

        files = self.manifest.get("files", {})
        dtypes = self.manifest.get("dtypes", {})
        shapes = self.manifest.get("shapes", {})

        self.embeddings_path = self.cache_root / files.get("embeddings", "")
        self.latlon_deg_path = self.cache_root / files.get("latlon_deg", "")
        self.row_index_path = self.cache_root / files.get("row_index", "")
        image_ids_name = files.get("image_ids")
        self.image_ids_path = self.cache_root / image_ids_name if image_ids_name else None

        required_paths = [self.embeddings_path, self.latlon_deg_path]
        if self.return_metadata:
            required_paths.append(self.row_index_path)
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Missing cache file: {path}")

        self.embeddings_shape = tuple(int(v) for v in shapes.get("embeddings", []))
        self.latlon_shape = tuple(
            int(v) for v in shapes.get("latlon_deg", shapes.get("latlon_norm", []))
        )
        self.row_index_shape = tuple(int(v) for v in shapes.get("row_index", []))
        if self.embeddings_shape != (self.num_samples, self.embedding_dim):
            raise ValueError(
                f"Unexpected embeddings shape in manifest: {self.embeddings_shape}, "
                f"expected ({self.num_samples}, {self.embedding_dim})"
            )
        if self.latlon_shape != (self.num_samples, 2):
            raise ValueError(
                f"Unexpected latlon_deg shape in manifest: {self.latlon_shape}, "
                f"expected ({self.num_samples}, 2)"
            )
        if self.return_metadata and self.row_index_shape != (self.num_samples,):
            raise ValueError(
                f"Unexpected row_index shape in manifest: {self.row_index_shape}, "
                f"expected ({self.num_samples},)"
            )

        self.embeddings_np_dtype = np.dtype(dtypes.get("embeddings", "float32"))
        self.row_index_np_dtype = np.dtype(dtypes.get("row_index", "int64"))

        self._indices = None
        self._selected_image_ids = None
        self._build_index()

        self._embeddings_mm = None
        self._latlon_deg_mm = None
        self._row_index_mm = None
        self._embeddings_t = None
        self._latlon_deg_t = None
        self.image_embedding_dim = int(self.embedding_dim)

    def _build_index(self):
        if self.split == "all" and not self.return_metadata:
            self._indices = np.arange(self.num_samples, dtype=np.int64)
            self._selected_image_ids = None
            return

        if self.image_ids_path is None or not self.image_ids_path.exists():
            raise FileNotFoundError(
                "Split and metadata lookups require image_ids.txt in the cache directory."
            )

        with self.image_ids_path.open("r") as f:
            all_image_ids = [line.rstrip("\n") for line in f]
        if len(all_image_ids) != self.num_samples:
            raise ValueError(
                f"image_ids length mismatch: got {len(all_image_ids)}, expected {self.num_samples}"
            )

        if self.split == "all":
            self._indices = np.arange(self.num_samples, dtype=np.int64)
            self._selected_image_ids = all_image_ids if self.return_metadata else None
            return

        keep_rows = []
        keep_image_ids = [] if self.return_metadata else None
        for row_idx, image_id in enumerate(all_image_ids):
            if keep_for_split(
                image_id,
                split=self.split,
                val_pct=self.val_pct,
            ):
                keep_rows.append(row_idx)
                if keep_image_ids is not None:
                    keep_image_ids.append(image_id)

        self._indices = np.asarray(keep_rows, dtype=np.int64)
        self._selected_image_ids = keep_image_ids
        # cross-fitting (geo_search_env lesson 75): train on one photographer half only; the retrieval index keeps every photo
        mask_path = os.environ.get("PINPOINT_TRAIN_ROW_MASK")
        if mask_path and self.split == "train":
            keep = np.load(mask_path)[self._indices]
            self._indices = self._indices[keep]
            if self._selected_image_ids is not None:
                self._selected_image_ids = [i for i, k in zip(self._selected_image_ids, keep) if k]
            print(f"PINPOINT_TRAIN_ROW_MASK {mask_path}: {len(self._indices)} training rows")

    def _ensure_memmaps(self):
        if self._embeddings_mm is not None:
            return

        self._embeddings_mm = np.memmap(
            self.embeddings_path,
            dtype=self.embeddings_np_dtype,
            mode="c",
            shape=self.embeddings_shape,
        )
        self._embeddings_t = torch.from_numpy(self._embeddings_mm)
        self._latlon_deg_mm = np.memmap(
            self.latlon_deg_path,
            dtype=np.float32,
            mode="c",
            shape=(self.num_samples, 2),
        )
        self._latlon_deg_t = torch.from_numpy(self._latlon_deg_mm)
        if self.return_metadata:
            self._row_index_mm = np.memmap(
                self.row_index_path,
                dtype=self.row_index_np_dtype,
                mode="r",
                shape=(self.num_samples,),
            )

    def __len__(self):
        return int(self._indices.shape[0])

    def __getitem__(self, idx):
        self._ensure_memmaps()
        row = int(self._indices[idx])

        embedding = self._embeddings_t[row]
        if self.embedding_dtype is not None and embedding.dtype != self.embedding_dtype:
            embedding = embedding.to(self.embedding_dtype)

        target = self._latlon_deg_t[row]
        if not self.return_metadata:
            return embedding, target

        lat_deg = float(target[0].item())
        lon_deg = float(target[1].item())
        image_id = self._selected_image_ids[idx]
        metadata = {
            "image_id": image_id,
            "cache_idx": row,
            "row_idx": int(self._row_index_mm[row]),
            "lat": lat_deg,
            "lon": lon_deg,
        }
        return embedding, target, metadata
