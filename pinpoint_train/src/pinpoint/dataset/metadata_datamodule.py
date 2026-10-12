"""Shared metadata-preserving DataModule utilities for training geolocation models."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import lightning as L
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from pinpoint.dataset.embed_data import (
    _build_single_dataset,
    infer_shared_embed_dim,
    resolve_data_selection,
)


_DATALOADER_SUPPORTS_IN_ORDER = "in_order" in inspect.signature(DataLoader).parameters


class SourceTaggedDataset(Dataset):
    """Wrap a dataset and inject a stable source label into returned metadata."""

    def __init__(self, base: Dataset, source_name: str):
        self.base = base
        self.source_name = str(source_name)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        if not isinstance(sample, (tuple, list)):
            raise TypeError(f"Expected dataset sample tuple/list, got {type(sample)}")
        if len(sample) < 2:
            raise ValueError(f"Unexpected sample size: {len(sample)}")
        if len(sample) == 2:
            emb, target = sample
            metadata = {"source": self.source_name}
            return emb, target, metadata

        emb, target, metadata = sample[0], sample[1], sample[2]
        if isinstance(metadata, dict):
            metadata["source"] = self.source_name
            return emb, target, metadata
        return emb, target, {"source": self.source_name}


def _metadata_collate_fn(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]],
    embedding_dtype: torch.dtype | None,
):
    inputs = torch.stack([item[0] for item in batch])
    if embedding_dtype is not None and inputs.dtype != embedding_dtype:
        inputs = inputs.to(embedding_dtype)
    targets = torch.stack([item[1] for item in batch])
    metadata = [item[2] if len(item) > 2 else {} for item in batch]
    return inputs, targets, metadata


class MetadataEmbedDataModule(L.LightningDataModule):
    """Build train/val loaders that preserve per-sample dataset identity metadata."""

    def __init__(
        self,
        *,
        cache_roots: dict[str, str],
        train_datasets: list[str],
        val_datasets: list[str],
        batch_size: int,
        num_workers: int,
        val_pct: int,
        embed_dtype: torch.dtype = torch.float32,
        train_shuffle: bool = True,
        prefetch_factor: int | None = None,
        train_mixing: str = "proportional",
    ):
        super().__init__()
        if train_mixing not in {"proportional", "uniform"}:
            raise ValueError("train_mixing must be one of: proportional, uniform")

        self.cache_roots = dict(cache_roots)
        self.train_datasets = list(train_datasets)
        self.val_datasets = list(val_datasets)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.val_pct = int(val_pct)
        self.embed_dtype = embed_dtype
        self.train_shuffle = bool(train_shuffle)
        self.prefetch_factor = prefetch_factor
        self.train_mixing = train_mixing

        self.train_ds = None
        self.val_ds = None
        self.train_sampler = None
        self._collate_fn = lambda batch: _metadata_collate_fn(batch, embedding_dtype=self.embed_dtype)

    def setup(self, stage=None):
        if self.train_ds is None:
            train_parts = []
            for name in self.train_datasets:
                ds = _build_single_dataset(
                    dataset_name=name,
                    split="train",
                    cache_roots=self.cache_roots,
                    val_pct=self.val_pct,
                    embedding_dtype=None,
                    return_metadata=True,
                )
                train_parts.append(SourceTaggedDataset(ds, source_name=name))
            self.train_ds, self.train_sampler = self._merge_train_parts(train_parts)

        if self.val_ds is None:
            val_parts = []
            for name in self.val_datasets:
                ds = _build_single_dataset(
                    dataset_name=name,
                    split="val",
                    cache_roots=self.cache_roots,
                    val_pct=self.val_pct,
                    embedding_dtype=None,
                    return_metadata=True,
                )
                val_parts.append(SourceTaggedDataset(ds, source_name=name))
            self.val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)

    def _merge_train_parts(self, datasets):
        if len(datasets) == 1:
            return datasets[0], None

        merged = ConcatDataset(datasets)
        if self.train_mixing == "proportional":
            return merged, None

        weights = torch.empty(len(merged), dtype=torch.double)
        start = 0
        n_parts = len(datasets)
        for ds in datasets:
            length = len(ds)
            if length <= 0:
                raise ValueError("Encountered empty dataset while building uniform sampler")
            weights[start : start + length] = 1.0 / (n_parts * length)
            start += length
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(merged), replacement=True)
        return merged, sampler

    def _loader_kwargs(self, worker_count):
        kwargs = {
            "num_workers": worker_count,
            "pin_memory": torch.cuda.is_available(),
        }
        if worker_count > 0:
            kwargs["persistent_workers"] = True
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return kwargs

    def _make_loader(self, dataset, worker_count, *, shuffle=False, sampler=None):
        loader_kwargs = self._loader_kwargs(worker_count)
        if shuffle and worker_count > 0 and _DATALOADER_SUPPORTS_IN_ORDER:
            loader_kwargs["in_order"] = False
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            collate_fn=self._collate_fn,
            **loader_kwargs,
        )

    def train_dataloader(self):
        return self._make_loader(
            self.train_ds,
            worker_count=self.num_workers,
            shuffle=(self.train_shuffle and self.train_sampler is None),
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return self._make_loader(
            self.val_ds,
            worker_count=min(2, self.num_workers),
        )


def build_metadata_datamodule_and_dim(
    args,
    *,
    default_train_mixing: str = "proportional",
    return_cache_context: bool = False,
):
    """Build the shared metadata-aware datamodule used by training entrypoints."""
    train_datasets, val_datasets, cache_roots = resolve_data_selection(args)
    embed_dim = infer_shared_embed_dim(
        sorted(set(train_datasets + val_datasets)),
        cache_roots,
    )
    datamodule = MetadataEmbedDataModule(
        cache_roots=cache_roots,
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_pct=args.val_pct,
        embed_dtype=torch.float32,
        train_shuffle=True,
        prefetch_factor=getattr(args, "prefetch_factor", None),
        train_mixing=getattr(args, "train_mixing", default_train_mixing),
    )
    if return_cache_context:
        return datamodule, embed_dim, train_datasets, val_datasets, cache_roots
    return datamodule, embed_dim, train_datasets, val_datasets
