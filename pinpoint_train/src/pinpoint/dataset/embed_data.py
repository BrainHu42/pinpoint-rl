"""Shared embedding-cache dataset mixing utilities for training scripts."""

from __future__ import annotations

import os

from pinpoint.dataset.mp16 import MP16EmbedDataset
from pinpoint.dataset.osv5m import OSVEmbedDataset
from pinpoint.utils import infer_embed_dim, resolve_backbone_name

_SUPPORTED_DATASETS = ("mp16", "osv5m")
_DEFAULT_BACKBONE_NAME = resolve_backbone_name()
_DEFAULT_DATA_ROOT = os.environ.get("PINPOINT_DATA_ROOT", "/data/pinpoint")
_DEFAULT_MP16_EMBED_ROOT = os.environ.get("PINPOINT_MP16_EMBED_ROOT", f"{_DEFAULT_DATA_ROOT}/mp16-embed")
_DEFAULT_OSV5M_EMBED_ROOT = os.environ.get("PINPOINT_OSV5M_EMBED_ROOT", f"{_DEFAULT_DATA_ROOT}/osv5m-embed")


def add_embed_data_args(parser):
    """Add shared embedding-cache CLI args used by train scripts."""
    parser.add_argument(
        "--train-datasets",
        type=str,
        default="mp16",
        help="Comma-separated list from {mp16,osv5m}; e.g. mp16 or mp16,osv5m",
    )
    parser.add_argument(
        "--val-datasets",
        type=str,
        default=None,
        help="Optional comma-separated list. Defaults to --train-datasets.",
    )
    parser.add_argument(
        "--mp16-cache-root",
        type=str,
        default=f"{_DEFAULT_MP16_EMBED_ROOT}/{_DEFAULT_BACKBONE_NAME}",
    )
    parser.add_argument(
        "--osv5m-cache-root",
        type=str,
        default=f"{_DEFAULT_OSV5M_EMBED_ROOT}/{_DEFAULT_BACKBONE_NAME}",
    )
    parser.add_argument(
        "--train-mixing",
        choices=("proportional", "uniform"),
        default="proportional",
        help=(
            "When multiple train datasets are used: proportional samples by dataset size, "
            "uniform samples each dataset equally."
        ),
    )
    parser.add_argument("--val-pct", type=int, default=1)


def parse_dataset_spec(spec: str, arg_name: str) -> list[str]:
    names = [part.strip().lower() for part in spec.split(",") if part.strip()]
    if not names:
        raise ValueError(f"{arg_name} must include at least one dataset: {', '.join(_SUPPORTED_DATASETS)}")

    deduped = []
    seen = set()
    for name in names:
        if name not in _SUPPORTED_DATASETS:
            raise ValueError(
                f"Unknown dataset '{name}' in {arg_name}. "
                f"Expected one of: {', '.join(_SUPPORTED_DATASETS)}"
            )
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _cache_root_for(name: str, cache_roots: dict[str, str]) -> str:
    root = cache_roots.get(name)
    if not root:
        raise ValueError(f"Missing cache root for dataset '{name}'")
    return root


def _build_single_dataset(
    dataset_name: str,
    split: str,
    cache_roots: dict[str, str],
    val_pct: int,
    embedding_dtype,
    return_metadata: bool,
):
    if dataset_name == "mp16":
        return MP16EmbedDataset(
            cache_root=_cache_root_for(dataset_name, cache_roots),
            split=split,
            val_pct=val_pct,
            embedding_dtype=embedding_dtype,
            return_metadata=return_metadata,
        )
    if dataset_name == "osv5m":
        cache_root = _cache_root_for(dataset_name, cache_roots)
        return OSVEmbedDataset(
            cache_root=cache_root,
            split=split,
            val_pct=val_pct,
            embedding_dtype=embedding_dtype,
            return_metadata=return_metadata,
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def infer_shared_embed_dim(
    dataset_names: list[str],
    cache_roots: dict[str, str],
) -> int:
    """Infer and validate a single embedding dim across all selected datasets."""
    embed_dims = {}
    for name in dataset_names:
        if name == "mp16":
            embed_dims[name] = infer_embed_dim(
                _cache_root_for("mp16", cache_roots),
                manifest_name="manifest.json",
            )
            continue

        cache_root = _cache_root_for(name, cache_roots)
        embed_dims[name] = infer_embed_dim(cache_root, manifest_name="manifest.json")

    unique_dims = sorted(set(embed_dims.values()))
    if len(unique_dims) != 1:
        details = ", ".join(f"{name}={dim}" for name, dim in sorted(embed_dims.items()))
        raise ValueError(
            "Selected datasets do not share one embedding_dim. "
            f"Got: {details}. Use compatible caches/models."
        )
    return unique_dims[0]


def resolve_data_selection(args) -> tuple[list[str], list[str], dict[str, str]]:
    train_datasets = parse_dataset_spec(args.train_datasets, "--train-datasets")
    if args.val_datasets:
        val_datasets = parse_dataset_spec(args.val_datasets, "--val-datasets")
    else:
        val_datasets = list(train_datasets)
    cache_roots = {
        "mp16": args.mp16_cache_root,
        "osv5m": args.osv5m_cache_root,
    }
    return train_datasets, val_datasets, cache_roots
