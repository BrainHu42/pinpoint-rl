from __future__ import annotations

from importlib import import_module
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any, Sequence

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pinpoint.attention_reranker.retrieval_index import (
    AttentionRetrievalIndex,
    _normalize_retrieval_db_source,
    ensure_attention_retrieval_index,
)
from pinpoint.utils import _resolve_device, _resolve_inference_dtype, geodesic_km


_SUPPORTED_HALF_DTYPES = (torch.float16, torch.bfloat16)
_OSV5M_LEAVE_ONE_OUT_EXCLUSION_KM = 1.0
_RETRIEVAL_BASELINE_IMAGE_WEIGHT = 0.7
_RETRIEVAL_BASELINE_GPS_WEIGHT = 0.3
_CONTRASTIVE_MODEL_MODULE = "pinpoint.contrastive_retrieval.lightning_module"


def _resolve_contrastive_geo_module_class():
    module = import_module(_CONTRASTIVE_MODEL_MODULE)
    return getattr(module, "GeoLightningModule")


def _latlon_deg_to_unit_xyz_np(latlon_deg: np.ndarray) -> np.ndarray:
    latlon = np.asarray(latlon_deg, dtype=np.float64)
    lat_rad = np.deg2rad(latlon[..., 0])
    lon_rad = np.deg2rad(latlon[..., 1])
    cos_lat = np.cos(lat_rad)

    xyz = np.empty(latlon.shape[:-1] + (3,), dtype=np.float64)
    xyz[..., 0] = cos_lat * np.cos(lon_rad)
    xyz[..., 1] = cos_lat * np.sin(lon_rad)
    xyz[..., 2] = np.sin(lat_rad)
    return xyz


def _chord_to_km_np(chord_dist: np.ndarray) -> np.ndarray:
    clamped = np.clip(np.asarray(chord_dist, dtype=np.float64) * 0.5, 0.0, 1.0)
    central_angle_rad = 2.0 * np.arcsin(clamped)
    return central_angle_rad * 6371.0088


class _OsvSupportCache:
    """CPU-side nearest-neighbor lookup over cached OSV image embeddings."""

    def __init__(self, cache_root: str | Path) -> None:
        try:
            from scipy.spatial import cKDTree  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "OSV support tokens require scipy for nearest-neighbor lookup. Install dependencies with `uv sync`."
            ) from exc

        self.cache_root = Path(cache_root).expanduser().resolve()
        manifest_path = self.cache_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing OSV cache manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", {})
        shapes = manifest.get("shapes", {})

        embedding_name = files.get("embeddings")
        latlon_name = files.get("latlon_deg")
        if not embedding_name or not latlon_name:
            raise ValueError(f"{manifest_path}: manifest must define files.embeddings and files.latlon_deg")

        embedding_shape = tuple(int(v) for v in shapes.get("embeddings", []))
        latlon_shape = tuple(int(v) for v in shapes.get("latlon_deg", []))
        if len(embedding_shape) != 2 or embedding_shape[0] <= 0 or embedding_shape[1] <= 0:
            raise ValueError(f"{manifest_path}: invalid shapes.embeddings={embedding_shape}")
        if latlon_shape != (embedding_shape[0], 2):
            raise ValueError(
                f"{manifest_path}: invalid shapes.latlon_deg={latlon_shape}; expected ({embedding_shape[0]}, 2)"
            )

        self.embedding_dim = int(embedding_shape[1])
        self.embedding_space = str(manifest.get("embedding_space", "backbone")).strip().lower()
        if self.embedding_space not in {"backbone", "contrastive_osv"}:
            raise ValueError(f"{manifest_path}: unsupported embedding_space={self.embedding_space!r}")
        self.image_embeddings_mm = np.memmap(
            self.cache_root / str(embedding_name),
            dtype=np.float16,
            mode="r",
            shape=embedding_shape,
        )
        self.latlon_mm = np.memmap(
            self.cache_root / str(latlon_name),
            dtype=np.float32,
            mode="r",
            shape=latlon_shape,
        )
        self.tree = cKDTree(
            _latlon_deg_to_unit_xyz_np(np.asarray(self.latlon_mm)),
            leafsize=64,
            compact_nodes=True,
            balanced_tree=True,
        )

    def query(
        self,
        latlon_deg: np.ndarray,
        *,
        neighbors_k: int,
        radius_km: float,
        workers: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if neighbors_k <= 0:
            raise ValueError("neighbors_k must be > 0")
        if radius_km <= 0.0:
            raise ValueError("radius_km must be > 0")

        query_xyz = _latlon_deg_to_unit_xyz_np(latlon_deg)
        chord_dist, neighbor_idx = self.tree.query(
            query_xyz,
            k=int(neighbors_k),
            workers=int(workers),
        )
        chord_dist = np.asarray(chord_dist, dtype=np.float64)
        neighbor_idx = np.asarray(neighbor_idx, dtype=np.int64)
        if chord_dist.ndim == 1:
            chord_dist = chord_dist[:, None]
            neighbor_idx = neighbor_idx[:, None]
        distance_km = _chord_to_km_np(chord_dist).astype(np.float32, copy=False)
        valid_mask = np.isfinite(distance_km) & (distance_km <= float(radius_km))
        neighbor_idx = np.where(valid_mask, neighbor_idx, -1)
        return neighbor_idx, distance_km, valid_mask


class AttentionInterpolateLightningModule(L.LightningModule):
    """Top-N geolocation reranker over retrieval-index candidates."""

    _OPTIONAL_STATE_DICT_PREFIXES = (
        "image_proj.",
        "query_proj.",
        "candidate_image_fused_proj.",
        "gps_fused_proj.",
        "raw_image_fused_proj.",
        "image_feature_gate.",
        "gps_feature_film.",
        "raw_image_feature_film.",
        "candidate_image_metadata_proj.",
        "raw_image_metadata_proj.",
        "gps_metadata_proj.",
    )

    _RETRIEVAL_FEATURE_NAMES = (
        "image_similarity",
        "raw_image_similarity",
        "gps_similarity",
        "rank_image",
        "rank_raw_image",
        "rank_gps",
        "in_image_topk",
        "in_raw_image_topk",
        "in_gps_topk",
    )
    _RETRIEVAL_FEATURE_GROUPS = {
        "image": (0, 3, 6),
        "raw_image": (1, 4, 7),
        "gps": (2, 5, 8),
    }
    _RETRIEVAL_FEATURE_IMAGE_IDX = _RETRIEVAL_FEATURE_GROUPS["image"]
    _RETRIEVAL_FEATURE_RAW_IMAGE_IDX = _RETRIEVAL_FEATURE_GROUPS["raw_image"]
    _RETRIEVAL_FEATURE_GPS_IDX = _RETRIEVAL_FEATURE_GROUPS["gps"]
    _RETRIEVAL_FEATURE_DIM = len(_RETRIEVAL_FEATURE_NAMES)
    _RETRIEVAL_FEATURE_GROUP_DIM = 3

    def __init__(
        self,
        embed_dim: int,
        contrastive_checkpoint_path: str,
        retrieval_index_dir: str,
        lr: float,
        weight_decay: float,
        total_steps: int,
        warmup_steps: int = 0,
        retrieval_chunk_size: int = 65536,
        inference_device: str = "cuda",
        inference_dtype: str = "bfloat16",
        contrastive_checkpoint_strict: bool = True,
        retrieval_db_source: str = "mp16",
        reranker_hidden_dim: int = 512,
        reranker_num_heads: int = 8,
        reranker_num_layers: int = 4,
        reranker_ffn_dim: int = 2048,
        reranker_dropout: float = 0.1,
        reranker_sequence_chunk_size: int | None = None,
        use_raw_image_features: bool = False,
        use_retrieval_features: bool = True,
        use_osv_support_token: bool = False,
        osv_support_cache_root: str | None = None,
        osv_support_neighbors_k: int = 4,
        osv_support_radius_km: float = 1.0,
        num_neighbors_image: int = 32,
        num_neighbors_raw_image: int = 16,
        num_neighbors_gps: int = 16,
        modality_pool_multiplier: int = 1,
        margin_loss_weight: float = 1.0,
        margin_loss_margin: float = 0.2,
        **unused_checkpoint_hparams: Any,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("unused_checkpoint_hparams",))

        if embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if retrieval_chunk_size <= 0:
            raise ValueError("retrieval_chunk_size must be > 0")
        if reranker_hidden_dim <= 0:
            raise ValueError("reranker_hidden_dim must be > 0")
        if reranker_num_heads <= 0:
            raise ValueError("reranker_num_heads must be > 0")
        if reranker_num_layers <= 0:
            raise ValueError("reranker_num_layers must be > 0")
        if reranker_ffn_dim <= 0:
            raise ValueError("reranker_ffn_dim must be > 0")
        if not (0.0 <= reranker_dropout < 1.0):
            raise ValueError("reranker_dropout must be in [0, 1)")
        if reranker_sequence_chunk_size is not None and reranker_sequence_chunk_size <= 0:
            raise ValueError("reranker_sequence_chunk_size must be > 0 when provided")
        if osv_support_neighbors_k <= 0:
            raise ValueError("osv_support_neighbors_k must be > 0")
        if osv_support_radius_km <= 0.0:
            raise ValueError("osv_support_radius_km must be > 0")
        if num_neighbors_image < 0:
            raise ValueError("num_neighbors_image must be >= 0")
        if num_neighbors_raw_image < 0:
            raise ValueError("num_neighbors_raw_image must be >= 0")
        if num_neighbors_gps < 0:
            raise ValueError("num_neighbors_gps must be >= 0")
        if modality_pool_multiplier <= 0:
            raise ValueError("modality_pool_multiplier must be > 0")
        if margin_loss_weight < 0.0:
            raise ValueError("margin_loss_weight must be >= 0")
        if margin_loss_margin <= 0.0:
            raise ValueError("margin_loss_margin must be > 0")
        if margin_loss_weight == 0.0:
            raise ValueError("margin_loss_weight must be > 0")

        self.embed_dim = int(embed_dim)
        self.retrieval_chunk_size = int(retrieval_chunk_size)

        self.use_raw_image_features = bool(use_raw_image_features)
        self.use_retrieval_features = bool(use_retrieval_features)
        self.use_osv_support_token = bool(use_osv_support_token)
        self.osv_support_neighbors_k = int(osv_support_neighbors_k)
        self.osv_support_radius_km = float(osv_support_radius_km)
        self.reranker_sequence_chunk_size = (
            int(reranker_sequence_chunk_size) if reranker_sequence_chunk_size is not None else None
        )
        self.num_neighbors_image = int(num_neighbors_image)
        self.num_neighbors_raw_image = int(num_neighbors_raw_image)
        self.num_neighbors_gps = int(num_neighbors_gps)
        self.modality_pool_multiplier = int(modality_pool_multiplier)
        self.margin_loss_weight = float(margin_loss_weight)
        self.margin_loss_margin = float(margin_loss_margin)

        if self.num_neighbors_image <= 0 and self.num_neighbors_raw_image <= 0 and self.num_neighbors_gps <= 0:
            raise ValueError("At least one modality candidate budget must be > 0")

        self.inference_device = _resolve_device(inference_device)
        self.inference_dtype = _resolve_inference_dtype(inference_dtype, self.inference_device)
        self.retrieval_db_source = _normalize_retrieval_db_source(retrieval_db_source)
        self.leave_one_out_exclusion_km = (
            _OSV5M_LEAVE_ONE_OUT_EXCLUSION_KM if self.retrieval_db_source == "osv5m" else 0.0
        )

        contrastive_geo_module_cls = _resolve_contrastive_geo_module_class()
        self.contrastive_model = contrastive_geo_module_cls.load_from_checkpoint(
            contrastive_checkpoint_path,
            map_location="cpu",
            strict=bool(contrastive_checkpoint_strict),
        )
        self.contrastive_model.eval()
        for parameter in self.contrastive_model.parameters():
            parameter.requires_grad = False

        self.image_embed_dim = int(getattr(self.contrastive_model, "image_embed_dim", 0))
        self.contrastive_embed_dim = int(getattr(self.contrastive_model, "embed_dim", 0))
        self.output_dim = int(getattr(self.contrastive_model, "output_dim", 0))
        if self.image_embed_dim <= 0:
            raise RuntimeError("Contrastive checkpoint does not expose image_embed_dim")
        if self.output_dim <= 0:
            raise RuntimeError("Contrastive checkpoint does not expose output_dim")
        if self.use_osv_support_token and self.image_embed_dim <= 0:
            raise RuntimeError("OSV support token requires a contrastive model with image embedding support")

        if self.embed_dim not in {self.image_embed_dim, self.contrastive_embed_dim}:
            expected = (
                f"{self.image_embed_dim} (image-only) or {self.contrastive_embed_dim} (image+text concat)"
            )
            raise ValueError(f"Embedding dim mismatch: expected {expected}, got {self.embed_dim}")

        self._osv_support_cache: _OsvSupportCache | None = None
        self._osv_support_row_cache_indices: np.ndarray | None = None
        self._osv_support_row_cache_distance_km: np.ndarray | None = None
        self._osv_support_row_cache_ready: np.ndarray | None = None
        if self.use_osv_support_token:
            if osv_support_cache_root is None:
                raise ValueError("use_osv_support_token=True requires osv_support_cache_root")
            self._osv_support_cache = _OsvSupportCache(osv_support_cache_root)
            expected_osv_dim = (
                self.output_dim
                if self._osv_support_cache.embedding_space == "contrastive_osv"
                else self.image_embed_dim
            )
            if self._osv_support_cache.embedding_dim != expected_osv_dim:
                raise ValueError(
                    "OSV support cache embedding dim does not match its embedding space: "
                    f"{self._osv_support_cache.embedding_dim} vs {expected_osv_dim}"
                )

        requested_index_dir = Path(retrieval_index_dir).expanduser().resolve()
        try:
            self.retrieval_index = AttentionRetrievalIndex(requested_index_dir)
        except FileNotFoundError:
            base_index_dir = (
                requested_index_dir.parent
                if requested_index_dir.parent.name == "attention_retrieval_index"
                else requested_index_dir
            )
            print(
                "Attention retrieval index missing at "
                f"{requested_index_dir}. Attempting to build it now under {base_index_dir}..."
            )
            resolved_index_dir = ensure_attention_retrieval_index(
                contrastive_checkpoint_path=contrastive_checkpoint_path,
                cache_roots=None,
                index_cache_dir=str(base_index_dir),
                split="train",
                val_pct=None,
                build_batch_size=8192,
                build_device=str(self.inference_device),
                build_dtype="auto",
                checkpoint_strict=bool(contrastive_checkpoint_strict),
                retrieval_db_source=self.retrieval_db_source,
            )
            self.retrieval_index = AttentionRetrievalIndex(resolved_index_dir)
        if self.retrieval_index.embedding_dim != self.output_dim:
            raise ValueError(
                f"Retrieval index embedding dim ({self.retrieval_index.embedding_dim}) does not match "
                f"contrastive output_dim ({self.output_dim})"
            )
        if self.num_neighbors_raw_image > 0 and self.retrieval_index.raw_image_embeddings_path is None:
            raise ValueError("per_modality_union retrieval with raw-image candidates requires raw cached image embeddings")
        if self.use_osv_support_token:
            retrieval_rows = int(self.retrieval_index.num_samples)
            self._osv_support_row_cache_indices = np.full(
                (retrieval_rows, self.osv_support_neighbors_k),
                -1,
                dtype=np.int32,
            )
            self._osv_support_row_cache_distance_km = np.full(
                (retrieval_rows, self.osv_support_neighbors_k),
                float(self.osv_support_radius_km),
                dtype=np.float16,
            )
            self._osv_support_row_cache_ready = np.zeros((retrieval_rows,), dtype=bool)

        hidden_dim = int(reranker_hidden_dim)
        if hidden_dim % int(reranker_num_heads) != 0:
            raise ValueError("reranker_hidden_dim must be divisible by reranker_num_heads")
        self.hidden_dim = hidden_dim

        self.query_proj = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, hidden_dim),
            nn.GELU(),
        )
        self.candidate_image_fused_proj = nn.Sequential(
            nn.LayerNorm(self.output_dim + self._RETRIEVAL_FEATURE_GROUP_DIM),
            nn.Linear(self.output_dim + self._RETRIEVAL_FEATURE_GROUP_DIM, hidden_dim),
            nn.GELU(),
        )
        self.candidate_image_metadata_proj = nn.Sequential(
            nn.LayerNorm(self._RETRIEVAL_FEATURE_GROUP_DIM),
            nn.Linear(self._RETRIEVAL_FEATURE_GROUP_DIM, hidden_dim),
            nn.GELU(),
        )
        if self.use_raw_image_features:
            self.query_raw_image_proj = nn.Sequential(
                nn.LayerNorm(self.image_embed_dim),
                nn.Linear(self.image_embed_dim, hidden_dim),
                nn.GELU(),
            )
            self.raw_image_fused_proj = nn.Sequential(
                nn.LayerNorm(self.image_embed_dim + self._RETRIEVAL_FEATURE_GROUP_DIM),
                nn.Linear(self.image_embed_dim + self._RETRIEVAL_FEATURE_GROUP_DIM, hidden_dim),
                nn.GELU(),
            )
            self.raw_image_metadata_proj = nn.Sequential(
                nn.LayerNorm(self._RETRIEVAL_FEATURE_GROUP_DIM),
                nn.Linear(self._RETRIEVAL_FEATURE_GROUP_DIM, hidden_dim),
                nn.GELU(),
            )
        else:
            self.query_raw_image_proj = None
            self.raw_image_fused_proj = None
            self.raw_image_metadata_proj = None
        if self.use_osv_support_token:
            self.osv_support_proj = nn.Sequential(
                nn.LayerNorm(self.output_dim + 4),
                nn.Linear(self.output_dim + 4, hidden_dim),
                nn.GELU(),
            )
        else:
            self.osv_support_proj = None
        self.gps_fused_proj = nn.Sequential(
            nn.LayerNorm(self.output_dim + self._RETRIEVAL_FEATURE_GROUP_DIM),
            nn.Linear(self.output_dim + self._RETRIEVAL_FEATURE_GROUP_DIM, hidden_dim),
            nn.GELU(),
        )
        self.gps_metadata_proj = nn.Sequential(
            nn.LayerNorm(self._RETRIEVAL_FEATURE_GROUP_DIM),
            nn.Linear(self._RETRIEVAL_FEATURE_GROUP_DIM, hidden_dim),
            nn.GELU(),
        )
        self.image_feature_gate = None
        self.gps_feature_film = None
        self.raw_image_feature_film = None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(reranker_num_heads),
            dim_feedforward=int(reranker_ffn_dim),
            dropout=float(reranker_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.interaction_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(reranker_num_layers),
            norm=nn.LayerNorm(hidden_dim),
        )

        head_hidden_dim = max(hidden_dim // 2, 1)
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(reranker_dropout)),
            nn.Linear(head_hidden_dim, 1),
        )

        self._retrieval_device: torch.device | None = None
        self._union_retrieval_cache: dict[str, dict[str, dict[str, torch.Tensor]]] | None = None
        self._union_retrieval_cache_source_sizes: dict[str, int] = {}
        self._union_retrieval_cache = self._build_union_retrieval_cache()

        self._val_preds: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []
        self.register_buffer(
            "best_val_median_km",
            torch.tensor(float("inf"), dtype=torch.float32),
            persistent=True,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.contrastive_model.eval()
        return self

    @classmethod
    def _is_optional_state_dict_key(cls, key: str) -> bool:
        return any(key.startswith(prefix) for prefix in cls._OPTIONAL_STATE_DICT_PREFIXES)

    def load_state_dict(self, state_dict, strict: bool = True):
        # These projections were added after the bundled checkpoint was
        # trained. Keep their contribution neutral when their weights are not
        # present, preserving that checkpoint's original behavior.
        for module_name in (
            "candidate_image_metadata_proj",
            "raw_image_metadata_proj",
            "gps_metadata_proj",
        ):
            prefix = f"{module_name}."
            if not any(key.startswith(prefix) for key in state_dict):
                projection = getattr(self, module_name, None)
                if projection is not None:
                    final_linear = next(
                        (layer for layer in reversed(projection) if isinstance(layer, nn.Linear)),
                        None,
                    )
                    if final_linear is not None:
                        nn.init.zeros_(final_linear.weight)
                        if final_linear.bias is not None:
                            nn.init.zeros_(final_linear.bias)

        own_state = self.state_dict()
        filtered_state_dict = state_dict.__class__() if hasattr(state_dict, "__class__") else {}
        for key, value in state_dict.items():
            current_value = own_state.get(key)
            if current_value is not None and hasattr(value, "shape") and tuple(value.shape) != tuple(current_value.shape):
                if self._is_optional_state_dict_key(key):
                    continue
            filtered_state_dict[key] = value

        incompatible = super().load_state_dict(filtered_state_dict, strict=False)
        missing_keys = [key for key in incompatible.missing_keys if not self._is_optional_state_dict_key(key)]
        unexpected_keys = [key for key in incompatible.unexpected_keys if not self._is_optional_state_dict_key(key)]
        if strict and (missing_keys or unexpected_keys):
            missing_str = ", ".join(missing_keys) if missing_keys else "none"
            unexpected_str = ", ".join(unexpected_keys) if unexpected_keys else "none"
            raise RuntimeError(
                "Error(s) in loading state_dict for "
                f"{self.__class__.__name__}: missing keys: {missing_str}; unexpected keys: {unexpected_str}"
            )
        incompatible.missing_keys[:] = missing_keys
        incompatible.unexpected_keys[:] = unexpected_keys
        return incompatible

    @staticmethod
    def _unpack_batch(batch):
        if not isinstance(batch, (tuple, list)):
            raise TypeError("Expected batch to be a tuple/list from DataLoader")
        if len(batch) == 2:
            return batch[0], batch[1], None
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
        raise ValueError(f"Unexpected batch size: {len(batch)}")

    def _split_query_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings [B, D], got {tuple(embeddings.shape)}")
        dim = int(embeddings.shape[1])
        if dim == self.image_embed_dim:
            return embeddings
        if dim == self.contrastive_embed_dim:
            return embeddings[:, : self.image_embed_dim]
        raise ValueError(
            f"Embedding dim mismatch: expected {self.image_embed_dim} or {self.contrastive_embed_dim}, got {dim}"
        )

    def _extract_query_raw_image(self, embeddings: torch.Tensor) -> torch.Tensor:
        image_only = self._split_query_embeddings(embeddings)
        image_only = torch.nan_to_num(image_only, nan=0.0, posinf=0.0, neginf=0.0)
        return F.normalize(image_only.to(torch.float32), p=2, dim=-1, eps=1e-8)

    def _encode_query_image(self, image_only: torch.Tensor) -> torch.Tensor:
        image_only = torch.nan_to_num(image_only, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.inference_mode():
            if self.device.type == "cuda" and self.inference_dtype in _SUPPORTED_HALF_DTYPES:
                with torch.autocast(device_type="cuda", dtype=self.inference_dtype):
                    if hasattr(self.contrastive_model, "encode_image_for_source"):
                        query_img = self.contrastive_model.encode_image_for_source(
                            image_only,
                            source_name=self.retrieval_db_source,
                        )
                    else:
                        query_img = self.contrastive_model.encode_image(image_only)
            else:
                image_only = image_only.to(torch.float32)
                if hasattr(self.contrastive_model, "encode_image_for_source"):
                    query_img = self.contrastive_model.encode_image_for_source(
                        image_only,
                        source_name=self.retrieval_db_source,
                    )
                else:
                    query_img = self.contrastive_model.encode_image(image_only)
        query_img = torch.nan_to_num(query_img, nan=0.0, posinf=0.0, neginf=0.0)
        return F.normalize(query_img.to(torch.float32), p=2, dim=-1, eps=1e-8)

    def _ensure_retrieval_ready(self) -> None:
        if self._retrieval_device is not None and self._retrieval_device == self.device:
            return

        self.retrieval_index.to_device(self.device)
        self._retrieval_device = self.device

    def _reranker_autocast_dtype(self) -> torch.dtype | None:
        if self.device.type != "cuda":
            return None
        if self.inference_dtype == torch.float32:
            return None
        return torch.bfloat16

    def _reranker_feature_dtype(self) -> torch.dtype:
        return self._reranker_autocast_dtype() or torch.float32

    def _reranker_autocast_context(self):
        dtype = self._reranker_autocast_dtype()
        if dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _encode_osv_support_images(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            if self.device.type == "cuda" and self.inference_dtype in _SUPPORTED_HALF_DTYPES:
                with torch.autocast(device_type="cuda", dtype=self.inference_dtype):
                    if hasattr(self.contrastive_model, "encode_osv_image"):
                        encoded = self.contrastive_model.encode_osv_image(image_embeddings.to(self.device))
                    else:
                        encoded = self.contrastive_model.encode_image(image_embeddings.to(self.device))
            else:
                image_embeddings_f32 = image_embeddings.to(self.device, dtype=torch.float32)
                if hasattr(self.contrastive_model, "encode_osv_image"):
                    encoded = self.contrastive_model.encode_osv_image(image_embeddings_f32)
                else:
                    encoded = self.contrastive_model.encode_image(image_embeddings_f32)
        encoded = torch.nan_to_num(encoded.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return F.normalize(encoded, p=2, dim=-1, eps=1e-8)

    def _build_osv_support_token(
        self,
        query_img: torch.Tensor,
        top_indices: torch.Tensor,
        candidate_latlon_deg: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.osv_support_proj is None
            or self._osv_support_cache is None
            or self._osv_support_row_cache_indices is None
            or self._osv_support_row_cache_distance_km is None
            or self._osv_support_row_cache_ready is None
        ):
            raise RuntimeError("OSV support token was requested, but OSV support cache/projection is unavailable")

        batch_size, num_candidates, _ = candidate_latlon_deg.shape
        flat_count = batch_size * num_candidates
        top_indices_np = top_indices.detach().to(torch.long).cpu().reshape(flat_count).numpy()
        candidate_mask_np = candidate_mask.detach().cpu().reshape(flat_count).numpy()
        valid_retrieval_mask = candidate_mask_np & (top_indices_np >= 0)
        missing_cache_mask = valid_retrieval_mask & (~self._osv_support_row_cache_ready[top_indices_np.clip(min=0)])
        if np.any(missing_cache_mask):
            missing_rows = top_indices_np[missing_cache_mask]
            unique_missing_rows, first_positions = np.unique(missing_rows, return_index=True)
            missing_latlon_np = (
                candidate_latlon_deg.detach()
                .to(torch.float32)
                .cpu()
                .reshape(flat_count, 2)
                .numpy()[missing_cache_mask]
            )
            unique_missing_latlon_np = missing_latlon_np[first_positions]
            queried_idx_np, queried_dist_np, queried_valid_np = self._osv_support_cache.query(
                unique_missing_latlon_np,
                neighbors_k=self.osv_support_neighbors_k,
                radius_km=self.osv_support_radius_km,
                workers=-1,
            )
            self._osv_support_row_cache_indices[unique_missing_rows] = queried_idx_np.astype(np.int32, copy=False)
            self._osv_support_row_cache_distance_km[unique_missing_rows] = queried_dist_np.astype(np.float16, copy=False)
            self._osv_support_row_cache_ready[unique_missing_rows] = True

        safe_top_indices_np = top_indices_np.clip(min=0)
        row_idx_np = self._osv_support_row_cache_indices[safe_top_indices_np].astype(np.int64, copy=False)
        distance_km_np = self._osv_support_row_cache_distance_km[safe_top_indices_np].astype(np.float32, copy=False)
        valid_mask_np = valid_retrieval_mask[:, None] & (row_idx_np >= 0)

        neighbor_count = int(row_idx_np.shape[1])
        support_embeddings = torch.zeros(
            (flat_count, neighbor_count, self.output_dim),
            device=self.device,
            dtype=torch.float32,
        )
        valid_flat_rows = row_idx_np[valid_mask_np]
        if valid_flat_rows.size > 0:
            unique_rows, inverse = np.unique(valid_flat_rows, return_inverse=True)
            support_input_np = np.array(
                self._osv_support_cache.image_embeddings_mm[unique_rows],
                dtype=np.float16,
                copy=True,
                order="C",
            )
            support_input_t = torch.from_numpy(support_input_np).to(
                self.device,
                dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                non_blocking=(self.device.type == "cuda"),
            )
            if self._osv_support_cache.embedding_space == "contrastive_osv":
                encoded_unique = F.normalize(
                    support_input_t.to(torch.float32),
                    p=2,
                    dim=-1,
                    eps=1e-8,
                )
            else:
                encoded_unique = self._encode_osv_support_images(support_input_t)
            support_embeddings[torch.from_numpy(valid_mask_np).to(self.device)] = encoded_unique[
                torch.from_numpy(inverse).to(self.device, dtype=torch.long)
            ]

        valid_mask = torch.from_numpy(valid_mask_np).to(self.device, dtype=torch.bool)
        distance_km = torch.from_numpy(distance_km_np).to(self.device, dtype=torch.float32)
        query_flat = (
            query_img.to(torch.float32)
            .unsqueeze(1)
            .expand(-1, num_candidates, -1)
            .reshape(flat_count, self.output_dim)
        )
        similarity = torch.sum(support_embeddings * query_flat.unsqueeze(1), dim=-1)
        masked_similarity = similarity.masked_fill(~valid_mask, -1e4)
        weights = torch.softmax(masked_similarity, dim=1)
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        pooled_support = torch.sum(weights.unsqueeze(-1) * support_embeddings, dim=1)

        valid_counts = valid_mask.to(torch.float32).sum(dim=1)
        has_support = valid_counts > 0
        best_similarity = torch.where(valid_mask, similarity, -1e4).amax(dim=1)
        best_similarity = torch.where(has_support, best_similarity, torch.zeros_like(best_similarity))
        mean_similarity = torch.where(valid_mask, similarity, 0.0).sum(dim=1) / valid_counts.clamp(min=1.0)
        nearest_distance = torch.where(
            valid_mask,
            distance_km,
            torch.full_like(distance_km, self.osv_support_radius_km),
        ).amin(dim=1)
        nearest_distance_feature = 1.0 - torch.clamp(nearest_distance / self.osv_support_radius_km, min=0.0, max=1.0)
        support_fraction = valid_counts / max(float(neighbor_count), 1.0)
        support_stats = torch.stack(
            (
                best_similarity,
                mean_similarity,
                nearest_distance_feature,
                support_fraction,
            ),
            dim=-1,
        )
        fused_support = torch.cat((pooled_support, support_stats), dim=-1)
        support_token = self.osv_support_proj(fused_support)
        support_token = support_token.masked_fill(
            ~candidate_mask.reshape(flat_count).unsqueeze(-1),
            0.0,
        )
        return support_token

    def _union_modality_pool_sizes(self) -> dict[str, int]:
        num_rows = int(self.retrieval_index.num_samples)
        pool_multiplier = int(self.modality_pool_multiplier)
        return {
            "image": (min(int(self.num_neighbors_image) * pool_multiplier, num_rows) if self.num_neighbors_image > 0 else 0),
            "raw_image": (
                min(int(self.num_neighbors_raw_image) * pool_multiplier, num_rows)
                if self.num_neighbors_raw_image > 0
                else 0
            ),
            "gps": (min(int(self.num_neighbors_gps) * pool_multiplier, num_rows) if self.num_neighbors_gps > 0 else 0),
        }

    def _union_candidate_capacity(self) -> int:
        return int(self.num_neighbors_image) + int(self.num_neighbors_raw_image) + int(self.num_neighbors_gps)

    def _build_union_retrieval_cache(self) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        modality_pool_sizes = self._union_modality_pool_sizes()
        cache: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        self._union_retrieval_cache_source_sizes = {}
        for source in self.retrieval_index.manifest.get("sources", []):
            source_name = str(source.get("name", ""))
            source_num_samples = int(source.get("num_samples", 0))
            if not source_name or source_num_samples <= 0:
                continue
            self._union_retrieval_cache_source_sizes[source_name] = source_num_samples
            source_cache: dict[str, dict[str, torch.Tensor]] = {}
            for modality_name, pool_size in modality_pool_sizes.items():
                if pool_size <= 0:
                    continue
                source_cache[modality_name] = {
                    "indices": torch.full((source_num_samples, pool_size), -1, dtype=torch.int32),
                    "scores": torch.full((source_num_samples, pool_size), -float("inf"), dtype=torch.float16),
                    "valid": torch.zeros((source_num_samples,), dtype=torch.bool),
                }
            if source_cache:
                cache[source_name] = source_cache
        return cache

    def _query_cache_entries_from_metadata(
        self,
        metadata_batch: Sequence[dict[str, Any]] | None,
    ) -> list[tuple[str, int] | None] | None:
        if self._union_retrieval_cache is None or metadata_batch is None:
            return None

        cache_entries: list[tuple[str, int] | None] = []
        for metadata in metadata_batch:
            if str(metadata.get("source_split", "")).lower() == "test":
                cache_entries.append(None)
                continue
            source_name = str(metadata.get("source", ""))
            cache_idx_raw = metadata.get("cache_idx", metadata.get("image_cache_idx"))
            if not source_name or cache_idx_raw is None:
                cache_entries.append(None)
                continue
            try:
                cache_idx = int(cache_idx_raw)
            except (TypeError, ValueError):
                cache_entries.append(None)
                continue
            source_size = self._union_retrieval_cache_source_sizes.get(source_name, 0)
            if cache_idx < 0 or cache_idx >= source_size or source_name not in self._union_retrieval_cache:
                cache_entries.append(None)
                continue
            cache_entries.append((source_name, cache_idx))
        return cache_entries

    def _topk_from_matrix_with_cache(
        self,
        *,
        modality_name: str,
        query: torch.Tensor,
        retrieval_mat: torch.Tensor,
        k: int,
        query_indices: torch.Tensor | None,
        query_exclusion_latlon_deg: torch.Tensor | None,
        query_cache_entries: list[tuple[str, int] | None] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        union_cache = self._union_retrieval_cache
        if query_cache_entries is None or union_cache is None:
            indices, scores = self._topk_from_matrix(
                query=query,
                retrieval_mat=retrieval_mat,
                k=k,
                query_indices=query_indices,
                query_exclusion_latlon_deg=query_exclusion_latlon_deg,
            )
            return (
                indices.to(device="cpu", dtype=torch.int32),
                scores.to(device="cpu", dtype=torch.float32),
            )

        batch_size = int(query.shape[0])
        indices_cpu = torch.empty((batch_size, k), dtype=torch.int32)
        scores_cpu = torch.empty((batch_size, k), dtype=torch.float16)
        cache_hits = torch.zeros((batch_size,), dtype=torch.bool)

        for batch_idx, cache_entry in enumerate(query_cache_entries):
            if cache_entry is None:
                continue
            source_name, cache_idx = cache_entry
            source_cache = union_cache.get(source_name)
            if source_cache is None:
                continue
            modality_cache = source_cache.get(modality_name)
            if modality_cache is None or not bool(modality_cache["valid"][cache_idx].item()):
                continue
            indices_cpu[batch_idx] = modality_cache["indices"][cache_idx]
            scores_cpu[batch_idx] = modality_cache["scores"][cache_idx]
            cache_hits[batch_idx] = True

        if not bool(cache_hits.all().item()):
            missing_rows_cpu = torch.nonzero(~cache_hits, as_tuple=False).squeeze(-1)
            missing_rows = missing_rows_cpu.to(device=query.device, dtype=torch.long)
            query_missing = torch.index_select(query, dim=0, index=missing_rows)
            query_indices_missing = None
            if query_indices is not None and query_indices.numel() == batch_size:
                query_indices_missing = torch.index_select(query_indices, dim=0, index=missing_rows)
            query_exclusion_missing = None
            if query_exclusion_latlon_deg is not None and query_exclusion_latlon_deg.shape[0] == batch_size:
                query_exclusion_missing = torch.index_select(
                    query_exclusion_latlon_deg,
                    dim=0,
                    index=missing_rows,
                )

            missing_indices, missing_scores = self._topk_from_matrix(
                query=query_missing,
                retrieval_mat=retrieval_mat,
                k=k,
                query_indices=query_indices_missing,
                query_exclusion_latlon_deg=query_exclusion_missing,
            )
            missing_indices_cpu = missing_indices.detach().to(device="cpu", dtype=torch.int32)
            missing_scores_cpu = missing_scores.detach().to(device="cpu", dtype=torch.float16)

            indices_cpu[missing_rows_cpu] = missing_indices_cpu
            scores_cpu[missing_rows_cpu] = missing_scores_cpu

            for local_row, batch_idx in enumerate(missing_rows_cpu.tolist()):
                cache_entry = query_cache_entries[batch_idx]
                if cache_entry is None:
                    continue
                source_name, cache_idx = cache_entry
                source_cache = union_cache.get(source_name)
                if source_cache is None:
                    continue
                modality_cache = source_cache.get(modality_name)
                if modality_cache is None:
                    continue
                modality_cache["indices"][cache_idx].copy_(missing_indices_cpu[local_row])
                modality_cache["scores"][cache_idx].copy_(missing_scores_cpu[local_row])
                modality_cache["valid"][cache_idx] = True

        return indices_cpu, scores_cpu.to(torch.float32)

    def _retrieve_candidates(
        self,
        query_img: torch.Tensor,
        query_raw_image: torch.Tensor,
        *,
        query_indices: torch.Tensor | None,
        query_exclusion_latlon_deg: torch.Tensor | None,
        query_cache_entries: list[tuple[str, int] | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._retrieve_candidates_union(
            query_img,
            query_raw_image,
            query_indices=query_indices,
            query_exclusion_latlon_deg=query_exclusion_latlon_deg,
            query_cache_entries=query_cache_entries,
        )

    def _retrieve_candidates_union(
        self,
        query_img: torch.Tensor,
        query_raw_image: torch.Tensor,
        *,
        query_indices: torch.Tensor | None,
        query_exclusion_latlon_deg: torch.Tensor | None,
        query_cache_entries: list[tuple[str, int] | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.retrieval_index.image_embeddings
        gps = self.retrieval_index.gps_embeddings
        raw_image = self.retrieval_index.raw_image_embeddings
        if image is None or gps is None:
            raise RuntimeError("Retrieval index tensors are not loaded")

        num_rows = int(image.shape[0])
        if num_rows <= 0:
            raise RuntimeError("Retrieval index is empty")

        batch_size = int(query_img.shape[0])
        candidate_capacity = min(self._union_candidate_capacity(), num_rows)
        if candidate_capacity <= 0:
            raise ValueError("At least one modality candidate budget must be > 0")

        query_indices_for_search = query_indices if query_indices is not None and query_indices.numel() == batch_size else None

        modality_results: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []
        pool_multiplier = int(self.modality_pool_multiplier)
        if self.num_neighbors_image > 0:
            image_k = min(int(self.num_neighbors_image) * pool_multiplier, num_rows)
            image_idx, image_scores = self._topk_from_matrix_with_cache(
                modality_name="image",
                query=query_img.to(image.dtype),
                retrieval_mat=image,
                k=image_k,
                query_indices=query_indices_for_search,
                query_exclusion_latlon_deg=query_exclusion_latlon_deg,
                query_cache_entries=query_cache_entries,
            )
            modality_results.append((0, image_idx, image_scores, int(self.num_neighbors_image)))

        if self.num_neighbors_raw_image > 0:
            if raw_image is None:
                raise RuntimeError("Raw-image retrieval candidates requested, but raw cached embeddings are unavailable")
            raw_image_k = min(int(self.num_neighbors_raw_image) * pool_multiplier, num_rows)
            raw_image_idx, raw_image_scores = self._topk_from_matrix_with_cache(
                modality_name="raw_image",
                query=query_raw_image.to(raw_image.dtype),
                retrieval_mat=raw_image,
                k=raw_image_k,
                query_indices=query_indices_for_search,
                query_exclusion_latlon_deg=query_exclusion_latlon_deg,
                query_cache_entries=query_cache_entries,
            )
            modality_results.append((1, raw_image_idx, raw_image_scores, int(self.num_neighbors_raw_image)))

        if self.num_neighbors_gps > 0:
            gps_k = min(int(self.num_neighbors_gps) * pool_multiplier, num_rows)
            gps_idx, gps_scores = self._topk_from_matrix_with_cache(
                modality_name="gps",
                query=query_img.to(gps.dtype),
                retrieval_mat=gps,
                k=gps_k,
                query_indices=query_indices_for_search,
                query_exclusion_latlon_deg=query_exclusion_latlon_deg,
                query_cache_entries=query_cache_entries,
            )
            modality_results.append((2, gps_idx, gps_scores, int(self.num_neighbors_gps)))

        if not modality_results:
            raise RuntimeError("per_modality_union retrieval produced no modality candidate lists")

        if len(modality_results) == 1:
            modality_id, selected_indices, selected_scores, target_k = modality_results[0]
            keep = min(int(target_k), candidate_capacity, int(selected_indices.shape[1]))
            top_indices = selected_indices[:, :keep].to(torch.long)
            top_scores = selected_scores[:, :keep].to(torch.float32)
            candidate_mask = torch.ones((batch_size, keep), dtype=torch.bool)
            rank_features = torch.ones((batch_size, keep, 3), dtype=torch.float32)
            shortlist_flags = torch.zeros((batch_size, keep, 3), dtype=torch.float32)
            rank_denom = max(float(keep - 1), 1.0)
            modality_ranks = (
                torch.arange(keep, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1) / rank_denom
            )
            rank_features[:, :, modality_id] = modality_ranks
            shortlist_flags[:, :, modality_id] = 1.0
            non_blocking = query_img.device.type == "cuda"
            return (
                top_indices.to(device=query_img.device, non_blocking=non_blocking),
                top_scores.to(device=query_img.device, non_blocking=non_blocking),
                candidate_mask.to(device=query_img.device, non_blocking=non_blocking),
                rank_features.to(device=query_img.device, non_blocking=non_blocking),
                shortlist_flags.to(device=query_img.device, non_blocking=non_blocking),
            )

        top_indices = torch.zeros((batch_size, candidate_capacity), dtype=torch.long)
        top_scores = torch.full((batch_size, candidate_capacity), -float("inf"), dtype=torch.float32)
        candidate_mask = torch.zeros((batch_size, candidate_capacity), dtype=torch.bool)
        rank_features = torch.ones((batch_size, candidate_capacity, 3), dtype=torch.float32)
        shortlist_flags = torch.zeros((batch_size, candidate_capacity, 3), dtype=torch.float32)
        sampled_indices_parts: list[torch.Tensor] = []
        sampled_scores_parts: list[torch.Tensor] = []
        sampled_modality_parts: list[torch.Tensor] = []
        sampled_rank_pos_parts: list[torch.Tensor] = []
        sampled_rank_parts: list[torch.Tensor] = []

        for modality_id, modality_indices, modality_scores, target_k in modality_results:
            selected_indices = modality_indices
            selected_scores = modality_scores
            pool_size = int(selected_indices.shape[1])
            if self.training and pool_size > target_k:
                random_scores = torch.rand((batch_size, pool_size), dtype=torch.float32)
                sample_order = torch.topk(random_scores, k=target_k, dim=1).indices
                selected_indices = torch.gather(selected_indices, dim=1, index=sample_order)
                selected_scores = torch.gather(selected_scores, dim=1, index=sample_order)
            elif pool_size > target_k:
                selected_indices = selected_indices[:, :target_k]
                selected_scores = selected_scores[:, :target_k]

            if selected_scores.shape[1] > 1:
                sorted_order = torch.argsort(selected_scores, dim=1, descending=True)
                selected_indices = torch.gather(selected_indices, dim=1, index=sorted_order)
                selected_scores = torch.gather(selected_scores, dim=1, index=sorted_order)

            rank_pos = torch.arange(selected_indices.shape[1], dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            rank_denom = max(float(target_k - 1), 1.0)
            rank_values = rank_pos.to(torch.float32) / rank_denom

            sampled_indices_parts.append(selected_indices.to(torch.long))
            sampled_scores_parts.append(selected_scores.to(torch.float32))
            sampled_modality_parts.append(torch.full_like(selected_indices, fill_value=modality_id, dtype=torch.long))
            sampled_rank_pos_parts.append(rank_pos)
            sampled_rank_parts.append(rank_values)

        all_indices = torch.cat(sampled_indices_parts, dim=1)
        all_scores = torch.cat(sampled_scores_parts, dim=1)
        all_modalities = torch.cat(sampled_modality_parts, dim=1)
        all_rank_pos = torch.cat(sampled_rank_pos_parts, dim=1)
        all_rank_values = torch.cat(sampled_rank_parts, dim=1)

        valid = (all_indices >= 0) & torch.isfinite(all_scores)
        if not torch.any(valid):
            raise RuntimeError("Failed to build any union retrieval candidates for the batch")

        batch_offsets = (
            torch.arange(batch_size, dtype=torch.int64).unsqueeze(1) * int(num_rows)
        )
        flat_keys = all_indices.to(torch.int64) + batch_offsets

        valid_keys = flat_keys[valid]
        valid_scores = all_scores[valid]
        valid_modalities = all_modalities[valid]
        valid_rank_pos = all_rank_pos[valid]
        valid_rank_values = all_rank_values[valid]

        unique_keys, inverse = torch.unique(valid_keys, sorted=True, return_inverse=True)
        unique_batch_idx = torch.div(unique_keys, int(num_rows), rounding_mode="floor").to(torch.long)
        unique_candidate_idx = torch.remainder(unique_keys, int(num_rows)).to(torch.long)

        unique_count = int(unique_keys.numel())
        max_scores = torch.full((unique_count,), -float("inf"), dtype=torch.float32)
        max_scores.scatter_reduce_(0, inverse, valid_scores, reduce="amax", include_self=True)

        large_rank = int(all_rank_pos.shape[1] + 1)
        seed_rank_pos = torch.where(
            valid_scores == max_scores[inverse],
            valid_rank_pos,
            torch.full_like(valid_rank_pos, large_rank),
        )
        best_rank = torch.full((unique_count,), large_rank, dtype=torch.long)
        best_rank.scatter_reduce_(0, inverse, seed_rank_pos, reduce="amin", include_self=True)

        merged_ranks = torch.ones((unique_count, 3), dtype=torch.float32)
        merged_flags = torch.zeros((unique_count, 3), dtype=torch.float32)
        for modality_id in range(3):
            modality_mask = valid_modalities == modality_id
            if not torch.any(modality_mask):
                continue
            modality_inverse = inverse[modality_mask]
            modality_rank_values = valid_rank_values[modality_mask]
            merged_ranks[:, modality_id].scatter_reduce_(
                0,
                modality_inverse,
                modality_rank_values,
                reduce="amin",
                include_self=True,
            )
            merged_flags[:, modality_id].scatter_reduce_(
                0,
                modality_inverse,
                torch.ones_like(modality_rank_values),
                reduce="amax",
                include_self=True,
            )

        order = torch.arange(unique_count, dtype=torch.long)
        order = order[torch.argsort(unique_candidate_idx[order], stable=True)]
        order = order[torch.argsort(best_rank[order], stable=True)]
        order = order[torch.argsort(max_scores[order], descending=True, stable=True)]
        order = order[torch.argsort(unique_batch_idx[order], stable=True)]

        ordered_batch_idx = unique_batch_idx[order]
        batch_counts = torch.bincount(ordered_batch_idx, minlength=batch_size)
        if torch.any(batch_counts == 0):
            raise RuntimeError("Failed to build any union retrieval candidates for at least one query")

        batch_starts = torch.cumsum(batch_counts, dim=0) - batch_counts
        pos_in_batch = torch.arange(order.shape[0], dtype=torch.long) - torch.repeat_interleave(batch_starts, batch_counts)
        keep_mask = pos_in_batch < int(candidate_capacity)

        kept_order = order[keep_mask]
        kept_batch_idx = unique_batch_idx[kept_order]
        kept_pos = pos_in_batch[keep_mask]

        top_indices[kept_batch_idx, kept_pos] = unique_candidate_idx[kept_order]
        top_scores[kept_batch_idx, kept_pos] = max_scores[kept_order]
        rank_features[kept_batch_idx, kept_pos] = merged_ranks[kept_order]
        shortlist_flags[kept_batch_idx, kept_pos] = merged_flags[kept_order]
        candidate_mask[kept_batch_idx, kept_pos] = True

        non_blocking = query_img.device.type == "cuda"
        return (
            top_indices.to(device=query_img.device, non_blocking=non_blocking),
            top_scores.to(device=query_img.device, non_blocking=non_blocking),
            candidate_mask.to(device=query_img.device, non_blocking=non_blocking),
            rank_features.to(device=query_img.device, non_blocking=non_blocking),
            shortlist_flags.to(device=query_img.device, non_blocking=non_blocking),
        )

    def _topk_from_matrix(
        self,
        *,
        query: torch.Tensor,
        retrieval_mat: torch.Tensor,
        k: int,
        query_indices: torch.Tensor | None,
        query_exclusion_latlon_deg: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_rows = int(retrieval_mat.shape[0])
        if num_rows <= 0:
            raise RuntimeError("Retrieval index is empty")

        k = min(int(k), num_rows)
        if k <= 0:
            raise ValueError("k must be > 0")

        batch_size = query.shape[0]
        has_query_mask = query_indices is not None and query_indices.numel() == batch_size
        best_scores = torch.full((batch_size, k), -float("inf"), device=query.device, dtype=query.dtype)
        best_indices = torch.zeros((batch_size, k), device=query.device, dtype=torch.long)

        chunk_size = min(int(self.retrieval_chunk_size), num_rows)
        if chunk_size <= 0:
            raise ValueError("retrieval_chunk_size must be > 0")

        for start in range(0, num_rows, chunk_size):
            end = min(start + chunk_size, num_rows)
            chunk = retrieval_mat[start:end]
            chunk_scores = torch.matmul(query, chunk.T)
            if has_query_mask:
                valid = (query_indices >= start) & (query_indices < end)
                if torch.any(valid):
                    row_ids = torch.nonzero(valid, as_tuple=False).squeeze(-1)
                    local_cols = (query_indices[row_ids] - start).to(torch.long)
                    chunk_scores[row_ids, local_cols] = -float("inf")
            retrieval_latlon = self.retrieval_index.latlon_deg
            candidate_latlon = None if retrieval_latlon is None else retrieval_latlon[start:end]
            self._mask_nearby_retrieval_scores(
                chunk_scores,
                query_exclusion_latlon_deg=query_exclusion_latlon_deg,
                candidate_latlon_deg=candidate_latlon,
                radius_km=self.leave_one_out_exclusion_km,
            )

            chunk_k = min(k, end - start)
            chunk_top_scores, chunk_top_local = torch.topk(chunk_scores, k=chunk_k, dim=-1)
            chunk_top_indices = chunk_top_local + int(start)

            merged_scores = torch.cat((best_scores, chunk_top_scores), dim=-1)
            merged_indices = torch.cat((best_indices, chunk_top_indices), dim=-1)
            best_scores, keep_pos = torch.topk(merged_scores, k=k, dim=-1)
            best_indices = torch.gather(merged_indices, dim=-1, index=keep_pos)

        return best_indices, best_scores

    @staticmethod
    def _mask_nearby_retrieval_scores(
        scores: torch.Tensor,
        *,
        query_exclusion_latlon_deg: torch.Tensor | None,
        candidate_latlon_deg: torch.Tensor | None,
        radius_km: float,
    ) -> None:
        if radius_km <= 0 or query_exclusion_latlon_deg is None or candidate_latlon_deg is None:
            return
        if query_exclusion_latlon_deg.ndim != 2 or query_exclusion_latlon_deg.shape[-1] != 2:
            return
        if query_exclusion_latlon_deg.shape[0] != scores.shape[0] or candidate_latlon_deg.shape[0] != scores.shape[1]:
            return

        query_deg = query_exclusion_latlon_deg.to(device=scores.device, dtype=torch.float32)
        candidate_deg = candidate_latlon_deg.to(device=scores.device, dtype=torch.float32)
        finite_query = torch.isfinite(query_deg).all(dim=-1)
        finite_candidate = torch.isfinite(candidate_deg).all(dim=-1)
        if not bool(torch.any(finite_query).item()) or not bool(torch.any(finite_candidate).item()):
            return

        q_lat = query_deg[:, 0]
        q_lon = query_deg[:, 1]
        c_lat = candidate_deg[:, 0]
        c_lon = candidate_deg[:, 1]

        radius_deg = float(radius_km) / 111.195
        lat_delta = torch.abs(q_lat.unsqueeze(1) - c_lat.unsqueeze(0))
        lon_delta = torch.abs(q_lon.unsqueeze(1) - c_lon.unsqueeze(0))
        lon_delta = torch.minimum(lon_delta, 360.0 - lon_delta)
        lon_radius_deg = radius_deg / torch.cos(torch.deg2rad(q_lat)).abs().clamp_min(0.01)
        close_box = (
            finite_query.unsqueeze(1)
            & finite_candidate.unsqueeze(0)
            & (lat_delta <= radius_deg)
            & (lon_delta <= lon_radius_deg.unsqueeze(1))
        )
        if not bool(torch.any(close_box).item()):
            return

        row_idx, col_idx = torch.nonzero(close_box, as_tuple=True)
        q_sel = query_deg[row_idx]
        c_sel = candidate_deg[col_idx]
        dlat = torch.deg2rad(c_sel[:, 0] - q_sel[:, 0])
        dlon = torch.remainder(c_sel[:, 1] - q_sel[:, 1] + 180.0, 360.0) - 180.0
        dlon = torch.deg2rad(dlon)
        q_lat_rad = torch.deg2rad(q_sel[:, 0])
        c_lat_rad = torch.deg2rad(c_sel[:, 0])
        a = (
            torch.sin(dlat * 0.5).square()
            + torch.cos(q_lat_rad) * torch.cos(c_lat_rad) * torch.sin(dlon * 0.5).square()
        )
        distance_km = 6371.0 * 2.0 * torch.asin(torch.sqrt(a.clamp_(0.0, 1.0)))
        nearby = distance_km <= float(radius_km)
        if bool(torch.any(nearby).item()):
            scores[row_idx[nearby], col_idx[nearby]] = -float("inf")

    def _gather_retrieval_tensors(
        self,
        top_indices: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.retrieval_index.image_embeddings
        gps = self.retrieval_index.gps_embeddings
        latlon_deg = self.retrieval_index.latlon_deg
        if image is None or gps is None or latlon_deg is None:
            raise RuntimeError("Retrieval index tensors are unavailable")

        feature_dtype = self._reranker_feature_dtype()
        non_blocking = self.device.type == "cuda"

        retrieved_image = image[top_indices].to(dtype=feature_dtype, non_blocking=non_blocking)
        retrieved_raw_image = self.retrieval_index.gather_raw_image_embeddings(
            top_indices,
            device=self.device,
            dtype=feature_dtype,
        )
        retrieved_gps = gps[top_indices].to(dtype=feature_dtype, non_blocking=non_blocking)
        retrieved_latlon = latlon_deg[top_indices].to(torch.float32)
        return retrieved_raw_image, retrieved_image, retrieved_gps, retrieved_latlon

    def _compute_candidate_retrieval_features(
        self,
        query_raw_image: torch.Tensor,
        query_img: torch.Tensor,
        retrieved_image: torch.Tensor,
        retrieved_raw_image: torch.Tensor | None,
        retrieved_gps: torch.Tensor,
        rank_features: torch.Tensor,
        shortlist_flags: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        query_img_f32 = query_img.to(torch.float32).unsqueeze(1)
        retrieved_image_f32 = retrieved_image.to(torch.float32)
        retrieved_gps_f32 = retrieved_gps.to(torch.float32)
        image_similarity = torch.sum(
            query_img_f32 * retrieved_image_f32,
            dim=-1,
        )
        gps_similarity = torch.sum(
            query_img_f32 * retrieved_gps_f32,
            dim=-1,
        )
        if retrieved_raw_image is not None:
            query_raw_image_f32 = query_raw_image.to(torch.float32).unsqueeze(1)
            retrieved_raw_image_f32 = retrieved_raw_image.to(torch.float32)
            raw_image_similarity = torch.sum(
                query_raw_image_f32 * retrieved_raw_image_f32,
                dim=-1,
            )
        else:
            raw_image_similarity = torch.zeros_like(image_similarity)

        features = torch.cat(
            (
                image_similarity.unsqueeze(-1),
                raw_image_similarity.unsqueeze(-1),
                gps_similarity.unsqueeze(-1),
                rank_features.to(torch.float32),
                shortlist_flags.to(torch.float32),
            ),
            dim=-1,
        )
        features = features.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
        return features

    def _compute_fused_retrieval_scores(
        self,
        query_img: torch.Tensor,
        retrieved_image: torch.Tensor,
        retrieved_gps: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        query_img_f32 = query_img.to(torch.float32).unsqueeze(1)
        image_similarity = torch.sum(query_img_f32 * retrieved_image.to(torch.float32), dim=-1)
        gps_similarity = torch.sum(query_img_f32 * retrieved_gps.to(torch.float32), dim=-1)
        fused_scores = (
            _RETRIEVAL_BASELINE_IMAGE_WEIGHT * image_similarity
            + _RETRIEVAL_BASELINE_GPS_WEIGHT * gps_similarity
        )
        return fused_scores.masked_fill(~candidate_mask, -float("inf"))

    @staticmethod
    def _candidate_haversine_km(candidate_deg: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
        if candidate_deg.ndim != 3 or candidate_deg.shape[-1] != 2:
            raise ValueError(f"Expected candidate_deg [B, N, 2], got {tuple(candidate_deg.shape)}")
        if target_deg.ndim != 2 or target_deg.shape[-1] != 2:
            raise ValueError(f"Expected target_deg [B, 2], got {tuple(target_deg.shape)}")

        cand_rad = torch.deg2rad(candidate_deg)
        tgt_rad = torch.deg2rad(target_deg).unsqueeze(1)
        dlat = cand_rad[..., 0] - tgt_rad[..., 0]
        dlon = cand_rad[..., 1] - tgt_rad[..., 1]
        hav = torch.sin(dlat / 2.0).pow(2) + torch.cos(tgt_rad[..., 0]) * torch.cos(cand_rad[..., 0]) * torch.sin(
            dlon / 2.0
        ).pow(2)
        hav = torch.clamp(hav, min=0.0, max=1.0)
        return 6371.0 * (2.0 * torch.arcsin(torch.sqrt(hav)))

    def _split_retrieval_feature_groups(
        self,
        retrieval_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if retrieval_features.ndim != 3 or retrieval_features.shape[-1] != self._RETRIEVAL_FEATURE_DIM:
            raise ValueError(
                f"Expected retrieval_features [B, N, {self._RETRIEVAL_FEATURE_DIM}], got {tuple(retrieval_features.shape)}"
            )
        features = retrieval_features.to(torch.float32)
        return (
            features[..., self._RETRIEVAL_FEATURE_IMAGE_IDX],
            features[..., self._RETRIEVAL_FEATURE_RAW_IMAGE_IDX],
            features[..., self._RETRIEVAL_FEATURE_GPS_IDX],
        )

    def _build_candidate_batch(
        self,
        embeddings: torch.Tensor,
        *,
        query_indices: torch.Tensor | None = None,
        metadata_batch: Sequence[dict[str, Any]] | None = None,
        query_latlon_deg: torch.Tensor | None = None,
        apply_leave_one_out: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        self._ensure_retrieval_ready()
        query_raw_image = self._extract_query_raw_image(embeddings)
        query_img = self._encode_query_image(query_raw_image)
        query_cache_entries = self._query_cache_entries_from_metadata(metadata_batch)

        query_indices_for_search = None
        if apply_leave_one_out and query_indices is not None and query_indices.numel() == query_img.shape[0]:
            query_indices_for_search = query_indices
        query_exclusion_latlon_deg = None
        if (
            apply_leave_one_out
            and self.leave_one_out_exclusion_km > 0
            and query_latlon_deg is not None
            and query_latlon_deg.shape[0] == query_img.shape[0]
        ):
            query_exclusion_latlon_deg = query_latlon_deg
        safe_top_indices, top_scores, candidate_mask, rank_features, shortlist_flags = self._retrieve_candidates(
            query_img,
            query_raw_image,
            query_indices=query_indices_for_search,
            query_exclusion_latlon_deg=query_exclusion_latlon_deg,
            query_cache_entries=query_cache_entries,
        )

        retrieved_raw_image, retrieved_image, retrieved_gps, candidate_latlon_deg = (
            self._gather_retrieval_tensors(safe_top_indices)
        )
        top_scores = self._compute_fused_retrieval_scores(
            query_img,
            retrieved_image,
            retrieved_gps,
            candidate_mask,
        )
        retrieval_features = self._compute_candidate_retrieval_features(
            query_raw_image,
            query_img,
            retrieved_image,
            retrieved_raw_image,
            retrieved_gps,
            rank_features,
            shortlist_flags,
            candidate_mask,
        )
        return {
            "query_raw_image": query_raw_image,
            "query_img": query_img,
            "top_indices": safe_top_indices,
            "top_scores": top_scores,
            "candidate_mask": candidate_mask,
            "shortlist_flags": shortlist_flags,
            "retrieved_raw_image": retrieved_raw_image,
            "retrieved_image": retrieved_image,
            "retrieved_gps": retrieved_gps,
            "candidate_latlon_deg": candidate_latlon_deg,
            "retrieval_features": retrieval_features,
        }

    @staticmethod
    def _gather_candidates(candidate_tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        gather_index = positions.unsqueeze(-1).expand(-1, -1, candidate_tensor.shape[-1])
        return torch.gather(candidate_tensor, dim=1, index=gather_index)

    def _rerank_outputs(
        self,
        query_raw_image: torch.Tensor | None,
        query_img: torch.Tensor,
        top_indices: torch.Tensor,
        retrieved_raw_image: torch.Tensor | None,
        retrieved_image: torch.Tensor,
        retrieved_gps: torch.Tensor,
        candidate_latlon_deg: torch.Tensor,
        retrieval_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_candidates, _ = retrieved_image.shape

        with self._reranker_autocast_context():
            if self.use_retrieval_features:
                image_features, raw_image_features, gps_features = self._split_retrieval_feature_groups(retrieval_features)
            else:
                zero_features = torch.zeros(
                    (batch_size, num_candidates, self._RETRIEVAL_FEATURE_GROUP_DIM),
                    device=retrieval_features.device,
                    dtype=torch.float32,
                )
                image_features = zero_features
                raw_image_features = zero_features
                gps_features = zero_features

            q = self.query_proj(query_img).unsqueeze(1).expand(-1, num_candidates, -1).reshape(-1, self.hidden_dim)
            i_inputs = torch.cat((retrieved_image.to(torch.float32), image_features), dim=-1)
            i = self.candidate_image_fused_proj(i_inputs) + self.candidate_image_metadata_proj(image_features)
            i = i.reshape(-1, self.hidden_dim)
            g_inputs = torch.cat((retrieved_gps.to(torch.float32), gps_features), dim=-1)
            g = self.gps_fused_proj(g_inputs) + self.gps_metadata_proj(gps_features)
            g = g.reshape(-1, self.hidden_dim)

            token_list = [q, i, g]
            if self.use_osv_support_token:
                osv_support = self._build_osv_support_token(query_img, top_indices, candidate_latlon_deg, candidate_mask)
                token_list.append(osv_support)
            if (
                self.use_raw_image_features
                and retrieved_raw_image is not None
                and self.raw_image_fused_proj is not None
            ):
                raw_inputs = torch.cat((retrieved_raw_image.to(torch.float32), raw_image_features), dim=-1)
                i_raw = self.raw_image_fused_proj(raw_inputs)
                if self.raw_image_metadata_proj is not None:
                    i_raw = i_raw + self.raw_image_metadata_proj(raw_image_features)
                i_raw = i_raw.reshape(-1, self.hidden_dim)
                token_list.append(i_raw)

            total_candidates = batch_size * num_candidates
            chunk_size = self.reranker_sequence_chunk_size or total_candidates
            score_chunks: list[torch.Tensor] = []
            for start in range(0, total_candidates, chunk_size):
                end = min(start + chunk_size, total_candidates)
                cls_token = self.cls_token.expand(end - start, -1, -1)
                stacked_chunk = torch.stack([token[start:end] for token in token_list], dim=1)
                seq_chunk = torch.cat((cls_token, stacked_chunk), dim=1)
                encoded_chunk = self.interaction_encoder(seq_chunk)
                cls_hidden_chunk = encoded_chunk[:, 0, :]
                score_chunks.append(self.score_head(cls_hidden_chunk))

            scores = score_chunks[0] if len(score_chunks) == 1 else torch.cat(score_chunks, dim=0)
            scores = scores.to(torch.float32).reshape(batch_size, num_candidates)
        return scores

    def _pairwise_margin_ranking_loss(
        self,
        rerank_scores: torch.Tensor,
        candidate_distance_km: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rerank_scores.shape != candidate_distance_km.shape:
            raise ValueError("rerank_scores and candidate_distance_km must share shape")
        if candidate_mask.shape != rerank_scores.shape:
            raise ValueError("candidate_mask must share shape with rerank_scores")

        batch_size, num_candidates = rerank_scores.shape
        if num_candidates < 2:
            zero = rerank_scores.new_tensor(0.0)
            one = rerank_scores.new_tensor(1.0)
            return zero, one

        masked_dist = candidate_distance_km.masked_fill(~candidate_mask, float("inf"))
        pos_idx = torch.argmin(masked_dist, dim=1, keepdim=True)
        pos_scores = torch.gather(rerank_scores, dim=1, index=pos_idx)

        all_idx = torch.arange(num_candidates, device=rerank_scores.device).unsqueeze(0).expand(batch_size, -1)
        neg_mask = candidate_mask & (all_idx != pos_idx)
        if not torch.any(neg_mask):
            zero = rerank_scores.new_tensor(0.0)
            one = rerank_scores.new_tensor(1.0)
            return zero, one

        pos_dist = torch.gather(masked_dist, dim=1, index=pos_idx)
        dist_gap = (candidate_distance_km - pos_dist).masked_fill(~neg_mask, 0.0)
        neg_scores = rerank_scores[neg_mask]
        pos_expanded = pos_scores.expand_as(rerank_scores)[neg_mask]
        pos_dist = pos_dist.expand_as(candidate_distance_km)[neg_mask]
        dist_gap = dist_gap[neg_mask]
        pair_losses = torch.relu(self.margin_loss_margin - (pos_expanded - neg_scores))
        loss = pair_losses.mean()
        pair_acc = (pos_expanded > neg_scores).to(torch.float32).mean()
        return loss, pair_acc

    def forward(
        self,
        embeddings: torch.Tensor,
        query_indices: torch.Tensor | None = None,
        metadata_batch: Sequence[dict[str, Any]] | None = None,
        query_latlon_deg: torch.Tensor | None = None,
        *,
        apply_leave_one_out: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch_outputs = self._build_candidate_batch(
            embeddings,
            query_indices=query_indices,
            metadata_batch=metadata_batch,
            query_latlon_deg=query_latlon_deg,
            apply_leave_one_out=apply_leave_one_out,
        )
        rerank_scores = self._rerank_outputs(
            batch_outputs["query_raw_image"] if self.use_raw_image_features else None,
            batch_outputs["query_img"],
            batch_outputs["top_indices"],
            batch_outputs["retrieved_raw_image"] if self.use_raw_image_features else None,
            batch_outputs["retrieved_image"],
            batch_outputs["retrieved_gps"],
            batch_outputs["candidate_latlon_deg"],
            batch_outputs["retrieval_features"],
            batch_outputs["candidate_mask"],
        )
        rerank_scores = rerank_scores.masked_fill(~batch_outputs["candidate_mask"], -float("inf"))

        return {
            "top_indices": batch_outputs["top_indices"].masked_fill(~batch_outputs["candidate_mask"], -1),
            "top_scores": batch_outputs["top_scores"].masked_fill(~batch_outputs["candidate_mask"], -float("inf")),
            "rerank_scores": rerank_scores,
            "candidate_latlon_deg": batch_outputs["candidate_latlon_deg"],
            "candidate_mask": batch_outputs["candidate_mask"],
            "shortlist_flags": batch_outputs["shortlist_flags"],
        }

    def _common_step(self, batch, *, apply_leave_one_out: bool) -> dict[str, torch.Tensor]:
        embeddings, targets_deg, metadata = self._unpack_batch(batch)
        query_indices = self.retrieval_index.query_indices_from_metadata(metadata, device=embeddings.device)
        outputs = self(
            embeddings,
            query_indices=query_indices,
            metadata_batch=metadata,
            query_latlon_deg=targets_deg,
            apply_leave_one_out=apply_leave_one_out,
        )

        candidate_mask = outputs["candidate_mask"]
        candidate_distance_km = self._candidate_haversine_km(outputs["candidate_latlon_deg"], targets_deg)
        candidate_distance_km = candidate_distance_km.masked_fill(~candidate_mask, float("inf"))
        margin_loss, pair_acc = self._pairwise_margin_ranking_loss(
            outputs["rerank_scores"],
            candidate_distance_km,
            candidate_mask,
        )
        rerank_order = torch.argsort(outputs["rerank_scores"], dim=-1, descending=True)
        rerank_top_pos = rerank_order[:, :1]

        loss = self.margin_loss_weight * margin_loss

        retrieval_top_pos = torch.argmax(outputs["top_scores"], dim=-1, keepdim=True)

        rerank_top_deg = self._gather_candidates(outputs["candidate_latlon_deg"], rerank_top_pos).squeeze(1)
        retrieval_top_deg = self._gather_candidates(outputs["candidate_latlon_deg"], retrieval_top_pos).squeeze(1)
        rerank_top_flags = self._gather_candidates(outputs["shortlist_flags"], rerank_top_pos).squeeze(1)
        retrieval_top_flags = self._gather_candidates(outputs["shortlist_flags"], retrieval_top_pos).squeeze(1)

        rerank_top_km = torch.gather(candidate_distance_km, dim=1, index=rerank_top_pos).squeeze(1)
        retrieval_top_km = torch.gather(candidate_distance_km, dim=1, index=retrieval_top_pos).squeeze(1)
        oracle_top_km = torch.min(candidate_distance_km, dim=1).values

        return {
            "loss": loss,
            "margin_loss": margin_loss,
            "final_pred_deg": rerank_top_deg,
            "pair_acc": pair_acc,
            "rerank_top_deg": rerank_top_deg,
            "retrieval_top_deg": retrieval_top_deg,
            "rerank_top_flags": rerank_top_flags,
            "retrieval_top_flags": retrieval_top_flags,
            "targets_deg": targets_deg,
            "rerank_top_km": rerank_top_km,
            "retrieval_top_km": retrieval_top_km,
            "oracle_top_km": oracle_top_km,
        }

    def training_step(self, batch, batch_idx):
        out = self._common_step(batch, apply_leave_one_out=True)
        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            batch_size = int(out["targets_deg"].shape[0])
            self.log(
                "train/loss",
                out["loss"],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "train/margin_loss",
                out["margin_loss"],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "train/pairwise_acc",
                out["pair_acc"],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "train/rerank_top1_mean_km",
                out["rerank_top_km"].mean(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "train/retrieval_top1_mean_km",
                out["retrieval_top_km"].mean(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "train/oracle_top1_mean_km",
                out["oracle_top_km"].mean(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "train/improvement_mean_km",
                (out["retrieval_top_km"] - out["rerank_top_km"]).mean(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "train/rerank_top1_in_image_topk",
                out["rerank_top_flags"][:, 0].mean(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "train/rerank_top1_in_raw_image_topk",
                out["rerank_top_flags"][:, 1].mean(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "train/rerank_top1_in_gps_topk",
                out["rerank_top_flags"][:, 2].mean(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "train/retrieval_top1_in_image_topk",
                out["retrieval_top_flags"][:, 0].mean(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "train/retrieval_top1_in_raw_image_topk",
                out["retrieval_top_flags"][:, 1].mean(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "train/retrieval_top1_in_gps_topk",
                out["retrieval_top_flags"][:, 2].mean(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            if trainer.optimizers:
                lr = float(trainer.optimizers[0].param_groups[0]["lr"])
                self.log(
                    "train/lr",
                    lr,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    batch_size=batch_size,
                )
        return out["loss"]

    def on_validation_epoch_start(self):
        self._val_preds = []
        self._val_targets = []

    def validation_step(self, batch, batch_idx):
        out = self._common_step(batch, apply_leave_one_out=True)
        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            batch_size = int(out["targets_deg"].shape[0])
            self.log(
                "val/loss",
                out["loss"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "val/margin_loss",
                out["margin_loss"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "val/pairwise_acc",
                out["pair_acc"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "val/rerank_top1_mean_km",
                out["rerank_top_km"].mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "val/retrieval_top1_mean_km",
                out["retrieval_top_km"].mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "val/oracle_top1_mean_km",
                out["oracle_top_km"].mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "val/improvement_mean_km",
                (out["retrieval_top_km"] - out["rerank_top_km"]).mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "val/rerank_top1_in_image_topk",
                out["rerank_top_flags"][:, 0].mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "val/rerank_top1_in_raw_image_topk",
                out["rerank_top_flags"][:, 1].mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "val/rerank_top1_in_gps_topk",
                out["rerank_top_flags"][:, 2].mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "val/retrieval_top1_in_image_topk",
                out["retrieval_top_flags"][:, 0].mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "val/retrieval_top1_in_raw_image_topk",
                out["retrieval_top_flags"][:, 1].mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "val/retrieval_top1_in_gps_topk",
                out["retrieval_top_flags"][:, 2].mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
        self._val_preds.append(out["final_pred_deg"].detach().cpu())
        self._val_targets.append(out["targets_deg"].detach().cpu())
        return out["loss"]

    def on_validation_epoch_end(self):
        if not self._val_targets:
            mean_km = float("inf")
            median_km = float("inf")
        else:
            pred_np = torch.cat(self._val_preds, dim=0).numpy()
            tgt_np = torch.cat(self._val_targets, dim=0).numpy()
            dists = geodesic_km(pred_np[:, 0], pred_np[:, 1], tgt_np[:, 0], tgt_np[:, 1])
            mean_km = float(np.mean(dists))
            median_km = float(np.median(dists))

        if getattr(self, "_trainer", None) is not None:
            self.log("val/mean_km", mean_km, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log("val/median_km", median_km, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        if median_km < float(self.best_val_median_km.item()):
            self.best_val_median_km.fill_(median_km)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        warmup_steps = int(self.hparams.warmup_steps)
        total_steps = int(self.hparams.total_steps)
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if warmup_steps >= total_steps:
            raise ValueError("warmup_steps must be < total_steps")

        if warmup_steps > 0:
            cosine_steps = total_steps - warmup_steps
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
