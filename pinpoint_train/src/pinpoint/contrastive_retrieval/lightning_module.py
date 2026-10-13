from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pinpoint.utils import geodesic_km


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP block."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.dropout1(y)
        y = self.fc2(y)
        y = self.dropout2(y)
        return x + y


class SharedImageTower(nn.Module):
    """Shared projection trunk with source-specific input adapters."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        trunk_hidden_dim: int,
        trunk_depth: int,
        trunk_expansion: int,
        trunk_dropout: float,
        adapter_depth: int,
        adapter_expansion: int,
        adapter_dropout: float,
        sources: Sequence[str],
    ):
        super().__init__()
        if trunk_depth < 1:
            raise ValueError("image tower depth must be >= 1")
        if adapter_depth < 1:
            raise ValueError("image adapter depth must be >= 1")
        if not sources:
            raise ValueError("sources must contain at least one source name")

        self.adapters = nn.ModuleDict(
            {
                str(source_name): nn.Sequential(
                    *self._build_adapter_layers(
                        in_dim,
                        trunk_hidden_dim,
                        adapter_depth,
                        adapter_expansion,
                        adapter_dropout,
                    )
                )
                for source_name in sources
            }
        )
        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(
                    trunk_hidden_dim,
                    expansion=trunk_expansion,
                    dropout=trunk_dropout,
                )
                for _ in range(trunk_depth)
            ]
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(trunk_hidden_dim),
            nn.Linear(trunk_hidden_dim, out_dim),
        )

    def forward(self, image_embeddings: torch.Tensor, *, source_name: str) -> torch.Tensor:
        source_key = str(source_name)
        if source_key not in self.adapters:
            raise ValueError(f"Unsupported image source '{source_name}'")
        x = self.adapters[source_key](image_embeddings)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(x)

    @staticmethod
    def _build_adapter_layers(
        in_dim: int,
        hidden_dim: int,
        adapter_depth: int,
        adapter_expansion: int,
        adapter_dropout: float,
    ) -> list[nn.Module]:
        layers: list[nn.Module] = [
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(adapter_dropout),
        ]
        for _ in range(adapter_depth - 1):
            layers.append(
                ResidualMLPBlock(
                    hidden_dim,
                    expansion=adapter_expansion,
                    dropout=adapter_dropout,
                )
            )
        return layers


class RandomFourierFeatures(nn.Module):
    """Fixed random Fourier feature map for low-dimensional coordinates."""

    def __init__(self, in_dim: int, num_features: int, sigma: float, seed: int):
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be > 0")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")

        generator = torch.Generator()
        generator.manual_seed(int(seed))
        weight = torch.randn(in_dim, num_features, generator=generator) * float(sigma)
        bias = torch.rand(num_features, generator=generator) * (2.0 * math.pi)
        self.register_buffer("weight", weight, persistent=True)
        self.register_buffer("bias", bias, persistent=True)

    @property
    def out_dim(self) -> int:
        return int(self.weight.shape[1]) * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase = (2.0 * math.pi) * (x @ self.weight) + self.bias
        return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)


class LearnedFourierFeatures(nn.Module):
    """Trainable Fourier feature map for low-dimensional coordinates."""

    def __init__(self, in_dim: int, num_features: int, sigma: float, seed: int):
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be > 0")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")

        generator = torch.Generator()
        generator.manual_seed(int(seed))
        weight = torch.randn(in_dim, num_features, generator=generator) * float(sigma)
        bias = torch.rand(num_features, generator=generator) * (2.0 * math.pi)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    @property
    def out_dim(self) -> int:
        return int(self.weight.shape[1]) * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase = (2.0 * math.pi) * (x @ self.weight) + self.bias
        return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)


class LocationScaleTower(nn.Module):
    """Residual sub-tower for one geographic feature scale."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        depth: int,
        expansion: int,
        dropout: float,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, expansion=expansion, dropout=dropout) for _ in range(depth)]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(features)
        for block in self.blocks:
            x = block(x)
        return x


class LocationTower(nn.Module):
    """Geographic tower: lat/lon -> xyz (+ optional Fourier features) -> per-scale towers -> shared space."""

    def __init__(
        self,
        out_dim: int,
        use_rff: bool,
        use_learned_rff: bool,
        rff_seed: int,
        rff_feature_sizes: Sequence[int] | None,
        rff_sigmas: Sequence[float] | None,
        hidden_dim: int,
        depth: int,
        expansion: int,
        dropout: float,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("location tower depth must be >= 1")
        if use_rff and use_learned_rff:
            raise ValueError("use_rff and use_learned_rff are mutually exclusive")
        if rff_feature_sizes is not None or rff_sigmas is not None:
            if rff_feature_sizes is None or rff_sigmas is None:
                raise ValueError("rff_feature_sizes and rff_sigmas must both be provided together")
            if len(rff_feature_sizes) != len(rff_sigmas):
                raise ValueError("rff_feature_sizes and rff_sigmas must have matching lengths")
            if any(int(v) <= 0 for v in rff_feature_sizes):
                raise ValueError("rff_feature_sizes entries must all be > 0")
            if any(float(v) <= 0 for v in rff_sigmas):
                raise ValueError("rff_sigmas entries must all be > 0")
            rff_specs = [
                (int(features), float(sigma))
                for features, sigma in zip(rff_feature_sizes, rff_sigmas, strict=True)
            ]
        else:
            rff_specs = []
            if use_rff or use_learned_rff:
                raise ValueError("List-based Fourier feature settings are required when Fourier features are enabled")
        if len(rff_specs) == 0 and (use_rff or use_learned_rff):
            raise ValueError("At least one Fourier feature block is required when Fourier features are enabled")
        if len(rff_specs) > 0 and not (use_rff or use_learned_rff):
            raise ValueError("Fourier feature blocks require use_rff or use_learned_rff")

        feature_cls = None
        if use_rff:
            feature_cls = RandomFourierFeatures
        elif use_learned_rff:
            feature_cls = LearnedFourierFeatures

        self.rff_layers = nn.ModuleList()
        if feature_cls is not None:
            for idx, (features, sigma) in enumerate(rff_specs):
                rff_layer = feature_cls(
                    in_dim=3,
                    num_features=int(features),
                    sigma=float(sigma),
                    seed=int(rff_seed) + idx,
                )
                self.rff_layers.append(rff_layer)
        self.rff = self.rff_layers[0] if len(self.rff_layers) > 0 else None
        self.scale_towers = nn.ModuleList()
        if len(self.rff_layers) == 0:
            self.scale_towers.append(
                LocationScaleTower(
                    in_dim=3,
                    hidden_dim=hidden_dim,
                    depth=depth,
                    expansion=expansion,
                    dropout=dropout,
                )
            )
        else:
            for rff_layer in self.rff_layers:
                self.scale_towers.append(
                    LocationScaleTower(
                        in_dim=3 + rff_layer.out_dim,
                        hidden_dim=hidden_dim,
                        depth=depth,
                        expansion=expansion,
                        dropout=dropout,
                    )
                )
        self.scale_gate = None
        if len(self.scale_towers) > 1:
            self.scale_gate = nn.Linear(3, len(self.scale_towers))
            nn.init.zeros_(self.scale_gate.weight)
            nn.init.zeros_(self.scale_gate.bias)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, unit_xyz: torch.Tensor) -> torch.Tensor:
        if len(self.rff_layers) == 0:
            hidden = self.scale_towers[0](unit_xyz)
        else:
            hidden_by_scale = []
            for scale_tower, rff_layer in zip(self.scale_towers, self.rff_layers, strict=True):
                scale_features = torch.cat((unit_xyz, rff_layer(unit_xyz)), dim=-1)
                hidden_by_scale.append(scale_tower(scale_features))
            hidden_stack = torch.stack(hidden_by_scale, dim=1)
            if self.scale_gate is None:
                hidden = hidden_stack[:, 0, :]
            else:
                gate_logits = self.scale_gate(unit_xyz)
                gate_weights = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)
                hidden = torch.sum(hidden_stack * gate_weights, dim=1)
        return self.out_proj(hidden)


class GeoLightningModule(L.LightningModule):
    """Dual-source contrastive model with a shared image trunk and source-specific adapters."""

    _SUPPORTED_SOURCES = ("mp16", "osv5m")

    def __init__(
        self,
        embed_dim: int,
        image_embed_dim: int | None,
        osv_image_embed_dim: int | None,
        lr: float,
        weight_decay: float,
        total_steps: int,
        warmup_steps: int = 0,
        output_dim: int = 512,
        shared_image_hidden_dim: int = 1024,
        shared_image_depth: int = 3,
        image_trunk_depth: int | None = None,
        shared_image_expansion: int = 2,
        shared_image_dropout: float = 0.15,
        image_adapter_depth: int = 1,
        image_adapter_expansion: int = 2,
        image_adapter_dropout: float = 0.15,
        use_osv_image_tower: bool = True,
        loc_hidden_dim: int = 1024,
        loc_depth: int = 3,
        loc_expansion: int = 2,
        loc_dropout: float = 0.10,
        use_rff: bool = False,
        use_learned_rff: bool = False,
        rff_seed: int = 21,
        rff_feature_sizes: Sequence[int] | None = None,
        rff_sigmas: Sequence[float] | None = None,
        logit_scale_init: float = 10.0,
        logit_scale_max: float = 100.0,
        logit_bias_init: float = -10.0,
        loss_weight_image_loc: float = 1.0,
        loss_weight_osv_image_loc: float = 1.0,
        **unused_checkpoint_hparams: Any,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=("unused_checkpoint_hparams",))

        if output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        if logit_scale_init <= 0:
            raise ValueError("logit_scale_init must be > 0")
        if logit_scale_max <= 0:
            raise ValueError("logit_scale_max must be > 0")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")

        # The bundled checkpoint records the shared trunk depth under this
        # legacy name. Use it when present so its residual-block weights align.
        if image_trunk_depth is not None:
            shared_image_depth = int(image_trunk_depth)

        self.embed_dim = int(embed_dim)
        self.image_embed_dim = int(image_embed_dim) if image_embed_dim is not None else self.embed_dim
        self.osv_image_embed_dim = (
            int(osv_image_embed_dim) if osv_image_embed_dim is not None else self.embed_dim
        )
        if self.image_embed_dim <= 0:
            raise ValueError("image_embed_dim must be > 0")
        if self.osv_image_embed_dim <= 0:
            raise ValueError("osv_image_embed_dim must be > 0")
        if self.image_embed_dim != self.embed_dim:
            raise ValueError("image_embed_dim must match embed_dim for source-tagged mixed-batch training")
        if self.osv_image_embed_dim != self.embed_dim:
            raise ValueError("osv_image_embed_dim must match embed_dim for source-tagged mixed-batch training")

        self.output_dim = int(output_dim)
        self.has_osv_image_tower = bool(use_osv_image_tower)

        if loss_weight_image_loc < 0:
            raise ValueError("loss_weight_image_loc must be >= 0")
        if loss_weight_osv_image_loc < 0:
            raise ValueError("loss_weight_osv_image_loc must be >= 0")
        if float(loss_weight_image_loc) <= 0.0 and float(loss_weight_osv_image_loc) <= 0.0:
            raise ValueError("At least one image/location loss weight must be > 0")
        if not self.has_osv_image_tower and float(loss_weight_osv_image_loc) > 0.0:
            raise ValueError("loss_weight_osv_image_loc requires use_osv_image_tower=True")

        if int(shared_image_hidden_dim) <= 0:
            raise ValueError("shared_image_hidden_dim must be > 0")
        if int(shared_image_depth) <= 0:
            raise ValueError("shared_image_depth must be > 0")
        if int(shared_image_expansion) <= 0:
            raise ValueError("shared_image_expansion must be > 0")
        if int(image_adapter_depth) <= 0:
            raise ValueError("image_adapter_depth must be > 0")
        if int(image_adapter_expansion) <= 0:
            raise ValueError("image_adapter_expansion must be > 0")
        if not (0.0 <= float(shared_image_dropout) < 1.0):
            raise ValueError("shared_image_dropout must be in [0, 1)")
        if not (0.0 <= float(image_adapter_dropout) < 1.0):
            raise ValueError("image_adapter_dropout must be in [0, 1)")

        image_sources = ("mp16", "osv5m") if self.has_osv_image_tower else ("mp16",)
        self.image_tower = SharedImageTower(
            in_dim=self.image_embed_dim,
            out_dim=self.output_dim,
            trunk_hidden_dim=int(shared_image_hidden_dim),
            trunk_depth=int(shared_image_depth),
            trunk_expansion=int(shared_image_expansion),
            trunk_dropout=float(shared_image_dropout),
            adapter_depth=int(image_adapter_depth),
            adapter_expansion=int(image_adapter_expansion),
            adapter_dropout=float(image_adapter_dropout),
            sources=image_sources,
        )
        self.location_tower = LocationTower(
            out_dim=self.output_dim,
            use_rff=bool(use_rff),
            use_learned_rff=bool(use_learned_rff),
            rff_seed=int(rff_seed),
            rff_feature_sizes=None if rff_feature_sizes is None else tuple(int(v) for v in rff_feature_sizes),
            rff_sigmas=None if rff_sigmas is None else tuple(float(v) for v in rff_sigmas),
            hidden_dim=int(loc_hidden_dim),
            depth=int(loc_depth),
            expansion=int(loc_expansion),
            dropout=float(loc_dropout),
        )

        self.logit_scale = nn.Parameter(torch.tensor(float(math.log(logit_scale_init)), dtype=torch.float32))
        self.logit_bias = nn.Parameter(torch.tensor(float(logit_bias_init), dtype=torch.float32))

        self._val_preds = []
        self._val_targets = []
        self._val_preds_by_source: dict[str, list[torch.Tensor]] = {}
        self._val_targets_by_source: dict[str, list[torch.Tensor]] = {}
        self.register_buffer(
            "best_val_median_km",
            torch.tensor(float("inf"), dtype=torch.float32),
            persistent=True,
        )

    @staticmethod
    def _unpack_batch(batch):
        if not isinstance(batch, (tuple, list)):
            raise TypeError("Expected batch to be a tuple/list from DataLoader")
        if len(batch) == 2:
            return batch[0], batch[1], None
        if len(batch) >= 3:
            return batch[0], batch[1], batch[2]
        raise ValueError(f"Unexpected batch size: {len(batch)}")

    @staticmethod
    def _latlon_deg_to_unit_xyz(latlon_deg: torch.Tensor) -> torch.Tensor:
        lat_rad = torch.deg2rad(latlon_deg[..., 0])
        lon_rad = torch.deg2rad(latlon_deg[..., 1])
        cos_lat = torch.cos(lat_rad)
        x = cos_lat * torch.cos(lon_rad)
        y = cos_lat * torch.sin(lon_rad)
        z = torch.sin(lat_rad)
        return torch.stack((x, y, z), dim=-1)

    @classmethod
    def _normalize_source_name(cls, raw_source: Any) -> str:
        source_name = str(raw_source or "mp16").strip().lower()
        if source_name not in cls._SUPPORTED_SOURCES:
            raise ValueError(
                f"Unsupported source '{source_name}'. Expected one of: {', '.join(cls._SUPPORTED_SOURCES)}"
            )
        return source_name

    @classmethod
    def _metadata_to_source_names(cls, metadata: Sequence[dict[str, Any]] | None, batch_size: int) -> list[str]:
        if metadata is None:
            return ["mp16"] * batch_size
        if len(metadata) != batch_size:
            raise ValueError(
                f"Metadata length mismatch: expected {batch_size}, got {len(metadata)}"
            )
        return [cls._normalize_source_name(item.get("source", "mp16")) for item in metadata]

    @classmethod
    def _source_masks(
        cls,
        metadata: Sequence[dict[str, Any]] | None,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        sources = cls._metadata_to_source_names(metadata, batch_size=batch_size)
        masks: dict[str, torch.Tensor] = {}
        for source_name in cls._SUPPORTED_SOURCES:
            masks[source_name] = torch.tensor(
                [name == source_name for name in sources],
                device=device,
                dtype=torch.bool,
            )
        return masks

    @classmethod
    def normalize_retrieval_source_name(cls, raw_source: Any) -> str:
        return cls._normalize_source_name(raw_source)

    def encode_image_for_source(self, image_embeddings: torch.Tensor, *, source_name: str) -> torch.Tensor:
        normalized_source = self.normalize_retrieval_source_name(source_name)
        if normalized_source == "osv5m" and not self.has_osv_image_tower:
            raise RuntimeError("OSV image tower requested but use_osv_image_tower=False")
        return F.normalize(self.image_tower(image_embeddings, source_name=normalized_source), dim=-1, eps=1e-8)

    def encode_image(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        return self.encode_image_for_source(image_embeddings, source_name="mp16")

    def encode_osv_image(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        return self.encode_image_for_source(image_embeddings, source_name="osv5m")

    def encode_location(self, latlon_deg: torch.Tensor) -> torch.Tensor:
        unit_xyz = self._latlon_deg_to_unit_xyz(latlon_deg)
        return F.normalize(self.location_tower(unit_xyz), dim=-1, eps=1e-8)

    def _compute_pair_logits(
        self,
        lhs_features: torch.Tensor,
        rhs_features: torch.Tensor,
    ) -> torch.Tensor:
        scale = torch.clamp(torch.exp(self.logit_scale), max=float(self.hparams.logit_scale_max))
        return (scale * (lhs_features @ rhs_features.T)) + self.logit_bias

    @staticmethod
    def _siglip_pairwise_loss(logits: torch.Tensor) -> torch.Tensor:
        batch_size = logits.shape[0]
        targets = torch.eye(batch_size, device=logits.device, dtype=logits.dtype)
        signs = (targets * 2.0) - 1.0
        per_pair_loss = -F.logsigmoid(signs * logits)
        return per_pair_loss.sum(dim=-1).mean()

    @staticmethod
    def _pairwise_accuracy(logits: torch.Tensor) -> torch.Tensor:
        batch_size = logits.shape[0]
        targets = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
        preds = logits > 0.0
        return (preds == targets).to(torch.float32).mean()

    @staticmethod
    def _retrieval_top1(logits: torch.Tensor) -> torch.Tensor:
        batch_size = logits.shape[0]
        top_idx = torch.argmax(logits, dim=-1)
        labels = torch.arange(batch_size, device=logits.device)
        return (top_idx == labels).to(torch.float32).mean()

    @staticmethod
    def _gather_pred_latlon(logits: torch.Tensor, candidate_latlon: torch.Tensor) -> torch.Tensor:
        top_idx = torch.argmax(logits, dim=-1)
        return candidate_latlon[top_idx]

    def _encode_source_embeddings(
        self,
        source_name: str,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if source_name == "mp16":
            return self.encode_image(embeddings)
        if source_name == "osv5m":
            return self.encode_osv_image(embeddings)
        raise ValueError(f"Unsupported source: {source_name}")

    def forward(
        self,
        embeddings: torch.Tensor,
        latlon_deg: torch.Tensor,
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected embedding tensor [B, D], got shape {tuple(embeddings.shape)}")
        if embeddings.shape[1] != self.embed_dim:
            raise ValueError(
                f"Expected embedding dim={self.embed_dim}, got {int(embeddings.shape[1])}"
            )

        batch_size = int(embeddings.shape[0])
        source_masks = self._source_masks(metadata, batch_size=batch_size, device=embeddings.device)
        location_features = self.encode_location(latlon_deg)
        source_outputs: dict[str, dict[str, torch.Tensor]] = {}
        for source_name, mask in source_masks.items():
            if not torch.any(mask):
                continue
            src_embeddings = embeddings[mask]
            src_targets = latlon_deg[mask]
            src_location_features = location_features[mask]
            src_image_features = self._encode_source_embeddings(source_name, src_embeddings)
            logits_image_loc = self._compute_pair_logits(src_image_features, src_location_features)
            source_outputs[source_name] = {
                "mask": mask,
                "targets_deg": src_targets,
                "logits_image_loc": logits_image_loc,
            }
        return {
            "source_outputs": source_outputs,
        }

    def _source_loss_weight(self, source_name: str) -> float:
        if source_name == "mp16":
            return float(self.hparams.loss_weight_image_loc)
        if source_name == "osv5m":
            return float(self.hparams.loss_weight_osv_image_loc)
        raise ValueError(f"Unsupported source: {source_name}")

    def _loss_from_outputs(
        self,
        outputs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        source_outputs = outputs["source_outputs"]
        if not source_outputs:
            raise RuntimeError("No source-specific outputs were produced for this batch")

        weighted_loss_sum = None
        weight_total = 0.0
        metric_weight_total = 0

        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}
        predictions: dict[str, torch.Tensor] = {}
        pair_acc_sum = None
        top1_sum = None

        for source_name, source_data in source_outputs.items():
            logits = source_data["logits_image_loc"]
            targets_deg = source_data["targets_deg"]
            prefix = f"{source_name}_loc"
            count = int(logits.shape[0])
            source_loss = self._siglip_pairwise_loss(logits)
            losses[f"loss_{prefix}"] = source_loss
            predictions[source_name] = self._gather_pred_latlon(logits, targets_deg)

            with torch.no_grad():
                pair_acc = self._pairwise_accuracy(logits)
                top1 = self._retrieval_top1(logits)
                diag = torch.diagonal(logits).mean()
                offdiag = (logits.sum() - torch.diagonal(logits).sum()) / max(
                    logits.numel() - logits.shape[0], 1
                )
            metrics[f"pair_acc_{prefix}"] = pair_acc
            metrics[f"top1_{prefix}"] = top1
            metrics[f"pos_logit_{prefix}"] = diag
            metrics[f"neg_logit_{prefix}"] = offdiag

            metric_weight_total += count
            pair_acc_sum = pair_acc * count if pair_acc_sum is None else pair_acc_sum + (pair_acc * count)
            top1_sum = top1 * count if top1_sum is None else top1_sum + (top1 * count)

            source_weight = self._source_loss_weight(source_name)
            if source_weight > 0.0:
                scaled = source_loss * float(source_weight * count)
                weighted_loss_sum = scaled if weighted_loss_sum is None else weighted_loss_sum + scaled
                weight_total += float(source_weight * count)

        if weighted_loss_sum is None or weight_total <= 0.0:
            raise RuntimeError("No active source loss weights were available for this batch")

        total_loss = weighted_loss_sum / weight_total
        losses["loss"] = total_loss
        if metric_weight_total > 0 and pair_acc_sum is not None and top1_sum is not None:
            metrics["pair_acc_loc"] = pair_acc_sum / metric_weight_total
            metrics["top1_loc"] = top1_sum / metric_weight_total
        return total_loss, losses, metrics, predictions

    def _log_metrics(
        self,
        stage: str,
        losses: dict[str, torch.Tensor],
        metrics: dict[str, torch.Tensor],
        *,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        self.log(f"{stage}/loss", losses["loss"], on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True)
        for name, value in losses.items():
            if name == "loss":
                continue
            self.log(f"{stage}/{name}", value, on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True)
        for name, value in metrics.items():
            self.log(f"{stage}/{name}", value, on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True)

    def training_step(self, batch, batch_idx):
        embeddings, targets_deg, metadata = self._unpack_batch(batch)
        outputs = self(embeddings, targets_deg, metadata=metadata)
        loss, losses, metrics, _predictions = self._loss_from_outputs(outputs)

        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            self._log_metrics("train", losses, metrics, on_step=True, on_epoch=False)
            scale = torch.clamp(torch.exp(self.logit_scale), max=float(self.hparams.logit_scale_max))
            self.log("train/logit_scale", scale, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log(
                "train/logit_bias",
                self.logit_bias.detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )
        if trainer is not None and trainer.optimizers:
            lr = float(trainer.optimizers[0].param_groups[0]["lr"])
            self.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_preds = []
        self._val_targets = []
        self._val_preds_by_source = {source_name: [] for source_name in self._SUPPORTED_SOURCES}
        self._val_targets_by_source = {source_name: [] for source_name in self._SUPPORTED_SOURCES}

    def validation_step(self, batch, batch_idx):
        embeddings, targets_deg, metadata = self._unpack_batch(batch)
        outputs = self(embeddings, targets_deg, metadata=metadata)
        loss, losses, metrics, predictions = self._loss_from_outputs(outputs)

        pred_deg = torch.empty_like(targets_deg)
        for source_name, source_data in outputs["source_outputs"].items():
            mask = source_data["mask"]
            source_targets = source_data["targets_deg"]
            pred_deg[mask] = predictions[source_name]
            self._val_preds_by_source[source_name].append(predictions[source_name].detach().cpu())
            self._val_targets_by_source[source_name].append(source_targets.detach().cpu())

        if getattr(self, "_trainer", None) is not None:
            self._log_metrics("val", losses, metrics, on_step=False, on_epoch=True)

        self._val_preds.append(pred_deg.detach().cpu())
        self._val_targets.append(targets_deg.detach().cpu())
        return loss

    def _median_km(self, pred_list: list[torch.Tensor], target_list: list[torch.Tensor]) -> float:
        if not pred_list or not target_list:
            return float("inf")
        pred_np = torch.cat(pred_list, dim=0).numpy()
        tgt_np = torch.cat(target_list, dim=0).numpy()
        dists = geodesic_km(pred_np[:, 0], pred_np[:, 1], tgt_np[:, 0], tgt_np[:, 1])
        return float(np.median(dists))

    def on_validation_epoch_end(self):
        median_km = self._median_km(self._val_preds, self._val_targets)
        median_by_source = {
            source_name: self._median_km(
                self._val_preds_by_source.get(source_name, []),
                self._val_targets_by_source.get(source_name, []),
            )
            for source_name in self._SUPPORTED_SOURCES
        }

        if getattr(self, "_trainer", None) is not None:
            self.log("val/median_km", median_km, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            for source_name, source_median in median_by_source.items():
                if np.isfinite(source_median):
                    self.log(
                        f"val/median_km_{source_name}_loc",
                        source_median,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                    )
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
