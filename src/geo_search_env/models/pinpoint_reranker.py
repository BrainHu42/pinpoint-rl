"""Pinpoint's attention reranker (the submission's final model), ported for batched offline inference.

Matches pinpoint-yogi `attention_reranker.AttentionInterpolateLightningModule` in eval mode, without Lightning:
  1. The query's raw SigLIP2 embedding is L2-normalized and passed through the contrastive image tower (mp16 adapter).
  2. Three exact searches over the attention retrieval index (4.08M MP16 photos; bucket 99 is not in it): the top 8 by
     contrastive image similarity, the top 2 by raw SigLIP2 similarity and the top 2 by similarity to each photo's GPS
     embedding. Their union (<= 12 photos) is the candidate pool.
  3. Each candidate is scored on its own: a 4-layer transformer over [CLS, query, candidate image, candidate GPS, OSV
     support, candidate raw image] tokens, read out from CLS. The prediction is the top-scoring candidate's coordinate.

Differences from the original are confined to how the index is read: it is streamed from disk a chunk at a time
(same float32 dequantization, same fp16 search) instead of being made GPU-resident, so it fits beside other jobs.
Optionally, index photos by the query's photographer are excluded from all three searches (our benchmark rule).

Checked against the original code (pinpoint-yogi 27a9bfe, Lightning stubbed, index fed lazily) on 320 benchmark photos:
identical query encodings, candidate rows and rank/flag features on the full index, and identical bf16 scores given
the same features.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..paths import PINPOINT_ROOT
from ._pinpoint_torch import _load_image_tower

RERANKER_CHECKPOINT = PINPOINT_ROOT / "exp/attention_reranker/checkpoints/ckpt_best.pt"
CONTRASTIVE_CHECKPOINT = PINPOINT_ROOT / "exp/contrastive_retrieval/checkpoints/ckpt_best.pt"
INDEX_ROOT = PINPOINT_ROOT / "exp/contrastive_retrieval/checkpoints/attention_retrieval_index"
MODALITIES = ("image", "raw_image", "gps")
# Retrieval feature columns per token: [similarity, rank, in-shortlist] of that modality (feature order is
# [sim_image, sim_raw, sim_gps, rank_image, rank_raw, rank_gps, in_image, in_raw, in_gps]).
FEATURE_GROUPS = {"image": (0, 3, 6), "raw_image": (1, 4, 7), "gps": (2, 5, 8)}
EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class RerankedCandidates:
    """Each query's candidate pool, sorted by reranker score (best first); empty slots have row -1."""

    rows: np.ndarray  # [Q, C] attention-index rows
    cache_rows: np.ndarray  # [Q, C] rows of the MP16 SigLIP2 embedding cache (`data/pinpoint/mp16-embed`)
    latlon: np.ndarray  # [Q, C, 2] degrees, NaN for empty slots
    scores: np.ndarray  # [Q, C] reranker scores, -inf for empty slots
    features: np.ndarray  # [Q, C, 9] retrieval features, in FEATURE_GROUPS order

    @property
    def valid(self) -> np.ndarray:
        return self.rows >= 0

    @property
    def top1(self) -> np.ndarray:
        return self.latlon[:, 0]


class _RerankerHead(nn.Module):
    """The scoring layers of the checkpoint (contrastive model and unused projections stripped)."""

    def __init__(self, *, output_dim: int, raw_dim: int, hidden: int, heads: int, layers: int, ffn: int) -> None:
        super().__init__()

        def projection(width: int) -> nn.Sequential:
            return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU())

        self.query_proj = projection(output_dim)
        self.candidate_image_fused_proj = projection(output_dim + 3)
        self.gps_fused_proj = projection(output_dim + 3)
        self.raw_image_fused_proj = projection(raw_dim + 3)
        self.osv_support_proj = projection(output_dim + 4)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=ffn, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.interaction_encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(hidden))
        self.score_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(0.0), nn.Linear(hidden // 2, 1))

    def forward(self, tokens: list[torch.Tensor]) -> torch.Tensor:
        """`tokens`: [M, hidden] each (query, image, gps, osv, raw); returns [M] scores."""

        sequence = torch.cat((self.cls_token.expand(len(tokens[0]), -1, -1), torch.stack(tokens, dim=1)), dim=1)
        return self.score_head(self.interaction_encoder(sequence)[:, 0, :]).squeeze(-1)


def _discover_index(root: Path = INDEX_ROOT) -> Path:
    manifests = sorted(root.glob("*/manifest.json"))
    if len(manifests) != 1:
        raise FileNotFoundError(f"expected exactly one attention retrieval index under {root}, found {len(manifests)}")
    return manifests[0].parent


def _unit_xyz(latlon_deg: np.ndarray) -> np.ndarray:
    lat, lon = np.deg2rad(np.asarray(latlon_deg, dtype=np.float64)).T
    return np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)


class PinpointReranker:
    """Batched inference of the attention reranker over its own retrieval index."""

    def __init__(self, *, device: str = "cuda", query_batch: int = 1_024, index_dir: Path | None = None) -> None:
        self.device = torch.device(device)
        self.query_batch = query_batch
        checkpoint = torch.load(RERANKER_CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
        hp = checkpoint["hyper_parameters"]
        self.targets = {"image": int(hp["num_neighbors_image"]), "raw_image": int(hp["num_neighbors_raw_image"]), "gps": int(hp["num_neighbors_gps"])}
        self.osv_k, self.osv_radius_km = int(hp["osv_support_neighbors_k"]), float(hp["osv_support_radius_km"])
        self.multiplier, self.chunk = int(hp["modality_pool_multiplier"]), int(hp["retrieval_chunk_size"])
        if not (hp["use_raw_image_features"] and hp["use_retrieval_features"] and hp["use_osv_support_token"]):
            raise ValueError("this port supports the submission's configuration (raw image, retrieval features, OSV support)")

        # The reranker checkpoint embeds a copy of the contrastive model; its image tower is identical to the
        # standalone contrastive checkpoint's, which `_load_image_tower` reads.
        self.tower, raw_dim, output_dim, _ = _load_image_tower(CONTRASTIVE_CHECKPOINT)
        self.tower.to(self.device)
        self.head = _RerankerHead(
            output_dim=output_dim, raw_dim=raw_dim, hidden=int(hp["reranker_hidden_dim"]), heads=int(hp["reranker_num_heads"]),
            layers=int(hp["reranker_num_layers"]), ffn=int(hp["reranker_ffn_dim"]),
        )
        skip = ("contrastive_model.", "query_raw_image_proj.", "best_val_median_km")  # unused at inference
        state = {k: v for k, v in checkpoint["state_dict"].items() if not k.startswith(skip)}
        if any("_metadata_proj." in k for k in state):
            raise ValueError("checkpoint has metadata projections, which this port drops (they are zero in the submission)")
        self.head.load_state_dict(state, strict=True)
        self.head.to(self.device).eval()
        tower_state = {k.removeprefix("contrastive_model.image_tower."): v for k, v in checkpoint["state_dict"].items() if k.startswith("contrastive_model.image_tower.")}
        if any(not torch.equal(tower_state[k], v.cpu()) for k, v in self.tower.state_dict().items()):
            raise ValueError("the reranker's contrastive image tower differs from the standalone contrastive checkpoint")
        # Reranker in bf16 on Ampere and newer (as the original), float32 otherwise; the tower in fp16 on CUDA.
        cuda = self.device.type == "cuda"
        self.feature_dtype = torch.bfloat16 if cuda and torch.cuda.get_device_capability(self.device)[0] >= 8 else torch.float32

        self.index_dir = index_dir or _discover_index()
        manifest = json.loads((self.index_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("version") != 6 or manifest.get("retrieval_db_source") != "mp16":
            raise ValueError(f"unsupported attention retrieval index: {self.index_dir}")
        count, raw_width = manifest["shapes"]["raw_image_embeddings"]

        def memmap(name: str, dtype, shape):
            return np.memmap(self.index_dir / manifest["files"][name], dtype=dtype, mode="r", shape=shape)

        self.image = memmap("image_embeddings", np.float16, (count, output_dim))
        self.gps = memmap("gps_embeddings", np.float16, (count, output_dim))
        self.raw = memmap("raw_image_embeddings", np.int8, (count, raw_width))
        self.raw_scales = memmap("raw_image_scales", np.float16, (count,))
        self.latlon = memmap("latlon_deg", np.float32, (count, 2))
        self.cache_indices = np.fromfile(self.index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
        osv_root = self.index_dir / "osv_support_cache"
        osv = json.loads((osv_root / "manifest.json").read_text(encoding="utf-8"))
        if osv.get("embedding_space") != "contrastive_osv":
            raise ValueError("expected an OSV support cache of contrastive embeddings")
        self.osv_embeddings = np.memmap(osv_root / osv["files"]["embeddings"], dtype=np.float16, mode="r", shape=tuple(osv["shapes"]["embeddings"]))
        self.osv_latlon = np.memmap(osv_root / osv["files"]["latlon_deg"], dtype=np.float32, mode="r", shape=tuple(osv["shapes"]["latlon_deg"]))
        self._osv_tree = None

    def _autocast(self, dtype):
        return torch.autocast(device_type="cuda", dtype=dtype) if self.device.type == "cuda" else nullcontext()

    def _dequantize(self, start: int, end: int | None = None, rows: np.ndarray | None = None) -> torch.Tensor:
        """Raw SigLIP2 index embeddings as the original makes them resident: int8 x scale, NaN-free, unit norm, fp16."""

        values = self.raw[start:end] if rows is None else self.raw[rows]
        scales = self.raw_scales[start:end] if rows is None else self.raw_scales[rows]
        raw = torch.as_tensor(np.asarray(values), device=self.device).float()
        raw.mul_(torch.as_tensor(np.asarray(scales), device=self.device).float()[:, None])
        torch.nan_to_num_(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return raw.div_(raw.norm(dim=1, keepdim=True).clamp_(min=1e-8)).half()

    @torch.inference_mode()
    def encode_queries(self, embeddings: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """(unit raw SigLIP2 embedding [Q, 1536], unit contrastive embedding [Q, 256]), both float32."""

        raw = torch.nan_to_num(torch.as_tensor(np.asarray(embeddings, dtype=np.float32), device=self.device), nan=0.0, posinf=0.0, neginf=0.0)
        raw = F.normalize(raw, dim=-1, eps=1e-8)
        with self._autocast(torch.float16):
            projected = F.normalize(self.tower(raw, "mp16"), dim=-1, eps=1e-8)
        return raw, F.normalize(torch.nan_to_num(projected.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=1e-8)

    @torch.inference_mode()
    def _search(self, raw: torch.Tensor, projected: torch.Tensor, query_author: np.ndarray | None, index_author: np.ndarray | None):
        """Top hits per modality, streaming the index once; returns {modality: (rows [Q, k], scores [Q, k])}.

        Many index scores tie exactly (photos at one coordinate share a GPS embedding; near-duplicates share an fp16
        image score), and the reranker reads each hit's rank, so tie order matters. This follows the original's
        sequence of operations to break ties the same way: a pool of `multiplier` x the target per modality from
        fp16 scores over chunks of `retrieval_chunk_size` rows (running top-k of [best so far, chunk top-k]), cut
        to the target and re-sorted on the CPU in float32.
        """

        queries = {"image": projected.half(), "raw_image": raw.half(), "gps": projected.half()}
        pools = {m: self.targets[m] * self.multiplier for m in MODALITIES}
        best = {
            m: (torch.full((len(raw), pools[m]), float("-inf"), device=self.device, dtype=torch.float16), torch.zeros((len(raw), pools[m]), dtype=torch.long, device=self.device))
            for m in MODALITIES
        }
        q_author = None if query_author is None else torch.as_tensor(query_author, device=self.device)
        count = len(self.latlon)
        for start in range(0, count, self.chunk):
            end = min(start + self.chunk, count)
            blocks = {
                "image": torch.as_tensor(np.asarray(self.image[start:end]), device=self.device),
                "raw_image": self._dequantize(start, end),
                "gps": torch.as_tensor(np.asarray(self.gps[start:end]), device=self.device),
            }
            excluded = None if q_author is None else torch.as_tensor(index_author[start:end], device=self.device)
            for m in MODALITIES:
                for b in range(0, len(raw), self.query_batch):
                    scores = queries[m][b : b + self.query_batch] @ blocks[m].T
                    if excluded is not None:
                        scores.masked_fill_(excluded.unsqueeze(0) == q_author[b : b + self.query_batch].unsqueeze(1), float("-inf"))
                    top_values, top_indices = torch.topk(scores, min(pools[m], end - start), dim=-1)
                    values, indices = best[m][0][b : b + self.query_batch], best[m][1][b : b + self.query_batch]
                    merged_values, keep = torch.topk(torch.cat((values, top_values), dim=-1), pools[m], dim=-1)
                    best[m][0][b : b + self.query_batch] = merged_values
                    best[m][1][b : b + self.query_batch] = torch.gather(torch.cat((indices, top_indices + start), dim=-1), -1, keep)
            if (start // self.chunk) % 8 == 0:
                print(f"  index {end}/{count}", flush=True)
        found = {}
        for m, (values, indices) in best.items():
            values, indices = values[:, : self.targets[m]].cpu().float(), indices[:, : self.targets[m]].cpu().to(torch.int32)
            order = torch.argsort(values, dim=1, descending=True)
            found[m] = (torch.gather(indices, 1, order).numpy().astype(np.int64), torch.gather(values, 1, order).numpy())
        return found

    def _union(self, found) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Deduplicated pool (rows [Q, C], per-modality rank [Q, C, 3], in-shortlist flag [Q, C, 3]), as the original."""

        capacity = sum(self.targets.values())
        queries = len(next(iter(found.values()))[0])
        rows = np.full((queries, capacity), -1, dtype=np.int64)
        ranks = np.ones((queries, capacity, 3), dtype=np.float32)
        flags = np.zeros((queries, capacity, 3), dtype=np.float32)
        for q in range(queries):
            slots: dict[int, int] = {}
            for m, name in enumerate(MODALITIES):
                indices, values = found[name]
                denominator = max(self.targets[name] - 1, 1)
                for position, (row, value) in enumerate(zip(indices[q], values[q])):
                    if not np.isfinite(value):
                        continue
                    slot = slots.setdefault(int(row), len(slots))
                    rows[q, slot] = row
                    ranks[q, slot, m] = min(ranks[q, slot, m], position / denominator)
                    flags[q, slot, m] = 1.0
        return rows, ranks, flags

    def _osv_support(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For each index row, its k nearest OSV-5M photos within the radius (index -1 beyond it) and their distance."""

        if self._osv_tree is None:
            from scipy.spatial import cKDTree

            print("building the OSV support KD-tree", flush=True)
            self._osv_tree = cKDTree(_unit_xyz(np.asarray(self.osv_latlon)), leafsize=64, compact_nodes=True, balanced_tree=True)
        unique, inverse = np.unique(rows, return_inverse=True)
        chord, neighbors = self._osv_tree.query(_unit_xyz(np.asarray(self.latlon[unique])), k=self.osv_k, workers=-1)
        chord, neighbors = chord.reshape(len(unique), -1), neighbors.reshape(len(unique), -1)
        distance = (2.0 * np.arcsin(np.clip(chord * 0.5, 0.0, 1.0)) * EARTH_RADIUS_KM).astype(np.float32)
        valid = np.isfinite(distance) & (distance <= self.osv_radius_km)
        # The original caches distances as float16 per index row.
        return np.where(valid, neighbors, -1)[inverse], distance.astype(np.float16).astype(np.float32)[inverse]

    @torch.inference_mode()
    def _score(self, raw, projected, rows, ranks, flags) -> tuple[np.ndarray, np.ndarray]:
        """Reranker scores [Q, C] (-inf on empty slots) and the 9 retrieval features [Q, C, 9]."""

        valid = rows >= 0
        safe = np.where(valid, rows, 0)  # the original also scores empty slots on row 0, then masks them
        all_scores, all_features = [], []
        for b in range(0, len(rows), self.query_batch):
            r, v = safe[b : b + self.query_batch], valid[b : b + self.query_batch]
            q_raw, q_img = raw[b : b + self.query_batch], projected[b : b + self.query_batch]
            n, c = r.shape
            flat = r.reshape(-1)
            mask = torch.as_tensor(v, device=self.device)
            # Candidate embeddings travel through the reranker's feature dtype, as in the original.
            image = torch.as_tensor(np.asarray(self.image[flat]), device=self.device).to(self.feature_dtype).view(n, c, -1)
            gps = torch.as_tensor(np.asarray(self.gps[flat]), device=self.device).to(self.feature_dtype).view(n, c, -1)
            raw_c = self._dequantize(0, rows=flat).to(self.feature_dtype).view(n, c, -1)
            sim_image = (q_img.unsqueeze(1) * image.float()).sum(-1)
            sim_gps = (q_img.unsqueeze(1) * gps.float()).sum(-1)
            sim_raw = (q_raw.unsqueeze(1) * raw_c.float()).sum(-1)
            features = torch.cat((
                torch.stack((sim_image, sim_raw, sim_gps), dim=-1),
                torch.as_tensor(ranks[b : b + self.query_batch], device=self.device),
                torch.as_tensor(flags[b : b + self.query_batch], device=self.device),
            ), dim=-1).masked_fill(~mask.unsqueeze(-1), 0.0)

            neighbors, distance = self._osv_support(flat)
            support_valid = torch.as_tensor(v.reshape(-1)[:, None] & (neighbors >= 0), device=self.device)
            support = torch.zeros((n * c, self.osv_k, q_img.shape[-1]), device=self.device)
            if support_valid.any():
                wanted = neighbors[support_valid.cpu().numpy()]
                unique, inverse = np.unique(wanted, return_inverse=True)
                encoded = F.normalize(torch.as_tensor(np.asarray(self.osv_embeddings[unique]), device=self.device).float(), dim=-1, eps=1e-8)
                support[support_valid] = encoded[torch.as_tensor(inverse, device=self.device)]
            distance = torch.as_tensor(distance, device=self.device)

            group = {name: features[..., list(columns)] for name, columns in FEATURE_GROUPS.items()}
            with self._autocast(self.feature_dtype):
                q_flat = q_img.unsqueeze(1).expand(-1, c, -1).reshape(n * c, -1)
                similarity = (support * q_flat.unsqueeze(1)).sum(-1)
                weights = torch.softmax(similarity.masked_fill(~support_valid, -1e4), dim=1)
                weights = torch.where(support_valid, weights, torch.zeros_like(weights))
                weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
                pooled = (weights.unsqueeze(-1) * support).sum(1)
                counts = support_valid.float().sum(1)
                best = torch.where(support_valid, similarity, -1e4).amax(1)
                best = torch.where(counts > 0, best, torch.zeros_like(best))
                mean = torch.where(support_valid, similarity, 0.0).sum(1) / counts.clamp(min=1.0)
                nearest = torch.where(support_valid, distance, torch.full_like(distance, self.osv_radius_km)).amin(1)
                stats = torch.stack((best, mean, 1.0 - torch.clamp(nearest / self.osv_radius_km, 0.0, 1.0), counts / self.osv_k), dim=-1)
                osv_token = self.head.osv_support_proj(torch.cat((pooled, stats), dim=-1)).masked_fill(~mask.reshape(-1, 1), 0.0)

                tokens = [
                    self.head.query_proj(q_img).unsqueeze(1).expand(-1, c, -1).reshape(n * c, -1),
                    self.head.candidate_image_fused_proj(torch.cat((image.float(), group["image"]), dim=-1)).reshape(n * c, -1),
                    self.head.gps_fused_proj(torch.cat((gps.float(), group["gps"]), dim=-1)).reshape(n * c, -1),
                    osv_token,
                    self.head.raw_image_fused_proj(torch.cat((raw_c.float(), group["raw_image"]), dim=-1)).reshape(n * c, -1),
                ]
                scores = self.head(tokens).float().view(n, c)
            all_scores.append(scores.masked_fill(~mask, float("-inf")).cpu().numpy())
            all_features.append(features.cpu().numpy())
        return np.concatenate(all_scores), np.concatenate(all_features)

    def rank(self, embeddings: np.ndarray, *, query_author: np.ndarray | None = None, index_author: np.ndarray | None = None) -> RerankedCandidates:
        """Rerank each query's pool. `embeddings`: raw SigLIP2 [Q, 1536]. With `query_author` and `index_author`
        (an author id per index row, e.g. `mp16_author[cache_indices]`), index photos by the same author are excluded."""

        if (query_author is None) != (index_author is None):
            raise ValueError("pass both query_author and index_author, or neither")
        raw, projected = self.encode_queries(embeddings)
        rows, ranks, flags = self._union(self._search(raw, projected, query_author, index_author))
        scores, features = self._score(raw, projected, rows, ranks, flags)
        order = np.argsort(-scores, axis=1, kind="stable")
        rows, scores, features = np.take_along_axis(rows, order, 1), np.take_along_axis(scores, order, 1), np.take_along_axis(features, order[..., None], 1)
        valid = rows >= 0
        latlon = np.where(valid[..., None], np.asarray(self.latlon[np.where(valid, rows, 0)]), np.nan)
        return RerankedCandidates(
            rows=rows, cache_rows=np.where(valid, self.cache_indices[np.where(valid, rows, 0)], -1), latlon=latlon, scores=scores, features=features
        )
