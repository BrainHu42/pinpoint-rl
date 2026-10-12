"""Set head for Pinpoint's reranker (PINPOINT_SET_PLAN.md step 4): one small transformer over the whole candidate pool, on top of the
reranker's per-candidate CLS states. Torch only, so the Pinpoint training code (pinpoint-main) and our offline evaluation share it.

Inputs per candidate: the reranker's CLS state, its score, and scalars derived from the pool (score - max, softmax at temperature 0.2,
log of the softmax mass within 1 / 5 / 25 / 100 km, the 9 retrieval features, in-pool flag). Attention between candidates gets a
learned bias per distance bucket (edges 1, 5, 25, 100, 500, 2500 km). Output logits = score / 0.2 + a zero-initialised residual, so an
untrained head reproduces the reranker's ranking. Trained with the expected-hit loss (`expected_hit_loss`)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

TEMPERATURE = 0.2
SUPPORT_KM = (1.0, 5.0, 25.0, 100.0)
DIST_EDGES_KM = (1.0, 5.0, 25.0, 100.0, 500.0, 2500.0)
EARTH_RADIUS_KM = 6371.0088
N_SCALAR = 3 + len(SUPPORT_KM) + 9 + 1


def pairwise_km(latlon_deg: torch.Tensor) -> torch.Tensor:
    """[B, N, 2] degrees -> [B, N, N] great-circle km."""

    lat, lon = torch.deg2rad(latlon_deg[..., 0]), torch.deg2rad(latlon_deg[..., 1])
    xyz = torch.stack((torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)), -1)
    cos = (xyz.unsqueeze(2) * xyz.unsqueeze(1)).sum(-1).clamp(-1.0, 1.0)
    return torch.arccos(cos) * EARTH_RADIUS_KM


class _Layer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.bucket_bias = nn.Embedding(len(DIST_EDGES_KM) + 1, heads)
        nn.init.zeros_(self.bucket_bias.weight)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, buckets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        bias = self.bucket_bias(buckets).permute(0, 3, 1, 2).to(q.dtype)  # [B, H, N, N]
        bias = bias.masked_fill(~mask[:, None, None, :], -1e4)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.out(attn.transpose(1, 2).reshape(b, n, d))
        return x + self.ffn(self.norm2(x))


class SetHead(nn.Module):
    def __init__(self, hidden_dim: int = 512, dim: int = 128, heads: int = 4, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_in = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, dim))
        self.scalar_in = nn.Sequential(nn.LayerNorm(N_SCALAR), nn.Linear(N_SCALAR, dim))
        self.layers = nn.ModuleList(_Layer(dim, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.readout = nn.Linear(dim, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.register_buffer("edges", torch.tensor(DIST_EDGES_KM), persistent=False)

    def scalars(self, scores: torch.Tensor, km: torch.Tensor, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        s = scores.masked_fill(~mask, -1e4)
        top = s.amax(1, keepdim=True)
        prob = torch.softmax(s / TEMPERATURE, 1) * mask
        support = [torch.log((prob.unsqueeze(1) * (km <= r)).sum(-1).clamp(min=1e-6)) for r in SUPPORT_KM]
        in_pool = (features[..., 6:9] > 0).any(-1).float()
        cols = [torch.where(mask, scores, 0.0), torch.where(mask, scores - top, 0.0), prob] + support
        out = torch.cat((torch.stack(cols, -1), features.float(), in_pool.unsqueeze(-1)), -1)
        return out.masked_fill(~mask.unsqueeze(-1), 0.0)

    def forward(self, hidden: torch.Tensor, scores: torch.Tensor, latlon_deg: torch.Tensor, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Logits [B, N] (-inf on empty slots). hidden [B, N, H]; scores [B, N]; latlon_deg [B, N, 2]; features [B, N, 9]; mask [B, N]."""

        scores = scores.float().masked_fill(~mask, 0.0)
        latlon = torch.where(mask.unsqueeze(-1), latlon_deg.float(), torch.zeros_like(latlon_deg, dtype=torch.float32))
        km = pairwise_km(latlon)
        buckets = torch.bucketize(km, self.edges)
        x = self.hidden_in(hidden.float()) + self.scalar_in(self.scalars(scores, km, features, mask))
        for layer in self.layers:
            x = layer(x, buckets, mask)
        logits = scores / TEMPERATURE + self.readout(self.norm(x)).squeeze(-1)
        return logits.masked_fill(~mask, float("-inf"))


def expected_hit_loss(logits: torch.Tensor, distance_km: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """-log of the softmax mass on candidates < 25 km (double weight < 1 km), averaged over photos with such a candidate.
    Returns (loss, fraction of photos used)."""

    reward = ((distance_km < 25.0).float() + (distance_km < 1.0).float()) / 2.0
    reward = torch.where(mask, reward, torch.zeros_like(reward))
    usable = reward.sum(1) > 0
    if not usable.any():
        return logits.new_tensor(0.0), logits.new_tensor(0.0)
    logp = torch.log_softmax(logits[usable].float(), 1)
    loss = -torch.logsumexp(logp + torch.log(reward[usable].clamp(min=1e-12)), 1)
    return loss.mean(), usable.float().mean()
