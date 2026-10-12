# Set-aware Pinpoint (lever 1 after lesson 69): Pinpoint's reranker scores each candidate on its own, trained to put the single nearest candidate
# first (every other candidate, even one 1 km from the truth, is a negative). Here its per-candidate encoder is kept and a set-level model on top
# sees all candidates at once (their encoder states, scores and pairwise distances) and is trained to put mass within 25 km of the truth.
# Usage: .venv/bin/python -m geo_search_env.experiment.pinpoint_set extract --name mp16 [--targets 32 8 8]
#        .venv/bin/python -m geo_search_env.experiment.pinpoint_set train ...

"""extract: Pinpoint's candidates for every query of a saved pool set (pinpoint_reference: mp16 = artifacts/sft/queries.json, mp16_extra), from a
wider search (top `targets` per modality instead of 8 / 2 / 2), each scored by the released reranker exactly as in its own pool (rank / shortlist
features relative to the original 8 / 2 / 2, so the original pool's candidates keep their scores), with the reranker's CLS state (512-d, the input
of its score head). Saved to artifacts/pinpoint_set/{name}_{targets}.npz: rows, latlon, scores, hidden (fp16), features, truth; candidates sorted
by score, empty slots row -1.

train: the set-level model on bucket-99 photos (photo sets train / dev3k; Pinpoint never trained on them). Per candidate: its CLS state, its
score (raw, minus the photo's best, softmax at T 0.2), the 9 retrieval features and whether it is in Pinpoint's own pool; two transformer layers
over the photo's candidates with a learned attention bias per pairwise-distance bucket (so it can count agreement); a zero-initialised residual
on Pinpoint's score. Loss: -log of the expected reward of a softmax pick (1 within 25 km, 2 within 1 km), GPS labels. 5-fold CV on train, then
trained on all of train and applied to dev3k. Reported: % < 25 / 1 km of the top candidate vs Pinpoint's top-1, and the oracle.
Variants: --pool original (only Pinpoint's <= 12 candidates), --no-hidden (no CLS states)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/pinpoint_set")
SFT_ROOT = Path("artifacts/sft")
NVME_INDEX = Path("/opt/dlami/nvme/pinpoint/attention_retrieval_index")  # staged copy of the index (hypotheses.NVME_INDEX), if present
QUERY_FILES = {"mp16": SFT_ROOT / "queries.json", "mp16_extra": SFT_ROOT / "queries_extra.json", "mp16_more": OUT / "queries_more.json"}


def more_queries(n: int, seed: int = 2) -> None:
    """n more training photos from buckets 0-98 with pinpoint_reference.extra's rules (no benchmark or dev photographers, no near-duplicates of
    benchmark photos), none already in queries.json or queries_extra.json -> artifacts/pinpoint_set/queries_more.json (extract --name mp16_more)."""

    import torch
    import torch.nn.functional as F

    from .sft_data import HELD_OUT_BUCKET, NEAR_DUPLICATE_SIM, VAL_AUTHOR_BUCKETS, _md5_bucket
    from .strategy_search import MP16_EMBED, load_world

    world = load_world()
    image_ids = (MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines()
    author_names = np.empty(len(world.vocab["author"]), dtype=object)
    for author, index in world.vocab["author"].items():
        author_names[index] = author
    used = {q["row"] for f in (QUERY_FILES["mp16"], QUERY_FILES["mp16_extra"]) for q in json.loads(f.read_text(encoding="utf-8"))}
    val_author = np.asarray([_md5_bucket(a, VAL_AUTHOR_BUCKETS) == 0 for a in author_names])[world.mp16["author"]]
    eligible = ~np.isin(world.mp16["author"], world.query_author[world.query_author >= 0]) & ~val_author
    rows = []
    for r in np.random.default_rng(seed).permutation(np.flatnonzero(eligible)):
        if int(r) not in used and _md5_bucket(image_ids[r], 100) < HELD_OUT_BUCKET:
            rows.append(int(r))
            if len(rows) >= int(n * 1.05):
                break
    rows = np.sort(np.asarray(rows))
    bench = F.normalize(torch.as_tensor(world.query_embeddings, device="cuda"), dim=-1).half()
    near = np.concatenate([
        (F.normalize(torch.as_tensor(np.asarray(world.mp16["embeddings"][rows[i : i + 8192]]), device="cuda").float(), dim=-1).half() @ bench.T)
        .max(dim=1).values.float().cpu().numpy()
        for i in range(0, len(rows), 8192)
    ])
    rows = rows[near < NEAR_DUPLICATE_SIM][:n]
    queries = [{"row": int(r), "image_id": image_ids[r], "group": "trained_on", "split": "train"} for r in rows]
    QUERY_FILES["mp16_more"].write_text(json.dumps(queries) + "\n", encoding="utf-8")
    print(f"{len(queries)} more photos from buckets 0-98 -> {QUERY_FILES['mp16_more']}", flush=True)


def extract(name: str, targets: Sequence[int], limit: int | None = None, tag: str = "") -> None:
    import torch

    from ..models.pinpoint_reranker import MODALITIES, PinpointReranker
    from .strategy_search import load_world

    class _Wide(PinpointReranker):
        def _union(self, found):
            """The pool over the wide lists; rank / shortlist features as the original computes them for its 8 / 2 / 2 lists (a hit
            beyond the original list size gets rank 1 and flag 0, as a candidate that list did not find)."""

            capacity = sum(len(found[m][0][0]) for m in MODALITIES)
            queries = len(found["image"][0])
            rows = np.full((queries, capacity), -1, dtype=np.int64)
            ranks = np.ones((queries, capacity, 3), dtype=np.float32)
            flags = np.zeros((queries, capacity, 3), dtype=np.float32)
            for q in range(queries):
                slots: dict[int, int] = {}
                for m, name_ in enumerate(MODALITIES):
                    indices, values = found[name_]
                    keep = self.original[name_]
                    for position, (row, value) in enumerate(zip(indices[q], values[q])):
                        if not np.isfinite(value):
                            continue
                        slot = slots.setdefault(int(row), len(slots))
                        rows[q, slot] = row
                        if position < keep:
                            ranks[q, slot, m] = min(ranks[q, slot, m], position / max(keep - 1, 1))
                            flags[q, slot, m] = 1.0
            return rows, ranks, flags

    if name == "bench":  # both halves of im2gps3k and yfcc4k, in the bench pools' order (photo set full: "index")
        world = load_world()
    else:
        world = load_world(mp16_queries=json.loads(QUERY_FILES[name].read_text(encoding="utf-8"))[:limit])
    model = _Wide(device="cuda", index_dir=NVME_INDEX if (NVME_INDEX / "manifest.json").exists() else None)
    model.original = dict(model.targets)
    # the released extraction searched 2x pools (tie order among equal fp16 scores depends on it); keep it for retrained rerankers
    model.multiplier = max(model.multiplier, 2)
    model.targets = dict(zip(MODALITIES, (int(t) for t in targets)))
    captured: list[torch.Tensor] = []
    head = model.head.score_head

    class _Capture(torch.nn.Module):
        def forward(self, x):
            captured.append(x.detach().to(torch.float16).cpu())
            return head(x)

    model.head.score_head = _Capture()
    index_author = world.mp16["author"][model.cache_indices]
    raw, projected = model.encode_queries(world.query_embeddings)
    found = model._search(raw, projected, world.query_author, index_author)
    rows, ranks, flags = model._union(found)
    scores, features = model._score(raw, projected, rows, ranks, flags)
    hidden = torch.cat(captured).numpy().reshape(rows.shape[0], rows.shape[1], -1)
    order = np.argsort(-scores, axis=1, kind="stable")
    rows, scores = np.take_along_axis(rows, order, 1), np.take_along_axis(scores, order, 1)
    features, hidden = np.take_along_axis(features, order[..., None], 1), np.take_along_axis(hidden, order[..., None], 1)
    valid = rows >= 0
    latlon = np.where(valid[..., None], np.asarray(model.latlon[np.where(valid, rows, 0)]), np.nan).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{name}_{'_'.join(str(t) for t in targets)}{f'_first{limit}' if limit else ''}{f'_{tag}' if tag else ''}.npz"
    np.savez(dst, rows=rows, latlon=latlon, scores=scores.astype(np.float32), hidden=hidden, features=features.astype(np.float16),
             truth=world.query_latlon.astype(np.float32))
    in_pool = (features[..., 6:9] > 0).any(-1)
    print(f"{name}: {len(rows)} queries, {valid.sum(1).mean():.1f} candidates ({in_pool.sum(1).mean():.1f} in the original pool) -> {dst}", flush=True)


DIST_EDGES_KM = (1.0, 5.0, 25.0, 100.0, 500.0, 2500.0)


def _haversine_matrix(latlon: np.ndarray) -> np.ndarray:
    """[Q, C, C] km between a photo's points."""

    lat, lon = np.deg2rad(latlon[..., 0]), np.deg2rad(latlon[..., 1])
    dlat = lat[:, :, None] - lat[:, None, :]
    dlon = lon[:, :, None] - lon[:, None, :]
    h = np.sin(dlat / 2) ** 2 + np.cos(lat[:, :, None]) * np.cos(lat[:, None, :]) * np.sin(dlon / 2) ** 2
    return (2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))).astype(np.float32)


def _load(path: Path | dict, pool: str) -> dict:
    z = dict(np.load(path)) if isinstance(path, Path) else dict(path)
    valid = z["rows"] >= 0
    in_pool = (z["features"][..., 6:9] > 0).any(-1)
    if "is_global" in z:  # look-zoom pools: Pinpoint's own pool is the global search's shortlist only
        in_pool &= z["is_global"]
    if pool == "original":
        valid &= in_pool
    s = np.where(valid, z["scores"], -np.inf)
    best = s.max(1, keepdims=True)
    soft = np.exp(np.where(valid, (s - best) / 0.2, -np.inf))
    soft /= soft.sum(1, keepdims=True)
    lat = np.where(valid[..., None], z["latlon"], 0.0).astype(np.float32)
    z["pair_bucket"] = np.zeros(valid.shape + valid.shape[-1:], np.int8)
    for c in range(0, len(lat), 1024):  # in chunks: [Q, C, C] float64 temporaries do not fit for ~100-candidate pools
        z["pair_bucket"][c:c + 1024] = np.digitize(_haversine_matrix(lat[c:c + 1024]), DIST_EDGES_KM)

    def mass_within(weights):
        """Softmax mass within 1 / 5 / 25 / 100 km of each candidate (bucket edges 1, 5, 25, 100 km = buckets <= 0..3)."""
        out = [np.zeros(weights.shape, np.float32) for _ in range(4)]
        for c in range(0, len(weights), 1024):
            for i in range(4):
                out[i][c:c + 1024] = np.einsum("qij,qj->qi", (z["pair_bucket"][c:c + 1024] <= i).astype(np.float32), weights[c:c + 1024])
        return out

    # agreement (lesson 68) as explicit inputs: softmax mass within 1 / 5 / 25 / 100 km of each candidate
    support = mass_within(soft)
    scalars = [np.where(valid, z["scores"], 0), np.where(valid, s - best, 0), soft] + [np.log(np.maximum(x, 1e-6)) for x in support]
    z["prior"] = np.where(valid, z["scores"], 0.0).astype(np.float32)
    if "group" in z:  # look-zoom pools: pool statistics from the global search only; zoomed candidates get their search's own too
        glob = valid & z["is_global"]
        s = np.where(glob, z["scores"], -np.inf)
        best = s.max(1, keepdims=True)
        soft = np.exp(np.where(glob, (s - best) / 0.2, -np.inf))
        soft /= soft.sum(1, keepdims=True)
        support = mass_within(soft)
        sv = np.where(valid, z["scores"], -np.inf)
        g_best, g_soft = np.zeros_like(z["prior"]), np.zeros_like(z["prior"])
        for g in np.unique(z["group"]):
            m = (z["group"] == g) & valid
            gb = np.where(m, sv, -np.inf).max(1, keepdims=True)
            e = np.exp(np.where(m, (sv - np.where(np.isfinite(gb), gb, 0)) / 0.2, -np.inf))
            g_best = np.where(m, sv - gb, g_best)
            g_soft = np.where(m, e / np.maximum(e.sum(1, keepdims=True), 1e-12), g_soft)
        scalars = [z["prior"], np.where(valid, z["scores"] - best, 0), soft * glob] + [np.log(np.maximum(x, 1e-6)) for x in support] + [g_best, g_soft]
        # zoomed candidates start 5 logits below their score: the untrained model is Pinpoint's global pick
        z["prior"] = np.where(glob | ~valid, z["prior"], z["prior"] - 1.0).astype(np.float32)
    parts = [np.stack(scalars, -1), z["features"].astype(np.float32), in_pool[..., None]] + ([z["cand_flags"]] if "cand_flags" in z else [])
    z["scalars"] = np.concatenate(parts, -1).astype(np.float32)
    z["valid"], z["in_pool"] = valid, in_pool & valid
    km = _haversine_matrix(np.concatenate([z["truth"][:, None, :], lat], 1))[:, 0, 1:]
    z["km"] = np.where(valid, km, np.inf).astype(np.float32)
    return z


HP = {"epochs": 30, "lr": 1e-3, "dim": 128, "layers": 2}  # set-model training settings (CLI: --epochs --lr --dim --layers)


def _set_model(n_scalar: int, hidden: bool, dim: int = 128, heads: int = 4, layers: int = 2):
    """Candidate encoder (CLS state + scalars) and transformer layers over the photo's candidates with a learned attention bias per
    distance bucket; the output is added to Pinpoint's score / 0.2 (zero-initialised, so the model starts as Pinpoint)."""

    import torch
    import torch.nn as nn

    torch.backends.mha.set_fastpath_enabled(False)  # the eval fast path returns NaN with these per-head float masks

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.h_proj = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, dim)) if hidden else None
            self.s_proj = nn.Sequential(nn.Linear(n_scalar, dim), nn.GELU(), nn.Linear(dim, dim))
            self.bias = nn.Embedding(len(DIST_EDGES_KM) + 1, heads * layers)
            nn.init.zeros_(self.bias.weight)
            self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(dim, heads, 4 * dim, 0.1, "gelu", batch_first=True, norm_first=True)
                                         for _ in range(layers)])
            self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
            nn.init.zeros_(self.out[1].weight)
            nn.init.zeros_(self.out[1].bias)

        def forward(self, h, x, bucket, valid, score):
            e = self.s_proj(x) + (self.h_proj(h) if self.h_proj is not None else 0)
            b = self.bias(bucket.long())  # [B, C, C, heads * layers]
            for i, block in enumerate(self.blocks):
                mask = b[..., i * heads:(i + 1) * heads].permute(0, 3, 1, 2).masked_fill(~valid[:, None, None, :], -1e4)  # finite: the eval fast path turns -inf masks into NaN
                e = block(e, src_mask=mask.reshape(-1, *mask.shape[2:]))
            return (score / 0.2 + self.out(e).squeeze(-1)).masked_fill(~valid, float("-inf"))

    return Net()


def _targets(km: np.ndarray) -> np.ndarray:
    """Reward of picking each candidate: 1 within 25 km of the truth, 2 within 1 km (halved to [0, 1])."""

    return ((km < 25).astype(np.float32) + (km < 1).astype(np.float32)) / 2


def _batch(z, b, mu, sd, device):
    import torch

    v = z["valid"][b]
    x = np.where(v[..., None], (z["scalars"][b] - mu) / sd, 0.0)
    return (torch.as_tensor(z["hidden"][b], device=device).float(), torch.as_tensor(x, device=device, dtype=torch.float32),
            torch.as_tensor(z["pair_bucket"][b], device=device), torch.as_tensor(v, device=device),
            torch.as_tensor(np.where(v, z["prior"][b], 0.0), device=device, dtype=torch.float32))


def _fit_set(z: dict, photos: np.ndarray, hidden: bool, seed: int, epochs: int | None = None, device: str = "cuda"):
    import torch

    torch.manual_seed(seed)
    epochs = epochs or HP["epochs"]
    usable = photos[(z["km"][photos] < 25).any(1)]
    model = _set_model(z["scalars"].shape[-1], hidden, dim=HP["dim"], layers=HP["layers"]).to(device)
    batches = max(1, len(usable) // 256)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=HP["lr"], total_steps=epochs * batches, pct_start=0.1)
    cells = z["scalars"][photos][z["valid"][photos]]
    mu, sd = cells.mean(0), cells.std(0) + 1e-6
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        for b in np.array_split(rng.permutation(usable), batches):
            logp = torch.log_softmax(model(*_batch(z, b, mu, sd, device)), 1)
            # -log(expected reward of a pick drawn from the softmax): free to concentrate on one good candidate (the per-candidate
            # cross-entropy to a uniform target over all good candidates spreads mass and picked worse than Pinpoint)
            reward = torch.as_tensor(_targets(z["km"][b]), device=device)
            loss = -torch.logsumexp(logp + torch.log(reward), 1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return model, mu, sd


def _predict_set(models, z, photos, mu, sd, device="cuda") -> np.ndarray:
    """Pick probabilities [photos, C] (an ensemble averages its softmaxes); argmax(1) is the answer."""

    import torch

    out = []
    with torch.no_grad():
        for b in np.array_split(photos, max(1, len(photos) // 512)):
            out.append((sum(torch.softmax(m(*_batch(z, b, mu, sd, device)), 1) for m in models) / len(models)).cpu().numpy())
    return np.concatenate(out)


def _report(label: str, z: dict, photos: np.ndarray, picks: np.ndarray) -> None:
    from .strategy_search import _bootstrap

    km = z["km"][photos]
    rows = np.arange(len(photos))
    base_km = km[rows, np.where(z["in_pool"][photos], z["scores"][photos], -np.inf).argmax(1)]
    chosen = km[rows, picks]
    parts = []
    for t in (25, 1):
        d = _bootstrap((chosen < t).astype(float) - (base_km < t))
        parts.append(f"< {t} km {np.mean(chosen < t):.1%} ({100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}])")
    print(f"  {label:32s} Pinpoint {np.mean(base_km < 25):.1%} / {np.mean(base_km < 1):.1%}; " + "; ".join(parts)
          + f"; oracle {np.mean((km < 25).any(1)):.1%}", flush=True)


def train(path: Path, pool: str, hidden: bool, ensemble: int, save: Path | None, extra: Sequence[Path] = (), dump: Path | None = None) -> None:
    z = _load(path, pool)
    queries = json.loads(QUERY_FILES["mp16"].read_text(encoding="utf-8"))
    assert len(queries) == len(z["rows"]), "train needs the extraction of all of queries.json"
    sets = {}
    for tag in ("train", "dev3k"):  # photo-set "index" = position in queries.json
        photos = json.loads((Path("artifacts/photo_sets") / tag / "photos.json").read_text(encoding="utf-8"))
        sets[tag] = np.asarray([p["index"] for p in photos])
        assert np.abs(np.asarray([p["truth"] for p in photos]) - z["truth"][sets[tag]]).max() < 1e-3, tag
    tr, dv = sets["train"], sets["dev3k"]
    fit_extra = np.zeros(0, dtype=int)
    if extra:  # more training photos (buckets 0-98, which Pinpoint trained on): appended, used for fitting only
        parts = [_load(path_, pool) for path_ in extra]
        fit_extra = len(z["rows"]) + np.arange(sum(len(e["rows"]) for e in parts))
        z = {k: np.concatenate([z[k]] + [e[k] for e in parts]) for k in z}
        del parts
        print(f"  + {len(fit_extra)} extra training photos ({', '.join(p_.name for p_ in extra)})")
    print(f"{path.name}: pool {pool}, hidden {hidden}, ensemble {ensemble}; {z['valid'][tr].sum(1).mean():.1f} candidates per photo", flush=True)
    folds = np.random.default_rng(0).permutation(len(tr)) % 5
    cv = np.zeros(z["rows"][tr].shape, dtype=np.float32)
    fold_fits = []
    for k in range(5):
        fits = [_fit_set(z, np.concatenate([tr[folds != k], fit_extra]), hidden, seed=10 * k + j) for j in range(ensemble)]
        cv[folds == k] = _predict_set([m for m, _, _ in fits], z, tr[folds == k], fits[0][1], fits[0][2])
        fold_fits.append(([m.state_dict() for m, _, _ in fits], fits[0][1], fits[0][2]))
    _report("train, 5-fold CV", z, tr, cv.argmax(1))
    fits = [_fit_set(z, np.concatenate([tr, fit_extra]), hidden, seed=100 + j) for j in range(ensemble)]
    dev = _predict_set([m for m, _, _ in fits], z, dv, fits[0][1], fits[0][2])
    _report("dev3k (trained on all of train)", z, dv, dev.argmax(1))
    if dump:  # out-of-fold pick probabilities for train, all-of-train model for dev3k (photo-set order), for the hypothesis comparator
        np.savez(dump, train_probs=cv, dev3k_probs=dev, train_latlon=z["latlon"][tr], dev3k_latlon=z["latlon"][dv],
                 train_valid=z["valid"][tr], dev3k_valid=z["valid"][dv])
        print(f"  pick probabilities -> {dump}")
    if save:
        import torch

        # fold models too: train photos must be scored by the model that did not see them (fold k of train photo i: train_folds[i])
        torch.save({"models": [m.state_dict() for m, _, _ in fits], "mu": fits[0][1], "sd": fits[0][2], "pool": pool, "hidden": hidden,
                    "n_scalar": z["scalars"].shape[-1], "hp": dict(HP), "fold_fits": fold_fits, "train_folds": folds}, save)
        print(f"  saved -> {save}")


def _logits_set(models, z, photos, mu, sd, device="cuda") -> np.ndarray:
    """Mean set-model logits [photos, C] (-inf on empty slots)."""

    import torch

    out = []
    with torch.no_grad():
        for b in np.array_split(photos, max(1, len(photos) // 512)):
            out.append((sum(m(*_batch(z, b, mu, sd, device)) for m in models) / len(models)).cpu().numpy())
    return np.concatenate(out)


def restricted(tag: str, extra_hyps: Sequence[str], model_path: Path | None) -> None:
    """The set model inside each hypothesis: every hypothesis's restricted candidates (hypotheses.py, merged as hypothesis_compare's E4
    features) are re-scored by Pinpoint to get their CLS states (rank / shortlist features as stored), then the set model scores them. Train
    photos are scored by the fold model that did not see them. Saved per hypothesis (merged order): setr_max (best logit), setr_lse
    (log-sum-exp of logits), km_setr (km of the set model's pick inside the hypothesis) -> artifacts/pinpoint_set/restricted_{tag}.npz."""

    import torch

    from ..models.pinpoint_reranker import PinpointReranker
    from . import hypothesis_compare
    from .strategy_search import load_world

    hypothesis_compare.EXTRA_HYPOTHESES.extend(extra_hyps)
    h = hypothesis_compare._merge_clue_hyps(tag)
    photos = json.loads((Path("artifacts/photo_sets") / tag / "photos.json").read_text(encoding="utf-8"))
    from .hypotheses import photo_world

    world, qrows = photo_world(photos)  # benchmark sets (full): the benchmark world
    model = PinpointReranker(device="cuda", index_dir=NVME_INDEX if (NVME_INDEX / "manifest.json").exists() else None)
    captured: list[torch.Tensor] = []
    head = model.head.score_head

    class _Capture(torch.nn.Module):
        def forward(self, x):
            captured.append(x.detach().to(torch.float16).cpu())
            return head(x)

    model.head.score_head = _Capture()
    query_index = qrows[h["photo"]]
    raw, projected = model.encode_queries(world.query_embeddings[np.unique(query_index)])
    position = np.searchsorted(np.unique(query_index), query_index)
    rows, features = h["rows"].astype(np.int64), h["features"].astype(np.float32)
    scores = np.concatenate([
        model._score(raw[position[i:i + 8192]], projected[position[i:i + 8192]], rows[i:i + 8192], features[i:i + 8192, :, 3:6],
                     features[i:i + 8192, :, 6:9])[0]
        for i in range(0, len(rows), 8192)
    ])
    valid = rows >= 0
    print(f"{tag}: {len(rows)} hypotheses re-scored; max |score - stored| {np.abs(scores - h['scores'])[valid].max():.4f}", flush=True)
    hidden = torch.cat(captured).numpy().reshape(rows.shape[0], rows.shape[1], -1)
    truth = np.asarray([p["truth"] for p in photos], np.float32)[h["photo"]]
    # the zoomed candidates with their CLS states, for the look-zoom-rerank model (PINPOINT_SET_PLAN.md step 1)
    np.savez(OUT / f"zoom_{tag}.npz", photo=h["photo"], source=h["source"], rank=h["rank"], support=h["support"], center=h["center"],
             rows=rows, latlon=h["latlon"].astype(np.float32), scores=scores.astype(np.float32), hidden=hidden, features=features.astype(np.float16))
    if model_path is None:  # zoom states only (retrained rerankers: no offline set model to run inside the hypotheses)
        print(f"  zoomed candidates -> {OUT / f'zoom_{tag}.npz'}", flush=True)
        return
    z = _load({"rows": rows, "latlon": h["latlon"].astype(np.float32), "scores": scores.astype(np.float32), "hidden": hidden,
               "features": features, "truth": truth}, "wide")
    ckpt = torch.load(model_path, map_location="cuda", weights_only=False)
    hp = ckpt.get("hp", HP)

    def build(states):
        nets = []
        for state in states:
            net = _set_model(ckpt["n_scalar"], ckpt["hidden"], dim=hp["dim"], layers=hp["layers"]).cuda()
            net.load_state_dict(state)
            nets.append(net.eval())
        return nets

    logits = np.full(rows.shape, -np.inf, dtype=np.float32)
    if tag == "train":  # out-of-fold: photo i of train was held out of fold model train_folds[i]
        fold_of = np.asarray(ckpt["train_folds"])[h["photo"]]
        for k, (states, mu, sd) in enumerate(ckpt["fold_fits"]):
            idx = np.flatnonzero(fold_of == k)
            logits[idx] = _logits_set(build(states), z, idx, mu, sd)
    else:
        logits = _logits_set(build(ckpt["models"]), z, np.arange(len(rows)), ckpt["mu"], ckpt["sd"])
    has = valid.any(1)
    pick = np.where(has, logits.argmax(1), 0)
    out = {"setr_max": np.where(has, logits.max(1), -1e3),
           "setr_lse": np.where(has, np.logaddexp.reduce(np.where(valid, logits, -np.inf), axis=1), -1e3),
           "km_setr": np.where(has, z["km"][np.arange(len(rows)), pick], np.inf), "photo": h["photo"], "source": h["source"]}
    dst = OUT / f"restricted_{tag}.npz"
    np.savez(dst, **out)
    print(f"  set model inside each hypothesis -> {dst}", flush=True)


def apply(tag: str, data: Path, model_path: Path) -> None:
    """The saved set model (trained on all of train) on a photo set's extracted candidates -> artifacts/pinpoint_set/set_probs_{tag}.npz in the
    train --dump format ({tag}_probs, {tag}_latlon, {tag}_valid; hypothesis_compare --set-dump), and its accuracy next to Pinpoint's."""

    import torch

    photos = json.loads((Path("artifacts/photo_sets") / tag / "photos.json").read_text(encoding="utf-8"))
    idx = np.asarray([p["index"] for p in photos])
    z = _load(data, "wide")
    assert np.abs(np.asarray([p["truth"] for p in photos]) - z["truth"][idx]).max() < 1e-3
    ckpt = torch.load(model_path, map_location="cuda", weights_only=False)
    hp = ckpt.get("hp", HP)
    models = []
    for state in ckpt["models"]:
        net = _set_model(ckpt["n_scalar"], ckpt["hidden"], dim=hp["dim"], layers=hp["layers"]).cuda()
        net.load_state_dict(state)
        models.append(net.eval())
    probs = _predict_set(models, z, idx, ckpt["mu"], ckpt["sd"])
    for bench in sorted({p.get("benchmark", tag) for p in photos}):
        sel = np.asarray([p.get("benchmark", tag) == bench for p in photos])
        _report(f"{tag} / {bench} ({sel.sum()} photos)", z, idx[sel], probs[sel].argmax(1))
    dst = OUT / f"set_probs_{tag}.npz"
    np.savez(dst, **{f"{tag}_probs": probs, f"{tag}_latlon": z["latlon"][idx], f"{tag}_valid": z["valid"][idx]})
    print(f"  -> {dst}")


def _zoom_pools(tag: str, k: int = 5, sources: Sequence[int] = (0,)) -> dict:
    """Look + zoom candidates per photo of a set (PINPOINT_SET_PLAN.md step 1): the global wide search (extract: mp16 / bench) followed by
    the restricted searches of its hypotheses (restricted: zoom_{tag}.npz; Pinpoint areas ranked < k, plus clue / VLM places if in
    sources), each candidate with its CLS state as scored in its own search. Per-candidate flags: zoomed, hypothesis source (3), area rank
    (k), log area support. A row found by both searches appears twice (two views)."""

    photos = json.loads((Path("artifacts/photo_sets") / tag / "photos.json").read_text(encoding="utf-8"))
    idx = np.asarray([p["index"] for p in photos])
    g = np.load(OUT / ("bench_32_8_8.npz" if "benchmark" in photos[0] else "mp16_32_8_8.npz"))
    glob = {key: g[key][idx] for key in ("rows", "latlon", "scores", "hidden", "features", "truth")}
    zm = np.load(OUT / f"zoom_{tag}.npz")
    keep = np.isin(zm["source"], sources) & ((zm["source"] != 0) | (zm["rank"] < k))
    zrows = np.flatnonzero(keep)
    by_photo: dict[int, list[int]] = {}
    for i in zrows:
        by_photo.setdefault(int(zm["photo"][i]), []).append(int(i))
    cg, cz = glob["rows"].shape[1], zm["rows"].shape[1]
    cap = cg + cz * max((len(v) for v in by_photo.values()), default=0)
    n = len(photos)
    n_flags = 1 + 3 + k + 1
    out = {"rows": np.full((n, cap), -1, np.int64), "latlon": np.full((n, cap, 2), np.nan, np.float32), "scores": np.full((n, cap), -np.inf, np.float32),
           "hidden": np.zeros((n, cap, glob["hidden"].shape[-1]), np.float16), "features": np.zeros((n, cap, 9), np.float16),
           "truth": glob["truth"].astype(np.float32), "cand_flags": np.zeros((n, cap, n_flags), np.float32), "is_global": np.zeros((n, cap), bool),
           "group": np.zeros((n, cap), np.int16)}
    for key in ("rows", "latlon", "scores", "hidden", "features"):
        out[key][:, :cg] = glob[key]
    out["is_global"][:, :cg] = glob["rows"] >= 0
    zh, zr, zl, zs, zf = zm["hidden"], zm["rows"], zm["latlon"], zm["scores"], zm["features"]
    for p, members in by_photo.items():
        for j, i in enumerate(members):
            sl = slice(cg + j * cz, cg + (j + 1) * cz)
            out["rows"][p, sl], out["latlon"][p, sl], out["scores"][p, sl] = zr[i], zl[i], zs[i]
            out["hidden"][p, sl], out["features"][p, sl] = zh[i], zf[i]
            out["group"][p, sl] = j + 1
            flags = np.zeros(n_flags, np.float32)
            flags[0], flags[1 + int(zm["source"][i])] = 1.0, 1.0
            if zm["source"][i] == 0:
                flags[4 + int(zm["rank"][i])] = 1.0
            flags[-1] = np.log(max(float(zm["support"][i]), 1e-6))
            out["cand_flags"][p, sl] = np.where((zr[i] >= 0)[:, None], flags, 0.0)
    out["scores"] = np.where(out["rows"] >= 0, out["scores"], 0.0).astype(np.float32)
    print(f"{tag}: look + zoom pools, up to {cap} candidates ({(out['rows'] >= 0).sum(1).mean():.1f} per photo; {len(zrows)} zoomed areas)", flush=True)
    return out


def train_zoom(k: int, sources: Sequence[int], ensemble: int, save: Path | None, apply_tag: str | None) -> None:
    """Step 1: the set model over look + zoom candidates. 5-fold CV on train, then trained on all of train and applied to dev3k (and, only
    when asked, to a test set: apply_tag)."""

    tr = _load(_zoom_pools("train", k, sources), "wide")
    dv = _load(_zoom_pools("dev3k", k, sources), "wide")
    n = len(tr["rows"])
    print(f"look + zoom: K {k}, sources {tuple(sources)}, ensemble {ensemble}, {tr['scalars'].shape[-1]} scalars per candidate", flush=True)
    folds = np.random.default_rng(0).permutation(n) % 5
    cv = np.zeros(tr["rows"].shape, np.float32)
    fold_fits = []
    for f in range(5):
        fits = [_fit_set(tr, np.flatnonzero(folds != f), True, seed=10 * f + j) for j in range(ensemble)]
        cv[folds == f] = _predict_set([m for m, _, _ in fits], tr, np.flatnonzero(folds == f), fits[0][1], fits[0][2])
        fold_fits.append(([m.state_dict() for m, _, _ in fits], fits[0][1], fits[0][2]))
    _report("train, 5-fold CV", tr, np.arange(n), cv.argmax(1))
    fits = [_fit_set(tr, np.arange(n), True, seed=100 + j) for j in range(ensemble)]
    models, mu, sd = [m for m, _, _ in fits], fits[0][1], fits[0][2]
    _report("dev3k (trained on all of train)", dv, np.arange(len(dv["rows"])), _predict_set(models, dv, np.arange(len(dv["rows"])), mu, sd).argmax(1))
    if save:
        import torch

        torch.save({"models": [m.state_dict() for m in models], "mu": mu, "sd": sd, "hidden": True, "n_scalar": tr["scalars"].shape[-1], "hp": dict(HP),
                    "k": k, "sources": tuple(sources), "fold_fits": fold_fits, "train_folds": folds}, save)
        print(f"  saved -> {save}")
    if apply_tag:
        te = _load(_zoom_pools(apply_tag, k, sources), "wide")
        photos = json.loads((Path("artifacts/photo_sets") / apply_tag / "photos.json").read_text(encoding="utf-8"))
        picks = _predict_set(models, te, np.arange(len(te["rows"])), mu, sd).argmax(1)
        for bench in sorted({p.get("benchmark", apply_tag) for p in photos}):
            sel = np.flatnonzero([p.get("benchmark", apply_tag) == bench for p in photos])
            _report(f"{apply_tag} / {bench} ({len(sel)})", te, sel, picks[sel])
        _report(f"{apply_tag} / all ({len(picks)})", te, np.arange(len(picks)), picks)


def photo_set(ckpt_tag: str, tags: Sequence[str] = ("train", "dev3k")) -> None:
    """The hypothesis pipeline on a retrained reranker (lesson 70; the merged-pipeline plan, step 1): its wide pool (extract --ckpt-tag) becomes
    the photo sets' Pinpoint pool. Writes pinpoint_reference/{mp16,bench}_{ckpt_tag}.npz (+ queries), photo_sets/{tag}_{ckpt_tag}/ (photos.json with the
    new pool sorted by score, labels.json and the clue inputs of the original set linked), and set_probs_{tag}_{ckpt_tag}.npz: the retrained
    reranker's softmax (T 0.2) over its pool, in the set-dump format (it replaces the offline set layer). Run hypotheses / hypothesis_compare /
    restricted with PINPOINT_RERANKER_CKPT set to the same checkpoint."""

    import os
    import shutil

    from ..models.pinpoint_reranker import _discover_index

    index_dir = _discover_index()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    cache_indices = np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
    ref = Path("artifacts/pinpoint_reference")
    sources = {}

    def source(kind):
        """The retrained reranker's wide pools for MP16 queries (mp16) or both benchmark halves (bench), saved as a pinpoint_reference set."""
        if kind not in sources:
            z = {k: v for k, v in np.load(OUT / f"{kind}_32_8_8_{ckpt_tag}.npz").items() if k != "hidden"}  # read each array once
            valid = z["rows"] >= 0
            np.savez(ref / f"{kind}_{ckpt_tag}.npz", rows=z["rows"], cache_rows=np.where(valid, cache_indices[np.where(valid, z["rows"], 0)], -1),
                     latlon=z["latlon"], scores=np.where(valid, z["scores"], -np.inf), features=z["features"], truth=z["truth"])
            shutil.copyfile(ref / f"{kind}_queries.json", ref / f"{kind}_{ckpt_tag}_queries.json")
            s_ = np.where(valid, z["scores"], -np.inf)
            probs = np.exp((s_ - s_.max(1, keepdims=True)) / 0.2)
            sources[kind] = (z, valid, probs / probs.sum(1, keepdims=True))
        return sources[kind]

    agent = Path("artifacts/agent")
    for tag in tags:
        src, dst = Path("artifacts/photo_sets") / tag, Path("artifacts/photo_sets") / f"{tag}_{ckpt_tag}"
        dst.mkdir(parents=True, exist_ok=True)
        photos = json.loads((src / "photos.json").read_text(encoding="utf-8"))
        kind = "bench" if "benchmark" in photos[0] else "mp16"
        z, valid, probs = source(kind)
        for e in photos:
            v = valid[e["index"]]
            ll = z["latlon"][e["index"]][v].astype(float).tolist()
            e["pool"], e["pool_rows"], e["pools"] = ll, z["rows"][e["index"]][v].tolist(), f"{kind}_{ckpt_tag}"
            e["candidates"] = [["", a, b] for a, b in ll]
        (dst / "photos.json").write_text(json.dumps(photos), encoding="utf-8")
        shutil.copyfile(src / "labels.json", dst / "labels.json")
        for name in (f"region_evidence_{tag}_4b.json", f"clue_judge_{tag}_4b_4b.json", f"observe_{tag}_4b.json", f"queries_{tag}_4b.json"):
            link = agent / name.replace(f"_{tag}_", f"_{tag}_{ckpt_tag}_", 1)
            if (agent / name).exists() and not link.exists():
                os.symlink((agent / name).resolve(), link)
        idx = np.asarray([e["index"] for e in photos])
        t = f"{tag}_{ckpt_tag}"
        np.savez(OUT / f"set_probs_{t}.npz", **{f"{t}_probs": probs[idx], f"{t}_latlon": z["latlon"][idx], f"{t}_valid": valid[idx]})
        print(f"{t}: {len(photos)} photos, {valid[idx].sum(1).mean():.1f} candidates per pool", flush=True)


def eval_head(data: Path, checkpoint: Path | None) -> None:
    """Step 4: a retrained reranker (pinpoint_train/) on train and dev3k. Its own pick (top score over the original 8 / 2 / 2 pool) and,
    with `checkpoint`, its set head's pick over the wide pool, against the released Pinpoint (mp16_32_8_8.npz)."""

    import torch

    from ..models.set_head import SetHead

    z = np.load(data)
    ref = np.load(OUT / "mp16_32_8_8.npz")
    head = None
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = {k.removeprefix("set_head."): v for k, v in state["state_dict"].items() if k.startswith("set_head.")}
        hp = state["hyper_parameters"]
        head = SetHead(512, int(hp.get("set_head_dim", 128)), 4, int(hp.get("set_head_layers", 2))).cuda().eval()
        head.load_state_dict(sd)
    valid = z["rows"] >= 0
    with torch.no_grad():
        logits = []
        for b in np.array_split(np.arange(len(valid)), max(1, len(valid) // 512)):
            if head is None:
                break
            t = lambda x, dt=torch.float32: torch.as_tensor(x[b], device="cuda", dtype=dt)
            logits.append(head(t(z["hidden"]), t(np.where(valid, z["scores"], 0.0)), t(np.nan_to_num(z["latlon"])), t(z["features"]),
                               t(valid, torch.bool)).cpu().numpy())
    set_pick = np.concatenate(logits).argmax(1) if logits else None

    def km_of(zz, pick):
        ll = zz["latlon"][np.arange(len(pick)), pick]
        return _haversine_matrix(np.stack([zz["truth"], ll], 1))[:, 0, 1]

    def own_pick(zz):
        in_pool = (zz["features"][..., 6:9] > 0).any(-1) & (zz["rows"] >= 0)
        return np.where(in_pool, zz["scores"], -np.inf).argmax(1)

    from .strategy_search import _bootstrap

    base_km = km_of(ref, own_pick(ref))
    for tag in ("train", "dev3k"):
        idx = np.asarray([p["index"] for p in json.loads((Path("artifacts/photo_sets") / tag / "photos.json").read_text(encoding="utf-8"))])
        wide_pick = np.where(z["rows"] >= 0, z["scores"], -np.inf).argmax(1)
        rows = [("retrained, own pool", km_of(z, own_pick(z))[idx]), ("retrained, wide pool", km_of(z, wide_pick)[idx])]
        rows += [("retrained + set head", km_of(z, set_pick)[idx])] if set_pick is not None else []
        b = base_km[idx]
        print(f"  {tag}: released Pinpoint {np.mean(b < 25):.1%} / {np.mean(b < 1):.1%} (25 / 1 km)")
        for label, k in rows:
            parts = []
            for t in (25, 1):
                d = _bootstrap((k < t).astype(float) - (b < t))
                parts.append(f"< {t} km {np.mean(k < t):.1%} ({100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}])")
            print(f"    {label:24s} " + "; ".join(parts), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("extract", "train", "more-queries", "restricted", "apply", "zoom", "eval-head", "photo-set"))
    parser.add_argument("--checkpoint", type=Path, help="eval-head: retrained reranker checkpoint with a set head")
    parser.add_argument("--tags", nargs="*", default=("train", "dev3k"), help="photo-set: photo sets to rebuild (full = test set, only when asked)")
    parser.add_argument("--ckpt-tag", default="", help="extract: output suffix (with PINPOINT_RERANKER_CKPT set to a retrained reranker)")
    parser.add_argument("--k", type=int, default=5, help="zoom: Pinpoint areas zoomed into")
    parser.add_argument("--sources", type=int, nargs="*", default=(0,), help="zoom: hypothesis sources (0 areas, 1 clue places, 2 VLM places)")
    parser.add_argument("--apply-tag", help="zoom: also apply to this photo set (test set only when the user asks)")
    parser.add_argument("--tag", default="dev3k", help="restricted: photo set (train, dev3k)")
    parser.add_argument("--extra-hyps", nargs="*", default=("vlmhyps",), help="restricted: hypothesis files merged as in hypothesis_compare")
    parser.add_argument("--model", type=Path, default=OUT / "set_wide.pt", help="restricted: saved set model (train --save)")
    parser.add_argument("--n", type=int, default=100_000, help="more-queries: how many")
    parser.add_argument("--name", default="mp16", choices=tuple(QUERY_FILES) + ("bench",))
    parser.add_argument("--targets", type=int, nargs=3, default=(32, 8, 8), help="hits per modality (image, raw image, GPS); Pinpoint uses 8 2 2")
    parser.add_argument("--limit", type=int, help="first N queries only (plumbing)")
    parser.add_argument("--data", type=Path, default=OUT / "mp16_32_8_8.npz", help="train: extracted candidates (all of queries.json)")
    parser.add_argument("--pool", choices=("wide", "original"), default="wide", help="train: all extracted candidates or Pinpoint's own pool")
    parser.add_argument("--no-hidden", action="store_true", help="train: scores, features and geometry only (no CLS states)")
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--dump", type=Path, help="train: save pick probabilities (train out-of-fold, dev3k)")
    parser.add_argument("--extra", type=Path, nargs="*", default=(), help="train: extracted candidates of more training photos (e.g. mp16_extra_32_8_8.npz)")
    for key, value in HP.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args(argv)
    HP.update({key: getattr(args, key) for key in HP})
    if args.cmd == "more-queries":
        more_queries(args.n)
    elif args.cmd == "zoom":
        train_zoom(args.k, args.sources, args.ensemble, args.save, args.apply_tag)
    elif args.cmd == "apply":
        apply(args.tag, args.data, args.model)
    elif args.cmd == "restricted":
        restricted(args.tag, args.extra_hyps, None if str(args.model) == "none" else args.model)
    elif args.cmd == "extract":
        extract(args.name, args.targets, args.limit, args.ckpt_tag)
    elif args.cmd == "photo-set":
        photo_set(args.ckpt_tag, args.tags)
    elif args.cmd == "eval-head":
        eval_head(args.data, args.checkpoint)
    else:
        train(args.data, args.pool, not args.no_hidden, args.ensemble, args.save, args.extra, args.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
