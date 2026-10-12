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


def extract(name: str, targets: Sequence[int], limit: int | None = None) -> None:
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

    queries = json.loads(QUERY_FILES[name].read_text(encoding="utf-8"))[:limit]
    world = load_world(mp16_queries=queries)
    model = _Wide(device="cuda", index_dir=NVME_INDEX if (NVME_INDEX / "manifest.json").exists() else None)
    model.original = dict(model.targets)
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
    dst = OUT / f"{name}_{'_'.join(str(t) for t in targets)}{f'_first{limit}' if limit else ''}.npz"
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


def _load(path: Path, pool: str) -> dict:
    z = dict(np.load(path))
    valid = z["rows"] >= 0
    in_pool = (z["features"][..., 6:9] > 0).any(-1)
    if pool == "original":
        valid &= in_pool
    s = np.where(valid, z["scores"], -np.inf)
    best = s.max(1, keepdims=True)
    soft = np.exp(np.where(valid, (s - best) / 0.2, -np.inf))
    soft /= soft.sum(1, keepdims=True)
    lat = np.where(valid[..., None], z["latlon"], 0.0)
    pair = _haversine_matrix(lat)
    # agreement (lesson 68) as explicit inputs: softmax mass within 1 / 5 / 25 / 100 km of each candidate
    support = [np.einsum("qij,qj->qi", (pair < r).astype(np.float32), soft) for r in (1.0, 5.0, 25.0, 100.0)]
    scalars = [np.where(valid, z["scores"], 0), np.where(valid, s - best, 0), soft] + [np.log(np.maximum(x, 1e-6)) for x in support]
    z["scalars"] = np.concatenate([np.stack(scalars, -1), z["features"].astype(np.float32), in_pool[..., None]], -1).astype(np.float32)
    z["valid"], z["in_pool"] = valid, in_pool & valid
    z["pair_bucket"] = np.digitize(pair, DIST_EDGES_KM).astype(np.int8)
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
            torch.as_tensor(np.where(v, z["scores"][b], 0.0), device=device, dtype=torch.float32))


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
    for k in range(5):
        fits = [_fit_set(z, np.concatenate([tr[folds != k], fit_extra]), hidden, seed=10 * k + j) for j in range(ensemble)]
        cv[folds == k] = _predict_set([m for m, _, _ in fits], z, tr[folds == k], fits[0][1], fits[0][2])
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

        torch.save({"models": [m.state_dict() for m, _, _ in fits], "mu": fits[0][1], "sd": fits[0][2], "pool": pool, "hidden": hidden,
                    "n_scalar": z["scalars"].shape[-1]}, save)
        print(f"  saved -> {save}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("extract", "train", "more-queries"))
    parser.add_argument("--n", type=int, default=100_000, help="more-queries: how many")
    parser.add_argument("--name", default="mp16", choices=tuple(QUERY_FILES))
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
    elif args.cmd == "extract":
        extract(args.name, args.targets, args.limit)
    else:
        train(args.data, args.pool, not args.no_hidden, args.ensemble, args.save, args.extra, args.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
