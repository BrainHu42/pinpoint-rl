# Geo-aware re-scoring of the raw neighbours: an adapter on frozen SigLIP2 trained to pull same-place photos together and push look-alikes apart.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.geo_embed data     (CPU + disk, gathers training rows)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.geo_embed train --name lr2e-3 --lr 2e-3   (GPU, minutes)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.geo_embed report --name lr2e-3   (GPU; dev / val report)

"""Training queries: the MP16 bucket-99 train photos (held-out group, train split; artifacts/sft/queries.json), each with its cached top-1000 raw
neighbours in MP16 and in OSV-5M (same-photographer rows already excluded). Gallery rows by dev or benchmark photographers are dropped from training
lists. A neighbour is positive if it lies within POS_KM of the query's truth. Per query: up to POS_MAX positives, the NEG_MAX most similar
non-positives (by raw similarity: the look-alikes) and NEG_RANDOM random non-positives from the rest of the list. Queries whose authors fall in a selection bucket are held out for model selection.

Model: f(x) = normalize(x + MLP(x)), the last layer zero-initialised so training starts from raw SigLIP2 similarity; score = cos(f(q), f(g)) / tau.
Loss: multi-positive InfoNCE over each query's sampled list.

report (dev / val photos, placeholders dropped): re-score all ~2,000 cached neighbours, cluster at 1 km (score order) and report
- recall: a cluster within 25 km of the truth among the first K = 10 / 25 / 50 clusters, raw vs adapter;
- the current pool + K extra clusters (K = 5 / 10 / 20) from the adapter vs from raw similarity + region prior (the pipeline's own raw source, the
  matched control of region_prior), oracle gain at 25 / 200 km with 95% intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from typing import Any, Sequence

import numpy as np

from .query_evidence import BENCH_ROOT, ROOT, SFT_ROOT
from .region_prior import _extend, _hit
from .stage1_eval import _bootstrap
from .strategy_search import PRIOR_FLOOR, _pool, load_world
from .wiki_backend import _km

OUT = ROOT / "geo_embed"
POS_KM = 25.0
POS_MAX = 32
NEG_MAX = 96
NEG_RANDOM = 128  # random non-positives from the rest of the list: without them the adapter lifts deep neighbours it never saw
SELECT_BUCKET = 1  # md5(author) % 10 == 1 among train photographers: model selection
EPOCHS = 6
BATCH = 256
KS = (10, 25, 50)
EXTRA = (5, 10, 20)


def _world(dev: list[dict[str, Any]]):
    return load_world(mp16_queries=[{"row": e["row"], "image_id": e["image_id"]} for e in dev])


def _lists(world, cache: dict[str, np.ndarray], q: int, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global row ids (OSV offset by len(MP16)), raw similarity and distance to `truth` of query q's cached neighbours, best first."""

    n_mp16 = len(world.mp16["latlon"])
    rows, sims, latlon = [], [], []
    for name, gallery, offset in (("mp16_raw", world.mp16, 0), ("osv_raw", world.osv, n_mp16)):
        idx, sim = cache[f"{name}_idx"][q], cache[f"{name}_sim"][q]
        keep = np.isfinite(sim)
        rows.append(idx[keep].astype(np.int64) + offset); sims.append(sim[keep]); latlon.append(gallery["latlon"][idx[keep]])
    rows, sims, latlon = np.concatenate(rows), np.concatenate(sims), np.concatenate(latlon)
    order = np.argsort(-sims, kind="stable")
    return rows[order], sims[order], _km(latlon[order], *truth)


def _read(world, rows: np.ndarray) -> np.ndarray:
    """Embeddings (float16) of global rows, read in sorted order from the two memmaps."""

    n_mp16 = len(world.mp16["latlon"])
    out = np.empty((len(rows), world.mp16["embeddings"].shape[1]), dtype=np.float16)
    order = np.argsort(rows)
    sorted_rows = rows[order]
    split = np.searchsorted(sorted_rows, n_mp16)
    chunks = [world.mp16["embeddings"][sorted_rows[:split]], world.osv["embeddings"][sorted_rows[split:] - n_mp16]]
    out[order] = np.concatenate([np.asarray(c, dtype=np.float16) for c in chunks])
    return out


def data() -> None:
    dev = json.loads((ROOT / "dev" / "dev.json").read_text(encoding="utf-8"))
    world = _world(dev)
    queries = json.loads((SFT_ROOT / "queries.json").read_text(encoding="utf-8"))
    rows_q = np.asarray([q["row"] for q in queries], dtype=np.int64)
    train = [i for i, q in enumerate(queries) if q["group"] == "held_out" and q["split"] == "train"]
    with np.load(SFT_ROOT / "neighbors.npz") as saved:
        cache = {k: saved[k] for k in ("mp16_raw_idx", "mp16_raw_sim", "osv_raw_idx", "osv_raw_sim")}
    # gallery rows by dev or benchmark photographers never enter a training list
    blocked_authors = np.union1d(world.mp16["author"][np.asarray([e["row"] for e in dev])], load_world().query_author)
    blocked = np.isin(world.mp16["author"], blocked_authors[blocked_authors >= 0])
    author_names = {v: k for k, v in world.vocab["author"].items()}
    lists, used = [], []
    rng = np.random.default_rng(0)
    for n, i in enumerate(train):
        truth = world.mp16["latlon"][rows_q[i]]
        rows, sims, dist = _lists(world, cache, i, truth)
        keep = (rows >= len(blocked)) | ~blocked[np.minimum(rows, len(blocked) - 1)]
        rows, sims, dist = rows[keep], sims[keep], dist[keep]
        pos, far = np.flatnonzero(dist < POS_KM)[:POS_MAX], np.flatnonzero(dist >= POS_KM)
        deep = far[NEG_MAX:]
        neg = np.concatenate((far[:NEG_MAX], np.sort(rng.choice(deep, min(NEG_RANDOM, len(deep)), replace=False))))
        if not len(pos):
            continue  # nothing to pull up: re-ranking can't help these
        author = author_names[int(world.mp16["author"][rows_q[i]])]
        lists.append({"query": i, "rows": np.concatenate((rows[pos], rows[neg])), "positive": len(pos),
                      "select": int(hashlib.md5(author.encode()).hexdigest(), 16) % 10 == SELECT_BUCKET})
        used.append(rows_q[i])
        if (n + 1) % 5000 == 0:
            print(f"lists {n + 1}/{len(train)}", flush=True)
    unique = np.unique(np.concatenate([l["rows"] for l in lists]))
    print(f"{len(lists)} of {len(train)} train photos have a positive in their top-1000; {len(unique)} distinct gallery rows", flush=True)
    started = time.time()
    gallery = np.concatenate([_read(world, unique[s : s + 500_000]) for s in range(0, len(unique), 500_000)])
    print(f"read gallery rows in {time.time() - started:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "gallery_rows.npy", unique)
    np.save(OUT / "gallery_emb.npy", gallery)
    np.save(OUT / "query_emb.npy", np.asarray(world.mp16["embeddings"][np.sort(np.asarray(used))], dtype=np.float16)[np.argsort(np.argsort(np.asarray(used)))])
    np.savez(OUT / "lists.npz", query=np.asarray([l["query"] for l in lists]), positive=np.asarray([l["positive"] for l in lists]),
             select=np.asarray([l["select"] for l in lists]), offsets=np.cumsum([0] + [len(l["rows"]) for l in lists]),
             index=np.searchsorted(unique, np.concatenate([l["rows"] for l in lists])))


class Adapter:
    """Residual MLP adapter (built lazily so the module imports without torch)."""

    @staticmethod
    def build(dim: int, hidden: int = 2048):
        import torch

        class _Adapter(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, dim))
                torch.nn.init.zeros_(self.mlp[-1].weight)
                torch.nn.init.zeros_(self.mlp[-1].bias)
                self.log_tau = torch.nn.Parameter(torch.tensor(math.log(0.05)))

            def forward(self, x):
                x = torch.nn.functional.normalize(x.float(), dim=-1)
                return torch.nn.functional.normalize(x + self.mlp(x), dim=-1)

        return _Adapter()


def train(name: str, lr: float, epochs: int, hidden: int) -> None:
    import torch

    device = torch.device("cuda")
    gallery = torch.as_tensor(np.load(OUT / "gallery_emb.npy"), device=device)
    queries = torch.as_tensor(np.load(OUT / "query_emb.npy"), device=device)
    saved = np.load(OUT / "lists.npz")
    offsets, positive, select, index = saved["offsets"], saved["positive"], saved["select"].astype(bool), torch.as_tensor(saved["index"], device=device)
    width = POS_MAX + NEG_MAX + NEG_RANDOM
    lengths = np.diff(offsets)
    table = np.full((len(lengths), width), -1, dtype=np.int64)  # per list: positions into `index`, padded
    for k, (o, n) in enumerate(zip(offsets[:-1], lengths)):
        table[k, :n] = np.arange(o, o + n)
    table = torch.as_tensor(table, device=device)
    is_pos = torch.as_tensor(np.arange(width)[None, :] < positive[:, None], device=device)
    valid = table >= 0
    train_ids, select_ids = np.flatnonzero(~select), np.flatnonzero(select)
    print(f"{len(train_ids)} training lists, {len(select_ids)} selection lists, {len(gallery)} gallery rows", flush=True)

    model = Adapter.build(gallery.shape[1], hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(train_ids) / BATCH)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps, pct_start=0.1)

    def scores(ids):
        ids = torch.as_tensor(ids, device=device)
        t = table[ids]
        g = model(gallery[index[t.clamp(min=0)]].view(-1, gallery.shape[1])).view(len(ids), width, -1)
        q = model(queries[ids])
        return (g @ q[:, :, None]).squeeze(-1), valid[ids], is_pos[ids]

    def evaluate() -> dict[str, float]:
        """Selection lists: mean reciprocal rank of the first positive and share with a positive ranked first, adapter vs raw (identity) order."""

        model.eval()
        out = {"mrr": [], "top1": []}
        with torch.no_grad():
            for s in range(0, len(select_ids), 1024):
                sc, v, p = scores(select_ids[s : s + 1024])
                sc = sc.masked_fill(~v, -1e9)
                rank = (sc[:, :, None] < sc[:, None, :]).sum(-1)  # 0 = best
                first = rank.masked_fill(~p, 10**6).min(1).values.float()
                out["mrr"].append((1 / (first + 1)).cpu()); out["top1"].append((first == 0).float().cpu())
        model.train()
        return {k: float(torch.cat(v).mean()) for k, v in out.items()}

    print(f"before training (raw SigLIP2 order within the sampled lists): {evaluate()}", flush=True)
    rng = np.random.default_rng(0)
    best, best_state = -1.0, None
    for epoch in range(epochs):
        losses = []
        order = rng.permutation(train_ids)
        for s in range(0, len(order), BATCH):
            sc, v, p = scores(order[s : s + BATCH])
            logits = (sc / model.log_tau.exp()).masked_fill(~v, -1e9)
            loss = -(torch.logsumexp(logits.masked_fill(~p, -1e9), 1) - torch.logsumexp(logits, 1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        m = evaluate()
        print(f"{name} epoch {epoch + 1}/{epochs} loss {np.mean(losses):.3f} tau {model.log_tau.exp().item():.3f} selection {m}", flush=True)
        if m["mrr"] > best:
            best, best_state = m["mrr"], {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save({"state": best_state, "hidden": hidden}, OUT / f"adapter_{name}.pt")
    print(f"saved the best epoch (selection MRR {best:.3f})", flush=True)


def eval_(name: str) -> None:
    import torch

    device = torch.device("cuda")
    dev = json.loads((ROOT / "dev" / "dev.json").read_text(encoding="utf-8"))
    world = _world(dev)
    saved_model = torch.load(OUT / f"adapter_{name}.pt")
    model = Adapter.build(world.mp16["embeddings"].shape[1], saved_model["hidden"]).to(device)
    model.load_state_dict(saved_model["state"])
    model.eval()
    excluded = set(json.loads((ROOT / "val" / "exclude.json").read_text(encoding="utf-8")))
    report: dict[str, Any] = {}
    for tag, root in (("dev", SFT_ROOT), ("val", BENCH_ROOT)):
        photos = [e for e in json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8")) if e["image_id"] not in excluded]
        with np.load(root / "neighbors.npz") as saved:
            cache = {k: saved[k] for k in ("mp16_raw_idx", "mp16_raw_sim", "osv_raw_idx", "osv_raw_sim")}
        heads = dict(np.load(root / "region_head.npz"))
        bench = load_world() if tag == "val" else None
        q_emb = np.stack([np.asarray(world.mp16["embeddings"][e["row"]], dtype=np.float32) if tag == "dev" else bench.query_embeddings[e["index"]] for e in photos])
        del bench
        with torch.no_grad():
            fq = model(torch.as_tensor(q_emb, device=device))
        regions = np.concatenate((world.mp16["region"], world.osv["region"]))
        latlon_all = (world.mp16["latlon"], world.osv["latlon"])
        n_mp16 = len(world.mp16["latlon"])
        arms: dict[str, list] = {"raw": [], "raw + prior": [], "adapter": [], "blend": []}
        for i, e in enumerate(photos):
            rows, sims, _ = _lists(world, cache, e["index"], np.asarray(e["truth"]))
            with torch.no_grad():
                score = (model(torch.as_tensor(_read(world, rows), device=device)) @ fq[i]).cpu().numpy()
            lat = np.where((rows < n_mp16)[:, None], latlon_all[0][np.minimum(rows, n_mp16 - 1)], latlon_all[1][np.maximum(rows - n_mp16, 0)])
            prior = {int(r): float(p) for r, p in zip(heads["head_mlp_regions"][e["index"]], heads["head_mlp_probs"][e["index"]])}
            log_prior = np.log(np.asarray([prior.get(int(r), 0.0) for r in regions[rows]]) + PRIOR_FLOOR)
            arms["raw"].append(_pool([(a, b, s) for (a, b), s in zip(lat, sims)], limit=max(KS)))
            arms["raw + prior"].append(_pool([(a, b, s + 0.01 * lp) for (a, b), s, lp in zip(lat, sims, log_prior)], limit=max(KS)))
            arms["adapter"].append(_pool([(a, b, s) for (a, b), s in zip(lat, score)], limit=max(KS)))
            arms["blend"].append(_pool([(a, b, s + t + 0.01 * lp) for (a, b), s, t, lp in zip(lat, sims, score, log_prior)], limit=max(KS)))  # raw + adapter + prior
            if (i + 1) % 200 == 0:
                print(f"{tag} {i + 1}/{len(photos)}", flush=True)
        truth = np.asarray([e["truth"] for e in photos])
        entry: dict[str, Any] = {"n": len(photos), "recall": {}, "extra": {}}
        for arm, pools in arms.items():
            entry["recall"][arm] = {f"@{k}": {f"<{t:g} km": float(np.mean([_hit(p[:k], truth[j], t) for j, p in enumerate(pools)])) for t in (25.0, 200.0)} for k in KS}
        current = [[tuple(c) for c in e["pool"]] for e in photos]
        cur_hit = {t: np.asarray([_hit(p, truth[j], t) for j, p in enumerate(current)]) for t in (25.0, 200.0)}
        entry["pool oracle"] = {f"<{t:g} km": float(h.mean()) for t, h in cur_hit.items()}
        for arm in ("raw + prior", "adapter", "blend"):
            entry["extra"][arm] = {}
            for k in EXTRA:
                entry["extra"][arm][f"+{k}"] = {}
                for t in (25.0, 200.0):
                    h = np.asarray([_hit(_extend(current[j], arms[arm][j], k), truth[j], t) for j in range(len(photos))])
                    entry["extra"][arm][f"+{k}"][f"<{t:g} km"] = list(_bootstrap((h & ~cur_hit[t]).astype(float)))
        report[tag] = entry
        print(f"\n== {tag} (n={len(photos)}) == pool oracle <25 / <200 km: {entry['pool oracle']['<25 km']:.1%} / {entry['pool oracle']['<200 km']:.1%}")
        print("  right-place cluster among the first K clusters (<25 km | <200 km):")
        for arm in arms:
            print(f"    {arm:12s} " + "  ".join(f"K={k}: {entry['recall'][arm][f'@{k}']['<25 km']:.1%} | {entry['recall'][arm][f'@{k}']['<200 km']:.1%}" for k in KS))
        print("  current pool + K extra clusters, oracle gain <25 km [95% CI] (and <200 km):")
        for arm in ("raw + prior", "adapter", "blend"):
            print(f"    {arm:12s} " + "  ".join("+{}: {:+.1f} [{:+.1f},{:+.1f}] ({:+.1f})".format(k, *(100 * x for x in entry["extra"][arm][f"+{k}"]["<25 km"]), 100 * entry["extra"][arm][f"+{k}"]["<200 km"][0]) for k in EXTRA))
        (OUT / f"report_{name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("data", "train", "report"))
    parser.add_argument("--name", default="base")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--hidden", type=int, default=2048)
    args = parser.parse_args(argv)
    if args.node == "data":
        data()
    elif args.node == "train":
        train(args.name, args.lr, args.epochs, args.hidden)
    else:
        eval_(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
