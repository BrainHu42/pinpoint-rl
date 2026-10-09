# Deeper retrieval votes for area agreement (after lesson 68): Pinpoint's pool has <= 12 candidates; agreement among them already adds +1.0 pts
# on train. Here the top K index photos by Pinpoint's contrastive image similarity (same photographer excluded) vote too: each candidate's
# support adds beta x the softmax(sim / tau) mass of the neighbours within RADIUS_KM of it. Settings chosen on train, measured on dev3k.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.deep_votes run --tags train dev3k   (GPU, one index pass)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.deep_votes report

"""run:    per photo of the sets, the top K neighbours (index rows, similarity, lat / lon) -> artifacts/agent/deep_votes_{tag}.npz.
report: support_j = sum_pool softmax(s / T)_k [d(j, k) < R] + beta sum_nb softmax(sim / tau)_n [d(j, n) < R]; pick = argmax over Pinpoint's
        candidates. Grid over beta, tau on train; the train choice applied to dev3k; also neighbours alone (vote for the candidate whose
        25 km disc holds the most neighbour mass)."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
PHOTO_SETS = Path("artifacts/photo_sets")
SFT_ROOT = Path("artifacts/sft")
NVME_INDEX = Path("/opt/dlami/nvme/pinpoint/attention_retrieval_index")
K = 100


def run(tags: Sequence[str]) -> None:
    import torch

    from ..models.pinpoint_reranker import PinpointReranker
    from .strategy_search import load_world

    queries = json.loads((SFT_ROOT / "queries.json").read_text(encoding="utf-8"))
    sets = {t: json.loads((PHOTO_SETS / t / "photos.json").read_text(encoding="utf-8")) for t in tags}
    allq = [queries[e["index"]] for t in tags for e in sets[t]]
    world = load_world(mp16_queries=allq)
    reranker = PinpointReranker(device="cuda", index_dir=NVME_INDEX if (NVME_INDEX / "manifest.json").exists() else None)
    _, projected = reranker.encode_queries(world.query_embeddings)
    q = projected.half()
    q_author = torch.as_tensor(world.query_author, device=reranker.device)
    index_author = world.mp16["author"][reranker.cache_indices]
    best_v = torch.full((len(q), K), float("-inf"), device=reranker.device, dtype=torch.float16)
    best_i = torch.zeros((len(q), K), dtype=torch.long, device=reranker.device)
    count = len(reranker.latlon)
    with torch.inference_mode():
        for start in range(0, count, reranker.chunk):
            end = min(start + reranker.chunk, count)
            block = torch.as_tensor(np.asarray(reranker.image[start:end]), device=reranker.device)
            authors = torch.as_tensor(index_author[start:end], device=reranker.device)
            for b in range(0, len(q), reranker.query_batch):
                s = q[b:b + reranker.query_batch] @ block.T
                s.masked_fill_(authors.unsqueeze(0) == q_author[b:b + reranker.query_batch].unsqueeze(1), float("-inf"))
                v, i = torch.topk(s, min(K, end - start), dim=-1)
                mv, keep = torch.topk(torch.cat((best_v[b:b + reranker.query_batch], v), -1), K, dim=-1)
                best_i[b:b + reranker.query_batch] = torch.gather(torch.cat((best_i[b:b + reranker.query_batch], i + start), -1), -1, keep)
                best_v[b:b + reranker.query_batch] = mv
            if (start // reranker.chunk) % 16 == 0:
                print(f"  index {end}/{count}", flush=True)
    rows, sims = best_i.cpu().numpy(), best_v.float().cpu().numpy()
    latlon = np.asarray(reranker.latlon)[rows.reshape(-1)].reshape(len(rows), K, 2)
    offset = 0
    for t in tags:
        n = len(sets[t])
        np.savez(OUT / f"deep_votes_{t}.npz", sims=sims[offset:offset + n].astype(np.float32), latlon=latlon[offset:offset + n].astype(np.float32))
        print(f"{t}: {n} photos x {K} neighbours -> {OUT / f'deep_votes_{t}.npz'}", flush=True)
        offset += n


def _prepare(tag: str):
    from .pinpoint_reference import load_pools
    from .pinpoint_tool import _km

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    z = dict(np.load(OUT / f"deep_votes_{tag}.npz"))
    pools = load_pools("mp16")
    out = []
    for i, e in enumerate(photos):
        valid = pools["valid"][e["index"]]
        cand = np.asarray(e["pool"], np.float64)
        nb = z["latlon"][i].astype(np.float64)
        both = _km(np.concatenate([cand, nb, np.asarray([e["truth"]])]))
        c = len(cand)
        out.append({"s": pools["scores"][e["index"]][valid].astype(np.float64), "cc": both[:c, :c] < 25, "cn": both[:c, c:c + K] < 25,
                    "sim": z["sims"][i].astype(np.float64), "km": both[:c, -1]})
    return out


def _picks(data, beta: float | None, tau: float, temp: float = 0.2) -> np.ndarray:
    """beta None: neighbours alone."""

    out = []
    for r in data:
        w = np.exp((r["s"] - r["s"].max()) / temp)
        w /= w.sum()
        sim = np.where(np.isfinite(r["sim"]), r["sim"], -np.inf)
        v = np.exp((sim - sim.max()) / tau)
        v /= v.sum()
        support = r["cn"] @ v if beta is None else r["cc"] @ w + beta * (r["cn"] @ v)
        out.append(int(np.argmax(support)))
    return np.asarray(out)


def report() -> None:
    from .strategy_search import _bootstrap

    train, dev = _prepare("train"), _prepare("dev3k")
    grid = list(itertools.product((0.0, 0.25, 0.5, 1.0, 2.0, 4.0, None), (0.01, 0.02, 0.05, 0.1)))
    gain = lambda data, p: np.asarray([int(r["km"][j] < 25) - int(r["km"][0] < 25) for r, j in zip(data, p)], float)
    scores = {g: gain(train, _picks(train, *g)).mean() for g in grid}
    for g, v in sorted(scores.items(), key=lambda x: -x[1])[:6]:
        print(f"  train: beta {g[0]}, tau {g[1]}: {100 * v:+.2f} pts")
    best = max(scores, key=scores.get)
    for name, g in (("pool agreement only (lesson 68)", (0.0, 0.05)), (f"with deep votes, train-chosen beta {best[0]}, tau {best[1]}", best)):
        for tag, data in (("train", train), ("dev3k", dev)):
            p = _picks(data, *g)
            d = gain(data, p)
            g1 = np.asarray([int(r["km"][j] < 1) - int(r["km"][0] < 1) for r, j in zip(data, p)], float)
            c = _bootstrap(d)
            print(f"  {tag:5s} {name:54s} < 25 km {100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}] (fixes {int((d > 0).sum())}, "
                  f"breaks {int((d < 0).sum())}); < 1 km {100 * g1.mean():+.1f}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("run", "report"))
    parser.add_argument("--tags", nargs="*", default=("train", "dev3k"))
    args = parser.parse_args(argv)
    run(args.tags) if args.cmd == "run" else report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
