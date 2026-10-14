# Region hypotheses for the pipeline (lesson 85; lesson 84's candidates): per photo the K most frequent (country, region) labels among its 1,000
# nearest index photos (Pinpoint's image embedding, photographer excluded), each searched with Pinpoint restricted to the region (tools.pinpoint).
# Regions whose restricted top-1 lies within BOX_KM of one of the photo's Pinpoint areas are dropped (already a hypothesis). Saved in the
# hypotheses.py format as source 2 -> artifacts/agent/hypotheses_{tag}_regionhyps.npz, merged by hypothesis_compare / pinpoint_set restricted
# with --extra-hyps regionhyps; hypothesis_agent reads them with HYP_VARIANT=regionhyps.
# Usage: PINPOINT_RERANKER_CKPT=artifacts/pinpoint_train/setft_a/checkpoints/ckpt_final.pt PYTHONPATH=src \
#            .venv/bin/python -m geo_search_env.experiment.region_hyps build --tag dev3k_setft_a [--k 16]
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.region_hyps oracle --tag dev3k_setft_a

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
PHOTO_SETS = Path("artifacts/photo_sets")
NEIGHBOURS = 1000
BOX_KM = 25.0


def _regions(tool, projected, authors, k: int, batch: int = 256) -> tuple[list[list[str]], list[list[float]]]:
    """The k most frequent regions among each query's NEIGHBOURS nearest index photos (photographer excluded) and their shares."""

    import torch

    res = tool.reranker._resident
    index_author = torch.as_tensor(tool.index_author(), device=res["image"].device)
    _, region = tool.labels()
    names, shares = [], []
    for b in range(0, len(projected), batch):
        q = projected[b:b + batch].half()
        s = res["image"] @ q.T  # [index, batch]
        s.masked_fill_(index_author[:, None] == torch.as_tensor(authors[b:b + batch], device=s.device)[None], -1e4)
        top = torch.topk(s, NEIGHBOURS, dim=0).indices.T.cpu().numpy()
        for row in top:
            labels, counts = np.unique([r for r in region[row] if r], return_counts=True)
            order = np.argsort(-counts, kind="stable")[:k]
            names.append([str(x) for x in labels[order]])
            shares.append([float(x) / NEIGHBOURS for x in counts[order]])
    return names, shares


def build(tag: str, k: int) -> None:
    from ..tools.pinpoint import Constraint, Pinpoint
    from .hypotheses import photo_world
    from .strategy_search import _haversine_km

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    world, qrows = photo_world(photos)
    emb = np.asarray(world.query_embeddings[qrows], np.float32)
    authors = np.asarray(world.query_author[qrows], np.int64)
    tool = Pinpoint(targets=None)  # the pipeline's hypothesis searches use the reranker's own list sizes
    tool.reranker.load_resident()
    tool._resident = True
    _, projected = tool.reranker.encode_queries(emb)
    names, shares = _regions(tool, projected, authors, k)
    del projected
    areas = dict(np.load(OUT / f"hypotheses_{tag}.npz"))
    centres: dict[int, np.ndarray] = {}
    for p in np.unique(areas["photo"]):
        m = (areas["photo"] == p) & (areas["source"] == 0)
        centres[int(p)] = areas["center"][m]
    jobs = [(p, r, region, share) for p in range(len(photos)) for r, (region, share) in enumerate(zip(names[p], shares[p]))]
    print(f"{tag}: {len(photos)} photos, {len(jobs)} region searches", flush=True)
    keep, parts = [], []
    step = 16384  # bounded memory per call (the results of all searches are kept, not the index)
    for s in range(0, len(jobs), step):
        chunk = jobs[s:s + step]
        idx = np.asarray([p for p, _, _, _ in chunk])
        found = tool.search(emb[idx], [Constraint(region=region) for _, _, region, _ in chunk], authors[idx])
        parts.append(found)
        print(f"  {s + len(chunk)}/{len(jobs)}", flush=True)
    cat = {key: np.concatenate([getattr(f, key) for f in parts]) for key in ("rows", "latlon", "scores", "features")}
    for j, (p, r, region, share) in enumerate(jobs):
        top = cat["latlon"][j, 0]
        if not np.isfinite(top).all():
            continue
        here = centres.get(p, np.zeros((0, 2)))
        if len(here) and (_haversine_km(top[0], top[1], here) < BOX_KM).any():
            continue
        keep.append(j)
    keep = np.asarray(keep)
    np.savez(OUT / f"hypotheses_{tag}_regionhyps.npz", photo=np.asarray([jobs[j][0] for j in keep]), source=np.full(len(keep), 2),
             rank=np.asarray([jobs[j][1] for j in keep]), support=np.asarray([jobs[j][3] for j in keep]),
             center=cat["latlon"][keep, 0].astype(np.float64), latlon=cat["latlon"][keep], scores=cat["scores"][keep],
             features=cat["features"][keep], rows=cat["rows"][keep], region=np.asarray([jobs[j][2] for j in keep]))
    print(f"-> {OUT / f'hypotheses_{tag}_regionhyps.npz'}: {len(keep)} of {len(jobs)} kept (others empty or within {BOX_KM:.0f} km of a "
          f"Pinpoint area)", flush=True)
    oracle(tag)


def oracle(tag: str) -> None:
    """% < 25 km for the best hypothesis (its restricted top-1): Pinpoint areas + clue places (the pipeline's), plus region hypotheses of
    rank < K."""

    from .strategy_search import _haversine_km

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    truth = np.asarray([e["truth"] for e in photos], np.float64)
    top1 = np.asarray([e["candidates"][0][1:] for e in photos], np.float64)
    base = np.asarray([_haversine_km(*truth[i], top1[i:i + 1])[0] < 25 for i in range(len(photos))])

    def best(files, max_rank=99):
        ok = base.copy()
        for name in files:
            path = OUT / f"hypotheses_{tag}{name}.npz"
            if not path.exists():
                continue
            z = np.load(path)
            for p, r, src, ll in zip(z["photo"], z["rank"], z["source"], z["latlon"][:, 0]):
                if (src != 2 or r < max_rank) and np.isfinite(ll).all() and _haversine_km(*truth[p], ll[None])[0] < 25:
                    ok[p] = True
        return ok.mean()

    pipeline = best(["", "_cluehyps"])
    print(f"{tag}: retrained top-1 {base.mean():.1%}; best of the pipeline's hypotheses {pipeline:.1%}")
    for kk in (4, 8, 16):
        print(f"  + region hypotheses of rank < {kk}: {best(['', '_cluehyps', '_regionhyps'], kk):.1%}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("build", "oracle"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--k", type=int, default=16)
    args = parser.parse_args(argv)
    build(args.tag, args.k) if args.cmd == "build" else oracle(args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
