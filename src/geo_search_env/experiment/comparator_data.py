# Training pairs for the fine-tuned comparator: (query photo, exemplar of a pool candidate) -> same place?
# Usage: .venv/bin/python -m geo_search_env.experiment.comparator_data   (CPU, ~10 min)

"""For every MP16 train photo (Pinpoint's held-out bucket 99, training photographers), its top TOPK pool candidates in reranker order, each with the
best exemplar: the database photo within 1 km of the candidate (not by the query's photographer) most similar to the query, as at test time
(`exemplar_judge`). The label comes from the distance of the candidate to the truth: < 1 km is the same place, >= 10 km is not, 1-10 km is dropped.

Writes artifacts/query_evidence/comparator/pairs_train.jsonl, one row per (photo, candidate, exemplar): query id, exemplar id, label, km, rank,
exemplar rank (0 = the most similar, 1 = the second; the second-best exemplar is kept only for positives, as augmentation).
"""

from __future__ import annotations

import json
import math

import numpy as np

from .query_evidence import ROOT, SFT_ROOT
from .strategy_search import EARTH_KM, MP16_EMBED, _xyz, load_world
from .wiki_backend import _km

TOPK = 8
EXEMPLAR_KM = 1.0
POSITIVE_KM, NEGATIVE_KM = 1.0, 10.0
OUT = ROOT / "comparator"


def main() -> int:
    from scipy.spatial import cKDTree

    photos = json.loads((ROOT / "train" / "dev.json").read_text(encoding="utf-8"))
    saved = dict(np.load(SFT_ROOT / "candidates.npz"))
    ids = (MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines()
    world = load_world(mp16_queries=[{"row": e["row"], "image_id": e["image_id"]} for e in photos])
    tree = cKDTree(_xyz(world.mp16["latlon"]))
    chord = 2 * math.sin(EXEMPLAR_KM / EARTH_KM / 2)
    OUT.mkdir(parents=True, exist_ok=True)
    kept = {"positive": 0, "negative": 0, "dropped": 0, "no exemplar": 0}
    with (OUT / "pairs_train.jsonl").open("w", encoding="utf-8") as out:
        for m, e in enumerate(photos):
            coords, valid, ranking = saved["coords"][e["index"]], saved["valid"][e["index"]], saved["ranking"][e["index"]]
            top = [c for c in ranking if valid[c]][:TOPK]
            q = world.query_embeddings[m] / np.linalg.norm(world.query_embeddings[m])
            author = int(world.query_author[m])
            for rank, c in enumerate(top):
                km = float(_km(coords[c][None, :], *e["truth"])[0])
                label = 1 if km < POSITIVE_KM else 0 if km >= NEGATIVE_KM else None
                if label is None:
                    kept["dropped"] += 1
                    continue
                rows = [r for r in tree.query_ball_point(_xyz(coords[c]), chord) if world.mp16["author"][r] != author]
                if not rows:
                    kept["no exemplar"] += 1
                    continue
                rows = np.sort(rows)
                emb = np.asarray(world.mp16["embeddings"][rows], dtype=np.float32)
                sims = emb @ q / np.linalg.norm(emb, axis=1)
                for k, i in enumerate(np.argsort(-sims)[: 2 if label else 1]):
                    out.write(json.dumps({"photo": m, "query": e["image_id"], "exemplar": ids[rows[i]], "label": label, "km": km, "rank": rank, "exemplar_rank": k,
                                          "sim": float(sims[i])}) + "\n")
                kept["positive" if label else "negative"] += 1
            if (m + 1) % 5000 == 0:
                print(f"  {m + 1}/{len(photos)} photos: {kept}", flush=True)
    print(f"candidates: {kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
