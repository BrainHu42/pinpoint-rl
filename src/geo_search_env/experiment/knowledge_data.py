# Knowledge SFT data: MP16 photos from Pinpoint's training buckets (0-98) -> "country > region > city > neighbourhood" (Overture labels, place_labels.py),
# in the format sft_train.py reads ({image_id, split, prompt, target}). The prompt and answer format are those name_score.py scores.
# Usage: PYTHONPATH=src ~/.venvs/sft/bin/python -m geo_search_env.experiment.knowledge_data select [--photos 200000]   (CPU, pyarrow; needs artifacts/place_labels/mp16.parquet)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.knowledge_data overlay                       (base VLM served at :8765; ~30 min)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.knowledge_data dataset

"""select:  eligible photos: md5 bucket < 99; photographer not in dev, im2gps3k / yfcc4k (all splits) or wikimedia; not a near-duplicate (SigLIP2 cosine
         >= DUP_SIM among the cached top-1000 MP16 neighbours) of any cached query (benchmarks, the MP16 pool incl. dev, wikimedia); a country and region
         label. At most PER_CITY photos per (country, region, city), then a random sample.
overlay: P(burned-in GPS) for the selected photos with the base VLM (sft_data's prompt and cut-off).
dataset: drop overlay photos, write artifacts/knowledge/knowledge.jsonl (VAL_ROWS rows split 'val' for eval loss, the rest 'train')."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .name_score import PROMPT, SEP, name_parts
from .place_labels import CANONICAL
from .query_evidence import ROOT
from .strategy_search import MP16_EMBED

OUT = Path("artifacts/knowledge")
MP16_CSV = Path("/data/hf/datasets/MP16-Pro/metadata/MP16_Pro_filtered.csv")
CACHES = (Path("artifacts/strategy_search/neighbors.npz"), Path("artifacts/sft/neighbors.npz"), Path("artifacts/wikimedia/neighbors.npz"))
DUP_SIM = 0.95
PER_CITY = 400
VAL_ROWS = 2_000


def _bucket(image_id: str) -> int:
    return int(hashlib.md5(image_id.encode("utf-8")).hexdigest(), 16) % 100


def select(photos: int, seed: int = 0) -> None:
    import pyarrow.parquet as pq

    from ..data.benchmarks import load_benchmark

    ids = (MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines()
    author_of: dict[str, str] = {}
    with MP16_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            author_of[r["IMG_ID"]] = r["AUTHOR"]
    authors = np.asarray([author_of.get(i, "") for i in ids])
    blocked_authors = {authors[e["row"]] for e in json.loads((ROOT / "dev" / "dev.json").read_text(encoding="utf-8"))}
    for name in ("im2gps3k", "yfcc4k", "wikimedia"):
        blocked_authors |= {a for a in load_benchmark(name).authors if a}
    blocked_authors.discard("")
    dup = np.zeros(len(ids), bool)
    for path in CACHES:
        z = np.load(path)
        idx, sim = z["mp16_raw_idx"], z["mp16_raw_sim"]
        dup[idx[sim >= DUP_SIM]] = True
    labels = pq.read_table(OUT.parent / "place_labels" / "mp16.parquet").to_pandas()
    assert (labels["row"].to_numpy() == np.arange(len(ids))).all()
    bucket = np.fromiter((_bucket(i) for i in ids), dtype=np.int16, count=len(ids))
    has_label = labels["country"].notna().to_numpy() & labels["region"].notna().to_numpy()
    eligible = (bucket < 99) & ~np.isin(authors, list(blocked_authors)) & ~dup & has_label
    print(f"{len(ids)} MP16 photos; bucket < 99 {(bucket < 99).mean():.1%}; blocked photographers {len(blocked_authors)} "
          f"({np.isin(authors, list(blocked_authors)).sum()} photos); near-duplicates of a cached query {dup.sum()}; eligible {eligible.sum()}")
    rng = np.random.default_rng(seed)
    rows = rng.permutation(np.flatnonzero(eligible))
    city = (labels["country"].fillna("") + "|" + labels["region"].fillna("") + "|" + labels["locality"].fillna("")).to_numpy()
    seen: dict[str, int] = {}
    chosen = []
    for r in rows:
        if seen.get(city[r], 0) < PER_CITY:
            seen[city[r]] = seen.get(city[r], 0) + 1
            chosen.append(int(r))
            if len(chosen) == photos:
                break
    chosen = np.sort(np.asarray(chosen))
    out = []
    for r in chosen:
        lab = {lv: (None if v != v else v) for lv, v in labels.iloc[r][list(CANONICAL)].items()}
        out.append({"row": int(r), "image_id": ids[r], "target": SEP.join(name_parts(lab))})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selected.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
    depth = np.bincount([len(o["target"].split(SEP)) for o in out], minlength=5)
    countries = len({o["target"].split(SEP)[0] for o in out})
    print(f"selected {len(out)} photos from {len(seen)} cities ({countries} countries); answer depth (levels) " + ", ".join(f"{d}: {depth[d]}" for d in range(1, 5)))


def overlay(server: str = "http://127.0.0.1:8765", workers: int = 64) -> None:
    from .sft_data import MP16Images, _p_yes

    selected = json.loads((OUT / "selected.json").read_text(encoding="utf-8"))
    images = MP16Images()

    def score(o: dict) -> float | None:
        data = images.read(o["image_id"])
        try:
            return _p_yes(server, data) if data else None
        except Exception:
            return None

    with ThreadPoolExecutor(workers) as pool:
        scores = list(pool.map(score, selected))
    (OUT / "overlay_scores.json").write_text(json.dumps({o["image_id"]: s for o, s in zip(selected, scores)}) + "\n", encoding="utf-8")
    print(f"overlay scores for {len(scores)} photos, {sum(s is None for s in scores)} failed")


def dataset(seed: int = 0) -> None:
    from .sft_data import OVERLAY_THRESHOLD

    selected = json.loads((OUT / "selected.json").read_text(encoding="utf-8"))
    scores = json.loads((OUT / "overlay_scores.json").read_text(encoding="utf-8"))
    keep = [o for o in selected if scores.get(o["image_id"]) is not None and scores[o["image_id"]] < OVERLAY_THRESHOLD]
    order = np.random.default_rng(seed).permutation(len(keep))
    with (OUT / "knowledge.jsonl").open("w", encoding="utf-8") as f:
        for n, i in enumerate(order):
            o = keep[i]
            f.write(json.dumps({"image_id": o["image_id"], "split": "val" if n < VAL_ROWS else "train", "prompt": PROMPT, "target": o["target"]}) + "\n")
    print(f"{len(keep)} rows ({len(selected) - len(keep)} dropped: overlay or unreadable) -> {OUT / 'knowledge.jsonl'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("select", "overlay", "dataset"))
    parser.add_argument("--photos", type=int, default=200_000)
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    args = parser.parse_args(argv)
    if args.node == "select":
        select(args.photos)
    elif args.node == "overlay":
        overlay(args.server)
    else:
        dataset()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
