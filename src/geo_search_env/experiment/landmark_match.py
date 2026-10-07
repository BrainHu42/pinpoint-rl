# Visual landmark matching as a tool (AGENT_DIRECTION.md): for photos a VLM flags as showing a landmark (observe.py landmarks), keypoint-match the query
# (SuperPoint + LightGlue) against the gallery photos nearest each retrieval candidate. Exemplars come from the neighbour cache (top-1000 SigLIP2 MP16
# neighbours per query, same-photographer rows already excluded): per candidate, the 2 most similar cached neighbours within 1 km.
# Usage: PYTHONPATH=src .venvs/sft/bin/python -m geo_search_env.experiment.landmark_match --tags dev val   (GPU; env with lightglue)

"""Writes artifacts/agent/kp_{tag}.npz: F [photos, C, 2] = max LightGlue matches (confidence > 0.2) over the candidate's exemplars, and whether the
candidate had an exemplar at all; zeros for photos without a landmark. geo_tools._load picks these up as the 'keypoints' features."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .photo_sets import load_photos
from .sft_data import MP16Images

OUT = Path("artifacts/agent")
CACHE = Path("artifacts/sft")
GALLERY = Path("data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384")
EXEMPLARS, NEAR_KM, MIN_CONF, LONG_SIDE = 2, 1.0, 0.2, 1024


def _km(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lats), np.radians(lons)
    h = np.sin((la - np.radians(lat)) / 2) ** 2 + np.cos(la) * np.cos(np.radians(lat)) * np.sin((lo - np.radians(lon)) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(h))


def main(argv: Sequence[str] | None = None) -> int:
    import torch
    from lightglue import LightGlue, SuperPoint
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=["dev", "val"])
    parser.add_argument("--clues", default="9b")
    args = parser.parse_args(argv)
    queries = json.loads((CACHE / "queries.json").read_text(encoding="utf-8"))
    q_of_row = {int(q["row"]): i for i, q in enumerate(queries)}
    cache = np.load(CACHE / "neighbors.npz", mmap_mode="r")
    raw_idx = cache["mp16_raw_idx"]
    latlon = np.fromfile(GALLERY / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2)
    ids = (GALLERY / "image_ids.txt").read_text(encoding="utf-8").split("\n")
    images = MP16Images()
    extractor = SuperPoint(max_num_keypoints=1024).eval().cuda()
    matcher = LightGlue(features="superpoint").eval().cuda()
    feats: dict[str, dict] = {}

    def features(image_id: str) -> dict | None:
        if image_id not in feats:
            data = images.read(image_id)
            if data is None:
                feats[image_id] = None
            else:
                im = Image.open(io.BytesIO(data)).convert("RGB")
                im.thumbnail((LONG_SIDE, LONG_SIDE))
                t = torch.as_tensor(np.asarray(im), dtype=torch.float32).permute(2, 0, 1).cuda() / 255
                with torch.inference_mode():
                    feats[image_id] = extractor.extract(t)
            if len(feats) > 4000:  # bound GPU memory: drop the oldest entries
                for k in list(feats)[:1000]:
                    del feats[k]
        return feats[image_id]

    for tag in args.tags:
        photos = load_photos(tag)
        lms = json.loads((OUT / f"landmark_{tag}_{args.clues}.json").read_text(encoding="utf-8"))
        C = max(len(e["pool"]) for e in photos)
        F = np.zeros((len(photos), C, 2), np.float32)
        done = 0
        for p, e in enumerate(photos):
            if not (lms.get(e["image_id"]) or {}).get("landmark"):
                continue
            q = q_of_row.get(int(e["row"]))
            fq = features(e["image_id"])
            if q is None or fq is None:
                continue
            nb = np.asarray(raw_idx[q])
            nb_ll = latlon[nb]
            for k, (lat, lon) in enumerate(e["pool"]):
                near = nb[_km(lat, lon, nb_ll[:, 0], nb_ll[:, 1]) <= NEAR_KM][:EXEMPLARS]
                if not len(near):
                    continue
                F[p, k, 1] = 1.0
                best = 0
                for g in near:
                    fg = features(ids[int(g)])
                    if fg is None:
                        continue
                    with torch.inference_mode():
                        out = matcher({"image0": fq, "image1": fg})
                    best = max(best, int((out["matching_scores0"][0] > MIN_CONF).sum()))
                F[p, k, 0] = best
            done += 1
            if done % 25 == 0:
                print(f"  {tag}: {done} landmark photos matched", flush=True)
        np.savez(OUT / f"kp_{tag}.npz", F=F, image_ids=np.asarray([e["image_id"] for e in photos]))
        print(f"{tag}: keypoint features for {done} landmark photos -> {OUT / f'kp_{tag}.npz'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
