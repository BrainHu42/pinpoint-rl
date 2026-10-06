# Keypoint verification of map-search candidates: DISK keypoints + LightGlue matches + MAGSAC fundamental-matrix inliers between the query photo and each gallery
# photo of map_search.py match-pairs (the 100 most similar gallery photos within 25 km of the reranker top-1).
# Usage: PYTHONPATH=src ~/.venvs/match/bin/python -m geo_search_env.experiment.geo_match score    (GPU, ~2 GB; ~30 min for the 342 dev photos)
#        PYTHONPATH=src ~/.venvs/match/bin/python -m geo_search_env.experiment.geo_match report
# ~/.venvs/match = the sft env's packages (sft.pth) + kornia, kornia_rs, opencv-python-headless (installed with --no-deps).

"""score:  inliers per (query, gallery photo) pair -> map_search_match_scores_dev.json.
report: within-photo AUC of inliers (gallery photo < 1 km from the truth vs 1-25 km) against SigLIP2 similarity, and the rule "move to the gallery photo with
        the most inliers if it has >= T, else keep the top-1" over T, against the reranker top-1 (exploratory: T is not chosen on held-out photos yet)."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from .query_evidence import ROOT

PAIRS, SCORES = ROOT / "map_search_match_pairs_dev.json", ROOT / "map_search_match_scores_dev.json"
OSV_IMAGES = Path("/data/hf/datasets/osv5m/images/train")
LONG_SIDE, KEYPOINTS = 1024, 2048
THRESHOLDS = (10, 15, 20, 30, 50, 80, 120)
LOGIT_THRESHOLDS = (-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0)


def _osv_paths(ids: set[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for folder in sorted(OSV_IMAGES.iterdir()):
        with os.scandir(folder) as entries:
            for entry in entries:
                stem = entry.name[:-4]
                if stem in ids:
                    found[stem] = Path(entry.path)
    return found


def score() -> None:
    import cv2
    from kornia.feature import DISK, LightGlueMatcher, laf_from_center_scale_ori
    from PIL import Image
    import torch

    from .sft_data import MP16Images

    photos = json.loads(PAIRS.read_text(encoding="utf-8"))
    device = torch.device("cuda")
    disk = DISK.from_pretrained("depth").to(device).eval()
    matcher = LightGlueMatcher("disk").to(device).eval()
    mp16 = MP16Images()
    osv = _osv_paths({g["id"] for p in photos for g in p["gallery"] if g["source"] == "osv"})
    print(f"{len(osv)} OSV image paths found", flush=True)

    def load(source: str, image_id: str):
        try:
            raw = mp16.read(image_id) if source == "mp16" else (osv[image_id].read_bytes() if image_id in osv else None)
            if raw is None:
                return None
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            scale = LONG_SIDE / max(img.size)
            if scale < 1:
                img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR)
            return np.asarray(img)
        except Exception:
            return None

    @torch.inference_mode()
    def features(arr: np.ndarray):
        t = torch.from_numpy(arr).permute(2, 0, 1)[None].float().div(255).to(device)
        f = disk(t, KEYPOINTS, pad_if_not_divisible=True)[0]
        kp = f.keypoints
        laf = laf_from_center_scale_ori(kp[None], torch.ones(1, len(kp), 1, 1, device=device))
        return kp, f.descriptors, laf, torch.tensor(arr.shape[:2], device=device)

    @torch.inference_mode()
    def inliers(a, b) -> tuple[int, int]:
        if len(a[0]) < 8 or len(b[0]) < 8:
            return 0, 0
        _, idx = matcher(a[1], b[1], a[2], b[2], hw1=a[3], hw2=b[3])
        if len(idx) < 8:
            return int(len(idx)), 0
        p1, p2 = a[0][idx[:, 0]].cpu().numpy(), b[0][idx[:, 1]].cpu().numpy()
        try:
            _, mask = cv2.findFundamentalMat(p1, p2, cv2.USAC_MAGSAC, 1.0, 0.999, 10000)
        except cv2.error:  # degenerate match sets
            return int(len(idx)), 0
        return int(len(idx)), int(mask.sum()) if mask is not None else 0

    done = {p["image_id"]: p for p in json.loads(SCORES.read_text(encoding="utf-8"))} if SCORES.exists() else {}
    out, start, pairs = [], time.time(), 0
    with ThreadPoolExecutor(8) as pool:
        for n, p in enumerate(photos):
            if p["image_id"] in done:
                out.append(done[p["image_id"]])
                continue
            query = load("mp16", p["image_id"])
            images = pool.map(lambda g: load(g["source"], g["id"]), p["gallery"])
            qf = features(query) if query is not None else None
            result = []
            for g, img in zip(p["gallery"], images):
                if qf is None or img is None:
                    result.append(None)
                    continue
                result.append(inliers(qf, features(img)))
                pairs += 1
            out.append({"image_id": p["image_id"], "matches": [r[0] if r else None for r in result], "inliers": [r[1] if r else None for r in result]})
            if (n + 1) % 10 == 0:
                SCORES.write_text(json.dumps(out) + "\n", encoding="utf-8")
                print(f"  {n + 1}/{len(photos)} photos, {pairs / (time.time() - start):.1f} pairs/s", flush=True)
    SCORES.write_text(json.dumps(out) + "\n", encoding="utf-8")
    missing = sum(x is None for p in out for x in p["inliers"])
    print(f"done: {len(out)} photos, {missing} pairs without an image")


def _bootstrap(x: np.ndarray, reps: int = 2000) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    means = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def report(verifier: str = "inliers") -> None:
    """verifier: 'inliers' (keypoint matching) or a comparator name whose multi_exemplar scores for tag maptop100_dev exist (logit of P(same place); vote
    weights are the probabilities). Gallery photos without a score (OSV for the comparator) never count."""

    photos = json.loads(PAIRS.read_text(encoding="utf-8"))
    if verifier == "inliers":
        scores = {p["image_id"]: p["inliers"] for p in json.loads(SCORES.read_text(encoding="utf-8"))}
        thresholds, weight = THRESHOLDS, lambda v: v
    else:
        from .multi_exemplar import _scores_path

        scored = json.loads(_scores_path(verifier, "maptop100_dev").read_text(encoding="utf-8"))
        scores = {p["image_id"]: [float(np.log(x[0] / (1 - x[0]))) if x and x[0] is not None and 0 < x[0] < 1 else None for x in p["p_same"]] for p in scored}
        thresholds, weight = LOGIT_THRESHOLDS, lambda v: 1 / (1 + np.exp(-v))
    photos = [p for p in photos if p["image_id"] in scores]
    km = [np.asarray([g["km"] for g in p["gallery"]]) for p in photos]
    sim = [np.asarray([g["sim"] for g in p["gallery"]]) for p in photos]
    inl = [np.asarray([x if x is not None else -1e9 for x in scores[p["image_id"]]], float) for p in photos]
    top1 = np.asarray([p["top1_km"] for p in photos])

    def auc(values: list[np.ndarray], sel: np.ndarray) -> float:
        wins = pairs = 0.0
        for v, d, s in zip(values, km, sel):
            if not s:
                continue
            ok = v > -1e8
            a, b = v[ok & (d < 1)], v[ok & (d >= 1) & (d < 25)]
            if len(a) and len(b):
                wins += (a[:, None] > b[None]).sum() + 0.5 * (a[:, None] == b[None]).sum(); pairs += len(a) * len(b)
        return wins / max(pairs, 1)

    for label, sel in (("top-1 < 25 km", top1 < 25), ("near-misses", top1 >= 1)):
        print(f"{label} (n={int(sel.sum())}): within-photo AUC, gallery photo < 1 km vs 1-25 km from the truth: {verifier} {auc(inl, sel):.3f}, SigLIP2 similarity {auc(sim, sel):.3f}")
        print(f"  a gallery photo < 1 km among the 100: {np.mean([(d < 1).any() for d, s in zip(km, sel) if s]):.1%}; the top-{verifier} photo is < 1 km: "
              f"{np.mean([d[np.argmax(v)] < 1 for d, v, s in zip(km, inl, sel) if s]):.1%}; the most similar photo is < 1 km: {np.mean([d[0] < 1 for d, s in zip(km, sel) if s]):.1%}")
    base = top1 < 1
    print(f"\nrule: move to the max-inlier gallery photo if it has >= T inliers, else keep the top-1 (n={len(photos)} photos with top-1 < 25 km; top-1 < 1 km {base.mean():.1%})")
    for t in thresholds:
        best =[int(np.argmax(v)) for v in inl]
        move = np.asarray([v[b] >= t for v, b in zip(inl, best)])
        chosen = np.where(move, [d[b] for d, b in zip(km, best)], top1)
        c = _bootstrap((chosen < 1).astype(float) - base)
        print(f"  T={t:4}: moves {move.mean():5.1%}, < 1 km {(chosen < 1).mean():.1%}, change {100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}] "
              f"(= {100 * c[0] * len(photos) / 1000:+.1f} over all 1,000 dev photos); fixes {int(((chosen < 1) & ~base).sum())}, breaks {int((~(chosen < 1) & base).sum())}; "
              f"moved-to photo < 1 km in {np.mean(chosen[move] < 1) if move.any() else 0:.0%} of moves")
    # consensus: a single gallery photo's GPS is noisy (the same landmark is often tagged km away), so strongly matched photos vote for the places within
    # VOTE_KM of them, weighted by inliers; the top-1 is a candidate too, and it is kept on ties or when nothing matches strongly
    print(f"\nrule: consensus of photos with >= T inliers (votes within {VOTE_KM} km, weighted by inliers); keep the top-1 unless another place gets more votes")
    for t in thresholds:
        chosen, moved = top1.copy(), np.zeros(len(photos), bool)
        for i, (p, v) in enumerate(zip(photos, inl)):
            strong = v >= t
            if not strong.any():
                continue
            locs = np.asarray([g["latlon"] for g in p["gallery"]])[strong]
            cands = np.concatenate((np.asarray(p["top"])[None], locs))
            votes = (_km(cands, locs) < VOTE_KM) @ weight(v[strong])
            best = int(np.argmax(votes))
            if votes[best] > votes[0]:
                chosen[i], moved[i] = _km(np.asarray(p["truth"])[None], cands[best][None])[0, 0], True
        c = _bootstrap((chosen < 1).astype(float) - base)
        print(f"  T={t:4}: moves {moved.mean():5.1%}, < 1 km {(chosen < 1).mean():.1%}, change {100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}] "
              f"(= {100 * c[0] * len(photos) / 1000:+.1f} over all 1,000 dev photos); fixes {int(((chosen < 1) & ~base).sum())}, breaks {int((~(chosen < 1) & base).sum())}")


VOTE_KM = 1.0


def _km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.radians(a), np.radians(b)
    xa = np.stack((np.cos(a[:, 0]) * np.cos(a[:, 1]), np.cos(a[:, 0]) * np.sin(a[:, 1]), np.sin(a[:, 0])), -1)
    xb = np.stack((np.cos(b[:, 0]) * np.cos(b[:, 1]), np.cos(b[:, 0]) * np.sin(b[:, 1]), np.sin(b[:, 0])), -1)
    return 6371.0088 * np.arccos(np.clip(xa @ xb.T, -1, 1))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("score", "report"))
    parser.add_argument("--verifier", default="inliers", help="inliers, or a comparator name scored on tag maptop100_dev")
    args = parser.parse_args(argv)
    score() if args.node == "score" else report(args.verifier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
