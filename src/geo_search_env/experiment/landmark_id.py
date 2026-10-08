# Landmark / object identification as a tool (AGENT_DIRECTION.md): the VLM boxes the most distinctive object in the photo (observe.py objects);
# the crop is searched against all MP16 gallery photos with SigLIP2 (same-photographer rows excluded, so the query itself never matches);
# the top hits are verified by keypoint matching (SuperPoint + LightGlue); a strong match's location replaces Pinpoint's top-1.
# Usage: PYTHONPATH=src .venvs/sft/bin/python -m geo_search_env.experiment.landmark_id search --tag dev3k   (GPU: crop embeddings, top-k, keypoints)
#        PYTHONPATH=src .venvs/sft/bin/python -m geo_search_env.experiment.landmark_id verify --tag dev3k   (GPU: RANSAC inliers for all top-k hits)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.landmark_id report --tag dev3k

"""search: per photo with an object box -> artifacts/agent/lmid_{tag}.npz: photo index, the gallery rows of the TOP_K nearest crops (SigLIP2 cosine),
        their similarities, and LightGlue matches (confidence > MIN_CONF) between the crop and each of the first VERIFY gallery photos.
report: the switch rule on the dev set: if the best verified gallery photo has >= T matches and lies >= 25 km from Pinpoint's top-1, answer with the
        nearest pool candidate within 25 km of it, else its own location; change in top-1 < 1 / 25 km vs Pinpoint, fixes and breaks, per threshold T."""

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
GALLERY = Path("data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384")
TOP_K, VERIFY, MIN_CONF, LONG_SIDE = 20, 5, 0.2, 1024


def _box(box: list[float], w: int, h: int, coords: str) -> tuple[int, int, int, int] | None:
    """Pixel box from the VLM's box (Qwen3 grounding: 0-1000 relative coordinates; 'pixels' for absolute), padded 10%."""

    x1, y1, x2, y2 = box
    if coords == "norm1000":
        x1, x2, y1, y2 = x1 * w / 1000, x2 * w / 1000, y1 * h / 1000, y2 * h / 1000
    x1, x2, y1, y2 = sorted((x1, x2)) + sorted((y1, y2))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    px, py = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
    return int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(w, x2 + px)), int(min(h, y2 + py))


def search(tag: str, clues: str, coords: str, top_k: int = TOP_K, verify_hits: bool = True) -> None:
    import torch
    import torch.nn.functional as F
    from lightglue import LightGlue, SuperPoint
    from PIL import Image

    from .embed_cache import _embed, _load_model
    from .strategy_search import _stream_topk, load_world

    photos = load_photos(tag)
    objects = json.loads((OUT / f"objects_{tag}_{clues}.json").read_text(encoding="utf-8"))
    images = MP16Images()
    model, processor = _load_model()
    crops, keep = [], []
    for p, e in enumerate(photos):
        o = objects.get(e["image_id"]) or {}
        if not (o.get("object") and o.get("bbox")):
            continue
        im = Image.open(io.BytesIO(images.read(e["image_id"]))).convert("RGB")
        box = _box(o["bbox"], *im.size, coords)
        if box is None:
            continue
        crops.append(im.crop(box))
        keep.append(p)
    print(f"{tag}: {len(keep)} photos with a usable object box", flush=True)
    feats = []
    for start in range(0, len(crops), 32):
        pixels = processor(images=crops[start:start + 32], return_tensors="pt")["pixel_values"]
        feats.append(_embed(model, pixels))
    del model
    torch.cuda.empty_cache()
    queries = F.normalize(torch.as_tensor(np.concatenate(feats), dtype=torch.float32), dim=-1).half().cuda()
    world = load_world()
    q_author = np.asarray([int(world.mp16["author"][photos[p]["row"]]) for p in keep], dtype=np.int64)
    rows, sims = _stream_topk(world.mp16["embeddings"], queries, q_author, world.mp16["author"], top_k, normalize=True)
    if not verify_hits:
        np.savez(OUT / f"lmid_{tag}_k{top_k}.npz", photo=np.asarray(keep), rows=rows, sims=sims)
        print(f"{tag}: crop search (top {top_k}, no verification) for {len(keep)} photos -> {OUT / f'lmid_{tag}_k{top_k}.npz'}", flush=True)
        return
    ids = (GALLERY / "image_ids.txt").read_text(encoding="utf-8").split("\n")
    extractor = SuperPoint(max_num_keypoints=1024).eval().cuda()
    matcher = LightGlue(features="superpoint").eval().cuda()

    def feature(im) -> dict:
        im = im.copy()
        im.thumbnail((LONG_SIDE, LONG_SIDE))
        t = torch.as_tensor(np.asarray(im).copy(), dtype=torch.float32).permute(2, 0, 1).cuda() / 255
        with torch.inference_mode():
            return extractor.extract(t)

    matches = np.zeros((len(keep), VERIFY), np.int32)
    for i, crop in enumerate(crops):
        fq = feature(crop)
        for j in range(VERIFY):
            data = images.read(ids[int(rows[i, j])])
            if data is None:
                continue
            fg = feature(Image.open(io.BytesIO(data)).convert("RGB"))
            with torch.inference_mode():
                out = matcher({"image0": fq, "image1": fg})
            matches[i, j] = int((out["matching_scores0"][0] > MIN_CONF).sum())
        if i % 100 == 0:
            print(f"  verified {i} / {len(keep)}", flush=True)
    np.savez(OUT / f"lmid_{tag}.npz", photo=np.asarray(keep), rows=rows, sims=sims, matches=matches)
    print(f"{tag}: crop search + verification for {len(keep)} photos -> {OUT / f'lmid_{tag}.npz'}", flush=True)


def verify(tag: str, clues: str, coords: str) -> None:
    """Geometric verification of all TOP_K crop hits: LightGlue matches filtered by a RANSAC fundamental matrix (MAGSAC); the inlier count per hit
    -> artifacts/agent/lmid_inliers_{tag}.npz (aligned with lmid_{tag}.npz)."""

    import cv2
    import torch
    from lightglue import LightGlue, SuperPoint
    from PIL import Image

    photos = load_photos(tag)
    objects = json.loads((OUT / f"objects_{tag}_{clues}.json").read_text(encoding="utf-8"))
    z = np.load(OUT / f"lmid_{tag}.npz")
    ids = (GALLERY / "image_ids.txt").read_text(encoding="utf-8").split("\n")
    images = MP16Images()
    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
    matcher = LightGlue(features="superpoint").eval().cuda()

    def feature(im) -> dict:
        im = im.copy()
        im.thumbnail((LONG_SIDE, LONG_SIDE))
        t = torch.as_tensor(np.asarray(im).copy(), dtype=torch.float32).permute(2, 0, 1).cuda() / 255
        with torch.inference_mode():
            return extractor.extract(t)

    inliers = np.zeros(z["rows"].shape, np.int32)
    for i, p in enumerate(z["photo"]):
        e = photos[p]
        im = Image.open(io.BytesIO(images.read(e["image_id"]))).convert("RGB")
        fq = feature(im.crop(_box(objects[e["image_id"]]["bbox"], *im.size, coords)))
        for j, row in enumerate(z["rows"][i]):
            data = images.read(ids[int(row)])
            if data is None:
                continue
            fg = feature(Image.open(io.BytesIO(data)).convert("RGB"))
            with torch.inference_mode():
                out = matcher({"image0": fq, "image1": fg})
            m = out["matches"][0].cpu().numpy()
            if len(m) < 8:
                continue
            a = fq["keypoints"][0].cpu().numpy()[m[:, 0]]
            b = fg["keypoints"][0].cpu().numpy()[m[:, 1]]
            try:
                _, mask = cv2.findFundamentalMat(a, b, cv2.USAC_MAGSAC, 2.0, 0.999, 10_000)
            except cv2.error:  # degenerate match sets raise in OpenCV 5's USAC; fall back to classic RANSAC
                try:
                    _, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 2.0, 0.999)
                except cv2.error:
                    mask = None
            inliers[i, j] = int(mask.sum()) if mask is not None else 0
        if i % 100 == 0:
            print(f"  verified {i} / {len(z['photo'])}", flush=True)
    np.savez(OUT / f"lmid_inliers_{tag}.npz", inliers=inliers)
    print(f"{tag}: RANSAC inliers for {len(z['photo'])} photos x {z['rows'].shape[1]} hits -> {OUT / f'lmid_inliers_{tag}.npz'}", flush=True)


def rescore(tag: str, clues: str, coords: str, top_k: int = 100, size: int = 336) -> None:
    """Re-score each crop's top-k SigLIP2 hits with DINOv2-large (instance-retrieval features): CLS token and GeM-pooled patch tokens, both
    L2-normalised, crop vs gallery photo at size x size -> artifacts/agent/lmid_dino_{tag}.npz (cls, gem: [photos, top_k] cosine)."""

    from concurrent.futures import ThreadPoolExecutor

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoModel

    photos = load_photos(tag)
    objects = json.loads((OUT / f"objects_{tag}_{clues}.json").read_text(encoding="utf-8"))
    z = np.load(OUT / f"lmid_{tag}_k{top_k}.npz")
    ids = (GALLERY / "image_ids.txt").read_text(encoding="utf-8").split("\n")
    images = MP16Images()
    model = AutoModel.from_pretrained("facebook/dinov2-large", dtype=torch.float16).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    def tensor(im) -> np.ndarray:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BICUBIC), dtype=np.uint8)

    def embed(batch: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        x = torch.as_tensor(np.stack(batch), device="cuda").permute(0, 3, 1, 2).float() / 255
        x = ((x - mean) / std).half()
        with torch.inference_mode():
            h = model(pixel_values=x).last_hidden_state.float()
        cls = F.normalize(h[:, 0], dim=-1)
        gem = F.normalize(h[:, 1:].clamp(min=1e-6).pow(3).mean(1).pow(1 / 3), dim=-1)
        return cls.cpu().numpy(), gem.cpu().numpy()

    def gallery(row: int):
        data = images.read(ids[int(row)])
        return None if data is None else tensor(Image.open(io.BytesIO(data)))

    cls_out = np.zeros(z["rows"].shape, np.float32)
    gem_out = np.zeros(z["rows"].shape, np.float32)
    with ThreadPoolExecutor(16) as pool:
        for i, p in enumerate(z["photo"]):
            e = photos[p]
            im = Image.open(io.BytesIO(images.read(e["image_id"]))).convert("RGB")
            qc, qg = embed([tensor(im.crop(_box(objects[e["image_id"]]["bbox"], *im.size, coords)))])
            hits = list(pool.map(gallery, z["rows"][i]))
            ok = [j for j, h in enumerate(hits) if h is not None]
            gc, gg = embed([hits[j] for j in ok])
            cls_out[i, ok] = gc @ qc[0]
            gem_out[i, ok] = gg @ qg[0]
            if i % 100 == 0:
                print(f"  re-scored {i} / {len(z['photo'])}", flush=True)
    np.savez(OUT / f"lmid_dino_{tag}.npz", cls=cls_out, gem=gem_out)
    print(f"{tag}: DINOv2 re-scores for {len(z['photo'])} photos x {top_k} hits -> {OUT / f'lmid_dino_{tag}.npz'}", flush=True)


def report_rescore(tag: str, top_k: int = 100) -> None:
    """On Pinpoint's mistakes: is a right hit (< 25 km) among the top-k at all, and does each score put a right hit first? Then the switch rule per
    score: the top hit replaces Pinpoint's top-1 (snapped to a pool candidate within 25 km) if its score >= threshold and >= V of the score's
    top-5 hits lie within 25 km of it."""

    from .strategy_search import _bootstrap, _haversine_km

    photos = load_photos(tag)
    z = np.load(OUT / f"lmid_{tag}_k{top_k}.npz")
    d = np.load(OUT / f"lmid_dino_{tag}.npz")
    latlon = np.fromfile(GALLERY / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2).astype(np.float64)
    base = np.asarray([_haversine_km(*e["truth"], np.asarray(e["pool"][:1]))[0] for e in photos])
    scores = {"siglip": z["sims"], "dino cls": d["cls"], "dino gem": d["gem"]}
    right = np.stack([_haversine_km(*photos[p]["truth"], latlon[z["rows"][i]]) < 25 for i, p in enumerate(z["photo"])])
    mist = base[z["photo"]] >= 25
    print(f"{tag}: Pinpoint mistakes searched {mist.sum()}; a right hit among the top {top_k}: {right[mist].any(1).mean():.1%}")
    for name, sc in scores.items():
        order = np.argsort(-sc, axis=1)
        top1_right = right[np.arange(len(order)), order[:, 0]]
        print(f"  {name:9s}: top-1 hit right on mistakes {top1_right[mist].mean():.1%}; on photos Pinpoint gets right {top1_right[~mist].mean():.1%}")
    for name, sc in scores.items():
        order = np.argsort(-sc, axis=1)
        best = sc[np.arange(len(order)), order[:, 0]]
        for q in (0.5, 0.8, 0.9, 0.95, 0.99):
            T = float(np.quantile(best, q))
            for V in (1, 2, 3):
                chosen, switched = base.copy(), 0
                for i, p in enumerate(z["photo"]):
                    if best[i] < T:
                        continue
                    target = latlon[z["rows"][i, order[i, 0]]]
                    if (_haversine_km(*target, latlon[z["rows"][i, order[i, :5]]]) < 25).sum() < V:
                        continue
                    e = photos[p]
                    if _haversine_km(*target, np.asarray(e["pool"][:1]))[0] < 25:
                        continue
                    d_pool = _haversine_km(*target, np.asarray(e["pool"]))
                    answer = np.asarray(e["pool"][int(d_pool.argmin())]) if d_pool.min() < 25 else target
                    chosen[p] = _haversine_km(*e["truth"], answer[None])[0]
                    switched += 1
                c = _bootstrap((chosen < 25) * 1.0 - (base < 25))
                fixes, breaks = ((chosen < 25) & (base >= 25)).sum(), ((chosen >= 25) & (base < 25)).sum()
                print(f"  {name:9s} score >= q{q:.2f} ({T:.3f}), top-5 agreeing >= {V}: switched {switched:4d}; < 25 km {100 * c[0]:+.1f} "
                      f"[{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]; fixes {fixes}, breaks {breaks}")


def report(tag: str) -> None:
    from .strategy_search import _bootstrap, _haversine_km

    photos = load_photos(tag)
    z = np.load(OUT / f"lmid_{tag}.npz")
    latlon = np.fromfile(GALLERY / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2).astype(np.float64)
    base = np.asarray([_haversine_km(*e["truth"], np.asarray(e["pool"][:1]))[0] for e in photos])
    n = len(photos)
    print(f"{tag}: n={n}, Pinpoint top-1 < 1 / 25 km {(base < 1).mean():.1%} / {(base < 25).mean():.1%}; photos searched {len(z['photo'])}")
    best_j = z["matches"].argmax(1)
    best_m = z["matches"].max(1)
    hit_ll = latlon[z["rows"][np.arange(len(best_j)), best_j]]
    top1_hit = latlon[z["rows"][:, 0]]
    print(f"  crop search alone (top-1 gallery hit) < 25 km on searched photos: {np.mean([_haversine_km(*photos[p]['truth'], top1_hit[i:i + 1])[0] < 25 for i, p in enumerate(z['photo'])]):.1%}"
          f"  (Pinpoint on the same photos {np.mean(base[z['photo']] < 25):.1%})")
    for T in (0, 25, 50, 100, 150, 200, 300):
        chosen = base.copy()
        switched = 0
        for i, p in enumerate(z["photo"]):
            if best_m[i] < T:
                continue
            e = photos[p]
            target = hit_ll[i]
            if _haversine_km(*target, np.asarray(e["pool"][:1]))[0] < 25:
                continue  # agrees with Pinpoint's top-1
            d_pool = _haversine_km(*target, np.asarray(e["pool"]))
            answer = np.asarray(e["pool"][int(d_pool.argmin())]) if d_pool.min() < 25 else target
            chosen[p] = _haversine_km(*e["truth"], answer[None])[0]
            switched += 1
        parts = []
        for km in (1, 25):
            c = _bootstrap((chosen < km) * 1.0 - (base < km))
            parts.append(f"< {km} km {100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]")
        fixes, breaks = ((chosen < 25) & (base >= 25)).sum(), ((chosen >= 25) & (base < 25)).sum()
        print(f"  matches >= {T:3d}: switched {switched:4d}; " + "; ".join(parts) + f"; 25 km fixes {fixes}, breaks {breaks}")


def report_inliers(tag: str) -> None:
    """The switch rule with RANSAC inliers: the hit with the most inliers is accepted if it has >= T inliers and >= V verified hits (>= T inliers)
    lie within 25 km of it; it replaces Pinpoint's top-1 (snapped to a pool candidate within 25 km) when it disagrees. Also: do right hits get more
    inliers than wrong ones on Pinpoint's mistakes?"""

    from .strategy_search import _bootstrap, _haversine_km

    photos = load_photos(tag)
    z = np.load(OUT / f"lmid_{tag}.npz")
    inl = np.load(OUT / f"lmid_inliers_{tag}.npz")["inliers"]
    latlon = np.fromfile(GALLERY / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2).astype(np.float64)
    base = np.asarray([_haversine_km(*e["truth"], np.asarray(e["pool"][:1]))[0] for e in photos])
    right_best, wrong_best = [], []
    for i, p in enumerate(z["photo"]):
        e = photos[p]
        if base[p] < 25:
            continue
        ok = _haversine_km(*e["truth"], latlon[z["rows"][i]]) < 25
        if ok.any() and (~ok).any():
            right_best.append(inl[i][ok].max()); wrong_best.append(inl[i][~ok].max())
    r, w = np.asarray(right_best), np.asarray(wrong_best)
    print(f"{tag}: on Pinpoint's mistakes with both right and wrong top-{inl.shape[1]} hits ({len(r)}): best right hit inliers median {np.median(r):.0f}, "
          f"best wrong {np.median(w):.0f}; right > wrong on {np.mean(r > w):.0%}")
    for T in (15, 25, 40, 60, 100):
        for V in (1, 2, 3):
            chosen, switched = base.copy(), 0
            for i, p in enumerate(z["photo"]):
                j = int(inl[i].argmax())
                if inl[i, j] < T:
                    continue
                target = latlon[z["rows"][i, j]]
                verified = latlon[z["rows"][i][inl[i] >= T]]
                if (_haversine_km(*target, verified) < 25).sum() < V:
                    continue
                e = photos[p]
                if _haversine_km(*target, np.asarray(e["pool"][:1]))[0] < 25:
                    continue
                d_pool = _haversine_km(*target, np.asarray(e["pool"]))
                answer = np.asarray(e["pool"][int(d_pool.argmin())]) if d_pool.min() < 25 else target
                chosen[p] = _haversine_km(*e["truth"], answer[None])[0]
                switched += 1
            c = _bootstrap((chosen < 25) * 1.0 - (base < 25))
            fixes, breaks = ((chosen < 25) & (base >= 25)).sum(), ((chosen >= 25) & (base < 25)).sum()
            print(f"  inliers >= {T:3d}, agreeing hits >= {V}: switched {switched:4d}; < 25 km {100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]; "
                  f"fixes {fixes}, breaks {breaks}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("search", "verify", "rescore", "report", "report-inliers", "report-rescore"))
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--no-verify", action="store_true", help="search: keep the top-k hits without LightGlue")
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--clues", default="9b")
    parser.add_argument("--coords", choices=("norm1000", "pixels"), default="norm1000")
    args = parser.parse_args(argv)
    if args.node == "search":
        search(args.tag, args.clues, args.coords, args.top_k, not args.no_verify)
    elif args.node == "verify":
        verify(args.tag, args.clues, args.coords)
    elif args.node == "rescore":
        rescore(args.tag, args.clues, args.coords, args.top_k)
    elif args.node == "report-rescore":
        report_rescore(args.tag, args.top_k)
    elif args.node == "report-inliers":
        report_inliers(args.tag)
    else:
        report(args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
