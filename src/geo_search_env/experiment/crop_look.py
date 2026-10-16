# Looking at parts of the photo (lesson 89, the user's go-ahead for the no-LLM line): the answer head sees Pinpoint's evidence from one
# embedding of the whole photo, squashed to 384 px, where a sign or a facade is a few pixels. Here every photo gets 10 SigLIP2-giant views:
# the full image and a 3 x 3 grid of half-size windows (stride a quarter), each embedded as Pinpoint embeds index photos. Each view is
# compared with each hypothesis's restricted-search photos (the zoom file's 12 rows, photographer excluded), giving per-hypothesis column
# groups for the answer head (hypothesis_agent --extra): "global" (full view) and "crop" (the 9 windows). Ceiling first (all views, all areas);
# an RL look policy only if the crops add >= 1 pt.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.crop_look embed --tag dev3k     (then --tag train)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.crop_look features --tag dev3k
#        PYTHONPATH=src PINPOINT_RERANKER_CKPT=... .venv/bin/python -m geo_search_env.experiment.hypothesis_agent --ckpt-tag setft_a --extra global crop

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
CKPT_TAG = "setft_a"
GRID = [(x, y) for y in (0.0, 0.25, 0.5) for x in (0.0, 0.25, 0.5)]  # window corners, as fractions; windows are half the width and height
TEMP = 0.02  # softmax temperature over a photo's hypotheses (cosine units)


def _views(data: bytes) -> list:
    from PIL import Image

    im = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = im.size
    return [im] + [im.crop((int(x * w), int(y * h), int(x * w) + w // 2, int(y * h) + h // 2)) for x, y in GRID]


def _crops_path(tag: str) -> Path:
    return OUT / f"crops_{tag}_{CKPT_TAG}.npy"


def embed(tag: str, batch: int = 6, workers: int = 3) -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset

    from .embed_cache import _embed, _load_model
    from .name_score import _read
    from .sft_data import MP16Images

    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
    model, processor = _load_model()

    class Views(Dataset):
        def __init__(self) -> None:
            self.images = None

        def __len__(self) -> int:
            return len(photos)

        def __getitem__(self, i: int):
            if self.images is None:  # one tar reader per worker
                self.images = MP16Images()
            try:
                return i, processor(images=_views(_read(photos[i], self.images)), return_tensors="pt")["pixel_values"]
            except Exception:  # noqa: BLE001
                return i, None

    def collate(items):
        ok = [(i, p) for i, p in items if p is not None]
        return [i for i, _ in items if _ is None], [i for i, _ in ok], torch.cat([p for _, p in ok]) if ok else None

    out = np.zeros((len(photos), 1 + len(GRID), 1536), np.float16)
    failed = []
    loader = DataLoader(Views(), batch_size=batch, num_workers=workers, collate_fn=collate)
    for k, (bad, idx, pixels) in enumerate(loader):
        failed += bad
        if pixels is not None:
            e = _embed(model, pixels).astype(np.float32).reshape(len(idx), 1 + len(GRID), -1)
            out[idx] = (e / np.linalg.norm(e, axis=-1, keepdims=True)).astype(np.float16)
        if k % 500 == 0:
            print(f"  {k * batch} / {len(photos)}", flush=True)
    np.save(_crops_path(tag), out)
    # check: the full view against the cached MP16 embedding of the same photo (same model, PIL vs torchvision resize)
    from .strategy_search import MP16_EMBED, _memmap_gallery

    g = _memmap_gallery(MP16_EMBED)["embeddings"]
    probe = np.arange(0, len(photos), max(1, len(photos) // 200))
    ref = np.asarray(g[np.asarray([photos[i]["row"] for i in probe])], np.float32)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    cos = (ref * out[probe, 0].astype(np.float32)).sum(1)
    print(f"{tag}: {len(photos)} photos, {len(failed)} failed -> {_crops_path(tag)}; full view vs cached embedding cosine median {np.median(cos):.4f}, "
          f"min {cos.min():.4f}")


def features(tag: str) -> None:
    from ..models.pinpoint_reranker import _discover_index
    from .strategy_search import MP16_EMBED, _memmap_gallery

    views = np.load(_crops_path(tag)).astype(np.float32)
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{CKPT_TAG}.npz")
    index_dir = _discover_index()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    cache = np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
    rows, photo = zm["rows"], zm["photo"]
    valid = rows >= 0
    need = np.unique(rows[valid])
    print(f"{tag}: {len(photo)} hypotheses, {len(need)} distinct restricted-search photos", flush=True)
    g = _memmap_gallery(MP16_EMBED)["embeddings"]
    mp = cache[need]
    order = np.argsort(mp)
    emb = np.empty((len(need), g.shape[1]), np.float32)
    for s in range(0, len(need), 200_000):  # sorted reads from the memmap
        o = order[s:s + 200_000]
        emb[o] = g[mp[o]]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    pos = np.searchsorted(need, np.where(valid, rows, need[0]))
    n_v = views.shape[1]
    best = np.full((len(photo), n_v), np.nan, np.float32)  # max cosine of each view over the hypothesis's photos
    top3 = np.full((len(photo), n_v), np.nan, np.float32)
    for j in range(len(photo)):
        v = valid[j]
        if not v.any():
            continue
        s = views[photo[j]] @ emb[pos[j][v]].T  # [views, k]
        s.sort(1)
        best[j], top3[j] = s[:, -1], s[:, -3:].mean(1)
    rel = np.zeros_like(best)  # margin over the photo's best other hypothesis, per view
    soft = np.zeros_like(best)
    arg = np.zeros_like(best)
    for p in np.unique(photo):
        J = np.flatnonzero(photo == p)
        b = np.nan_to_num(best[J], nan=-1.0)
        for k, j in enumerate(J):
            others = np.delete(b, k, 0)
            rel[j] = b[k] - (others.max(0) if len(others) else b[k])
        e = np.exp((b - b.max(0)) / TEMP)
        soft[J] = e / e.sum(0)
        arg[J] = (b == b.max(0)).astype(np.float32)
    b0 = np.nan_to_num(best, nan=0.0)
    t0 = np.nan_to_num(top3, nan=0.0)
    c = slice(1, n_v)
    cols = {
        "global": [b0[:, 0], t0[:, 0], rel[:, 0], soft[:, 0]],
        "crop": [b0[:, c].mean(1), b0[:, c].max(1), t0[:, c].mean(1), rel[:, c].mean(1), rel[:, c].max(1), rel[:, c].min(1), arg[:, c].mean(1),
                 soft[:, c].mean(1), b0[:, c].mean(1) - b0[:, 0]],
    }
    x = np.stack([v for group in cols.values() for v in group], 1).astype(np.float32)
    groups = np.asarray([k for k, group in cols.items() for _ in group])
    dst = OUT / f"cropcheck_{tag}_{CKPT_TAG}.npz"
    np.savez(dst, photo=photo, source=zm["source"], x=x, groups=groups)
    print(f"-> {dst}: {x.shape[1]} columns ({', '.join(f'{k} {len(v)}' for k, v in cols.items())})")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("embed", "features"))
    parser.add_argument("--tag", default="dev3k")
    args = parser.parse_args(argv)
    if args.cmd == "embed":
        embed(args.tag)
    else:
        features(args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
