# MegaLoc as a proposer (lesson 95): can a same-place search find right places Pinpoint's areas miss? Measured on a deeper shortlist instead
# of the full index: per dev3k query, the K = 200 nearest index photos in raw SigLIP2 space (photographer excluded), re-ranked by MegaLoc.
# Reports the shortlist's ceiling, MegaLoc's top-1 vs SigLIP2's top-1 within it, and on photos where no Pinpoint area is right, how often
# MegaLoc's top match is right (and how precise it is by similarity), i.e. what a "deep same-place search" action could add as new areas.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.megaloc_deep shortlist --tag dev3k
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.megaloc_deep embed --tag dev3k
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.megaloc_deep report --tag dev3k

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .refine_headroom import _hav

OUT = Path("artifacts/agent")
CKPT_TAG = "setft_a"
K = 200


def _paths(tag: str):
    return OUT / f"megaloc_deep_{tag}_{CKPT_TAG}.npz", OUT / f"megaloc_deep_emb_{tag}_{CKPT_TAG}.npy"


def shortlist(tag: str, chunk: int = 500_000) -> None:
    import torch

    from ..tools.pinpoint import Pinpoint

    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
    tool = Pinpoint()
    rr = tool.reranker
    q_author = torch.as_tensor(tool.mp16_author(np.asarray([e["row"] for e in photos])), device="cuda")
    A = tool.index_author()
    views = np.load(OUT / f"crops_{tag}_{CKPT_TAG}.npy", mmap_mode="r")
    raw, _ = rr.encode_queries(np.asarray(views[:, 0], np.float32))
    q = raw.half()
    best_v = torch.full((len(q), K), float("-inf"), device="cuda", dtype=torch.float16)
    best_i = torch.zeros((len(q), K), dtype=torch.long, device="cuda")
    n = len(rr.latlon)
    with torch.inference_mode():
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            block = rr._dequantize(s, e)
            sc = q @ block.T
            sc.masked_fill_(torch.as_tensor(A[s:e], device="cuda")[None, :] == q_author[:, None], float("-inf"))
            v, i = torch.topk(sc, K, dim=1)
            mv, keep = torch.topk(torch.cat([best_v, v], 1), K, dim=1)
            best_i = torch.gather(torch.cat([best_i, i + s], 1), 1, keep)
            best_v = mv
            print(f"  index {e} / {n}", flush=True)
    rows = best_i.cpu().numpy()
    latlon = np.asarray(rr.latlon)[rows]
    meta, _ = _paths(tag)
    np.savez(meta, rows=rows, sims=best_v.float().cpu().numpy(), latlon=latlon, cache_rows=rr.cache_indices[rows])
    print(f"-> {meta}: {rows.shape}, {len(np.unique(rows))} distinct index photos")


def embed(tag: str, batch: int = 64, workers: int = 3) -> None:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from .megaloc_gate import SIZE, _model
    from .sft_data import MP16Images
    from .strategy_search import MP16_CSV, MP16_EMBED, _memmap_gallery

    meta, emb = _paths(tag)
    z = np.load(meta)
    uniq = np.unique(z["cache_rows"])
    row_index = _memmap_gallery(MP16_EMBED)["row_index"]
    with MP16_CSV.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        csv_ids = [r[0] for r in reader]
    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
    ids = [e["image_id"] for e in photos] + [csv_ids[row_index[c]] for c in uniq]
    mean, std = np.asarray([0.485, 0.456, 0.406], np.float32), np.asarray([0.229, 0.224, 0.225], np.float32)

    class Images(Dataset):
        images = None

        def __len__(self) -> int:
            return len(ids)

        def __getitem__(self, i: int):
            if self.images is None:
                self.images = MP16Images()
            try:
                im = Image.open(io.BytesIO(self.images.read(ids[i]))).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
                a = (np.asarray(im, np.float32) / 255.0 - mean) / std
                return i, torch.from_numpy(a.transpose(2, 0, 1).copy())
            except Exception:  # noqa: BLE001
                return i, torch.zeros(3, SIZE, SIZE)

    model = _model()
    out = np.lib.format.open_memmap(emb, mode="w+", dtype=np.float16, shape=(len(ids), 8448))
    with torch.inference_mode():
        for k, (idx, x) in enumerate(DataLoader(Images(), batch_size=batch, num_workers=workers)):
            out[idx.numpy()] = model(x.cuda()).half().cpu().numpy()
            if k % 500 == 0:
                print(f"  {k * batch} / {len(ids)}", flush=True)
    out.flush()
    np.save(OUT / f"megaloc_deep_uniq_{tag}_{CKPT_TAG}.npy", uniq)
    print(f"-> {emb}")


def report(tag: str) -> None:
    meta, emb = _paths(tag)
    z = np.load(meta)
    e = np.load(emb, mmap_mode="r")
    uniq = np.load(OUT / f"megaloc_deep_uniq_{tag}_{CKPT_TAG}.npy")
    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
    nq = len(photos)
    truth = np.asarray([p["truth"] for p in photos], np.float64)
    top1 = np.asarray([p["candidates"][0][1:] for p in photos], np.float64)
    q = np.asarray(e[:nq], np.float32)
    pos = np.searchsorted(uniq, z["cache_rows"])
    ml = np.zeros(z["rows"].shape, np.float32)
    for i in range(nq):
        ml[i] = np.asarray(e[nq + pos[i]], np.float32) @ q[i]
    km = _hav(truth[:, 0, None], truth[:, 1, None], z["latlon"][..., 0], z["latlon"][..., 1])
    base = _hav(truth[:, 0], truth[:, 1], top1[:, 0], top1[:, 1])
    areas = {o["image_id"]: o for o in json.loads((OUT / "tag_reader_dev3k_setft_a.json").read_text(encoding="utf-8"))}
    area_ll = [np.asarray([a["answer"] for a in areas[p["image_id"]]["areas"]], np.float64).reshape(-1, 2) for p in photos]
    area_right = np.asarray([any(a["km"] < 25 for a in areas[p["image_id"]]["areas"]) for p in photos])
    o = np.argsort(-ml, 1)
    ml_top = np.take_along_axis(km, o[:, :1], 1)[:, 0]
    ml_sim = np.take_along_axis(ml, o[:, :1], 1)[:, 0]
    ml_gap = ml_sim - np.take_along_axis(ml, o[:, 1:2], 1)[:, 0]
    top_ll = np.take_along_axis(z["latlon"], o[:, :1, None], 1)[:, 0]
    novel = np.asarray([len(a) == 0 or _hav(t[0], t[1], a[:, 0], a[:, 1]).min() > 25 for t, a in zip(top_ll, area_ll)])
    print(f"{tag}: {nq} queries, shortlist K = {K} (raw SigLIP2, photographer excluded)")
    for thr in (1, 25):
        print(f"  < {thr} km: retrained top-1 {np.mean(base < thr):.1%}; shortlist SigLIP2 top-1 {np.mean(km[:, 0] < thr):.1%}; "
              f"MegaLoc top-1 of the shortlist {np.mean(ml_top < thr):.1%}; any of the {K} {np.mean((km < thr).any(1)):.1%}; "
              f"any Pinpoint area {np.mean(area_right):.1%}" if thr == 25 else
              f"  < {thr} km: retrained top-1 {np.mean(base < thr):.1%}; MegaLoc top-1 {np.mean(ml_top < thr):.1%}; any of the {K} {np.mean((km < thr).any(1)):.1%}")
    miss = ~area_right
    print(f"\n  photos where no Pinpoint area is right: {miss.sum()} ({miss.mean():.1%}); shortlist has a right photo for {(km[miss] < 25).any(1).mean():.1%}; "
          f"MegaLoc top-1 right {np.mean(ml_top[miss] < 25):.1%} (SigLIP2 top-1 {np.mean(km[miss, 0] < 25):.1%})")
    print("  MegaLoc top match as a new area (> 25 km from every Pinpoint area), by its similarity:")
    print(f"   {'min sim':>8s} {'photos':>7s} {'share':>6s} {'right':>6s} {'fixes':>6s} {'of all':>7s}")
    for t in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7):
        sel = novel & (ml_sim >= t)
        right = ml_top[sel] < 25
        fixes = (sel & (ml_top < 25) & miss).sum()
        print(f"   {t:8.1f} {sel.sum():7d} {sel.mean():6.1%} {right.mean() if sel.any() else 0:6.1%} {fixes:6d} {fixes / nq:7.1%}")
    np.savez(OUT / f"megaloc_deep_report_{tag}_{CKPT_TAG}.npz", ml=ml, km=km, ml_sim=ml_sim, ml_gap=ml_gap, novel=novel, area_right=area_right)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("shortlist", "embed", "report"))
    parser.add_argument("--tag", default="dev3k")
    args = parser.parse_args(argv)
    {"shortlist": shortlist, "embed": embed, "report": report}[args.cmd](args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
