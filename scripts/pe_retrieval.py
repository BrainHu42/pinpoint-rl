"""PE_PLAN.md step B1: does PE-G find right photos SigLIP2's searches miss? A pre-check on a random 10% of Pinpoint's index before a full
PE-G index (B2, ~4.1M photos). All outputs on EBS (artifacts/retrieval/b1/), so the embed survives the daily 02:00 UTC (7 PM Pacific) stop.

select: a random 10% of the released index rows (seed 0) -> gallery.npz (rows, MP16 ids, lat/lon, authors).
embed (.venvs/sft, timm on PYTHONPATH): PE-G 378 px pooled output, unit norm, fp16 -> pe_g378.npy; resumable: rerun the same command after
        a restart and it continues from embed.progress.
recall: dev3k queries (original protocol, the photographer's photos kept): per search, recall@K of a gallery photo within 1 / 25 / 200 km,
        K = 8 / 32 / 256; searches: raw SigLIP2 cosine, Pinpoint's contrastive image embedding (the reranker's main search), PE-G cosine, and
        unions at equal total K. Works on a partial embed (only embedded gallery photos count, reported).

  .venv/bin/python scripts/pe_retrieval.py select
  PYTHONPATH=.venvs/pylib_pe:src .venvs/sft/bin/python scripts/pe_retrieval.py embed      # from the main checkout, or with paths below
  .venv/bin/python scripts/pe_retrieval.py recall
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path

import numpy as np

OUT = Path("artifacts/retrieval/b1")
MODEL, SIZE = "vit_pe_core_gigantic_patch14_448.fb", 378
FRACTION = 0.10


def select(seed: int = 0) -> None:
    from geo_search_env.experiment.strategy_search import MP16_CSV, MP16_EMBED, _memmap_gallery
    from geo_search_env.models.pinpoint_reranker import _discover_index

    index_dir = _discover_index()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    cache = np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
    row_index = _memmap_gallery(MP16_EMBED)["row_index"]
    rows = np.sort(np.random.default_rng(seed).choice(len(cache), int(FRACTION * len(cache)), replace=False))
    csv_rows = np.asarray(row_index[cache[rows]])
    want = {int(r): i for i, r in enumerate(csv_rows)}
    ids, authors, latlon = [None] * len(rows), [None] * len(rows), np.zeros((len(rows), 2), np.float32)
    with MP16_CSV.open(encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        next(r)
        for k, rec in enumerate(r):
            i = want.get(k)
            if i is not None:
                ids[i], authors[i], latlon[i] = rec[0], rec[1], (float(rec[2]), float(rec[3]))
    assert all(i is not None for i in ids)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "gallery.npz", rows=rows, cache=cache[rows], ids=np.asarray(ids), authors=np.asarray(authors), latlon=latlon)
    print(f"-> {OUT / 'gallery.npz'}: {len(rows):,} of {len(cache):,} index photos", flush=True)


def embed(batch: int = 64, workers: int = 4) -> None:
    os.environ.setdefault("HF_HOME", str(Path("data/hf").resolve()))
    os.environ["HF_HUB_OFFLINE"] = "1"
    import timm
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from geo_search_env.experiment.sft_data import MP16Images

    ids = list(np.load(OUT / "gallery.npz")["ids"])
    n = len(ids)
    net = timm.create_model(MODEL, pretrained=True, img_size=SIZE).cuda().eval().to(torch.bfloat16)
    tf = timm.data.create_transform(**(timm.data.resolve_data_config(model=net) | {"input_size": (3, SIZE, SIZE)}))

    class Images(Dataset):
        images = None

        def __len__(self) -> int:
            return n

        def __getitem__(self, i: int):
            if self.images is None:
                self.images = MP16Images()
            try:
                return i, tf(Image.open(io.BytesIO(self.images.read(ids[i]))).convert("RGB")), True
            except Exception:  # noqa: BLE001
                return i, torch.zeros(3, SIZE, SIZE), False

    path, prog, okp = OUT / "pe_g378.npy", OUT / "embed.progress", OUT / "ok.npy"
    if prog.exists():
        out, ok, start = np.lib.format.open_memmap(path, mode="r+"), np.load(okp), int(prog.read_text())
    else:
        out, ok, start = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, 1280)), np.zeros(n, bool), 0
    print(f"b1 embed: {n:,} gallery photos at {SIZE} px; starting at {start:,}", flush=True)
    done = start
    with torch.inference_mode():
        for b, (idx, x, good) in enumerate(DataLoader(torch.utils.data.Subset(Images(), list(range(start, n))), batch_size=batch,
                                                      num_workers=workers)):
            i = idx.numpy()
            out[i] = torch.nn.functional.normalize(net(x.cuda().to(torch.bfloat16)).float(), dim=-1).cpu().numpy().astype(np.float16)
            ok[i] = good.numpy()
            done += len(i)
            if b % 100 == 99 or done == n:  # checkpoint every 6,400 photos (~2-3 min)
                out.flush()
                np.save(okp, ok)
                prog.write_text(str(done))
                print(f"  {done:,} / {n:,}", flush=True)
    print(f"-> {path}: {np.mean(~ok):.2%} unreadable", flush=True)


def recall(ks: tuple[int, ...] = (8, 32, 256), radii: tuple[float, ...] = (1.0, 25.0, 200.0)) -> None:
    import torch

    from geo_search_env.experiment.strategy_search import MP16_CSV
    from geo_search_env.models.pinpoint_reranker import PinpointReranker

    g = np.load(OUT / "gallery.npz")
    ok = np.load(OUT / "ok.npy") & (np.arange(len(g["rows"])) < int((OUT / "embed.progress").read_text()))
    use = np.flatnonzero(ok)
    photos = json.loads(Path("artifacts/photo_sets/dev3k_setft_own/photos.json").read_text(encoding="utf-8"))
    qids = [p["image_id"] for p in photos]
    truth = np.asarray([p["truth"] for p in photos], np.float64)
    need = set(qids)
    qauth = {}
    with MP16_CSV.open(encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        next(r)
        for rec in r:
            if rec[0] in need:
                qauth[rec[0]] = rec[1]
    qauth = np.asarray([qauth[i] for i in qids])
    # queries: raw SigLIP2 from the MP16 cache, PE-G 378 px from step 5b's embed (same model and size, dev3k rows last)
    em = Path("data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384")
    m = json.loads((em / "manifest.json").read_text(encoding="utf-8"))
    pos = {k: i for i, k in enumerate((em / m["files"]["image_ids"]).read_text(encoding="utf-8").splitlines())}
    sig = np.memmap(em / m["files"]["embeddings"], dtype=np.float16, mode="r", shape=tuple(m["shapes"]["embeddings"]))
    q_sig = np.asarray(sig[np.asarray([pos[i] for i in qids])], np.float32)
    probe = Path("artifacts/retrieval/probe")
    n_p = len(np.load(probe / "photos.npz")["ids"])
    q_pe = np.asarray(np.load(probe / "pooled.npy", mmap_mode="r")[n_p:n_p + len(qids)], np.float32)
    rr = PinpointReranker()
    raw_q, con_q = rr.encode_queries(q_sig)
    rows = g["rows"][use]
    dev = rr.device
    galleries = {"siglip2": rr._dequantize(0, rows=rows), "contrastive": torch.as_tensor(np.asarray(rr.image[rows]), device=dev),
                 "pe_g": torch.as_tensor(np.asarray(np.load(OUT / "pe_g378.npy", mmap_mode="r")[use]), device=dev)}
    queries = {"siglip2": raw_q.half(), "contrastive": con_q.half(),
               "pe_g": torch.nn.functional.normalize(torch.as_tensor(q_pe, device=dev), dim=-1).half()}
    glat = g["latlon"][use].astype(np.float64)
    same = qauth[:, None] == g["authors"][use][None, :]  # [Q, G] the photographer's own photos (original protocol keeps them)
    kmax = max(ks)

    def top(name: str, exclude_own: bool) -> np.ndarray:
        s = (queries[name] @ galleries[name].T).float()
        if exclude_own:
            s.masked_fill_(torch.as_tensor(same, device=dev), -9.0)
        return s.topk(kmax, dim=1).indices.cpu().numpy()

    def dist(idx: np.ndarray) -> np.ndarray:
        a, b = np.radians(truth[:, None, :]), np.radians(glat[idx])
        h = np.sin((b[..., 0] - a[..., 0]) / 2) ** 2 + np.cos(a[..., 0]) * np.cos(b[..., 0]) * np.sin((b[..., 1] - a[..., 1]) / 2) ** 2
        return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

    print(f"b1 recall: {len(use):,} of {len(ok):,} gallery photos embedded; dev3k {len(qids)} queries; % with a gallery photo within "
          f"{' / '.join(f'{r:g}' for r in radii)} km among the top K", flush=True)
    for exclude_own in (False, True):
        print(f"\n{'photographer excluded' if exclude_own else 'original protocol (photographer kept)'}:", flush=True)
        d = {name: dist(top(name, exclude_own)) for name in galleries}
        for k in ks:
            arms = {name: d[name][:, :k].min(1) for name in d}
            h = k // 2
            arms["contrastive + pe_g (K/2 each)"] = np.minimum(d["contrastive"][:, :h].min(1), d["pe_g"][:, :h].min(1))
            arms["siglip2 + pe_g (K/2 each)"] = np.minimum(d["siglip2"][:, :h].min(1), d["pe_g"][:, :h].min(1))
            arms["contrastive + siglip2 (K/2 each)"] = np.minimum(d["contrastive"][:, :h].min(1), d["siglip2"][:, :h].min(1))
            for name, v in arms.items():
                print(f"  K={k:3d} {name:34s} " + " / ".join(f"{100 * np.mean(v < r):5.1f}" for r in radii), flush=True)
            miss = arms["contrastive"] >= 25
            print(f"  K={k:3d} of the {miss.sum()} photos the contrastive top-K misses at 25 km, PE-G's top-K finds one for "
                  f"{100 * np.mean(arms['pe_g'][miss] < 25):.1f}% ({100 * np.mean(miss & (arms['pe_g'] < 25)):.1f} pts of all photos)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("node", choices=("select", "embed", "recall"))
    args = ap.parse_args()
    {"select": select, "embed": embed, "recall": recall}[args.node]()


if __name__ == "__main__":
    main()
