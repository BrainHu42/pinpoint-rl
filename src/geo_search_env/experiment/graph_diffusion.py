"""Evidence from the structure of the index, not only query-to-photo similarity: diffusion over a k-NN graph of the MP16 gallery.

knn:     every MP16 photo's K most SigLIP2-similar photos by other photographers (cosine; a photographer's own burst can't form a cluster)
         -> artifacts/diffusion/knn_k{K}_{idx,sim}.npy (cache-row order, int32 / float16). GPU, resumable per chunk.
diffuse: per photo of a photo set, personalized PageRank from its S nearest gallery photos (the photo itself excluded, as the original
         protocol does), T steps over the row-normalised graph (edge weights exp((cos - 1) / TAU)), truncated to the top M nodes per step;
         then per area token the mass within BOX_KM of the area centre (diffused and seeds-only), aligned with the hypothesis features
         -> artifacts/diffusion/diffusion_{tag}{_variant}.npz.

  PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.graph_diffusion knn [--k 32]
  PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.graph_diffusion diffuse --tag dev3k --ckpt-tag setft_own
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/diffusion")
EMB = Path("data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384")
MP16_CSV = Path("data/hf/datasets/MP16-Pro/metadata/MP16_Pro_filtered.csv")
BOX_KM = 25.0
CHUNK = 262_144  # rows per saved part of the graph


def _gallery(device: str = "cuda"):
    """The MP16 SigLIP2 gallery on the GPU, unit-normalised (stored vectors are not unit norm), with author ids and coordinates."""

    import torch

    man = json.loads((EMB / "manifest.json").read_text())
    n, dim = man["shapes"]["embeddings"]
    e = np.memmap(EMB / man["files"]["embeddings"], dtype=np.float16, mode="r", shape=(n, dim))
    g = torch.empty((n, dim), dtype=torch.float16, device=device)
    for s in range(0, n, 500_000):
        g[s:s + 500_000] = torch.nn.functional.normalize(torch.as_tensor(np.array(e[s:s + 500_000]), device=device).float(), dim=1).half()
    row_index = np.fromfile(EMB / man["files"]["row_index"], dtype=np.int64)
    auth, vocab = [], {}
    with MP16_CSV.open(newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        a = next(rd).index("AUTHOR")
        for row in rd:
            auth.append(vocab.setdefault(row[a], len(vocab)))
    author = np.asarray(auth, np.int64)[row_index]
    latlon = np.fromfile(EMB / man["files"]["latlon_deg"], dtype=np.float32).reshape(n, 2)
    return g, author, latlon


def knn(k: int = 32, batch: int = 512) -> None:
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    g, author, _ = _gallery()
    n = g.shape[0]
    auth = torch.as_tensor(author, device="cuda")
    for c0 in range(0, n, CHUNK):
        part = OUT / f"knn_k{k}_part{c0 // CHUNK:03d}.npz"
        if part.exists():
            continue
        c1 = min(n, c0 + CHUNK)
        idx = np.empty((c1 - c0, k), np.int32)
        sim = np.empty((c1 - c0, k), np.float16)
        with torch.inference_mode():
            for s in range(c0, c1, batch):
                q = torch.arange(s, min(c1, s + batch), device="cuda")
                x = g[q] @ g.T
                x[auth[q][:, None] == auth[None]] = -2  # other photographers only (drops the photo itself)
                v, i = x.topk(k, dim=1)
                idx[s - c0:s - c0 + len(q)] = i.cpu().numpy()
                sim[s - c0:s - c0 + len(q)] = v.cpu().numpy()
        np.savez(part, idx=idx, sim=sim)
        print(f"  rows {c0}-{c1} / {n} -> {part}", flush=True)
    parts = sorted(OUT.glob(f"knn_k{k}_part*.npz"))
    np.save(OUT / f"knn_k{k}_idx.npy", np.concatenate([np.load(p)["idx"] for p in parts]))
    np.save(OUT / f"knn_k{k}_sim.npy", np.concatenate([np.load(p)["sim"] for p in parts]))
    for p in parts:
        p.unlink()
    print(f"-> {OUT / f'knn_k{k}_idx.npy'}", flush=True)


def _hav(lat1, lon1, lat2, lon2):
    import torch

    lat1, lon1, lat2, lon2 = (torch.deg2rad(x) for x in (lat1, lon1, lat2, lon2))
    h = torch.sin((lat2 - lat1) / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * torch.asin(torch.sqrt(h.clamp(0, 1)))


def diffuse(tag: str, ckpt_tag: str, k: int = 32, seeds: int = 64, steps: int = 3, alpha: float = 0.7, tau: float = 0.05, top_m: int = 4096,
            variant: str = "", geo_km: float = 0.0) -> None:
    import torch

    from .area_model import _paths

    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{ckpt_tag}" / "photos.json").read_text(encoding="utf-8"))
    assert "row" in photos[0], "MP16 photo sets only (queries are gallery rows)"
    zm = np.load(_paths(tag, ckpt_tag)[1])
    photo, centre = zm["photo"], zm["center"].astype(np.float32)
    g, _, latlon = _gallery()
    ll = torch.as_tensor(latlon, device="cuda")
    nbr = torch.as_tensor(np.load(OUT / f"knn_k{k}_idx.npy"), device="cuda")
    w = torch.exp((torch.as_tensor(np.load(OUT / f"knn_k{k}_sim.npy"), device="cuda").float() - 1) / tau)
    w = (w / w.sum(1, keepdim=True)).half()
    by_photo = [np.flatnonzero(photo == p) for p in range(len(photos))]
    mass = {"diff": np.zeros(len(photo), np.float32), "seed": np.zeros(len(photo), np.float32)}
    pick = {"diff": np.zeros((len(photos), 2), np.float32), "seed": np.zeros((len(photos), 2), np.float32)}
    rows = np.asarray([e["row"] for e in photos])
    with torch.inference_mode():
        for b0 in range(0, len(photos), 256):
            q = torch.as_tensor(rows[b0:b0 + 256], device="cuda")
            x = g[q] @ g.T
            x[torch.arange(len(q), device="cuda"), q] = -2  # the photo itself (the original protocol keeps the photographer's other photos)
            sv, si = x.topk(seeds, dim=1)
            del x
            for j in range(len(q)):
                p = b0 + j
                own = q[j]
                f0_idx, f0 = si[j], torch.softmax((sv[j].float() - 1) / tau, 0)
                a_idx, a_val = f0_idx, f0
                for _ in range(steps):
                    to2 = nbr[a_idx].long()
                    ww = w[a_idx].float()
                    if geo_km:  # only edges between photos within geo_km of each other; the cut weight stays at the node
                        near = _hav(ll[a_idx, None, 0], ll[a_idx, None, 1], ll[to2, 0], ll[to2, 1]) < geo_km
                        stay = (ww * ~near).sum(1)
                        ww = ww * near
                        to2 = torch.cat([to2, a_idx[:, None]], 1)
                        ww = torch.cat([ww, stay[:, None]], 1)
                    to = to2.reshape(-1)
                    val = (a_val[:, None] * ww).reshape(-1) * alpha
                    cat_i = torch.cat([to, f0_idx])
                    cat_v = torch.cat([val, (1 - alpha) * f0])
                    u, inv = torch.unique(cat_i, return_inverse=True)
                    v = torch.zeros(len(u), device="cuda").scatter_add_(0, inv, cat_v)
                    v[u == own] = 0
                    if len(u) > top_m:
                        v, keep = v.topk(top_m)
                        u = u[keep]
                    a_idx, a_val = u, v / v.sum()
                for name, (ni, nv) in (("diff", (a_idx, a_val)), ("seed", (f0_idx, f0))):
                    nl = ll[ni]
                    h = by_photo[p]
                    if len(h):
                        c = torch.as_tensor(centre[h], device="cuda")
                        d = _hav(c[:, None, 0], c[:, None, 1], nl[None, :, 0], nl[None, :, 1])
                        mass[name][h] = ((d < BOX_KM).float() * nv[None]).sum(1).cpu().numpy()
                    top = nv.topk(min(512, len(nv))).indices
                    dd = _hav(nl[top, None, 0], nl[top, None, 1], nl[None, top, 0], nl[None, top, 1])
                    pick[name][p] = nl[top[((dd < BOX_KM).float() * nv[top][None]).sum(1).argmax()]].cpu().numpy()
            if b0 % 5120 == 0:
                print(f"  {b0} / {len(photos)}", flush=True)
    dst = OUT / f"diffusion_{tag}_{ckpt_tag}{'_' + variant if variant else ''}.npz"
    np.savez(dst, photo=photo, source=zm["source"], mass_diff=mass["diff"], mass_seed=mass["seed"], pick_diff=pick["diff"], pick_seed=pick["seed"],
             params=np.asarray([k, seeds, steps, alpha, tau, top_m, geo_km]))
    print(f"-> {dst}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("knn", "diffuse"))
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--ckpt-tag", default="setft_own")
    parser.add_argument("--alpha", type=float, default=0.7, help="diffuse: share of mass passed along the graph per step")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--variant", default="", help="diffuse: output suffix")
    parser.add_argument("--geo-km", type=float, default=0.0, help="diffuse: only edges between photos within this distance (0: all)")
    args = parser.parse_args(argv)
    if args.cmd == "knn":
        knn(args.k)
    else:
        diffuse(args.tag, args.ckpt_tag, args.k, steps=args.steps, alpha=args.alpha, variant=args.variant, geo_km=args.geo_km)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
