"""VALUE_PLAN regime idea R4 (user, 2026-10-24): a stronger region expert from more data. Lesson V12: the region probe (track R's pe_probe,
PE-G 378 px pooled output -> 2,048 k-means cells, ~98k MP16 photos) is the judge that picks right regions where retrieval fails, and its
region accuracy is now the limit. Track R has PE-G embeddings of far more index photos: B1 (a random 10% of Pinpoint's index, 408k) and B2
(the full index, ~4.1M, in progress). Same probe recipe (pe_probe._fit_mlp, imported read-only), more photos and finer cells.

Leak guard (track R, 2026-10-24): B2 embedded the dev3k / commons26 dev pools' and zoom searches' index photos first; they cluster around
the eval photos' true places, so every row in b2/first_{dev3k,commons26dev}_{pool,zoom}.npy is dropped, with dev3k photos and every photo
by a dev3k photographer. Only rows marked done & ok are read.

train --data probe|b1|b2 --cells K [--max N]: fit -> artifacts/value/region_expert_{data}_{K}[_{N}].npz (dev3k log-probs, cell centres)
        and print dev3k top-cell accuracy at 25 / 200 / 750 / 2500 km.
Then `hedge_search.py regime --probe <file>` judges V9's boxes with it (region / place on area-oracle misses, net of nearby-truth luck).

  PYTHONPATH=src .venv/bin/python scripts/region_expert.py train --data b2 --cells 8192
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

RET = Path("artifacts/retrieval")
OUT = Path("artifacts/value")
CSV = Path("data/hf/datasets/MP16-Pro/metadata/MP16_Pro_filtered.csv")
EVAL_FIRST = ("first_dev3k_pool", "first_dev3k_zoom", "first_commons26dev_pool", "first_commons26dev_zoom")


def _pe_probe():
    spec = importlib.util.spec_from_file_location("pe_probe", Path(__file__).resolve().parents[2] / "retrieval-track" / "scripts" / "pe_probe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _xyz(ll: np.ndarray) -> np.ndarray:
    la, lo = np.radians(ll[:, 0]), np.radians(ll[:, 1])
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)


def _km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 6371.0 * np.arccos(np.clip((_xyz(a) * _xyz(b)).sum(-1), -1, 1))


def _cells(latlon: np.ndarray, k: int, seed: int = 0, iters: int = 30, chunk: int = 200_000):
    """Spherical k-means on unit xyz (GPU, assignments in chunks so millions of photos x 8k cells fit): centres [k, 2] and each photo's cell."""
    import torch

    x = torch.tensor(_xyz(latlon), device="cuda", dtype=torch.float32)
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k].cuda()].clone()

    def assign(c):
        return torch.cat([(x[s:s + chunk] @ c.T).argmax(1) for s in range(0, len(x), chunk)])

    for _ in range(iters):
        a = assign(c)
        s = torch.zeros_like(c).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=k)
        c = torch.where(cnt[:, None] > 0, torch.nn.functional.normalize(s, dim=1), c)
    a = assign(c).cpu().numpy()
    c = c.cpu().numpy()
    return np.stack([np.degrees(np.arcsin(np.clip(c[:, 2], -1, 1))), np.degrees(np.arctan2(c[:, 1], c[:, 0]))], 1), a


def _dev_guard() -> tuple[set, set]:
    """dev3k photo ids and their photographers (from MP16's metadata)."""
    dev = set(json.loads(Path("artifacts/agent/splits.json").read_text(encoding="utf-8"))["dev"])
    authors = set()
    with CSV.open(encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row[0] in dev:
                authors.add(row[1])
    return dev, authors


def _siglip():
    """MP16's SigLIP2 cache (the retriever's model), [4.1M, 1536] fp16, indexed by MP16 row (B1 / B2 `cache`, photo sets' `row`)."""
    root = Path("data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384")
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return np.memmap(root / m["files"]["embeddings"], dtype=np.float16, mode="r", shape=tuple(m["shapes"]["embeddings"]))


def _load_rows(e, rows: np.ndarray) -> np.ndarray:
    o = np.argsort(rows)  # sorted reads from the memmap, then back to the given order
    x = np.concatenate([np.asarray(e[rows[o][s:s + 200_000]]) for s in range(0, len(rows), 200_000)])
    out = np.empty_like(x)
    out[o] = x
    return out


def _training_set(data: str, max_n: int, seed: int = 0, backbone: str = "pe"):
    """(embeddings fp16 [N, D], latlon [N, 2]) of the training photos, leak-guarded; backbone pe (PE-G 378 px pooled), siglip (SigLIP2
    from MP16's cache) or concat (both, each unit-normalised)."""
    if data == "probe":  # track R's probe set (already leak-guarded at select): the baseline, refit with this script's settings
        p = np.load(RET / "probe" / "photos.npz")
        e = np.load(RET / "probe" / "pooled.npy", mmap_mode="r")
        return np.asarray(e[: len(p["ids"])]), p["latlon"].astype(np.float64)
    d = RET / data
    g = np.load(d / "gallery.npz")
    ok = np.load(d / "ok.npy")
    use = ok & (np.load(d / "done.npy") if (d / "done.npy").exists() else np.ones(len(ok), bool))
    drop = np.zeros(len(use), bool)
    if data == "b2":
        for f in EVAL_FIRST:
            drop[np.load(d / f"{f}.npy")] = True
    else:  # B1 rows are B2's rows reused: map B2's eval-first positions to B1 by photo id
        b2 = np.load(RET / "b2" / "gallery.npz")["ids"]
        ev = np.concatenate([np.load(RET / "b2" / f"{f}.npy") for f in EVAL_FIRST])
        drop |= np.isin(g["ids"], b2[ev])
    dev, dev_auth = _dev_guard()
    drop |= np.isin(g["ids"], list(dev)) | np.isin(g["authors"], list(dev_auth))
    rows = np.flatnonzero(use & ~drop)
    print(f"{data}: {use.sum():,} embedded rows, {(use & drop).sum():,} dropped by the leak guard, {len(rows):,} kept", flush=True)
    if max_n and len(rows) > max_n:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_n, replace=False))
    t = time.perf_counter()
    parts = []
    if backbone in ("pe", "concat"):
        parts.append(_load_rows(np.load(d / "pe_g378.npy", mmap_mode="r"), rows))
    if backbone in ("siglip", "concat"):
        parts.append(_load_rows(_siglip(), g["cache"][rows].astype(np.int64)))
    X = _join(parts)
    print(f"  loaded {len(rows):,} {backbone} embeddings ({X.shape[1]}-d) in {time.perf_counter() - t:.0f} s", flush=True)
    return X, g["latlon"][rows].astype(np.float64)


def _join(parts: list[np.ndarray]) -> np.ndarray:
    if len(parts) == 1:
        return parts[0]
    out = np.empty((len(parts[0]), sum(p.shape[1] for p in parts)), np.float16)
    c = 0
    for p in parts:  # unit-normalise each part in chunks (no full float32 copy)
        for s in range(0, len(p), 100_000):
            x = p[s:s + 100_000].astype(np.float32)
            out[s:s + 100_000, c:c + p.shape[1]] = x / np.linalg.norm(x, axis=1, keepdims=True)
        c += p.shape[1]
    return out


def train(data: str, k: int, max_n: int = 0, epochs: int = 0, batch: int = 0, seed: int = 0, backbone: str = "pe") -> None:
    import torch

    pp = _pe_probe()
    pp.CELLS = k
    X, ll = _training_set(data, max_n, seed, backbone)
    parts = []
    if backbone in ("pe", "concat"):  # dev3k queries, appended last at the probe's embed (same model, 378 px)
        parts.append(np.asarray(np.load(RET / "probe" / "pooled.npy", mmap_mode="r")[-3000:]))
    if backbone in ("siglip", "concat"):
        photos = json.loads((Path("artifacts/photo_sets") / "dev3k_setft_own" / "photos.json").read_text(encoding="utf-8"))
        parts.append(_load_rows(_siglip(), np.asarray([int(e["row"]) for e in photos])))
    q = _join(parts)
    _, truth = pp._dev()
    truth = truth.astype(np.float64)
    cent, cell = _cells(ll, k, seed)
    n = len(X)
    Xt = torch.tensor(np.concatenate([X, q]), device="cuda", dtype=torch.float16)
    y = torch.tensor(np.concatenate([cell, np.zeros(3000, np.int64)]), device="cuda")
    epochs = epochs or max(3, min(15, round(15 * 100_000 / n)))  # ~the probe's 15 epochs over 98k photos; fewer passes over more data
    batch = batch or (512 if n < 200_000 else 1024)
    print(f"  {n:,} photos, {k} cells (median {int(np.median(np.bincount(cell, minlength=k)))} photos), {epochs} epochs, batch {batch}", flush=True)
    t = time.perf_counter()
    logp = pp._fit_mlp(Xt, y, np.arange(n), seed, epochs=epochs, batch=batch)
    lp = logp(np.arange(n, n + 3000)).astype(np.float32)
    top = cent[lp.argmax(1)]
    acc = [100 * np.mean(_km(top, truth) < r) for r in (25, 200, 750, 2500)]
    print(f"  trained in {time.perf_counter() - t:.0f} s; dev3k top cell within 25 / 200 / 750 / 2500 km: "
          + " / ".join(f"{a:.1f}" for a in acc), flush=True)
    out = OUT / f"region_expert_{data}{'' if backbone == 'pe' else '_' + backbone}_{k}{f'_{max_n}' if max_n else ''}.npz"
    np.savez(out, logp=lp, cells=cent.astype(np.float32), n=n, acc=np.asarray(acc))
    print(f"-> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("node", choices=("train",))
    ap.add_argument("--data", default="b2", choices=("probe", "b1", "b2"))
    ap.add_argument("--cells", type=int, default=2048)
    ap.add_argument("--max", type=int, default=0, help="subsample the training photos to this many")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--backbone", default="pe", choices=("pe", "siglip", "concat"))
    args = ap.parse_args()
    train(args.data, args.cells, args.max, args.epochs, args.batch, backbone=args.backbone)


if __name__ == "__main__":
    main()
