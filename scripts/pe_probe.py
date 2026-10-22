"""PE_PLAN.md step 5b: the Perception Encoder paper's middle-layer claim, tested as the paper tests it. Every layer result in the paper
(Fig. 8) is a readout *trained* on that layer's frozen features; step 5 (lesson R16) only compared mean-pooled vectors by cosine. Here the same
attention probe is trained on each layer's patch tokens to predict a geographic cell, and compared with the last block and the pooled output.

select: the 35,359 train photos + 65k random MP16 photos, dropping dev3k photos and every photo by a dev3k photographer (leak guard)
        -> artifacts/retrieval/probe/photos.npz (ids, latlon); the 3,000 dev3k queries are appended at embed time.
embed (.venvs/sft, timm on PYTHONPATH): PE-G at 378 px; patch tokens (27 x 27) of blocks 32 / 38 / 44 / 47 and the final normed tokens,
        average-pooled to 7 x 7 -> tok_{name}.npy [N, 49, 1536] fp16, plus the pooled output -> pooled.npy [N, 1280]; resumable.
train:  k-means cells over the probe photos' GPS; per feature the same probe (attention probe on tokens; MLP on the pooled output), the best
        epoch on a 5% held-out split, then dev3k: % of queries whose top cell's centre is within 25 / 200 / 750 / 2500 km; dev3k log-probs saved.
gate:   per candidate area the log-prob of its cell (and of cells within 100 km), raw and relative to the photo's other areas, stacked on the
        answer head's option logits as in scripts/siglip_gate.py (conditional logit, 2-fold x 3 seeds).

  .venv/bin/python scripts/pe_probe.py select
  PYTHONPATH=<timm>:<value-agent>/src .venvs/sft/bin/python scripts/pe_probe.py embed
  PYTHONPATH=<value-agent>/src .venvs/sft/bin/python scripts/pe_probe.py train
  PYTHONPATH=<value-agent>/src:<value-agent>/scripts .venv/bin/python scripts/pe_probe.py gate
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path

import numpy as np

OUT = Path("artifacts/retrieval/probe")
CSV = Path("data/hf/datasets/MP16-Pro/metadata/MP16_Pro_filtered.csv")
MODEL = "vit_pe_core_gigantic_patch14_448.fb"
SIZE, GRID = 378, 7
LAYERS = {"l32": 31, "l38": 37, "l44": 43, "l47": 46}  # name -> block index (0-based); "last" = the final normed tokens
FEATURES = ("l32", "l38", "l44", "l47", "last", "pooled")
N_EXTRA, CELLS = 65_000, 2048


def _xyz(latlon: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(latlon[:, 0]), np.radians(latlon[:, 1])
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 1)


def _km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle km between rows of a and b ([n, 2] lat/lon degrees)."""
    return 6371.0 * np.arccos(np.clip((_xyz(a) * _xyz(b)).sum(1), -1, 1))


def select(seed: int = 0) -> None:
    splits = json.loads(Path("artifacts/agent/splits.json").read_text(encoding="utf-8"))
    dev = set(splits["dev"])
    ids, authors, latlon = [], [], []
    with CSV.open(encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            ids.append(row[0])
            authors.append(row[1])
            latlon.append((float(row[2]), float(row[3])))
    ids, authors, latlon = np.asarray(ids), np.asarray(authors), np.asarray(latlon, np.float32)
    dev_authors = set(authors[np.isin(ids, list(dev))])
    clean = ~np.isin(ids, list(dev)) & ~np.isin(authors, list(dev_authors))
    is_train = np.isin(ids, splits["train"])
    rng = np.random.default_rng(seed)
    extra = rng.choice(np.flatnonzero(clean & ~is_train), N_EXTRA, replace=False)
    keep = np.concatenate([np.flatnonzero(clean & is_train), np.sort(extra)])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "photos.npz", ids=ids[keep], latlon=latlon[keep])
    print(f"-> {OUT / 'photos.npz'}: {int((clean & is_train).sum())} train + {N_EXTRA} random MP16 photos "
          f"({int((is_train & ~clean).sum())} train photos dropped: dev3k photographers)", flush=True)


def _dev(tag: str = "dev3k", ckpt: str = "setft_own") -> tuple[list[str], np.ndarray]:
    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{ckpt}" / "photos.json").read_text(encoding="utf-8"))
    return [p["image_id"] for p in photos], np.asarray([p["truth"] for p in photos], np.float32)


def embed(batch: int = 64, workers: int = 4) -> None:
    os.environ.setdefault("HF_HOME", str(Path("data/hf").resolve()))
    os.environ["HF_HUB_OFFLINE"] = "1"
    import timm
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from geo_search_env.experiment.sft_data import MP16Images

    p = np.load(OUT / "photos.npz")
    dev_ids, _ = _dev()
    ids = list(p["ids"]) + dev_ids
    n = len(ids)
    net = timm.create_model(MODEL, pretrained=True, img_size=SIZE).cuda().eval().to(torch.bfloat16)
    cfg = timm.data.resolve_data_config(model=net) | {"input_size": (3, SIZE, SIZE)}
    tf = timm.data.create_transform(**cfg)

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

    names = list(LAYERS) + ["last"]
    prog = OUT / "embed.progress"
    if prog.exists():
        toks = {k: np.lib.format.open_memmap(OUT / f"tok_{k}.npy", mode="r+") for k in names}
        pooled = np.lib.format.open_memmap(OUT / "pooled.npy", mode="r+")
        ok = np.load(OUT / "ok.npy")
        start = int(prog.read_text())
    else:
        toks = {k: np.lib.format.open_memmap(OUT / f"tok_{k}.npy", mode="w+", dtype=np.float16, shape=(n, GRID * GRID, 1536)) for k in names}
        pooled = np.lib.format.open_memmap(OUT / "pooled.npy", mode="w+", dtype=np.float16, shape=(n, 1280))
        ok = np.zeros(n, bool)
        start = 0
    print(f"probe embed: {len(p['ids'])} probe photos + {len(dev_ids)} dev3k queries at {SIZE} px; starting at {start}", flush=True)
    side = SIZE // 14

    def grid(t):  # [B, 729, C] -> [B, 49, C]
        t = t.float().transpose(1, 2).reshape(len(t), -1, side, side)
        return F.adaptive_avg_pool2d(t, GRID).flatten(2).transpose(1, 2).cpu().numpy().astype(np.float16)

    done = start
    with torch.inference_mode():
        for b, (idx, x, good) in enumerate(DataLoader(torch.utils.data.Subset(Images(), list(range(start, n))), batch_size=batch,
                                                      num_workers=workers)):
            final, inter = net.forward_intermediates(x.cuda().to(torch.bfloat16), indices=list(LAYERS.values()), output_fmt="NLC")
            i = idx.numpy()
            for k, t in zip(LAYERS, inter):
                toks[k][i] = grid(t)
            toks["last"][i] = grid(final)
            pooled[i] = net.forward_head(final).float().cpu().numpy().astype(np.float16)
            ok[i] = good.numpy()
            done += len(i)
            if b % 200 == 199 or done == n:
                for t in toks.values():
                    t.flush()
                pooled.flush()
                np.save(OUT / "ok.npy", ok)
                prog.write_text(str(done))
                print(f"  {done:,} / {n:,}", flush=True)
    print(f"-> {OUT}: {np.mean(~ok):.2%} unreadable", flush=True)


def _cells(latlon: np.ndarray, k: int, seed: int = 0):
    """Spherical k-means on unit xyz (GPU): centroids [k, 2] lat/lon and each photo's cell."""
    import torch

    x = torch.tensor(_xyz(latlon), device="cuda", dtype=torch.float32)
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k].cuda()].clone()
    for _ in range(30):
        a = (x @ c.T).argmax(1)
        s = torch.zeros_like(c).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=k)
        c = torch.where(cnt[:, None] > 0, torch.nn.functional.normalize(s, dim=1), c)
    a = (x @ c.T).argmax(1).cpu().numpy()
    c = c.cpu().numpy()
    return np.stack([np.degrees(np.arcsin(np.clip(c[:, 2], -1, 1))), np.degrees(np.arctan2(c[:, 1], c[:, 0]))], 1), a


def _probe(dim: int, k: int, tokens: bool):
    import torch
    from torch import nn

    class AttnProbe(nn.Module):  # the paper's classification probe: one randomly initialised cross-attention block on frozen tokens
        def __init__(self, width: int = 768, heads: int = 8) -> None:
            super().__init__()
            self.inp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width))
            self.pos = nn.Parameter(torch.zeros(1, GRID * GRID, width))
            self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
            self.attn = nn.MultiheadAttention(width, heads, dropout=0.1, batch_first=True)
            self.mlp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(0.1), nn.Linear(4 * width, width))
            self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, k))

        def forward(self, t):
            t = self.inp(t) + self.pos
            q = self.query.expand(len(t), -1, -1)
            q = q + self.attn(q, t, t, need_weights=False)[0]
            q = q + self.mlp(q)
            return self.out(q[:, 0])

    if tokens:
        return AttnProbe()
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 768), nn.GELU(), nn.Dropout(0.1), nn.Linear(768, 768), nn.GELU(),
                         nn.LayerNorm(768), nn.Linear(768, k))


def train(epochs: int = 15, batch: int = 512, seed: int = 0) -> None:
    import torch
    import torch.nn.functional as F

    p = np.load(OUT / "photos.npz")
    _, dev_ll = _dev()
    n_p, ok = len(p["ids"]), np.load(OUT / "ok.npy")
    cent, cell = _cells(p["latlon"], CELLS, seed)
    rng = np.random.default_rng(seed)
    use = np.flatnonzero(ok[:n_p])
    va = rng.choice(use, len(use) // 20, replace=False)
    tr = np.setdiff1d(use, va)
    dev_rows = n_p + np.arange(len(dev_ll))
    y = torch.tensor(cell, device="cuda")
    print(f"probe train: {len(tr)} train / {len(va)} held-out photos, {CELLS} cells (median {int(np.median(np.bincount(cell)))} photos); "
          f"dev3k {len(dev_ll)} queries ({int(ok[dev_rows].sum())} readable)", flush=True)
    results = {}
    for name in FEATURES:
        src = np.load(OUT / ("pooled.npy" if name == "pooled" else f"tok_{name}.npy"), mmap_mode="r")
        X = torch.empty(src.shape, dtype=torch.float16, device="cuda")  # whole feature on the GPU (~15 GB for tokens)
        for s in range(0, len(src), 8192):
            X[s:s + 8192] = torch.from_numpy(np.ascontiguousarray(src[s:s + 8192]))
        torch.manual_seed(seed)
        net = _probe(src.shape[-1], CELLS, name != "pooled").cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
        steps = epochs * (len(tr) // batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
        best, best_state = np.inf, None

        def logp(rows: np.ndarray) -> torch.Tensor:
            net.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = torch.cat([F.log_softmax(net(X[torch.as_tensor(rows[s:s + 2048], device="cuda")].float()).float(), 1)
                                 for s in range(0, len(rows), 2048)])
            net.train()
            return out

        for ep in range(epochs):
            perm = rng.permutation(tr)
            for s in range(0, len(perm) - batch + 1, batch):
                b = torch.as_tensor(perm[s:s + batch], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(net(X[b].float()).float(), y[b], label_smoothing=0.1)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
            v = F.nll_loss(logp(va), y[torch.as_tensor(va, device="cuda")]).item()
            if v < best:
                best, best_state = v, {k: t.detach().clone() for k, t in net.state_dict().items()}
        net.load_state_dict(best_state)
        lp = logp(dev_rows).cpu().numpy()
        km = _km(cent[lp.argmax(1)], dev_ll)
        km = np.where(ok[dev_rows], km, np.inf)
        vacc = (logp(va).argmax(1) == y[torch.as_tensor(va, device="cuda")]).float().mean().item()
        results[name] = {r: float(np.mean(km < r)) for r in (25, 200, 750, 2500)}
        np.save(OUT / f"dev_logp_{name}.npy", lp.astype(np.float16))
        print(f"  {name:6s}: held-out loss {best:.3f}, cell acc {vacc:.1%}; dev3k top cell within 25 / 200 / 750 / 2500 km: "
              + " / ".join(f"{100 * results[name][r]:.1f}" for r in (25, 200, 750, 2500)), flush=True)
        del X, net, opt
        torch.cuda.empty_cache()
    np.save(OUT / "cells.npy", cent)
    (OUT / "train_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


def embed_pooled(batch: int = 64, workers: int = 4) -> None:
    """Step 5c: PE-G 378 px pooled output for the photos the probe set lacks: train photos by dev3k photographers (dropped from the probe
    set) and the commons26 dev queries -> pooled_extra.npz (keys, embeddings)."""

    os.environ.setdefault("HF_HOME", str(Path("data/hf").resolve()))
    os.environ["HF_HUB_OFFLINE"] = "1"
    import timm
    import torch
    from PIL import Image

    from geo_search_env.experiment import megaloc_gate
    from geo_search_env.experiment.sft_data import MP16Images

    have = set(np.load(OUT / "photos.npz")["ids"])
    train = [p["image_id"] for p in json.loads(Path("artifacts/photo_sets/train_setft_own/photos.json").read_text(encoding="utf-8"))]
    keys = [i for i in train if i not in have]
    megaloc_gate.CKPT_TAG = "setft_own"
    c26, nq, _ = megaloc_gate._image_ids("commons26dev", np.zeros(0, np.int64))  # benchmark queries: resolved file paths
    keys += [f"commons26dev:{i}" for i in range(nq)]
    net = timm.create_model(MODEL, pretrained=True, img_size=SIZE).cuda().eval().to(torch.bfloat16)
    tf = timm.data.create_transform(**(timm.data.resolve_data_config(model=net) | {"input_size": (3, SIZE, SIZE)}))
    images = MP16Images()

    def load(k: str):
        data = Path(c26[int(k.split(":")[1])]).read_bytes() if k.startswith("commons26dev:") else images.read(k)
        return tf(Image.open(io.BytesIO(data)).convert("RGB"))

    out = np.zeros((len(keys), 1280), np.float16)
    with torch.inference_mode():
        for s in range(0, len(keys), batch):
            x = torch.stack([load(k) for k in keys[s:s + batch]]).cuda().to(torch.bfloat16)
            out[s:s + batch] = net(x).float().cpu().numpy().astype(np.float16)
    np.savez(OUT / "pooled_extra.npz", keys=np.asarray(keys), embeddings=out)
    print(f"-> {OUT / 'pooled_extra.npz'}: {len(keys) - nq} train photos + {nq} commons26 dev queries", flush=True)


def _fit_mlp(X, y, rows: np.ndarray, seed: int, epochs: int = 15, batch: int = 512):
    """The pooled-output probe of train(), on rows of X (GPU fp16); 5% of rows held out for the epoch. Returns rows -> log-probs."""
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    va = rng.choice(rows, len(rows) // 20, replace=False)
    tr = np.setdiff1d(rows, va)
    torch.manual_seed(seed)
    net = _probe(X.shape[-1], CELLS, False).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * (len(tr) // batch), pct_start=0.05)

    def logp(r: np.ndarray) -> np.ndarray:
        net.eval()
        with torch.no_grad():
            out = torch.cat([F.log_softmax(net(X[torch.as_tensor(r[s:s + 4096], device="cuda")].float()), 1) for s in range(0, len(r), 4096)])
        net.train()
        return out.cpu().numpy()

    best, state = np.inf, None
    for _ in range(epochs):
        perm = rng.permutation(tr)
        for s in range(0, len(perm) - batch + 1, batch):
            b = torch.as_tensor(perm[s:s + batch], device="cuda")
            loss = F.cross_entropy(net(X[b].float()), y[b], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        v = -logp(va)[np.arange(len(va)), y[torch.as_tensor(va, device="cuda")].cpu().numpy()].mean()
        if v < best:
            best, state = v, {k: t.detach().clone() for k, t in net.state_dict().items()}
    net.load_state_dict(state)
    return logp


def _columns(lp: np.ndarray, zm, cent: np.ndarray) -> np.ndarray:
    """Per zoom-file hypothesis: log-prob of the area's cell and log mass of cells within 100 km of its centre, raw and relative to the best
    of the photo's other (non-street) areas. lp: [photos, cells] log-probs in photo-set order."""
    photo, street = zm["photo"], zm["source"] == 2
    cx = _xyz(cent)
    own, mass = np.zeros(len(photo), np.float32), np.zeros(len(photo), np.float32)
    for s in range(0, len(photo), 8192):
        sl = slice(s, s + 8192)
        km = 6371.0 * np.arccos(np.clip(_xyz(zm["center"][sl]) @ cx.T, -1, 1))
        l = lp[photo[sl]]
        own[sl] = l[np.arange(len(km)), km.argmin(1)]
        mass[sl] = np.log(np.maximum(np.where(km < 100, np.exp(l), 0).sum(1), 1e-12))
    cols = [own, mass]
    order = np.argsort(photo, kind="stable")
    bounds = np.flatnonzero(np.diff(photo[order])) + 1
    for v in (own, mass):
        rel = np.zeros_like(v)
        for J in np.split(order, bounds):
            for i, j in enumerate(J):
                o = np.delete(v[J], i)
                if not street[j]:
                    o = o[~np.delete(street[J], i)]
                rel[j] = v[j] - (o.max() if len(o) else v[j])
        cols.append(rel)
    return np.stack(cols, 1).astype(np.float32)


def crossfit(folds: int = 5, seed: int = 0, ckpt: str = "setft_own") -> None:
    """Step 5c: region-probe columns for the trained answer head (track V's head(), hypothesis_agent.EXTRA_FILES format). Train photos get
    log-probs from a probe trained without their fold, where folds group photographers and every probe photo by a fold's photographers is
    left out of that fold's probe; dev3k and commons26 dev get the probe trained on all probe photos (no dev3k photo or photographer)."""
    import torch

    p = np.load(OUT / "photos.npz")
    ok, cent = np.load(OUT / "ok.npy"), np.load(OUT / "cells.npy")
    n_p = len(p["ids"])
    base = np.load(OUT / "pooled.npy", mmap_mode="r")
    extra = np.load(OUT / "pooled_extra.npz")
    ekeys = list(extra["keys"])
    X = torch.tensor(np.concatenate([np.asarray(base), extra["embeddings"]]), device="cuda")  # probe photos, dev3k queries, extras
    y = torch.tensor((_xyz(p["latlon"]) @ _xyz(cent).T).argmax(1), device="cuda")
    row = {k: i for i, k in enumerate(p["ids"])} | {k: len(base) + i for i, k in enumerate(ekeys)}
    authors = {}
    with CSV.open(encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        next(r)
        for rec in r:
            authors[rec[0]] = rec[1]
    probe_auth = np.asarray([authors[i] for i in p["ids"]])
    train = [e["image_id"] for e in json.loads(Path(f"artifacts/photo_sets/train_{ckpt}/photos.json").read_text(encoding="utf-8"))]
    t_auth = np.asarray([authors[i] for i in train])
    uniq = np.unique(t_auth)
    fold_of = dict(zip(uniq, np.random.default_rng(seed).permutation(len(uniq)) % folds))
    t_fold = np.asarray([fold_of[a] for a in t_auth])
    usable = np.flatnonzero(ok[:n_p])
    lp_train = np.zeros((len(train), CELLS), np.float32)
    for k in range(folds):
        held = set(uniq[[fold_of[a] == k for a in uniq]])
        rows = usable[~np.isin(probe_auth[usable], list(held))]
        logp = _fit_mlp(X, y, rows, seed + k)
        J = np.flatnonzero(t_fold == k)
        lp_train[J] = logp(np.asarray([row[train[j]] for j in J]))
        print(f"  fold {k}: probe on {len(rows)} photos (photographers of {len(J)} train photos left out)", flush=True)
    logp = _fit_mlp(X, y, usable, seed)
    lp = {"train": lp_train, "dev3k": logp(n_p + np.arange(len(base) - n_p)),
          "commons26dev": logp(np.asarray([row[k] for k in ekeys if k.startswith("commons26dev:")]))}
    for tag, l in lp.items():
        zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{ckpt}.npz")
        x = _columns(l, zm, cent)
        assert np.isfinite(x).all()
        dst = OUT / f"pe_probe_cols_{tag}_{ckpt}.npz"
        np.savez(dst, photo=zm["photo"], source=zm["source"], x=x, groups=np.asarray(["pe_probe"] * x.shape[1]))
        if tag != "train":
            truth = np.asarray([e["truth"] for e in json.loads((Path("artifacts/photo_sets") / f"{tag}_{ckpt}" / "photos.json").read_text())])
            print(f"  {tag}: top cell within 200 / 750 km {np.mean(_km(cent[l.argmax(1)], truth) < 200):.1%} / "
                  f"{np.mean(_km(cent[l.argmax(1)], truth) < 750):.1%}", flush=True)
        else:
            truth = np.asarray([e["truth"] for e in json.loads(Path(f"artifacts/photo_sets/train_{ckpt}/photos.json").read_text())])
            print(f"  train (cross-fitted): top cell within 200 / 750 km {np.mean(_km(cent[l.argmax(1)], truth) < 200):.1%} / "
                  f"{np.mean(_km(cent[l.argmax(1)], truth) < 750):.1%}", flush=True)
        print(f"-> {dst}: {x.shape}", flush=True)


def gate(tag: str = "dev3k", ckpt: str = "setft_own") -> None:
    import second_model_gate as g

    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{ckpt}.npz")
    cent = np.load(OUT / "cells.npy")
    photo, street = zm["photo"], zm["source"] == 2
    near = np.stack([_km(np.repeat(zm["center"][i:i + 1], len(cent), 0), cent) for i in range(len(photo))])  # [areas, cells]
    logits, km, hyp = g.head_options(tag, ckpt)
    valid = np.isfinite(logits)
    base = g.clogit(np.where(valid, logits, 0.0)[..., None], valid, km)
    print(f"{tag}: stacked head only {np.mean(base < 25):.2%} / {np.mean(base < 1):.2%} (< 25 / 1 km, 2-fold x 3 seeds)", flush=True)

    def columns(name: str) -> np.ndarray:
        lp = np.load(OUT / f"dev_logp_{name}.npy").astype(np.float32)[photo]  # [areas, cells]
        own = lp[np.arange(len(photo)), near.argmin(1)]
        mass = np.log(np.maximum(np.where(near < 100, np.exp(lp), 0).sum(1), 1e-12))
        cols = [own, mass]
        for v in (own, mass):
            rel = np.zeros_like(v)
            for p in np.unique(photo):
                J = np.flatnonzero(photo == p)
                for i, j in enumerate(J):
                    o = np.delete(v[J], i)
                    if not street[j]:
                        o = o[~np.delete(street[J], i)]
                    rel[j] = v[j] - (o.max() if len(o) else v[j])
            cols.append(rel)
        return np.stack(cols, 1).astype(np.float32)

    groups = {name: columns(name) for name in FEATURES}
    groups["pooled + l47"] = np.concatenate([groups["pooled"], groups["l47"]], 1)
    # does the probe's region prediction stack on V5's best columns (PE-G + SigLIP2 cosine to each area's photos)?
    g.OUT = Path("artifacts/retrieval/gate")
    cos = np.concatenate([g.columns("pe_g", tag, ckpt, zm), g.columns("siglip_final", tag, ckpt, zm)], 1)
    groups["cosine pe_g + siglip"] = cos
    groups["cosine + probe pooled"] = np.concatenate([cos, groups["pooled"]], 1)
    for name, c in groups.items():
        X = np.concatenate([np.where(valid, logits, 0.0)[..., None], np.where((hyp >= 0)[..., None], c[np.maximum(hyp, 0)], 0.0)], -1)
        res = g.clogit(X, valid, km)
        d25, d1 = (res < 25).mean(0) - (base < 25).mean(0), (res < 1).mean(0) - (base < 1).mean(0)
        print(f"  + probe {name:14s}: {100 * d25.mean():+.2f} / {100 * d1.mean():+.2f} pts (paired SE {100 * d25.std() / np.sqrt(len(d25)):.2f})",
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("node", choices=("select", "embed", "train", "gate", "embed_pooled", "crossfit"))
    args = ap.parse_args()
    {"select": select, "embed": embed, "train": train, "gate": gate, "embed_pooled": embed_pooled, "crossfit": crossfit}[args.node]()


if __name__ == "__main__":
    main()
