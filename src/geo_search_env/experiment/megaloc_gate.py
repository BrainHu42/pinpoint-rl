# Same-place recognition without an LLM (lesson 93, the user's go-ahead): MegaLoc (Berton & Masone, CVPR-W 2025; gberton/MegaLoc, MIT), a
# visual place-recognition model (DINOv2 ViT-B/14 + optimal-transport aggregation of patch features, 8,448-d, trained to match the same place
# across viewpoint, time and season). SigLIP2 (Pinpoint's space) matches "the same kind of scene"; MegaLoc should match "the same building".
# Gate (dev3k): embed the queries and every photo of their hypotheses' restricted searches (the zoom file's 12 per area, photographer
# excluded), then per area the best / top-3 MegaLoc similarity; does it separate right from wrong areas better than SigLIP2 and add to a
# conditional logit over Pinpoint's features?
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.megaloc_gate embed --tag dev3k
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.megaloc_gate gate --tag dev3k

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
CKPT_TAG = "setft_a"
BASE = ""  # features --base: the run whose similarities a variant reuses
SNAPSHOT = Path("data/hf/hub/models--gberton--MegaLoc/snapshots/a0f34722c4297ff787e022433799250180860af7")
SIZE = 322  # the paper's evaluation resolution


def _model():
    import torch
    from safetensors.torch import load_file

    import sys
    import types

    if "torchvision" not in sys.modules:  # the repo env has no torchvision; the model file only uses its resize (inputs here are already 322 px)
        tv, tvt, fn = types.ModuleType("torchvision"), types.ModuleType("torchvision.transforms"), types.ModuleType("torchvision.transforms.functional")
        fn.resize = lambda x, size, antialias=True: torch.nn.functional.interpolate(x, size=size, mode="bilinear", antialias=antialias)
        tv.transforms, tvt.functional = tvt, fn
        sys.modules.update({"torchvision": tv, "torchvision.transforms": tvt, "torchvision.transforms.functional": fn})
    spec = importlib.util.spec_from_file_location("megaloc_model", SNAPSHOT / "megaloc_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.MegaLoc()
    missing, unexpected = model.load_state_dict(load_file(SNAPSHOT / "model.safetensors"), strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return model.cuda().eval()  # float32: the optimal-transport aggregation mixes dtypes under .half()


def _paths(tag: str) -> tuple[Path, Path]:
    return OUT / f"megaloc_meta_{tag}_{CKPT_TAG}.npz", OUT / f"megaloc_emb_{tag}_{CKPT_TAG}.npy"


def embed(tag: str, batch: int = 64, workers: int = 3) -> None:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from ..models.pinpoint_reranker import _discover_index
    from .sft_data import MP16Images
    from .strategy_search import MP16_CSV, MP16_EMBED, _memmap_gallery

    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{CKPT_TAG}.npz")
    rows = zm["rows"]
    need = np.unique(rows[rows >= 0])
    index_dir = _discover_index()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    cache = np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
    row_index = _memmap_gallery(MP16_EMBED)["row_index"]
    with MP16_CSV.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        csv_ids = [r[0] for r in reader]
    ids = [e["image_id"] for e in photos] + [csv_ids[row_index[cache[r]]] for r in need]
    print(f"{tag}: {len(photos)} queries + {len(need)} area photos", flush=True)
    mean, std = np.asarray([0.485, 0.456, 0.406], np.float32), np.asarray([0.229, 0.224, 0.225], np.float32)

    def tf(im):  # ImageNet normalisation at 322 x 322 (the model card's preprocessing, PIL resize instead of torchvision's)
        a = (np.asarray(im.resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255.0 - mean) / std
        return torch.from_numpy(a.transpose(2, 0, 1).copy())

    class Images(Dataset):
        images = None

        def __len__(self) -> int:
            return len(ids)

        def __getitem__(self, i: int):
            if self.images is None:
                self.images = MP16Images()
            data = self.images.read(ids[i])
            try:
                return i, tf(Image.open(io.BytesIO(data)).convert("RGB"))
            except Exception:  # noqa: BLE001
                return i, torch.zeros(3, SIZE, SIZE)

    model = _model()
    out = np.zeros((len(ids), 8448), np.float16)
    loader = DataLoader(Images(), batch_size=batch, num_workers=workers)
    with torch.inference_mode():
        for k, (idx, x) in enumerate(loader):
            out[idx.numpy()] = model(x.cuda()).float().cpu().numpy().astype(np.float16)
            if k % 200 == 0:
                print(f"  {k * batch} / {len(ids)}", flush=True)
    meta, emb = _paths(tag)
    np.save(emb, out)
    np.savez(meta, n_queries=len(photos), rows=need)
    print(f"-> {emb}")


def features(tag: str, batch: int = 64, workers: int = 3) -> None:
    """Streaming version for large sets (train: ~1.2M area photos): embeds the queries, then the area photos in index-row order, keeping only
    each (hypothesis, slot) similarity; resumable (partial state saved every ~50k photos). Writes megaloc_{tag}_{CKPT_TAG}.npz in the column
    group format hypothesis_agent --extra reads (group "megaloc")."""

    import torch
    from torch.utils.data import DataLoader

    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{CKPT_TAG}.npz")
    rows, photo = zm["rows"], zm["photo"]
    sims = np.full(rows.shape, np.nan, np.float32)
    todo = rows >= 0
    base_x = None
    if BASE:  # a variant whose zoom file starts with the base's rows (extra hypotheses appended): reuse those, embed only the new rows' photos
        b, bz = np.load(OUT / f"megaloc_{tag}_{BASE}.npz"), np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{BASE}.npz")
        nb = len(bz["photo"])
        assert np.array_equal(bz["rows"], rows[:nb]) and np.array_equal(bz["photo"], photo[:nb]), "variant does not extend the base"
        base_x = b["x"]  # older base files keep only the per-row columns (best, top-3 mean), which is all the variant needs
        todo[:nb] = False
    need = np.unique(rows[todo])
    h_idx, k_idx = np.nonzero(todo)
    pos = np.searchsorted(need, rows[h_idx, k_idx])
    order = np.argsort(pos, kind="stable")
    h_idx, k_idx, pos = h_idx[order], k_idx[order], pos[order]
    state = OUT / f"megaloc_partial_{tag}_{CKPT_TAG}.npz"
    done = 0
    if state.exists():
        st = np.load(state)
        sims, done = st["sims"], int(st["done"])
        print(f"resuming at {done} / {len(need)}", flush=True)
    ids, nq, dataset_cls = _image_ids(tag, need)
    model = _model()
    qpath = OUT / f"megaloc_queries_{tag}_{CKPT_TAG}.npy"
    if BASE and not qpath.exists():
        qpath = OUT / f"megaloc_queries_{tag}_{BASE}.npy"
    if qpath.exists():
        q = torch.as_tensor(np.load(qpath), device="cuda")
    else:
        q = _embed_all(model, dataset_cls(ids[:nq]), batch, workers).half()
        np.save(qpath, q.cpu().numpy())
    print(f"{tag}: {nq} queries embedded; {len(need)} area photos, {len(h_idx)} (hypothesis, photo) pairs", flush=True)
    loader = DataLoader(dataset_cls(ids[nq + done:]), batch_size=batch, num_workers=workers)
    ph = torch.as_tensor(photo, device="cuda")
    ptr = int(np.searchsorted(pos, done))
    start = done
    with torch.inference_mode():
        for b, (_, x) in enumerate(loader):
            e = model(x.cuda()).half()
            end = start + len(e)
            stop = int(np.searchsorted(pos, end))
            if stop > ptr:
                hs, ks, ps = h_idx[ptr:stop], k_idx[ptr:stop], pos[ptr:stop] - start
                v = (e[torch.as_tensor(ps, device="cuda")].float() * q[ph[torch.as_tensor(hs, device="cuda")]].float()).sum(1)
                sims[hs, ks] = v.cpu().numpy()
            ptr, start = stop, end
            if b % 800 == 799:
                np.savez(state, sims=sims, done=start)
                print(f"  {start} / {len(need)}", flush=True)
    _write_columns(tag, zm, sims, base_x)
    state.unlink(missing_ok=True)


def _image_ids(tag: str, need: np.ndarray):
    import torch
    from PIL import Image
    from torch.utils.data import Dataset

    from ..models.pinpoint_reranker import _discover_index
    from .sft_data import MP16Images
    from .strategy_search import MP16_CSV, MP16_EMBED, _memmap_gallery

    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
    index_dir = _discover_index()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    cache = np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
    row_index = _memmap_gallery(MP16_EMBED)["row_index"]
    with MP16_CSV.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        csv_ids = [r[0] for r in reader]
    # benchmark queries (photo set full) are read from their files, MP16 photos from the tar shards
    def bench_path(e) -> str:  # stored paths can point into deleted worktrees; commons26 images live under release/
        options = [e.get("path"), Path("data/pinpoint") / e["benchmark"] / "images" / e["image_id"],
                   Path("data/pinpoint") / e["benchmark"] / "release" / "images" / e["image_id"]]
        found = next((str(p) for p in options if p and Path(p).exists()), None)
        if found is None:
            raise FileNotFoundError(f"no image file for {e['benchmark']} {e['image_id']}")
        return found

    ids = [bench_path(e) if "benchmark" in e else e["image_id"] for e in photos]
    ids += [csv_ids[row_index[cache[r]]] for r in need]
    mean, std = np.asarray([0.485, 0.456, 0.406], np.float32), np.asarray([0.229, 0.224, 0.225], np.float32)

    class Images(Dataset):
        images = None

        def __init__(self, keys):
            self.keys = keys

        def __len__(self) -> int:
            return len(self.keys)

        def __getitem__(self, i: int):
            if self.images is None:
                self.images = MP16Images()
            try:
                key = self.keys[i]
                data = Path(key).read_bytes() if key.startswith("data/pinpoint/") or key.startswith("/") else self.images.read(key)
                im = Image.open(io.BytesIO(data)).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
                a = (np.asarray(im, np.float32) / 255.0 - mean) / std
                return i, torch.from_numpy(a.transpose(2, 0, 1).copy())
            except Exception:  # noqa: BLE001
                return i, torch.zeros(3, SIZE, SIZE)

    return ids, len(photos), Images


def _embed_all(model, dataset, batch: int, workers: int):
    import torch
    from torch.utils.data import DataLoader

    out = []
    with torch.inference_mode():
        for _, x in DataLoader(dataset, batch_size=batch, num_workers=workers):
            out.append(model(x.cuda()))
    return torch.cat(out)


def _write_columns(tag: str, zm, sims: np.ndarray, base_x: np.ndarray | None = None) -> None:
    photo = zm["photo"]
    srt = np.sort(np.where(np.isnan(sims), -np.inf, sims), 1)[:, -3:]
    best = np.where(np.isfinite(srt[:, -1]), srt[:, -1], -1.0)
    with np.errstate(invalid="ignore"):
        top3 = np.nan_to_num(np.nanmean(np.where(np.isfinite(srt), srt, np.nan), 1), nan=-1.0)
    if base_x is not None:  # the base run's rows (features --base)
        best[:len(base_x)], top3[:len(base_x)] = base_x[:, 0], base_x[:, 1]
    cols = [best, top3]
    street = zm["source"] == 2  # street-search areas: out of the other rows' reference (they are hidden until that action)
    for v in (best, top3):
        rel = np.zeros_like(v)
        for p in np.unique(photo):
            J = np.flatnonzero(photo == p)
            for k, j in enumerate(J):
                o = np.delete(v[J], k)
                if not street[j]:
                    o = o[~np.delete(street[J], k)]
                rel[j] = v[j] - (o.max() if len(o) else v[j])
        cols.append(rel)
    x = np.stack(cols, 1).astype(np.float32)
    dst = OUT / f"megaloc_{tag}_{CKPT_TAG}.npz"
    np.savez(dst, photo=photo, source=zm["source"], x=x, groups=np.asarray(["megaloc"] * x.shape[1]), sims=sims.astype(np.float16))
    print(f"-> {dst}: {np.mean(np.isnan(sims[zm['rows'] >= 0])):.2%} pairs missing")


def columns_from_gate(tag: str) -> None:
    """dev3k: the same columns from the gate's stored embeddings."""

    meta, emb = _paths(tag)
    m = np.load(meta)
    e = np.load(emb, mmap_mode="r")
    nq, need = int(m["n_queries"]), m["rows"]
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{CKPT_TAG}.npz")
    rows, photo = zm["rows"], zm["photo"]
    q = np.asarray(e[:nq], np.float32)
    sims = np.full(rows.shape, np.nan, np.float32)
    for h in range(len(photo)):
        v = np.flatnonzero(rows[h] >= 0)
        if len(v):
            sims[h, v] = np.asarray(e[nq + np.searchsorted(need, rows[h, v])], np.float32) @ q[photo[h]]
    _write_columns(tag, zm, sims)


def per_hypothesis(tag: str) -> tuple[np.ndarray, list[str]]:
    meta, emb = _paths(tag)
    m = np.load(meta)
    e = np.load(emb, mmap_mode="r")
    nq, need = int(m["n_queries"]), m["rows"]
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{CKPT_TAG}.npz")
    rows, photo = zm["rows"], zm["photo"]
    q = np.asarray(e[:nq], np.float32)
    best = np.full(len(photo), -1.0, np.float32)
    top3 = np.full(len(photo), -1.0, np.float32)
    for h in range(len(photo)):
        v = rows[h][rows[h] >= 0]
        if not len(v):
            continue
        a = np.asarray(e[nq + np.searchsorted(need, v)], np.float32)
        s = np.sort(a @ q[photo[h]])
        best[h], top3[h] = s[-1], s[-3:].mean()
    return np.stack([best, top3], 1), ["ml_best", "ml_top3"]


def gate(tag: str) -> None:
    import torch

    x, names = per_hypothesis(tag)
    f = np.load(OUT / f"hypothesis_features_{tag}_{CKPT_TAG}_e4_set.npz")
    keep = np.isin(f["source"], (0, 1))
    photo, right, x = f["photo"][keep], f["km_agree"][keep] < 25, x[keep]
    P = photo.max() + 1

    def rel(v):
        out = np.zeros_like(v)
        for q in range(P):
            J = np.flatnonzero(photo == q)
            for k, j in enumerate(J):
                o = np.delete(v[J], k)
                out[j] = v[j] - (o.max() if len(o) else v[j])
        return out

    feats = {n: x[:, k].astype(np.float64) for k, n in enumerate(names)}
    feats |= {f"{n}_rel": rel(v) for n, v in list(feats.items())}
    rank = f["rank"][keep]
    choice = np.asarray([right[photo == q].any() and not right[photo == q].all() for q in range(P)])
    top_wrong = np.asarray([right[photo == q].any() and not right[photo == q][np.argmin(rank[photo == q])] for q in range(P)])
    print(f"{tag}: {choice.sum()} photos with a right and a wrong area; {top_wrong.sum()} where the top area is wrong and a right one exists")
    for label, grp in (("right + wrong", choice), ("top area wrong", top_wrong)):
        sel = grp[photo]
        for n, v in feats.items():
            a, b = v[sel & right], v[sel & ~right]
            print(f"  [{label}] {n:12s} AUC {(a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean():.3f}")
    # how often the right area beats Pinpoint's top area on MegaLoc, where the top area is wrong; and the reverse where it is right
    fix = brk = 0
    n_fix = n_brk = 0
    for q in range(P):
        J = np.flatnonzero(photo == q)
        top = J[np.argmin(rank[J])]
        if top_wrong[q]:
            n_fix += 1
            fix += feats["ml_best"][J[right[J]]].max() > feats["ml_best"][top] + 0.05
        elif right[top] and len(J) > 1:
            n_brk += 1
            brk += feats["ml_best"][J[J != top]].max() > feats["ml_best"][top] + 0.05
    print(f"  right area beats the wrong top area by > 0.05: {fix} / {n_fix}; a wrong area beats the right top area by > 0.05: {brk} / {n_brk}")

    base = np.stack([np.minimum(rank, 8), rank == 0, f["source"][keep], f["support"][keep], f["mass_0.2"][keep], f["photographers"][keep]], 1)
    clue = np.stack([f[k][keep] for k in ("read_n", "read_p", "read_share", "read_rare", "recall_n", "recall_p", "recall_share", "recall_rare")], 1)
    ml = np.stack(list(feats.values()), 1)
    H = np.bincount(photo).max()
    slot = np.zeros(len(photo), int)
    seen = np.zeros(P, int)
    for j, q in enumerate(photo):
        slot[j] = seen[q]
        seen[q] += 1

    def run(X, label):
        X = ((X - X.mean(0)) / (X.std(0) + 1e-6)).astype(np.float32)
        Tt = np.zeros((P, H, X.shape[1]), np.float32)
        Tt[photo, slot] = X
        M = np.zeros((P, H), bool)
        M[photo, slot] = True
        Y = np.zeros((P, H), bool)
        Y[photo, slot] = right
        has = Y.any(1)
        hits = np.zeros(P)
        for seed in range(3):
            folds = np.random.default_rng(seed).permutation(P) % 2
            for k in range(2):
                tr, te = (folds != k) & has, folds == k
                w = torch.zeros(X.shape[1], requires_grad=True)
                Xt, Mt, Yt = torch.tensor(Tt), torch.tensor(M), torch.tensor(Y)
                opt = torch.optim.LBFGS([w], max_iter=200)

                def closure():
                    opt.zero_grad()
                    lp = torch.log_softmax((Xt[tr] @ w).masked_fill(~Mt[tr], -1e9), 1)
                    loss = -torch.logsumexp(lp.masked_fill(~Yt[tr], -1e9), 1).mean() + 1e-3 * (w ** 2).sum()
                    loss.backward()
                    return loss

                opt.step(closure)
                s = (Xt[te] @ w.detach()).masked_fill(~Mt[te], -1e9)
                hits[te] += Y[te][np.arange(te.sum()), s.argmax(1).numpy()] / 3
        print(f"  {label:34s} pick right {hits.mean():.2%}")

    print("conditional logit (< 25 km), 2-fold x 3 seeds:")
    run(base, "Pinpoint features")
    run(np.concatenate([base, clue], 1), "+ clue features")
    run(np.concatenate([base, ml], 1), "+ MegaLoc")
    run(np.concatenate([base, clue, ml], 1), "+ clue features + MegaLoc")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("embed", "gate", "features", "columns"))
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--ckpt-tag", default=CKPT_TAG, help="photo set / zoom file suffix (setft_aown: the test set with the photographer's photos kept)")
    parser.add_argument("--base", default="", help="features: ckpt-tag of the base run a variant extends (e.g. setft_own for setft_own_moreosv)")
    args = parser.parse_args(argv)
    globals()["CKPT_TAG"] = args.ckpt_tag
    globals()["BASE"] = args.base
    {"embed": embed, "gate": gate, "features": features, "columns": columns_from_gate}[args.cmd](args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
