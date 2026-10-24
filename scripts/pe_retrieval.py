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
B2 (full index): select --fraction 1 --out artifacts/retrieval/b2, then embed --out artifacts/retrieval/b2 --reuse artifacts/retrieval/b1
(scripts/b2_embed.sh). Resume uses done.npy (B1's first run used the embed.progress counter; both are read).
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
QSET = "dev3k"  # query set: dev3k (MP16) or commons26dev (rerank / recall / trained queries; --set)
SETS = {"dev3k": ("dev3k_setft_own", "artifacts/pinpoint_set/mp16_32_8_8_setft_own.npz"),
        "commons26dev": ("commons26dev_setft_own", "artifacts/pinpoint_set/commons26_32_8_8_setft_own.npz")}


def select(seed: int = 0, fraction: float = FRACTION) -> None:
    from geo_search_env.experiment.strategy_search import MP16_CSV, MP16_EMBED, _memmap_gallery
    from geo_search_env.models.pinpoint_reranker import _discover_index

    index_dir = _discover_index()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    cache = np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
    row_index = _memmap_gallery(MP16_EMBED)["row_index"]
    rows = (np.arange(len(cache)) if fraction >= 1 else
            np.sort(np.random.default_rng(seed).choice(len(cache), int(fraction * len(cache)), replace=False)))
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


def embed(reuse: str | None = None, first: tuple[str, ...] = (), only_first: bool = False, batch: int = 64, workers: int = 4) -> None:
    """first: .npy files of gallery positions to embed before the rest, in order (B2: the dev3k, then train pools' index rows);
    only_first: stop after them."""
    os.environ.setdefault("HF_HOME", str(Path("data/hf").resolve()))
    os.environ["HF_HUB_OFFLINE"] = "1"
    import pandas as pd
    import timm
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from geo_search_env.experiment.sft_data import MP16Images

    g = np.load(OUT / "gallery.npz")
    ids, g_rows = list(g["ids"]), g["rows"]
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

    path, prog, okp, donep = OUT / f"pe_g{SIZE}.npy", OUT / "embed.progress", OUT / "ok.npy", OUT / "done.npy"
    if donep.exists() or prog.exists():
        out, ok = np.lib.format.open_memmap(path, mode="r+"), np.load(okp)
        done = np.load(donep) if donep.exists() else np.arange(n) < int(prog.read_text())  # B1 ran with a counter
    else:
        out, ok, done = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, 1280)), np.zeros(n, bool), np.zeros(n, bool)
        if reuse:  # B2: copy rows another gallery (B1) already embedded
            src = Path(reuse)
            sg, sd = np.load(src / "gallery.npz")["rows"], np.load(src / "done.npy") if (src / "done.npy").exists() else None
            sd = sd if sd is not None else np.arange(len(sg)) < int((src / "embed.progress").read_text())
            pos = np.searchsorted(g_rows, sg[sd])
            assert np.array_equal(g_rows[pos], sg[sd])
            out[pos] = np.load(src / f"pe_g{SIZE}.npy", mmap_mode="r")[np.flatnonzero(sd)]
            ok[pos], done[pos] = np.load(src / "ok.npy")[sd], True
            out.flush()
            np.save(okp, ok)
            np.save(donep, done)
            print(f"  reused {sd.sum():,} photos from {src}", flush=True)
    todo = np.flatnonzero(~done)
    if first:
        pri = pd.unique(np.concatenate([np.load(f) for f in first]).astype(np.int64))
        pri = pri[~done[pri]]
        todo = pri if only_first else np.concatenate([pri, todo[~np.isin(todo, pri)]])
        print(f"  {len(pri):,} priority photos first ({', '.join(first)}){'; only those' if only_first else ''}", flush=True)
    print(f"embed {OUT}: {n:,} gallery photos at {SIZE} px; {done.sum():,} done, {len(todo):,} to go", flush=True)
    last = (len(todo) - 1) // batch
    with torch.inference_mode():
        for b, (idx, x, good) in enumerate(DataLoader(torch.utils.data.Subset(Images(), todo.tolist()), batch_size=batch,
                                                      num_workers=workers)):
            i = idx.numpy()
            out[i] = torch.nn.functional.normalize(net(x.cuda().to(torch.bfloat16)).float(), dim=-1).cpu().numpy().astype(np.float16)
            ok[i], done[i] = good.numpy(), True
            if b % 100 == 99 or b == last:  # checkpoint every 6,400 photos (~4 min)
                out.flush()
                np.save(okp, ok)
                np.save(donep, done)
                prog.write_text(str(int(done.sum())))
                print(f"  {int(done.sum()):,} / {n:,}", flush=True)
    print(f"-> {path}: {np.mean(~ok):.2%} unreadable", flush=True)


def _setup(kmax: int = 256) -> dict:
    """dev3k queries and the embedded gallery for every search, plus top-K / distance helpers (recall, trained)."""
    import torch

    from geo_search_env.experiment.strategy_search import MP16_CSV
    from geo_search_env.models.pinpoint_reranker import PinpointReranker

    g = np.load(OUT / "gallery.npz")
    ok = np.load(OUT / "ok.npy") & (np.load(OUT / "done.npy") if (OUT / "done.npy").exists()
                                    else np.arange(len(g["rows"])) < int((OUT / "embed.progress").read_text()))
    use = np.flatnonzero(ok)
    photos = json.loads(Path(f"artifacts/photo_sets/{SETS[QSET][0]}/photos.json").read_text(encoding="utf-8"))
    qids = [p["image_id"] for p in photos]
    truth = np.asarray([p["truth"] for p in photos], np.float64)
    if QSET == "dev3k":
        need = set(qids)
        qauth = {}
        with MP16_CSV.open(encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            next(r)
            for rec in r:
                if rec[0] in need:
                    qauth[rec[0]] = rec[1]
        qauth = np.asarray([qauth[i] for i in qids])
    else:  # benchmark queries carry no MP16 photographer
        qauth = np.asarray([f"query:{i}" for i in range(len(qids))])
    # queries: raw SigLIP2 from the MP16 cache, PE-G 378 px from step 5b's embed (same model and size, dev3k rows last)
    em = Path("data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384")
    m = json.loads((em / "manifest.json").read_text(encoding="utf-8"))
    pos = {k: i for i, k in enumerate((em / m["files"]["image_ids"]).read_text(encoding="utf-8").splitlines())}
    sig = np.memmap(em / m["files"]["embeddings"], dtype=np.float16, mode="r", shape=tuple(m["shapes"]["embeddings"]))
    if QSET == "dev3k":
        q_sig = np.asarray(sig[np.asarray([pos[i] for i in qids])], np.float32)
    else:
        from geo_search_env.data.benchmarks import load_benchmark

        q_sig = np.asarray(load_benchmark("commons26").embeddings, np.float32)[np.asarray([e["index"] for e in photos])]
    if QSET != "dev3k":  # commons26 dev queries at 378 px from pe_probe embed_pooled
        assert SIZE == 378
        extra = np.load("artifacts/retrieval/probe/pooled_extra.npz")
        at = {k: i for i, k in enumerate(extra["keys"])}
        q_pe = extra["embeddings"][np.asarray([at[f"commons26dev:{i}"] for i in range(len(qids))])].astype(np.float32)
    elif SIZE == 378:
        probe = Path("artifacts/retrieval/probe")
        n_p = len(np.load(probe / "photos.npz")["ids"])
        q_pe = np.asarray(np.load(probe / "pooled.npy", mmap_mode="r")[n_p:n_p + len(qids)], np.float32)
    else:  # dev3k queries embedded at SIZE by `embed --out artifacts/retrieval/q{SIZE}` (gallery.npz holding the dev3k ids in order)
        qd = OUT.parent / f"q{SIZE}"
        assert list(np.load(qd / "gallery.npz")["ids"]) == qids
        q_pe = np.load(qd / f"pe_g{SIZE}.npy").astype(np.float32)
    rr = PinpointReranker()
    raw_q, con_q = rr.encode_queries(q_sig)
    rows = g["rows"][use]
    dev = rr.device
    galleries = {"siglip2": rr._dequantize(0, rows=rows), "contrastive": torch.as_tensor(np.asarray(rr.image[rows]), device=dev),
                 "pe_g": torch.as_tensor(np.asarray(np.load(OUT / f"pe_g{SIZE}.npy", mmap_mode="r")[use]), device=dev)}
    queries = {"siglip2": raw_q.half(), "contrastive": con_q.half(),
               "pe_g": torch.nn.functional.normalize(torch.as_tensor(q_pe, device=dev), dim=-1).half()}
    glat = g["latlon"][use].astype(np.float64)
    same = qauth[:, None] == g["authors"][use][None, :]  # [Q, G] the photographer's own photos (original protocol keeps them)

    def top(name: str, exclude_own: bool, cols: np.ndarray | None = None) -> np.ndarray:
        """Top-kmax gallery positions (into use) per query; cols restricts the gallery to those positions."""
        gal = galleries[name] if cols is None else galleries[name][torch.as_tensor(cols, device=dev)]
        s = (queries[name] @ gal.T).float()
        if exclude_own:
            s.masked_fill_(torch.as_tensor(same if cols is None else same[:, cols], device=dev), -9.0)
        i = s.topk(kmax, dim=1).indices.cpu().numpy()
        return i if cols is None else cols[i]

    def dist(idx: np.ndarray) -> np.ndarray:
        a, b = np.radians(truth[:, None, :]), np.radians(glat[idx])
        h = np.sin((b[..., 0] - a[..., 0]) / 2) ** 2 + np.cos(a[..., 0]) * np.cos(b[..., 0]) * np.sin((b[..., 1] - a[..., 1]) / 2) ** 2
        return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

    return dict(g=g, ok=ok, use=use, qids=qids, qauth=qauth, rr=rr, dev=dev, galleries=galleries, queries=queries, glat=glat,
                top=top, dist=dist, raw_q=raw_q, con_q=con_q)


def _report(d: dict, ks: tuple[int, ...], radii: tuple[float, ...], pairs: list[tuple[str, str]], new: str) -> None:
    """d: search -> [Q, kmax] km of each top-K photo. Recall@K per search, K/2 + K/2 unions, and what `new` adds to contrastive's misses."""
    for k in ks:
        arms = {name: d[name][:, :k].min(1) for name in d}
        h = k // 2
        for x, y in pairs:
            arms[f"{x} + {y} (K/2 each)"] = np.minimum(d[x][:, :h].min(1), d[y][:, :h].min(1))
        for name, v in arms.items():
            print(f"  K={k:3d} {name:40s} " + " / ".join(f"{100 * np.mean(v < r):5.1f}" for r in radii), flush=True)
        miss = arms["contrastive"] >= 25
        print(f"  K={k:3d} of the {miss.sum()} photos the contrastive top-K misses at 25 km, {new}'s top-K finds one for "
              f"{100 * np.mean(arms[new][miss] < 25):.1f}% ({100 * np.mean(miss & (arms[new] < 25)):.1f} pts of all photos)", flush=True)


def recall(ks: tuple[int, ...] = (8, 32, 256), radii: tuple[float, ...] = (1.0, 25.0, 200.0)) -> None:
    S = _setup(max(ks))
    print(f"b1 recall: {len(S['use']):,} of {len(S['ok']):,} gallery photos embedded; dev3k {len(S['qids'])} queries; % with a gallery photo "
          f"within {' / '.join(f'{r:g}' for r in radii)} km among the top K", flush=True)
    for exclude_own in (False, True):
        print(f"\n{'photographer excluded' if exclude_own else 'original protocol (photographer kept)'}:", flush=True)
        d = {name: S["dist"](S["top"](name, exclude_own)) for name in S["galleries"]}
        _report(d, ks, radii, [("contrastive", "pe_g"), ("siglip2", "pe_g"), ("contrastive", "siglip2")], "pe_g")


def trained(epochs: int = 20, batch: int = 4096, dim: int = 256, cell_deg: float = 0.2, ks: tuple[int, ...] = (8, 32, 256),
            radii: tuple[float, ...] = (1.0, 25.0, 200.0), seed: int = 0, backbones: tuple[str, ...] = ("pe_g", "siglip2"),
            train_frac: float = 1.0, eval_fracs: tuple[float, ...] = (1.0,), tag: str = "") -> None:
    """B1 fallback (PE_PLAN, Oct 22 decisions): is PE-G a better *trainable* search backbone than SigLIP2? The same small projection
    (MLP -> 256-d, InfoNCE) trained on each backbone's raw embedding of one half of the gallery (photographers split by hash; dev3k
    photographers kept out of training), positives by distance only (another photo in the same ~22 km cell), batch pairs within 25 km
    masked as negatives. Evaluated on the other half with dev3k queries, next to Pinpoint's contrastive search (trained on all of MP16).
    backbones: pe_g / siglip2 / concat (both raw embeddings side by side). train_frac: a random share of the train half's photos, trained
    for the same number of steps. eval_fracs: nested random shares of the eval half (gallery density), K = 32 at 25 km for each."""
    import hashlib

    import torch
    import torch.nn.functional as F

    S = _setup(max(ks))
    g, use, dev = S["g"], S["use"], S["dev"]
    auth = g["authors"][use]
    half = np.asarray([hashlib.md5(a.encode()).digest()[0] & 1 for a in auth], bool)
    tr = np.flatnonzero(half & ~np.isin(auth, np.unique(S["qauth"])))
    ev = np.flatnonzero(~half)
    if train_frac < 1:
        tr = np.sort(np.random.default_rng(seed + 1).choice(tr, int(train_frac * len(tr)), replace=False))
        epochs = round(epochs / train_frac)
    if "concat" in backbones:
        S["galleries"]["concat"] = torch.cat([S["galleries"]["siglip2"], S["galleries"]["pe_g"]], 1)
        S["queries"]["concat"] = torch.cat([S["queries"]["siglip2"], S["queries"]["pe_g"]], 1)
    lat = S["glat"]
    cell = np.floor(lat[:, 0] / cell_deg).astype(np.int64) * 100000 + np.floor(
        lat[:, 1] * np.cos(np.radians(lat[:, 0])) / cell_deg).astype(np.int64)
    _, inv, cnt = np.unique(cell[tr], return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    anchors = np.flatnonzero(cnt[inv] > 1)  # positions in tr with a same-cell partner
    print(f"trained: train half {len(tr):,} photos ({len(anchors):,} anchors with a same-cell positive), eval half {len(ev):,}", flush=True)
    rng = np.random.default_rng(seed)
    lat_t = torch.as_tensor(np.radians(lat), device=dev, dtype=torch.float32)

    def km(i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        a, b = lat_t[i][:, None], lat_t[j][None]
        h = torch.sin((b[..., 0] - a[..., 0]) / 2) ** 2 + torch.cos(a[..., 0]) * torch.cos(b[..., 0]) * torch.sin((b[..., 1] - a[..., 1]) / 2) ** 2
        return 2 * 6371.0 * torch.asin(h.clamp(0, 1).sqrt())

    out = {}
    for name in backbones:
        torch.manual_seed(seed)
        X = S["galleries"][name].float()
        head = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 2048), torch.nn.GELU(), torch.nn.Linear(2048, dim)).to(dev)
        logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / 0.07), device=dev, dtype=torch.float32))
        opt = torch.optim.AdamW([*head.parameters(), logit_scale], lr=1e-3, weight_decay=0.01)
        steps = epochs * (len(anchors) // batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
        for ep in range(epochs):
            perm = rng.permutation(anchors)
            tot = 0.0
            for b0 in range(0, len(perm) - batch + 1, batch):
                a_pos = perm[b0:b0 + batch]
                c = inv[a_pos]
                p_pos = order[starts[c] + rng.integers(0, cnt[c])]
                same = p_pos == a_pos  # redraw once; a leftover self-pair is still a valid (trivial) positive
                p_pos[same] = order[starts[c[same]] + rng.integers(0, cnt[c[same]])]
                ia, ip = torch.as_tensor(tr[a_pos], device=dev), torch.as_tensor(tr[p_pos], device=dev)
                za, zp = F.normalize(head(X[ia]), dim=-1), F.normalize(head(X[ip]), dim=-1)
                logits = logit_scale.exp().clamp(max=100) * za @ zp.T
                near = (km(ia, ip) < 25) & ~torch.eye(len(ia), dtype=torch.bool, device=dev)
                logits = logits.masked_fill(near, -1e4)
                y = torch.arange(len(ia), device=dev)
                loss = (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item()
            print(f"  {name} epoch {ep + 1}: loss {tot / max(1, len(perm) // batch):.3f}", flush=True)
        with torch.no_grad():
            S["galleries"][f"{name}_trained"] = torch.cat([F.normalize(head(X[i:i + 65536]), dim=-1) for i in range(0, len(X), 65536)]).half()
            S["queries"][f"{name}_trained"] = F.normalize(head(S["queries"][name].float()), dim=-1).half()
        out[name] = head
    names = ["contrastive", "siglip2", "pe_g"] + [f"{b}_trained" for b in backbones]
    new = "pe_g_trained" if "pe_g" in backbones else names[-1]
    if len(eval_fracs) > 1 or eval_fracs[0] < 1:
        ev_order = np.random.default_rng(seed + 2).permutation(ev)
        print(f"\ngallery density (share of the index = 0.05 x fraction of the eval half): % within 25 km among the top 32", flush=True)
        for exclude_own in (False, True):
            print(f"  {'photographer excluded' if exclude_own else 'original protocol'}:", flush=True)
            for f in eval_fracs:
                cols = np.sort(ev_order[:int(f * len(ev))])
                arms = {name: S["dist"](S["top"](name, exclude_own, cols=cols))[:, :32].min(1) for name in names}
                print(f"    {100 * 0.05 * f:5.2f}% of the index ({len(cols):,} photos): " +
                      "  ".join(f"{name} {100 * np.mean(v < 25):.1f}" for name, v in arms.items()), flush=True)
    print(f"\nb1 trained: eval half {len(ev):,} photos; train {len(tr):,} photos; dev3k {len(S['qids'])} queries; % with a gallery photo "
          f"within {' / '.join(f'{r:g}' for r in radii)} km among the top K", flush=True)
    for exclude_own in (False, True):
        print(f"\n{'photographer excluded' if exclude_own else 'original protocol (photographer kept)'}:", flush=True)
        d = {name: S["dist"](S["top"](name, exclude_own, cols=ev)) for name in names}
        _report(d, ks, radii, [("contrastive", n) for n in names[3:]] + [("contrastive", "pe_g")], new)
    torch.save({k: v.state_dict() for k, v in out.items()}, OUT / f"trained_heads{tag}.pt")


def rerank(ks: tuple[int, ...] = (8, 16), names: tuple[str, ...] = ("contrastive", "siglip2_trained", "pe_g_trained"), combo: bool = False,
           fixed: tuple[float, ...] = ()) -> None:
    """Does the trained PE-G search's recall reach the reranker's top-1? dev3k's setft_own pool (the pipeline's, photographer kept), plus each
    search's top K from this gallery (new photos only), all rescored by the setft_own reranker (run with PINPOINT_RERANKER_CKPT set to it).
    Added photos get rank 1 / not-in-shortlist features, as the wide lists' extra hits it was trained on. Top-1 = best score over the pool.
    combo: also pick by reranker score + lambda x the query's PE-G cosine to each candidate (raw, and in the trained projection), lambda by
    2-fold cross-validation on dev3k (folds by photo parity); candidate PE-G embeddings from B2's file (the dev3k pools embedded first).
    fixed: also report these lambdas as chosen elsewhere (commons26 is final validation only: lambda from dev3k, no tuning on its dev)."""
    import torch
    import torch.nn.functional as F

    S = _setup(max(ks))
    rr, dev = S["rr"], S["dev"]
    heads, mods = torch.load(OUT / "trained_heads.pt"), {}
    for name in ("pe_g", "siglip2"):
        X = S["galleries"][name].float()
        head = mods[name] = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 2048), torch.nn.GELU(), torch.nn.Linear(2048, 256)).to(dev)
        head.load_state_dict(heads[name])
        with torch.no_grad():
            S["galleries"][f"{name}_trained"] = torch.cat([F.normalize(head(X[i:i + 65536]), dim=-1) for i in range(0, len(X), 65536)]).half()
            S["queries"][f"{name}_trained"] = F.normalize(head(S["queries"][name].float()), dim=-1).half()
    photos = json.loads(Path(f"artifacts/photo_sets/{SETS[QSET][0]}/photos.json").read_text(encoding="utf-8"))
    idx = np.asarray([e["index"] for e in photos])
    z = np.load(SETS[QSET][1])
    rows0, feats0, scores0 = z["rows"][idx], z["features"][idx].astype(np.float32), z["scores"][idx]
    truth = np.asarray([e["truth"] for e in photos], np.float64)
    raw, con = S["raw_q"], S["con_q"]

    def km(rows: np.ndarray, pick: np.ndarray) -> np.ndarray:
        ll = np.asarray(rr.latlon[rows[np.arange(len(rows)), pick]], np.float64)
        a, b = np.radians(truth), np.radians(ll)
        h = np.sin((b[:, 0] - a[:, 0]) / 2) ** 2 + np.cos(a[:, 0]) * np.cos(b[:, 0]) * np.sin((b[:, 1] - a[:, 1]) / 2) ** 2
        return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

    def rescore(rows, ranks, flags):
        s, _ = rr._score(raw, con, rows, ranks, flags)
        return s

    s_base = rescore(rows0, feats0[..., 3:6], feats0[..., 6:9])
    v0 = rows0 >= 0
    agree = np.mean(np.where(v0, s_base, -np.inf).argmax(1) == np.where(v0, scores0, -np.inf).argmax(1))
    print(f"rerank: {QSET} {len(idx)} photos, pool {v0.sum(1).mean():.1f} candidates; rescoring the stored pool reproduces its top-1 for "
          f"{100 * agree:.1f}% (max |score diff| {np.nanmax(np.abs(np.where(v0, s_base - scores0, 0))):.3f})", flush=True)
    base = km(rows0, np.where(v0, s_base, -np.inf).argmax(1))
    print(f"  reranker top-1, today's pool: 25 km {100 * np.mean(base < 25):.2f}  1 km {100 * np.mean(base < 1):.2f}", flush=True)
    if combo:
        b2 = Path("artifacts/retrieval/b2")
        pe_all, pe_done = np.load(b2 / "pe_g378.npy", mmap_mode="r"), np.load(b2 / "done.npy")
        q_raw = S["queries"]["pe_g"].float()
        q_tr = S["queries"]["pe_g_trained"].float()
        q_sig_tr = S["queries"]["siglip2_trained"].float()
        lams = np.asarray([0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128])
        fold = np.arange(len(idx)) % 2

        def combos(label: str, rows: np.ndarray, s: np.ndarray) -> None:
            v = rows >= 0
            have = pe_done[np.where(v, rows, 0)] & v
            u, inv = np.unique(np.where(v, rows, 0), return_inverse=True)
            e = torch.as_tensor(np.asarray(pe_all[u]), device=dev).float()
            with torch.no_grad():
                e_tr = F.normalize(mods["pe_g"](e), dim=-1)
            inv = torch.as_tensor(inv.reshape(rows.shape), device=dev)
            with torch.no_grad():  # control: the same projection trained on SigLIP2 (index vectors exist for every row)
                e_sig = F.normalize(mods["siglip2"](rr._dequantize(0, rows=u).float()), dim=-1)
            cos = {"raw": (q_raw[:, None] * e[inv]).sum(-1).cpu().numpy(), "trained": (q_tr[:, None] * e_tr[inv]).sum(-1).cpu().numpy(),
                   "siglip2 trained": (q_sig_tr[:, None] * e_sig[inv]).sum(-1).cpu().numpy()}
            cos["trained + siglip2 trained"] = cos["trained"] + cos["siglip2 trained"]
            out, lam_of = [], {}
            for kind, c in cos.items():
                if "raw" in kind or kind.startswith("trained"):  # a candidate without PE-G: the pool mean
                    c = np.where(have, c, np.nanmean(np.where(have, c, np.nan), 1, keepdims=True))
                d = np.stack([km(rows, np.where(v, s + lam * c, -np.inf).argmax(1)) for lam in lams])  # [L, Q]
                pick, chosen = np.zeros(len(idx)), []
                for f in (0, 1):
                    j = int(np.argmax((d[:, fold != f] < 25).mean(1)))
                    chosen.append(lams[j])
                    pick[fold == f] = d[j, fold == f]
                for lam in fixed:
                    j = int(np.flatnonzero(lams == lam)[0])
                    f25, f1 = (d[j] < 25).astype(float) - (base < 25), (d[j] < 1).astype(float) - (base < 1)
                    out.append(f"{kind} cos (fixed lambda {lam:g}): 25 km {100 * np.mean(d[j] < 25):.2f} ({100 * f25.mean():+.2f} +- "
                               f"{100 * f25.std() / np.sqrt(len(idx)):.2f})  1 km {100 * np.mean(d[j] < 1):.2f} ({100 * f1.mean():+.2f})")
                lam_of[kind] = chosen
                diff25, diff1 = (pick < 25).astype(float) - (base < 25), (pick < 1).astype(float) - (base < 1)
                out.append(f"{kind} cos (lambda {chosen[0]:g}/{chosen[1]:g}): 25 km {100 * np.mean(pick < 25):.2f} ({100 * diff25.mean():+.2f} +- "
                           f"{100 * diff25.std() / np.sqrt(len(idx)):.2f})  1 km {100 * np.mean(pick < 1):.2f} ({100 * diff1.mean():+.2f})")
            print(f"    combo {label}: PE-G for {100 * have[v].mean():.1f}% of candidates; " + ";  ".join(out), flush=True)
            if label == "today's pool":  # for the fix analysis (scripts/pe_fixes.py): what the trained similarity changes, and why
                feats = feats0 if rows.shape == rows0.shape else None
                np.savez(OUT / f"combo_dump_{QSET}.npz", rows=rows, score=s, base_km=base, truth=truth, fold=fold,
                         latlon=np.asarray(rr.latlon[np.where(v, rows, 0)]), cos_raw=cos["raw"], cos_trained=cos["trained"],
                         cos_siglip2_trained=cos["siglip2 trained"], lam_trained=np.asarray(lam_of["trained"]),
                         sim_contrastive=feats[..., 0], sim_siglip2=feats[..., 1], sim_gps=feats[..., 2])
                print(f"    -> {OUT / f'combo_dump_{QSET}.npz'}", flush=True)

        combos("today's pool", rows0, s_base)
    grow = S["g"]["rows"][S["use"]]
    for name in names:
        top = S["top"](name, False)  # photographer kept, as the pool
        for k in ks:
            add = grow[top[:, :k]]
            add = np.where(np.asarray([np.isin(a, r) for a, r in zip(add, rows0)]), -1, add)  # new photos only
            rows = np.concatenate([rows0, add], 1)
            ranks = np.concatenate([feats0[..., 3:6], np.ones(add.shape + (3,), np.float32)], 1)
            flags = np.concatenate([feats0[..., 6:9], np.zeros(add.shape + (3,), np.float32)], 1)
            s = rescore(rows, ranks, flags)
            pick = np.where(rows >= 0, s, -np.inf).argmax(1)
            d = km(rows, pick)
            new_pick = np.mean(pick >= rows0.shape[1])
            parts = []
            for t in (25, 1):
                diff = (d < t).astype(float) - (base < t)
                parts.append(f"{t} km {100 * np.mean(d < t):.2f} ({100 * diff.mean():+.2f} +- {100 * diff.std() / np.sqrt(len(diff)):.2f})")
            print(f"  + {name:16s} top {k:2d} ({(add >= 0).sum(1).mean():.1f} new): " + "  ".join(parts) +
                  f"  top-1 is an added photo for {100 * new_pick:.1f}%", flush=True)
            if combo:
                combos(f"+ {name} top {k}", rows, s)


def main() -> None:
    global OUT, SIZE, QSET
    ap = argparse.ArgumentParser()
    ap.add_argument("node", choices=("select", "embed", "recall", "trained", "rerank"))
    ap.add_argument("--out", default=str(OUT), help="gallery directory (B1 artifacts/retrieval/b1, B2 artifacts/retrieval/b2)")
    ap.add_argument("--fraction", type=float, default=FRACTION, help="select: share of the index (1 = all, B2)")
    ap.add_argument("--reuse", default=None, help="embed: copy rows another gallery already embedded (B2: artifacts/retrieval/b1)")
    ap.add_argument("--first", default="", help="embed: comma list of .npy gallery positions to embed first (B2 priority)")
    ap.add_argument("--only-first", action="store_true", help="embed: stop after the --first photos")
    ap.add_argument("--size", type=int, default=SIZE, help="PE-G input resolution (files pe_g{size}.npy; queries from q{size}/ unless 378)")
    ap.add_argument("--backbones", default="pe_g,siglip2", help="trained: comma list of pe_g / siglip2 / concat")
    ap.add_argument("--train-frac", type=float, default=1.0, help="trained: share of the train half (same number of steps)")
    ap.add_argument("--eval-fracs", default="1", help="trained: comma list of nested shares of the eval half (gallery density)")
    ap.add_argument("--set", default="dev3k", choices=tuple(SETS), help="query set (rerank, recall, trained)")
    ap.add_argument("--fixed", default="", help="rerank --combo: comma list of lambdas chosen elsewhere (e.g. on dev3k)")
    ap.add_argument("--combo", action="store_true", help="rerank: also reranker score + lambda x PE-G cosine (2-fold lambda)")
    ap.add_argument("--names", default="contrastive,siglip2_trained,pe_g_trained", help="rerank: searches whose top K are added")
    ap.add_argument("--tag", default="", help="trained: suffix of the saved heads file")
    args = ap.parse_args()
    OUT, SIZE, QSET = Path(args.out), args.size, args.set
    tr = lambda: trained(backbones=tuple(args.backbones.split(",")), train_frac=args.train_frac,  # noqa: E731
                         eval_fracs=tuple(float(f) for f in args.eval_fracs.split(",")), tag=args.tag)
    first = tuple(f for f in args.first.split(",") if f)
    {"select": lambda: select(fraction=args.fraction), "embed": lambda: embed(reuse=args.reuse, first=first, only_first=args.only_first), "recall": recall, "trained": tr, "rerank": lambda: rerank(names=tuple(args.names.split(",")), combo=args.combo,
                                                                   fixed=tuple(float(x) for x in args.fixed.split(",") if x))}[args.node]()


if __name__ == "__main__":
    main()
