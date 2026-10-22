"""A second image model on the candidate areas (VALUE_PLAN.md idea 2; user, 2026-10-22). Every Pinpoint score comes from SigLIP2 (plus MegaLoc's
same-place columns, lesson 93, the largest evidence gain since the retrained reranker). Meta's Perception Encoder (PE-Core, 5.4B image-text
pairs, apache-2.0; timm `vit_pe_core_{gigantic_patch14_448,large_patch14_336}.fb`) learned from other data: does it know which candidate
area is right where SigLIP2 and MegaLoc don't? Also DINOv3 (self-supervised on 1.7B images; ViT-7B and its distilled ViT-L; gated: HF_TOKEN).

embed (.venvs/sft, timm on PYTHONPATH): the dev3k queries and every photo of their areas' restricted searches (the zoom file's 12 per area,
        as megaloc_gate), unit-norm embeddings -> artifacts/value/pe_{model}_{tag}_{ckpt}.npy (+ _keys.npy); resumable.
gate (.venv): per area the best / top-3 cosine and both relative to the photo's other areas (megaloc_gate's columns). The answer head of
        the current pipeline (action_agent, --one-km 0.5, trained on all train, seed 7) with every tool revealed gives a logit per answer
        option (area x point); a conditional logit over the options on dev3k (2-fold x 3 seeds) with [head logit] vs [head logit + PE
        columns]; control: [head logit + MegaLoc columns], already inside the head, should add ~0. Pre-declared bar: PE adds >= +1.0 pt
        < 25 km over head-only, < 1 km not down by more than 0.3, and the MegaLoc control within +-0.3.

  PYTHONPATH=<timm>:src .venvs/sft/bin/python scripts/second_model_gate.py embed --model pe_g
  PYTHONPATH=src .venv/bin/python scripts/second_model_gate.py gate --models pe_g pe_l
head (after embedding train with --tag train --photos artifacts/value/train_informative_setft_own.npy): see head().
  PYTHONPATH=src .venv/bin/python scripts/second_model_gate.py head --models pe_g dino_7b
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np

OUT = Path("artifacts/value")
MODELS = {"pe_g": "vit_pe_core_gigantic_patch14_448.fb", "pe_l": "vit_pe_core_large_patch14_336.fb",
          "dino_7b": "facebook/dinov3-vit7b16-pretrain-lvd1689m", "dino_l": "facebook/dinov3-vitl16-pretrain-lvd1689m"}
DINO_SIZE = 224  # the DINOv3 processor's default (square resize, ImageNet normalisation); its pooled CLS token is the embedding


def _paths(model: str, tag: str, ckpt: str) -> tuple[Path, Path]:
    return OUT / f"{model}_{tag}_{ckpt}.npy", OUT / f"{model}_{tag}_{ckpt}_keys.npz"


def embed(model: str, tag: str, ckpt: str, batch: int = 64, workers: int = 4, photos: str = "") -> None:
    """photos: a boolean .npy over the photo set; only those queries and their areas' photos are embedded (train: the photos whose full-view
    options are neither all right nor all wrong at 25 / 1 km, the only ones that move the answer head's loss)."""

    os.environ.setdefault("HF_HOME", str(Path("data/hf").resolve()))
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from geo_search_env.experiment import megaloc_gate
    from geo_search_env.experiment.sft_data import MP16Images

    megaloc_gate.CKPT_TAG = ckpt
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{ckpt}.npz")
    rows = zm["rows"]
    keep = np.isin(zm["photo"], np.flatnonzero(np.load(photos))) if photos else np.ones(len(rows), bool)
    need = np.unique(rows[keep][rows[keep] >= 0])
    ids, nq, _ = megaloc_gate._image_ids(tag, need)  # queries first, then the area photos in index-row order
    if model.startswith("pe_"):
        import timm

        net = timm.create_model(MODELS[model], pretrained=True).cuda().eval().to(torch.bfloat16)
        cfg = timm.data.resolve_data_config(model=net)
        tf = timm.data.create_transform(**cfg)
        size = cfg["input_size"][1]
    else:  # DINOv3 (gated on the Hub: HF_TOKEN)
        from torchvision import transforms as T
        from transformers import AutoModel

        dino = AutoModel.from_pretrained(MODELS[model], dtype=torch.bfloat16).cuda().eval()
        net = lambda x: dino(pixel_values=x).pooler_output
        size = DINO_SIZE
        tf = T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    class Images(Dataset):
        images = None

        def __len__(self) -> int:
            return len(ids)

        def __getitem__(self, i: int):
            if self.images is None:
                self.images = MP16Images()
            try:
                key = ids[i]
                data = Path(key).read_bytes() if key.startswith("data/pinpoint/") or key.startswith("/") else self.images.read(key)
                return i, tf(Image.open(io.BytesIO(data)).convert("RGB")), True
            except Exception:  # noqa: BLE001
                return i, torch.zeros(3, size, size), False

    OUT.mkdir(parents=True, exist_ok=True)
    emb_path, key_path = _paths(model, tag, ckpt)
    prog = emb_path.with_suffix(".progress")
    with torch.inference_mode():
        dim = net(torch.zeros(1, 3, size, size, device="cuda", dtype=torch.bfloat16)).shape[1]
    if prog.exists():
        out = np.lib.format.open_memmap(emb_path, mode="r+")
        ok = np.load(key_path)["ok"]
        start = int(prog.read_text())
    else:
        out = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16, shape=(len(ids), dim))
        ok = np.zeros(len(ids), bool)
        start = 0
        np.savez(key_path, n_queries=nq, rows=need, ok=ok)
    print(f"{model} {tag}: {nq} queries + {len(need)} area photos, dim {dim}; starting at {start}", flush=True)
    order = np.concatenate([np.flatnonzero(np.load(photos)) if photos else np.arange(nq), nq + np.arange(len(need))])  # unselected queries stay 0
    sub = torch.utils.data.Subset(Images(), order[start:].tolist())
    done = start
    with torch.inference_mode():
        for b, (idx, x, good) in enumerate(DataLoader(sub, batch_size=batch, num_workers=workers)):
            e = torch.nn.functional.normalize(net(x.cuda().to(torch.bfloat16)).float(), dim=-1)
            out[idx.numpy()] = e.cpu().numpy().astype(np.float16)
            ok[idx.numpy()] = good.numpy()
            done += len(idx)
            if b % 200 == 199 or done == len(order):
                out.flush()
                np.savez(key_path, n_queries=nq, rows=need, ok=ok)
                prog.write_text(str(done))
                print(f"  {done:,} / {len(order):,}", flush=True)
    print(f"-> {emb_path}: {np.mean(~ok):.2%} unreadable", flush=True)


def columns(model: str, tag: str, ckpt: str, zm, per_area: int = 12) -> np.ndarray:
    """Per zoom-file hypothesis: best, top-3 mean, and both minus the best of the photo's other non-street areas (megaloc_gate._write_columns);
    per_area: only each area's first photos (Pinpoint's restricted-search order), to price a cheaper train-side embedding."""

    emb_path, key_path = _paths(model, tag, ckpt)
    k = np.load(key_path)
    e = np.load(emb_path, mmap_mode="r")
    nq, need, ok = int(k["n_queries"]), k["rows"], k["ok"]
    rows, photo = zm["rows"], zm["photo"]
    q = np.asarray(e[:nq], np.float32)
    sims = np.full(rows.shape, np.nan, np.float32)
    for h in range(len(photo)):
        v = np.flatnonzero(rows[h, :per_area] >= 0)
        if len(v):
            pos = nq + np.searchsorted(need, rows[h, v])
            s = np.asarray(e[pos], np.float32) @ q[photo[h]]
            sims[h, v] = np.where(ok[pos], s, np.nan)
    srt = np.sort(np.where(np.isnan(sims), -np.inf, sims), 1)[:, -3:]
    best = np.where(np.isfinite(srt[:, -1]), srt[:, -1], -1.0)
    with np.errstate(invalid="ignore"):
        top3 = np.nan_to_num(np.nanmean(np.where(np.isfinite(srt), srt, np.nan), 1), nan=-1.0)
    street = zm["source"] == 2
    cols = [best, top3]
    for v in (best, top3):
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


def head_options(tag: str, ckpt: str):
    """The current pipeline's answer head (action_agent, --one-km 0.5, all train, seed 7) with read + recognise + every area zoomed: per photo
    the 2h option logits, option km, and each option's zoom-file hypothesis index."""

    import torch

    from geo_search_env.experiment import action_agent as aa
    from geo_search_env.experiment import hypothesis_agent as ha

    aa.CKPT_TAG, aa.ONE_KM = ckpt, 0.5
    tr = aa._data("train", ckpt=ckpt)
    dv = aa._data(tag, stats=tr["stats"], ckpt=ckpt)
    net = aa.fit_answer(tr, np.arange(tr["n"]), seed=7)
    n, h = dv["mask"].shape
    on = torch.ones(n, dtype=torch.bool, device="cuda")
    v = aa.view(dv, on, on)
    zoomed = v["mask"].clone()
    with torch.no_grad():
        logits = aa._options(net, v, zoomed).float().cpu().numpy()
    km = torch.cat([v["km_unzoom"], v["km_zoom"]], 1).cpu().numpy()
    # padded slot -> zoom-file hypothesis (load(): rows = the kept sources in file order, slots in order per photo)
    f = np.load(ha._paths(tag, ckpt)[0])
    rows = np.flatnonzero(np.isin(f["source"], ha.SOURCES))
    hyp = np.full((n, h), -1)
    seen = np.zeros(n, int)
    for r in rows:
        p = int(f["photo"][r])
        hyp[p, seen[p]] = r
        seen[p] += 1
    assert (hyp >= 0).sum() == int(dv["mask"].sum().item())
    return logits, km, np.concatenate([hyp, hyp], 1)


def clogit(X: np.ndarray, valid: np.ndarray, km: np.ndarray, seeds: int = 3) -> np.ndarray:
    """Conditional logit over options, 2-fold x seeds on photos; loss as the head's (< 25 km + 0.5 x < 1 km). Returns the chosen km per photo
    for each seed (seeds, n)."""

    import torch

    X = np.where(valid[..., None], X, 0.0)
    mu, sd = X[valid].mean(0), X[valid].std(0) + 1e-6
    Xt = torch.tensor(np.where(valid[..., None], (X - mu) / sd, 0.0), dtype=torch.float32)
    Mt = torch.tensor(valid)
    out = np.zeros((seeds, len(X)))
    for seed in range(seeds):
        folds = np.random.default_rng(seed).permutation(len(X)) % 2
        for k in range(2):
            tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            w = torch.zeros(X.shape[-1], requires_grad=True)
            w.data[0] = 1.0
            opt = torch.optim.LBFGS([w], max_iter=300)

            def closure():
                opt.zero_grad()
                lp = torch.log_softmax((Xt[tr] @ w).masked_fill(~Mt[tr], -1e9), 1)
                loss = 1e-3 * (w ** 2).sum()
                for r, wt in ((25.0, 1.0), (1.0, 0.5)):
                    ok = torch.tensor((km[tr] < r) & valid[tr])
                    use = ok.any(1)
                    if use.any():
                        loss = loss - wt * torch.logsumexp(lp.masked_fill(~ok, -1e9), 1)[use].mean()
                loss.backward()
                return loss

            opt.step(closure)
            s = (Xt[te] @ w.detach()).masked_fill(~Mt[te], -1e9).numpy()
            out[seed, te] = km[te, s.argmax(1)]
    return out


def gate(models: list[str], tag: str, ckpt: str, ks: tuple[int, ...] = (12,)) -> None:
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{ckpt}.npz")
    logits, km, hyp = head_options(tag, ckpt)
    valid = np.isfinite(logits)
    head_km = km[np.arange(len(km)), np.where(valid, logits, -np.inf).argmax(1)]
    print(f"\n{tag}: head argmax (read + recognise + zoom all) {np.mean(head_km < 25):.1%} < 25 km, {np.mean(head_km < 1):.1%} < 1 km; "
          f"option oracle {np.mean((km < 25).any(1)):.1%}", flush=True)
    groups = {"MegaLoc (control)": np.load(Path("artifacts/agent") / f"megaloc_{tag}_{ckpt}.npz")["x"]}
    groups |= {m + ("" if k == 12 else f" @{k}"): columns(m, tag, ckpt, zm, k) for m in models if _paths(m, tag, ckpt)[0].exists() for k in ks}
    if len(models) > 1:  # all models together
        groups["+".join(models)] = np.concatenate([columns(m, tag, ckpt, zm) for m in models if _paths(m, tag, ckpt)[0].exists()], 1)
    # per area: does the new model separate right from wrong areas where the head's pick is wrong and a right area exists?
    a_right = np.zeros(len(zm["photo"]), bool)
    for p in range(len(km)):
        for j in range(km.shape[1] // 2):
            if hyp[p, j] >= 0:
                a_right[hyp[p, j]] = min(km[p, j], km[p, j + km.shape[1] // 2]) < 25
    base = clogit(np.where(valid, logits, 0.0)[..., None], valid, km)
    print(f"  stacked head only: {np.mean(base < 25):.2%} / {np.mean(base < 1):.2%} (< 25 / 1 km, 2-fold x 3 seeds)")
    for name, cols in groups.items():
        hw = (head_km >= 25) & (km < 25).any(1)
        sel = np.isin(zm["photo"], np.flatnonzero(hw)) & np.isin(np.arange(len(zm["photo"])), hyp[hyp >= 0])
        a, b = cols[sel & a_right, 0], cols[sel & ~a_right, 0]
        auc = (a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()
        X = np.concatenate([np.where(valid, logits, 0.0)[..., None], np.where((hyp >= 0)[..., None], cols[np.maximum(hyp, 0)], 0.0)], -1)
        res = clogit(X, valid, km)
        diff = (res < 25).mean(0) - (base < 25).mean(0)  # per photo, averaged over seeds
        se = diff.std() / np.sqrt(len(diff))
        print(f"  + {name:18s}: {np.mean(res < 25):.2%} / {np.mean(res < 1):.2%}; vs head only {100 * np.mean((res < 25).mean(0) - (base < 25).mean(0)):+.2f} "
              f"/ {100 * np.mean((res < 1).mean(0) - (base < 1).mean(0)):+.2f} pts (paired SE {100 * se:.2f}); "
              f"area AUC (best) where the head is wrong and a right area exists {auc:.3f}", flush=True)


def write(model: str, tag: str, ckpt: str, photos: str = "") -> Path:
    """The model's columns in megaloc_gate's format for hypothesis_agent.EXTRA_FILES -> artifacts/value/{model}_cols_{tag}_{ckpt}.npz; with
    photos (train), the unembedded photos' rows get the embedded rows' column means (they carry no answer-head gradient; this keeps the
    normalisation statistics those of real rows)."""

    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{ckpt}.npz")
    x = columns(model, tag, ckpt, zm)
    if photos:
        sel = np.isin(zm["photo"], np.flatnonzero(np.load(photos)))
        x[~sel] = x[sel].mean(0)
    dst = OUT / f"{model}_cols_{tag}_{ckpt}.npz"
    np.savez(dst, photo=zm["photo"], source=zm["source"], x=x, groups=np.asarray([model] * x.shape[1]))
    print(f"-> {dst}", flush=True)
    return dst


def siglip_cached(tag: str, ckpt: str) -> None:
    """SigLIP2-g's final pooled embeddings (Pinpoint's own space, data/pinpoint/mp16-embed) for the rows of the pe_g file, in the same format:
    no GPU (after the retrieval-track session's scripts/siglip_gate.py cached)."""

    from geo_search_env.models.pinpoint_reranker import _discover_index

    embed_dir = Path("data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384")
    ref = np.load(OUT / f"pe_g_{tag}_{ckpt}_keys.npz")
    nq, need, ok = int(ref["n_queries"]), ref["rows"], ref["ok"]
    index_dir = _discover_index()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    cache = np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
    m = json.loads((embed_dir / "manifest.json").read_text(encoding="utf-8"))
    emb = np.memmap(embed_dir / m["files"]["embeddings"], dtype=np.float16, mode="r", shape=tuple(m["shapes"]["embeddings"]))
    pos = {k: i for i, k in enumerate((embed_dir / m["files"]["image_ids"]).read_text(encoding="utf-8").splitlines())}
    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{ckpt}" / "photos.json").read_text(encoding="utf-8"))
    bench = "benchmark" in photos[0]
    rows = np.concatenate([np.zeros(nq, np.int64) if bench else np.asarray([pos[p["image_id"]] for p in photos]), cache[need]])
    out = np.lib.format.open_memmap(OUT / f"siglip_final_{tag}_{ckpt}.npy", mode="w+", dtype=np.float16, shape=(len(rows), emb.shape[1]))
    for a in range(0, len(rows), 100_000):
        r = rows[a: a + 100_000]
        o = np.argsort(r)
        x = np.empty((len(r), emb.shape[1]), np.float32)
        x[o] = np.asarray(emb[r[o]], np.float32)
        out[a: a + len(r)] = (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float16)
    if bench:  # benchmark queries: their cached SigLIP2 embedding (the one Pinpoint queries with; action_agent._data's glance)
        from geo_search_env.data.benchmarks import load_benchmark

        names = sorted({p["benchmark"] for p in photos})
        assert len(names) == 1, names
        q = np.asarray(load_benchmark(names[0]).embeddings, np.float32)[[p["index"] for p in photos]]
        out[:nq] = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float16)
    out.flush()
    np.savez(OUT / f"siglip_final_{tag}_{ckpt}_keys.npz", n_queries=nq, rows=need, ok=ok)  # the pe_g file's coverage (train: informative photos)
    print(f"-> siglip_final_{tag}_{ckpt}: {nq} queries + {len(need)} area photos", flush=True)


def head(models: list[str], ckpt: str, seeds: tuple[int, ...] = (7, 8, 9), photos: str = "artifacts/value/train_informative_setft_own.npy",
         tags: tuple[str, ...] = ("dev3k",), configs: list[str] | None = None) -> None:
    """The real test: the answer head (action_agent, --one-km 0.5) trained on train with the new columns revealed with MegaLoc (recognise), every
    tool revealed on each tag (take-everything = the RL policy at lambda 0, lesson 98), vs the same head without them; 3 seeds each. Pre-declared bar
    (dev3k): PE-G + DINOv3-7B >= +1.0 pt < 25 km over the head without them, < 1 km not down by more than 0.3 (passed, lesson V5). siglip_final
    (SigLIP2-g's own pooled embedding in the same columns; the retrieval-track session's control) was a secondary arm, chosen after the gate.
    configs: e.g. ["none", "pe_g", "pe_g+siglip_final"]; default none, each model, PE-G with each other model, all together."""

    import torch

    from geo_search_env.experiment import action_agent as aa
    from geo_search_env.experiment import hypothesis_agent as ha

    if configs:
        cfgs = [() if c == "none" else tuple(c.split("+")) for c in configs]
    else:
        cfgs = [()] + [(m,) for m in models] + [("pe_g", m) for m in models if m != "pe_g" and "pe_g" in models]
        cfgs += [tuple(models)] if len(models) > 2 or (len(models) == 2 and "pe_g" not in models) else []
    for m in sorted({m for c in cfgs for m in c}):
        if not _paths(m, "train", ckpt)[0].exists():  # column files built elsewhere (pe_probe: the retrieval session's PE-G region probe, R16)
            assert all((OUT / f"{m}_cols_{t}_{ckpt}.npz").exists() for t in ("train",) + tuple(tags)), m
            continue
        write(m, "train", ckpt, photos)
        for t in tags:
            write(m, t, ckpt)
    aa.CKPT_TAG, aa.ONE_KM = ckpt, 0.5
    res = {}
    for cfg in cfgs:
        ha.EXTRA_FILES = [str(OUT / f"{m}_cols_{{tag}}_{{ckpt}}.npz") for m in cfg]
        aa.N_ML = 4 + 4 * len(cfg)
        tr = aa._data("train", ckpt=ckpt)
        sets = {t: aa._data(t, stats=tr["stats"], ckpt=ckpt) for t in tags}
        kms = {t: [] for t in tags}
        for seed in seeds:
            net = aa.fit_answer(tr, np.arange(tr["n"]), seed=seed)
            for t, dv in sets.items():
                kms[t].append(aa.rollout(net, None, dv, 0.0, "fixed", fixed=(1, 1, "all"))[0].cpu().numpy())
        for t in tags:
            k = res[t, cfg] = np.stack(kms[t])
            line = f"  {t:13s} {'+'.join(cfg) or 'head (MegaLoc, no new columns)':32s} {np.mean(k < 25):.2%} / {np.mean(k < 1):.2%} (< 25 / 1 km; seeds " + \
                   " ".join(f"{np.mean(x < 25):.1%}" for x in k) + ")"
            if cfg and (t, ()) in res:
                b = res[t, ()]
                d25, d1 = (k < 25).mean(0) - (b < 25).mean(0), (k < 1).mean(0) - (b < 1).mean(0)
                line += f"; vs head {100 * d25.mean():+.2f} / {100 * d1.mean():+.2f} pts (paired SE {100 * d25.std() / np.sqrt(len(d25)):.2f})"
            print(line, flush=True)
        del tr, sets
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("node", choices=("embed", "gate", "head", "siglip"))
    ap.add_argument("--model", default="pe_g", choices=tuple(MODELS))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--tag", default="dev3k")
    ap.add_argument("--ckpt", default="setft_own")
    ap.add_argument("--photos", default="", help="embed: boolean .npy over the photo set (only those queries and their areas)")
    ap.add_argument("--tags", nargs="*", default=["dev3k"], help="head: photo sets to evaluate on")
    ap.add_argument("--configs", nargs="*", default=None, help="head: e.g. none pe_g pe_g+siglip_final")
    ap.add_argument("--ks", type=int, nargs="*", default=[12], help="gate: photos per area used (first k)")
    args = ap.parse_args()
    if args.node == "embed":
        embed(args.model, args.tag, args.ckpt, photos=args.photos)
    elif args.node == "gate":
        gate(args.models, args.tag, args.ckpt, tuple(args.ks))
    elif args.node == "siglip":
        siglip_cached(args.tag, args.ckpt)
    else:
        head(args.models, args.ckpt, tags=tuple(args.tags), configs=args.configs)


if __name__ == "__main__":
    main()
