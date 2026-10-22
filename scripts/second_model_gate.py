"""A second image model on the candidate areas (VALUE_PLAN.md idea 2; user, 2026-10-22). Every Pinpoint score comes from SigLIP2 (plus MegaLoc's
same-place columns, lesson 93, the largest evidence gain since the retrained reranker). Meta's Perception Encoder (PE-Core, 5.4B image-text
pairs, apache-2.0; timm `vit_pe_core_{gigantic_patch14_448,large_patch14_336}.fb`) learned from other data: does it know which candidate
area is right where SigLIP2 and MegaLoc don't? DINOv3 is gated on the Hub (needs a licence click), so PE first.

embed (.venvs/sft, timm on PYTHONPATH): the dev3k queries and every photo of their areas' restricted searches (the zoom file's 12 per area,
        as megaloc_gate), unit-norm embeddings -> artifacts/value/pe_{model}_{tag}_{ckpt}.npy (+ _keys.npy); resumable.
gate (.venv): per area the best / top-3 cosine and both relative to the photo's other areas (megaloc_gate's columns). The answer head of
        the current pipeline (action_agent, --one-km 0.5, trained on all train, seed 7) with every tool revealed gives a logit per answer
        option (area x point); a conditional logit over the options on dev3k (2-fold x 3 seeds) with [head logit] vs [head logit + PE
        columns]; control: [head logit + MegaLoc columns], already inside the head, should add ~0. Pre-declared bar: PE adds >= +1.0 pt
        < 25 km over head-only, < 1 km not down by more than 0.3, and the MegaLoc control within +-0.3.

  PYTHONPATH=<timm>:src .venvs/sft/bin/python scripts/second_model_gate.py embed --model pe_g
  PYTHONPATH=src .venv/bin/python scripts/second_model_gate.py gate --models pe_g pe_l
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import numpy as np

OUT = Path("artifacts/value")
MODELS = {"pe_g": "vit_pe_core_gigantic_patch14_448.fb", "pe_l": "vit_pe_core_large_patch14_336.fb"}


def _paths(model: str, tag: str, ckpt: str) -> tuple[Path, Path]:
    return OUT / f"{model}_{tag}_{ckpt}.npy", OUT / f"{model}_{tag}_{ckpt}_keys.npz"


def embed(model: str, tag: str, ckpt: str, batch: int = 64, workers: int = 4) -> None:
    os.environ.setdefault("HF_HOME", str(Path("data/hf").resolve()))
    os.environ["HF_HUB_OFFLINE"] = "1"
    import timm
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from geo_search_env.experiment import megaloc_gate
    from geo_search_env.experiment.sft_data import MP16Images

    megaloc_gate.CKPT_TAG = ckpt
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{ckpt}.npz")
    rows = zm["rows"]
    need = np.unique(rows[rows >= 0])
    ids, nq, _ = megaloc_gate._image_ids(tag, need)  # queries first, then the area photos in index-row order
    net = timm.create_model(MODELS[model], pretrained=True).cuda().eval().to(torch.bfloat16)
    cfg = timm.data.resolve_data_config(model=net)
    tf = timm.data.create_transform(**cfg)
    size = cfg["input_size"][1]

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
    sub = torch.utils.data.Subset(Images(), range(start, len(ids)))
    with torch.inference_mode():
        for b, (idx, x, good) in enumerate(DataLoader(sub, batch_size=batch, num_workers=workers)):
            e = torch.nn.functional.normalize(net(x.cuda().to(torch.bfloat16)).float(), dim=-1)
            out[idx.numpy()] = e.cpu().numpy().astype(np.float16)
            ok[idx.numpy()] = good.numpy()
            if b % 200 == 199 or int(idx[-1]) == len(ids) - 1:
                out.flush()
                np.savez(key_path, n_queries=nq, rows=need, ok=ok)
                prog.write_text(str(int(idx[-1]) + 1))
                print(f"  {int(idx[-1]) + 1:,} / {len(ids):,}", flush=True)
    print(f"-> {emb_path}: {np.mean(~ok):.2%} unreadable", flush=True)


def columns(model: str, tag: str, ckpt: str, zm) -> np.ndarray:
    """Per zoom-file hypothesis: best, top-3 mean, and both minus the best of the photo's other non-street areas (megaloc_gate._write_columns)."""

    emb_path, key_path = _paths(model, tag, ckpt)
    k = np.load(key_path)
    e = np.load(emb_path, mmap_mode="r")
    nq, need, ok = int(k["n_queries"]), k["rows"], k["ok"]
    rows, photo = zm["rows"], zm["photo"]
    q = np.asarray(e[:nq], np.float32)
    sims = np.full(rows.shape, np.nan, np.float32)
    for h in range(len(photo)):
        v = np.flatnonzero(rows[h] >= 0)
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


def gate(models: list[str], tag: str, ckpt: str) -> None:
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{ckpt}.npz")
    logits, km, hyp = head_options(tag, ckpt)
    valid = np.isfinite(logits)
    head_km = km[np.arange(len(km)), np.where(valid, logits, -np.inf).argmax(1)]
    print(f"\n{tag}: head argmax (read + recognise + zoom all) {np.mean(head_km < 25):.1%} < 25 km, {np.mean(head_km < 1):.1%} < 1 km; "
          f"option oracle {np.mean((km < 25).any(1)):.1%}", flush=True)
    groups = {"MegaLoc (control)": np.load(Path("artifacts/agent") / f"megaloc_{tag}_{ckpt}.npz")["x"]}
    groups |= {m: columns(m, tag, ckpt, zm) for m in models if _paths(m, tag, ckpt)[0].exists()}
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("node", choices=("embed", "gate"))
    ap.add_argument("--model", default="pe_g", choices=tuple(MODELS))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--tag", default="dev3k")
    ap.add_argument("--ckpt", default="setft_own")
    args = ap.parse_args()
    if args.node == "embed":
        embed(args.model, args.tag, args.ckpt)
    else:
        gate(args.models, args.tag, args.ckpt)


if __name__ == "__main__":
    main()
