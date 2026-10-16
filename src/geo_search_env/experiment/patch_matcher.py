# A learned patch-level matcher for the contrast step (EXPERT_PLAN.md; lessons 99-100: global MegaLoc matching of fetched references adds
# nothing beyond the answer head). Global descriptors squeeze a photo into one vector, so "same kind of scene" and "same building" look
# alike; matching parts can tell them apart. Each image (query, and the 8 most MegaLoc-similar fetched YFCC references of each of the head's
# top two areas) is encoded once by MegaLoc's frozen DINOv2 backbone into 12 x 12 patch tokens (768-d). A small trained head projects the
# tokens (768 -> 128), finds for every query patch its best-matching patch in each reference (late interaction), pools over references and
# weights query patches by a learned saliency (sky and pets matter less than facades and signs), giving one score per area. The pair logit
# is score(A) - score(B) + w * the head's logit margin, trained with ground-truth labels on the 5k decisive train pairs (contrast_model).
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.patch_matcher tokens
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.patch_matcher train

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .contrast_gate import REFS, _pairs_path
from .contrast_model import TRAIN_PAIRS

OUT = Path("artifacts/agent")
CKPT_TAG = "setft_a"
K_USE, GRID = 8, 12
TOKENS = OUT / "patch_tokens.npy"  # [images, GRID * GRID, 768] fp16
INDEX = OUT / "patch_tokens_index.json"


def _layout():
    """Per split the pairs, and per area the K_USE references to use: the most MegaLoc-similar to the query among the first 12 (< 0.95)."""

    from .megaloc_gate import _paths as gate_paths

    z = np.load(OUT / "contrast_ref_emb.npz")
    pos = {int(p): i for i, p in enumerate(z["ids"])}
    emb = z["emb"]
    out = {}
    for split, pairs_path, refs_name, qpath in (
        ("train", TRAIN_PAIRS, "contrast_refs_train.json", OUT / f"megaloc_queries_train_{CKPT_TAG}.npy"),
        ("dev3k", _pairs_path(), "contrast_refs_dev3k.json", gate_paths("dev3k")[1]),
    ):
        pr = json.loads(pairs_path.read_text(encoding="utf-8"))
        plan = json.loads((OUT / refs_name).read_text(encoding="utf-8"))
        Q = np.load(qpath, mmap_mode="r")
        items = []
        for o in pr:
            q = np.asarray(Q[o["photo"]], np.float32)
            areas = []
            for h in o["rows"]:
                ids = [p for p in plan.get(str(h), [])[:12] if p in pos]
                s = np.asarray([emb[pos[p]].astype(np.float32) @ q for p in ids])
                areas.append([p for v, p in sorted(zip(s, ids), reverse=True) if v < 0.95][:K_USE])
            items.append({"photo": o["photo"], "areas": areas, "km": o["km"], "margin": o["logit"][0] - o["logit"][1]})
        out[split] = items
    return out


def tokens(batch: int = 128, workers: int = 8) -> None:
    """Frozen MegaLoc backbone patch tokens (avg-pooled to GRID x GRID) for every query and used reference."""

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from .megaloc_gate import SIZE, _model
    from .name_score import _read
    from .sft_data import MP16Images

    lay = _layout()
    photos = {s: json.loads((Path("artifacts/photo_sets") / f"{s}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8")) for s in lay}
    keys = []  # ("q", split, photo) or ("r", photo id)
    for s, items in lay.items():
        keys += [("q", s, o["photo"]) for o in items]
    keys += [("r", p) for p in sorted({p for items in lay.values() for o in items for a in o["areas"] for p in a})]
    mean, std = np.asarray([0.485, 0.456, 0.406], np.float32), np.asarray([0.229, 0.224, 0.225], np.float32)

    class Images(Dataset):
        images = None

        def __len__(self) -> int:
            return len(keys)

        def __getitem__(self, i: int):
            import io

            k = keys[i]
            try:
                if k[0] == "q":
                    if self.images is None:
                        self.images = MP16Images()
                    im = Image.open(io.BytesIO(_read(photos[k[1]][k[2]], self.images)))
                else:
                    im = Image.open(REFS / f"{k[1]}.jpg")
                im = im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
                return i, torch.from_numpy(((np.asarray(im, np.float32) / 255.0 - mean) / std).transpose(2, 0, 1).copy())
            except Exception:  # noqa: BLE001
                return i, torch.zeros(3, SIZE, SIZE)

    backbone = _model().backbone
    out = np.lib.format.open_memmap(TOKENS, mode="w+", dtype=np.float16, shape=(len(keys), GRID * GRID, 768))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for k, (idx, x) in enumerate(DataLoader(Images(), batch_size=batch, num_workers=workers)):
            patches, _ = backbone(x.cuda())
            t = torch.nn.functional.adaptive_avg_pool2d(patches.float(), GRID).flatten(2).transpose(1, 2)
            out[idx.numpy()] = t.half().cpu().numpy()
            if k % 100 == 0:
                print(f"  {k * batch} / {len(keys)}", flush=True)
    out.flush()
    index = {"q": {f"{k[1]}:{k[2]}": i for i, k in enumerate(keys) if k[0] == "q"}, "r": {str(k[1]): i for i, k in enumerate(keys) if k[0] == "r"}}
    INDEX.write_text(json.dumps({"index": index, "layout": lay}), encoding="utf-8")
    print(f"-> {TOKENS}: {len(keys)} images")


def _tensors(split: str, idx: dict, lay: dict):
    """Per pair: query token row, [2, K_USE] reference rows (-1 padded), km of both areas, head margin."""

    items = lay[split]
    q = np.asarray([idx["q"][f"{split}:{o['photo']}"] for o in items])
    r = np.full((len(items), 2, K_USE), -1)
    for i, o in enumerate(items):
        for a, refs in enumerate(o["areas"]):
            for j, p in enumerate(refs):
                r[i, a, j] = idx["r"][str(p)]
    return q, r, np.asarray([o["km"] for o in items]), np.asarray([o["margin"] for o in items], np.float32)


def _matcher(dim: int = 128):
    import torch
    import torch.nn as nn

    class Matcher(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, dim))
            self.sal = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 1))
            self.tau = nn.Parameter(torch.tensor(10.0))
            self.w_score, self.w_margin = nn.Parameter(torch.tensor(1.0)), nn.Parameter(torch.tensor(1.0))

        def area_score(self, qt, rt, rmask):
            """qt [B, P, 768], rt [B, K, P, 768], rmask [B, K] -> [B]."""
            qn = nn.functional.normalize(self.proj(qt), dim=-1)
            rn = nn.functional.normalize(self.proj(rt), dim=-1)
            sim = torch.einsum("bpd,bkqd->bpkq", qn, rn).amax(-1)  # best patch per (query patch, reference)
            sim = sim.masked_fill(~rmask[:, None, :], -1.0)
            best = torch.logsumexp(self.tau * sim, -1) / self.tau  # soft best reference per query patch
            w = torch.softmax(self.sal(qt).squeeze(-1), -1)
            return (w * best).sum(-1)

        def forward(self, qt, rt, rmask, margin):
            sa = self.area_score(qt, rt[:, 0], rmask[:, 0])
            sb = self.area_score(qt, rt[:, 1], rmask[:, 1])
            return self.w_score * (sa - sb) * 10 + self.w_margin * margin

    return Matcher().cuda()


def train(epochs: int = 12, batch: int = 32, seeds: int = 3) -> None:
    import torch

    meta = json.loads(INDEX.read_text(encoding="utf-8"))
    idx, lay = meta["index"], meta["layout"]
    T = np.load(TOKENS, mmap_mode="r")
    G = torch.empty(T.shape, dtype=torch.float16, device="cuda")  # all tokens on the GPU (~21 GB)
    for s in range(0, len(T), 4096):
        G[s:s + 4096] = torch.as_tensor(np.asarray(T[s:s + 4096]), device="cuda")
    data = {s: _tensors(s, idx, lay) for s in ("train", "dev3k")}

    def batch_of(split, sel):
        q, r, km, m = data[split]
        rr = torch.as_tensor(r[sel], device="cuda")
        rt = G[rr.clamp(min=0)].float()
        return G[torch.as_tensor(q[sel], device="cuda")].float(), rt, rr >= 0, torch.as_tensor(m[sel], device="cuda"), km[sel]

    def predict(net, split, patch_only=False):
        n = len(data[split][0])
        out = []
        with torch.no_grad():
            for s in range(0, n, 64):
                qt, rt, rm, m, _ = batch_of(split, np.arange(s, min(n, s + 64)))
                out.append((net(qt, rt, rm, m * 0) if patch_only else torch.sigmoid(net(qt, rt, rm, m))).cpu().numpy())
        return np.concatenate(out)

    def fit(sel, seed, zero_shot=False):
        torch.manual_seed(seed)
        net = _matcher()
        if zero_shot:
            return net
        y = torch.as_tensor((data["train"][2][:, 0] < 25).astype(np.float32), device="cuda")
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            net.train()
            for bi in np.array_split(rng.permutation(sel), max(1, len(sel) // batch)):
                qt, rt, rm, m, _ = batch_of("train", bi)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(qt, rt, rm, m), y[torch.as_tensor(bi, device="cuda")])
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        return net

    def auc(s, yv):
        a, b = s[yv == 1], s[yv == 0]
        return (a[:, None] > b[None]).mean() + 0.5 * (a[:, None] == b[None]).mean()

    yt = (data["train"][2][:, 0] < 25).astype(int)
    n = len(yt)
    folds = np.random.default_rng(1).permutation(n) % 5
    p, pp = np.zeros(n), np.zeros(n)
    for f in range(5):
        nets = [fit(np.flatnonzero(folds != f), seed=10 * f + s) for s in range(seeds)]
        sub = np.flatnonzero(folds == f)
        q_, r_, km_, m_ = data["train"]
        data["cv"] = (q_[sub], r_[sub], km_[sub], m_[sub])
        p[sub] = np.mean([predict(net, "cv") for net in nets], 0)
        pp[sub] = np.mean([predict(net, "cv", patch_only=True) for net in nets], 0)
        print(f"  fold {f}: AUC {auc(p[sub], yt[sub]):.3f}", flush=True)
    m_all = data["train"][3]
    print(f"train CV ({n} decisive pairs, head right {yt.mean():.1%}): head margin AUC {auc(m_all, yt):.3f}; "
          f"matcher + margin AUC {auc(p, yt):.3f}, accuracy {np.mean((p >= 0.5) == yt):.1%}")
    nets = [fit(np.arange(n), seed=100 + s) for s in range(seeds)]
    pd = np.mean([predict(net, "dev3k") for net in nets], 0)
    km = data["dev3k"][2]
    for thr in (25, 1):
        right = km < thr
        dec = right[:, 0] != right[:, 1]
        sw = pd < 0.5
        fixes, breaks = int((sw & right[:, 1] & ~right[:, 0]).sum()), int((sw & right[:, 0] & ~right[:, 1]).sum())
        print(f"dev3k < {thr:2d} km: decisive {dec.sum()}, head right {np.mean(right[dec, 0]):.1%}, matcher {np.mean(np.where(pd[dec] >= 0.5, right[dec, 0], right[dec, 1])):.1%} "
              f"(AUC {auc(pd[dec], right[dec, 0].astype(int)):.3f} vs margin {auc(data['dev3k'][3][dec], right[dec, 0].astype(int)):.3f}); "
              f"switches {sw.sum()}, fixes {fixes}, breaks {breaks}, net {100 * (fixes - breaks) / len(pd):+.2f} pts")
    # patch evidence alone (no margin): does it separate the right area at all?
    ps = np.mean([predict(net, "dev3k", patch_only=True) for net in nets], 0)
    right = km < 25
    dec = right[:, 0] != right[:, 1]
    print(f"patch score alone (margin weight zeroed): train CV AUC {auc(pp, yt):.3f}, dev3k AUC {auc(ps[dec], right[dec, 0].astype(int)):.3f}")
    # stacking: logistic regression on [head margin, out-of-fold patch score], fit on train, applied to dev3k
    X = torch.tensor(np.column_stack([m_all, pp]), dtype=torch.float64)
    w = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w[:2] + w[2], torch.tensor(yt, dtype=torch.float64))
        loss.backward()
        return loss

    opt.step(closure)
    wv = w.detach().numpy()
    print(f"stack weights margin {wv[0]:.3f}, patch {wv[1]:.3f}, bias {wv[2]:.3f}")
    pdl = data["dev3k"][3] * wv[0] + ps * wv[1] + wv[2]
    for thr in (25, 1):
        right = km < thr
        dec = right[:, 0] != right[:, 1]
        sw = pdl < 0
        fixes, breaks = int((sw & right[:, 1] & ~right[:, 0]).sum()), int((sw & right[:, 0] & ~right[:, 1]).sum())
        print(f"stacked dev3k < {thr:2d} km: decisive accuracy {np.mean(np.where(pdl[dec] >= 0, right[dec, 0], right[dec, 1])):.1%} "
              f"(AUC {auc(pdl[dec], right[dec, 0].astype(int)):.3f}); switches {sw.sum()}, fixes {fixes}, breaks {breaks}, "
              f"net {100 * (fixes - breaks) / len(pdl):+.2f} pts")
    torch.save([net.state_dict() for net in nets], OUT / "patch_matcher.pt")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("tokens", "train"))
    args = parser.parse_args(argv)
    {"tokens": tokens, "train": train}[args.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
