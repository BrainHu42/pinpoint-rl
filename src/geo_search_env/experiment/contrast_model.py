# Our own contrast model instead of an LLM (EXPERT_PLAN.md; lesson 99: the 4B reader is position-biased and below the answer head). Given
# the head's top two areas and fetched YFCC reference photos of each (contrast_gate.fetch), a small model trained on ground truth decides
# which area the query was taken in. Evidence per area: MegaLoc similarity of the query to each reference, globally and per aggregation
# cluster (its descriptor is 64 cluster vectors of 128 + a 256-d scene token, so cluster-wise cosines are part-level matches), pooled over
# the references; plus the head's logit margin, so the model learns when to override the head.
# Train: ~5k "decisive" train photos (exactly one of the top two areas < 25 km), the head's logits out of fold (5-fold cross-fit).
# Dev: the 500 dev3k pairs of contrast_gate (head trained on all of train), the first 12 references per area.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.contrast_model pairs
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.contrast_model fetch
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.contrast_model embed
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.contrast_model train

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .contrast_gate import REFS, _pairs_path

OUT = Path("artifacts/agent")
CKPT_TAG = "setft_a"
N_TRAIN, K_REF = 5000, 12
TRAIN_PAIRS = OUT / "contrast_pairs_train.json"


def pairs(n: int = N_TRAIN) -> None:
    """Out-of-fold answer-head logits on train (5 folds, read + recognise, every area zoomed); n photos where exactly one of the top two
    areas is right (< 25 km), seed 0."""

    import torch

    from .action_agent import _data, fit_answer, view
    from .hypothesis_agent import SOURCES

    tr = _data("train")
    folds = np.random.default_rng(0).permutation(tr["n"]) % 5
    logits = np.full(tuple(tr["mask"].shape), -np.inf, np.float32)
    for f in range(5):
        net = fit_answer(tr, np.flatnonzero(folds != f), seed=7 + f)
        idx = torch.as_tensor(np.flatnonzero(folds == f), device="cuda")
        from .hypothesis_agent import _sub

        b = _sub(tr, idx)
        with torch.no_grad():
            on = torch.ones(len(idx), dtype=torch.bool, device="cuda")
            v = view(b, on, on)
            logits[folds == f] = net.answer_logits(v, v["mask"]).masked_fill(~v["mask"], float("-inf")).cpu().numpy()
        print(f"  fold {f} done", flush=True)
    km = tr["km_zoom"].cpu().numpy()
    fz = np.load(OUT / f"hypothesis_features_train_{CKPT_TAG}_e4_set.npz")
    rows = np.flatnonzero(np.isin(fz["source"], SOURCES))
    slot_row = np.full(logits.shape, -1)
    seen = np.zeros(tr["n"], int)
    for r in rows:
        p = fz["photo"][r]
        slot_row[p, seen[p]] = r
        seen[p] += 1
    top2 = np.argsort(-logits, 1)[:, :2]
    k2 = np.take_along_axis(km, top2, 1)
    ok = np.isfinite(np.take_along_axis(logits, top2, 1)).all(1)
    head = np.take_along_axis(km, logits.argmax(1)[:, None], 1)[:, 0]
    print(f"train out of fold: head pick < 25 km {np.mean(head < 25):.1%}; exactly one of the top two right {np.mean(ok & ((k2[:, 0] < 25) != (k2[:, 1] < 25))):.1%}")
    decisive = np.flatnonzero(ok & ((k2[:, 0] < 25) != (k2[:, 1] < 25)))
    pick = np.sort(np.random.default_rng(0).choice(decisive, min(n, len(decisive)), replace=False))
    out = [{"photo": int(p), "rows": slot_row[p, top2[p]].tolist(), "km": k2[p].tolist(), "logit": logits[p, top2[p]].tolist()} for p in pick]
    TRAIN_PAIRS.write_text(json.dumps(out), encoding="utf-8")
    print(f"-> {TRAIN_PAIRS}: {len(out)} decisive photos, head right on {np.mean([o['km'][0] < 25 for o in out]):.1%}")


def fetch() -> None:
    from .contrast_gate import fetch as _fetch

    _fetch(workers=128, tag="train", pairs_path=TRAIN_PAIRS, out_name="contrast_refs_train.json", k_fetch=K_REF, size="n")


def embed(batch: int = 128, workers: int = 6) -> None:
    """MegaLoc on every reference of the train and dev3k pairs (dev3k: the first K_REF per area)."""

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from .megaloc_gate import SIZE, _model

    ids = set()
    for name in ("contrast_refs_train.json", "contrast_refs_dev3k.json"):
        for v in json.loads((OUT / name).read_text(encoding="utf-8")).values():
            ids.update(v[:K_REF])
    ids = sorted(ids)
    mean, std = np.asarray([0.485, 0.456, 0.406], np.float32), np.asarray([0.229, 0.224, 0.225], np.float32)

    class Refs(Dataset):
        def __len__(self) -> int:
            return len(ids)

        def __getitem__(self, i: int):
            try:
                im = Image.open(REFS / f"{ids[i]}.jpg").convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
                return i, torch.from_numpy(((np.asarray(im, np.float32) / 255.0 - mean) / std).transpose(2, 0, 1).copy())
            except Exception:  # noqa: BLE001
                return i, torch.zeros(3, SIZE, SIZE)

    model = _model()
    out = np.zeros((len(ids), 8448), np.float16)
    with torch.inference_mode():
        for k, (idx, x) in enumerate(DataLoader(Refs(), batch_size=batch, num_workers=workers)):
            out[idx.numpy()] = model(x.cuda()).half().cpu().numpy()
            if k % 200 == 0:
                print(f"  {k * batch} / {len(ids)}", flush=True)
    np.savez(OUT / "contrast_ref_emb.npz", ids=np.asarray(ids, np.int64), emb=out)
    print(f"-> contrast_ref_emb.npz: {len(ids)} references")


def _area_features(q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Query vs an area's references: global cosine (best, top-3 mean), scene-token cosine (best), reference count, and per aggregation
    cluster the cosine (best, top-3 mean over references) -> 4 + 128 features."""

    if not len(R):
        return np.concatenate([[-1.0, -1.0, -1.0, 0.0], np.full(128, -1.0)]).astype(np.float32)
    g = R @ q
    tq, tr = q[:256] / (np.linalg.norm(q[:256]) + 1e-8), R[:, :256] / (np.linalg.norm(R[:, :256], axis=1, keepdims=True) + 1e-8)
    cq = q[256:].reshape(128, 64)
    cq = cq / (np.linalg.norm(cq, axis=0, keepdims=True) + 1e-8)
    cr = R[:, 256:].reshape(len(R), 128, 64)
    cr = cr / (np.linalg.norm(cr, axis=1, keepdims=True) + 1e-8)
    cc = np.einsum("dk,mdk->mk", cq, cr)  # [refs, clusters]
    srt = np.sort(cc, 0)[::-1]
    gs = np.sort(g)[::-1]
    return np.concatenate([[gs[0], gs[:3].mean(), (tr @ tq).max(), len(R) / K_REF], srt[0], srt[:3].mean(0)]).astype(np.float32)


def _build(split: str):
    z = np.load(OUT / "contrast_ref_emb.npz")
    pos = {int(p): i for i, p in enumerate(z["ids"])}
    emb = z["emb"]
    if split == "train":
        pr = json.loads(TRAIN_PAIRS.read_text(encoding="utf-8"))
        plan = json.loads((OUT / "contrast_refs_train.json").read_text(encoding="utf-8"))
        Q = np.load(OUT / f"megaloc_queries_train_{CKPT_TAG}.npy", mmap_mode="r")
    else:
        from .megaloc_gate import _paths as gate_paths

        pr = json.loads(_pairs_path().read_text(encoding="utf-8"))
        plan = json.loads((OUT / "contrast_refs_dev3k.json").read_text(encoding="utf-8"))
        Q = np.load(gate_paths("dev3k")[1], mmap_mode="r")
    feats, margin, km = [], [], []
    for o in pr:
        q = np.asarray(Q[o["photo"]], np.float32)
        fa = []
        for h in o["rows"]:
            refs = [pos[p] for p in plan.get(str(h), [])[:K_REF] if p in pos]
            R = np.asarray(emb[refs], np.float32)
            if len(R):
                R = R[(R @ q) < 0.95]  # near-duplicates of the query
            fa.append(_area_features(q, R))
        feats.append(np.stack(fa))
        margin.append(o["logit"][0] - o["logit"][1])
        km.append(o["km"])
    return np.stack(feats), np.asarray(margin, np.float32), np.asarray(km)


def _fit(F, M, y, seed: int, epochs: int = 60, use_refs: bool = True):
    """Symmetric pair scorer: s(a) - s(b) + w * margin, s an MLP over an area's features (and the other's, so it can compare); both orders."""

    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    d = F.shape[-1]
    net = nn.Sequential(nn.Linear(2 * d, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1))
    w = nn.Parameter(torch.ones(1))
    mu, sd = F.reshape(-1, d).mean(0), F.reshape(-1, d).std(0) + 1e-6
    Ft = torch.as_tensor((F - mu) / sd, dtype=torch.float32)
    Mt, yt = torch.as_tensor(M), torch.as_tensor(y, dtype=torch.float32)
    opt = torch.optim.AdamW(list(net.parameters()) + [w], lr=1e-3, weight_decay=1e-2)

    def logit(Ft_, Mt_):
        a, b = Ft_[:, 0], Ft_[:, 1]
        s = (net(torch.cat([a, b], 1)) - net(torch.cat([b, a], 1))).squeeze(1) if use_refs else 0.0
        return s + w * Mt_

    for _ in range(epochs):
        net.train()
        for bi in torch.randperm(len(yt)).split(128):
            loss = nn.functional.binary_cross_entropy_with_logits(logit(Ft[bi], Mt[bi]), yt[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()

    def predict(F2, M2):
        with torch.no_grad():
            return torch.sigmoid(logit(torch.as_tensor((F2 - mu) / sd, dtype=torch.float32), torch.as_tensor(M2))).numpy()

    return predict


def train(seeds: int = 5) -> None:
    Ft, Mt, Kt = _build("train")
    Fd, Md, Kd = _build("dev3k")
    yt = (Kt[:, 0] < 25).astype(np.float32)  # 1: the head's top area is the right one
    print(f"train: {len(yt)} decisive pairs, head right {yt.mean():.1%}; references per area median {np.median(Ft[:, :, 3]) * K_REF:.0f}")
    # train CV (5 folds) for the head-margin-only baseline and the full model
    folds = np.random.default_rng(1).permutation(len(yt)) % 5
    for label, use in (("head margin only", False), ("+ fetched-reference MegaLoc", True)):
        p = np.zeros(len(yt))
        for f in range(5):
            pr = [_fit(Ft[folds != f], Mt[folds != f], yt[folds != f], seed=s, use_refs=use) for s in range(seeds)]
            p[folds == f] = np.mean([q(Ft[folds == f], Mt[folds == f]) for q in pr], 0)
        print(f"  train CV, {label:28s}: pair accuracy {np.mean((p >= 0.5) == (yt == 1)):.1%}")
    # dev3k: all 500 pairs; accuracy where exactly one is right, net fixes / breaks of switching to the second area
    for label, use in (("head margin only", False), ("+ fetched-reference MegaLoc", True)):
        pr = [_fit(Ft, Mt, yt, seed=s, use_refs=use) for s in range(seeds)]
        p = np.mean([q(Fd, Md) for q in pr], 0)
        print(f"\n dev3k ({len(p)} pairs), {label}: switches to the second area on {np.sum(p < 0.5)}")
        for thr in (25, 1):
            right = Kd < thr
            dec = right[:, 0] != right[:, 1]
            acc = np.mean(np.where(p[dec] >= 0.5, right[dec, 0], right[dec, 1]))
            sw = p < 0.5
            fixes, breaks = int((sw & right[:, 1] & ~right[:, 0]).sum()), int((sw & right[:, 0] & ~right[:, 1]).sum())
            print(f"   < {thr:2d} km: decisive {dec.sum()}, head right {np.mean(right[dec, 0]):.1%}, model {acc:.1%}; "
                  f"fixes {fixes}, breaks {breaks}, net {100 * (fixes - breaks) / len(p):+.2f} pts of photos")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("pairs", "fetch", "embed", "train"))
    args = parser.parse_args(argv)
    {"pairs": pairs, "fetch": fetch, "embed": embed, "train": train}[args.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
