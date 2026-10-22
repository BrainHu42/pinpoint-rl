"""SigLIP2-g as a second model on the candidate areas: the control for track V's PE-G gate (scripts/second_model_gate.py on branch
worktree-value-agent). PE-G's per-area columns add +0.79 pts < 25 km over the answer head on dev3k; is that PE's knowledge, or does the column
form itself add (query vs each area's 12 restricted-search photos), which the head never sees for SigLIP2 either?

cached: SigLIP2-g final (pooled) embeddings of the dev3k queries and every area photo, straight from the MP16 cache (no GPU), written in the
        gate's format to artifacts/retrieval/gate/siglip_final_dev3k_setft_own.npy (+ _keys.npz); PE / DINOv3 files are symlinked there.
gate:   second_model_gate.gate with OUT pointed at artifacts/retrieval/gate.
solo:   each model alone, per photo with a right area: does the area with the highest best-cosine hold a right answer (< 25 km)?

  PYTHONPATH=<value-agent>/src:<value-agent>/scripts .venv/bin/python scripts/siglip_gate.py cached
  PYTHONPATH=<value-agent>/src:<value-agent>/scripts .venv/bin/python scripts/siglip_gate.py gate --models siglip_final pe_g
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path("artifacts/retrieval/gate")
VALUE = Path("artifacts/value")
EMBED = Path("data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384")


def cached(tag: str, ckpt: str) -> None:
    from geo_search_env.models.pinpoint_reranker import _discover_index

    OUT.mkdir(parents=True, exist_ok=True)
    ref = np.load(VALUE / f"pe_g_{tag}_{ckpt}_keys.npz")  # same rows as track V's files, so the gate lines up
    nq, need = int(ref["n_queries"]), ref["rows"]
    index_dir = _discover_index()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    cache = np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
    m = json.loads((EMBED / "manifest.json").read_text(encoding="utf-8"))
    emb = np.memmap(EMBED / m["files"]["embeddings"], dtype=np.float16, mode="r", shape=tuple(m["shapes"]["embeddings"]))
    ids = (EMBED / m["files"]["image_ids"]).read_text(encoding="utf-8").splitlines()
    pos = {k: i for i, k in enumerate(ids)}
    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{ckpt}" / "photos.json").read_text(encoding="utf-8"))
    qrows = np.asarray([pos[p["image_id"]] for p in photos])
    assert len(qrows) == nq
    arows = cache[need]
    rows = np.concatenate([qrows, arows])
    order = np.argsort(rows)
    x = np.empty((len(rows), emb.shape[1]), np.float32)
    x[order] = np.asarray(emb[rows[order]], np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    np.save(OUT / f"siglip_final_{tag}_{ckpt}.npy", x.astype(np.float16))
    np.savez(OUT / f"siglip_final_{tag}_{ckpt}_keys.npz", n_queries=nq, rows=need, ok=np.isfinite(x).all(1))
    for model in ("pe_g", "pe_l", "dino_7b", "dino_l"):
        for f in (VALUE / f"{model}_{tag}_{ckpt}.npy", VALUE / f"{model}_{tag}_{ckpt}_keys.npz"):
            if f.exists() and not (OUT / f.name).exists():
                (OUT / f.name).symlink_to(f.resolve())
    print(f"-> {OUT}: {nq} queries + {len(need)} area photos from the MP16 cache", flush=True)


def solo(models: list[str], tag: str, ckpt: str) -> None:
    import second_model_gate as g

    g.OUT = OUT
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{ckpt}.npz")
    logits, km, hyp = g.head_options(tag, ckpt)
    valid = np.isfinite(logits)
    head_km = km[np.arange(len(km)), np.where(valid, logits, -np.inf).argmax(1)]
    half = km.shape[1] // 2
    akm = np.full(len(zm["photo"]), np.inf)
    for p in range(len(km)):
        for j in range(half):
            if hyp[p, j] >= 0:
                akm[hyp[p, j]] = min(km[p, j], km[p, j + half])
    has = np.zeros(len(km), bool)
    has[np.unique(zm["photo"][akm < 25])] = True
    hw = has & (head_km >= 25)
    print(f"{tag}: photos with a right area {has.mean():.1%}, of which the head is wrong on {hw.sum()}", flush=True)
    for m in models:
        best = g.columns(m, tag, ckpt, zm)[:, 0]
        pick = np.full(len(km), np.inf)
        for p in np.flatnonzero(has):
            J = np.flatnonzero((zm["photo"] == p) & np.isfinite(akm))
            pick[p] = akm[J[np.argmax(best[J])]]
        print(f"  {m:14s} top area by best cosine right (< 25 km): {np.mean(pick[has] < 25):.1%} of photos with a right area; "
              f"{np.mean(pick[hw] < 25):.1%} where the head is wrong", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("node", choices=("cached", "gate", "solo"))
    ap.add_argument("--models", nargs="*", default=["siglip_final", "pe_g"])
    ap.add_argument("--tag", default="dev3k")
    ap.add_argument("--ckpt", default="setft_own")
    args = ap.parse_args()
    if args.node == "cached":
        cached(args.tag, args.ckpt)
    elif args.node == "gate":
        import second_model_gate as g

        g.OUT = OUT
        g.gate(args.models, args.tag, args.ckpt)
    else:
        solo(args.models, args.tag, args.ckpt)


if __name__ == "__main__":
    main()
