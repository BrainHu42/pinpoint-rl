# OSV-5M for spot refinement (1 km): inside the area the agent chose, the 50 OSV images Pinpoint's contrastive space ranks highest, re-ranked
# by MegaLoc same-place similarity (images read from the Hugging Face zips, osv_remote); when MegaLoc is sure, answer at that OSV image.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.osv_refine fetch --tag dev3k
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.osv_refine report

"""Why: inside the agent's area an OSV image within 1 km of the truth exists for 80% of dev3k / 69% of commons26 dev photos the agent has
within 25 km, and the contrastive top 50 holds one where the agent misses 1 km on 15.9% / 21.5% of them (50 random OSV images from the
area: 10.1% / 15.0%; scripts/clue_verify/osv_refine_ceiling.py). The contrastive top-1 alone is within 1 km on only 13% (agent 65%).

fetch:  per photo (all of dev3k / commons26 dev, so a switch rule is judged on every photo), the baseline agent's area centre (setft_own,
        --one-km 0.5, RL lambda 0), the K OSV images within R_KM of it with the highest contrastive similarity, their MegaLoc cosine to the
        query (the pipeline's MegaLoc query embedding) -> artifacts/osv_refine/{tag}.npz (rows, latlon, csim, msim; NaN = unreadable).
report: MegaLoc's pick vs the contrastive top-1, and a switch (answer at MegaLoc's best OSV image if its cosine > t and the agent's
        confidence < c), thresholds by 5-fold CV over photos; < 1 / 25 km vs the agent.

Results (2026-10-17; 142,911 / 44,116 (photo, OSV image) pairs, all read): on photos the agent has within 25 km, < 1 km: dev3k agent 65.2%,
contrastive top-1 13.0%, MegaLoc pick 14.4% (any of the 50: 61.2%); commons26 dev agent 30.5%, 6.1%, 11.0% (43.9%). MegaLoc finds no
same-place OSV image (best cosine < 0.1) on 76% / 75% of them; above 0.1 its pick is right 23% / 26% vs the agent's 55% / 27%. CV switch:
dev3k -0.1 / -0.0 at 1 / 25 km (switched 0.8%), commons26 dev -0.4 / +0.0 (11.5%). On the 5% most street-like dev3k photos (OSV top-1
SigLIP above MP16's) the MegaLoc pick is right 8.5% vs the agent's 28.8%. An OSV image within 1 km usually exists, but it does not show
what the photo shows; not adopted.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .clue_verify import OSV_EMBED, _haversine, _megaloc_tf

OUT = Path("artifacts/osv_refine")
BASE = Path("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz")
KEYS = {"dev3k": "dev3k", "commons26dev": "commons26dev:setft_own"}
K, R_KM = 50, 25.0


def fetch(tag: str, workers: int = 64, batch: int = 128) -> None:
    import torch
    from scipy.spatial import cKDTree

    from ..models.pinpoint_reranker import CONTRASTIVE_CHECKPOINT, _discover_index, _load_image_tower, _unit_xyz
    from .hypotheses import photo_world
    from .megaloc_gate import _model
    from .osv_remote import INDEX, read

    photos = json.loads(Path(f"artifacts/photo_sets/{tag}_setft_own/photos.json").read_text(encoding="utf-8"))
    centre = np.load(BASE)[f"{KEYS[tag]}|RL policy, lambda 0|center"].astype(np.float64)
    root = _discover_index() / "osv_support_cache"
    man = json.loads((root / "manifest.json").read_text())
    emb = torch.as_tensor(np.fromfile(root / man["files"]["embeddings"], dtype=np.float16).reshape(man["shapes"]["embeddings"]), device="cuda")
    ll = np.fromfile(root / man["files"]["latlon_deg"], dtype=np.float32).reshape(man["shapes"]["latlon_deg"])
    ids = (OSV_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines()  # same row order as the cache (identical lat / lon)
    world, qrows = photo_world(photos)
    tower = _load_image_tower(CONTRASTIVE_CHECKPOINT)[0].cuda().eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        q = tower(torch.nn.functional.normalize(torch.as_tensor(np.asarray(world.query_embeddings[qrows], np.float32), device="cuda"), dim=-1), "mp16")
    q = torch.nn.functional.normalize(q.float(), dim=-1).half()
    tree = cKDTree(_unit_xyz(ll.astype(np.float64)))
    chord = 2 * np.sin(R_KM / 6371.0 / 2)
    rows = np.full((len(photos), K), -1, np.int64)
    csim = np.full((len(photos), K), np.nan, np.float32)
    for p in range(len(photos)):
        idx = np.asarray(tree.query_ball_point(_unit_xyz(centre[p:p + 1])[0], chord), np.int64)
        if len(idx):
            s = (emb[torch.as_tensor(idx, device="cuda")] @ q[p]).float().cpu().numpy()
            o = np.argsort(-s)[:K]
            rows[p, :len(o)], csim[p, :len(o)] = idx[o], s[o]
    del emb
    zindex = json.loads(INDEX.read_text())
    need = np.unique(rows[rows >= 0])
    need = need[[ids[r] in zindex for r in need]]
    print(f"{tag}: {len(photos)} photos, {int((rows >= 0).sum()):,} (photo, OSV image) pairs, {len(need):,} images to read", flush=True)
    qml = torch.as_tensor(np.load(f"artifacts/agent/megaloc_queries_{tag}_setft_own.npy"), device="cuda").float()
    qml = torch.nn.functional.normalize(qml, dim=-1)
    msim = np.full((len(photos), K), np.nan, np.float32)
    part = OUT / f"{tag}_partial.npz"
    done = 0
    if part.exists():
        st = np.load(part)
        msim, done = st["msim"], int(st["done"])
        print(f"  resuming at {done:,}", flush=True)
    pos = np.searchsorted(need, np.where(rows >= 0, rows, 0))
    hit = (rows >= 0) & (pos < len(need)) & (need[np.minimum(pos, len(need) - 1)] == rows)
    model, tf = _model(), _megaloc_tf()
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for a in range(done, len(need), batch):
            chunk = need[a:a + batch]
            xs = [(r, tf(b)) for r, b in zip(chunk, pool.map(lambda r: read(zindex[ids[r]]), chunk))]
            xs = [(r, x) for r, x in xs if x is not None]
            if xs:
                with torch.inference_mode():
                    e = torch.nn.functional.normalize(model(torch.stack([x for _, x in xs]).cuda()).float(), dim=-1)
                got = np.asarray([r for r, _ in xs])
                pp, kk = np.nonzero(hit & np.isin(rows, got))
                j = np.searchsorted(got, rows[pp, kk])
                msim[pp, kk] = (e[torch.as_tensor(j, device="cuda")] * qml[torch.as_tensor(pp, device="cuda")]).sum(1).cpu().numpy()
            if a // batch % 20 == 19:
                np.savez(part, msim=msim, done=a + batch)
                print(f"  {a + len(chunk):,}/{len(need):,} images, {(a + len(chunk) - done) / (time.time() - t0):.0f}/s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{tag}.npz", rows=rows, latlon=np.where(rows[..., None] >= 0, ll[np.maximum(rows, 0)], np.nan), csim=csim, msim=msim,
             centre=centre)
    part.unlink(missing_ok=True)
    print(f"-> {OUT / f'{tag}.npz'}: MegaLoc on {np.mean(np.isfinite(msim[rows >= 0])):.1%} of pairs ({time.time() - t0:.0f}s)", flush=True)


def report() -> None:
    base = np.load(BASE)
    for tag, key in KEYS.items():
        z = np.load(OUT / f"{tag}.npz")
        photos = json.loads(Path(f"artifacts/photo_sets/{tag}_setft_own/photos.json").read_text(encoding="utf-8"))
        truth = np.asarray([p["truth"] for p in photos])
        km, conf = base[f"{key}|RL policy, lambda 0"], base[f"{key}|RL policy, lambda 0|conf"]
        d = _haversine(truth[:, None, 0], truth[:, None, 1], z["latlon"][..., 0], z["latlon"][..., 1])
        m = np.where(np.isfinite(z["msim"]), z["msim"], -np.inf)
        best = m.argmax(1)
        has = np.isfinite(m.max(1))
        n = len(km)
        d_ml = np.where(has, d[np.arange(n), best], np.inf)
        d_c = np.where(np.isfinite(z["csim"][:, 0]), d[:, 0], np.inf)
        s_best = np.where(has, m.max(1), -1.0)
        in25 = km < 25
        print(f"== {tag} ({n} photos): agent < 1 / 25 km {np.mean(km < 1):.1%} / {np.mean(km < 25):.1%}; on photos within 25 km ({in25.sum()}): "
              f"agent < 1 km {np.mean(km[in25] < 1):.1%}, contrastive top-1 {np.mean(d_c[in25] < 1):.1%}, MegaLoc pick {np.mean(d_ml[in25] < 1):.1%}, "
              f"any of the {K} {np.mean((d[in25] < 1).any(1)):.1%}")
        ok = in25 & has
        for lo, hi in ((-1, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 1.01)):
            b = ok & (s_best >= lo) & (s_best < hi)
            if b.sum():
                print(f"   MegaLoc best cosine [{lo:.1f}, {hi:.1f}): n={b.sum():4d}, its image < 1 km {np.mean(d_ml[b] < 1):5.1%}, agent < 1 km {np.mean(km[b] < 1):5.1%}")
        fold = np.random.default_rng(0).integers(0, 5, n)
        grid = [(t, c) for t in np.arange(0.1, 0.61, 0.025) for c in (*np.quantile(conf, [0.25, 0.5, 0.75]), np.inf)]
        ret = lambda k: (k < 1).astype(float) + (k < 25)  # the switch must not cost 25 km accuracy
        new_km = np.zeros(n)
        for f in range(5):
            tr, te = fold != f, fold == f
            gain = lambda g, idx: np.mean(np.where((s_best[idx] > g[0]) & (conf[idx] < g[1]), ret(d_ml[idx]), ret(km[idx])))
            g = max(grid, key=lambda g: gain(g, tr))
            sw = (s_best > g[0]) & (conf < g[1])
            new_km[te] = np.where(sw[te], d_ml[te], km[te])
        sw_rate = np.mean(new_km != km)
        print(f"   CV switch (MegaLoc cosine > t, agent confidence < c): < 1 km {np.mean(new_km < 1):.1%} ({100 * (np.mean(new_km < 1) - np.mean(km < 1)):+.1f}), "
              f"< 25 km {np.mean(new_km < 25):.1%} ({100 * (np.mean(new_km < 25) - np.mean(km < 25)):+.1f}), switched {sw_rate:.1%}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("fetch", "report"))
    ap.add_argument("--tag", choices=tuple(KEYS), default="dev3k")
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args(argv)
    fetch(a.tag, a.workers) if a.cmd == "fetch" else report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
