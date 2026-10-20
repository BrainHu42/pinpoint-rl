# OSV-5M as a candidate generator: nearest-neighbour retrieval over the 4.89M street-level SigLIP2 embeddings, and whether it proposes
# a right place (< 25 km) where the pipeline is wrong and none of its areas is right.
# Usage: .venv/bin/python -m geo_search_env.experiment.osv_propose {search,report}

"""Pinpoint retrieves candidates from MP16 only (OSV enters only as a support token around MP16 candidates, 4 nearest within 1 km), so a
place with no MP16 photo nearby is never proposed. Lesson 110: nothing within 25 km retrieved on 25.9% of dev3k misses.

search: top-K OSV images per query by raw SigLIP2 cosine (dev3k from the MP16 cache, commons26 dev from its cache), and the same for MP16
        (Pinpoint's index, photographer kept as in the original protocol, the query photo itself excluded) for comparison.
report: top-1 accuracy; recall of a < 25 km candidate in the top k; on photos where the pipeline (lesson 111 / 113 answer) is wrong,
        how often OSV proposes a right place and whether it is new (> 25 km from every area the pipeline considered).

Results (2026-10-17; dev3k 3,000 / commons26 dev 977; pipeline < 25 km 47.3 / 24.8%; scripts/clue_verify/osv_{unique,clusters,chance}.py):
- OSV top-1 < 25 km: raw 11.1 / 7.9%, contrastive 15.9 / 8.0% (MP16 33.0-34.8 / 12.5-17.8%). Recall top 20 / 50 (contrastive):
  dev3k 41.8 / 51.4%, commons26 30.0 / 40.5%.
- Pipeline wrong and OSV top 50 has a right place: dev3k 15.9% (chance, another photo's list: 1.7%), commons26 21.7% (0.7%).
  Right where no pipeline area and no MP16 top-50 photo is: dev3k 2.7-3.1%, commons26 5.3-7.6% (raw / contrastive).
- As extra areas (agreement clusters of the OSV top 50 not covered by a pipeline area): a right new area with the top 1 / 3 / 5 added
  on dev3k 0.7 / 1.5 / 2.6%, commons26 1.0 / 2.7 / 3.9% (raw similar). Given how rarely the head picks added areas (lessons 111-112),
  expect ~+0.3 on dev3k and ~+0.5 on commons26; an end-to-end test needs the areas through restricted search, zoom, MegaLoc and a refit.
- Gating by photo type (scripts/clue_verify/osv_gate.py): even on the most street-like 5% of dev3k (OSV top-1 sim above MP16's), OSV's
  best cluster is right 23.2% vs the pipeline's 39.1% (OSV right where the pipeline is wrong 6.0%, the reverse 21.9%); the same holds by
  OSV agreement and by pipeline confidence. A 5-fold CV switch (street-likeness, agreement, pipeline confidence) gains -0.03 / -0.10 pts
  (contrastive) and +0.03 / +0.10 (raw) at 25 km on dev3k / commons26 dev. OSV retrieval never beats the pipeline on any subset.
- Street-search action end to end (osv-search / hyps, scripts/osv_street_pipeline.sh, HYP_VARIANT=moreosv action_agent --ckpt-tag setft_own
  --one-km 0.5 --apply commons26dev:setft_own; trained on all of train): up to 3 OSV clusters per photo outside the existing areas become
  area tokens (source 2) with restricted search, zoom states and MegaLoc, revealed by a street action (cost 1). Ceiling: an OSV area right
  where the baseline agent is wrong and no existing area is right on 1.7% of dev3k / 2.9% of commons26 dev. % < 25 / 1 km, baseline (same
  run settings; commons26 dev MegaLoc recomputed, the earlier queries were blank) -> variant: dev3k RL lambda 0 47.3 / 30.8 -> 47.3 / 30.3,
  lambda 0.005 47.0 / 30.2 -> 47.3 / 30.2; commons26 dev lambda 0 25.2 / 7.7 -> 24.9 / 7.5, lambda 0.005 24.9 / 7.9 -> 24.3 / 7.6; fixed
  "everything" with vs without street +0.2 / +0.1. The head answers from an OSV area on 0.5% / 1.0% of photos (right 2 of 15 / 0 of 10;
  scripts/clue_verify/osv_street_picks.py). Why (osv_street_auc.py, train): 2.4% of OSV areas are right (Pinpoint's areas 13.9%); MP16
  evidence inside them ranks them (restricted top-1 AUC 0.85) but OSV's own vote is anti-informative (cluster support AUC 0.42: heavy OSV
  clusters away from Pinpoint's areas are confidently wrong). Not adopted; the action stays off unless HYP_VARIANT names OSV areas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .clue_verify import OSV_EMBED, _haversine
from .density_probe import _index_rows, _mp16_photo_ids
from .strategy_search import MP16_EMBED

OUT = Path("artifacts/osv_propose")
K = 50
CKPT_TAG = "setft_own"


def _queries():
    """[(set, photo index, truth, SigLIP2 embedding)] for dev3k and commons26 dev."""
    from ..data.benchmarks import DATA_ROOT, EMBEDDING_KEY

    out = []
    mp16 = np.memmap(MP16_EMBED / "embeddings.f16.bin", dtype=np.float16, mode="r").reshape(-1, 1536)
    for i, p in enumerate(json.loads(Path(f"artifacts/photo_sets/dev3k_{CKPT_TAG}/photos.json").read_text())):
        out.append(("dev3k", i, p["truth"], np.asarray(mp16[int(p["row"])]), int(p["image_id"].rsplit("_", 1)[-1][:-4])))
    root = DATA_ROOT / "commons26" / "release" / "image_embeddings" / EMBEDDING_KEY
    ids = (root / "image_ids.txt").read_text().splitlines()
    emb = np.fromfile(root / "embeddings.f16.bin", dtype=np.float16).reshape(len(ids), -1)
    row = {x: k for k, x in enumerate(ids)}
    for i, p in enumerate(json.loads(Path(f"artifacts/photo_sets/commons26_{CKPT_TAG}/photos.json").read_text())):
        if p.get("split") == "dev":
            out.append(("commons26", i, p["truth"], emb[row[p["image_id"]]], -1))
    return out


def _topk(q, emb, rows, chunk, exclude=None):
    import torch

    dev = torch.device("cuda")
    qt = torch.nn.functional.normalize(torch.as_tensor(q, device=dev).float(), dim=1).half()
    best_s = torch.full((len(q), K), -2.0, device=dev)
    best_i = torch.zeros((len(q), K), dtype=torch.int64, device=dev)
    t0 = time.time()
    for a in range(0, len(rows), chunk):
        r = rows[a:a + chunk]
        e = torch.nn.functional.normalize(torch.as_tensor(np.asarray(emb[r[0]:r[-1] + 1][r - r[0]], np.float32), device=dev), dim=1).half()
        s = (qt @ e.T).float()
        if exclude is not None:
            s.masked_fill_(exclude(a, a + len(r)), -2.0)
        cat_s = torch.cat([best_s, s], 1)
        cat_i = torch.cat([best_i, torch.arange(a, a + len(r), device=dev).expand(len(q), -1)], 1)
        best_s, j = cat_s.topk(K, dim=1)
        best_i = cat_i.gather(1, j)
        if a // chunk % 25 == 0:
            print(f"  {a + len(r):,}/{len(rows):,} ({time.time() - t0:.0f}s)", flush=True)
    return best_s.cpu().numpy(), best_i.cpu().numpy()


def search(chunk: int, space: str) -> None:
    """space raw: SigLIP2 cosine over the raw caches. space contrastive: Pinpoint's contrastive space (query through the image tower's mp16
    adapter, as the reranker does; OSV from the index's osv_support_cache, MP16 from the index's image embeddings)."""
    import torch

    qs = _queries()
    q = np.stack([x[3] for x in qs]).astype(np.float32)
    rows = _index_rows()
    mp16_pid = _mp16_photo_ids()
    mp16_ll = np.fromfile(MP16_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2)
    n = int(json.loads((OSV_EMBED / "manifest.json").read_text())["num_samples"])
    osv_ll = np.fromfile(OSV_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(n, 2)
    if space == "raw":
        osv_emb = np.memmap(OSV_EMBED / "embeddings.f16.bin", dtype=np.float16, mode="r").reshape(n, -1)
        mp16_emb, mp16_rows = np.memmap(MP16_EMBED / "embeddings.f16.bin", dtype=np.float16, mode="r").reshape(-1, 1536), rows
    else:
        from ..models.pinpoint_reranker import CONTRASTIVE_CHECKPOINT, _discover_index, _load_image_tower

        tower = _load_image_tower(CONTRASTIVE_CHECKPOINT)[0].cuda().eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            raw = torch.nn.functional.normalize(torch.as_tensor(q, device="cuda"), dim=-1)
            q = torch.nn.functional.normalize(tower(raw, "mp16").float(), dim=-1).cpu().numpy()
        index_dir = _discover_index()
        osv_root = index_dir / "osv_support_cache"
        man = json.loads((osv_root / "manifest.json").read_text())
        osv_emb = np.memmap(osv_root / man["files"]["embeddings"], dtype=np.float16, mode="r", shape=tuple(man["shapes"]["embeddings"]))
        osv_ll = np.memmap(osv_root / man["files"]["latlon_deg"], dtype=np.float32, mode="r", shape=tuple(man["shapes"]["latlon_deg"]))
        man = json.loads((index_dir / "manifest.json").read_text())
        mp16_emb = np.memmap(index_dir / man["files"]["image_embeddings"], dtype=np.float16, mode="r", shape=tuple(man["shapes"]["image_embeddings"]))
        mp16_rows = np.arange(len(rows))  # index order; index row r is MP16 cache row rows[r]
    print(f"{len(qs)} queries; OSV {len(osv_ll):,}; space {space}", flush=True)
    s_osv, i_osv = _topk(q, osv_emb, np.arange(len(osv_ll)), chunk)
    qpid = torch.as_tensor(np.asarray([x[4] for x in qs]), device="cuda")[:, None]
    pid_rows = torch.as_tensor(mp16_pid[rows], device="cuda")
    s_mp, i_mp = _topk(q, mp16_emb, mp16_rows, chunk, exclude=lambda a, b: qpid == pid_rows[a:b][None])
    OUT.mkdir(parents=True, exist_ok=True)
    name = "search.npz" if space == "raw" else f"search_{space}.npz"
    np.savez(OUT / name, set=np.asarray([x[0] for x in qs]), photo=np.asarray([x[1] for x in qs]),
             truth=np.asarray([x[2] for x in qs], np.float64), osv_sim=s_osv, osv_ll=np.asarray(osv_ll)[i_osv], mp16_sim=s_mp,
             mp16_ll=mp16_ll[rows[i_mp]])
    print(f"wrote {OUT / name}")


def _clusters(ll, sims, radius=25.0):
    """Greedy agreement clusters over a ranked list: [(centre lat, lon, summed weight)], heaviest first."""
    out = []
    for (la, lo), s in zip(ll, sims):
        w = float(np.exp((s - sims[0]) / 0.02))
        for c in out:
            if _haversine(la, lo, c[0], c[1]) < radius:
                c[2] += w
                break
        else:
            out.append([float(la), float(lo), w])
    return sorted(out, key=lambda c: -c[2])


def osv_search(tag: str, batch: int = 2048) -> None:
    """Top-K OSV images per photo of a photo set (e.g. train_setft_own) in Pinpoint's contrastive space, the query embedding as the reranker
    sees it (hypotheses.photo_world) -> OUT/osv_{tag}.npz (sim, latlon), photo-set order. All OSV embeddings sit on the GPU (2.5 GB)."""
    import torch

    from ..models.pinpoint_reranker import CONTRASTIVE_CHECKPOINT, _discover_index, _load_image_tower
    from .hypotheses import PHOTO_SETS, photo_world

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    world, qrows = photo_world(photos)
    q = np.asarray(world.query_embeddings[qrows], np.float32)
    root = _discover_index() / "osv_support_cache"
    man = json.loads((root / "manifest.json").read_text())
    emb = torch.as_tensor(np.fromfile(root / man["files"]["embeddings"], dtype=np.float16).reshape(man["shapes"]["embeddings"]), device="cuda")
    ll = np.fromfile(root / man["files"]["latlon_deg"], dtype=np.float32).reshape(man["shapes"]["latlon_deg"])
    tower = _load_image_tower(CONTRASTIVE_CHECKPOINT)[0].cuda().eval()
    sims, idx = [], []
    t0 = time.time()
    with torch.inference_mode():
        for a in range(0, len(q), batch):
            with torch.autocast("cuda", dtype=torch.float16):
                x = tower(torch.nn.functional.normalize(torch.as_tensor(q[a:a + batch], device="cuda"), dim=-1), "mp16")
            x = torch.nn.functional.normalize(x.float(), dim=-1).half()
            best_s, best_i = [], []
            for c in range(0, len(emb), 1_000_000):
                s, i = (x @ emb[c:c + 1_000_000].T).float().topk(K, dim=1)
                best_s.append(s)
                best_i.append(i + c)
            s, j = torch.cat(best_s, 1).topk(K, dim=1)
            sims.append(s.cpu().numpy())
            idx.append(torch.cat(best_i, 1).gather(1, j).cpu().numpy())
            print(f"  {a + len(x)} / {len(q)} ({time.time() - t0:.0f}s)", flush=True)
    idx = np.concatenate(idx)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"osv_{tag}.npz", sim=np.concatenate(sims), latlon=ll[idx])
    print(f"-> {OUT / f'osv_{tag}.npz'}", flush=True)


def hypotheses(tag: str, k: int = 3) -> None:
    """The street-search action's areas: per photo, the heaviest agreement clusters of its OSV top K not within BOX_KM of a Pinpoint area
    (rank < 5) or a clue place, up to k, as hypotheses (source 2, support = the cluster's share of the list's weight) with Pinpoint's
    restricted search around each -> artifacts/agent/hypotheses_{tag}_moreosv.npz (merged by hypothesis_compare / pinpoint_set
    --extra-hyps moreosv; HYP_VARIANT=moreosv downstream)."""
    from .hypotheses import BOX_KM, OUT as AGENT, PHOTO_SETS, restricted_search

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    d = np.load(OUT / f"osv_{tag}.npz")
    taken: dict[int, list] = {}
    for name in ("", "_cluehyps"):
        z = np.load(AGENT / f"hypotheses_{tag}{name}.npz")
        for p, src, r, c in zip(z["photo"], z["source"], z["rank"], z["center"]):
            if src == 1 or r < 5:
                taken.setdefault(int(p), []).append(c)
    hyps = []
    for p in range(len(photos)):
        cs = _clusters(d["latlon"][p], d["sim"][p], BOX_KM)
        total = sum(c[2] for c in cs)
        kept = []
        for c in cs:
            if all(_haversine(c[0], c[1], t[0], t[1]) >= BOX_KM for t in taken.get(p, []) + kept):
                kept.append(c)
                if len(kept) == k:
                    break
        hyps += [{"photo": p, "source": 2, "rank": r, "support": c[2] / total, "center": c[:2]} for r, c in enumerate(kept)]
    truth = np.asarray([e["truth"] for e in photos])
    right = np.zeros(len(photos), bool)
    for h in hyps:
        right[h["photo"]] |= _haversine(*truth[h["photo"]], *h["center"]) < BOX_KM
    print(f"{tag}: {len(hyps)} OSV hypotheses on {len({h['photo'] for h in hyps})} photos; a centre < {BOX_KM:g} km on {right.mean():.1%}", flush=True)
    restricted_search(photos, hyps, AGENT / f"hypotheses_{tag}_moreosv.npz")


def report(space: str = "raw") -> None:
    d = np.load(OUT / ("search.npz" if space == "raw" else f"search_{space}.npz"))
    ans = np.load(f"artifacts/agent/action_agent_apply_km_{CKPT_TAG}_1km0.5_commons26.npz")
    for st in ("dev3k", "commons26"):
        m = np.flatnonzero(d["set"] == st)
        truth = d["truth"][m]
        key = next((k for k in (f"{st}:{CKPT_TAG}|RL policy, lambda 0", f"{st}|RL policy, lambda 0") if k in ans.files), None)
        pipe = ans[key][d["photo"][m]] if key else None
        zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{st}_{CKPT_TAG}.npz")
        areas: dict[int, list] = {}
        for p, c in zip(zm["photo"], zm["center"]):
            areas.setdefault(int(p), []).append(c)
        print(f"\n== {st}: {len(m)} photos; pipeline < 25 km {100 * np.mean(pipe < 25):.1f}%" if pipe is not None else f"\n== {st}")
        for src in ("mp16", "osv"):
            ll = d[f"{src}_ll"][m]
            dist = _haversine(truth[:, None, 0], truth[:, None, 1], ll[..., 0], ll[..., 1])
            rec = {k: np.mean((dist[:, :k] < 25).any(1)) for k in (1, 5, 20, 50)}
            rec1 = np.mean(dist[:, 0] < 1)
            top_cluster = []
            for j in range(len(m)):
                c = _clusters(ll[j], d[f"{src}_sim"][m][j])[0]
                top_cluster.append(_haversine(truth[j, 0], truth[j, 1], c[0], c[1]) < 25)
            print(f"  {src:4s} {space}: top-1 < 1 km {100 * rec1:4.1f}%, < 25 km {100 * rec[1]:4.1f}%; top cluster < 25 km "
                  f"{100 * np.mean(top_cluster):4.1f}%; any of top 5 / 20 / 50 < 25 km {100 * rec[5]:4.1f} / {100 * rec[20]:4.1f} / {100 * rec[50]:4.1f}%")
            if src == "osv" and pipe is not None:
                wrong = pipe >= 25
                for k in (5, 20, 50):
                    hit = (dist[:, :k] < 25).any(1)
                    new = np.zeros(len(m), bool)
                    for j in np.flatnonzero(wrong & hit):
                        right = [(la, lo) for (la, lo), dd in zip(ll[j, :k], dist[j, :k]) if dd < 25]
                        cs = areas.get(int(d["photo"][m][j]), [])
                        new[j] = any(all(_haversine(la, lo, c[0], c[1]) >= 25 for c in cs) for la, lo in right)
                    print(f"    pipeline wrong ({100 * wrong.mean():.1f}%): OSV top {k:2d} has a right place on {100 * np.mean(wrong & hit):4.1f}% of photos, "
                          f"one outside every pipeline area on {100 * np.mean(new):4.1f}%")
                c_ok = np.asarray(top_cluster)
                print(f"    OSV top cluster right where the pipeline is wrong: {100 * np.mean(wrong & c_ok):.1f}% of photos; "
                      f"where right, OSV top cluster wrong {100 * np.mean(~wrong & ~c_ok):.1f}%")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("search", "report", "osv-search", "hyps"))
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--space", choices=("raw", "contrastive"), default="raw")
    ap.add_argument("--tag", help="osv-search / hyps: photo set (train_setft_own, dev3k_setft_own, commons26dev_setft_own)")
    ap.add_argument("--k", type=int, default=3, help="hyps: OSV areas per photo")
    a = ap.parse_args(argv)
    if a.cmd == "search":
        search(a.chunk, a.space)
    elif a.cmd == "report":
        report(a.space)
    elif a.cmd == "osv-search":
        osv_search(a.tag)
    else:
        hypotheses(a.tag, a.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
