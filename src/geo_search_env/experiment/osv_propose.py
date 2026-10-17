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
    ap.add_argument("cmd", choices=("search", "report"))
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--space", choices=("raw", "contrastive"), default="raw")
    a = ap.parse_args(argv)
    search(a.chunk, a.space) if a.cmd == "search" else report(a.space)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
