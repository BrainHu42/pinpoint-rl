# GLDv2 as a landmark gallery, the full index (lesson R10 on): every landmark with Wikidata coordinates, <= 50 photos each (~2.7M photos).
# Usage: PYTHONPATH=src .venv/bin/python scripts/gldv2_index.py {keep,stream,embed,search} [--set commons26dev|dev3k|train|...]
#   keep / stream: as gldv2_pilot (photos already on disk from the pilot are skipped) -> index_keep.json, images/
#   embed: MegaLoc in resumable shards of 100k photos -> index/shard_XXX.npz (fp16 8448-d, ids)
#   search: per query photo, the top landmarks by best-photo MegaLoc similarity over the whole index -> search_<set>.npz
#           (landmark ids, lat / lon, best similarity, photos of that landmark among the query's top 2,048 photos; and the raw top photos)
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import gldv2_pilot as p  # noqa: E402

OUT = p.OUT
INDEX = OUT / "index"
SHARD, TOP_PHOTOS, TOP_LM = 100_000, 2048, 32
# query MegaLoc embeddings per photo set (megaloc_gate's caches, same preprocessing); commons26dev from the pilot
QUERIES = {"dev3k": Path("artifacts/agent/megaloc_queries_dev3k_setft_own.npy"), "train": Path("artifacts/agent/megaloc_queries_train_setft_own.npy")}


def keep() -> None:
    ids, _ = p._landmarks()
    lm_set = set(ids.tolist())
    by_lm = {}
    for r in csv.DictReader(open(OUT / "train.csv")):
        if int(r["landmark_id"]) in lm_set:
            by_lm.setdefault(int(r["landmark_id"]), []).append(r["id"])
    want = {}
    for lm, imgs in by_lm.items():
        imgs = sorted(imgs)
        if len(imgs) > p.PER_LANDMARK:  # the same per-landmark sample as the pilot
            imgs = sorted(np.random.default_rng(lm).choice(imgs, p.PER_LANDMARK, replace=False).tolist())
        for i in imgs:
            want[i] = lm
    (OUT / "index_keep.json").write_text(json.dumps({"images": want}))
    print(f"{len(by_lm)} landmarks, {len(want)} photos")


def stream(workers: int = 8) -> None:
    import gldv2_pilot

    keep_path, done_path = OUT / "index_keep.json", OUT / "index_stream_done.json"
    want = set(json.loads(keep_path.read_text())["images"])
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    from concurrent.futures import ThreadPoolExecutor

    t0, total = time.time(), 0
    with ThreadPoolExecutor(workers) as pool:
        for n, (k, got) in enumerate(pool.map(lambda k: gldv2_pilot._stream(k, want), [k for k in range(500) if k not in done])):
            if got >= 0:
                done.add(k)
                total += got
                done_path.write_text(json.dumps(sorted(done)))
            if n % 10 == 0:
                print(f"  tars {len(done)}/500, written {total} this run ({time.time() - t0:.0f}s)", flush=True)
    print(f"tars done {len(done)}/500 ({time.time() - t0:.0f}s)")


def embed(batch: int = 64) -> None:
    import torch
    from PIL import Image

    from geo_search_env.experiment.megaloc_gate import SIZE, _model

    want = json.loads((OUT / "index_keep.json").read_text())["images"]
    ids = sorted(i for i in want if (p.IMAGES / i[:2] / f"{i}.jpg").exists())
    print(f"{len(ids)} of {len(want)} photos on disk", flush=True)
    INDEX.mkdir(exist_ok=True)
    mean, std = np.asarray([0.485, 0.456, 0.406], np.float32), np.asarray([0.229, 0.224, 0.225], np.float32)

    class D(torch.utils.data.Dataset):
        def __init__(self, part):
            self.part = part

        def __len__(self):
            return len(self.part)

        def __getitem__(self, i):
            try:
                im = Image.open(p.IMAGES / self.part[i][:2] / f"{self.part[i]}.jpg").convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
                return torch.from_numpy(((np.asarray(im, np.float32) / 255.0 - mean) / std).transpose(2, 0, 1).copy()), True
            except Exception:  # noqa: BLE001
                return torch.zeros(3, SIZE, SIZE), False

    model = _model()
    t0 = time.time()
    for k, s in enumerate(range(0, len(ids), SHARD)):
        path = INDEX / f"shard_{k:03d}.npz"
        if path.exists():
            continue
        part = ids[s : s + SHARD]
        out = np.zeros((len(part), 8448), np.float16)
        ok = np.zeros(len(part), bool)
        j = 0
        with torch.inference_mode():
            for x, good in torch.utils.data.DataLoader(D(part), batch_size=batch, num_workers=4):
                out[j : j + len(x)] = model(x.cuda()).float().cpu().numpy()
                ok[j : j + len(x)] = good.numpy()
                j += len(x)
        np.savez(path, emb=out[ok], ids=np.asarray(part)[ok], lm=np.asarray([want[i] for i in np.asarray(part)[ok]]))
        print(f"  shard {k}: {ok.sum()} photos ({time.time() - t0:.0f}s, {s + len(part)}/{len(ids)})", flush=True)


def _queries(name: str) -> np.ndarray:
    if name == "commons26dev":
        z = np.load(OUT / "megaloc.npz")
        return z["queries"]
    return np.load(QUERIES[name], mmap_mode="r")


def search(name: str, chunk: int = 4096) -> None:
    import torch

    ids, ll = p._landmarks()
    loc = dict(zip(ids.tolist(), ll))
    Qall = _queries(name)
    nq = len(Qall)
    shards = sorted(INDEX.glob("shard_*.npz"))
    top_s = torch.full((nq, TOP_PHOTOS), -2.0)
    top_l = torch.zeros((nq, TOP_PHOTOS), dtype=torch.long)
    t0 = time.time()
    for sp in shards:
        z = np.load(sp)
        G = torch.nn.functional.normalize(torch.as_tensor(z["emb"]).cuda().float(), dim=-1).half()
        lm = torch.as_tensor(z["lm"].astype(np.int64)).cuda()
        for s in range(0, nq, chunk):
            q = torch.nn.functional.normalize(torch.as_tensor(np.asarray(Qall[s : s + chunk], np.float32)).cuda(), dim=-1).half()
            sims = (q @ G.T).float()
            v, i = sims.topk(TOP_PHOTOS, 1)
            cat_s = torch.cat([top_s[s : s + chunk].cuda(), v], 1)
            cat_l = torch.cat([top_l[s : s + chunk].cuda(), lm[i]], 1)
            v2, i2 = cat_s.topk(TOP_PHOTOS, 1)
            top_s[s : s + chunk], top_l[s : s + chunk] = v2.cpu(), cat_l.gather(1, i2).cpu()
        del G
        print(f"  {sp.name} ({time.time() - t0:.0f}s)", flush=True)
    S, L = top_s.numpy(), top_l.numpy()
    out_l = np.zeros((nq, TOP_LM), np.int64)
    out_s = np.full((nq, TOP_LM), -2.0, np.float32)
    out_n = np.zeros((nq, TOP_LM), np.int16)
    for q in range(nq):  # photos come sorted by similarity: the first occurrence of a landmark is its best photo
        u, first, cnt = np.unique(L[q], return_index=True, return_counts=True)
        keep_ = np.argsort(first)[:TOP_LM]
        m = len(keep_)
        out_l[q, :m], out_s[q, :m], out_n[q, :m] = u[keep_], S[q, first[keep_]], cnt[keep_]
    lat_lon = np.asarray([[loc.get(int(l_), (np.nan, np.nan)) for l_ in row] for row in out_l], np.float64)
    np.savez(OUT / f"search_{name}.npz", landmark=out_l, sim=out_s, count=out_n, latlon=lat_lon, photo_sim=S.astype(np.float16),
             photo_lm=L.astype(np.int32))
    print(f"{name}: {nq} queries searched over {len(shards)} shards in {time.time() - t0:.0f}s -> {OUT / f'search_{name}.npz'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("keep", "stream", "embed", "search"))
    ap.add_argument("--set", default="commons26dev")
    a = ap.parse_args()
    {"keep": keep, "stream": stream, "embed": embed}.get(a.cmd, lambda: search(a.set))()
