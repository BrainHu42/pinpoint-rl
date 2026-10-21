# GLDv2 as a landmark gallery, step 2a (lesson R10): does same-landmark matching (MegaLoc) find the right place on commons26 dev?
# Usage: [GLD_EMB=megaloc|siglip|both] PYTHONPATH=src .venv/bin/python scripts/gldv2_pilot.py {keep,stream,embed,embed-siglip,report}
#   keep: a mini-gallery: the GLDv2 landmarks within 1 km of each commons26 dev photo's truth or of the pipeline's answer, plus random
#         distractor landmarks; <= 50 photos per landmark -> keep.json
#   stream: stream the 500 train tars from the public S3 bucket and keep only those photos (no tar is stored) -> images/
#   embed: MegaLoc for the gallery and the 977 queries (322 px, ImageNet normalisation, as megaloc_gate) -> megaloc.npz
#   report: retrieval over the mini-gallery (best photo per landmark), the switch "to the top landmark when its match beats a threshold", and how
#           the top-1 precision falls as distractors are added (the real gallery has 184k landmarks)
import csv
import io
import json
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import gldv2_ceiling as g  # noqa: E402

OUT = g.OUT
IMAGES = OUT / "images"
TAG, KEY = "commons26dev_setft_own", "commons26dev:setft_own|RL policy, lambda 0"
APPLY = Path("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz")
RELEASE = Path("data/pinpoint/commons26/release")
N_DISTRACT, PER_LANDMARK, RADIUS = 15000, 50, 1.0


def _landmarks():
    cats = {int(r["landmark_id"]): urllib.parse.unquote(r["category"].rsplit("Category:", 1)[-1]).replace("_", " ")
            for r in csv.DictReader(open(OUT / "train_label_to_category.csv"))}
    found = json.loads((OUT / "coords.json").read_text())
    ids = np.asarray(sorted(k for k in cats if cats[k] in found))
    return ids, np.asarray([found[cats[k]] for k in ids], np.float64)


def _photos():
    photos = json.loads((Path("artifacts/photo_sets") / TAG / "photos.json").read_text())
    z = np.load(APPLY)
    return photos, np.asarray([e["truth"] for e in photos], np.float64), z[f"{KEY}|center"].astype(np.float64), z[KEY]


def keep() -> None:
    from scipy.spatial import cKDTree

    ids, ll = _landmarks()
    photos, truth, centre, _ = _photos()
    tree = cKDTree(g._xyz(ll))
    near_t = tree.query_ball_point(g._xyz(truth), g._chord(RADIUS))
    near_a = tree.query_ball_point(g._xyz(centre), g._chord(RADIUS))
    local = sorted({int(ids[j]) for x in list(near_t) + list(near_a) for j in x})
    rng = np.random.default_rng(0)
    rest = np.setdiff1d(ids, local)
    distract = sorted(int(x) for x in rng.choice(rest, N_DISTRACT, replace=False))
    by_lm = {}
    for r in csv.DictReader(open(OUT / "train.csv")):
        by_lm.setdefault(int(r["landmark_id"]), []).append(r["id"])
    want = {}
    for lm in local + distract:
        imgs = sorted(by_lm.get(lm, []))
        if len(imgs) > PER_LANDMARK:
            imgs = sorted(np.random.default_rng(lm).choice(imgs, PER_LANDMARK, replace=False).tolist())
        for i in imgs:
            want[i] = lm
    (OUT / "keep.json").write_text(json.dumps({"local": local, "distract": distract, "images": want}))
    print(f"local landmarks {len(local)} (near a truth or an answer), distractors {len(distract)}; photos to keep {len(want)} "
          f"(local {sum(v in set(local) for v in want.values())})")


def _stream(k: int, want: set) -> tuple[int, int]:
    url = f"https://s3.amazonaws.com/google-landmark/train/images_{k:03d}.tar"
    for attempt in range(5):
        try:
            got = 0
            with urllib.request.urlopen(url, timeout=120) as r, tarfile.open(fileobj=r, mode="r|") as tar:
                for m in tar:
                    name = m.name.rsplit("/", 1)[-1][:-4]
                    if m.isfile() and name in want:
                        dst = IMAGES / name[:2] / f"{name}.jpg"
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(tar.extractfile(m).read())
                        got += 1
            return k, got
        except Exception as error:  # noqa: BLE001
            print(f"  tar {k} attempt {attempt}: {error}", flush=True)
            time.sleep(10 * (attempt + 1))
    return k, -1


def stream(workers: int = 8) -> None:
    want = set(json.loads((OUT / "keep.json").read_text())["images"])
    done_path = OUT / "stream_done.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    todo = [k for k in range(500) if k not in done]
    t0, total = time.time(), 0
    with ThreadPoolExecutor(workers) as pool:
        for n, (k, got) in enumerate(pool.map(lambda k: _stream(k, want), todo)):
            if got >= 0:
                done.add(k)
                total += got
                done_path.write_text(json.dumps(sorted(done)))
            if n % 10 == 0:
                print(f"  tars {len(done)}/500, kept {total} this run ({time.time() - t0:.0f}s)", flush=True)
    have = sum(1 for _ in IMAGES.rglob("*.jpg"))
    print(f"tars done {len(done)}/500; photos on disk {have} of {len(want)} ({time.time() - t0:.0f}s)")


def embed(batch: int = 64) -> None:
    import torch
    from PIL import Image

    from geo_search_env.experiment.megaloc_gate import SIZE, _model

    keepj = json.loads((OUT / "keep.json").read_text())
    photos, *_ = _photos()
    gallery = sorted(i for i in keepj["images"] if (IMAGES / i[:2] / f"{i}.jpg").exists())
    paths = [str(RELEASE / "images" / e["image_id"]) for e in photos] + [str(IMAGES / i[:2] / f"{i}.jpg") for i in gallery]
    mean, std = np.asarray([0.485, 0.456, 0.406], np.float32), np.asarray([0.229, 0.224, 0.225], np.float32)

    class D(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            try:
                im = Image.open(paths[i]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
                return torch.from_numpy(((np.asarray(im, np.float32) / 255.0 - mean) / std).transpose(2, 0, 1).copy()), True
            except Exception:  # noqa: BLE001
                return torch.zeros(3, SIZE, SIZE), False

    model = _model()
    out = np.zeros((len(paths), 8448), np.float16)
    ok = np.zeros(len(paths), bool)
    t0, s = time.time(), 0
    with torch.inference_mode():
        for x, good in torch.utils.data.DataLoader(D(), batch_size=batch, num_workers=4):
            out[s : s + len(x)] = model(x.cuda()).float().cpu().numpy()
            ok[s : s + len(x)] = good.numpy()
            s += len(x)
            if (s // batch) % 200 == 0:
                print(f"  {s}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(OUT / "megaloc.npz", queries=out[: len(photos)], gallery=out[len(photos) :], gallery_ids=np.asarray(gallery),
             ok=ok, n_queries=len(photos))
    print(f"embedded {len(photos)} queries + {len(gallery)} gallery photos ({(~ok).sum()} unreadable) in {time.time() - t0:.0f}s")


def embed_siglip(batch: int = 64) -> None:
    """SigLIP2 (Pinpoint's embedding, embed_cache) for the same queries and gallery photos as megaloc.npz -> siglip.npz."""

    import torch

    from geo_search_env.experiment.embed_cache import _embed, _load_model, _pixels

    z = np.load(OUT / "megaloc.npz")
    photos, *_ = _photos()
    gallery = z["gallery_ids"]
    paths = [str(RELEASE / "images" / e["image_id"]) for e in photos] + [str(IMAGES / i[:2] / f"{i}.jpg") for i in gallery]
    model, processor = _load_model()

    class D(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            x = _pixels(processor, Path(paths[i]).read_bytes())
            return (x, True) if x is not None else (torch.zeros(3, 384, 384), False)

    out = np.zeros((len(paths), 1536), np.float16)
    ok = np.zeros(len(paths), bool)
    t0, s = time.time(), 0
    for x, good in torch.utils.data.DataLoader(D(), batch_size=batch, num_workers=4):
        out[s : s + len(x)] = _embed(model, x)
        ok[s : s + len(x)] = good.numpy()
        s += len(x)
        if (s // batch) % 200 == 0:
            print(f"  {s}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(OUT / "siglip.npz", queries=out[: len(photos)], gallery=out[len(photos) :], gallery_ids=gallery, ok=ok & z["ok"],
             n_queries=len(photos))
    print(f"embedded {len(paths)} images with SigLIP2 in {time.time() - t0:.0f}s ({(~ok).sum()} unreadable)")


def _emb() -> str:
    """The embedding to evaluate: GLD_EMB=megaloc (default), siglip, or both (the two similarities averaged, see _sims)."""

    import os

    return os.environ.get("GLD_EMB", "megaloc")


def _sims(nq: int):
    """[gallery x queries] cosine similarity on the GPU, and the gallery's ok mask, for GLD_EMB."""

    import torch

    unit = lambda x: torch.nn.functional.normalize(torch.as_tensor(x.astype(np.float32)).cuda(), dim=-1)
    out, ok = None, None
    names = ("megaloc", "siglip") if _emb() == "both" else (_emb(),)
    for name in names:
        z = np.load(OUT / f"{name}.npz")
        s = unit(z["gallery"]) @ unit(z["queries"]).T
        out = s if out is None else out + s
        ok = z["ok"][nq:] if ok is None else ok & z["ok"][nq:]
    return out / len(names), ok


def report() -> None:
    import torch

    keepj = json.loads((OUT / "keep.json").read_text())
    z = np.load(OUT / "megaloc.npz")
    ids, ll = _landmarks()
    loc = dict(zip(ids.tolist(), ll))
    photos, truth, centre, km = _photos()
    nq = int(z["n_queries"])
    sims_t, ok = _sims(nq)  # gallery x 977
    sims_t = sims_t[torch.as_tensor(ok, device="cuda")]
    lm_of = np.asarray([keepj["images"][i] for i in z["gallery_ids"]])[ok]
    lms, inv = np.unique(lm_of, return_inverse=True)
    print(f"embedding: {_emb()}")
    idx = torch.as_tensor(inv, device="cuda")[:, None].expand_as(sims_t)
    best = torch.full((len(lms), nq), -1.0, device="cuda").scatter_reduce(0, idx, sims_t, "amax").T.cpu().numpy()  # best photo per landmark
    lm_ll = np.asarray([loc[int(x)] for x in lms])
    bm = {r["IMG_ID"]: r for r in csv.DictReader(open(RELEASE / "benchmark.csv"))}
    tier = np.asarray([bm[e["image_id"]]["tier"] for e in photos])
    distract = np.isin(lms, keepj["distract"])
    print(f"mini-gallery: {len(lms)} landmarks ({(~distract).sum()} near a truth or an answer, {distract.sum()} random distractors), "
          f"{len(G)} photos; queries {nq}; pipeline < 25 km {100 * np.mean(km < 25):.1f}%, < 1 km {100 * np.mean(km < 1):.1f}%")

    def run(cols, label, verbose):
        b = best[:, cols]
        top = b.argmax(1)
        s1 = b[np.arange(nq), top]
        tll = lm_ll[cols][top]
        d_truth = _dist(truth, tll)
        d_ans = _dist(centre, tll)
        print(f" {label}: top landmark < 1 / 25 km of the truth {100 * np.mean(d_truth < 1):.1f} / {100 * np.mean(d_truth < 25):.1f}%")
        if not verbose:
            return
        for t in ("landmark", "city", "region", "none"):
            m = tier == t
            print(f"   tier {t:8s} {m.sum():3d}: top landmark < 25 km {100 * np.mean(d_truth[m] < 25):5.1f}% (pipeline {100 * np.mean(km[m] < 25):5.1f}%)")
        print("   switch to the top landmark when its best match >= t (and it lies > 25 km from the pipeline's answer); pts of 977 at 25 km, "
              "and the 1 km refinement (move to the landmark when it is within 25 km of the answer)")
        for t in np.quantile(s1, [0.5, 0.75, 0.9, 0.95, 0.98, 0.99]):
            sw = (s1 >= t) & (d_ans > 25)
            new = np.where(sw, d_truth, km)
            fixes, breaks = int(((new < 25) & (km >= 25)).sum()), int(((new >= 25) & (km < 25)).sum())
            rf = (s1 >= t) & (d_ans <= 25)
            new1 = np.where(rf, d_truth, km)
            f1, b1 = int(((new1 < 1) & (km >= 1)).sum()), int(((new1 >= 1) & (km < 1)).sum())
            print(f"     t {t:.3f} ({100 * np.mean(s1 >= t):4.1f}% of photos): switches {sw.sum():3d}, fixes {fixes:3d}, breaks {breaks:3d} -> "
                  f"{100 * (fixes - breaks) / nq:+.2f} pts | 1 km refinement {rf.sum():3d}: fixes {f1}, breaks {b1} -> {100 * (f1 - b1) / nq:+.2f}")
        right = d_truth < 1
        print(f"   top match similarity: right landmark (< 1 km) median {np.median(s1[right]):.3f}, wrong {np.median(s1[~right]):.3f}; "
              f"AUC {np.mean(s1[right][:, None] > s1[~right][None]):.3f}")

    loc_cols = np.flatnonzero(~distract)
    dis_cols = np.flatnonzero(distract)
    rng = np.random.default_rng(1)
    for frac in (0.0, 0.25, 0.5, 1.0):
        cols = np.concatenate([loc_cols, rng.permutation(dis_cols)[: int(frac * len(dis_cols))]])
        run(cols, f"local + {int(frac * len(dis_cols))} distractor landmarks", verbose=frac == 1.0)


def _dist(a, b):
    return _hav(a[:, 0], a[:, 1], b[:, 0], b[:, 1])


def _hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


if __name__ == "__main__":
    {"keep": keep, "stream": stream, "embed": embed, "embed-siglip": embed_siglip, "report": report}[sys.argv[1]]()
