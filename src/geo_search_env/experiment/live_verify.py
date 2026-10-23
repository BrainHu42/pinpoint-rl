# Live verification of text-proposed places with fresh Commons imagery (TOOL_USE_DESIGN.md, first test). Lesson S3: on 116 commons26 dev
# photos a right text candidate exists but the pipeline is wrong, and offline checks (restricted Pinpoint, MegaLoc of its top) rank the right
# place above the text ranker's wrong pick on only 16-37%. Lesson 135 tried fetched 2004-14 Flickr photos (MegaLoc AUC 0.60). Here: the nearest
# guarded Commons files (tools.imagery: uploaded before the query was taken, not by its uploader) at three places per photo (the best right
# text candidate, the ranker's wrong pick, the pipeline's answer), scored by MegaLoc against the query.
# Pre-declared go: the right place beats the wrong pick on well over half of the 116 (then the switch on all 977 dev photos, bar +1 pt).
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.live_verify targets   (CPU; text_verify's T ranker, as lesson S3)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.live_verify fetch     (network, resumable through the tool's cache)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.live_verify score     (GPU, MegaLoc)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.live_verify report

"""targets: the 116 photos and their three places -> artifacts/tool_use/targets_commons26dev.json.
fetch:   up to N guarded files within KM of each place, thumbnails cached -> artifacts/tool_use/fetched_commons26dev.json.
score:   MegaLoc of every thumbnail vs its query -> artifacts/tool_use/scores_commons26dev.json.
report:  right vs wrong pick vs the pipeline's place: coverage, who scores higher, AUC, thresholds -> artifacts/tool_use/report_commons26dev.txt."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/tool_use")
AGENT = Path("artifacts/agent")
KM = {"commons": 10.0, "mapillary": 1.0}  # radius per source (Commons: the API's cap; Mapillary: street level, +-1 km box)
N = 20  # files per place (--n)
SOURCE = "commons"  # --source
FETCH = ("right", "wrong pick", "pipeline", "truth")  # --roles


def _out(kind: str, ext: str = "json") -> Path:
    """artifacts/tool_use/{kind}_commons26dev[_mapillary][_n200].{ext}"""

    return OUT / f"{kind}_commons26dev{'' if SOURCE == 'commons' else '_' + SOURCE}{'' if N == 20 else f'_n{N}'}.{ext}"
ROLES = ("right", "wrong pick", "pipeline")


def targets() -> None:
    from . import text_verify as tv
    from .search_tree import _km, _photos

    z = np.load(tv.BUILT)
    photos = _photos(tv.TAG)
    n = len(photos)
    bench = {r["IMG_ID"]: r for r in csv.DictReader(open("data/pinpoint/commons26/release/benchmark.csv", encoding="utf-8"))}
    fold = tv._folds([bench[e["image_id"]]["group"] or e["image_id"] for e in photos])
    apply = np.load(tv.APPLY)
    ok = apply[tv.KEY] < 25
    answer = apply[f"{tv.KEY}|center"]
    photo, y, m = z["photo"], z["y"], z["source"] == 0
    tp = [np.flatnonzero(m & (photo == p)) for p in range(n)]
    width = max(len(t) for t in tp)
    X = np.zeros((n, width, z["X"].shape[1]), np.float32)
    M, Y, idx = np.zeros((n, width), bool), np.zeros((n, width), bool), np.full((n, width), -1)
    for p, t in enumerate(tp):
        X[p, : len(t)], M[p, : len(t)], Y[p, : len(t)], idx[p, : len(t)] = z["X"][t], True, y[t], t
    P = np.zeros(M.shape, np.float32)
    for k in range(5):  # lesson 119's linear ranker, uploader 5-fold (text_verify evaluate, section 4)
        tr, te = fold != k, fold == k
        P[te] = tv._fit_predict(X[tr], M[tr], Y[tr], X[te], M[te], 0)
    top = P.argmax(1)
    top_right = M.any(1) & Y[np.arange(n), top]
    sel = np.flatnonzero(Y.any(1) & ~top_right & ~ok)
    out = []
    for p in sel:
        right = idx[p][M[p] & Y[p]][np.argmax(P[p][M[p] & Y[p]])]
        e = photos[p]
        b = bench[e["image_id"]]
        places = [z["centre"][right], z["centre"][idx[p, top[p]]], answer[p]]
        out.append({"photo": int(p), "image_id": e["image_id"], "group": b["group"], "taken": b["taken"], "truth": list(map(float, e["truth"])),
                    "places": [{"role": r, "lat": float(c[0]), "lon": float(c[1]), "km_to_truth": float(_km(np.asarray(c), np.asarray(e["truth"])))}
                               for r, c in zip(ROLES, places)]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "targets_commons26dev.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(out)} target photos (lesson S3: 116) -> {OUT / 'targets_commons26dev.json'}")


def fetch(workers: int = 6) -> None:
    """Places run `workers` at a time: API calls stay paced by the tool's shared limiter, thumbnails (CDN) download unpaced."""

    from ..tools.imagery import SOURCES

    tool = SOURCES[SOURCE]()
    tg = json.loads((OUT / "targets_commons26dev.json").read_text(encoding="utf-8"))
    truth = lambda t: {"role": "truth", "lat": t["truth"][0], "lon": t["truth"][1]}  # control: imagery at the photo's own location
    jobs = [(t, pl) for t in tg for pl in t["places"] + [truth(t)] if pl["role"] in FETCH]

    def one(job):
        t, pl = job
        try:
            photos = tool.near(pl["lat"], pl["lon"], KM[SOURCE], N, before=t["taken"], exclude_group=t["group"])
            error = ""
        except RuntimeError as e:  # a place the source can't serve counts as uncovered (reported), not a crash
            photos, error = [], str(e)
            print(f"  {error}", flush=True)
        with ThreadPoolExecutor(4) as tx:
            paths = list(tx.map(tool.thumb, photos))
        return {"photo": t["photo"], "role": pl["role"], "error": error, "files": [
            {"page_id": p.page_id, "title": p.title, "dist_m": p.dist_m, "uploaded": p.uploaded, "path": str(path)}
            for p, path in zip(photos, paths) if path is not None]}

    fetched = []
    with ThreadPoolExecutor(workers) as ex:
        for i, row in enumerate(ex.map(one, jobs)):
            fetched.append(row)
            if i % 20 == 0:
                print(f"  {i + 1} / {len(jobs)} places; {sum(len(f['files']) for f in fetched)} files", flush=True)
    _out("fetched").write_text(json.dumps(fetched), encoding="utf-8")
    print(f"-> {_out('fetched')}: {sum(len(f['files']) for f in fetched)} files for {len(fetched)} places")


def score(batch: int = 64) -> None:
    import io

    from PIL import Image
    import torch

    from . import megaloc_gate as mg

    fetched = json.loads(_out("fetched").read_text(encoding="utf-8"))
    q = torch.as_tensor(np.load(AGENT / "megaloc_queries_commons26dev_setft_own.npy"), device="cuda").float()  # commons26dev photo order
    assert float((q[:100] @ q[:100].T).mean()) < 0.5, "blank MegaLoc query embeddings"
    paths = sorted({f["path"] for pl in fetched for f in pl["files"]})
    mean, std = np.asarray([0.485, 0.456, 0.406], np.float32), np.asarray([0.229, 0.224, 0.225], np.float32)

    def load(path):
        try:
            im = Image.open(io.BytesIO(Path(path).read_bytes())).convert("RGB").resize((mg.SIZE, mg.SIZE), Image.BILINEAR)
            return torch.from_numpy(((np.asarray(im, np.float32) / 255.0 - mean) / std).transpose(2, 0, 1).copy())
        except Exception:  # noqa: BLE001
            return None

    model = mg._model()
    emb, bad = {}, 0
    with torch.inference_mode():
        for s in range(0, len(paths), batch):
            xs = [(p, load(p)) for p in paths[s: s + batch]]
            good = [(p, x) for p, x in xs if x is not None]
            bad += len(xs) - len(good)
            if good:
                e = model(torch.stack([x for _, x in good]).cuda()).float()
                emb.update({p: v for (p, _), v in zip(good, e)})
    others = sorted({pl["photo"] for pl in fetched})
    for pl in fetched:
        for f in pl["files"]:
            f["megaloc"] = float((emb[f["path"]] * q[pl["photo"]]).sum()) if f["path"] in emb else None
        e = [emb[f["path"]] for f in pl["files"] if f["path"] in emb]
        if e:  # null: the same imagery's best match to the other target queries (generic or plentiful imagery matches everyone)
            best = (torch.stack(e) @ q[[o for o in others if o != pl["photo"]]].T).max(0).values
            pl["null_mu"], pl["null_sd"] = float(best.mean()), float(best.std())
    _out("scores").write_text(json.dumps(fetched), encoding="utf-8")
    print(f"-> {_out('scores')}: {len(emb)} thumbnails embedded, {bad} unreadable")


def report() -> None:
    """Per place: best MegaLoc match (and top-3 mean), and z = (best - null mean) / null sd against the same imagery's best matches to the
    other target queries. Each role vs the wrong pick: share of photos where it scores higher (no imagery = lowest) and AUC. Truth is the
    control: imagery at the photo's own location, the ceiling of any candidate."""

    from .search_tree import _auc

    sc = json.loads(_out("scores").read_text(encoding="utf-8"))
    tg = {t["photo"]: t for t in json.loads((OUT / "targets_commons26dev.json").read_text(encoding="utf-8"))}
    photos = sorted({pl["photo"] for pl in sc})
    roles = [r for r in ROLES + ("truth",) if any(pl["role"] == r for pl in sc)]
    place = {(pl["photo"], pl["role"]): pl for pl in sc}

    def stat(p, r, kind):
        pl = place.get((p, r))
        v = sorted((f["megaloc"] for f in pl["files"] if f["megaloc"] is not None), reverse=True) if pl else []
        if not v:
            return -1.0 if kind != "z" else -9.0  # no imagery: lowest
        if kind == "z":
            return (v[0] - pl["null_mu"]) / max(pl["null_sd"], 1e-3)
        return float(np.mean(v[: 1 if kind == "best" else 3]))

    lines = [f"Live verification with guarded {SOURCE} imagery (<= {N} files within {KM[SOURCE]:g} km, made before the query was taken"
             f"{', not by its uploader' if SOURCE == 'commons' else ''}); {len(photos)} commons26 dev photos with a right text candidate and a "
             f"wrong pipeline answer (lesson S3); truth = control at the photo's own location"]
    for r in roles:
        cnt = np.asarray([len(place[(p, r)]["files"]) if (p, r) in place else 0 for p in photos])
        best = np.asarray([stat(p, r, "best") for p in photos])
        z = np.asarray([stat(p, r, "z") for p in photos])
        lines.append(f"  {r:10s}: any file {100 * np.mean(cnt > 0):.0f}%, files median {np.median(cnt):.0f}; best MegaLoc median "
                     f"{np.median(best[cnt > 0]):.3f}, >= 0.2 / 0.3 / 0.4 on {100 * np.mean(best >= 0.2):.0f} / {100 * np.mean(best >= 0.3):.0f} / "
                     f"{100 * np.mean(best >= 0.4):.0f}%; z median {np.median(z[cnt > 0]):.2f}, z >= 3 on {100 * np.mean(z >= 3):.0f}%")
    if "wrong pick" in roles:
        for kind in ("best", "top3", "z"):
            w = np.asarray([stat(p, "wrong pick", kind) for p in photos])
            for r in [r for r in roles if r != "wrong pick"]:
                v = np.asarray([stat(p, r, kind) for p in photos])
                lab = np.r_[np.ones(len(v), bool), np.zeros(len(w), bool)]
                lines.append(f"  {kind:4s}: {r:8s} above the wrong pick on {100 * np.mean(v > w):.0f}% of photos, AUC {_auc(np.r_[v, w], lab):.3f}"
                             f"{'  (offline checks 16-37%; lesson 135 AUC 0.60)' if r == 'right' else ''}")
    km = np.asarray([next(pl["km_to_truth"] for pl in tg[p]["places"] if pl["role"] == "right") for p in photos])
    lines.append(f"  right place to the photo's location: median {np.median(km):.1f} km, within 1 km {100 * np.mean(km < 1):.0f}%, "
                 f"within {KM[SOURCE]:g} km {100 * np.mean(km < KM[SOURCE]):.0f}%")
    text = "\n".join(lines)
    _out("report", "txt").write_text(text + "\n")
    print(text)


def main(argv: Sequence[str] | None = None) -> int:
    global SOURCE, N, FETCH
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("targets", "fetch", "score", "report"))
    parser.add_argument("--source", choices=tuple(KM), default="commons", help="imagery source (outputs of non-commons sources get a suffix)")
    parser.add_argument("--n", type=int, default=N, help="files per place (outputs of other values get a suffix)")
    parser.add_argument("--roles", default=",".join(FETCH), help="places to fetch, comma-separated (wrong_pick = 'wrong pick')")
    args = parser.parse_args(argv)
    SOURCE, N, FETCH = args.source, args.n, tuple(r.replace("_", " ") for r in args.roles.split(","))
    {"targets": targets, "fetch": fetch, "score": score, "report": report}[args.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
