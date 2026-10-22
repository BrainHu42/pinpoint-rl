# Mapillary street-level imagery as a verifier on dev3k (lesson 138 tested only commons26 text places). The answer head's top two areas
# (25 km head trained on the full state, lesson R11) where exactly one is right (< 25 km): fetch the nearest Mapillary frames around each
# area's point (its restricted top-1) and the truth (control), score each place by its best MegaLoc match to the query, and z against the
# same frames' best matches to the other target queries. Pre-declared go: the right area beats the wrong one on well over half of the pairs
# AND the verifier adds to the head on the same pairs (the head ranks the right one first on X%; a combined rule must beat X).
# Usage: PYTHONPATH=src .venv/bin/python scripts/mapillary_dev3k.py targets
#        PYTHONPATH=src .venv/bin/python scripts/mapillary_dev3k.py fetch [--n 30] [--workers 6]
#        PYTHONPATH=src .venv/bin/python scripts/mapillary_dev3k.py score
#        PYTHONPATH=src .venv/bin/python scripts/mapillary_dev3k.py report
import argparse
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
OUT = Path("artifacts/tool_use/dev3k")
TAG, CKPT = "dev3k", "setft_own"
KM = 1.0
ROLES = ("top", "second", "truth")


def _hav(a, b):
    a, b = np.radians(np.asarray(a, np.float64)), np.radians(np.asarray(b, np.float64))
    h = np.sin((b[..., 0] - a[..., 0]) / 2) ** 2 + np.cos(a[..., 0]) * np.cos(b[..., 0]) * np.sin((b[..., 1] - a[..., 1]) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def targets() -> None:
    import torch

    import structure_probe as sp
    from geo_search_env.experiment import action_agent as aa
    from geo_search_env.experiment.hypothesis_agent import SOURCES, _sub, area_points

    aa.CKPT_TAG = CKPT
    tr = aa._data("train")
    dv = aa._data(TAG, stats=tr["stats"])
    net = sp.fit(tr, np.arange(tr["n"]), 7, 0.0, True)
    b = _sub(dv, torch.arange(dv["n"], device="cuda"))
    on = torch.ones(dv["n"], dtype=torch.bool, device="cuda")
    v = aa.view(b, on, on)
    with torch.no_grad():
        prob = torch.softmax(net.answer_logits(v, v["mask"]), 1).cpu().numpy()
    # area points in the padded slot order of hypothesis_agent.load
    photo, source, point = area_points(TAG, CKPT)
    keep = np.isin(source, SOURCES)
    photo, point = photo[keep], point[keep]
    n, h = prob.shape
    pts = np.full((n, h, 2), np.nan)
    seen = np.zeros(n, int)
    for p, ll in zip(photo, point):
        pts[p, seen[p]] = ll
        seen[p] += 1
    assert (seen == v["mask"].sum(1).cpu().numpy()).all()
    photos = json.loads((Path("artifacts/photo_sets") / f"{TAG}_{CKPT}" / "photos.json").read_text(encoding="utf-8"))
    truth = np.asarray([e["truth"] for e in photos], np.float64)
    right = _hav(truth[:, None], pts) < 25
    order = np.argsort(-prob, 1)
    out = []
    for i in range(n):
        if seen[i] < 2:
            continue
        a, c = order[i, 0], order[i, 1]
        if right[i, a] == right[i, c]:
            continue
        out.append({"photo": i, "image_id": photos[i]["image_id"], "truth": truth[i].tolist(), "head_right": bool(right[i, a]),
                    "p_top": float(prob[i, a]), "p_second": float(prob[i, c]),
                    "places": [{"role": "top", "lat": float(pts[i, a, 0]), "lon": float(pts[i, a, 1]), "right": bool(right[i, a])},
                               {"role": "second", "lat": float(pts[i, c, 0]), "lon": float(pts[i, c, 1]), "right": bool(right[i, c])},
                               {"role": "truth", "lat": float(truth[i, 0]), "lon": float(truth[i, 1]), "right": True}]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "targets.json").write_text(json.dumps(out, indent=1))
    hr = np.mean([t["head_right"] for t in out])
    print(f"{len(out)} dev3k photos with exactly one right area in the head's top two (head picks the right one on {hr:.1%}; "
          f"head top-1 right overall {np.mean(right[np.arange(n), order[:, 0]]):.1%}) -> {OUT / 'targets.json'}", flush=True)


def fetch(n_files: int, workers: int) -> None:
    from geo_search_env.tools.imagery import MapillaryImagery

    tool = MapillaryImagery()
    tg = json.loads((OUT / "targets.json").read_text())
    jobs = [(t, pl) for t in tg for pl in t["places"]]

    def one(job):
        t, pl = job
        try:
            photos, error = tool.near(pl["lat"], pl["lon"], KM, n_files), ""
        except Exception as e:  # noqa: BLE001  (a failed or malformed tile: the place counts as uncovered)
            photos, error = [], f"{type(e).__name__}: {e}"[:200]
        with ThreadPoolExecutor(8) as tx:
            paths = list(tx.map(tool.thumb, photos))
        return {"photo": t["photo"], "role": pl["role"], "error": error,
                "files": [{"id": p.page_id, "dist_m": p.dist_m, "captured": p.uploaded, "path": str(q)} for p, q in zip(photos, paths) if q is not None]}

    partial = OUT / "fetched_partial.json"  # progress, saved every 200 places; error-free places are reused on a restart
    done = {(r["photo"], r["role"]): r for r in (json.loads(partial.read_text()) if partial.exists() else []) if not r["error"]}

    def save(rows):
        tmp = partial.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows))
        tmp.replace(partial)

    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for i, row in enumerate(ex.map(lambda j: done.get((j[0]["photo"], j[1]["role"])) or one(j), jobs)):
            rows.append(row)
            if i % 200 == 0:
                print(f"  {i + 1} / {len(jobs)} places; {sum(len(r['files']) for r in rows)} files", flush=True)
                save(rows)
    save(rows)
    (OUT / "fetched.json").write_text(json.dumps(rows))
    print(f"-> {OUT / 'fetched.json'}: {sum(len(r['files']) for r in rows)} files for {len(rows)} places", flush=True)


def score(batch: int = 64) -> None:
    import torch
    from PIL import Image

    from geo_search_env.experiment import megaloc_gate as mg

    rows = json.loads((OUT / "fetched.json").read_text())
    q = torch.as_tensor(np.load(f"artifacts/agent/megaloc_queries_{TAG}_{CKPT}.npy"), device="cuda").float()
    assert float((q[:100] @ q[:100].T).mean()) < 0.5, "blank MegaLoc query embeddings"
    paths = sorted({f["path"] for r in rows for f in r["files"]})
    mean, std = np.asarray([0.485, 0.456, 0.406], np.float32), np.asarray([0.229, 0.224, 0.225], np.float32)

    def load(path):
        try:
            im = Image.open(io.BytesIO(Path(path).read_bytes())).convert("RGB").resize((mg.SIZE, mg.SIZE), Image.BILINEAR)
            return torch.from_numpy(((np.asarray(im, np.float32) / 255.0 - mean) / std).transpose(2, 0, 1).copy())
        except Exception:  # noqa: BLE001
            return None

    model = mg._model()
    emb = {}
    with torch.inference_mode(), ThreadPoolExecutor(4) as ex:
        for s in range(0, len(paths), batch):
            good = [(p, x) for p, x in zip(paths[s: s + batch], ex.map(load, paths[s: s + batch])) if x is not None]
            if good:
                e = model(torch.stack([x for _, x in good]).cuda()).float()
                emb.update({p: v for (p, _), v in zip(good, e)})
    others = np.asarray(sorted({r["photo"] for r in rows}))
    rng = np.random.default_rng(0)
    null_q = q[rng.choice(others, min(300, len(others)), replace=False)]
    for r in rows:
        e = [emb[f["path"]] for f in r["files"] if f["path"] in emb]
        for f in r["files"]:
            f["megaloc"] = float((emb[f["path"]] * q[r["photo"]]).sum()) if f["path"] in emb else None
        if e:  # null: the same frames' best match to other target queries
            best = (torch.stack(e) @ null_q.T).max(0).values
            r["null_mu"], r["null_sd"] = float(best.mean()), float(best.std())
    (OUT / "scores.json").write_text(json.dumps(rows))
    print(f"-> {OUT / 'scores.json'}: {len(emb)} of {len(paths)} frames embedded", flush=True)


def _auc(s, y):
    from scipy.stats import rankdata

    r = rankdata(s)
    npos, nneg = y.sum(), (~y).sum()
    return (r[y].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1)


def report() -> None:
    sc = json.loads((OUT / "scores.json").read_text())
    tg = {t["photo"]: t for t in json.loads((OUT / "targets.json").read_text())}
    place = {(r["photo"], r["role"]): r for r in sc}
    photos = sorted(tg)

    def stat(p, role, kind):
        r = place.get((p, role))
        v = sorted((f["megaloc"] for f in r["files"] if f["megaloc"] is not None), reverse=True) if r else []
        if not v:
            return np.nan
        return (v[0] - r["null_mu"]) / max(r["null_sd"], 1e-3) if kind == "z" else v[0]

    head_right = np.asarray([tg[p]["head_right"] for p in photos])
    lines = [f"Mapillary verifier on dev3k: {len(photos)} photos whose head top two areas hold exactly one right area (< 25 km); <= frames "
             f"within {KM:g} km of each area's point, no panoramas, <= 2 per sequence; MegaLoc. Head picks the right one on {head_right.mean():.1%}."]
    for role in ROLES:
        cnt = np.asarray([len(place[(p, role)]["files"]) if (p, role) in place else 0 for p in photos])
        best = np.asarray([stat(p, role, "best") for p in photos])
        z = np.asarray([stat(p, role, "z") for p in photos])
        lines.append(f"  {role:6s}: any frame {np.mean(cnt > 0):.0%}, median {np.median(cnt):.0f}; best median {np.nanmedian(best):.3f}, >= 0.3 / 0.4 on "
                     f"{np.mean(best >= 0.3):.1%} / {np.mean(best >= 0.4):.1%}; z >= 3 on {np.mean(z >= 3):.1%}")
    for kind in ("best", "z"):
        val = {(p, role): stat(p, role, kind) for p in photos for role in ("top", "second", "truth")}
        rgt = np.asarray([val[(p, "top")] if tg[p]["head_right"] else val[(p, "second")] for p in photos])
        wrg = np.asarray([val[(p, "second")] if tg[p]["head_right"] else val[(p, "top")] for p in photos])
        tru = np.asarray([val[(p, "truth")] for p in photos])
        both = np.isfinite(rgt) & np.isfinite(wrg)
        lines.append(f"  {kind:4s}: both covered {both.mean():.0%}; right area above the wrong one {np.mean(rgt[both] > wrg[both]):.1%} "
                     f"(AUC {_auc(np.r_[rgt[both], wrg[both]], np.r_[np.ones(both.sum(), bool), np.zeros(both.sum(), bool)]):.3f}); "
                     f"truth above the wrong one {np.mean(tru[np.isfinite(tru) & np.isfinite(wrg)] > wrg[np.isfinite(tru) & np.isfinite(wrg)]):.1%}")
        # switch: take the head's second when its verifier score beats the top's by > margin (both covered); fixes / breaks on the pairs
        top_v = np.asarray([val[(p, "top")] for p in photos])
        sec_v = np.asarray([val[(p, "second")] for p in photos])
        for m in ((0.0, 0.1, 0.2) if kind == "best" else (0.0, 2.0, 4.0)):
            sw = both & (sec_v > top_v + m)
            fx, br = int((sw & ~head_right).sum()), int((sw & head_right).sum())
            lines.append(f"      switch to the second when it leads by > {m:g}: {sw.sum()} switches, {fx} fixes / {br} breaks "
                         f"({100 * (fx - br) / 3000:+.2f} pts of dev3k)")
    text = "\n".join(lines)
    print(text, flush=True)
    (OUT / "report.txt").write_text(text + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("targets", "fetch", "score", "report"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    {"targets": targets, "fetch": lambda: fetch(a.n, a.workers), "score": score, "report": report}[a.cmd]()


if __name__ == "__main__":
    main()
