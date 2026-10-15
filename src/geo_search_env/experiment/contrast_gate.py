# EXPERT_PLAN.md, first experiment: contrast the answer head's top two areas the way an expert does. Per photo the reader sees the query
# and reference photos of each area, names the feature that distinguishes them and picks one. References: YFCC100M photos fetched live from
# Flickr near each area's top restricted-search candidates (photo, photographer and near-duplicates excluded), the 4 most MegaLoc-similar
# to the query shown. Same-recipe control: the 4 most MegaLoc-similar of the area's existing 12 restricted-search (Pinpoint index) photos.
# Measured on 500 dev3k photos against the answer head (lesson 94's net, all evidence, all areas zoomed): pair accuracy where exactly one
# area is right, net fixes / breaks of following the reader, at 25 and 1 km.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.contrast_gate pairs
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.contrast_gate fetch
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.contrast_gate embed
#        LIMIT_MM=9 RUN="contrast_gate ask" bash scripts/clue_judge.sh      (4B served at :8765)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.contrast_gate report

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .refine_headroom import _hav

OUT = Path("artifacts/agent")
REFS = Path("data/yfcc_refs")  # fetched reference photos (EBS, durable)
CKPT_TAG = "setft_a"
N_PHOTOS, K_SHOW, K_FETCH, RADIUS_KM = 500, 4, 24, 0.3


def _pairs_path() -> Path:
    return OUT / "contrast_pairs_dev3k.json"


def pairs(n: int = N_PHOTOS) -> None:
    """The answer head's top two areas per photo (trained on all of train; read + recognise, every area zoomed), with their zoom-file rows
    and km; n photos with at least two areas, seed 0."""

    import torch

    from .action_agent import _data, fit_answer, view
    from .hypothesis_agent import SOURCES

    tr = _data("train")
    dv = _data("dev3k", stats=tr["stats"])
    net = fit_answer(tr, np.arange(tr["n"]), seed=7)
    with torch.no_grad():
        on = torch.ones(dv["n"], dtype=torch.bool, device="cuda")
        v = view(dv, on, on)
        logits = net.answer_logits(v, v["mask"]).masked_fill(~v["mask"], float("-inf")).cpu().numpy()
    km = dv["km_zoom"].cpu().numpy()
    f = np.load(OUT / f"hypothesis_features_dev3k_{CKPT_TAG}_e4_set.npz")
    rows = np.flatnonzero(np.isin(f["source"], SOURCES))
    slot_row = np.full(logits.shape, -1)
    seen = np.zeros(dv["n"], int)
    for r in rows:
        p = f["photo"][r]
        slot_row[p, seen[p]] = r
        seen[p] += 1
    eligible = np.flatnonzero(np.isfinite(logits).sum(1) >= 2)
    pick = np.sort(np.random.default_rng(0).choice(eligible, n, replace=False))
    out = []
    for p in pick:
        o = np.argsort(-logits[p])[:2]
        out.append({"photo": int(p), "rows": slot_row[p, o].tolist(), "km": km[p, o].tolist(), "logit": logits[p, o].tolist()})
    _pairs_path().write_text(json.dumps(out), encoding="utf-8")
    head = np.asarray([o["km"][0] for o in out])
    print(f"{len(out)} photos -> {_pairs_path()}; head pick < 25 km {np.mean(head < 25):.1%}, < 1 km {np.mean(head < 1):.1%}; "
          f"exactly one of the pair right (25 km) {np.mean([(a < 25) != (b < 25) for a, b in (o['km'] for o in out)]):.1%}")


def fetch(workers: int = 32, tag: str = "dev3k", pairs_path: Path | None = None, out_name: str = "contrast_refs_dev3k.json",
          k_fetch: int = K_FETCH, size: str = "z") -> None:
    """YFCC photos within RADIUS_KM of each pair area's top 3 restricted candidates, nearest first (round robin over the points), excluding
    the query photo and its photographer; up to k_fetch downloaded per area (Flickr size suffix: z 640 px, n 320 px). Files are shared by
    photo id across runs (REFS); contrast_model fetches the training pairs with this."""

    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    from scipy.spatial import cKDTree

    from .strategy_search import load_world

    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
    pr = json.loads((pairs_path or _pairs_path()).read_text(encoding="utf-8"))
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{CKPT_TAG}.npz")
    world = load_world()
    inv = {i: a for a, i in world.vocab["author"].items()}
    g = np.load("data/yfcc100m/geo.npz")
    uid_of = {u: i for i, u in enumerate(json.loads(Path("data/yfcc100m/geo_uid_vocab.json").read_text()))}
    lat, lon, uid, pid, server, secret = g["lat"], g["lon"], g["uid"], g["photo_id"], g["server"], g["secret"]

    def xyz(a, b):
        a, b = np.deg2rad(np.asarray(a, np.float64)), np.deg2rad(np.asarray(b, np.float64))
        return np.stack([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)], -1)

    tree = cKDTree(xyz(lat, lon))
    r = 2 * np.sin(RADIUS_KM / 6371.0 / 2)
    jobs, plan = [], {}
    for o in pr:
        e = photos[o["photo"]]
        q_uid = uid_of.get(inv[int(world.mp16["author"][e["row"]])], -1)
        q_pid = int(e["image_id"].split("_")[-1].split(".")[0])
        for h in o["rows"]:
            pts = zm["latlon"][h][:3]
            pts = pts[np.isfinite(pts[:, 0])]
            lists = []
            for pt in pts:
                hits = np.asarray(tree.query_ball_point(xyz(*pt), r), int)
                hits = hits[(uid[hits] != q_uid) & (pid[hits] != q_pid)]
                d = _hav(pt[0], pt[1], lat[hits], lon[hits])
                lists.append(hits[np.argsort(d)])
            chosen, seen = [], set()
            for k in range(max((len(x) for x in lists), default=0)):
                for x in lists:
                    if k < len(x) and int(x[k]) not in seen and len(chosen) < k_fetch:
                        seen.add(int(x[k]))
                        chosen.append(int(x[k]))
            plan[str(h)] = [int(pid[i]) for i in chosen]
            jobs += [(int(pid[i]), int(server[i]), secret[i].decode()) for i in chosen]
    REFS.mkdir(parents=True, exist_ok=True)
    todo = [j for j in {j[0]: j for j in jobs}.values() if not (REFS / f"{j[0]}.jpg").exists() and not (REFS / f"{j[0]}.missing").exists()]
    print(f"{len(pr)} photos, {len(plan)} areas, {sum(map(len, plan.values()))} references ({len(todo)} to download)", flush=True)

    def get(job):
        p, srv, sec = job
        try:
            data = urllib.request.urlopen(f"https://live.staticflickr.com/{srv}/{p}_{sec}_{size}.jpg", timeout=20).read()
            if len(data) > 5000 and data[:2] == b"\xff\xd8":
                (REFS / f"{p}.jpg").write_bytes(data)
                return True
        except Exception:  # noqa: BLE001
            pass
        (REFS / f"{p}.missing").touch()
        return False

    with ThreadPoolExecutor(workers) as ex:
        got = 0
        for k, ok in enumerate(ex.map(get, todo)):
            got += ok
            if k % 2000 == 0:
                print(f"  {k} / {len(todo)} ({got} ok)", flush=True)
    plan = {h: [p for p in v if (REFS / f"{p}.jpg").exists()] for h, v in plan.items()}
    (OUT / out_name).write_text(json.dumps(plan), encoding="utf-8")
    n = np.asarray([len(v) for v in plan.values()])
    print(f"-> {out_name}: references per area median {np.median(n):.0f}, none for {np.mean(n == 0):.1%} of areas", flush=True)


def embed(batch: int = 64) -> None:
    """MegaLoc on the fetched references; per area the K_SHOW most similar to the query (cosine < 0.95: near-duplicates dropped), and the
    control: the K_SHOW most similar of the area's 12 restricted-search photos."""

    import torch
    from PIL import Image

    from .megaloc_gate import SIZE, _model
    from .megaloc_gate import _paths as gate_paths

    pr = json.loads(_pairs_path().read_text(encoding="utf-8"))
    plan = json.loads((OUT / "contrast_refs_dev3k.json").read_text(encoding="utf-8"))
    gm = np.load(gate_paths("dev3k")[0])
    ge = np.load(gate_paths("dev3k")[1], mmap_mode="r")
    q = np.asarray(ge[: int(gm["n_queries"])], np.float32)
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_dev3k_{CKPT_TAG}.npz")
    sims_zoom = np.load(OUT / f"megaloc_area_sims_dev3k_{CKPT_TAG}.npz")["zoom"]
    model = _model()
    mean, std = np.asarray([0.485, 0.456, 0.406], np.float32), np.asarray([0.229, 0.224, 0.225], np.float32)
    ids = sorted({p for v in plan.values() for p in v})
    emb = {}
    with torch.inference_mode():
        for s in range(0, len(ids), batch):
            xs = []
            for p in ids[s:s + batch]:
                im = Image.open(REFS / f"{p}.jpg").convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
                xs.append(((np.asarray(im, np.float32) / 255.0 - mean) / std).transpose(2, 0, 1))
            e = model(torch.as_tensor(np.stack(xs), device="cuda")).float().cpu().numpy()
            emb.update(zip(ids[s:s + batch], e))
    out = {}
    for o in pr:
        for h in o["rows"]:
            refs = plan.get(str(h), [])
            s = np.asarray([emb[p] @ q[o["photo"]] for p in refs]) if refs else np.zeros(0)
            keep = [(float(v), p) for v, p in sorted(zip(s, refs), reverse=True) if v < 0.95][:K_SHOW]
            sz = sims_zoom[h]
            ok = np.flatnonzero(np.isfinite(sz))
            ctrl = ok[np.argsort(-sz[ok])][:K_SHOW]
            out[str(h)] = {"fetched": [p for _, p in keep], "fetched_sim": [v for v, _ in keep],
                           "control": [int(zm["rows"][h][k]) for k in ctrl], "control_sim": [float(sz[k]) for k in ctrl]}
    (OUT / "contrast_show_dev3k.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"-> contrast_show_dev3k.json ({len(out)} areas, {len(emb)} references embedded)")


PROMPT = """You are an expert geolocator. The first image is the query photo. It was taken in one of two places, A or B.
The next images show reference photos taken in place A, then reference photos taken in place B.
First name the visible feature that best distinguishes place A from place B (architecture, landscape, vegetation, signs, road
features, specific buildings or landmarks), then decide where the query photo was taken.
Answer with JSON only: {"feature": "<one short phrase>", "answer": "A" or "B"}"""


SCHEMA = {"type": "json_schema", "json_schema": {"name": "contrast", "schema": {
    "type": "object", "properties": {"feature": {"type": "string", "maxLength": 120}, "answer": {"type": "string", "enum": ["A", "B"]}},
    "required": ["feature", "answer"], "additionalProperties": False}}}


def _jpeg(data: bytes, side: int) -> str:
    from PIL import Image

    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(tag: str = "dev3k", name: str = "4b", server: str = "http://127.0.0.1:8765", workers: int = 16) -> None:
    """The reader on each pair, both conditions, both orders (A / B swapped)."""

    from concurrent.futures import ThreadPoolExecutor

    from ..models.pinpoint_reranker import _discover_index
    from .name_score import _post, _read
    from .sft_data import MP16Images
    from .strategy_search import MP16_CSV, MP16_EMBED, _memmap_gallery

    del tag
    photos = json.loads((Path("artifacts/photo_sets") / f"dev3k_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
    pr = json.loads(_pairs_path().read_text(encoding="utf-8"))
    show = json.loads((OUT / "contrast_show_dev3k.json").read_text(encoding="utf-8"))
    index_dir = _discover_index()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    cache = np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
    row_index = _memmap_gallery(MP16_EMBED)["row_index"]
    import csv

    with MP16_CSV.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        csv_ids = [r[0] for r in reader]
    images = MP16Images()

    def ref_images(h: int, cond: str) -> list[str]:
        if cond == "fetched":
            return [_jpeg((REFS / f"{p}.jpg").read_bytes(), 384) for p in show[str(h)]["fetched"]]
        return [_jpeg(images.read(csv_ids[row_index[cache[r]]]), 384) for r in show[str(h)]["control"]]

    jobs = []
    for i, o in enumerate(pr):
        for cond in ("fetched", "control"):
            if any(not show[str(h)][cond] for h in o["rows"]):
                continue
            for order in (0, 1):
                jobs.append((i, cond, order))
    print(f"{len(jobs)} requests ({name})", flush=True)

    def run(job):
        i, cond, order = job
        o = pr[i]
        a, b = (o["rows"][0], o["rows"][1]) if order == 0 else (o["rows"][1], o["rows"][0])
        content = [{"type": "text", "text": "Query photo:"},
                   {"type": "image_url", "image_url": {"url": _jpeg(_read(photos[o["photo"]], images), 768)}},
                   {"type": "text", "text": "Place A:"}] + [{"type": "image_url", "image_url": {"url": u}} for u in ref_images(a, cond)]
        content += [{"type": "text", "text": "Place B:"}] + [{"type": "image_url", "image_url": {"url": u}} for u in ref_images(b, cond)]
        content += [{"type": "text", "text": PROMPT}]
        body = {"model": "vlm", "temperature": 0.0, "max_tokens": 120, "messages": [{"role": "user", "content": content}],
                "chat_template_kwargs": {"enable_thinking": False}, "response_format": SCHEMA}  # the 4B ignores "JSON only" without it
        try:
            text = _post(server, "/v1/chat/completions", body)["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            return {"i": i, "cond": cond, "order": order, "error": str(e)[:200]}
        ans = None
        try:
            js = json.loads(text[text.index("{"): text.rindex("}") + 1])
            ans = str(js.get("answer", "")).strip().upper()[:1]
        except Exception:  # noqa: BLE001
            js = {}
        first = None if ans not in ("A", "B") else int((ans == "A") == (order == 0))  # 1: the head's top area, 0: its second
        return {"i": i, "cond": cond, "order": order, "text": text, "feature": js.get("feature"), "picks_top": first}

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(run, jobs))
    (OUT / f"contrast_answers_dev3k_{name}.json").write_text(json.dumps(res), encoding="utf-8")
    print(f"-> contrast_answers_dev3k_{name}.json; parsed {np.mean([r.get('picks_top') is not None for r in res]):.1%}", flush=True)


def report(name: str = "4b") -> None:
    pr = json.loads(_pairs_path().read_text(encoding="utf-8"))
    show = json.loads((OUT / "contrast_show_dev3k.json").read_text(encoding="utf-8"))
    res = json.loads((OUT / f"contrast_answers_dev3k_{name}.json").read_text(encoding="utf-8"))
    km = np.asarray([o["km"] for o in pr])
    n = len(pr)
    score = {c: np.full(n, np.nan) for c in ("fetched", "control")}  # P(reader prefers the head's top area), mean of both orders
    for c in score:
        acc = {}
        for r in res:
            if r["cond"] == c and r.get("picks_top") is not None:
                acc.setdefault(r["i"], []).append(r["picks_top"])
        for i, v in acc.items():
            if len(v) == 2:
                score[c][i] = np.mean(v)
    # MegaLoc alone: the area whose best shown photo is more similar
    for c in ("fetched", "control"):
        s = np.full(n, np.nan)
        for i, o in enumerate(pr):
            a, b = (show[str(h)][f"{c}_sim"] for h in o["rows"])
            if a and b:
                s[i] = 1.0 if a[0] > b[0] else 0.0 if b[0] > a[0] else 0.5
        score[f"MegaLoc best, {c}"] = s
    both = np.isfinite(score["fetched"]) & np.isfinite(score["control"])
    print(f"{name}: {n} photos; both conditions answered for {both.sum()}; order-consistent: fetched "
          f"{np.mean(np.isin(score['fetched'][both], (0, 1))):.1%}, control {np.mean(np.isin(score['control'][both], (0, 1))):.1%}")
    for thr in (25, 1):
        right = km < thr
        decisive = both & (right[:, 0] != right[:, 1])
        print(f"\n< {thr} km: head's top area right {np.mean(right[both, 0]):.1%}; exactly one of the pair right on {decisive.sum()} photos "
              f"(head right on {np.mean(right[decisive, 0]):.1%} of them)")
        print(f"   {'reader':26s} {'pair acc.':>9s} {'switches':>8s} {'fixes':>6s} {'breaks':>6s} {'net (pts of photos)':>20s}")
        for c, s in score.items():
            sel = both & np.isfinite(s)
            acc = np.mean(np.where(s[decisive & sel] >= 0.5, right[decisive & sel, 0], right[decisive & sel, 1])) if (decisive & sel).any() else np.nan
            sw = sel & (s == 0)  # both orders prefer the second area
            fixes = int((sw & right[:, 1] & ~right[:, 0]).sum())
            breaks = int((sw & right[:, 0] & ~right[:, 1]).sum())
            print(f"   {c:26s} {acc:9.1%} {sw.sum():8d} {fixes:6d} {breaks:6d} {100 * (fixes - breaks) / sel.sum():+20.2f}")
    feats = [r.get("feature") for r in res if r["cond"] == "fetched" and r.get("feature")]
    print("\nexample distinguishing features (fetched):", "; ".join(map(str, feats[:15])))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("pairs", "fetch", "embed", "ask", "report"))
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--name", default="4b")
    args = parser.parse_args(argv)
    if args.cmd == "ask":
        ask(args.tag, args.name)
    elif args.cmd == "report":
        report(args.name)
    else:
        {"pairs": pairs, "fetch": fetch, "embed": embed}[args.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
