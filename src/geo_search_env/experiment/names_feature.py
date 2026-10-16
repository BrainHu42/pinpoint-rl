# Named-area choice by a local VLM as answer-head evidence (after lessons 104-106: Sonnet told the candidates' town names fixed head
# mistakes; local readers were right more often than chance but uncalibrated, so every switching threshold lost). Here the reader's answer is
# not a switch: the photo and all of a photo's areas (the head's rows: Pinpoint areas + clue places), each named "near <town> (lat, lon),
# country", one letter per area; P(letter) from the first-token logprobs, forward and reversed order averaged (position bias, lesson 99).
# The answer head learns how far to trust it from the 35k train photos (extra column group "names").
# Usage: MODEL=data/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a NAME=27b TAG=dev3k_setft_own \
#          RUN="names_feature ask" bash scripts/clue_judge.sh         (then TAG=train_setft_own; resumable)
#        .venv/bin/python -m geo_search_env.experiment.names_feature columns --tag dev3k_setft_own --name 27b
#        action_agent --extra names ... (hypothesis_agent.load reads artifacts/agent/names_{tag}.npz)

"""ask: answers appended to artifacts/agent/names_{tag}_{name}.jsonl (one line per photo and order; reruns skip done photos).
columns: names_{tag}.npz aligned with the zoom file's rows (photo, source; x = P, log P, P - best other P; groups "names")."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
LETTERS = "ABCDEFGHIJ"
PROMPT = """You are an expert geolocator. This photo was taken within about 25 km of at most one of these places:
{options}
Use text and language, brands, landmarks, events, architecture, landscape and terrain, vegetation, climate and what you know about each
place. Answer with the single letter of the most likely place."""


def _names(centres: np.ndarray) -> list[str]:
    """ "near <town> (lat, lon), <country>": the nearest GeoNames town of >= 5,000 people (artifacts/agent/cities5k.tsv)."""

    names, lat, lon, cc = [], [], [], []
    for line in (OUT / "cities5k.tsv").read_text(encoding="utf-8").splitlines():
        n, a, b, c = line.split("\t")
        names.append(n)
        lat.append(float(a))
        lon.append(float(b))
        cc.append(c)
    country = {}
    for line in Path("data/pinpoint/geonames/countryInfo.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            f = line.split("\t")
            country[f[0]] = f[4]
    from scipy.spatial import cKDTree

    def xyz(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)

    _, idx = cKDTree(xyz(np.asarray(lat), np.asarray(lon))).query(xyz(centres[:, 0], centres[:, 1]))
    return [f"near {names[i]} ({a:.2f}, {b:.2f}), {country.get(cc[i], cc[i])}" for i, (a, b) in zip(idx, centres)]


def _zoom(tag: str):
    return np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}.npz")


def ask(tag: str, name: str, server: str = "http://127.0.0.1:8765", workers: int = 24, limit: int | None = None) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    from .name_score import _post, _read
    from .sft_data import MP16Images

    photos = json.loads((Path("artifacts/photo_sets") / tag / "photos.json").read_text(encoding="utf-8"))
    zm = _zoom(tag)
    names = _names(zm["center"].astype(np.float64))
    by_photo: dict[int, list[int]] = {}
    for j, p in enumerate(zm["photo"]):
        by_photo.setdefault(int(p), []).append(j)
    path = OUT / f"names_{tag}_{name}.jsonl"
    done = set()
    if path.exists():
        done = {(r["photo"], r["order"]) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if "p" in r}
    todo = [p for p in sorted(by_photo) if len(by_photo[p]) > 1 and not all((p, o) in done for o in (0, 1))][:limit]
    print(f"{tag}: {len(todo)} photos to ask ({len(done) // 2} done)", flush=True)
    images = MP16Images()

    def image(p):
        im = Image.open(io.BytesIO(_read(photos[p], images))).convert("RGB")
        im.thumbnail((768, 768))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def run(p):
        rows = by_photo[p][: len(LETTERS)]
        try:
            url = image(p)
        except Exception as e:  # noqa: BLE001
            return [{"photo": p, "order": o, "error": str(e)[:200]} for o in (0, 1)]
        out = []
        for o in (0, 1):
            if (p, o) in done:
                continue
            order = rows if o == 0 else rows[::-1]
            options = "\n".join(f"{LETTERS[k]} = {names[j]}" for k, j in enumerate(order))
            body = {"model": "vlm", "temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                              {"type": "text", "text": PROMPT.format(options=options)}]}]}
            try:
                top = _post(server, "/v1/chat/completions", body)["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            except Exception as e:  # noqa: BLE001
                out.append({"photo": p, "order": o, "error": str(e)[:200]})
                continue
            q = np.zeros(len(order))
            for t in top:
                s = t["token"].strip().upper()
                if len(s) == 1 and s in LETTERS[: len(order)]:
                    q[LETTERS.index(s)] += math.exp(t["logprob"])
            q = q / q.sum() if q.sum() > 0 else np.full(len(order), 1 / len(order))
            out.append({"photo": p, "order": o, "rows": [int(j) for j in order], "p": [round(float(x), 5) for x in q]})
        return out

    n = 0
    with path.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(workers) as pool:
        for res in pool.map(run, todo):
            for r in res:
                stream.write(json.dumps(r) + "\n")
            n += 1
            if n % 500 == 0:
                stream.flush()
                print(f"  {n} / {len(todo)}", flush=True)
    print(f"-> {path}", flush=True)


def columns(tag: str, name: str) -> None:
    zm = _zoom(tag)
    photo = zm["photo"]
    acc = np.zeros(len(photo))
    cnt = np.zeros(len(photo))
    for r in map(json.loads, (OUT / f"names_{tag}_{name}.jsonl").read_text(encoding="utf-8").splitlines()):
        if "p" in r:
            acc[r["rows"]] += r["p"]
            cnt[r["rows"]] += 1
    n_rows = np.bincount(photo, minlength=photo.max() + 1)[photo]
    p = np.where(cnt > 0, acc / np.maximum(cnt, 1), 1.0 / n_rows)  # photos with one row (or unanswered): uniform
    rel = np.zeros_like(p)
    for q in np.unique(photo):
        J = np.flatnonzero(photo == q)
        for k, j in enumerate(J):
            o = np.delete(p[J], k)
            rel[j] = p[j] - (o.max() if len(o) else p[j])
    x = np.stack([p, np.log(np.maximum(p, 1e-4)), rel], 1).astype(np.float32)
    dst = OUT / f"names_{tag}.npz"
    np.savez(dst, photo=photo, source=zm["source"], x=x, groups=np.asarray(["names"] * x.shape[1]))
    print(f"-> {dst}: {int((cnt > 0).sum())} of {len(photo)} rows answered")


def report(tag: str) -> None:
    """How often the reader's top area is right (< 25 km restricted top-1), vs the first (Pinpoint-ranked) area, on photos with > 1 area."""

    from .refine_headroom import _hav

    zm = _zoom(tag)
    x = np.load(OUT / f"names_{tag}.npz")["x"]
    photos = json.loads((Path("artifacts/photo_sets") / tag / "photos.json").read_text(encoding="utf-8"))
    truth = np.asarray([e["truth"] for e in photos], np.float64)[zm["photo"]]
    ok = _hav(truth[:, 0], truth[:, 1], zm["center"][:, 0], zm["center"][:, 1]) < 25
    reader, first, oracle = [], [], []
    for q in np.unique(zm["photo"]):
        J = np.flatnonzero(zm["photo"] == q)
        if len(J) > 1:
            reader.append(ok[J[np.argmax(x[J, 0])]])
            first.append(ok[J[0]])
            oracle.append(ok[J].any())
    print(f"{tag}: {len(reader)} photos; area centre < 25 km: reader's pick {np.mean(reader):.1%}, first area {np.mean(first):.1%}, any {np.mean(oracle):.1%}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("ask", "columns", "report"))
    parser.add_argument("--tag", default="dev3k_setft_own")
    parser.add_argument("--name", default="27b")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.cmd == "ask":
        ask(args.tag, args.name, limit=args.limit)
    elif args.cmd == "columns":
        columns(args.tag, args.name)
    else:
        report(args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
