"""Text visible in the index photos as area evidence (idea 1; user, 2026-10-21). Gazetteers hold official place names; the index's own photos
show the strings people actually see at a place (shop fronts, posters, menus, brand signage). For every dev3k photo with read text, the 4B
reads all visible text in each candidate area's 12 index photos (hypotheses rows); the query's text (the 4B's verbatim text + place names)
is fuzzy-matched against each area's strings.

read:   OCR the area photos with the 4B served at :8765 (scripts/gallery_text.sh) -> artifacts/value/gallery_text_{tag}.jsonl (resumable),
        then gallery_text_{tag}.json {row: [strings]}
report: per area the best full-string rapidfuzz ratio and the number of query strings matched >= 90 (strings of >= 5 letters, not digits);
        AUC right vs wrong areas; on the pipeline's misses with a right area, how often the right area beats the pipeline's area; an in-sample
        switch sweep (upper bound; pre-declared bar >= +1 pt of dev3k); luck = the query's text against another photo's area photos.

  PYTHONPATH=src .venv/bin/python scripts/gallery_text_check.py {read,report} [--tag dev3k_setft_own]
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from geo_search_env.experiment.text_search import haversine_km, norm

OUT = Path("artifacts/value")
KM = {"dev3k_setft_own": "dev3k|RL policy, lambda 0"}
OBS = {"dev3k_setft_own": "observe_dev3k_setft_own_4b.json"}
PROMPT = ('Read all text visible in this photo (signs, shop names, plates, posters, menus, labels). Answer with JSON only: {"text": ["verbatim '
          'text", ...]}, at most 15 items, no repeats; {"text": []} if there is none.')


def query_strings(parsed: dict) -> list[str]:
    raw = [s for s in (parsed.get("text") or []) + (parsed.get("place_names") or []) if isinstance(s, str)]
    return sorted({norm(s) for s in raw if len(norm(s).replace(" ", "")) >= 5 and not norm(s).replace(" ", "").isdigit()})


def setup(tag: str):
    photos = json.loads(Path(f"artifacts/photo_sets/{tag}/photos.json").read_text(encoding="utf-8"))
    obs = json.loads(Path("artifacts/agent", OBS[tag]).read_text(encoding="utf-8"))
    h = np.load(f"artifacts/agent/hypotheses_{tag}.npz")
    qs = {i: query_strings((obs.get(p["image_id"]) or {}).get("parsed") or {}) for i, p in enumerate(photos)}
    with_text = [i for i, s in qs.items() if s]
    return photos, h, qs, with_text


def read(tag: str, server: str = "http://127.0.0.1:8765", workers: int = 32) -> None:
    from geo_search_env.experiment.sft_data import MP16Images
    from geo_search_env.experiment.strategy_search import MP16_CSV, MP16_EMBED, _memmap_gallery
    from geo_search_env.models.pinpoint_reranker import _discover_index

    photos, h, qs, with_text = setup(tag)
    keep = np.isin(h["photo"], with_text)
    rows = h["rows"][keep]
    need = np.unique(rows[rows >= 0])
    index_dir = _discover_index()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    cache = np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
    row_index = _memmap_gallery(MP16_EMBED)["row_index"]
    with MP16_CSV.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        csv_ids = [r[0] for r in reader]
    ids = [csv_ids[row_index[cache[r]]] for r in need]
    print(f"{tag}: {len(with_text)} photos with text, {keep.sum()} areas, {len(need)} area photos to read", flush=True)
    images = MP16Images()

    def run(k: int):
        data = images.read(ids[k])
        if not data:
            return None
        body = {"model": "vlm", "temperature": 0.0, "max_tokens": 250, "repetition_penalty": 1.05, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode()}},
                                                          {"type": "text", "text": PROMPT}]}]}
        req = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                text = json.loads(r.read())["choices"][0]["message"]["content"]
            a, b = text.find("{"), text.rfind("}")
            return [s for s in json.loads(text[a: b + 1]).get("text", []) if isinstance(s, str)]
        except Exception:
            return None

    OUT.mkdir(parents=True, exist_ok=True)
    part = OUT / f"gallery_text_{tag}.jsonl"  # appended as answers arrive, so a killed run resumes (the first run died at 35k / 50k)
    done = {json.loads(line)["row"] for line in part.open(encoding="utf-8")} if part.exists() else set()
    todo = [k for k in range(len(need)) if int(need[k]) not in done]
    print(f"  {len(done):,} already read, {len(todo):,} to go", flush=True)
    with ThreadPoolExecutor(workers) as pool, part.open("a", encoding="utf-8") as f:
        for n, (k, res) in enumerate(zip(todo, pool.map(run, todo))):
            if res is not None:
                f.write(json.dumps({"row": int(need[k]), "text": res}, ensure_ascii=False) + "\n")
            if n % 2000 == 0:
                f.flush()
                print(f"  {n:,} / {len(todo):,}", flush=True)
    out = {}
    for line in part.open(encoding="utf-8"):
        d = json.loads(line)
        out[d["row"]] = d["text"]
    (OUT / f"gallery_text_{tag}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"done: {len(out):,} of {len(need):,} read, {sum(bool(v) for v in out.values()):,} with text", flush=True)


def report(tag: str) -> None:
    from rapidfuzz import fuzz, process

    photos, h, qs, with_text = setup(tag)
    text = {int(k): v or [] for k, v in json.loads((OUT / f"gallery_text_{tag}.json").read_text(encoding="utf-8")).items()}
    z = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz")
    km, chosen = z[KM[tag]], z[KM[tag] + "|center"]
    truths = [tuple(json.loads(p["truth"]) if isinstance(p["truth"], str) else p["truth"]) for p in photos]
    by_photo: dict[int, list[int]] = {}
    for j, p in enumerate(h["photo"]):
        by_photo.setdefault(int(p), []).append(j)

    def area_strings(j: int) -> list[str]:
        return sorted({norm(s) for r in h["rows"][j] if r >= 0 for s in text.get(int(r), []) if len(norm(s).replace(" ", "")) >= 5})

    def score(q: list[str], vocab: list[str]) -> tuple[float, int]:
        if not q or not vocab:
            return 0.0, 0
        m = process.cdist(q, vocab, scorer=fuzz.ratio, dtype=np.uint8).max(1)
        return float(m.max()), int((m >= 90).sum())

    rng = random.Random(0)
    rows = []
    for i in with_text:
        areas = by_photo.get(i, [])
        if not areas:
            continue
        t = truths[i]
        dist = lambda c: float(haversine_km(t[0], t[1], np.asarray([c[0]]), np.asarray([c[1]]))[0])
        cd = [float(haversine_km(chosen[i][0], chosen[i][1], np.asarray([h["center"][j][0]]), np.asarray([h["center"][j][1]]))[0]) for j in areas]
        ch = areas[int(np.argmin(cd))] if min(cd) < 10 else None  # the pipeline's area among the candidate areas (None: a clue / zoomed point elsewhere)
        o = rng.choice([k for k in with_text if k != i and by_photo.get(k)])
        rows.append({"i": i, "km": float(km[i]),
                     "areas": [{"j": j, "km": dist(h["center"][j]), "s": score(qs[i], area_strings(j))} for j in areas],
                     "chosen": ch, "luck": score(qs[i], area_strings(by_photo[o][0]))})
    n_set = len(photos)
    print(f"{tag}: {len(rows)} photos with text and areas; area photos with any text "
          f"{np.mean([bool(v) for v in text.values()]):.0%}; pipeline's area found among the areas for {np.mean([r['chosen'] is not None for r in rows]):.0%}")
    pos = [a["s"][0] for r in rows for a in r["areas"] if a["km"] < 25 and any(b["km"] >= 25 for b in r["areas"])]
    neg = [a["s"][0] for r in rows for a in r["areas"] if a["km"] >= 25 and any(b["km"] < 25 for b in r["areas"])]
    pa, na = np.asarray(pos), np.asarray(neg)
    auc = ((pa[:, None] > na[None, :]).mean() + 0.5 * (pa[:, None] == na[None, :]).mean()) if len(pa) and len(na) else float("nan")
    print(f"  AUC right vs wrong areas (best ratio): {auc:.3f} ({len(pa)} / {len(na)}); match >= 90: right areas {np.mean(pa >= 90):.1%}, "
          f"wrong {np.mean(na >= 90):.1%}, another photo's area (luck) {np.mean([r['luck'][0] >= 90 for r in rows]):.1%}")
    fix = [r for r in rows if r["km"] >= 25 and any(a["km"] < 25 for a in r["areas"]) and r["chosen"] is not None]
    sc = lambda r, j: next(a["s"] for a in r["areas"] if a["j"] == j)
    better = [max(a["s"] for a in r["areas"] if a["km"] < 25) > sc(r, r["chosen"]) for r in fix]
    tied = [max(a["s"] for a in r["areas"] if a["km"] < 25) == sc(r, r["chosen"]) for r in fix]
    print(f"  pipeline wrong, a right area offered: {len(fix)} photos ({100 * len(fix) / n_set:.1f} pts); right area above the pipeline's area "
          f"{np.mean(better):.0%}, tied {np.mean(tied):.0%}")
    best = None
    for thr in (85, 90, 95, 100):
        for need in (1, 2):
            for margin in (5, 10, 20):
                fixes = breaks = 0
                for r in rows:
                    if r["chosen"] is None:
                        continue
                    top = max(r["areas"], key=lambda a: (a["s"][1], a["s"][0]))
                    c = sc(r, r["chosen"])
                    if top["j"] != r["chosen"] and top["s"][0] >= thr and top["s"][1] >= need and top["s"][0] - c[0] >= margin:
                        fixes += r["km"] >= 25 and top["km"] < 25
                        breaks += r["km"] < 25 and top["km"] >= 25
                if best is None or fixes - breaks > best[0]:
                    best = (fixes - breaks, thr, need, margin, fixes, breaks)
    print(f"  best in-sample switch (ratio >= {best[1]}, >= {best[2]} strings matched, beats the pipeline's area by >= {best[3]}): {best[4]} fixes / "
          f"{best[5]} breaks = {100 * best[0] / n_set:+.2f} pts of dev3k (upper bound; bar +1)")
    (OUT / f"gallery_text_report_{tag}.json").write_text(json.dumps(rows), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("node", choices=("read", "report"))
    ap.add_argument("--tag", default="dev3k_setft_own")
    args = ap.parse_args()
    (read if args.node == "read" else report)(args.tag)


if __name__ == "__main__":
    main()
