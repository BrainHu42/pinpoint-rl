"""SigLIP2's text side as a place-name proposer (idea 2; user, 2026-10-21). Pinpoint only compares images with images; SigLIP2's text
tower learned what named places look like from web captions. Each dev3k photo (its cached full-image SigLIP2 embedding, crops view 0) is
scored against every geotagged name: GeoNames towns with population >= 1000 ("This is a photo of {town}, {country}.") and geotagged
English Wikipedia pages ("This is a photo of {title}."). Proposals = the top names merged within 25 km, best first.

Reported on dev3k (pre-declared bar: >= 3 pts of dev3k, net of luck, of photos the pipeline gets wrong where no candidate area is right
and some top-5 proposal is < 25 km): standalone top-1 accuracy, coverage on the pipeline's misses by loss stage, and luck (the same
proposals against another photo's truth, lesson 120).

  PYTHONPATH=src .venv/bin/python scripts/siglip_names_probe.py [--tag dev3k_setft_own]
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path

import numpy as np

from geo_search_env.experiment.text_search import _countries, haversine_km

MODEL = "google/siglip2-giant-opt-patch16-384"
OUT = Path("artifacts/value")
TEXT = OUT / "siglip_names_text.f16.npy"
META = OUT / "siglip_names_meta.json"
KM_KEY = {"dev3k_setft_own": "dev3k|RL policy, lambda 0"}


def names() -> list[tuple[str, float, float, str]]:
    """(prompt, lat, lon, source) for every name."""

    cc_name = _countries()[0]
    out = []
    towns = sqlite3.connect("file:/opt/dlami/nvme/text_search/places.db?mode=ro", uri=True)
    for lat, lon, name, cc, _pop in towns.execute("SELECT lat, lon, name, cc, pop FROM towns"):
        country = cc_name.get(cc, cc)
        out.append((f"This is a photo of {name}, {country}.", lat, lon, "town"))
    wiki = sqlite3.connect("file:data/web_search/wiki_en.db?mode=ro", uri=True)
    for title, lat, lon in wiki.execute("SELECT title, lat, lon FROM pages WHERE source = 'wikipedia' AND lat IS NOT NULL"):
        out.append((f"This is a photo of {title}.", lat, lon, "wiki"))
    return out


def embed_text(prompts: list[str], batch: int = 1024) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    out = np.zeros((len(prompts), model.config.text_config.projection_size if hasattr(model.config.text_config, "projection_size")
                    else 1536), np.float16)
    with torch.no_grad():
        for i in range(0, len(prompts), batch):
            tok = proc(text=prompts[i: i + batch], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            f = model.get_text_features(input_ids=tok["input_ids"].cuda())
            f = f if torch.is_tensor(f) else f.pooler_output
            out[i: i + len(f)] = torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy().astype(np.float16)
            if i % (batch * 200) == 0:
                print(f"  text {i:,} / {len(prompts):,}", flush=True)
    return out


def proposals(img: np.ndarray, txt: np.ndarray, lat: np.ndarray, lon: np.ndarray, top: int = 200, k: int = 10) -> list[list[tuple]]:
    """Per photo: up to k (lat, lon, score, name index) merged within 25 km, best first."""

    import torch

    t = torch.as_tensor(txt, device="cuda")
    res = []
    for i in range(0, len(img), 256):
        s = torch.as_tensor(img[i: i + 256], device="cuda").half() @ t.T
        v, idx = s.topk(top, dim=1)
        for vv, ii in zip(v.float().cpu().numpy(), idx.cpu().numpy()):
            picked: list[tuple] = []
            for score, j in zip(vv, ii):
                if all(haversine_km(lat[j], lon[j], np.asarray([p[0]]), np.asarray([p[1]]))[0] >= 25 for p in picked):
                    picked.append((float(lat[j]), float(lon[j]), float(score), int(j)))
                    if len(picked) == k:
                        break
            res.append(picked)
    return res


def hit(props: list[tuple], truth, k: int) -> bool:
    return any(haversine_km(truth[0], truth[1], np.asarray([p[0]]), np.asarray([p[1]]))[0] < 25 for p in props[:k])


def run(tag: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not TEXT.exists():
        rows = names()
        print(f"names: {len(rows):,} ({sum(r[3] == 'town' for r in rows):,} towns, {sum(r[3] == 'wiki' for r in rows):,} Wikipedia)", flush=True)
        np.save(TEXT, embed_text([r[0].lower() for r in rows]))  # SigLIP2 was trained on lowercased text: cased prompts drop 24 / 40 -> 4 / 40 on a 20-city check
        META.write_text(json.dumps([[r[0], r[1], r[2], r[3]] for r in rows], ensure_ascii=False), encoding="utf-8")
    meta = json.loads(META.read_text(encoding="utf-8"))
    txt = np.load(TEXT)
    lat, lon = np.asarray([m[1] for m in meta]), np.asarray([m[2] for m in meta])
    photos = json.loads(Path(f"artifacts/photo_sets/{tag}/photos.json").read_text(encoding="utf-8"))
    img = np.asarray(np.load(f"artifacts/agent/crops_{tag}.npy", mmap_mode="r")[:, 0], np.float32)
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    props = proposals(img, txt, lat, lon)
    truths = [tuple(json.loads(p["truth"]) if isinstance(p["truth"], str) else p["truth"]) for p in photos]
    km = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz")[KM_KEY[tag]]
    h = np.load(f"artifacts/agent/hypotheses_{tag}.npz")
    area_right = np.zeros(len(photos), bool)
    for p, c in zip(h["photo"], h["center"]):
        area_right[p] |= haversine_km(truths[p][0], truths[p][1], np.asarray([c[0]]), np.asarray([c[1]]))[0] < 25
    pool_right = np.asarray([any(haversine_km(t[0], t[1], np.asarray([q[0]]), np.asarray([q[1]]))[0] < 25
                                 for q in (json.loads(p["pool"]) if isinstance(p["pool"], str) else p["pool"])) for p, t in zip(photos, truths)])
    rng = random.Random(0)
    other = []
    for t in truths:
        while True:
            o = truths[rng.randrange(len(truths))]
            if haversine_km(t[0], t[1], np.asarray([o[0]]), np.asarray([o[1]]))[0] > 100:
                other.append(o)
                break
    n = len(photos)
    wrong = km >= 25
    with open(OUT / f"siglip_names_{tag}.jsonl", "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"image_id": photos[i]["image_id"], "truth": truths[i], "km": float(km[i]), "area_right": bool(area_right[i]),
                                "pool_right": bool(pool_right[i]), "props": [[p[0], p[1], round(p[2], 4), meta[p[3]][0]] for p in props[i]]},
                               ensure_ascii=False) + "\n")
    print(f"\n{tag}: {n} photos; pipeline {100 * np.mean(~wrong):.1f}% < 25 km, area oracle {100 * np.mean(area_right | ~wrong):.1f}, "
          f"pool oracle {100 * np.mean(pool_right | ~wrong):.1f}")
    for k in (1, 5, 10):
        real = np.asarray([hit(props[i], truths[i], k) for i in range(n)])
        luck = np.asarray([hit(props[i], other[i], k) for i in range(n)])
        line = f"  top-{k:<2d} proposal < 25 km: all {100 * real.mean():5.1f} (luck {100 * luck.mean():4.1f})"
        for name, m in (("pipeline wrong", wrong), ("wrong, no area right", wrong & ~area_right), ("wrong, no pool candidate right", wrong & ~pool_right)):
            line += f" | {name} {100 * (real & m).mean():4.1f} pts (luck {100 * (luck & m).mean():3.1f})"
        print(line)
    real5 = np.asarray([hit(props[i], truths[i], 5) for i in range(n)])
    luck5 = np.asarray([hit(props[i], other[i], 5) for i in range(n)])
    m = wrong & ~area_right
    net = 100 * ((real5 & m).mean() - (luck5 & m).mean())
    print(f"  bar (wrong, no area right, top-5 right, net of luck >= 3 pts): {net:+.2f} pts -> {'GO' if net >= 3 else 'NO-GO'}")
    src = np.asarray([meta[props[i][0][3]][3] if props[i] else "" for i in range(n)])
    r1 = np.asarray([hit(props[i], truths[i], 1) for i in range(n)])
    for s in ("town", "wiki"):
        print(f"  top-1 from {s}: {np.mean(src == s):.0%} of photos, right {100 * r1[src == s].mean():.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dev3k_setft_own")
    args = ap.parse_args()
    run(args.tag)


if __name__ == "__main__":
    main()
