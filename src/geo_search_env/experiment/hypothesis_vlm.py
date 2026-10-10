# VLM plausibility of each hypothesis (HYPOTHESIS_PLAN.md, step 1): the base 4B sees the photo and a hypothesis's place name (nearest town,
# region, country of its centre) and answers whether the photo could have been taken there; P(Yes) becomes a comparator feature. It adds the
# model's knowledge (architecture, vegetation, signs) as evidence next to retrieval, judged per hypothesis. dev sets only.
# Usage: RUN="hypothesis_vlm judge" NAME=4b TAG=dev3k bash scripts/clue_judge.sh      (serves the 4B; all hypothesis files of the set)

"""judge: for every hypotheses_{tag}{,_clues,_cluehyps,_more10,_vlmhyps}.npz, P(Yes) per row -> artifacts/agent/hypvlm_{tag}_{file}.npy
(aligned with that file's rows). Place names: nearest GeoNames town (>= 1,000 people) + the Overture region / country of the centre."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
PHOTO_SETS = Path("artifacts/photo_sets")
FILES = ("", "_clues", "_cluehyps", "_more10", "_vlmhyps")
PROMPT = "Could this photo have been taken in or near {place}? Answer Yes or No."


def judge(tag: str, name: str, server: str = "http://127.0.0.1:8765", workers: int = 48) -> None:
    from .name_score import _post
    from .region_evidence import _Labeller
    from .sft_data import MP16Images
    from .text_search import TextSearch

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    images = MP16Images()
    towns = TextSearch(yfcc=False)
    regions = _Labeller()
    import collections
    import threading

    cache: collections.OrderedDict[int, str] = collections.OrderedDict()  # bounded: jobs run photo-major
    lock = threading.Lock()

    def place(lat: float, lon: float) -> str:
        town = towns.label(lat, lon)
        country, region = regions(lat, lon)
        name_ = town.rsplit(",", 1)[0] if town != "?" else ""
        return ", ".join(x for x in (name_, region, country) if x) or f"{lat:.2f}, {lon:.2f}"

    def ask(job: tuple[int, str]) -> float:
        p, where = job
        with lock:
            image = cache.get(p)
        if image is None:
            image = base64.b64encode(images.read(photos[p]["image_id"])).decode()
            with lock:
                cache[p] = image
                while len(cache) > 400:
                    cache.popitem(last=False)
        body = {"model": "vlm", "temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image}},
                                                          {"type": "text", "text": PROMPT.format(place=where)}]}]}
        try:
            top = _post(server, "/v1/chat/completions", body)["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except Exception:  # noqa: BLE001
            return float("nan")
        pr = {"yes": 0.0, "no": 0.0}
        for t in top:
            k = t["token"].strip().lower()
            if k in pr:
                pr[k] += math.exp(t["logprob"])
        return pr["yes"] / (pr["yes"] + pr["no"]) if pr["yes"] + pr["no"] > 0 else float("nan")

    for suffix in FILES:
        path = OUT / f"hypotheses_{tag}{suffix}.npz"
        dst = OUT / f"hypvlm_{tag}{suffix or '_base'}.npy"
        if not path.exists() or dst.exists():
            continue
        z = np.load(path)
        jobs = [(int(p), place(*c)) for p, c in zip(z["photo"], z["center"])]
        order = np.argsort([j[0] for j in jobs], kind="stable")  # photo-major, so a photo's image stays cached
        with ThreadPoolExecutor(workers) as pool:
            got = list(pool.map(ask, [jobs[i] for i in order]))
        out = np.empty(len(jobs))
        out[order] = got
        np.save(dst, out)
        print(f"{tag}{suffix}: {len(jobs)} hypotheses judged, {int(np.isnan(out).sum())} failed -> {dst}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("judge",))
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--name", default="4b")
    args = parser.parse_args(argv)
    judge(args.tag, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
