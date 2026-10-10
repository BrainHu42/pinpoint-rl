# VLM plausibility of each hypothesis (HYPOTHESIS_PLAN.md, step 1): the base 4B sees the photo and a hypothesis's place name (nearest town,
# region, country of its centre) and answers whether the photo could have been taken there; P(Yes) becomes a comparator feature. It adds the
# model's knowledge (architecture, vegetation, signs) as evidence next to retrieval, judged per hypothesis. dev sets only.
# Usage: RUN="hypothesis_vlm judge" NAME=4b TAG=dev3k bash scripts/clue_judge.sh      (serves the 4B; all hypothesis files of the set)
#        .venv/bin/python -m geo_search_env.experiment.hypothesis_vlm sft-data --tag train_extra   (fine-tuning rows; then sft_train, and
#        MODEL=data/pinpoint/sft/<run>/merged NAME=<run> RUN="hypothesis_vlm judge" ... to judge with the fine-tuned model)
#        .venv/bin/python -m geo_search_env.experiment.hypothesis_vlm sft-data --tag train_extra   (fine-tuning rows; then sft_train, and
#        MODEL=data/pinpoint/sft/<run>/merged NAME=<run> RUN="hypothesis_vlm judge" ... to judge with the fine-tuned model)

"""judge: for every hypotheses_{tag}{,_clues,_cluehyps,_more10,_vlmhyps}.npz, P(Yes) per row -> artifacts/agent/hypvlm_{tag}_{file}.npy
(aligned with that file's rows; a --name other than 4b adds _{name}). Place names: nearest GeoNames town (>= 1,000 people) + the Overture
region / country of the centre.
sft-data: the same prompt on the E4 hypotheses of a photo set (photos never evaluated: train_extra), target Yes when the hypothesis's restricted
top-1 is < 25 km from the truth (GPS only), uniformly subsampled so P(Yes) stays calibrated -> artifacts/agent/hypvlm_sft.jsonl."""

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


def _placer():
    from .region_evidence import _Labeller
    from .text_search import TextSearch

    towns = TextSearch(yfcc=False)
    regions = _Labeller()

    def place(lat: float, lon: float) -> str:
        town = towns.label(lat, lon)
        country, region = regions(lat, lon)
        name_ = town.rsplit(",", 1)[0] if town != "?" else ""
        return ", ".join(x for x in (name_, region, country) if x) or f"{lat:.2f}, {lon:.2f}"

    return place


def sft_data(tag: str, out: Path, max_examples: int, val_share: float = 0.02) -> None:
    from .hypothesis_compare import _merge_clue_hyps

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    z = _merge_clue_hyps(tag)
    km = np.load(OUT / f"hypothesis_features_{tag}_e4_vlmhyps.npz")["km"]
    assert len(km) == len(z["photo"]), (len(km), len(z["photo"]))
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(km), min(max_examples, len(km)), replace=False))
    val = set(rng.choice(len(photos), int(len(photos) * val_share), replace=False).tolist())
    place = _placer()
    rows = [{"image_id": photos[int(z["photo"][i])]["image_id"], "split": "val" if int(z["photo"][i]) in val else "train",
             "prompt": PROMPT.format(place=place(*z["center"][i])), "target": "Yes" if km[i] < 25 else "No"} for i in pick]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"{tag}: {len(km)} hypotheses; wrote {len(rows)} rows ({sum(r['target'] == 'Yes' for r in rows)} Yes, "
          f"{sum(r['split'] == 'val' for r in rows)} val) -> {out}")


def judge(tag: str, name: str, server: str = "http://127.0.0.1:8765", workers: int = 48) -> None:
    from .name_score import _post
    from .sft_data import MP16Images

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    images = MP16Images()
    place = _placer()
    import collections
    import threading

    cache: collections.OrderedDict[int, str] = collections.OrderedDict()  # bounded: jobs run photo-major
    lock = threading.Lock()

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
        dst = OUT / f"hypvlm_{tag}{suffix or '_base'}{'' if name == '4b' else '_' + name}.npy"
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
    parser.add_argument("cmd", choices=("judge", "sft-data"))
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--name", default="4b")
    parser.add_argument("--out", type=Path, default=OUT / "hypvlm_sft.jsonl")
    parser.add_argument("--max-examples", type=int, default=48000)
    args = parser.parse_args(argv)
    judge(args.tag, args.name) if args.cmd == "judge" else sft_data(args.tag, args.out, args.max_examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
