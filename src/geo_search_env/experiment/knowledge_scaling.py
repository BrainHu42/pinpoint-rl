# How much does model size help as a chooser? Zero-shot Qwen models answer with the photo and the reranker's top-10 candidates as text.
# Usage: .venv/bin/python -m geo_search_env.experiment.knowledge_scaling run --name qwen3.5-9b --server http://127.0.0.1:8765 --tag dev
#        .venv/bin/python -m geo_search_env.experiment.knowledge_scaling report --names qwen3.5-4b qwen3.5-9b qwen3.6-27b

"""The `default` prompt of pivot_diagnostics (candidates best first, think briefly, JSON coordinates), temperature 0, on the dev and val photos
(val without the placeholders). The 27B is served by llama.cpp on :8766 (`--max-tokens` is raised there, since it writes longer analyses).

report: top-1 within 1 / 25 / 200 km of each model's answer vs the reranker's top-1 and the top-10 candidate oracle, how often the answer
lies within 25 km of a shown candidate, and paired intervals against the reranker.
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

import numpy as np

from .pivot_diagnostics import PROMPT, parse_coordinates
from .query_evidence import ROOT, MP16Images, _chat, _load, _options, _parallel, _photo
from .stage1_eval import _bootstrap
from .wiki_backend import _km


def run(name: str, server: str, tag: str, max_tokens: int) -> None:
    photos = _load(tag, "dev.json")
    images = MP16Images()
    raws = _parallel(lambda e: _chat(server, [_photo(images, e), {"type": "text", "text": PROMPT.format(options=_options(e))}], max_tokens), photos, workers=16 if "27b" in name else 64)
    out = {e["image_id"]: {"raw": raw, "answer": parse_coordinates(raw)} for e, raw in zip(photos, raws)}
    (ROOT / tag / f"scaling_{name}.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"{name} {tag}: {len(out)} photos, unparsed {sum(v['answer'] is None for v in out.values())}")


def report(names: Sequence[str]) -> None:
    excluded = set(json.loads((ROOT / "val" / "exclude.json").read_text(encoding="utf-8")))
    for tag in ("dev", "val"):
        photos = [e for e in _load(tag, "dev.json") if e["image_id"] not in excluded]
        top1 = np.asarray([_km(np.asarray(e["candidates"][:1])[:, 1:], *e["truth"])[0] for e in photos])
        oracle = np.asarray([_km(np.asarray(e["candidates"])[:, 1:], *e["truth"]).min() for e in photos])
        print(f"\n{tag} (n={len(photos)}):                      <1 km  <25 km <200 km   <25 km vs reranker [95% CI]   unparsed   answer within 25 km of a shown candidate")
        print(f"  {'reranker top-1':26s} " + " ".join(f"{(top1 < t).mean():6.1%}" for t in (1, 25, 200)))
        print(f"  {'top-10 candidate oracle':26s} " + " ".join(f"{(oracle < t).mean():6.1%}" for t in (1, 25, 200)))
        for name in names:
            answers = json.loads((ROOT / tag / f"scaling_{name}.json").read_text(encoding="utf-8"))
            dist, near, bad = [], [], 0
            for e, k in zip(photos, top1):
                a = answers[e["image_id"]]["answer"]
                if a is None:
                    bad += 1
                    dist.append(k)  # unparsed: the reranker's top-1
                    near.append(True)
                    continue
                dist.append(_km(np.asarray([a]), *e["truth"])[0])
                near.append(bool((_km(np.asarray(e["candidates"])[:, 1:], *a) < 25).any()))
            dist = np.asarray(dist)
            ci = _bootstrap((dist < 25).astype(float) - (top1 < 25))
            print(f"  {name:26s} " + " ".join(f"{(dist < t).mean():6.1%}" for t in (1, 25, 200)) + f"   {ci[0]:+6.1%} [{ci[1]:+.1%}, {ci[2]:+.1%}]   {bad / len(photos):7.1%}   {np.mean(near):.0%}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("run", "report"))
    parser.add_argument("--name")
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--tag", default="dev")
    parser.add_argument("--max-tokens", type=int, default=600)
    args = parser.parse_args(argv)
    run(args.name, args.server, args.tag, args.max_tokens) if args.node == "run" else report(args.names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
