# Does the VLM know where a photo is when it is not shown the candidates? Candidate-free guesses as an independent vote (threshold_headroom).
# Usage: .venv/bin/python -m geo_search_env.experiment.free_guess --name qwen3.5-4b --tag dev    (vLLM on :8765; scripts/free_guess.sh runs dev + val)

"""Prompt: the photo only, no candidates. One greedy answer and SAMPLES answers at temperature 0.7 per photo, saved to <tag>/free_<name>.json as
{"greedy": [lat, lon] or null, "samples": [[lat, lon] or null, ...]}. threshold_headroom adds them as vote features when the file exists."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from typing import Any, Sequence

from .pivot_diagnostics import parse_coordinates
from .query_evidence import ROOT, MP16Images, _load, _parallel, _photo

SAMPLES = 8
PROMPT = """Where was this photo taken? Use readable text, landmarks, architecture, vegetation, terrain and any other cues.
Think briefly (under 120 words): name the country and region, then end with exactly one JSON block: ```json
{"lat": <latitude>, "lon": <longitude>}
```"""


def _complete(server: str, content: list[dict[str, Any]], max_tokens: int, n: int, temperature: float) -> list[str]:
    body = {"model": "vlm", "temperature": temperature, "seed": 0, "n": n, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": content}]}
    request = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return [c["message"]["content"] or "" for c in json.loads(response.read())["choices"]]
        except Exception:
            time.sleep(5 * 2**attempt)
    return [""] * n


def run(name: str, server: str, tag: str, max_tokens: int) -> None:
    photos = _load(tag, "dev.json")
    images = MP16Images()

    def one(e: dict[str, Any]) -> dict[str, Any]:
        content = [_photo(images, e), {"type": "text", "text": PROMPT}]
        greedy = _complete(server, content, max_tokens, 1, 0.0)[0]
        samples = _complete(server, content, max_tokens, SAMPLES, 0.7)
        return {"greedy": parse_coordinates(greedy), "samples": [parse_coordinates(s) for s in samples], "raw": greedy}

    out = dict(zip((e["image_id"] for e in photos), _parallel(one, photos, workers=32)))
    (ROOT / tag / f"free_{name}.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"{name} {tag}: {len(out)} photos, greedy unparsed {sum(v['greedy'] is None for v in out.values())}, "
          f"samples unparsed {sum(s is None for v in out.values() for s in v['samples'])} of {SAMPLES * len(out)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="qwen3.5-4b")
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--tag", default="dev")
    parser.add_argument("--max-tokens", type=int, default=400)
    args = parser.parse_args(argv)
    run(args.name, args.server, args.tag, args.max_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
