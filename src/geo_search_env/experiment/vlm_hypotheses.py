# Collect weighted location hypotheses from a local OpenAI-compatible VLM server for coarse-filter queries.
# Usage: PYTHONPATH=src python -m geo_search_env.experiment.vlm_hypotheses --queries artifacts/coarse_filter/v1/rows.jsonl --output artifacts/vlm_guesses/gemma4_26b_a4b.jsonl
#    or: ... --benchmark-queries artifacts/strategy_search/queries.json --per-benchmark 500 --output artifacts/strategy_search/vlm_gemma4.jsonl

"""Zero-shot VLM coarse hypotheses (lat/lon + probability) for the coarse-filter study."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import urllib.error
import urllib.request
from typing import Any, Sequence

import numpy as np

from .feasibility import OSVTestReferenceDataset


OSV_ROOT = Path("/data/hf/datasets/osv5m")
PROMPT = """You are an expert GeoGuessr player. Geolocate this {kind}.
First, briefly transcribe any readable text and list the geographic clues (language/script, driving side, road markings, poles, bollards, vegetation, terrain, architecture).
Then give 5 distinct hypotheses for where it was taken, each with a probability; probabilities must sum to 1. Spread probability across countries when uncertain.
Finish with exactly one JSON block:
```json
{"text": "...", "gps_overlay": false, "hypotheses": [{"country": "...", "region": "...", "lat": 0.0, "lon": 0.0, "p": 0.0}]}
```
Set "gps_overlay" to true only if the image itself shows printed GPS coordinates."""


def _ask(server: str, model: str, image_path: Path, prompt: str) -> dict[str, Any]:
    image = base64.b64encode(image_path.read_bytes()).decode()
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 1500,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    request = urllib.request.Request(
        f"{server}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            answer = json.loads(response.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as error:
        # llama.cpp rejects generations containing invalid UTF-8 but echoes the full text in the error body.
        message = json.loads(error.read())["error"]["message"]
        if "<channel|>" not in message:
            raise
        answer = message.split("<channel|>", 1)[1]
    return {**parse_answer(answer), "raw": answer}


def parse_answer(answer: str) -> dict[str, Any]:
    """Parse the trailing JSON block, tolerating the malformed keys small VLMs emit (e.g. `"lat: 6.5`)."""

    block = answer[answer.rfind("hypotheses"):] if "hypotheses" in answer else ""
    hypotheses = []
    for item in re.findall(r"\{[^{}]*\}", block):
        lat = re.search(r"lat\W*?(-?\d+(?:\.\d+)?)", item)
        lon = re.search(r"lon\W*?(-?\d+(?:\.\d+)?)", item)
        prob = re.search(r"\"p\"?\s*:\s*\"?(\d*\.?\d+)", item)
        if lat and lon and abs(float(lat.group(1))) <= 90 and abs(float(lon.group(1))) <= 180:
            hypotheses.append((float(lat.group(1)), float(lon.group(1)), float(prob.group(1)) if prob else 0.0))
    total = sum(p for _, _, p in hypotheses)
    hypotheses = [(lat, lon, p / total if total > 0 else 1 / len(hypotheses)) for lat, lon, p in hypotheses]
    text = re.search(r"\"text\"\s*:\s*\"([^\"]*)\"", answer)
    return {
        "guesses": [[lat, lon] for lat, lon, _ in hypotheses],
        "probabilities": [p for _, _, p in hypotheses],
        "overlay": bool(re.search(r"gps_overlay\W*true", answer)),
        "text": text.group(1) if text else "",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, help="OSV coarse-filter rows (query_id, index)")
    parser.add_argument("--benchmark-queries", type=Path, help="strategy_search queries.json (benchmark, image_id, path)")
    parser.add_argument("--per-benchmark", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--model", default="gemma-4-26b-a4b")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    if args.benchmark_queries is not None:
        # Flickr benchmark photos: a fixed random sample per benchmark, addressed by path.
        dataset, kind = None, "photo"
        rows = json.loads(args.benchmark_queries.read_text(encoding="utf-8"))
        rng = np.random.default_rng(args.seed)
        queries = []
        for benchmark in sorted({row["benchmark"] for row in rows}):
            members = [row for row in rows if row["benchmark"] == benchmark]
            for i in sorted(rng.choice(len(members), size=min(args.per_benchmark, len(members)), replace=False)):
                queries.append({"query_id": f"{benchmark}:{members[i]['image_id']}", "path": members[i]["path"]})
    else:
        dataset, kind = OSVTestReferenceDataset(OSV_ROOT), "street-level photo"
        queries = [json.loads(line) for line in args.queries.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompt = PROMPT.replace("{kind}", kind)
    done = set()
    if args.output.exists():
        # Re-parse earlier answers with the current parser and drop transport errors so they are retried.
        kept = []
        for row in map(json.loads, args.output.read_text(encoding="utf-8").splitlines()):
            if row["raw"].startswith("error:"):
                continue
            kept.append({**row, **parse_answer(row["raw"])})
            done.add(row["episode_id"])
        args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")
    pending = [q for q in queries if q["query_id"] not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def work(query: dict[str, Any]) -> dict[str, Any]:
        path = Path(query["path"]) if "path" in query else dataset.image_path_for_id(dataset.image_id_at(query["index"]))
        try:
            result = _ask(args.server, args.model, path, prompt)
        except Exception as error:  # keep the sweep going; failures are recorded as empty hypotheses
            result = {"guesses": [], "probabilities": [], "overlay": False, "text": "", "raw": f"error: {error}"}
        return {"episode_id": query["query_id"], "model": args.model, **result}

    with args.output.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(args.workers) as pool:
        for number, row in enumerate(pool.map(work, pending), start=1):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{number}/{len(pending)} {row['episode_id']} hypotheses={len(row['guesses'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
