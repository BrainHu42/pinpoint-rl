# RL on the 4B's free-text search queries (lesson 96; the user: free-text queries as an action trained with RL, the case where only RL can
# train it). The 4B sees the photo only (lesson 82's blind prompt) and writes up to 2 search queries or none; each query goes through the
# text-search tool (text_search top cluster; the photo and its photographer excluded). Reward per sample, scored against the truth (not
# against Pinpoint, unlike lesson 87): the best query lands < 25 km of the truth and > 25 km from every Pinpoint area (a new right place) +1,
# < 25 km but inside a Pinpoint area (confirms one) +0.2; each query landing elsewhere -0.1, each finding nothing -0.05; none 0.
# GRPO (TRL, vLLM colocated, LoRA on the base Qwen3.5-4B, thinking off), as clue_grpo.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.query_grpo data
#        PYTHONPATH=src .venvs/grpo/bin/python -m geo_search_env.experiment.query_grpo train --run query-grpo-a --hours 2.5
#        PYTHONPATH=src .venvs/grpo/bin/python -m geo_search_env.experiment.query_grpo merge --run query-grpo-a
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.query_grpo answer --name base     (VLM served at :8765)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.query_grpo evaluate --name base

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
from pathlib import Path
import re
import time
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
DATA = OUT / "query_grpo_data.jsonl"
BASE = Path("data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
PROMPT = ("If the photo shows something a map or web search could find (a landmark, building, street, business, named place, or text naming a "
          "place), give up to 2 search queries for it, each as specific as you can (name plus town / country if you know them). If nothing in "
          'the photo is place-specific, answer none. Answer with JSON only: {"queries": ["...", "..."]} or {"queries": []}.')  # lesson 82's blind prompt
R_NEW, R_CONFIRM, R_WRONG, R_EMPTY, R_UNPARSED = 1.0, 0.2, -0.1, -0.05, -0.1
KM = 25.0


def _hav(a, b, c, d) -> float:
    a, b, c, d = map(math.radians, (a, b, c, d))
    h = math.sin((c - a) / 2) ** 2 + math.cos(a) * math.cos(c) * math.sin((d - b) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(min(1.0, h)))


def data() -> None:
    """Train and dev3k photos with truth, Pinpoint's area centres (zoom file, source 0) and the photo / photographer to exclude."""

    from .region_evidence import _owns

    with DATA.open("w", encoding="utf-8") as out:
        for split, tag in (("train", "train_setft_a"), ("dev", "dev3k_setft_a")):
            photos = json.loads((Path("artifacts/photo_sets") / tag / "photos.json").read_text(encoding="utf-8"))
            zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}.npz")
            areas: dict[int, list] = {}
            for p, s, c in zip(zm["photo"], zm["source"], zm["center"]):
                if s == 0:
                    areas.setdefault(int(p), []).append([round(float(c[0]), 5), round(float(c[1]), 5)])
            owns = _owns([{"image_id": e["image_id"]} for e in photos])
            for i, e in enumerate(photos):
                own = owns.get(e["image_id"])
                out.write(json.dumps({"split": split, "index": i, "image_id": e["image_id"], "truth": e["truth"], "areas": areas.get(i, []),
                                      "own": [int(own[0]), own[1]] if own else None}) + "\n")
    print(f"-> {DATA}")


def parse(text: str | None) -> list[str] | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        obj = json.loads(m.group(0)) if m else None
    except Exception:  # noqa: BLE001
        obj = None
    if not isinstance(obj, dict):
        return None
    return [str(q).strip() for q in (obj.get("queries") or []) if str(q).strip()][:2]


class Resolver:
    """Query -> km to the truth and to the nearest Pinpoint area of the text-search top cluster (cached per photo and query)."""

    def __init__(self) -> None:
        from .text_search import TextSearch

        self.ts = TextSearch()
        self.cache: dict[tuple[str, str], tuple[float, float] | None] = {}

    def __call__(self, q: str, row: dict) -> tuple[float, float] | None:
        key = (row["image_id"], q)
        if key not in self.cache:
            try:
                c = self.ts.search(q, exclude=tuple(row["own"]) if row["own"] else None).clusters
            except Exception:  # noqa: BLE001
                c = []
            if c:
                t = row["truth"]
                to_area = min((_hav(c[0].lat, c[0].lon, a[0], a[1]) for a in row["areas"]), default=1e9)
                self.cache[key] = (_hav(t[0], t[1], c[0].lat, c[0].lon), to_area)
            else:
                self.cache[key] = None
        return self.cache[key]


def reward_of(queries: list[str] | None, row: dict, resolve: Resolver) -> tuple[float, str]:
    """(reward, outcome): outcome new / confirm / wrong / empty / none / unparsed."""

    if queries is None:
        return R_UNPARSED, "unparsed"
    if not queries:
        return 0.0, "none"
    best, penalty, outcome = 0.0, 0.0, "wrong"
    for q in queries:
        res = resolve(q, row)
        if res is None:
            penalty += R_EMPTY
            outcome = "empty" if outcome == "wrong" and best == 0 else outcome
            continue
        to_truth, to_area = res
        if to_truth < KM and to_area > KM:
            best, outcome = max(best, R_NEW), "new"
        elif to_truth < KM:
            if best < R_CONFIRM:
                best, outcome = R_CONFIRM, "confirm"
        else:
            penalty += R_WRONG
    return best + penalty, outcome


def train(run: str, hours: float, lr: float, beta: float, batch: int, accumulation: int, rank: int = 32, limit: int = 5000) -> None:
    import functools

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    import trl.generation.vllm_generation as trl_vllm

    from .sft_data import MP16Images
    from .sft_train import LORA_TARGETS, MAX_PIXELS, RUNS

    trl_vllm.LLM = functools.partial(trl_vllm.LLM, mm_processor_kwargs={"max_pixels": MAX_PIXELS}, limit_mm_per_prompt={"image": 1})
    rows = [json.loads(x) for x in DATA.read_text(encoding="utf-8").splitlines()]
    train_rows = [r for r in rows if r["split"] == "train"]
    train_rows = [train_rows[i] for i in np.random.default_rng(1).permutation(len(train_rows))[:limit or None]]
    out = RUNS / run
    processor = AutoProcessor.from_pretrained(BASE, max_pixels=MAX_PIXELS)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                             target_modules=r".*language_model.*\.(" + "|".join(LORA_TARGETS) + ")$"))
    reader: list[MP16Images] = []
    resolve = Resolver()

    def with_images(b: dict[str, list]) -> dict[str, list]:
        reader or reader.append(MP16Images())
        return b | {"image": [Image.open(io.BytesIO(reader[0].read(i))).convert("RGB") for i in b["image_id"]]}

    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": PROMPT}], "image_id": r["image_id"], "row": json.dumps(r)}
                                 for r in train_rows])
    dataset.set_transform(with_images)

    def text_of(completion) -> str:
        return completion[0]["content"] if isinstance(completion, list) else completion

    def scored(completions, row):
        return [reward_of(parse(text_of(c)), json.loads(r), resolve) for c, r in zip(completions, row)]

    def query_reward(completions, row, **_):
        return [s[0] for s in scored(completions, row)]

    def new_place(completions, row, **_):  # logged only
        return [float(s[1] == "new") for s in scored(completions, row)]

    def silent(completions, row, **_):  # logged only
        return [float(s[1] == "none") for s in scored(completions, row)]

    class TimeLimit(TrainerCallback):
        def __init__(self) -> None:
            self.started = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            if time.time() - self.started > hours * 3600:
                control.should_training_stop = True

    config = GRPOConfig(
        output_dir=str(out), per_device_train_batch_size=batch, gradient_accumulation_steps=accumulation, use_vllm=True, vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35, vllm_enable_sleep_mode=True, vllm_max_model_length=4096, num_generations=8, max_completion_length=80,
        temperature=1.0, chat_template_kwargs={"enable_thinking": False}, reward_weights=[1.0, 0.0, 0.0], learning_rate=lr,
        lr_scheduler_type="constant_with_warmup", warmup_steps=5, max_steps=100_000, beta=beta, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, eval_strategy="no", save_strategy="steps", save_steps=50,
        save_only_model=True, save_total_limit=12, report_to="none", remove_unused_columns=False, seed=0,
    )
    trainer = GRPOTrainer(model=model, args=config, processing_class=processor, train_dataset=dataset,
                          reward_funcs=[query_reward, new_place, silent], callbacks=[TimeLimit()])
    trainer.train()
    out.mkdir(parents=True, exist_ok=True)
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1) + "\n", encoding="utf-8")
    trainer.save_model(str(out / "adapter"))


def answer(name: str, server: str = "http://127.0.0.1:8765", workers: int = 48) -> None:
    """Greedy queries of the served model for every dev3k photo -> artifacts/agent/query_grpo_dev3k_{name}.json."""

    from .name_score import _post, _read
    from .sft_data import MP16Images

    rows = [r for r in map(json.loads, DATA.read_text(encoding="utf-8").splitlines()) if r["split"] == "dev"]
    images = MP16Images()

    def one(r):
        body = {"model": "vlm", "temperature": 0.0, "max_tokens": 80, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(_read(r, images)).decode()}},
                    {"type": "text", "text": PROMPT}]}]}
        try:
            return _post(server, "/v1/chat/completions", body)["choices"][0]["message"]["content"]
        except Exception:  # noqa: BLE001
            return None

    with ThreadPoolExecutor(workers) as pool:
        texts = list(pool.map(one, rows))
    dst = OUT / f"query_grpo_dev3k_{name}.json"
    dst.write_text(json.dumps(texts, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{name}: {sum(t is None for t in texts)} failed -> {dst}")


def evaluate(name: str) -> None:
    import collections

    rows = [r for r in map(json.loads, DATA.read_text(encoding="utf-8").splitlines()) if r["split"] == "dev"]
    texts = json.loads((OUT / f"query_grpo_dev3k_{name}.json").read_text(encoding="utf-8"))
    resolve = Resolver()
    res = [reward_of(parse(t), r, resolve) for t, r in zip(texts, rows)]
    outcome = np.asarray([o for _, o in res])
    reward = np.asarray([x for x, _ in res])
    areas = {o["image_id"]: o for o in json.loads((OUT / "tag_reader_dev3k_setft_a.json").read_text(encoding="utf-8"))}
    no_area = np.asarray([not any(a["km"] < 25 for a in areas[r["image_id"]]["areas"]) for r in rows])
    asked = np.isin(outcome, ("new", "confirm", "wrong", "empty"))
    c = collections.Counter(outcome.tolist())
    print(f"{name}: {len(rows)} dev3k photos; mean reward {reward.mean():+.4f}; " + ", ".join(f"{k} {v / len(rows):.1%}" for k, v in c.most_common()))
    print(f"  queries on {asked.mean():.1%} of photos; precision (a query lands < 25 km) {np.isin(outcome[asked], ('new', 'confirm')).mean():.1%}; "
          f"new right place {np.mean(outcome == 'new'):.2%} of photos ({(outcome == 'new').sum()}), of them where no Pinpoint area is right "
          f"{((outcome == 'new') & no_area).sum()} ({((outcome == 'new') & no_area).sum() / len(rows):.2%} of photos)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("data", "train", "merge", "answer", "evaluate"))
    parser.add_argument("--run", default="query-grpo-a")
    parser.add_argument("--name", default="base")
    parser.add_argument("--tag", default="dev3k")  # dev3k only (scripts/clue_judge.sh passes it)
    parser.add_argument("--hours", type=float, default=2.5)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args(argv)
    if args.cmd == "data":
        data()
    elif args.cmd == "train":
        train(args.run, args.hours, args.lr, args.beta, args.batch, args.accumulation, limit=args.limit)
    elif args.cmd == "merge":
        from .clue_grpo import merge

        merge(args.run)
    elif args.cmd == "answer":
        answer(args.name)
    else:
        evaluate(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
