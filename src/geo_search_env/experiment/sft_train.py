# LoRA SFT of Qwen3.5-4B on the MP16 SFT set, LoRA merge for vLLM, and decoding eval on MP16 validation photos.
# Usage: ~/.venvs/sft/bin/python -m geo_search_env.experiment.sft_train train --examples 5000 --run sft-5k
#        ~/.venvs/sft/bin/python -m geo_search_env.experiment.sft_train merge --run sft-5k
#        .venv/bin/python -m geo_search_env.experiment.sft_train val_eval --model sft-5k  (vLLM serving the merged model)

"""Supervised fine-tuning on `artifacts/sft/sft.jsonl` (built by `sft_data dataset`).

Prompts are rendered with the chat template exactly as vLLM renders them at test time (thinking off), and the loss
covers only the answer and its end-of-turn token. Training subsets are nested (the first N of one fixed shuffle), so
runs of different sizes form a learning curve.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import re
import shutil
import urllib.request
from typing import Any, Sequence

import numpy as np

from ..data.benchmarks import compute_metrics
from .pivot_diagnostics import THRESHOLDS_KM, parse_coordinates
from .sft_data import MP16Images
from .strategy_search import _haversine_km


BASE_MODEL = Path("/data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
DATA = Path("artifacts/sft/sft.jsonl")
RUNS = Path("/data/pinpoint/sft")
MAX_PIXELS = 1024 * 768  # vLLM serves with the same cap (--mm-processor-kwargs '{"max_pixels": 786432}')
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
EVAL_LOSS_EXAMPLES = 500
VAL_DECODE_EXAMPLES = 1_000
SAMPLES = 8
TEMPERATURE = 1.0
TOKENIZER_FILES = ("chat_template.jinja", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "preprocessor_config.json", "video_preprocessor_config.json")


def load_rows(split: str, limit: int | None = None, seed: int = 0, data: Path = DATA) -> list[dict[str, Any]]:
    """Rows of one split in a fixed shuffled order, so smaller training sets are prefixes of larger ones."""

    rows = [row for row in map(json.loads, data.read_text(encoding="utf-8").splitlines()) if row["split"] == split]
    order = np.random.default_rng(seed).permutation(len(rows))
    return [rows[i] for i in order[:limit]]


class Collator:
    """Chat-template rendering, image loading and answer-only labels for a batch of SFT rows."""

    def __init__(self, processor) -> None:
        from PIL import Image

        self.processor = processor
        self.image_open = Image.open
        self.images: MP16Images | None = None  # opened lazily, once per dataloader worker
        self.think_end = processor.tokenizer.convert_tokens_to_ids("</think>")
        self.turn_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        self.images = self.images or MP16Images()
        texts, images = [], []
        for row in rows:
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["prompt"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": row["target"]}]},
            ]
            # enable_thinking must be passed directly: processors ignore chat_template_kwargs.
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, enable_thinking=False))
            images.append(self.image_open(io.BytesIO(self.images.read(row["image_id"]))).convert("RGB"))
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = torch.full_like(batch["input_ids"], -100)
        for i, ids in enumerate(batch["input_ids"].tolist()):
            # The answer follows the empty think block "</think>\n\n" and ends with <|im_end|> (kept, so the model stops).
            start = len(ids) - ids[::-1].index(self.think_end) + 1
            stop = start + ids[start:].index(self.turn_end) + 1
            labels[i, start:stop] = batch["input_ids"][i, start:stop]
        batch["labels"] = labels
        return batch


def train(
    examples: int, run: str, *, data: Path = DATA, lr: float = 1e-4, batch: int = 8, accumulation: int = 4, rank: int = 32, seed: int = 0,
    checkpointing: bool = True, max_steps: int = -1, max_pixels: int = MAX_PIXELS,
) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import SFTConfig, SFTTrainer

    out = RUNS / run
    processor = AutoProcessor.from_pretrained(BASE_MODEL, max_pixels=max_pixels)  # serve with the same --mm-processor-kwargs max_pixels
    model = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
        target_modules=r".*language_model.*\.(" + "|".join(LORA_TARGETS) + ")$",  # not lm_head: chunked CE reads its weight directly
    ))
    model.print_trainable_parameters()
    train_rows, eval_rows = load_rows("train", examples, seed, data), load_rows("val", EVAL_LOSS_EXAMPLES, seed, data)
    steps = len(train_rows) // (batch * accumulation)
    config = SFTConfig(
        output_dir=str(out),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=accumulation,
        num_train_epochs=1,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, steps * 3 // 100),
        weight_decay=0.0,
        bf16=True,
        gradient_checkpointing=checkpointing,
        max_steps=max_steps,  # > 0 only for speed tests
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="steps" if max_steps < 0 else "no",
        eval_steps=max(10, steps // 5),
        save_strategy="no",
        report_to="none",
        seed=seed,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_num_workers=4,
        max_length=None,
    )
    trainer = SFTTrainer(
        model=model, args=config, processing_class=processor, data_collator=Collator(processor),
        train_dataset=Dataset.from_list(train_rows), eval_dataset=Dataset.from_list(eval_rows),
    )
    print(f"loss_type={trainer.args.loss_type} steps={steps} train={len(train_rows)} eval={len(eval_rows)}", flush=True)
    trainer.train()
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    if max_steps > 0:
        return  # speed test: nothing worth keeping
    trainer.save_model(str(out / "adapter"))
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1) + "\n", encoding="utf-8")


def merge(run: str) -> None:
    """Fold the LoRA into the base weights and save a checkpoint vLLM can serve."""

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    out = RUNS / run / "merged"
    model = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(RUNS / run / "adapter")).merge_and_unload()
    model.save_pretrained(out)
    for name in TOKENIZER_FILES:
        shutil.copy(BASE_MODEL / name, out / name)
    print(f"merged into {out}")


def _decode(server: str, model: str, image: bytes, prompt: str, temperature: float, n: int) -> list[tuple[float, float] | None]:
    body = {
        "model": model, "temperature": temperature, "seed": 1, "n": n, "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}},
            {"type": "text", "text": prompt},
        ]}],
    }
    request = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return [parse_coordinates(c["message"]["content"] or "") for c in json.loads(response.read())["choices"]]


def val_eval(model: str, *, data: Path = DATA, server: str = "http://127.0.0.1:8765", examples: int = VAL_DECODE_EXAMPLES, workers: int = 64, temperature: float = TEMPERATURE) -> None:
    """Greedy, single-sample and best-of-8 accuracy on MP16 validation photos (held-out photographers)."""

    rows = load_rows("val", examples, seed=1, data=data)
    images = MP16Images()

    def run(row):
        image = images.read(row["image_id"])
        return _decode(server, model, image, row["prompt"], 0.0, 1)[0], _decode(server, model, image, row["prompt"], temperature, SAMPLES)

    with ThreadPoolExecutor(workers) as pool:
        answers = list(pool.map(run, rows))
    truth = np.asarray([(r["meta"]["lat"], r["meta"]["lon"]) for r in rows])
    first = np.asarray([[float(v) for v in re.search(r"^1\. .*?\((-?[\d.]+), (-?[\d.]+)\)", r["prompt"], re.M).groups()] for r in rows])
    reranker = np.asarray([r["meta"].get("reranker_top1", f) for r, f in zip(rows, first)])  # older datasets show it first
    point = lambda a, i: a if a is not None else tuple(first[i])  # unparseable answers fall back to the first shown candidate
    greedy = np.asarray([point(g, i) for i, (g, _) in enumerate(answers)])
    sampled = np.asarray([[point(s, i) for s in samples] for i, (_, samples) in enumerate(answers)])
    sample_km = np.stack([_haversine_km(*t, s) for t, s in zip(truth, sampled)])
    report: dict[str, Any] = {
        "model": model, "n": len(rows), "temperature": temperature,
        "unparsed_rate": float(np.mean([a is None for g, s in answers for a in [g, *s]])),
        "reranker top-1": compute_metrics(reranker, truth),
        "greedy": compute_metrics(greedy, truth),
        "mean single sample": {f"Under_{int(t)}_km": float((sample_km < t).mean()) for t in THRESHOLDS_KM},
        f"best of {SAMPLES} (oracle)": {f"Under_{int(t)}_km": float((sample_km < t).any(1).mean()) for t in THRESHOLDS_KM},
    }
    path = Path("artifacts/sft") / (f"val_eval_{model}.json" if temperature == TEMPERATURE else f"val_eval_{model}_t{temperature:g}.json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"MP16 val n={len(rows)} (unparsed {report['unparsed_rate']:.1%})   <1km  <25km <200km <750km")
    for arm in ("reranker top-1", "greedy", "mean single sample", f"best of {SAMPLES} (oracle)"):
        print(f"  {arm:24s}" + "".join(f"{report[arm][f'Under_{int(t)}_km']:7.1%}" for t in THRESHOLDS_KM))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("train", "merge", "val_eval"))
    parser.add_argument("--run", help="run name (checkpoints under /data/pinpoint/sft/<run>)")
    parser.add_argument("--examples", type=int, help="training examples (a prefix of one fixed shuffle)")
    parser.add_argument("--data", type=Path, default=DATA, help="SFT jsonl (e.g. artifacts/sft/sft_evidence.jsonl)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--no-checkpointing", action="store_true", help="disable gradient checkpointing")
    parser.add_argument("--max-steps", type=int, default=-1, help="stop early (speed tests)")
    parser.add_argument("--max-pixels", type=int, default=MAX_PIXELS, help="image size cap in training (serve with the same max_pixels)")
    parser.add_argument("--model", help="served model name (val_eval)")
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="sampling temperature (val_eval)")
    args = parser.parse_args(argv)
    if args.node == "train":
        train(
            args.examples, args.run, data=args.data, lr=args.lr, batch=args.batch, accumulation=args.accumulation,
            checkpointing=not args.no_checkpointing, max_steps=args.max_steps, max_pixels=args.max_pixels,
        )
    elif args.node == "merge":
        merge(args.run)
    else:
        val_eval(args.model, data=args.data, server=args.server, temperature=args.temperature)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
