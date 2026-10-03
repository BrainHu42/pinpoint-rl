# LoRA fine-tuning of Qwen3.5-4B as a "same place?" comparator for (query photo, exemplar photo) pairs.
# Usage: ~/.venvs/sft/bin/python -m geo_search_env.experiment.comparator_train train --run comparator-a --samples 40000
#        ~/.venvs/sft/bin/python -m geo_search_env.experiment.sft_train merge --run comparator-a     (then serve the merged model with vLLM)
#        PYTHONPATH=src ~/.venvs/sft/bin/python -m geo_search_env.experiment.comparator_train train --run speed --samples 400 --max-steps 12   (speed test)

"""The prompt is rendered exactly as vLLM renders it at test time (two images, short question, thinking off); the loss is a binary cross-entropy on
logit("Yes") - logit("No") at the generation position, with positives and negatives weighted equally. At test time P(same) = P(Yes) / (P(Yes) + P(No)).

Training rows come from comparator_data.py. Photos with a positive contribute their positives (both exemplars) and up to NEG_PER_POSITIVE_PHOTO
negatives; photos without one contribute one negative. A fixed 3% of photos is held out for monitoring (pair accuracy and AUC).
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from .query_evidence import ROOT
from .sft_data import MP16Images
from .sft_train import BASE_MODEL, LORA_TARGETS, RUNS

PROMPT = "Were these two photos taken at the same place? Answer yes or no."
SIZE = 448
NEG_PER_POSITIVE_PHOTO = 3
PAIRS = ROOT / "comparator" / "pairs_train.jsonl"


def load_rows(samples: int | None, seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in PAIRS.read_text(encoding="utf-8").splitlines()]
    by_photo: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_photo.setdefault(r["photo"], []).append(r)
    rng = np.random.default_rng(seed)
    train, held = [], []
    for photo, rs in by_photo.items():
        pos, neg = [r for r in rs if r["label"] == 1], [r for r in rs if r["label"] == 0]
        take = pos + [neg[i] for i in rng.permutation(len(neg))[: NEG_PER_POSITIVE_PHOTO if pos else 1]]
        (held if photo % 33 == 0 else train).extend(take)
    order = rng.permutation(len(train))
    return [train[i] for i in order[:samples]], held


class Collator:
    """Two images per row (query first, exemplar second), the chat template with thinking off, left padding so the last position generates the answer."""

    def __init__(self, processor) -> None:
        self.processor = processor
        self.images: MP16Images | None = None  # opened lazily, once per dataloader worker
        self.ids = {w: processor.tokenizer.convert_tokens_to_ids(w) for w in ("Yes", "No")}
        processor.tokenizer.padding_side = "left"

    def _image(self, image_id: str):
        from PIL import Image

        image = Image.open(io.BytesIO(self.images.read(image_id))).convert("RGB")
        image.thumbnail((SIZE, SIZE))
        return image

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        self.images = self.images or MP16Images()
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        images = [im for r in rows for im in (self._image(r["query"]), self._image(r["exemplar"]))]
        batch = self.processor(text=[text] * len(rows), images=images, return_tensors="pt", padding=True)
        batch["labels"] = torch.tensor([r["label"] for r in rows], dtype=torch.float32)
        return batch


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label == 1][:, None], score[label == 0][None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def train(run: str, samples: int, *, lr: float = 1e-4, batch: int = 8, accumulation: int = 2, rank: int = 32, seed: int = 0, max_steps: int = -1) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForImageTextToText, AutoProcessor

    out = RUNS / run
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, max_pixels=SIZE * SIZE)
    model = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).cuda()
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=r".*language_model.*\.(" + "|".join(LORA_TARGETS) + ")$"))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    train_rows, held_rows = load_rows(samples, seed)
    pos = sum(r["label"] for r in train_rows)
    print(f"train rows {len(train_rows)} ({pos} positive), held-out rows {len(held_rows)}", flush=True)
    collator = Collator(processor)
    yes, no = collator.ids["Yes"], collator.ids["No"]
    loader = DataLoader(train_rows, batch_size=batch, shuffle=False, collate_fn=collator, num_workers=6, prefetch_factor=4, persistent_workers=True)
    steps = len(loader) // accumulation if max_steps < 0 else max_steps
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, steps * 0.03)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    weight = torch.tensor([pos / max(1, len(train_rows) - pos), 1.0]).cuda()  # [negative, positive] weights: equal total weight

    def logits_of(batch_: dict[str, Any]) -> torch.Tensor:
        labels = batch_.pop("labels")
        out_ = model(**{k: v.cuda() for k, v in batch_.items()}, logits_to_keep=1)
        z = out_.logits[:, -1, :].float()
        return z[:, yes] - z[:, no], labels.cuda()

    model.train()
    started, seen, running = time.time(), 0, []
    for i, batch_ in enumerate(loader):
        z, labels = logits_of(batch_)
        w = weight[labels.long()]
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(z, labels, reduction="none") * w).sum() / w.sum() / accumulation
        loss.backward()
        running.append(loss.item() * accumulation)
        seen += len(labels)
        if (i + 1) % accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            step = (i + 1) // accumulation
            if step % 10 == 0:
                print(f"step {step}/{steps} loss {np.mean(running[-20:]):.4f} lr {scheduler.get_last_lr()[0]:.2e} {seen / (time.time() - started):.1f} samples/s "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            if step % 500 == 0:
                model.save_pretrained(str(out / "adapter"))
            if step >= steps:
                break
    print(f"{seen} samples in {time.time() - started:.0f} s ({seen / (time.time() - started):.1f}/s); peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    if max_steps > 0:
        return
    model.save_pretrained(str(out / "adapter"))

    model.eval()
    held = held_rows[:2000]
    scores, labels_ = [], []
    with torch.no_grad():
        for start in range(0, len(held), batch):
            z, labels = logits_of(collator(held[start : start + batch]))
            scores += z.cpu().tolist(); labels_ += labels.cpu().tolist()
    s, l = np.asarray(scores), np.asarray(labels_)
    print(f"held-out train-photo pairs (n={len(l)}, {int(l.sum())} positive): accuracy {np.mean((s > 0) == (l == 1)):.1%}, AUC {auc(s, l):.3f}", flush=True)
    (out / "held_out.json").write_text(json.dumps({"n": len(l), "auc": auc(s, l), "accuracy": float(np.mean((s > 0) == (l == 1)))}) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("train",))
    parser.add_argument("--run", required=True)
    parser.add_argument("--samples", type=int, default=40_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1, help="speed test: stop after this many optimizer steps and keep nothing")
    args = parser.parse_args(argv)
    train(args.run, args.samples, lr=args.lr, batch=args.batch, accumulation=args.accumulation, max_steps=args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
