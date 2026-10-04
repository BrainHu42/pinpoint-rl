# LoRA fine-tuning of Qwen3.5-4B as a "same place?" comparator for (query photo, exemplar photo) pairs.
# Usage: ~/.venvs/sft/bin/python -m geo_search_env.experiment.comparator_train train --run comparator-a --samples 40000
#        ~/.venvs/sft/bin/python -m geo_search_env.experiment.comparator_train train --run comparator-b --mode multi --samples 0 --init-adapter /data/pinpoint/sft/comparator-a/adapter
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
PROMPT_PAIR = "Which of photos 2 and 3 was taken at the same place as photo 1? Answer 2 or 3."
PAIRWISE = ROOT / "comparator" / "pairwise_train.jsonl"
SIZE = 448
NEG_PER_POSITIVE_PHOTO = 3
PAIRS = ROOT / "comparator" / "pairs_train.jsonl"
PAIRS_ALL = ROOT / "comparator" / "pairs_train_all.jsonl"  # every top-8 candidate with its distance (comparator_data.py all)
PAIRS_MULTI = ROOT / "comparator" / "pairs_train_multi.jsonl"  # up to 4 exemplars per labelled candidate (comparator_data.py multi)
HARD_RANKED = 2  # multi mode: best-ranked negatives kept with all their exemplars


def _multi_rows(seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rows from pairs_train_multi.jsonl. Photos with a positive: every exemplar of every positive, every exemplar of the HARD_RANKED best-ranked
    negatives (the reranker's most plausible mistakes) and the first exemplar of up to 2 other negatives. Photos without one: the first exemplar of
    the best-ranked negative and of one random other. Held-out photos as in load_rows."""

    by_candidate: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for line in PAIRS_MULTI.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        by_candidate.setdefault((r["photo"], r["rank"]), []).append(r)
    by_photo: dict[int, list[list[dict[str, Any]]]] = {}
    for (photo, _), rs in sorted(by_candidate.items()):
        by_photo.setdefault(photo, []).append(rs)
    rng = np.random.default_rng(seed)
    train, held = [], []
    for photo, candidates in by_photo.items():
        pos = [rs for rs in candidates if rs[0]["label"] == 1]
        neg = [rs for rs in candidates if rs[0]["label"] == 0]  # in reranker order
        if pos:
            others = neg[HARD_RANKED:]
            take = [r for rs in pos + neg[:HARD_RANKED] for r in rs] + [others[i][0] for i in rng.permutation(len(others))[:2]]
        else:
            take = [neg[0][0]] + [neg[1:][i][0] for i in rng.permutation(len(neg) - 1)[:1]] if neg else []
        (held if photo % 33 == 0 else train).extend(take)
    return train, held


def load_rows(samples: int | None, seed: int = 0, mode: str = "same", label_km: tuple[float, float] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if mode == "multi":
        train, held = _multi_rows(seed)
        order = np.random.default_rng(seed).permutation(len(train))
        return [train[i] for i in order[: samples or None]], held
    if mode == "pairwise":  # rows are (query, right exemplar, wrong exemplar); the same 3% of photos is held out
        rows = [json.loads(line) for line in PAIRWISE.read_text(encoding="utf-8").splitlines()]
        order = np.random.default_rng(seed).permutation(len(rows))
        rows = [rows[i] for i in order]
        return [r for r in rows if r["photo"] % 33 != 0][:samples], [r for r in rows if r["photo"] % 33 == 0]
    if label_km:  # positive below label_km[0], negative from label_km[1] on, the band between dropped
        rows = [json.loads(line) for line in PAIRS_ALL.read_text(encoding="utf-8").splitlines()]
        rows = [dict(r, label=1 if r["km"] < label_km[0] else 0) for r in rows if r["km"] < label_km[0] or r["km"] >= label_km[1]]
    else:
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

    def __init__(self, processor, mode: str = "same") -> None:
        self.processor = processor
        self.mode = mode
        self.images: MP16Images | None = None  # opened lazily, once per dataloader worker
        self.ids = {w: processor.tokenizer.convert_tokens_to_ids(w) for w in (("3", "2") if mode == "pairwise" else ("Yes", "No"))}  # (positive, negative) answer tokens
        processor.tokenizer.padding_side = "left"

    def _image(self, image_id: str):
        from PIL import Image

        image = Image.open(io.BytesIO(self.images.read(image_id))).convert("RGB")
        image.thumbnail((SIZE, SIZE))
        return image

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        self.images = self.images or MP16Images()
        pair = self.mode == "pairwise"
        content = [{"type": "image"}] * (3 if pair else 2) + [{"type": "text", "text": PROMPT_PAIR if pair else PROMPT}]
        text = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if pair:
            third_right = [(r["photo"] + 7 * r["right_rank"] + r["wrong_rank"]) % 2 == 0 for r in rows]  # which of the two exemplars comes third: fixed per row, half and half
            images = [im for r, t in zip(rows, third_right)
                      for im in (self._image(r["query"]), self._image(r["wrong"] if t else r["right"]), self._image(r["right"] if t else r["wrong"]))]
            labels = [float(t) for t in third_right]
        else:
            images = [im for r in rows for im in (self._image(r["query"]), self._image(r["exemplar"]))]
            labels = [r["label"] for r in rows]
        batch = self.processor(text=[text] * len(rows), images=images, return_tensors="pt", padding=True)
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label == 1][:, None], score[label == 0][None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def train(run: str, samples: int, *, lr: float = 1e-4, batch: int = 8, accumulation: int = 2, rank: int = 32, seed: int = 0, max_steps: int = -1, mode: str = "same", init_adapter: str | None = None,
          label_km: tuple[float, float] | None = None) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForImageTextToText, AutoProcessor

    out = RUNS / run
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, max_pixels=SIZE * SIZE)
    model = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).cuda()
    if init_adapter:  # continue from an earlier adapter (e.g. the pointwise comparator for the pairwise task)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=r".*language_model.*\.(" + "|".join(LORA_TARGETS) + ")$"))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    train_rows, held_rows = load_rows(samples, seed, mode, label_km)
    pos = sum(r["label"] for r in train_rows) if mode != "pairwise" else len(train_rows) // 2
    print(f"train rows {len(train_rows)} ({pos} positive), held-out rows {len(held_rows)}", flush=True)
    collator = Collator(processor, mode)
    yes, no = list(collator.ids.values())
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
    parser.add_argument("--mode", choices=("same", "pairwise", "multi"), default="same")
    parser.add_argument("--init-adapter", help="start from this LoRA adapter instead of a fresh one")
    parser.add_argument("--label-km", type=float, nargs=2, metavar=("POSITIVE", "NEGATIVE"), help="pointwise labels from pairs_train_all.jsonl: positive below the first distance, negative from the second")
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1, help="speed test: stop after this many optimizer steps and keep nothing")
    args = parser.parse_args(argv)
    train(args.run, args.samples, lr=args.lr, batch=args.batch, accumulation=args.accumulation, max_steps=args.max_steps, mode=args.mode, init_adapter=args.init_adapter,
          label_km=tuple(args.label_km) if args.label_km else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
