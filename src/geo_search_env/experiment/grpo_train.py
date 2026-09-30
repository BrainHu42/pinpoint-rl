# Single-turn GRPO on top of a merged SFT checkpoint, then a LoRA merge for vLLM evaluation.
# Usage: ~/.venvs/grpo/bin/python -m geo_search_env.experiment.grpo_train train --init sft-34k-retrieval --data artifacts/sft/sft_retrieval.jsonl --run grpo-kl --hours 5
#        ~/.venvs/grpo/bin/python -m geo_search_env.experiment.grpo_train merge --init sft-34k-retrieval --run grpo-kl

"""GRPO with TRL on MP16 training photos (the SFT prompts), reward GeoGuessr/5000 + 0.5·[<25 km] + 0.5·[<1 km].

Rollouts use vLLM colocated in the trainer process (env `~/.venvs/grpo`: the vLLM 0.30 env plus trl/peft/fla), sleeping
while the trainer steps. A fresh LoRA trains on the merged SFT weights and is merged into vLLM's copy after every step;
`merge` folds it in on disk so `eval_run.sh` can serve the result.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import shutil
import time
from typing import Any, Sequence

import numpy as np

from .pivot_diagnostics import parse_coordinates
from .sft_data import MP16Images
from .sft_train import DATA, LORA_TARGETS, MAX_PIXELS, RUNS, TOKENIZER_FILES, load_rows
from .strategy_search import _geoguessr, _haversine_km


SAMPLES = 8
TEMPERATURE = 1.0  # 0.7 collapsed in the pilot (entropy 0.13 -> 0.025 in 187 steps)
MAX_COMPLETION_TOKENS = 96  # SFT answers are ~46 tokens
EVAL_PHOTOS = 64  # MP16 val photos scored every EVAL_STEPS (single-sample accuracy at the rollout temperature)
EVAL_STEPS = 50
COLLAPSE_RATIO = 0.3  # stop when the 10-step mean entropy falls below this fraction of the first 10 steps


def _distances(completions: list, lat: list[float], lon: list[float]) -> list[float | None]:
    out = []
    for completion, la, lo in zip(completions, lat, lon, strict=True):
        point = parse_coordinates(completion[0]["content"] if isinstance(completion, list) else completion)
        out.append(None if point is None else float(_haversine_km(la, lo, np.asarray([point]))[0]))
    return out


def geo_reward(completions: list, lat: list[float], lon: list[float], **_) -> list[float]:
    """GeoGuessr/5000 + 0.5·[<25 km] + 0.5·[<1 km]; unparseable answers get 0."""

    return [0.0 if d is None else float(_geoguessr(np.asarray([d]))[0]) / 5000 + 0.5 * (d < 25) + 0.5 * (d < 1) for d in _distances(completions, lat, lon)]


def within_25km(completions: list, lat: list[float], lon: list[float], **_) -> list[float]:  # logged only (weight 0)
    return [float(d is not None and d < 25) for d in _distances(completions, lat, lon)]


def within_1km(completions: list, lat: list[float], lon: list[float], **_) -> list[float]:  # logged only (weight 0)
    return [float(d is not None and d < 1) for d in _distances(completions, lat, lon)]


def train(
    init: str, run: str, *, data: Path = DATA, hours: float = 2.0, max_steps: int = -1, lr: float = 5e-6, beta: float = 0.04,
    temperature: float = TEMPERATURE, batch: int = 8, accumulation: int = 8, rank: int = 32,
) -> None:
    import functools

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    import trl.generation.vllm_generation as trl_vllm

    # TRL builds the colocated engine without image settings; vLLM must cap images like the trainer's processor, or the
    # image-token counts in the prompt ids disagree.
    trl_vllm.LLM = functools.partial(trl_vllm.LLM, mm_processor_kwargs={"max_pixels": MAX_PIXELS}, limit_mm_per_prompt={"image": 1})

    out = RUNS / run
    source = RUNS / init / "merged"
    processor = AutoProcessor.from_pretrained(source, max_pixels=MAX_PIXELS)
    processor.tokenizer.padding_side = "left"  # generation needs left padding
    model = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=r".*language_model.*\.(" + "|".join(LORA_TARGETS) + ")$",
    ))
    reader: list[MP16Images] = []

    def with_images(batch: dict[str, list]) -> dict[str, list]:
        reader or reader.append(MP16Images())
        return batch | {"image": [Image.open(io.BytesIO(reader[0].read(i))).convert("RGB") for i in batch["image_id"]]}

    def as_dataset(rows: list[dict[str, Any]]):
        dataset = Dataset.from_list([
            {"prompt": [{"role": "user", "content": r["prompt"]}], "image_id": r["image_id"], "lat": r["meta"]["lat"], "lon": r["meta"]["lon"]} for r in rows
        ])
        dataset.set_transform(with_images)
        return dataset

    dataset = as_dataset(load_rows("train", data=data, seed=1))  # a different order from SFT's
    eval_dataset = as_dataset(load_rows("val", EVAL_PHOTOS, seed=2, data=data))

    class TimeLimit(TrainerCallback):
        def __init__(self) -> None:
            self.started = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            if time.time() - self.started > hours * 3600:
                control.should_training_stop = True

    class CollapseStop(TrainerCallback):
        def __init__(self) -> None:
            self.entropy: list[float] = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "entropy" in logs and "eval_entropy" not in logs:
                self.entropy.append(float(logs["entropy"]))
                if len(self.entropy) >= 20 and np.mean(self.entropy[-10:]) < COLLAPSE_RATIO * np.mean(self.entropy[:10]):
                    print(f"entropy collapsed ({np.mean(self.entropy[:10]):.3f} -> {np.mean(self.entropy[-10:]):.3f}); stopping", flush=True)
                    control.should_training_stop = True

    config = GRPOConfig(
        output_dir=str(out),
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accumulation,  # batch * accumulation completions = that / SAMPLES prompts per step
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35,
        vllm_enable_sleep_mode=True,  # frees vLLM's weights and KV cache during the trainer's forward/backward
        vllm_max_model_length=4096,
        num_generations=SAMPLES,
        max_completion_length=MAX_COMPLETION_TOKENS,
        temperature=temperature,
        chat_template_kwargs={"enable_thinking": False},  # TRL passes these straight to apply_chat_template
        reward_weights=[1.0, 0.0, 0.0],
        learning_rate=lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        max_steps=max_steps if max_steps > 0 else 100_000,  # the time limit ends the pilot
        beta=beta,  # KL to the SFT policy (the LoRA-disabled model, so no extra weights in memory)
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        per_device_eval_batch_size=2 * SAMPLES,
        eval_strategy="steps" if max_steps < 0 else "no",
        eval_steps=EVAL_STEPS,
        save_strategy="steps" if max_steps < 0 else "no",
        save_steps=EVAL_STEPS,  # LoRA-only checkpoints (~270 MB each), to pick one from before any collapse
        save_only_model=True,
        save_total_limit=12,
        report_to="none",
        remove_unused_columns=False,
        seed=0,
    )
    trainer = GRPOTrainer(
        model=model, args=config, processing_class=processor, train_dataset=dataset, eval_dataset=eval_dataset,
        reward_funcs=[geo_reward, within_25km, within_1km], callbacks=[TimeLimit(), CollapseStop()],
    )
    trainer.train()
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1) + "\n", encoding="utf-8")
    if max_steps < 0:
        trainer.save_model(str(out / "adapter"))


def merge(init: str, run: str, adapter: str = "adapter") -> None:
    """Fold the GRPO LoRA (the final `adapter` or a `checkpoint-N`) into the merged SFT weights it trained on."""

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    source, out = RUNS / init / "merged", RUNS / run / "merged"
    model = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(RUNS / run / adapter)).merge_and_unload()
    model.save_pretrained(out)
    for name in TOKENIZER_FILES:
        shutil.copy(source / name, out / name)
    print(f"merged into {out}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("train", "merge"))
    parser.add_argument("--init", required=True, help="SFT run whose merged weights GRPO starts from")
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", type=Path, default=DATA, help="SFT jsonl whose prompt format matches --init")
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="smoke test: stop after N steps and save nothing")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.04, help="KL coefficient to the SFT policy")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--adapter", default="adapter", help="merge: final adapter or a checkpoint-N directory")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=8)
    args = parser.parse_args(argv)
    if args.node == "train":
        train(args.init, args.run, data=args.data, hours=args.hours, max_steps=args.max_steps, lr=args.lr, beta=args.beta,
              temperature=args.temperature, batch=args.batch, accumulation=args.accumulation)
    else:
        merge(args.init, args.run, args.adapter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
