# RL on the 4B's clue output (lesson 87; the user's design): the 4B sees the photo and Pinpoint's top-1 region and first says whether anything
# in the photo is place-specific. If not, it outputs nothing and Pinpoint's answer stands; otherwise it names the evidence and either keeps
# Pinpoint's region or overrides it with a country / region. GRPO (TRL, vLLM colocated, LoRA on the base Qwen3.5-4B, thinking off), reward from
# labels only (no tools in the loop): a right override where Pinpoint's region is wrong +1, a wrong override -1 where Pinpoint was right and
# -0.5 where both are wrong, none / keep 0. Evaluation turns overrides into answers with Pinpoint restricted to the region (tools.pinpoint).
# Usage: PINPOINT_RERANKER_CKPT=artifacts/pinpoint_train/setft_a/checkpoints/ckpt_final.pt PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.clue_grpo data
#        PYTHONPATH=src .venvs/grpo/bin/python -m geo_search_env.experiment.clue_grpo train --run clue-grpo-a --hours 2.5
#        PYTHONPATH=src .venvs/grpo/bin/python -m geo_search_env.experiment.clue_grpo merge --run clue-grpo-a
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.clue_grpo answer --model <dir> --name base   (VLM served at :8765)
#        PINPOINT_RERANKER_CKPT=... PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.clue_grpo evaluate --name base

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, Sequence

import numpy as np

OUT = Path("artifacts/agent")
DATA = OUT / "clue_grpo_data.jsonl"
VOCAB = OUT / "region_vocab.json"
BASE = Path("data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
PROMPT = ("A visual search engine places this photo in {pin}.\n"
          "First decide whether the photo shows anything place-specific: readable text, a recognisable landmark or building, or a feature "
          "typical of one country or region (signs, plates, architecture, vegetation, vehicles). Most photos (people, pets, food, flowers, "
          "interiors, generic landscapes) show nothing place-specific; then answer {{\"evidence\": \"none\"}}.\n"
          "Otherwise name the evidence briefly and either keep the search engine's region, {{\"evidence\": \"...\", \"answer\": \"keep\"}}, or "
          "give the country and first-level region you are confident in, {{\"evidence\": \"...\", \"country\": \"...\", \"region\": \"...\"}}. "
          "Only override when the evidence clearly points elsewhere. Answer with JSON only.")
R_FIX, R_BREAK, R_BOTH_WRONG, R_UNPARSED = 1.0, -1.0, -0.5, -0.5


def _pin_text(country: str, region: str) -> str:
    return f"{region.split('|', 1)[1]}, {country}" if region else (country or "an unknown place")


def data() -> None:
    """Train and dev3k photos with the truth's and Pinpoint's (retrained top-1) country / region labels, and the label vocabulary."""

    from ..tools.pinpoint import Pinpoint

    tool = Pinpoint(targets=None)
    country, region = tool.labels()
    pairs = sorted({(c, r.split("|", 1)[1]) for c, r in zip(country, region) if c and r})
    VOCAB.write_text(json.dumps({"countries": sorted(set(country) - {""}), "regions": [f"{c}|{r}" for c, r in pairs]}, ensure_ascii=False),
                     encoding="utf-8")
    with DATA.open("w", encoding="utf-8") as out:
        for split, tag in (("train", "train_setft_a"), ("dev", "dev3k_setft_a")):
            photos = json.loads((Path("artifacts/photo_sets") / tag / "photos.json").read_text(encoding="utf-8"))
            tc, tr = tool.mp16_labels(np.asarray([e["row"] for e in photos]))
            pc, pr = tool.labels_at(np.asarray([e["candidates"][0][1:] for e in photos], np.float64))
            for i, e in enumerate(photos):
                out.write(json.dumps({"split": split, "index": i, "image_id": e["image_id"], "truth": e["truth"], "top1": e["candidates"][0][1:],
                                      "truth_country": str(tc[i]), "truth_region": str(tr[i]), "pin_country": str(pc[i]),
                                      "pin_region": str(pr[i])}, ensure_ascii=False) + "\n")
    print(f"-> {DATA}, {VOCAB}")


def _names():
    from .nested_passk import _Names

    v = json.loads(VOCAB.read_text(encoding="utf-8"))
    return _Names(np.asarray(v["countries"]), np.asarray(v["regions"]))


def parse(text: str, names) -> tuple[str, str, str]:
    """-> (kind, country, region): kind none / keep / override / unparsed; names mapped to the labels ('' if unknown)."""

    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        obj = json.loads(m.group(0)) if m else None
    except Exception:  # noqa: BLE001
        obj = None
    if not isinstance(obj, dict):
        return "unparsed", "", ""
    if str(obj.get("evidence", "")).strip().lower() in ("none", "") and not obj.get("country"):
        return "none", "", ""
    if str(obj.get("answer", "")).strip().lower() == "keep" or not obj.get("country"):
        return "keep", "", ""
    c = names.country(str(obj.get("country")))
    r = names.region(c, str(obj.get("region") or "") or None, str(obj.get("country")))
    return "override", c or "", r or ""


def reward_of(kind: str, c: str, r: str, row: dict) -> float:
    if kind == "unparsed":
        return R_UNPARSED
    if kind in ("none", "keep"):
        return 0.0
    if r:  # region-level override
        if r == row["pin_region"]:
            return 0.0
        if r == row["truth_region"]:
            return R_FIX
        return R_BREAK if row["pin_region"] == row["truth_region"] else R_BOTH_WRONG
    if c and c != row["pin_country"]:  # country only
        if c == row["truth_country"]:
            return 0.5 * R_FIX
        return R_BREAK if row["pin_country"] == row["truth_country"] else R_BOTH_WRONG
    return 0.0 if c else R_BOTH_WRONG  # same country as Pinpoint = keep; unknown names cost


def train(run: str, hours: float, lr: float, beta: float, batch: int, accumulation: int, rank: int = 32, limit: int = 5000,
          init_adapter: str = "", resume: bool = False) -> None:
    """init_adapter: start from an earlier run's LoRA (e.g. the pilot's), resume: continue this run from its last checkpoint (after a
    restart; the photo order is fixed, so the run picks up where it stopped)."""

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
    names = _names()
    rows = [json.loads(x) for x in DATA.read_text(encoding="utf-8").splitlines()]
    train_rows = [r for r in rows if r["split"] == "train" and r["truth_region"]]  # unlabelled truths can't score an override
    train_rows = [train_rows[i] for i in np.random.default_rng(1).permutation(len(train_rows))[:limit or None]]
    out = RUNS / run
    processor = AutoProcessor.from_pretrained(BASE, max_pixels=MAX_PIXELS)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16)
    if init_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(RUNS / init_adapter), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                                 target_modules=r".*language_model.*\.(" + "|".join(LORA_TARGETS) + ")$"))
    reader: list[MP16Images] = []

    def with_images(b: dict[str, list]) -> dict[str, list]:
        reader or reader.append(MP16Images())
        return b | {"image": [Image.open(io.BytesIO(reader[0].read(i))).convert("RGB") for i in b["image_id"]]}

    dataset = Dataset.from_list([{
        "prompt": [{"role": "user", "content": PROMPT.format(pin=_pin_text(r["pin_country"], r["pin_region"]))}],  # TRL adds the image
        "image_id": r["image_id"], "row": json.dumps(r)} for r in train_rows])
    dataset.set_transform(with_images)

    def text_of(completion) -> str:
        return completion[0]["content"] if isinstance(completion, list) else completion

    def clue_reward(completions, row, **_):
        return [reward_of(*parse(text_of(c), names), json.loads(r)) for c, r in zip(completions, row)]

    def overrides(completions, row, **_):  # logged only
        return [float(parse(text_of(c), names)[0] == "override") for c in completions]

    def silent(completions, row, **_):  # logged only
        return [float(parse(text_of(c), names)[0] == "none") for c in completions]

    class TimeLimit(TrainerCallback):
        def __init__(self) -> None:
            self.started = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            if time.time() - self.started > hours * 3600:
                control.should_training_stop = True

    config = GRPOConfig(
        output_dir=str(out), per_device_train_batch_size=batch, gradient_accumulation_steps=accumulation, use_vllm=True, vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35, vllm_enable_sleep_mode=True, vllm_max_model_length=4096, num_generations=8, max_completion_length=96,
        temperature=1.0, chat_template_kwargs={"enable_thinking": False}, reward_weights=[1.0, 0.0, 0.0], learning_rate=lr,
        lr_scheduler_type="constant_with_warmup", warmup_steps=5, max_steps=100_000, beta=beta, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, eval_strategy="no", save_strategy="steps", save_steps=50,
        save_only_model=True, save_total_limit=12, report_to="none", remove_unused_columns=False, seed=0,
    )
    trainer = GRPOTrainer(model=model, args=config, processing_class=processor, train_dataset=dataset,
                          reward_funcs=[clue_reward, overrides, silent], callbacks=[TimeLimit()])
    trainer.train(resume_from_checkpoint=resume or None)
    out.mkdir(parents=True, exist_ok=True)
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1) + "\n", encoding="utf-8")
    trainer.save_model(str(out / "adapter"))


def merge(run: str, adapter: str = "adapter") -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    from .sft_train import RUNS, TOKENIZER_FILES

    out = RUNS / run / "merged"
    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(RUNS / run / adapter)).merge_and_unload()
    model.save_pretrained(out)
    for name in TOKENIZER_FILES:
        if (BASE / name).exists():
            shutil.copy(BASE / name, out / name)
    print(f"merged into {out}")


def answer(name: str, server: str = "http://127.0.0.1:8765", workers: int = 48) -> None:
    """Greedy answers of the served model on dev3k -> artifacts/agent/clue_grpo_dev3k_{name}.json."""

    from .name_score import _post, _read
    from .sft_data import MP16Images

    rows = [r for r in map(json.loads, DATA.read_text(encoding="utf-8").splitlines()) if r["split"] == "dev"]
    images = MP16Images()

    def one(r):
        body = {"model": "vlm", "temperature": 0.0, "max_tokens": 96, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(_read(r, images)).decode()}},
                    {"type": "text", "text": PROMPT.format(pin=_pin_text(r["pin_country"], r["pin_region"]))}]}]}
        try:
            return _post(server, "/v1/chat/completions", body)["choices"][0]["message"]["content"]
        except Exception:  # noqa: BLE001
            return None

    with ThreadPoolExecutor(workers) as pool:
        texts = list(pool.map(one, rows))
    dst = OUT / f"clue_grpo_dev3k_{name}.json"
    dst.write_text(json.dumps(texts, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{name}: {sum(t is None for t in texts)} failed -> {dst}")


def evaluate(name: str) -> None:
    """Overrides -> Pinpoint restricted to the region (or country); % < 25 km vs the retrained top-1; silence on lesson 67's labelled photos."""

    from ..tools.pinpoint import Constraint, Pinpoint
    from .strategy_search import _haversine_km, load_world

    names = _names()
    rows = [r for r in map(json.loads, DATA.read_text(encoding="utf-8").splitlines()) if r["split"] == "dev"]
    texts = json.loads((OUT / f"clue_grpo_dev3k_{name}.json").read_text(encoding="utf-8"))
    parsed = [parse(t, names) for t in texts]
    kinds = np.asarray([p[0] for p in parsed])
    rewards = np.asarray([reward_of(*p, r) for p, r in zip(parsed, rows)])
    truth = np.asarray([r["truth"] for r in rows], np.float64)
    base = np.asarray([_haversine_km(*truth[i], np.asarray([r["top1"]], np.float64))[0] < 25 for i, r in enumerate(rows)])
    final = base.copy()
    jobs = [(i, Constraint(region=p[2]) if p[2] else Constraint(country=p[1])) for i, p in enumerate(parsed)
            if p[0] == "override" and (p[2] and p[2] != rows[i]["pin_region"] or not p[2] and p[1] and p[1] != rows[i]["pin_country"])]
    if jobs:
        photos = json.loads(Path("artifacts/photo_sets/dev3k_setft_a/photos.json").read_text(encoding="utf-8"))
        tool = Pinpoint()
        mp16 = np.asarray([photos[i]["row"] for i, _ in jobs])
        found = tool.search(np.asarray(load_world().mp16["embeddings"][mp16], np.float32), [c for _, c in jobs], tool.mp16_author(mp16))
        for (i, _), ll in zip(jobs, found.latlon[:, 0]):
            if np.isfinite(ll).all():
                final[i] = _haversine_km(*truth[i], ll[None].astype(np.float64))[0] < 25
    moved = np.zeros(len(rows), bool)
    moved[[i for i, _ in jobs]] = True
    print(f"{name}: none {np.mean(kinds == 'none'):.1%}, keep {np.mean(kinds == 'keep'):.1%}, override {np.mean(kinds == 'override'):.1%} "
          f"(effective {moved.mean():.1%}), unparsed {np.mean(kinds == 'unparsed'):.1%}; mean reward {rewards.mean():+.3f}")
    right_region = np.asarray([p[2] == r["truth_region"] for p, r in zip(parsed, rows)])
    print(f"  effective overrides: region right {right_region[moved].mean() if moved.any() else 0:.1%}; % < 25 km {final.mean():.1%} vs retrained "
          f"top-1 {base.mean():.1%} ({100 * (final.mean() - base.mean()):+.2f}); fixes {(final & ~base).sum()}, breaks {(~final & base).sum()}")
    loc = Path("fixtures/locatability_dev3k.json")
    if loc.exists():
        tiers = json.loads(loc.read_text(encoding="utf-8"))
        tier_of = {t["image_id"]: t.get("tier") for t in (tiers["photos"] if isinstance(tiers, dict) else tiers)}
        idx = {r["image_id"]: i for i, r in enumerate(rows)}
        for label, wanted in (("nothing place-specific (X)", ("X",)), ("place-specific (P / C / R / N / K)", ("P", "C", "R", "N", "K"))):
            sel = [idx[k] for k, t in tier_of.items() if k in idx and t in wanted]
            if sel:
                print(f"  lesson 67 photos, {label}: {len(sel)}; silent (none) {np.mean(kinds[sel] == 'none'):.0%}, override {np.mean(moved[sel]):.0%}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("data", "train", "merge", "answer", "evaluate"))
    parser.add_argument("--run", default="clue-grpo-a")
    parser.add_argument("--adapter", default="adapter")
    parser.add_argument("--name", default="base")
    parser.add_argument("--hours", type=float, default=2.5)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--limit", type=int, default=5000, help="train photos (0: all)")
    parser.add_argument("--init-adapter", default="", help="train: start from this LoRA (a run's adapter dir relative to the runs root)")
    parser.add_argument("--resume", action="store_true", help="train: continue the run from its last checkpoint")
    args = parser.parse_args(argv)
    if args.cmd == "data":
        data()
    elif args.cmd == "train":
        train(args.run, args.hours, args.lr, args.beta, args.batch, args.accumulation, limit=args.limit, init_adapter=args.init_adapter,
              resume=args.resume)
    elif args.cmd == "merge":
        merge(args.run, args.adapter)
    elif args.cmd == "answer":
        answer(args.name)
    else:
        evaluate(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
