# "Ask an expert" as a trained judge (after lessons 104-106 and 114: Sonnet told the head's top two areas by town name gained +0.94 pts at
# 25 km with calibrated confidence; local models zero-shot were at chance). The 4B sees the photo and two named candidate areas and answers A
# or B; it is trained with GRPO on ground truth (reward: the right letter), LoRA on the base Qwen3.5-4B, thinking off.
# Usage: PYTHONPATH=src PINPOINT_KEEP_OWN=1 .venv/bin/python -m geo_search_env.experiment.names_grpo pairs
#        NAME=base RUN="names_grpo ask" bash scripts/clue_judge.sh        (zero-shot; MODEL=<merged dir> NAME=<run> for a trained judge)
#        PYTHONPATH=src .venvs/grpo/bin/python -m geo_search_env.experiment.names_grpo train --run names-grpo-a --hours 4
#        PYTHONPATH=src .venvs/grpo/bin/python -m geo_search_env.experiment.names_grpo merge --run names-grpo-a
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.names_grpo report --name <name>

"""pairs:  the answer head's (setft_own, --one-km 0.5; out-of-fold on train, all of train for dev3k) two most probable areas per photo under
        the full view (read, recognise, every area zoomed), each at the point the head would answer there, named "near <town> (lat, lon),
        <country>" -> artifacts/agent/names_grpo_pairs.jsonl. Splits: grpo (train used for GRPO), held (5,000 train photos kept out of GRPO,
        for choosing the switch threshold), dev (dev3k). A pair is decisive when exactly one area is < 25 km.
ask:    P(A) from the first-token logprobs, both orders, on the held and dev decisive pairs -> artifacts/agent/names_grpo_{name}.jsonl.
train:  GRPO on grpo decisive pairs, balanced between the head's top area being right and its second, A / B order random.
report: accuracy on decisive pairs (head-right / head-wrong), and following the judge when P(head's second) >= t, t chosen on held,
        fixes / breaks / net pts of dev3k at 25 km.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path
import shutil
import time
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
PAIRS = OUT / "names_grpo_pairs.jsonl"
BASE = Path("data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
CKPT = "setft_own"
N_HELD = 5000
PROMPT = """You are an expert geolocator. This photo was taken within about 25 km of exactly one of two places:
A = {a}
B = {b}
Use text and language, brands, landmarks, events, architecture, landscape and terrain, vegetation, climate and what you know about each
place. Reply with exactly one letter, A or B, and nothing else. Do not describe the photo."""


def _names(points: np.ndarray) -> list[str]:
    """ "near <town> (lat, lon), <country>": the nearest GeoNames town of >= 5,000 people (as names_feature, lesson 114)."""

    from scipy.spatial import cKDTree

    from ..models.pinpoint_reranker import _unit_xyz

    rows = [line.split("\t") for line in (OUT / "cities5k.tsv").read_text(encoding="utf-8").splitlines()]
    country = {}
    for line in Path("data/pinpoint/geonames/countryInfo.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            f = line.split("\t")
            country[f[0]] = f[4]
    ll = np.asarray([[float(r[1]), float(r[2])] for r in rows])
    _, idx = cKDTree(_unit_xyz(ll)).query(_unit_xyz(points))
    return [f"near {rows[i][0]} ({a:.2f}, {b:.2f}), {country.get(rows[i][3], rows[i][3])}" for i, (a, b) in zip(idx, points)]


def _points(tag: str, d: dict) -> tuple[np.ndarray, np.ndarray]:
    """[n, h, 2] lat / lon of each area's unzoomed and zoomed answer points, in the slot layout of hypothesis_agent.load (gated zoom rule)."""

    from .hypothesis_agent import SOURCES, _kernel_km, _paths, _unzoomed_km

    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT}" / "photos.json").read_text(encoding="utf-8"))
    zm = np.load(_paths(tag, CKPT)[1])
    rows = np.flatnonzero(np.isin(zm["source"], SOURCES))
    photo = zm["photo"][rows]
    truth = np.asarray([e["truth"] for e in photos], np.float64)
    _, pick = _kernel_km(zm, rows, truth, gate=True, return_pick=True)
    zoomed = np.where((pick >= 0)[:, None], zm["latlon"][rows, np.maximum(pick, 0)], zm["center"][rows])
    _, upick = _unzoomed_km({"photo": photo, "center": zm["center"][rows]}, photos, return_pick=True)
    unzoomed = np.asarray([photos[p]["pool"][k] if k >= 0 else zm["center"][r] for p, k, r in zip(photo, upick, rows)], np.float64)
    n, h = d["mask"].shape
    out_u, out_z = np.full((n, h, 2), np.nan), np.full((n, h, 2), np.nan)
    slot = np.zeros(len(rows), int)
    seen = np.zeros(n, int)
    for j, p in enumerate(photo):
        slot[j] = seen[p]
        seen[p] += 1
    out_u[photo, slot], out_z[photo, slot] = unzoomed, zoomed
    return out_u, out_z


def pairs(seed: int = 0) -> None:
    import torch

    from . import action_agent as aa
    from .hypothesis_agent import _sub

    aa.CKPT_TAG, aa.ONE_KM = CKPT, 0.5
    tr = aa._data("train")
    dv = aa._data("dev3k", stats=tr["stats"])
    n = tr["n"]
    rng = np.random.default_rng(seed)
    fold = rng.permutation(n) % 5
    held = np.zeros(n, bool)
    held[rng.permutation(n)[:N_HELD]] = True

    def area_logp(net, d, photos):
        out_a, out_z = [], []
        with torch.no_grad():
            for c in np.array_split(photos, max(1, len(photos) // 512)):
                b = _sub(d, torch.as_tensor(c, device="cuda"))
                on = torch.ones(len(c), dtype=torch.bool, device="cuda")
                v = aa.view(b, on, on)
                lp = torch.log_softmax(aa._options(net, v, v["mask"]), 1)
                h = v["mask"].shape[1]
                out_a.append(torch.logsumexp(torch.stack([lp[:, :h], lp[:, h:]]), 0).cpu().numpy())
                out_z.append((lp[:, h:] > lp[:, :h]).cpu().numpy())
        return np.concatenate(out_a), np.concatenate(out_z)

    logp = {"train": np.full(tr["mask"].shape, -np.inf), "dev3k": None}
    use_z = {"train": np.zeros(tr["mask"].shape, bool), "dev3k": None}
    for k in range(5):
        net = aa.fit_answer(tr, np.flatnonzero(fold != k), seed=7)
        te = np.flatnonzero(fold == k)
        logp["train"][te], use_z["train"][te] = area_logp(net, tr, te)
        print(f"  fold {k} done", flush=True)
    net = aa.fit_answer(tr, np.arange(n), seed=7)
    logp["dev3k"], use_z["dev3k"] = area_logp(net, dv, np.arange(dv["n"]))
    rows = []
    for tag, d in (("train", tr), ("dev3k", dv)):
        photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT}" / "photos.json").read_text(encoding="utf-8"))
        pu, pz = _points(tag, d)
        km = np.where(use_z[tag], d["km_zoom"].cpu().numpy(), d["km_unzoom"].cpu().numpy())
        pt = np.where(use_z[tag][..., None], pz, pu)
        top = np.argsort(-logp[tag], 1)[:, :2]
        for p in range(d["n"]):
            a, b = top[p]
            if not np.isfinite(logp[tag][p, b]):
                continue  # one area only
            split = "dev" if tag == "dev3k" else ("held" if held[p] else "grpo")
            rows.append({"split": split, "photo": p, "image_id": photos[p]["image_id"], "truth": photos[p]["truth"],
                         "p": [float(np.exp(logp[tag][p, a])), float(np.exp(logp[tag][p, b]))], "km": [float(km[p, a]), float(km[p, b])],
                         "point": [pt[p, a].tolist(), pt[p, b].tolist()]})
    names = _names(np.asarray([x for r in rows for x in r["point"]], np.float64))
    for i, r in enumerate(rows):
        r["names"] = names[2 * i:2 * i + 2]
    PAIRS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    for split in ("grpo", "held", "dev"):
        s = [r for r in rows if r["split"] == split]
        dec = [r for r in s if (r["km"][0] < 25) != (r["km"][1] < 25)]
        print(f"{split}: {len(s)} photos with >= 2 areas, decisive at 25 km {len(dec)} ({len(dec) / max(len(s), 1):.1%}), head's top right on "
              f"{np.mean([r['km'][0] < 25 for r in dec]):.1%} of them")
    print(f"-> {PAIRS}")


def _image_url(e: dict, images) -> str:
    from PIL import Image

    from .name_score import _read

    im = Image.open(io.BytesIO(_read(e, images))).convert("RGB")
    im.thumbnail((768, 768))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _decisive(split: str | tuple[str, ...]) -> list[dict]:
    splits = (split,) if isinstance(split, str) else split
    return [r for r in map(json.loads, PAIRS.read_text(encoding="utf-8").splitlines())
            if r["split"] in splits and (r["km"][0] < 25) != (r["km"][1] < 25)]


def ask(name: str, server: str = "http://127.0.0.1:8765", workers: int = 32) -> None:
    """P(A) by first-token logprobs, both orders, on the held and dev decisive pairs (resumable)."""

    from concurrent.futures import ThreadPoolExecutor

    from .name_score import _post
    from .sft_data import MP16Images

    rows = _decisive(("held", "dev"))
    path = OUT / f"names_grpo_{name}.jsonl"
    done = {(r["split"], r["photo"]) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())} if path.exists() else set()
    todo = [r for r in rows if (r["split"], r["photo"]) not in done]
    print(f"{name}: {len(todo)} pairs to ask ({len(done)} done)", flush=True)
    images = MP16Images()

    def run(r):
        try:
            url = _image_url(r, images)
        except Exception as e:  # noqa: BLE001
            return {"split": r["split"], "photo": r["photo"], "error": str(e)[:200]}
        p_top = []
        for order in (0, 1):
            a, b = r["names"] if order == 0 else r["names"][::-1]
            body = {"model": "vlm", "temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                              {"type": "text", "text": PROMPT.format(a=a, b=b)}]}]}
            try:
                top = _post(server, "/v1/chat/completions", body)["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            except Exception as e:  # noqa: BLE001
                return {"split": r["split"], "photo": r["photo"], "error": str(e)[:200]}
            q = {"A": 0.0, "B": 0.0}
            for t in top:
                s = t["token"].strip().upper()
                if s in q:
                    q[s] += math.exp(t["logprob"])
            pa = q["A"] / max(q["A"] + q["B"], 1e-9)
            p_top.append(pa if order == 0 else 1 - pa)  # P(the head's top area) from this order
        return {"split": r["split"], "photo": r["photo"], "p_top": p_top}

    n = 0
    with path.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(workers) as pool:
        for res in pool.map(run, todo):
            stream.write(json.dumps(res) + "\n")
            n += 1
            if n % 500 == 0:
                stream.flush()
                print(f"  {n} / {len(todo)}", flush=True)
    print(f"-> {path}", flush=True)


def report(name: str) -> None:
    pairs_ = {(r["split"], r["photo"]): r for r in _decisive(("held", "dev"))}
    ans = [r for r in map(json.loads, (OUT / f"names_grpo_{name}.jsonl").read_text(encoding="utf-8").splitlines()) if "p_top" in r]
    n_dev = sum(1 for r in map(json.loads, PAIRS.read_text(encoding="utf-8").splitlines()) if r["split"] == "dev")
    n_dev = max(n_dev, 3000)
    print(f"{name}: {len(ans)} answered pairs; mean P(first-shown) {np.mean([r['p_top'][0] for r in ans] + [1 - r['p_top'][1] for r in ans]):.2f} "
          f"(0.5 = no position bias); orders agree {np.mean([(r['p_top'][0] >= 0.5) == (r['p_top'][1] >= 0.5) for r in ans]):.0%}")
    best_t = None
    for split in ("held", "dev"):
        rs = [(float(np.mean(r["p_top"])), pairs_[(split, r["photo"])]["km"][0] < 25) for r in ans if r["split"] == split]
        if not rs:
            continue
        p, head_ok = map(np.asarray, zip(*rs))
        judge_ok = np.where(p >= 0.5, head_ok, ~head_ok)

        def roc_auc_score(y, s):
            r = np.argsort(np.argsort(s)) + 1
            return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())

        print(f"  {split}: {len(p)} decisive pairs, head right {head_ok.mean():.1%}; judge right {judge_ok.mean():.1%} (head-right pairs "
              f"{judge_ok[head_ok].mean():.1%}, head-wrong {judge_ok[~head_ok].mean():.1%}); AUC of P(head's top) for the head being right "
              f"{roc_auc_score(head_ok, p):.3f}")
        total = 3000 if split == "dev" else N_HELD
        rows_t = []
        for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            sw = p <= 1 - t
            fx, br = int((sw & ~head_ok).sum()), int((sw & head_ok).sum())
            rows_t.append((t, fx - br))
            print(f"    follow when P(head's second) >= {t}: switches {sw.sum()}, fixes {fx}, breaks {br}, net {100 * (fx - br) / total:+.2f} pts")
        if split == "held":
            best_t = max(rows_t, key=lambda x: x[1])[0]
        elif best_t is not None:
            print(f"    threshold chosen on held: {best_t}")


def train(run: str, hours: float, lr: float, beta: float, batch: int, accumulation: int, rank: int = 32, limit: int = 0,
          resume: bool = False) -> None:
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
    rng = np.random.default_rng(1)
    rows = _decisive("grpo")
    right_top = [r for r in rows if r["km"][0] < 25]
    right_second = [r for r in rows if r["km"][1] < 25]
    k = min(len(right_top), len(right_second))  # balanced: the head's preference carries no information about the answer
    rows = [right_top[i] for i in rng.permutation(len(right_top))[:k]] + [right_second[i] for i in rng.permutation(len(right_second))[:k]]
    rows = [rows[i] for i in rng.permutation(len(rows))][:limit or None]
    examples = []
    for r in rows:  # both orders, so position carries no information about the answer
        for swap in (False, True):
            a, b = r["names"][::-1] if swap else r["names"]
            right = ("B" if swap else "A") if r["km"][0] < 25 else ("A" if swap else "B")
            examples.append({"prompt": [{"role": "user", "content": PROMPT.format(a=a, b=b)}], "image_id": r["image_id"], "right": right})
    print(f"{len(examples)} GRPO examples ({k} with the head's top right, {k} with its second right)", flush=True)
    out = RUNS / run
    processor = AutoProcessor.from_pretrained(BASE, max_pixels=MAX_PIXELS)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                             target_modules=r".*language_model.*\.(" + "|".join(LORA_TARGETS) + ")$"))
    reader: list[MP16Images] = []

    def with_images(b: dict[str, list]) -> dict[str, list]:
        reader or reader.append(MP16Images())
        ims = []
        for i in b["image_id"]:
            im = Image.open(io.BytesIO(reader[0].read(i))).convert("RGB")
            im.thumbnail((768, 768))
            ims.append(im)
        return b | {"image": ims}

    dataset = Dataset.from_list(examples)
    dataset.set_transform(with_images)

    def letter(completion) -> str:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        return text.strip()[:1].upper()

    def judge_reward(completions, right, **_):
        return [1.0 if letter(c) == r else 0.0 for c, r in zip(completions, right)]

    def picks_a(completions, right, **_):  # logged only: position bias
        return [float(letter(c) == "A") for c in completions]

    class TimeLimit(TrainerCallback):
        def __init__(self) -> None:
            self.started = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            if time.time() - self.started > hours * 3600:
                control.should_training_stop = True

    config = GRPOConfig(
        output_dir=str(out), per_device_train_batch_size=batch, gradient_accumulation_steps=accumulation, use_vllm=True, vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35, vllm_enable_sleep_mode=True, vllm_max_model_length=4096, num_generations=8, max_completion_length=2,
        temperature=1.0, chat_template_kwargs={"enable_thinking": False}, reward_weights=[1.0, 0.0], learning_rate=lr,
        lr_scheduler_type="constant_with_warmup", warmup_steps=5, max_steps=100_000, beta=beta, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, eval_strategy="no", save_strategy="steps", save_steps=50,
        save_only_model=True, save_total_limit=12, report_to="none", remove_unused_columns=False, seed=0,
    )
    trainer = GRPOTrainer(model=model, args=config, processing_class=processor, train_dataset=dataset, reward_funcs=[judge_reward, picks_a],
                          callbacks=[TimeLimit()])
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("pairs", "ask", "report", "train", "merge"))
    parser.add_argument("--name", default="base")
    parser.add_argument("--tag", default="", help="ignored (scripts/clue_judge.sh passes --tag)")
    parser.add_argument("--run", default="names-grpo-a")
    parser.add_argument("--adapter", default="adapter")
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="train: examples (0: all)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "pairs":
        pairs()
    elif args.cmd == "ask":
        ask(args.name)
    elif args.cmd == "report":
        report(args.name)
    elif args.cmd == "train":
        train(args.run, args.hours, args.lr, args.beta, args.batch, args.accumulation, limit=args.limit, resume=args.resume)
    else:
        merge(args.run, args.adapter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
