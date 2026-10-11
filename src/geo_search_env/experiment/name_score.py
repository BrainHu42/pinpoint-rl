# Knowledge as a score per candidate: name each candidate by the Overture divisions containing it (place_labels.py label-all) and score the name
# "country > region > city > neighbourhood" as the VLM's answer for the photo (log-probability per level from vLLM prompt logprobs), with and without the
# photo (the difference corrects for how common a name is). Then a cross-validated combiner of rank + these scores against the reranker top-1.
# Usage: RUN=base TAG=dev bash scripts/name_scores.sh     (serves the model, runs `score`, then `report`)

"""score:  per photo of a set, per distinct candidate name: log-probabilities of each level's tokens, with the photo and text-only.
report: within-photo AUC of each score (candidate < 25 km vs >= 25 km from the truth), top-1 by the score alone, and a cross-validated listwise
        combiner (5 folds x 3 seeds) of rank features + name scores, change in top-1 < 1 / 25 / 200 km vs the reranker top-1."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .exemplar_judge import _jpeg
from .multi_exemplar import _post
from .query_evidence import ROOT
from .sft_data import MP16Images

LABELS = Path("artifacts/place_labels")
LEVELS = ("country", "region", "city", "neighbourhood")
PROMPT = "Where was this photo taken? Answer as: country > region > city > neighbourhood."
SEP = " > "


def name_parts(lab: dict[str, str | None]) -> list[str]:
    """The answer's parts, coarse to fine; a level without a polygon is left out (and everything finer, so the parts stay a prefix of LEVELS)."""

    parts = []
    for value in (lab.get("country"), lab.get("region"), lab.get("locality"), lab.get("neighborhood") or lab.get("macrohood")):
        if not value:
            break
        parts.append(value.replace(">", " "))
    return parts


def _level_logprobs(server: str, parts: list[str], image_b64: str | None) -> list[float] | None:
    """Sum of the answer tokens' log-probabilities per part, from the prompt logprobs of a chat whose final assistant message is the answer."""

    answer = SEP.join(parts)
    content: list[dict[str, Any]] = [{"type": "text", "text": PROMPT}]
    if image_b64 is not None:
        content.insert(0, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}})
    body = {"model": "vlm", "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0, "add_generation_prompt": False, "continue_final_message": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": content}, {"role": "assistant", "content": answer}]}
    try:
        entries = _post(server, "/v1/chat/completions", body)["prompt_logprobs"]
    except Exception:
        return None
    # walk back from the end over the answer's tokens, collecting (decoded text, logprob)
    tokens: list[tuple[str, float]] = []
    text = ""
    for entry in reversed(entries):
        if entry is None:
            return None
        tok = next(iter(entry.values()))
        tokens.append((tok["decoded_token"], tok["logprob"]))
        text = tok["decoded_token"] + text
        if len(text) >= len(answer):
            break
    if not text.endswith(answer):
        return None
    tokens.reverse()
    offset = len(text) - len(answer)  # characters of the first token that precede the answer
    bounds = np.cumsum([len(p) + len(SEP) for p in parts])  # part k spans [bounds[k-1], bounds[k]) of the answer, its trailing separator included
    out = [0.0] * len(parts)
    pos = -offset
    for decoded, lp in tokens:  # by the token's last character: " Maryland" starts in the separator but belongs to the next part
        end = pos + len(decoded) - 1
        out[min(int(np.searchsorted(bounds, max(end, 0), side="right")), len(parts) - 1)] += lp
        pos += len(decoded)
    return out


def score(server: str, tag: str, name: str) -> None:
    sets = json.loads((LABELS / f"candidates_{tag}.json").read_text(encoding="utf-8"))
    photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
    by_id = {e["image_id"]: e for e in photos}
    images = MP16Images()
    names = sorted({SEP.join(name_parts(lab)) for p in sets for lab in p["pool"]} - {""})
    with ThreadPoolExecutor(32) as pool:  # text-only, once per distinct name
        text_only = dict(zip(names, pool.map(lambda n: _level_logprobs(server, n.split(SEP), None), names)))
    jobs = [(i, n) for i, p in enumerate(sets) for n in sorted({SEP.join(name_parts(lab)) for lab in p["pool"]} - {""})]
    cache: dict[int, str] = {}

    def run(job: tuple[int, str]) -> list[float] | None:
        i, n = job
        e = by_id[sets[i]["image_id"]]
        if i not in cache:
            cache[i] = _jpeg(Path(e["path"]).read_bytes() if e.get("path") else images.read(e["image_id"]))
        return _level_logprobs(server, n.split(SEP), cache[i])

    with ThreadPoolExecutor(32) as pool:
        with_image = list(pool.map(run, jobs))
    out = [{"image_id": p["image_id"], "image": {}, "text": {}} for p in sets]
    for (i, n), lp in zip(jobs, with_image):
        out[i]["image"][n], out[i]["text"][n] = lp, text_only.get(n)
    path = ROOT / f"name_scores_{tag}_{name}.json"
    path.write_text(json.dumps(out) + "\n", encoding="utf-8")
    failed = sum(lp is None for lp in with_image)
    print(f"{len(jobs)} photo-name scorings ({len(names)} distinct names), {failed} failed -> {path}")


def _features(tag: str, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """[photos, candidates, F] features, [photos, candidates] km to the truth, validity mask, feature names."""

    sets =json.loads((LABELS / f"candidates_{tag}.json").read_text(encoding="utf-8"))
    photos = {e["image_id"]: e for e in json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))}
    scored = {p["image_id"]: p for p in json.loads((ROOT / f"name_scores_{tag}_{name}.json").read_text(encoding="utf-8"))}
    C = max(len(p["pool"]) for p in sets)
    names = ["rank", "log rank", "top-1"] + [f"{kind} {lv}" for kind in ("image", "pmi", "cum pmi") for lv in LEVELS] + ["levels"]
    F, D, V = np.zeros((len(sets), C, len(names)), np.float32), np.full((len(sets), C), 1e5), np.zeros((len(sets), C), bool)
    for i, p in enumerate(sets):
        e, s = photos[p["image_id"]], scored[p["image_id"]]
        truth = np.radians(np.asarray(e["truth"]))
        for k, (lab, c) in enumerate(zip(p["pool"], e["pool"])):
            c = np.radians(np.asarray(c))
            D[i, k] = 6371.0088 * 2 * np.arcsin(np.sqrt(np.sin((c[0] - truth[0]) / 2) ** 2 + np.cos(c[0]) * np.cos(truth[0]) * np.sin((c[1] - truth[1]) / 2) ** 2))
            V[i, k] = True
            F[i, k, :3] = k, np.log1p(k), k == 0
            n = SEP.join(name_parts(lab))
            img, txt = s["image"].get(n), s["text"].get(n)
            if not img or not txt:
                continue
            for lv in range(len(img)):
                F[i, k, 3 + lv] = img[lv]
                F[i, k, 3 + len(LEVELS) + lv] = img[lv] - txt[lv]
                F[i, k, 3 + 2 * len(LEVELS) + lv] = sum(img[: lv + 1]) - sum(txt[: lv + 1])
            F[i, k, -1] = len(img)
    return F, D, V, names


def _fit(F: np.ndarray, D: np.ndarray, V: np.ndarray, seed: int, steps: int = 400):
    import torch

    reward = np.where(V, (D < 25.0) * 1.0 + (D < 1.0) * 0.25 + (D < 200.0) * 0.25, 0.0).astype(np.float32)
    mean, std = F[V].mean(0), F[V].std(0) + 1e-6
    X, M, R = torch.as_tensor((F - mean) / std), torch.as_tensor(V), torch.as_tensor(reward)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(F.shape[-1], 32), torch.nn.GELU(), torch.nn.Linear(32, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(steps):
        loss = -(torch.softmax(net(X).squeeze(-1).masked_fill(~M, float("-inf")), -1) * R).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lambda Fn, Vn: net(torch.as_tensor((Fn - mean) / std)).squeeze(-1).masked_fill(~torch.as_tensor(Vn), float("-inf")).detach().numpy()


def report(tag: str, name: str) -> None:
    from .stage1_eval import _bootstrap

    F, D, V, names = _features(tag, name)
    n = len(D)
    base = D[:, 0]
    print(f"{tag} (n={n}), {name}: reranker top-1 < 1 / 25 / 200 km {(base < 1).mean():.1%} / {(base < 25).mean():.1%} / {(base < 200).mean():.1%}; "
          f"pool oracle {(D.min(1) < 1).mean():.1%} / {(D.min(1) < 25).mean():.1%} / {(D.min(1) < 200).mean():.1%}")
    print("within-photo AUC (candidate < 25 km vs >= 25 km from the truth) and top-1 < 25 km by the score alone:")
    for j, f in enumerate(names[3:-1], start=3):
        wins = pairs = 0.0
        for i in range(n):
            a, b = F[i, V[i] & (D[i] < 25), j], F[i, V[i] & (D[i] >= 25), j]
            if len(a) and len(b):
                wins += (a[:, None] > b[None]).sum() + 0.5 * (a[:, None] == b[None]).sum(); pairs += len(a) * len(b)
        pick = np.argmax(np.where(V, F[..., j], -np.inf), 1)
        print(f"  {f:22s} AUC {wins / max(pairs, 1):.3f}   top-1 < 25 km {(D[np.arange(n), pick] < 25).mean():.1%}")
    folds = np.array_split(np.random.default_rng(0).permutation(n), 5)
    for label, cols in (("rank only", [0, 1, 2]), ("rank + name scores", list(range(len(names))))):
        hits = {km: np.zeros(n) for km in (1, 25, 200)}
        for seed in range(3):
            for f in folds:
                tr = np.setdiff1d(np.arange(n), f)
                s = _fit(F[tr][..., cols], D[tr], V[tr], seed)
                pick = np.argmax(s(F[f][..., cols], V[f]), 1)
                for km in hits:
                    hits[km][f] += (D[f, pick] < km) / 3
        parts = []
        for km in (1, 25, 200):
            c = _bootstrap(hits[km] - (base < km))
            parts.append(f"< {km} km {100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]")
        print(f"CV combiner, {label:20s}: " + "; ".join(parts))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("score", "report"))
    parser.add_argument("--tag", default="dev")
    parser.add_argument("--name", default="base")
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    args = parser.parse_args(argv)
    score(args.server, args.tag, args.name) if args.node == "score" else report(args.tag, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
