# Does a bigger, longer-trained region head rank the true region higher? Region classifiers on frozen MP16 SigLIP2 embeddings (GPU, ~13 GB).
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.region_head_big   then   region_prior --head <name>

"""Region = MP16-Pro (state, country), as in strategy_search.region_head. Training data per config: "sft-out" = MP16 minus photos by any benchmark or
MP16-query (artifacts/sft/queries.json) photographer (1.5M, what the dev pipeline's head used); "eval-out" = MP16 minus only benchmark, dev and selection
photographers (3.75M); "+osv" adds OSV_SAMPLE OSV-5M street photos (grid region labels). Model selection on 2,000 held-out MP16 query photos that are
not in the dev set (bucket 99, train split). Wider heads trained 10-20 epochs overfit (selection top-1 41.5 -> 39%), so they get 3 epochs here.
Writes each head's top-50 regions for every benchmark query and every MP16 query in the region_head.npz layout (key prefix head_mlp_), so
region_prior can swap it in.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np

from .coarse_filter import UNKNOWN
from .query_evidence import ROOT, SFT_ROOT
from .strategy_search import load_world

OUT = ROOT / "region"
SELECT = 2_000
BATCH = 8_192
SCALE = 30.0
OSV_SAMPLE = 2_000_000

BASE = dict(hidden=(2048,), epochs=2, lr=2e-3, dropout=0.0, smoothing=0.0)  # the existing head's architecture and schedule
WIDE = dict(hidden=(4096, 4096), epochs=3, lr=1e-3, dropout=0.1, smoothing=0.1)
CONFIGS: dict[str, dict[str, Any]] = {
    "base": dict(BASE, data="sft-out"),
    "base-eval-out": dict(BASE, data="eval-out"),
    "wide2-eval-out": dict(WIDE, data="eval-out"),
    "base-eval-out-osv": dict(BASE, data="eval-out+osv"),
    "wide2-eval-out-osv": dict(WIDE, data="eval-out+osv"),
}


def _model(dim: int, n: int, hidden: tuple[int, ...], dropout: float):
    import torch

    layers: list[Any] = []
    for h in hidden:
        layers += [torch.nn.Linear(dim, h), torch.nn.LayerNorm(h) if len(hidden) > 1 else torch.nn.Identity(), torch.nn.GELU(), torch.nn.Dropout(dropout)]
        dim = h
    return torch.nn.Sequential(*layers, torch.nn.Linear(dim, n))


def _ranks(logits, truth):
    import torch

    t = torch.as_tensor(truth, device=logits.device)
    known = t != UNKNOWN
    true_logit = logits.gather(1, t.clamp(min=0)[:, None])
    rank = (logits > true_logit).sum(1) + 1
    return rank[known].cpu().numpy()


def main() -> None:
    import torch
    import torch.nn.functional as F

    world = load_world()
    sft = json.loads((SFT_ROOT / "queries.json").read_text(encoding="utf-8"))
    sft_rows = np.asarray([q["row"] for q in sft], dtype=np.int64)
    dev_index = {e["index"] for e in json.loads((ROOT / "dev" / "dev.json").read_text(encoding="utf-8"))}
    candidates = [i for i, q in enumerate(sft) if q["group"] == "held_out" and q["split"] == "train" and i not in dev_index]
    select = np.sort(np.random.default_rng(0).choice(candidates, SELECT, replace=False))

    dev_rows = np.asarray([e["row"] for e in json.loads((ROOT / "dev" / "dev.json").read_text(encoding="utf-8"))], dtype=np.int64)
    eval_authors = np.union1d(world.query_author[world.query_author >= 0], world.mp16["author"][np.concatenate((dev_rows, sft_rows[select]))])
    mp16_rows = np.flatnonzero((world.mp16["region"] != UNKNOWN) & ~np.isin(world.mp16["author"], eval_authors))
    sft_out = ~np.isin(world.mp16["author"][mp16_rows], world.mp16["author"][sft_rows])
    osv_labels = world.osv["region"]
    osv_rows = np.sort(np.random.default_rng(0).choice(np.flatnonzero(osv_labels != UNKNOWN), OSV_SAMPLE, replace=False))
    subsets = {
        "sft-out": np.flatnonzero(sft_out),
        "eval-out": np.arange(len(mp16_rows)),
        "eval-out+osv": np.arange(len(mp16_rows) + len(osv_rows)),
    }
    n_regions = len(world.vocab["region"])
    print(", ".join(f"{k}: {len(v)} training photos" for k, v in subsets.items()) + f"; {n_regions} regions", flush=True)

    device = torch.device("cuda")
    started = time.time()
    x = torch.empty((len(mp16_rows) + len(osv_rows), world.mp16["embeddings"].shape[1]), dtype=torch.float16, device=device)
    offset = 0
    for array, rows in ((world.mp16["embeddings"], mp16_rows), (world.osv["embeddings"], osv_rows)):
        for s in range(0, len(rows), 262_144):
            chunk = rows[s : s + 262_144]
            x[offset : offset + len(chunk)] = torch.as_tensor(np.asarray(array[chunk]), device=device)
            offset += len(chunk)
    y = torch.as_tensor(np.concatenate((world.mp16["region"][mp16_rows], osv_labels[osv_rows])), device=device)
    print(f"loaded embeddings to GPU in {time.time() - started:.0f}s", flush=True)

    def embed(array):
        return F.normalize(torch.as_tensor(np.asarray(array, dtype=np.float32), device=device), dim=-1) * SCALE

    sft_x = embed(world.mp16["embeddings"][np.sort(sft_rows)])[np.argsort(np.argsort(sft_rows))]  # memmap reads want sorted rows
    sft_truth = world.mp16["region"][sft_rows]
    bench_x = embed(world.query_embeddings)
    report: dict[str, Any] = {}
    for name, cfg in CONFIGS.items():
        torch.manual_seed(0)
        model = _model(x.shape[1], n_regions, cfg["hidden"], cfg["dropout"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        subset = torch.as_tensor(subsets[cfg["data"]], device=device)
        steps = cfg["epochs"] * math.ceil(len(subset) / BATCH)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["lr"], total_steps=steps)
        started = time.time()
        for epoch in range(cfg["epochs"]):
            model.train()
            order = subset[torch.randperm(len(subset), device=device)]
            losses = []
            for i in range(0, len(order), BATCH):
                take = order[i : i + BATCH]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(F.normalize(x[take].float(), dim=-1) * SCALE), y[take], label_smoothing=cfg["smoothing"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.detach())
            model.eval()
            with torch.inference_mode():
                r = _ranks(model(sft_x[select]).float(), sft_truth[select])
            print(f"{name} epoch {epoch + 1}/{cfg['epochs']} loss {torch.stack(losses).mean().item():.3f} select top1/5/10 "
                  f"{(r <= 1).mean():.1%} / {(r <= 5).mean():.1%} / {(r <= 10).mean():.1%} ({time.time() - started:.0f}s)", flush=True)
        with torch.inference_mode():
            for split, queries, out in (("sft", sft_x, f"head_{name}_sft.npz"), ("bench", bench_x, f"head_{name}_bench.npz")):
                top = torch.topk(torch.softmax(torch.cat([model(queries[i : i + 4096]).float() for i in range(0, len(queries), 4096)]), -1), 50, dim=-1)
                np.savez_compressed(OUT / out, head_mlp_regions=top.indices.cpu().numpy(), head_mlp_probs=top.values.cpu().numpy())
        report[name] = {"select top1/5/10": [float((r <= k).mean()) for k in (1, 5, 10)], "config": {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()}}
        (OUT / "heads.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        del model, optimizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
