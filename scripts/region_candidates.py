"""Lesson 84: lesson 83's best-of ceiling with regions proposed by retrieval instead of the 4B. Regions = the 16 most frequent (country,
region) labels among the photo's 1,000 nearest index photos (Pinpoint's image embedding, photographer excluded); Pinpoint restricted to each.
Run with PINPOINT_RERANKER_CKPT=artifacts/pinpoint_train/setft_a/checkpoints/ckpt_final.pt PYTHONPATH=src (after nested_passk evaluate)."""
import json
from collections import Counter

import numpy as np
import torch

from geo_search_env.experiment.strategy_search import load_world
from geo_search_env.experiment.text_search import haversine_km
from geo_search_env.tools.pinpoint import Constraint, Pinpoint

rows = json.loads(open("artifacts/agent/nested_passk_dev3k.json").read())
z = np.load("artifacts/agent/nested_passk_dev3k_answers.npz")
tool = Pinpoint()
country, region = tool.labels()
tool.reranker.load_resident()
res = tool.reranker._resident
world = load_world()
mp16_rows = np.asarray([r["row"] for r in rows])
emb = np.asarray(world.mp16["embeddings"][mp16_rows], np.float32)
authors = tool.mp16_author(mp16_rows)
raw, projected = tool.reranker.encode_queries(emb)
index_author = tool.index_author()
N = 1000
props = []
for i in range(len(rows)):
    s = (res["image"] @ projected[i].half()).float()
    s[torch.as_tensor(index_author == authors[i], device=s.device)] = -1e9
    top = torch.topk(s, N).indices.cpu().numpy()
    c = Counter(r for r in region[top] if r)
    props.append([r for r, _ in c.most_common(16)])
keys = [(i, r) for i, p in enumerate(props) for r in p]
print(len(keys), "restricted searches", flush=True)
found = tool.search(emb[[i for i, _ in keys]], [Constraint(region=r) for _, r in keys], authors[[i for i, _ in keys]])
truth = np.asarray([r["truth"] for r in rows])
top1 = np.asarray([r["top1"] for r in rows])
group = np.asarray([r["group"] for r in rows])
w = np.where(group == "fixable", 302 / 3000 / (group == "fixable").sum(), 2698 / 3000 / (group == "control").sum())
base = np.asarray([haversine_km(*truth[i], top1[i:i + 1, 0], top1[i:i + 1, 1])[0] < 25 for i in range(len(rows))])
hit_ret = np.zeros(len(rows), bool)
for j, (i, r) in enumerate(keys):
    ll = found.latlon[j, 0]
    if np.isfinite(ll).all() and haversine_km(*truth[i], ll[:1], ll[1:])[0] < 25:
        hit_ret[i] = True
ll4 = z["ll"]
hit_4b = np.asarray([any(haversine_km(*truth[i], np.nan_to_num(ll4[i, :, 0]), np.nan_to_num(ll4[i, :, 1])) < 25) for i in range(len(rows))])
true_in_ret = np.asarray([z["truth_region"][i] in props[i] for i in range(len(rows))])
true_in_4b = np.asarray([z["truth_region"][i] in set(z["sample_region"][i]) for i in range(len(rows))])
for name, h in (("retrieval regions (16)", hit_ret), ("4B samples (16)", hit_4b), ("union", hit_ret | hit_4b)):
    o = base | h
    print(f"{name}: best-of ceiling dev3k-weighted {np.sum(w * (o.astype(float) - base)) / w.sum():+.2%}; fixable {np.mean(o[group == 'fixable']) - np.mean(base[group == 'fixable']):+.1%}, "
          f"control {np.mean(o[group == 'control']) - np.mean(base[group == 'control']):+.1%}")
wrong = ~base
print(f"true region among proposals where top-1 wrong: retrieval {np.average(true_in_ret[wrong], weights=w[wrong]):.1%}, 4B {np.average(true_in_4b[wrong], weights=w[wrong]):.1%}; "
      f"4B-only fixes {np.sum(w * (hit_4b & ~hit_ret & ~base)) / w.sum():+.2%}, retrieval-only {np.sum(w * (hit_ret & ~hit_4b & ~base)) / w.sum():+.2%}")
