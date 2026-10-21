# GLDv2 pilot (lesson R10) sanity check: does the matcher find a GLDv2 photo's own landmark in the mini-gallery (the photo itself left out)?
# Usage: [GLD_EMB=megaloc|siglip] PYTHONPATH=src .venv/bin/python scripts/gldv2_selfcheck.py
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import gldv2_pilot as p  # noqa: E402

keepj = json.loads((p.OUT / "keep.json").read_text())
z = np.load(p.OUT / f"{p._emb()}.npz")
nq = int(z["n_queries"])
ok = z["ok"][nq:]
lm_of = np.asarray([keepj["images"][i] for i in z["gallery_ids"]])
G = torch.nn.functional.normalize(torch.as_tensor(z["gallery"].astype(np.float16)).cuda().float(), dim=-1).half()
cnt = np.bincount(np.unique(lm_of, return_inverse=True)[1])[np.unique(lm_of, return_inverse=True)[1]]
cand = np.flatnonzero(ok & (cnt >= 2) & np.isin(lm_of, keepj["local"]))
q = np.random.default_rng(0).choice(cand, 2000, replace=False)
sims = (G[torch.as_tensor(q).cuda()] @ G.T).float()
sims[torch.arange(len(q)), torch.as_tensor(q).cuda()] = -2  # leave the photo out
top = sims.topk(10, 1).indices.cpu().numpy()
same = lm_of[top] == lm_of[q][:, None]
print(f"{p._emb()}: GLDv2 photos of local landmarks as queries (2,000, landmark has >= 2 photos): own landmark top-1 photo {100 * same[:, 0].mean():.1f}%, "
      f"in the top 10 photos {100 * same.any(1).mean():.1f}%")
