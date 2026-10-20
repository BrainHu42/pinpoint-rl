# Hubness of pool candidates: are generic gallery photos (near neighbours of many photos) the look-alike wrong candidates? For every dev3k /
# train-sample candidate (setft_own pools), r_g = mean SigLIP2 cosine to its 10 nearest gallery photos by other photographers; per photo, does
# a CSLS-style re-score (reranker score - beta * r_g) pick a right (< 25 km) candidate more often than the score? beta chosen on train.
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from geo_search_env.experiment.pinpoint_reference import load_pools  # noqa: E402

K = 10
emb_dir = Path("data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384")
man = json.loads((emb_dir / "manifest.json").read_text())
n_rows = len(np.fromfile(emb_dir / "row_index.i64.bin", dtype=np.int64))
E = np.memmap(emb_dir / "embeddings.f16.bin", dtype=np.float16, mode="r", shape=(n_rows, man["embedding_dim"]))
row_index = np.fromfile(emb_dir / "row_index.i64.bin", dtype=np.int64)
print(f"gallery {n_rows} x {man['embedding_dim']}", flush=True)

# authors per cache row (MP16-Pro CSV order -> cache order)
import csv  # noqa: E402

auth, vocab = [], {}
with open("data/hf/datasets/MP16-Pro/metadata/MP16_Pro_filtered.csv", newline="", encoding="utf-8") as fh:
    rd = csv.reader(fh)
    a_i = next(rd).index("AUTHOR")
    for row in rd:
        auth.append(vocab.setdefault(row[a_i], len(vocab)))
author = np.asarray(auth, np.int64)[row_index]

pools = load_pools("mp16_setft_own")
rng = np.random.default_rng(0)
sets = {}
for tag, n in (("train_setft_own", 6000), ("dev3k_setft_own", None)):
    photos = json.loads(Path(f"artifacts/photo_sets/{tag}/photos.json").read_text())
    if n:
        photos = [photos[i] for i in sorted(rng.choice(len(photos), n, replace=False))]
    sets[tag] = photos
need = np.unique(np.concatenate([pools["cache_rows"][e["index"]][pools["valid"][e["index"]]] for ps in sets.values() for e in ps]))
print(f"{len(need)} distinct candidates", flush=True)

G = torch.empty((n_rows, E.shape[1]), dtype=torch.float16, device="cuda")
for s in range(0, n_rows, 500_000):  # stored embeddings are not unit norm: normalise for cosine
    G[s:s + 500_000] = torch.nn.functional.normalize(torch.as_tensor(np.array(E[s:s + 500_000]), device="cuda").float(), dim=1).half()
print("norm check", G[:1000].float().norm(dim=1).mean().item(), flush=True)
auth_t = torch.as_tensor(author, device="cuda")
r_g = np.zeros(len(need), np.float32)
for s in range(0, len(need), 512):
    q = torch.as_tensor(need[s:s + 512], device="cuda")
    sim = G[q] @ G.T  # [b, n_rows]
    sim[auth_t[q][:, None] == auth_t[None]] = -2  # other photographers only (also drops the photo itself)
    r_g[s:s + 512] = sim.topk(K, dim=1).values.float().mean(1).cpu().numpy()
    if s % 51200 == 0:
        print(f"  {s} / {len(need)}", flush=True)
hub = dict(zip(need.tolist(), r_g.tolist()))


def per_photo(photos):
    out = []
    for e in photos:
        i = e["index"]
        v = pools["valid"][i]
        s = pools["scores"][i][v].astype(np.float64)
        ll = pools["latlon"][i][v].astype(np.float64)
        t = np.radians(np.asarray(e["truth"]))
        la, lo = np.radians(ll[:, 0]), np.radians(ll[:, 1])
        h = np.sin((la - t[0]) / 2) ** 2 + np.cos(t[0]) * np.cos(la) * np.sin((lo - t[1]) / 2) ** 2
        km = 2 * 6371 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        out.append((s, np.asarray([hub[c] for c in pools["cache_rows"][i][v]]), km))
    return out


data = {t: per_photo(ps) for t, ps in sets.items()}
for t, d in data.items():
    allh = np.concatenate([x[1] for x in d])
    right = np.concatenate([x[2] < 25 for x in d])
    top = np.asarray([x[1][np.argmax(x[0])] for x in d])
    topok = np.asarray([x[2][np.argmax(x[0])] < 25 for x in d])
    q = np.quantile(top, [0.2, 0.4, 0.6, 0.8])
    band = np.digitize(top, q)
    print(f"\n{t}: candidates right {right.mean():.1%}; r_g right {allh[right].mean():.3f} vs wrong {allh[~right].mean():.3f}")
    print("  top-1 accuracy by its hubness quintile (low -> high): " + " ".join(f"{100 * topok[band == b].mean():.1f}" for b in range(5)))


def pick_acc(d, beta):
    return np.mean([x[2][np.argmax(x[0] - beta * x[1])] < 25 for x in d])


betas = [0, 0.5, 1, 2, 3, 5, 8, 12]
tr = {b: pick_acc(data["train_setft_own"], b) for b in betas}
best = max(betas, key=lambda b: tr[b])
print("\nre-score s - beta * r_g, top-1 < 25 km: " + " ".join(f"beta {b}: train {100 * tr[b]:.1f} / dev3k {100 * pick_acc(data['dev3k_setft_own'], b):.1f}" for b in betas))
d = data["dev3k_setft_own"]
g = np.asarray([float(x[2][np.argmax(x[0] - best * x[1])] < 25) - float(x[2][np.argmax(x[0])] < 25) for x in d])
idx = rng.integers(0, len(g), (2000, len(g)))
m = g[idx].mean(1)
print(f"train-chosen beta {best}: dev3k {100 * g.mean():+.1f} [{100 * np.percentile(m, 2.5):+.1f}, {100 * np.percentile(m, 97.5):+.1f}]; "
      f"fixes {(g > 0).sum()}, breaks {(g < 0).sum()}")
