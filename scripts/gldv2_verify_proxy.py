# Dense Commons imagery as a verifier, without the bot limit: GLDv2 (2.71M Commons landmark photos, MegaLoc index of lesson R10) stands in for
# "every Commons photo near the place". Lesson 138's test: the 116 commons26 dev photos with a right text place and a wrong pipeline answer
# (artifacts/tool_use/targets_commons26dev.json: right text place, the text ranker's wrong pick, the pipeline's answer, and the truth as a
# control). Per place and radius: the best MegaLoc match among GLDv2 photos within the radius (hits > 0.7 dropped as near-duplicates).
# Pre-declared go (lesson 138): the right place beats the wrong pick on well over half of the 116.
# Usage: PYTHONPATH=src .venv/bin/python scripts/gldv2_verify_proxy.py
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
import gldv2_head as gh  # noqa: E402
import gldv2_pilot as p  # noqa: E402

RADII = (1.0, 10.0, 25.0)
ROLES = ("right", "wrong pick", "pipeline", "truth")
CUT = 0.7


def main() -> None:
    tg = json.loads(Path("artifacts/tool_use/targets_commons26dev.json").read_text())
    photos, *_ = p._photos()
    assert all(photos[t["photo"]]["image_id"] == t["image_id"] for t in tg)
    Q = torch.as_tensor(np.load(p.OUT / "megaloc.npz")["queries"][[t["photo"] for t in tg]], dtype=torch.float32, device="cuda")
    places = np.asarray([[[pl["lat"], pl["lon"]] for pl in t["places"]] + [t["truth"]] for t in tg], np.float64)  # (116, 4, 2)
    assert [pl["role"] for pl in tg[0]["places"]] == list(ROLES[:3])
    P = torch.as_tensor(np.radians(places), device="cuda")
    tab = gh._lm_table()
    best = torch.full((len(tg), 4, len(RADII)), -2.0, device="cuda")
    count = torch.zeros((len(tg), 4, len(RADII)), device="cuda")
    for sp in sorted((p.OUT / "index").glob("shard_*.npz")):
        z = np.load(sp)
        ll = tab[np.minimum(z["lm"], len(tab) - 1)]
        ok = np.isfinite(ll[:, 0]) & (z["lm"] < len(tab))
        E = torch.as_tensor(z["emb"][ok], device="cuda").float()
        G = torch.as_tensor(np.radians(ll[ok]), device="cuda")
        S = Q @ E.T  # (116, n)
        S = torch.where(S > CUT, torch.full_like(S, -2.0), S)
        for r in range(4):
            a = P[:, r]
            h = (torch.sin((G[None, :, 0] - a[:, None, 0]) / 2) ** 2
                 + torch.cos(a[:, None, 0]) * torch.cos(G[None, :, 0]) * torch.sin((G[None, :, 1] - a[:, None, 1]) / 2) ** 2)
            km = 2 * 6371.0 * torch.arcsin(torch.sqrt(h.clamp(0, 1)))
            for k, rad in enumerate(RADII):
                m = km < rad
                best[:, r, k] = torch.maximum(best[:, r, k], torch.where(m, S, torch.full_like(S, -2.0)).max(1).values)
                count[:, r, k] += m.sum(1)
        print(f"{sp.name}", flush=True)
    best, count = best.cpu().numpy(), count.cpu().numpy()
    lines = [f"GLDv2 as dense Commons imagery: lesson 138's {len(tg)} photos (right text place vs the text ranker's wrong pick; MegaLoc, hits > {CUT} dropped)"]
    for k, rad in enumerate(RADII):
        cov = (count[:, :, k] > 0).mean(0)
        lines.append(f"  within {rad:g} km: covered " + ", ".join(f"{r} {c:.0%}" for r, c in zip(ROLES, cov))
                     + "; median best " + ", ".join(f"{np.median(best[count[:, i, k] > 0, i, k]) if (count[:, i, k] > 0).any() else float('nan'):.3f}"
                                                   for i in range(4)))
        for ref, name in ((0, "right"), (3, "truth")):
            b = best[:, ref, k] > best[:, 1, k]
            tie = best[:, ref, k] == best[:, 1, k]
            both = (count[:, ref, k] > 0) & (count[:, 1, k] > 0)
            lines.append(f"    {name} beats the wrong pick: {b.mean():.0%} of all (ties {tie.mean():.0%}, both uncovered), "
                         f"{b[both].mean() if both.any() else float('nan'):.0%} of the {both.sum()} where both are covered")
    text = "\n".join(lines)
    print(text, flush=True)
    (p.OUT / "verify_proxy_commons26dev.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
