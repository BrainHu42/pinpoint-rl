"""RL action policy over the agent's tools (lesson 94; the user: text clues and MegaLoc fit in as costly actions, the policy trained with RL).

Propose / probe / decide on the merged pipeline's area tokens (hypothesis_agent):
  - glance (free): Pinpoint's areas with their pre-zoom features;
  - zoom area k: Pinpoint restricted to the area (post-zoom states, as hypothesis_agent);
  - read: the 4B's clues (clue-place tokens appear, clue columns of every area revealed; read_agent);
  - recognise: MegaLoc same-place similarity of every area (megaloc_gate columns, lesson 93);
  - stop: the answer head picks an area (supervised, trained on random reveal patterns so it answers from any state).
Action policy: per-area zoom logits + read / recognise / stop logits from the frozen answer net's encoding, trained with REINFORCE
(self-critical: the greedy rollout's return is the baseline, small entropy bonus); return = [answer < 25 km] - cost of the actions taken
(units: zoom 1, recognise 1, read 2; times lambda). Reported as an accuracy-vs-cost curve against fixed tool combinations.

  PYTHONPATH=src PINPOINT_RERANKER_CKPT=... python -m geo_search_env.experiment.action_agent [--lambdas 0 0.005 0.01 0.02 0.04]
"""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np

from .hypothesis_agent import CLUE_COLS, _net, _random_zoom, _right, _sub, load

CKPT_TAG = "setft_a"
N_ML = 4  # megaloc columns, appended last by load(extra=("megaloc",))
UNITS = {"zoom": 1.0, "read": 2.0, "recognise": 1.0}
MAX_STEPS = 8


def _data(tag: str, stats=None):
    import torch

    d = load(tag, CKPT_TAG, stats=stats, extra=("megaloc",))
    assert d["pre"].shape[-1] == d["pre_noclue"].shape[-1]
    # the free glance: the photo's own SigLIP2 embedding (crop_look view 0 = the full image, unit norm), as read_agent found it decides
    # whether reading pays
    g = np.load(f"artifacts/agent/crops_{tag}_{CKPT_TAG}.npy", mmap_mode="r")[:, 0]
    d["glance"] = torch.as_tensor(np.asarray(g, np.float32), device="cuda")
    return d


def view(b, read, rec):
    """The batch as the agent sees it: clue columns / clue-place tokens only if read, MegaLoc columns only if recognised; two flag columns."""

    import torch

    pre = b["pre"].clone()
    pre[..., CLUE_COLS] = torch.where(read[:, None, None], pre[..., CLUE_COLS], b["pre_noclue"][..., CLUE_COLS])
    pre[..., -N_ML:] = torch.where(rec[:, None, None], pre[..., -N_ML:], torch.zeros_like(pre[..., -N_ML:]))
    flags = torch.stack([read, rec], 1).float()[:, None, :].expand(-1, pre.shape[1], -1)
    v = dict(b)
    v["pre"] = torch.cat([pre, flags], -1)
    v["mask"] = b["mask"] & (b["is_area"] | read[:, None])
    return v


def fit_answer(d, photos, seed, epochs=25):
    import torch

    torch.manual_seed(seed)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    net = _net(d["pre"].shape[-1] + 2, d["post"].shape[-1], d["cand"].shape[-1]).cuda()
    batches = max(1, len(photos) // 128)
    params = [p for n, p in net.named_parameters() if not n.startswith("action.")]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * batches, pct_start=0.1)
    rng = np.random.default_rng(seed)
    net.train()
    for _ in range(epochs):
        for bi in np.array_split(rng.permutation(photos), batches):
            b = _sub(d, torch.as_tensor(bi, device="cuda"))
            u = torch.rand(len(bi), 2, device="cuda", generator=gen) < 0.5
            v = view(b, u[:, 0], u[:, 1])
            z = _random_zoom(v, gen)
            ok = _right(v, z) & v["mask"]
            use = ok.any(1) & ~(ok == v["mask"]).all(1)
            if not use.any():
                continue
            logp = torch.log_softmax(net.answer_logits(v, z), 1)
            loss = -torch.logsumexp(logp.masked_fill(~ok, float("-inf")), 1)[use].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _km(net, v, zoomed):
    import torch

    with torch.no_grad():
        pick = net.answer_logits(v, zoomed).argmax(1)
    return torch.where(zoomed, v["km_zoom"], v["km_unzoom"]).gather(1, pick[:, None]).squeeze(1)


def _policy(dim: int = 64):
    import torch
    import torch.nn as nn

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.zoom = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
            self.glance = nn.Sequential(nn.Linear(1536, dim), nn.GELU())
            self.glob = nn.Sequential(nn.LayerNorm(2 * dim + 3), nn.Linear(2 * dim + 3, dim), nn.GELU(), nn.Linear(dim, 3))  # read, recognise, stop

        def logits(self, net, b, zoomed, read, rec):
            v = view(b, read, rec)
            with torch.no_grad():
                e = net.encode(v, zoomed)
            m = v["mask"].float().unsqueeze(-1)
            pooled = (e * m).sum(1) / m.sum(1).clamp(min=1)
            state = torch.stack([read.float(), rec.float(), zoomed.float().sum(1) / 8], 1)
            gl = self.glance(b["glance"])
            z = self.zoom(e + gl[:, None, :]).squeeze(-1).masked_fill(~v["mask"] | zoomed, float("-inf"))
            g = self.glob(torch.cat([pooled, gl, state], 1))
            g = torch.stack([g[:, 0].masked_fill(read, float("-inf")), g[:, 1].masked_fill(rec, float("-inf")), g[:, 2]], 1)
            return torch.cat([z, g], 1), v

    return Policy().cuda()


def rollout(net, pol, b, lam: float, mode: str, gen=None, fixed=None):
    """mode: 'sample' / 'greedy' (policy) or 'fixed' (fixed = (read, recognise, n top areas to zoom)). Returns km, cost, logp, entropy, counts."""

    import torch

    n, h = b["mask"].shape
    dev = b["mask"].device
    zoomed = torch.zeros_like(b["mask"])
    read = torch.zeros(n, dtype=torch.bool, device=dev)
    rec = torch.zeros_like(read)
    cost = torch.zeros(n, device=dev)
    logp = torch.zeros(n, device=dev)
    ent = torch.zeros(n, device=dev)
    if mode == "fixed":
        r, c, k = fixed
        read[:], rec[:] = bool(r), bool(c)
        v = view(b, read, rec)
        prio = torch.where(v["is_area"], -v["rank"], -v["rank"] - 10.0).masked_fill(~v["mask"], float("-inf"))
        order = prio.argsort(1, descending=True)
        kk = h if k == "all" else k
        zoomed = torch.zeros_like(v["mask"]).scatter(1, order[:, :kk], True) & v["mask"]
        cost = lam * (UNITS["read"] * read + UNITS["recognise"] * rec + UNITS["zoom"] * zoomed.sum(1))
        return _km(net, v, zoomed), cost, logp, ent, (zoomed.sum(1).float(), read.float(), rec.float())
    done = torch.zeros_like(read)
    for _ in range(MAX_STEPS):
        logits, v = pol.logits(net, b, zoomed, read, rec)
        if mode == "sample":
            g = -torch.log(-torch.log(torch.rand(logits.shape, device=dev, generator=gen).clamp(1e-9, 1 - 1e-9)))
            a = (logits + g).argmax(1)
        else:
            a = logits.argmax(1)
        lp = torch.log_softmax(logits, 1)
        live = ~done
        logp = logp + torch.where(live, lp.gather(1, a[:, None]).squeeze(1), 0.0)
        ent = ent + torch.where(live, -(lp.exp() * lp.nan_to_num(0.0, neginf=0.0)).sum(1), 0.0)
        is_zoom, is_read, is_rec, is_stop = a < h, a == h, a == h + 1, a == h + 2
        zoomed = zoomed | (torch.nn.functional.one_hot(a.clamp(max=h - 1), h).bool() & (is_zoom & live)[:, None])
        read = read | (is_read & live)
        rec = rec | (is_rec & live)
        cost = cost + lam * torch.where(live, UNITS["zoom"] * is_zoom + UNITS["read"] * is_read + UNITS["recognise"] * is_rec, 0.0)
        done = done | is_stop
        if done.all():
            break
    v = view(b, read, rec)
    return _km(net, v, zoomed & v["mask"]), cost, logp, ent, (zoomed.sum(1).float(), read.float(), rec.float())


def fit_policy(net, d, photos, lam, seed, epochs=10, beta=0.01):
    import torch

    torch.manual_seed(seed)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    pol = _policy()
    opt = torch.optim.AdamW(pol.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for bi in np.array_split(rng.permutation(photos), max(1, len(photos) // 256)):
            b = _sub(d, torch.as_tensor(bi, device="cuda"))
            km, cost, logp, ent, _ = rollout(net, pol, b, lam, "sample", gen)
            with torch.no_grad():
                km_g, cost_g, _, _, _ = rollout(net, pol, b, lam, "greedy")
                adv = ((km < 25).float() - cost) - ((km_g < 25).float() - cost_g)
            loss = -(adv * logp).mean() - beta * ent.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return pol


def evaluate(net, pols: dict, d, photos, base_km) -> list[tuple]:
    import torch

    rows = []
    configs = [(r, c, k) for k in (0, 2, "all") for r in (0, 1) for c in (0, 1)]
    with torch.no_grad():
        b = _sub(d, torch.as_tensor(np.asarray(photos), device="cuda"))
        for cfg in configs:
            km, _, _, _, cnt = rollout(net, None, b, 0.0, "fixed", fixed=cfg)
            rows.append((f"fixed: read {cfg[0]}, recognise {cfg[1]}, zoom {cfg[2]}", km.cpu().numpy(), [c.mean().item() for c in cnt]))
        for lam, pol in pols.items():
            km, _, _, _, cnt = rollout(net, pol, b, lam, "greedy")
            rows.append((f"RL policy, lambda {lam:g}", km.cpu().numpy(), [c.mean().item() for c in cnt]))
    return rows


def report(name, rows, base_km, rt_km):
    from .strategy_search import _bootstrap

    print(f"\n {name}: released Pinpoint {np.mean(base_km < 25):.1%}, retrained top-1 {np.mean(rt_km < 25):.1%} (< 25 km)")
    print(f"   {'':40s} {'<25 km':>7s} {'vs rel.':>8s} {'<1 km':>6s} {'zooms':>6s} {'read':>5s} {'recog':>6s} {'cost':>5s}")
    for label, km, (nz, nr, nc) in rows:
        c = _bootstrap((km < 25).astype(float) - (base_km < 25))
        units = UNITS["zoom"] * nz + UNITS["read"] * nr + UNITS["recognise"] * nc
        print(f"   {label:40s} {np.mean(km < 25):7.1%} {100 * c[0]:+7.1f} {np.mean(km < 1):6.1%} {nz:6.2f} {nr:5.0%} {nc:6.0%} {units:5.2f}", flush=True)


def run(lambdas: Sequence[float]) -> None:
    tr = _data("train")
    dv = _data("dev3k", stats=tr["stats"])
    n = tr["n"]
    print(f"action agent: {n} train photos, actions zoom (1) / read (2) / recognise (1) / stop; lambdas {list(lambdas)}", flush=True)
    held = np.random.default_rng(0).permutation(n) % 5 == 0  # one held-out fifth of train as a second estimate
    for name, fit, ev, d_ev in (("train, held-out fifth", np.flatnonzero(~held), np.flatnonzero(held), tr), ("dev3k", np.arange(n), None, dv)):
        net = fit_answer(tr, fit, seed=7)
        pols = {lam: fit_policy(net, tr, fit, lam, seed=11) for lam in lambdas}
        photos = ev if ev is not None else np.arange(d_ev["n"])
        rows = evaluate(net, pols, d_ev, photos, d_ev["base_km"][photos])
        report(name, rows, d_ev["base_km"][photos], d_ev["rt_base_km"][photos])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lambdas", type=float, nargs="*", default=(0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015))
    args = parser.parse_args(argv)
    run(args.lambdas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
