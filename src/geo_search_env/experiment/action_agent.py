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

from ..data.benchmarks import COMMONS
from .area_model import VARIANT
from .hypothesis_agent import _net, _random_zoom, _right, _sub, load

CKPT_TAG = "setft_a"
N_ML = 4  # megaloc columns, appended last by load(extra=("megaloc",))
UNITS = {"zoom": 1.0, "read": 2.0, "recognise": 1.0, "street": 1.0}
STREET = "osv" in VARIANT  # HYP_VARIANT=moreosv: a street-search action reveals the OSV-5M areas (source 2 tokens, osv_propose hyps)
MAX_STEPS = 8
ONE_KM = 0.0  # --one-km W (lesson 111): the answer chooses a point per area (the global pool's best near the area, or the zoomed kernel peak)
# and both the answer head's loss and the policy's return add W x [answer < 1 km]; 0 = lesson 94's area-only answer.


def _data(tag: str, stats=None, ckpt: str | None = None):
    import json
    from pathlib import Path

    import torch

    from . import hypothesis_agent

    ckpt = ckpt or CKPT_TAG
    d = load(tag, ckpt, stats=stats, extra=("megaloc",))
    if tag == "dev3k" and ckpt.endswith("own"):  # original protocol: compare with the released Pinpoint under it (lesson 108)
        d["base_km"] = _released_own_dev3k_km()
    assert d["pre"].shape[-1] == d["pre_noclue"].shape[-1]
    # the free glance: the photo's own SigLIP2 embedding (crop_look view 0 = the full image, unit norm), as read_agent found it decides
    # whether reading pays; benchmark photos (no crop file): their cached SigLIP2 embedding, the same full view
    crops = Path(f"artifacts/agent/crops_{tag}_{ckpt}.npy")
    if crops.exists():
        g = np.asarray(np.load(crops, mmap_mode="r")[:, 0], np.float32)
    else:
        from ..data.benchmarks import load_benchmark
        from .strategy_search import BENCHMARK_NAMES

        photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{ckpt}" / "photos.json").read_text(encoding="utf-8"))
        names = (photos[0]["benchmark"],) if photos[0].get("benchmark") in COMMONS else BENCHMARK_NAMES  # the photo set's pools' order ("index")
        g = np.concatenate([load_benchmark(n).embeddings for n in names])[[e["index"] for e in photos]]
        g /= np.linalg.norm(g, axis=1, keepdims=True)
    d["glance"] = torch.as_tensor(g, device="cuda")
    return d


def _released_own_km(tag: str, ckpt: str) -> np.ndarray:
    """The released Pinpoint's top-1 km with the query photographer's photos kept in the index (pinpoint_reference bench-own; lesson 54)."""

    import json
    from pathlib import Path

    from .refine_headroom import _hav

    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{ckpt}" / "photos.json").read_text(encoding="utf-8"))
    bench = photos[0].get("benchmark")
    z = np.load(f"artifacts/pinpoint_reference/{bench if bench in COMMONS else 'bench_own'}.npz")  # Commons: own = excluded
    idx = np.asarray([e["index"] for e in photos])
    top = np.where(z["rows"][idx] >= 0, z["scores"][idx], -np.inf).argmax(1)
    ll = z["latlon"][idx, top]
    truth = np.asarray([e["truth"] for e in photos], np.float64)
    return _hav(truth[:, 0], truth[:, 1], ll[:, 0], ll[:, 1])


def _released_own_dev3k_km() -> np.ndarray:
    """The released Pinpoint's top-1 km on dev3k with only the query photo excluded (lesson 108's extraction, dev3k photo set order)."""

    from .refine_headroom import _hav

    z = np.load("artifacts/pinpoint_set/mp16_32_8_8_dev3k_released_own.npz")
    ok = (z["rows"] >= 0) & (z["features"][..., 6:9] > 0).any(-1)
    top = np.where(ok, z["scores"], -np.inf).argmax(1)
    ll = z["latlon"][np.arange(len(top)), top]
    return _hav(z["truth"][:, 0], z["truth"][:, 1], ll[:, 0], ll[:, 1])


def view(b, read, rec, street=None):
    """The batch as the agent sees it: clue columns / clue-place tokens only if read, MegaLoc columns only if recognised, OSV areas only after
    street search (STREET); one flag column per action."""

    import torch

    pre = b["pre"].clone()
    # every column the read reveals (pre_noclue differs from pre only there: hypothesis_agent.CLUE_COLS, and combined_head's text columns)
    pre[..., :-N_ML] = torch.where(read[:, None, None], pre[..., :-N_ML], b["pre_noclue"][..., :-N_ML])
    pre[..., -N_ML:] = torch.where(rec[:, None, None], pre[..., -N_ML:], torch.zeros_like(pre[..., -N_ML:]))
    flags = torch.stack([read, rec] + ([street] if STREET else []), 1).float()[:, None, :].expand(-1, pre.shape[1], -1)
    v = dict(b)
    v["pre"] = torch.cat([pre, flags], -1)
    v["mask"] = b["mask"] & (b["is_area"] | (read[:, None] & ~b["is_osv"]) | (street[:, None] & b["is_osv"] if STREET else False))
    return v


def fit_answer(d, photos, seed, epochs=25):
    import torch

    torch.manual_seed(seed)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    net = _net(d["pre"].shape[-1] + 2 + STREET, d["post"].shape[-1], d["cand"].shape[-1]).cuda()
    if ONE_KM:
        net.point = torch.nn.Sequential(torch.nn.LayerNorm(64), torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 2)).cuda()
    batches = max(1, len(photos) // 128)
    params = [p for n, p in net.named_parameters() if not n.startswith("action.")]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * batches, pct_start=0.1)
    rng = np.random.default_rng(seed)
    net.train()
    for _ in range(epochs):
        for bi in np.array_split(rng.permutation(photos), batches):
            b = _sub(d, torch.as_tensor(bi, device="cuda"))
            u = torch.rand(len(bi), 2 + STREET, device="cuda", generator=gen) < 0.5
            v = view(b, u[:, 0], u[:, 1], u[:, 2] if STREET else None)
            z = _random_zoom(v, gen)
            if ONE_KM:
                logp = torch.log_softmax(_options(net, v, z), 1)
                okm = lambda r: torch.cat([v["km_unzoom"] < r, (v["km_zoom"] < r) & z], 1) & torch.cat([v["mask"], v["mask"] & z], 1)
                loss = 0.0
                for r, w in ((25.0, 1.0), (1.0, ONE_KM)):
                    ok = okm(r)
                    use = ok.any(1)
                    if use.any():
                        loss = loss - w * torch.logsumexp(logp.masked_fill(~ok, float("-inf")), 1)[use].mean()
                if not torch.is_tensor(loss):
                    continue
            else:
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


def _options(net, v, zoomed):
    """ONE_KM: logits over 2h answers, (area, its unzoomed point) then (area, its zoomed point; zoomed areas only)."""

    import torch

    e = net.encode(v, zoomed)
    a = net.answer(e).squeeze(-1)
    pt = net.point(e)
    return torch.cat([(a + pt[..., 0]).masked_fill(~v["mask"], float("-inf")), (a + pt[..., 1]).masked_fill(~(v["mask"] & zoomed), float("-inf"))], 1)


DETAILS: list | None = None  # evaluate: each answer's probability and its area's centre, appended here (text override test, lesson 121)


def _km(net, v, zoomed):
    import torch

    with torch.no_grad():
        if ONE_KM:
            logits = _options(net, v, zoomed)
            pick = logits.argmax(1)
            km = torch.cat([v["km_unzoom"], v["km_zoom"]], 1).gather(1, pick[:, None]).squeeze(1)
            area = pick % v["mask"].shape[1]
        else:
            logits = net.answer_logits(v, zoomed)
            pick = area = logits.argmax(1)
            km = torch.where(zoomed, v["km_zoom"], v["km_unzoom"]).gather(1, pick[:, None]).squeeze(1)
        if DETAILS is not None:
            DETAILS.append({"conf": logits.softmax(1).max(1).values.cpu().numpy(),
                            "center": v["center"][torch.arange(len(area), device=area.device), area].cpu().numpy()})
    return km


def _policy(dim: int = 64):
    import torch
    import torch.nn as nn

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.zoom = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
            self.glance = nn.Sequential(nn.Linear(1536, dim), nn.GELU())
            n = 3 + STREET  # read, recognise, [street,] stop
            self.glob = nn.Sequential(nn.LayerNorm(2 * dim + n), nn.Linear(2 * dim + n, dim), nn.GELU(), nn.Linear(dim, n))

        def logits(self, net, b, zoomed, read, rec, street=None):
            v = view(b, read, rec, street)
            with torch.no_grad():
                e = net.encode(v, zoomed)
            m = v["mask"].float().unsqueeze(-1)
            pooled = (e * m).sum(1) / m.sum(1).clamp(min=1)
            state = torch.stack([read.float(), rec.float()] + ([street.float()] if STREET else []) + [zoomed.float().sum(1) / 8], 1)
            gl = self.glance(b["glance"])
            z = self.zoom(e + gl[:, None, :]).squeeze(-1).masked_fill(~v["mask"] | zoomed, float("-inf"))
            g = self.glob(torch.cat([pooled, gl, state], 1))
            g = torch.stack([g[:, 0].masked_fill(read, float("-inf")), g[:, 1].masked_fill(rec, float("-inf"))]
                            + ([g[:, 2].masked_fill(street, float("-inf"))] if STREET else []) + [g[:, -1]], 1)
            return torch.cat([z, g], 1), v

    return Policy().cuda()


def rollout(net, pol, b, lam: float, mode: str, gen=None, fixed=None):
    """mode: 'sample' / 'greedy' (policy) or 'fixed' (fixed = (read, recognise, [street,] n top areas to zoom)). Returns km, cost, logp, entropy,
    counts (zooms, read, recognise[, street])."""

    import torch

    n, h = b["mask"].shape
    dev = b["mask"].device
    zoomed = torch.zeros_like(b["mask"])
    read = torch.zeros(n, dtype=torch.bool, device=dev)
    rec = torch.zeros_like(read)
    street = torch.zeros_like(read)
    cost = torch.zeros(n, device=dev)
    logp = torch.zeros(n, device=dev)
    ent = torch.zeros(n, device=dev)
    if mode == "fixed":
        r, c, *st, k = fixed
        read[:], rec[:], street[:] = bool(r), bool(c), bool(st and st[0])
        v = view(b, read, rec, street)
        prio = torch.where(v["is_area"], -v["rank"], -v["rank"] - 10.0).masked_fill(~v["mask"], float("-inf"))
        order = prio.argsort(1, descending=True)
        kk = h if k == "all" else k
        zoomed = torch.zeros_like(v["mask"]).scatter(1, order[:, :kk], True) & v["mask"]
        cost = lam * (UNITS["read"] * read + UNITS["recognise"] * rec + UNITS["street"] * street + UNITS["zoom"] * zoomed.sum(1))
        return _km(net, v, zoomed), cost, logp, ent, _counts(zoomed, read, rec, street)
    done = torch.zeros_like(read)
    for _ in range(MAX_STEPS):
        logits, v = pol.logits(net, b, zoomed, read, rec, street)
        if mode == "sample":
            g = -torch.log(-torch.log(torch.rand(logits.shape, device=dev, generator=gen).clamp(1e-9, 1 - 1e-9)))
            a = (logits + g).argmax(1)
        else:
            a = logits.argmax(1)
        lp = torch.log_softmax(logits, 1)
        live = ~done
        logp = logp + torch.where(live, lp.gather(1, a[:, None]).squeeze(1), 0.0)
        ent = ent + torch.where(live, -(lp.exp() * lp.nan_to_num(0.0, neginf=0.0)).sum(1), 0.0)
        is_zoom, is_read, is_rec, is_stop = a < h, a == h, a == h + 1, a == h + 2 + STREET
        is_street = (a == h + 2) & STREET
        zoomed = zoomed | (torch.nn.functional.one_hot(a.clamp(max=h - 1), h).bool() & (is_zoom & live)[:, None])
        read = read | (is_read & live)
        rec = rec | (is_rec & live)
        street = street | (is_street & live)
        cost = cost + lam * torch.where(live, UNITS["zoom"] * is_zoom + UNITS["read"] * is_read + UNITS["recognise"] * is_rec
                                        + UNITS["street"] * is_street, 0.0)
        done = done | is_stop
        if done.all():
            break
    v = view(b, read, rec, street)
    return _km(net, v, zoomed & v["mask"]), cost, logp, ent, _counts(zoomed, read, rec, street)


def _counts(zoomed, read, rec, street):
    return (zoomed.sum(1).float(), read.float(), rec.float()) + ((street.float(),) if STREET else ())


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
                ret = lambda k: (k < 25).float() + ONE_KM * (k < 1).float()
                adv = (ret(km) - cost) - (ret(km_g) - cost_g)
            loss = -(adv * logp).mean() - beta * ent.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return pol


def evaluate(net, pols: dict, d, photos, base_km) -> list[tuple]:
    import torch

    rows = []
    configs = [(r, c, k) for k in (0, 2, "all") for r in (0, 1) for c in (0, 1)]
    if STREET:
        configs = [(r, c, s, k) for k in (0, 2, "all") for r in (0, 1) for c in (0, 1) for s in (0, 1)]
    with torch.no_grad():
        b = _sub(d, torch.as_tensor(np.asarray(photos), device="cuda"))
        for cfg in configs:
            km, _, _, _, cnt = rollout(net, None, b, 0.0, "fixed", fixed=cfg)
            label = f"fixed: read {cfg[0]}, recognise {cfg[1]}, " + (f"street {cfg[2]}, " if STREET else "") + f"zoom {cfg[-1]}"
            rows.append((label, km.cpu().numpy(), [c.mean().item() for c in cnt]))
        for lam, pol in pols.items():
            globals()["DETAILS"] = []
            km, _, _, _, cnt = rollout(net, pol, b, lam, "greedy")
            last, globals()["DETAILS"] = DETAILS[-1], None
            rows.append((f"RL policy, lambda {lam:g}", km.cpu().numpy(), [c.mean().item() for c in cnt], last))
    return rows


def report(name, rows, base_km, rt_km):
    from .strategy_search import _bootstrap

    print(f"\n {name}: released Pinpoint {np.mean(base_km < 25):.1%} / {np.mean(base_km < 1):.1%}, retrained top-1 {np.mean(rt_km < 25):.1%} / "
          f"{np.mean(rt_km < 1):.1%} (< 25 / 1 km)")
    print(f"   {'':50s} {'<25 km':>7s} {'vs rel.':>8s} {'<1 km':>6s} {'vs rel.':>8s} {'zooms':>6s} {'read':>5s} {'recog':>6s}"
          + (f" {'street':>6s}" if STREET else "") + f" {'cost':>5s}")
    for label, km, (nz, nr, nc, *ns), *_ in rows:
        c = _bootstrap((km < 25).astype(float) - (base_km < 25))
        units = UNITS["zoom"] * nz + UNITS["read"] * nr + UNITS["recognise"] * nc + UNITS["street"] * sum(ns)
        print(f"   {label:50s} {np.mean(km < 25):7.1%} {100 * c[0]:+7.1f} {np.mean(km < 1):6.1%} {100 * (np.mean(km < 1) - np.mean(base_km < 1)):+7.1f} "
              f"{nz:6.2f} {nr:5.0%} {nc:6.0%}" + "".join(f" {x:6.0%}" for x in ns) + f" {units:5.2f}", flush=True)


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


def apply(lambdas: Sequence[float], targets: Sequence[str]) -> None:
    """The answer net and policies trained on all of train (as for dev3k), applied once to each target "tag:ckpt" (test set only when the user
    asks; full:setft_aown = the photographer's photos kept, compared with the released Pinpoint under the same protocol). dev3k is re-run as a check."""

    tr = _data("train")
    sets = {"dev3k": _data("dev3k", stats=tr["stats"])}
    for t in targets:
        tag, ckpt = t.split(":")
        sets[t] = _data(tag, stats=tr["stats"], ckpt=ckpt)
        if ckpt.endswith("own"):
            sets[t]["base_km"] = _released_own_km(tag, ckpt)
    n = tr["n"]
    print(f"action agent: trained on all {n} train photos; lambdas {list(lambdas)}; applied to {list(sets)}", flush=True)
    net = fit_answer(tr, np.arange(n), seed=7)
    pols = {lam: fit_policy(net, tr, np.arange(n), lam, seed=11) for lam in lambdas}
    saved = {}
    for name, d in sets.items():
        groups = [("all", np.arange(d["n"]))]
        groups += [(b, np.flatnonzero(d["benchmark"] == b)) for b in sorted(set(d["benchmark"])) if len(set(d["benchmark"])) > 1]
        for label, photos in groups:
            rows = evaluate(net, pols, d, photos, d["base_km"][photos])
            report(f"{name} / {label} ({len(photos)} photos)", rows, d["base_km"][photos], d["rt_base_km"][photos])
            if label == "all":
                saved |= {f"{name}|{r[0]}": r[1] for r in rows} | {f"{name}|released": d["base_km"], f"{name}|retrained": d["rt_base_km"]}
                saved |= {f"{name}|{r[0]}|{k}": v for r in rows if len(r) > 3 for k, v in r[3].items()}
    np.savez(f"artifacts/agent/action_agent_apply_km{'' if CKPT_TAG == 'setft_a' else '_' + CKPT_TAG}{'_' + VARIANT if VARIANT else ''}{f'_1km{ONE_KM:g}' if ONE_KM else ''}{''.join('_' + t.split(':')[0] for t in targets if not t.startswith('full:'))}.npz", **saved)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lambdas", type=float, nargs="*", default=(0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015))
    parser.add_argument("--apply", nargs="*", help="tag:ckpt photo sets to apply the train-fitted agent to (e.g. full:setft_a full:setft_aown)")
    parser.add_argument("--one-km", type=float, default=0.0, help="weight of [answer < 1 km] in the answer loss and the return (lesson 111)")
    parser.add_argument("--ckpt-tag", default=CKPT_TAG, help="train / dev3k photo sets (setft_own: retrained without photographer exclusion, lesson 108)")
    args = parser.parse_args(argv)
    globals()["CKPT_TAG"] = args.ckpt_tag
    globals()["ONE_KM"] = args.one_km
    if args.apply:
        apply(args.lambdas, args.apply)
    else:
        run(args.lambdas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
