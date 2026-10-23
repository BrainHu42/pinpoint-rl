"""One value network over action_agent's actions and answers (VALUE_PLAN step 1): Q(state, action) for zoom k / read / recognise and for
"commit to area k with its unzoomed / zoomed point", replacing the supervised answer head + REINFORCE policy.

  - state: action_agent.view (clue columns / clue-place tokens only if read, MegaLoc (+ EXTRA_FILES) columns only if recognised) + zoomed set;
  - commit values: per token and point option, P(< 25 km) and P(< 1 km) by logistic regression on the cached outcomes; value = P25 + W x P1
    (W = --one-km, the return of lesson 111);
  - information values (conditioned on lambda): fitted Q iteration, target = -lambda x units(a) + V(s'), s' read from the cached reveals;
    V(s') = the target network's value of the online network's best action at s' (double Q; commit included). Never the realised outcome of
    a choice made with hindsight;
  - states: random partial reveal sets (as the answer head's training) plus the network's own epsilon-greedy rollouts;
  - policy: argmax Q, stop when committing beats every further action. The accuracy-vs-cost curve comes from one network (lambda is an input).
Compared on dev3k and commons26 dev with action_agent's answer head + REINFORCE at the same seeds, and the same fixed recipes.

  PYTHONPATH=src python -m geo_search_env.experiment.value_agent [--seeds 7 8 9] [--extra pe_g siglip_final pe_probe]
"""

from __future__ import annotations

import argparse
import copy
from typing import Sequence

import numpy as np

from . import action_agent as aa
from .hypothesis_agent import _net, _random_zoom, _sub

LAMBDAS = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015)
GROUPS: list[tuple[str, float]] = [("recognise", 1.0)]  # reveal actions over the last action_agent.N_ML columns: (name, units) per block;
# one group = action_agent's recognise (MegaLoc + EXTRA_FILES together); --split: one action per model, 4 columns each in load order
# (MegaLoc first, then --extra), VALUE_PLAN step 2
LAM_SCALE = 100.0  # lambda as a network input: lambda x 100 (0 .. 1.5)


def _qnet(d, dim: int = 64):
    import torch
    import torch.nn as nn

    G = len(GROUPS)
    net = _net(d["pre"].shape[-1] + 1 + G, d["post"].shape[-1], d["cand"].shape[-1]).cuda()
    del net.answer, net.action
    net.commit = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 4)).cuda()  # (unzoomed, zoomed) x (25, 1 km) logits
    net.glance = nn.Sequential(nn.Linear(1536, dim), nn.GELU()).cuda()
    net.zoom = nn.Sequential(nn.LayerNorm(dim + 1), nn.Linear(dim + 1, dim), nn.GELU(), nn.Linear(dim, 1)).cuda()
    n_g = 1 + G  # read, one reveal per group
    net.glob = nn.Sequential(nn.LayerNorm(2 * dim + n_g + 2), nn.Linear(2 * dim + n_g + 2, dim), nn.GELU(), nn.Linear(dim, n_g)).cuda()
    return net


def view(b, read, rec):
    """action_agent.view with one reveal flag per group: rec (n, G) bool; group g's columns are zero until revealed."""

    import torch

    from .hypothesis_agent import CLUE_COLS

    pre = b["pre"].clone()
    pre[..., CLUE_COLS] = torch.where(read[:, None, None], pre[..., CLUE_COLS], b["pre_noclue"][..., CLUE_COLS])
    G, F = len(GROUPS), pre.shape[-1]
    width = aa.N_ML // G
    for g in range(G):
        lo = F - aa.N_ML + g * width
        pre[..., lo:lo + width] = torch.where(rec[:, g, None, None], pre[..., lo:lo + width], 0.0)
    flags = torch.cat([read[:, None], rec], 1).float()[:, None, :].expand(-1, pre.shape[1], -1)
    v = dict(b)
    v["pre"] = torch.cat([pre, flags], -1)
    v["mask"] = b["mask"] & (b["is_area"] | read[:, None])
    return v


def _pooled(net, b):
    """The attention-pooled restricted-search states per token (hypothesis_agent._net.encode); independent of the reveal state."""

    import torch

    t = net.h(b["hidden"].float()) + net.c(b["cand"])
    att = (t @ net.query / t.shape[-1] ** 0.5).masked_fill(~b["cvalid"], -1e4)
    return (torch.softmax(att, -1).unsqueeze(-1) * t).sum(-2)


GAIN = False  # --gain: information values as the best commit value now (detached) + a learned gain (advantage form; costs are ~0.01)
DELTA = False  # --delta: the zoomed point's logits are the unzoomed point's plus a learned difference
MC = False  # --mc: information targets = the realised gain of acting (greedy continuation's return from s' minus the greedy commit's reward
# at s, added to the commit value now); no max over estimates in the targets (the overestimation of the plain fitted Q iteration)
LISTWISE = 0.0  # --listwise W: + W x the old answer head's listwise loss on log commit values (ranking of options within a photo)


def _q(net, b, pooled, zoomed, read, rec, lam, detach_info: bool = True):
    """Q values at a state: (commit (n, 2h) over (token, unzoomed point) then (token, zoomed point), P25 / P1 logits (n, h, 4), information
    actions (n, h + 2): zoom token k, read, recognise; -inf where not allowed)."""

    import torch

    v = view(b, read, rec)
    z = zoomed.float().unsqueeze(-1)
    e = net.pre(v["pre"] * net.pre_mask) + net.post(torch.cat([v["post"] * z, pooled * z, z], -1))
    for layer in net.layers:
        e = layer(e, src_key_padding_mask=~v["mask"])
    logits = net.commit(e)
    if DELTA:  # zoomed point = unzoomed point + a learned difference (where the 1 km deficit sits: unzoomed vs zoomed point of an area)
        logits = torch.cat([logits[..., :2], logits[..., :2] + logits[..., 2:]], -1)
    p = torch.sigmoid(logits)
    w = aa.ONE_KM
    commit = torch.cat([(p[..., 0] + w * p[..., 1]).masked_fill(~v["mask"], float("-inf")),
                        (p[..., 2] + w * p[..., 3]).masked_fill(~(v["mask"] & zoomed), float("-inf"))], 1)
    ei = e.detach() if detach_info else e
    gl = net.glance(b["glance"])
    lf = (lam * LAM_SCALE)[:, None]
    zq = net.zoom(torch.cat([ei + gl[:, None, :], lf[:, None, :].expand(-1, ei.shape[1], -1)], -1)).squeeze(-1)
    zq = zq.masked_fill(~v["mask"] | zoomed, float("-inf"))  # (GAIN: -inf + finite stays -inf)
    m = v["mask"].float().unsqueeze(-1)
    pooled_e = (ei * m).sum(1) / m.sum(1).clamp(min=1)
    state = torch.cat([read.float()[:, None], rec.float(), zoomed.float().sum(1, keepdim=True) / 8], 1)
    g = net.glob(torch.cat([pooled_e, gl, state, lf], 1))
    info = torch.cat([zq, g], 1)
    if GAIN:
        info = info + commit.max(1, keepdim=True).values.detach()
    h = zq.shape[1]
    info = torch.cat([info[:, :h], info[:, h:h + 1].masked_fill(read[:, None], float("-inf")), info[:, h + 1:].masked_fill(rec, float("-inf"))], 1)
    return commit, logits, info, v


def _step(zoomed, read, rec, a, h, live):
    """Apply information action a (< h zoom, h read, h + 1 + g reveal group g) where live."""

    import torch

    zoomed = zoomed | (torch.nn.functional.one_hot(a.clamp(max=h - 1), h).bool() & ((a < h) & live)[:, None])
    G = rec.shape[1]
    rec = rec | (torch.nn.functional.one_hot((a - h - 1).clamp(0, G - 1), G).bool() & ((a > h) & live)[:, None])
    return zoomed, read | ((a == h) & live), rec


def _units(a, h):
    import torch

    gu = torch.as_tensor([u for _, u in GROUPS], device=a.device)
    return torch.where(a < h, aa.UNITS["zoom"], torch.where(a == h, aa.UNITS["read"], gu[(a - h - 1).clamp(0, len(GROUPS) - 1)]))


def _random_state(b, gen):
    import torch

    n = b["mask"].shape[0]
    u = torch.rand(n, 1 + len(GROUPS), device="cuda", generator=gen) < 0.5
    v = view(b, u[:, 0], u[:, 1:])
    return _random_zoom(v, gen), u[:, 0], u[:, 1:]


def _policy_state(net, b, pooled, lam, gen, eps=0.2):
    """A state on the network's own trajectories: epsilon-greedy for t ~ U{0..5} information steps (a commit stops early)."""

    import torch

    n, h = b["mask"].shape
    zoomed = torch.zeros_like(b["mask"])
    read = torch.zeros(n, dtype=torch.bool, device="cuda")
    rec = torch.zeros(n, len(GROUPS), dtype=torch.bool, device="cuda")
    t = torch.randint(0, 6, (n,), device="cuda", generator=gen)
    live = torch.ones_like(read)
    with torch.no_grad():
        for s in range(5):
            live = live & (t > s)
            if not live.any():
                break
            commit, _, info, _ = _q(net, b, pooled, zoomed, read, rec, lam)
            best = info.argmax(1)
            stop = commit.max(1).values >= info.max(1).values
            rnd = torch.rand(info.shape, device="cuda", generator=gen).masked_fill(~torch.isfinite(info), -1.0).argmax(1)
            explore = torch.rand(n, device="cuda", generator=gen) < eps
            ok = torch.isfinite(info).any(1)
            a = torch.where(explore, rnd, best)
            live = live & ok & (explore | ~stop)
            zoomed, read, rec = _step(zoomed, read, rec, a, h, live)
    return zoomed & view(b, read, rec)["mask"], read, rec


def _value(net, tgt, b, pooled_on, pooled_tg, zoomed, read, rec, lam):
    """V(s) with the online network's best action valued by the target network."""

    import torch

    c_on, _, i_on, _ = _q(net, b, pooled_on, zoomed, read, rec, lam)
    c_tg, _, i_tg, _ = _q(tgt, b, pooled_tg, zoomed, read, rec, lam)
    commit_best = c_on.max(1).values >= i_on.max(1).values
    vc = c_tg.gather(1, c_on.argmax(1, keepdim=True)).squeeze(1)
    vi = i_tg.gather(1, i_on.argmax(1, keepdim=True)).squeeze(1)
    return torch.where(commit_best, vc, vi)


def _greedy_return(net, b, pooled, zoomed, read, rec, lam, steps):
    """Realised return of the online network's greedy continuation from a state: [< 25 km] + W x [< 1 km] of its commit minus lambda x the
    units it spends on the way (at most `steps` more information actions). The choices never see the truth."""

    import torch

    h = b["mask"].shape[1]
    cost = torch.zeros_like(lam)
    live = torch.ones_like(read)
    for _ in range(steps):
        commit, _, info, _ = _q(net, b, pooled, zoomed, read, rec, lam)
        live = live & (info.max(1).values > commit.max(1).values)
        if not live.any():
            break
        a = info.argmax(1)
        cost = cost + torch.where(live, lam * _units(a, h), 0.0)
        zoomed, read, rec = _step(zoomed, read, rec, a, h, live)
    commit, _, _, v = _q(net, b, pooled, zoomed & view(b, read, rec)["mask"], read, rec, lam)
    km = _km(v, commit)
    return (km < 25).float() + aa.ONE_KM * (km < 1).float() - cost


def _expand(b, k):
    return {key: x.repeat_interleave(k, 0) for key, x in b.items()}


def fit_q(d, photos, seed, epochs=30, batch=128, tau=0.01, w_info=1.0, detach_info=True, on_policy_from=5):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    net = _qnet(d)
    tgt = copy.deepcopy(net)
    for p in tgt.parameters():
        p.requires_grad_(False)
    batches = max(1, len(photos) // batch)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * batches, pct_start=0.1)
    lams = torch.as_tensor(LAMBDAS, device="cuda")
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        tot_c = tot_i = 0.0
        for bi in np.array_split(rng.permutation(photos), batches):
            b = _sub(d, torch.as_tensor(bi, device="cuda"))
            n, h = b["mask"].shape
            lam = lams[torch.randint(0, len(lams), (n,), device="cuda", generator=gen)]
            net.train()
            zoomed, read, rec = _random_state(b, gen)
            if ep >= on_policy_from:  # half the batch on the network's own trajectories
                net.eval()
                with torch.no_grad():
                    zp, rp, cp = _policy_state(net, b, _pooled(net, b), lam, gen)
                net.train()
                half = torch.rand(n, device="cuda", generator=gen) < 0.5
                zoomed, read, rec = torch.where(half[:, None], zp, zoomed), torch.where(half, rp, read), torch.where(half[:, None], cp, rec)
            pooled = _pooled(net, b)
            commit, logits, info, v = _q(net, b, pooled, zoomed, read, rec, lam, detach_info)
            # commit: every allowed option's outcome
            y = torch.stack([v["km_unzoom"] < 25, v["km_unzoom"] < 1, v["km_zoom"] < 25, v["km_zoom"] < 1], -1).float()
            okm = torch.stack([v["mask"]] * 2 + [v["mask"] & zoomed] * 2, -1)
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            wts = torch.as_tensor([1.0, aa.ONE_KM or 1.0, 1.0, aa.ONE_KM or 1.0], device="cuda")
            loss_c = (bce * okm * wts).sum() / okm.sum().clamp(min=1)
            if LISTWISE:  # action_agent.fit_answer's loss with log commit values as the option logits
                logp = torch.log_softmax(torch.log(commit.clamp(min=1e-6)).masked_fill(~torch.isfinite(commit), float("-inf")), 1)
                for r, wr in ((25.0, 1.0), (1.0, aa.ONE_KM)):
                    ok = torch.cat([v["km_unzoom"] < r, (v["km_zoom"] < r) & zoomed], 1) & torch.isfinite(commit)
                    use = ok.any(1)
                    if use.any() and wr:
                        loss_c = loss_c - LISTWISE * wr * torch.logsumexp(logp.masked_fill(~ok, float("-inf")), 1)[use].mean()
            # information: every allowed action's target from its next state
            allowed = torch.isfinite(info)
            A = h + 1 + len(GROUPS)
            with torch.no_grad():
                net.eval()
                be = _expand(b, A)
                acts = torch.arange(A, device="cuda").repeat(n)
                z2, r2, c2 = _step(zoomed.repeat_interleave(A, 0), read.repeat_interleave(A, 0), rec.repeat_interleave(A, 0), acts, h,
                                   allowed.reshape(-1))
                l2 = lam.repeat_interleave(A)
                z2 = z2 & view(be, r2, c2)["mask"]
                if MC:  # realised gain of acting: greedy continuation from s' minus the greedy commit at s (paired, so mostly 0)
                    r0 = _greedy_return(net, b, _pooled(net, b), zoomed, read, rec, lam, 0)
                    ret = _greedy_return(net, be, _pooled(net, be), z2, r2, c2, l2, aa.MAX_STEPS - 1)
                    target = (ret - l2 * _units(acts, h)).reshape(n, A) - r0[:, None] + commit.max(1, keepdim=True).values.detach()
                else:
                    v2 = _value(net, tgt, be, _pooled(net, be), _pooled(tgt, be), z2, r2, c2, l2)
                    target = (v2 - l2 * _units(acts, h)).reshape(n, A)
                net.train()
            loss_i = ((info - target).masked_fill(~allowed, 0.0) ** 2).sum() / allowed.sum().clamp(min=1)
            loss = loss_c + w_info * loss_i
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                for p, q in zip(net.parameters(), tgt.parameters()):
                    q.lerp_(p, tau)
            tot_c += loss_c.item()
            tot_i += loss_i.item()
        print(f"    q seed {seed} epoch {ep}: commit {tot_c / batches:.4f}, info {tot_i / batches:.5f}", flush=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _km(v, commit):
    pick = commit.argmax(1)
    return __import__("torch").cat([v["km_unzoom"], v["km_zoom"]], 1).gather(1, pick[:, None]).squeeze(1)


def rollout_q(net, b, lam: float, fixed=None, gate=None, open_loop: bool = False):
    """Answer by the commit values after one of:
      - the greedy policy: argmax Q from the empty state, MAX_STEPS information actions at most (as action_agent);
      - fixed = (read, groups, k): read or not, reveal groups (0 none, 1 all, or a tuple of group indices), zoom the k top-ranked areas;
      - gate = (fixed, g, t): the fixed recipe, then reveal group g only where the top two commit values are within t (a simple gate);
      - open_loop: the actions worth more than committing at the empty state, all chosen there before any result (top MAX_STEPS by value).
    Returns km and counts (zooms, read, then one per group)."""

    import torch

    n, h = b["mask"].shape
    G = len(GROUPS)
    zoomed = torch.zeros_like(b["mask"])
    read = torch.zeros(n, dtype=torch.bool, device="cuda")
    rec = torch.zeros(n, G, dtype=torch.bool, device="cuda")
    lamt = torch.full((n,), lam, device="cuda")
    with torch.no_grad():
        pooled = _pooled(net, b)
        if fixed is not None or gate is not None:
            r, c, k = fixed if fixed is not None else gate[0]
            read[:] = bool(r)
            for g in (range(G) if c == 1 else () if c == 0 else c):
                rec[:, g] = True
            v = view(b, read, rec)
            prio = torch.where(v["is_area"], -v["rank"], -v["rank"] - 10.0).masked_fill(~v["mask"], float("-inf"))
            order = prio.argsort(1, descending=True)
            zoomed = torch.zeros_like(v["mask"]).scatter(1, order[:, : (h if k == "all" else k)], True) & v["mask"]
            if gate is not None:
                commit, _, _, _ = _q(net, b, pooled, zoomed, read, rec, lamt)
                top2 = commit.topk(2, 1).values
                rec[:, gate[1]] |= (top2[:, 0] - top2[:, 1]).nan_to_num(0.0, posinf=1e9) < gate[2]
        elif open_loop:
            commit, _, info, _ = _q(net, b, pooled, zoomed, read, rec, lamt)
            val, order = info.topk(min(aa.MAX_STEPS, info.shape[1]), 1)
            for j in range(order.shape[1]):
                zoomed, read, rec = _step(zoomed, read, rec, order[:, j], h, val[:, j] > commit.max(1).values)
        else:
            live = torch.ones_like(read)
            for _ in range(aa.MAX_STEPS):
                commit, _, info, _ = _q(net, b, pooled, zoomed, read, rec, lamt)
                live = live & (info.max(1).values > commit.max(1).values)
                if not live.any():
                    break
                zoomed, read, rec = _step(zoomed, read, rec, info.argmax(1), h, live)
        zoomed = zoomed & view(b, read, rec)["mask"]
        commit, _, _, v = _q(net, b, pooled, zoomed, read, rec, lamt)
        km = _km(v, commit)
    return km, (zoomed.sum(1).float(), read.float()) + tuple(rec[:, g].float() for g in range(G))


def _units_of(cnt) -> float:
    return aa.UNITS["zoom"] * cnt[0] + aa.UNITS["read"] * cnt[1] + sum(u * c for (_, u), c in zip(GROUPS, cnt[2:]))


def _configs():
    G = len(GROUPS)
    groups = (0, 1) if G == 1 else (0,) + tuple((g,) for g in range(G)) + (1,)
    return [(r, c, k) for k in (0, 2, "all") for r in (0, 1) for c in groups]


def _gname(c) -> str:
    return str(c) if c in (0, 1) else "+".join(GROUPS[g][0] for g in c)


GATE_T = (0.02, 0.05, 0.1, 0.2)  # top-two commit-value margins below which the gated group is revealed


def evaluate(qnet, rl, d, photos) -> dict:
    """km per row: fixed recipes, gates (G > 1), and the policy and its open-loop control per lambda, for the value network ('Q'), and the
    answer head + REINFORCE ('RL', one group only)."""

    import torch

    rows = {}
    G = len(GROUPS)
    with torch.no_grad():
        b = _sub(d, torch.as_tensor(np.asarray(photos), device="cuda"))
        for cfg in _configs():
            label = f"fixed: read {cfg[0]}, recognise {_gname(cfg[1])}, zoom {cfg[2]}"
            km, cnt = rollout_q(qnet, b, 0.0, fixed=cfg)
            rows["Q", label] = (km.cpu().numpy(), [c.mean().item() for c in cnt])
            if rl is not None:
                km, _, _, _, cnt = aa.rollout(rl[0], None, b, 0.0, "fixed", fixed=cfg)
                rows["RL", label] = (km.cpu().numpy(), [c.mean().item() for c in cnt])
        if G > 1:  # every group but the costliest revealed (read, all zooms), the costliest only where the answer is close
            g = int(np.argmax([u for _, u in GROUPS]))
            rest = tuple(i for i in range(G) if i != g)
            for k in (0, "all"):
                for t in GATE_T:
                    km, cnt = rollout_q(qnet, b, 0.0, gate=((1, rest, k), g, t))
                    rows["Q", f"gate: {GROUPS[g][0]} if margin < {t:g}, zoom {k}"] = (km.cpu().numpy(), [c.mean().item() for c in cnt])
        for lam in LAMBDAS:
            km, cnt = rollout_q(qnet, b, lam)
            rows["Q", f"policy, lambda {lam:g}"] = (km.cpu().numpy(), [c.mean().item() for c in cnt])
            km, cnt = rollout_q(qnet, b, lam, open_loop=True)
            rows["Q", f"open-loop, lambda {lam:g}"] = (km.cpu().numpy(), [c.mean().item() for c in cnt])
            if rl is not None:
                km, _, _, _, cnt = aa.rollout(rl[0], rl[1][lam], b, lam, "greedy")
                rows["RL", f"policy, lambda {lam:g}"] = (km.cpu().numpy(), [c.mean().item() for c in cnt])
    return rows


def report(name, runs: list[dict], base_km):
    """Seed-averaged rows; Q vs RL paired over photos (seed-mean per photo), bootstrap SE."""

    from .strategy_search import _bootstrap

    w = aa.ONE_KM
    gn = [g[:6] for g, _ in GROUPS]
    print(f"\n {name} ({len(base_km)} photos): released Pinpoint {np.mean(base_km < 25):.2%} / {np.mean(base_km < 1):.2%}; {len(runs)} seeds; "
          f"units: zoom {aa.UNITS['zoom']:g}, read {aa.UNITS['read']:g}, " + ", ".join(f"{g} {u:g}" for g, u in GROUPS))
    print(f"   {'':52s} {'<25 km':>7s} {'<1 km':>6s} {'units':>6s} {'zooms':>6s} {'read':>5s} " + " ".join(f"{g:>6s}" for g in gn)
          + f"   {'Q - RL <25':>11s} {'(SE)':>6s} {'<1':>6s}")
    for label in dict.fromkeys(k[1] for k in runs[0]):
        for who in ("RL", "Q"):
            if (who, label) not in runs[0]:
                continue
            km = np.stack([r[who, label][0] for r in runs])
            cnt = np.mean([r[who, label][1] for r in runs], 0)
            line = f"   {who + ' ' + label:52s} {np.mean(km < 25):7.2%} {np.mean(km < 1):6.2%} {_units_of(cnt):6.2f} {cnt[0]:6.2f} {cnt[1]:5.0%} " \
                   + " ".join(f"{c:6.0%}" for c in cnt[2:])
            if who == "Q" and ("RL", label) in runs[0]:
                rl = np.stack([r["RL", label][0] for r in runs])
                dd = (km < 25).mean(0) - (rl < 25).mean(0)
                c = _bootstrap(dd)
                line += f"   {100 * dd.mean():+11.2f} {100 * (c[2] - c[1]) / 3.92:6.2f} {100 * ((km < 1).mean() - (rl < 1).mean()):+6.2f}"
            print(line, flush=True)
    print(f"   return - lambda x units (reward = [<25] + {w:g} x [<1]):")
    for lam in LAMBDAS:
        out = []
        for who, kind in (("RL", "policy"), ("Q", "policy"), ("Q", "open-loop")):
            key = (who, f"{kind}, lambda {lam:g}")
            if key not in runs[0]:
                continue
            km = np.stack([r[key][0] for r in runs])
            cnt = np.mean([r[key][1] for r in runs], 0)
            out.append(f"{who} {kind} {np.mean((km < 25) + w * (km < 1)) - lam * _units_of(cnt):.4f}")
        print(f"     lambda {lam:<7g} " + "   ".join(out), flush=True)


def run(seeds: Sequence[int], extra: Sequence[str], targets: Sequence[str], baseline: bool = True, **kw) -> None:
    from . import hypothesis_agent as ha

    ha.EXTRA_FILES = [f"artifacts/value/{m}_cols_{{tag}}_{{ckpt}}.npz" for m in extra]
    aa.N_ML = 4 + 4 * len(extra)
    assert len(GROUPS) in (1, 1 + len(extra)), GROUPS
    baseline = baseline and len(GROUPS) == 1  # REINFORCE knows one recognise action
    tr = aa._data("train")
    sets = {"dev3k": aa._data("dev3k", stats=tr["stats"])}
    for t in targets:
        tag, ckpt = t.split(":")
        sets[t] = aa._data(tag, stats=tr["stats"], ckpt=ckpt)
        sets[t]["base_km"] = aa._released_own_km(tag, ckpt)
    print(f"value agent: {tr['n']} train photos, ckpt {aa.CKPT_TAG}, one-km {aa.ONE_KM:g}, extra {list(extra)}, groups {GROUPS}, seeds {list(seeds)}, "
          f"lambdas {LAMBDAS}, {kw}", flush=True)
    runs = {name: [] for name in sets}
    for seed in seeds:
        q = fit_q(tr, np.arange(tr["n"]), seed=seed, **kw)
        rl = None
        if baseline:
            net = aa.fit_answer(tr, np.arange(tr["n"]), seed=seed)
            rl = (net, {lam: aa.fit_policy(net, tr, np.arange(tr["n"]), lam, seed=seed + 4) for lam in LAMBDAS})
        for name, d in sets.items():
            runs[name].append(evaluate(q, rl, d, np.arange(d["n"])))
    for name, d in sets.items():
        report(name, runs[name], d["base_km"])
    out = f"artifacts/value/value_agent_{aa.CKPT_TAG}{'_' + '+'.join(extra) if extra else ''}{'_gain' if GAIN else ''}{'_mc' if MC else ''}{'_delta' if DELTA else ''}{f'_lw{LISTWISE:g}' if LISTWISE else ''}{'_split' if len(GROUPS) > 1 else ''}.npz"
    np.savez(out, **{f"{name}|{s}|{who}|{label}": r[who, label][0] for name, rs in runs.items() for s, r in zip(seeds, rs) for (who, label) in r})
    print(f"saved {out}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="*", default=[7, 8, 9])
    parser.add_argument("--extra", nargs="*", default=[], help="V5 column files in artifacts/value (e.g. pe_g siglip_final pe_probe), revealed by recognise")
    parser.add_argument("--targets", nargs="*", default=["commons26dev:setft_own"])
    parser.add_argument("--ckpt-tag", default="setft_own")
    parser.add_argument("--one-km", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--w-info", type=float, default=1.0)
    parser.add_argument("--shared", action="store_true", help="information losses also train the encoder (default: detached, as the old action head)")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--gain", action="store_true", help="information values in advantage form (best commit value now + gain)")
    parser.add_argument("--split", nargs="*", type=float, default=None,
                        help="one recognise action per model with these units, MegaLoc first then --extra (e.g. 1 9 0.05 0.17; VALUE_PLAN step 2)")
    parser.add_argument("--lambdas", type=float, nargs="*", default=list(LAMBDAS), help="costs per unit to train on and report")
    parser.add_argument("--delta", action="store_true", help="zoomed-point logits = unzoomed-point logits + a learned difference")
    parser.add_argument("--mc", action="store_true", help="information targets from realised greedy continuations (paired with the commit now)")
    parser.add_argument("--listwise", type=float, default=0.0, help="weight of the listwise ranking loss on commit values")
    args = parser.parse_args(argv)
    aa.CKPT_TAG, aa.ONE_KM = args.ckpt_tag, args.one_km
    globals()["GAIN"], globals()["LISTWISE"], globals()["MC"] = args.gain, args.listwise, args.mc
    globals()["DELTA"], globals()["LAMBDAS"] = args.delta, tuple(args.lambdas)
    if args.split is not None:
        globals()["GROUPS"] = list(zip(["megaloc"] + list(args.extra), args.split))
    run(args.seeds, args.extra, args.targets, baseline=not args.no_baseline, epochs=args.epochs, w_info=args.w_info, detach_info=not args.shared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
