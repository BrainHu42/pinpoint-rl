"""Budgeted hypothesis-testing agent (the merged pipeline, step 2): one network over a photo's area tokens with an answer head and an action head.

Area tokens (Pinpoint areas + clue places; hypothesis_compare features, pinpoint_set zoom states):
  - pre-zoom part, always visible: source, area rank, pool support, gallery density, clue-search statistics, distance to the global top-1,
    the global pool's probability mass in the area (zoom_agent._pre_zoom);
  - post-zoom part, visible once the area is zoomed: attention-pooled CLS states of its restricted search + its own restricted-search
    statistics (zoom_agent.POST_ZOOM; nothing relative to other areas, so unzoomed searches never leak) and a zoomed flag.
Answer head: a transformer layer over the tokens -> which area. A zoomed area answers with its agreement pick (set rule: the global top pick
if it falls inside); an unzoomed area with the best global-pool candidate inside it (its centre if none). Trained supervised with random zoom
subsets, so it answers from any partial state.
Action head: from the current tokens, which unzoomed area to zoom next (sequential: each zoom's result is seen before the next choice).
Trained with REINFORCE (self-critical: the greedy rollout's reward is the baseline) against the frozen answer head; reward = the final answer
< 25 km. Compared at each budget with zooming Pinpoint's top-B areas and with zooming everything.

  python -m geo_search_env.experiment.hypothesis_agent [--ckpt-tag setft_a] [--budgets 0 1 2 3] [--apply-tag full]

--zoom-rule soft: a zoomed area answers with a soft-kernel pick instead of the box agreement pick (score-weighted Laplace density, 10 km,
T 0.1, over its restricted candidates; lesson 72 / 73: keeps 1 km, chosen on train).
--apply-tag: the models trained on all of train applied to another photo set (full = the test set, only when the user asks), per benchmark at
1 / 25 / 200 / 750 / 2500 km; inputs are normalised with train statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .area_model import _paths
from .zoom_agent import POST_ZOOM, _pre_zoom

SOURCES = (0, 1)  # Pinpoint areas and clue places (the 4B's own place proposals add nothing, lesson 70)
BOX_KM = 25.0
CLUE_COLS = slice(12, 20)  # zoom_agent._pre_zoom columns of hypothesis_learn.CLUE_KEYS (clue evidence on each hypothesis)
KERNEL_KM, KERNEL_T, GATE_KM = 10.0, 0.1, 5.0  # soft zoom rule (per-area pick accuracy on train: 4.9% < 1 km vs 4.2% for the box rule, same at 25 km)


def _kernel_km(zm, rows: np.ndarray, truth: np.ndarray, in_group: bool = False, gate: bool = False) -> np.ndarray:
    """km to the truth of each zoomed area's soft-kernel pick: argmax over its restricted candidates of sum_j softmax(s / T)_j exp(-d_ij / KERNEL_KM).
    in_group (hybrid rule): only among the box rule's winning group (the founder with the most softmax(s / 0.2) mass within BOX_KM, and its
    members), so the 25 km decision stays the box rule's and the kernel only places the point.
    gate (gated rule): the kernel peak only if it lies within GATE_KM of the box founder, else the founder (train, top areas: the box rule's
    25 km accuracy, 1.2 of the kernel's 2.1 points at 1 km)."""

    from .pinpoint_set import _haversine_matrix

    out = np.full(len(rows), np.inf)
    for c in range(0, len(rows), 8192):
        r = rows[c:c + 8192]
        valid = zm["rows"][r] >= 0
        s = np.where(valid, zm["scores"][r], -np.inf)
        best = np.where(valid.any(1, keepdims=True), s.max(1, keepdims=True), 0.0)
        w = np.where(valid, np.exp((s - best) / KERNEL_T), 0.0)
        lat = np.where(valid[..., None], zm["latlon"][r], 0.0)
        dens = (w[:, None, :] * np.exp(-_haversine_matrix(lat) / KERNEL_KM)).sum(2)
        allowed = valid
        if in_group or gate:
            pair = _haversine_matrix(lat)
            wb = np.where(valid, np.exp((s - best) / 0.2), 0.0)
            near = (pair < BOX_KM) & valid[:, None, :] & valid[:, :, None]
            founder = np.where(valid, (near * wb[:, None, :]).sum(2), -1).argmax(1)
            allowed = near[np.arange(len(r)), founder]
        pick = np.where(allowed, dens, -np.inf).argmax(1)
        if gate:
            pick = np.where(pair[np.arange(len(r)), pick, founder] <= GATE_KM, pick, founder)
        both = np.stack([truth[zm["photo"][r]], lat[np.arange(len(r)), pick]], 1)
        km = _haversine_matrix(both)[:, 0, 1]
        out[c:c + 8192] = np.where(valid.any(1), km, np.inf)
    return out


def _post(f: dict) -> np.ndarray:
    cols = []
    for k in POST_ZOOM:
        if k not in f:
            continue
        x = f[k].astype(np.float64)
        cols.append(np.log(np.maximum(x, 1e-12)) if k.startswith("mass") or k in ("photographers", "n_cand") else np.where(np.isfinite(x), x, -1.0))
    return np.nan_to_num(np.stack(cols, 1)).astype(np.float32)


def _unzoomed_km(f: dict, photos: list[dict]) -> np.ndarray:
    """Answer km of an area that was not zoomed: the global pool's best-scored candidate within BOX_KM of its centre (the centre if none)."""

    from .strategy_search import _haversine_km

    out = np.zeros(len(f["photo"]))
    for i, (ph, c) in enumerate(zip(f["photo"], f["center"])):
        e = photos[ph]
        pool = np.asarray(e["pool"], np.float64)  # sorted by score
        inside = np.flatnonzero(_haversine_km(c[0], c[1], pool) < BOX_KM)
        point = pool[inside[0]] if len(inside) else np.asarray(c, np.float64)
        out[i] = _haversine_km(*e["truth"], point[None])[0]
    return out


def load(tag: str, ckpt_tag: str, device: str = "cuda", stats: dict | None = None, zoom_rule: str = "gated", extra: tuple[str, ...] = ()) -> dict:
    import torch

    feat_path, zoom_path = _paths(tag, ckpt_tag)
    f = dict(np.load(feat_path))
    zm = np.load(zoom_path)
    assert np.array_equal(zm["photo"], f["photo"]) and np.array_equal(zm["source"], f["source"])
    photo_tag = f"{tag}_{ckpt_tag}" if ckpt_tag else tag
    photos = json.loads((Path("artifacts/photo_sets") / photo_tag / "photos.json").read_text(encoding="utf-8"))
    from .area_model import FEATURES

    released = FEATURES.get(tag, hl_out() / f"hypothesis_features_{tag}_e4_vlmhyps_set.npz")
    released_base = np.load(released)["base_km"] if released.exists() else f["base_km"]  # no released-Pinpoint run (cross-fitting halves)
    keep = np.isin(f["source"], SOURCES)
    if "center" not in f:
        f["center"] = zm["center"]
    rows = np.flatnonzero(keep)
    photo = f["photo"][rows]
    n = len(photos)
    counts = np.bincount(photo, minlength=n)
    h_max = int(counts.max())
    slot = np.zeros(len(rows), int)
    seen = np.zeros(n, int)
    for j, p in enumerate(photo):
        slot[j] = seen[p]
        seen[p] += 1
    pre, post = _pre_zoom(f)[rows], _post(f)[rows]
    pre_noclue = pre.copy()
    pre_noclue[:, CLUE_COLS] = 0.0  # before reading clues (read_agent)
    if extra:  # map checks (map_checks.features): column groups appended to the pre-zoom part; checks need the clue reader, density doesn't
        mc = np.load(Path("artifacts/agent") / f"mapcheck_{tag}_{ckpt_tag}.npz")
        assert np.array_equal(mc["photo"], f["photo"]) and np.array_equal(mc["source"], f["source"])
        cols = np.isin(mc["groups"], extra)
        x = mc["x"][rows][:, cols]
        pre = np.concatenate([pre, x], 1)
        pre_noclue = np.concatenate([pre_noclue, np.where(mc["groups"][cols] != "density", 0.0, x)], 1)
    valid = zm["rows"][rows] >= 0
    s = np.where(valid, zm["scores"][rows], -np.inf)
    best = np.where(np.isfinite(s.max(1, keepdims=True)), s.max(1, keepdims=True), 0.0)
    soft = np.exp(np.where(valid, (s - best) / 0.2, -np.inf))
    soft /= np.maximum(soft.sum(1, keepdims=True), 1e-12)
    cand = np.concatenate([np.stack([np.where(valid, zm["scores"][rows], 0), np.where(valid, s - best, 0), soft], -1), zm["features"][rows].astype(np.float32)], -1)
    un_km = _unzoomed_km({"photo": photo, "center": f["center"][rows]}, photos)
    zoom_km = f["km_agree"][rows]
    if zoom_rule in ("soft", "hybrid", "gated"):
        soft_km = _kernel_km(zm, rows, np.asarray([e["truth"] for e in photos], np.float64), in_group=zoom_rule == "hybrid", gate=zoom_rule == "gated")
        zoom_km = np.where(np.isfinite(soft_km), soft_km, zoom_km)
    set_km = np.where(f["set_top_in"][rows] > 0, f["set_km"][photo], zoom_km)
    cell = cand[valid]
    if stats is None:  # train: its own statistics; other sets reuse them
        stats = {"pre": (pre.mean(0), pre.std(0) + 1e-6), "post": (post.mean(0), post.std(0) + 1e-6), "cand": (cell.mean(0), cell.std(0) + 1e-6)}
    norm = lambda x, key: (x - stats[key][0]) / stats[key][1]
    cand = np.where(valid[..., None], norm(cand, "cand"), 0.0)

    def pad(x, fill=0.0, dtype=np.float32):
        out = np.full((n, h_max) + x.shape[1:], fill, dtype)
        out[photo, slot] = x
        return out

    d = {
        "pre": pad(norm(pre, "pre")), "pre_noclue": pad(norm(pre_noclue, "pre")), "post": pad(norm(post, "post")), "hidden": pad(zm["hidden"][rows], dtype=np.float16), "cand": pad(cand),
        "cvalid": pad(valid, False, bool), "mask": pad(np.ones(len(rows), bool), False, bool),
        "km_zoom": pad(zoom_km, np.inf), "km_zoom_set": pad(set_km, np.inf), "km_unzoom": pad(un_km, np.inf),
        "is_area": pad(f["source"][rows] == 0, False, bool), "rank": pad(f["rank"][rows].astype(np.float32), 99.0),
    }
    d["cvalid"][..., 0] |= d["mask"]  # an empty search keeps one zero slot so pooling is defined
    t = {k: torch.as_tensor(v, device=device) for k, v in d.items()}
    t["n"] = n
    t["base_km"] = released_base
    t["rt_base_km"] = f["base_km"]
    t["stats"] = stats
    t["benchmark"] = np.asarray([e.get("benchmark", tag) for e in photos])
    return t


def _net(n_pre: int, n_post: int, n_c: int, dim: int = 64):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.h = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, dim))
            self.c = nn.Sequential(nn.Linear(n_c, dim), nn.GELU(), nn.Linear(dim, dim))
            self.query = nn.Parameter(torch.zeros(dim))
            self.pre = nn.Sequential(nn.Linear(n_pre, dim), nn.GELU(), nn.Linear(dim, dim))
            self.post = nn.Sequential(nn.Linear(n_post + dim + 1, dim), nn.GELU(), nn.Linear(dim, dim))
            self.layers = nn.ModuleList(nn.TransformerEncoderLayer(dim, 4, 4 * dim, 0.1, "gelu", batch_first=True, norm_first=True) for _ in range(2))
            self.answer = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
            self.action = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
            self.register_buffer("pre_mask", torch.ones(n_pre), persistent=False)  # pre-zoom columns this model sees (fit_answers variants)

        def forward(self, b, zoomed):
            return self.answer_logits(b, zoomed)

        def encode(self, b, zoomed):
            t = self.h(b["hidden"].float()) + self.c(b["cand"])
            att = (t @ self.query / t.shape[-1] ** 0.5).masked_fill(~b["cvalid"], -1e4)
            pooled = (torch.softmax(att, -1).unsqueeze(-1) * t).sum(-2)
            z = zoomed.float().unsqueeze(-1)
            e = self.pre(b["pre"] * self.pre_mask) + self.post(torch.cat([b["post"] * z, pooled * z, z], -1))
            for layer in self.layers:
                e = layer(e, src_key_padding_mask=~b["mask"])
            return e

        def answer_logits(self, b, zoomed):
            return self.answer(self.encode(b, zoomed)).squeeze(-1).masked_fill(~b["mask"], float("-inf"))

        def action_logits(self, b, zoomed):
            e = self.encode(b, zoomed)
            return self.action(e.detach()).squeeze(-1).masked_fill(~b["mask"] | zoomed, float("-inf"))

    torch.backends.mha.set_fastpath_enabled(False)
    return Net()


def _sub(d, idx):
    return {k: v[idx] for k, v in d.items() if hasattr(v, "shape") and k not in ("base_km", "rt_base_km", "benchmark")}


def hl_out() -> Path:
    from .hypothesis_learn import OUT as out

    return out


def _right(b, zoomed, rule="agree", r=25.0):
    import torch

    zkm = b["km_zoom_set"] if rule == "set" else b["km_zoom"]
    return torch.where(zoomed, zkm, b["km_unzoom"]) < r


def _random_zoom(b, gen):
    import torch

    # per photo: zoom everything (1/3), nothing (1/6), or a random subset (1/2)
    u = torch.rand(b["mask"].shape[0], 1, device=b["mask"].device, generator=gen)
    p = torch.rand(b["mask"].shape, device=b["mask"].device, generator=gen) < torch.rand_like(u)
    z = torch.where(u < 1 / 3, torch.ones_like(p), torch.where(u < 0.5, torch.zeros_like(p), p))
    return z & b["mask"]


def _view(b, read):
    """The batch as seen before / after reading clues (read_agent, lesson 76): unread photos see no clue-place tokens and zeroed clue features."""

    import torch

    if "pre_noclue" not in b:
        return b
    v = dict(b)
    v["pre"] = torch.where(read[:, None, None], b["pre"], b["pre_noclue"])
    v["mask"] = b["mask"] & (b["is_area"] | read[:, None])
    return v


def fit_answer(d, photos, seed, epochs=25, read_mode: str | None = None):
    import torch

    torch.manual_seed(seed)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    net = _net(d["pre"].shape[-1], d["post"].shape[-1], d["cand"].shape[-1]).cuda()
    batches = max(1, len(photos) // 128)
    params = [p for n, p in net.named_parameters() if not n.startswith("action.")]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * batches, pct_start=0.1)
    rng = np.random.default_rng(seed)
    net.train()
    for _ in range(epochs):
        for bi in np.array_split(rng.permutation(photos), batches):
            b = _sub(d, torch.as_tensor(bi, device="cuda"))
            if read_mode == "random":  # answer from either state: clues read for half the photos
                b = _view(b, torch.rand(len(bi), device="cuda", generator=gen) < 0.5)
            z = _random_zoom(b, gen)
            ok = _right(b, z) & b["mask"]
            use = ok.any(1) & ~(ok == b["mask"]).all(1)
            if not use.any():
                continue
            logp = torch.log_softmax(net.answer_logits(b, z), 1)
            loss = -torch.logsumexp(logp.masked_fill(~ok, float("-inf")), 1)[use].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    return net


def fit_answers(d, photos, seeds, pre_masks, epochs=25):
    """Several answer heads trained at once as one batched model (torch.func.vmap over stacked weights), so the GPU isn't idle on one small
    head: model k starts from seed seeds[k] and sees the pre-zoom columns where pre_masks[k] is 1 (the others are zeroed, a constant input).
    Same recipe as fit_answer; all models see the same batches and random zooms. Returns one ordinary net per model (pre_mask set)."""

    import copy

    import torch
    from torch.func import functional_call, vmap

    nets = []
    for seed, m in zip(seeds, pre_masks):
        torch.manual_seed(seed)
        net = _net(d["pre"].shape[-1], d["post"].shape[-1], d["cand"].shape[-1]).cuda()
        net.pre_mask.copy_(torch.as_tensor(np.asarray(m, np.float32)))
        nets.append(net)
    # stacked by hand: each _net call defines its own class, which stack_module_state rejects
    params = {n: torch.stack([dict(m.named_parameters())[n].detach() for m in nets]).requires_grad_() for n, _ in nets[0].named_parameters()}
    buffers = {n: torch.stack([dict(m.named_buffers())[n] for m in nets]) for n, _ in nets[0].named_buffers()}
    base = copy.deepcopy(nets[0]).to("meta")
    gen = torch.Generator(device="cuda").manual_seed(int(seeds[0]))
    batches = max(1, len(photos) // 128)
    train = [v for k, v in params.items() if not k.startswith("action.")]
    opt = torch.optim.AdamW(train, lr=1e-3, weight_decay=1e-2)  # elementwise: the same as one optimiser per model
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * batches, pct_start=0.1)
    rng = np.random.default_rng(int(seeds[0]))
    base.train()

    def one(p, buf, b, z):
        return functional_call(base, (p, buf), (b, z))

    batched = vmap(one, in_dims=(0, 0, None, None), randomness="different")
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):  # fused kernels reject the vmapped mask strides
        for _ in range(epochs):
            for bi in np.array_split(rng.permutation(photos), batches):
                b = _sub(d, torch.as_tensor(bi, device="cuda"))
                z = _random_zoom(b, gen)
                ok = _right(b, z) & b["mask"]
                use = ok.any(1) & ~(ok == b["mask"]).all(1)
                if not use.any():
                    continue
                logp = torch.log_softmax(batched(params, buffers, b, z), -1)  # [models, photos, areas]
                loss = -torch.logsumexp(logp.masked_fill(~ok, float("-inf")), -1)[:, use].mean(1).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
    out = []
    for k, net in enumerate(nets):
        with torch.no_grad():
            for name, prm in net.named_parameters():
                prm.copy_(params[name][k])
        net.eval()
        out.append(net)
    return out


def _rollout(net, b, budget, policy: str, gen=None):
    """Zoom `budget` areas one at a time. policy: 'top' (Pinpoint's best-ranked areas first, then clue places), 'greedy' or 'sample' (action
    head). Returns (zoomed mask, summed log-probabilities of the sampled actions)."""

    import torch

    zoomed = torch.zeros_like(b["mask"])
    logp = torch.zeros(b["mask"].shape[0], device=b["mask"].device)
    for _ in range(budget):
        avail = b["mask"] & ~zoomed
        if not avail.any():
            break
        if policy == "top":
            prio = torch.where(b["is_area"], -b["rank"], -b["rank"] - 10.0).masked_fill(~avail, float("-inf"))
            choice = prio.argmax(1)
        else:
            logits = net.action_logits(b, zoomed)
            if policy == "sample":
                g = -torch.log(-torch.log(torch.rand(logits.shape, device=logits.device, generator=gen).clamp(1e-9, 1 - 1e-9)))
                choice = (logits + g).argmax(1)
                logp = logp + torch.where(avail.any(1), torch.log_softmax(logits, 1).gather(1, choice[:, None]).squeeze(1).nan_to_num(0.0), 0.0)
            else:
                choice = logits.argmax(1)
        has = avail.any(1)
        zoomed = zoomed | (torch.nn.functional.one_hot(choice, zoomed.shape[1]).bool() & has[:, None] & avail)
    return zoomed, logp


def _answer(net, b, zoomed, rule="agree"):
    import torch

    with torch.no_grad():
        pick = net.answer_logits(b, zoomed).argmax(1)
    zkm = b["km_zoom_set"] if rule == "set" else b["km_zoom"]
    km = torch.where(zoomed, zkm, b["km_unzoom"]).gather(1, pick[:, None]).squeeze(1)
    return km


def fit_policy(net, d, photos, budget, seed, epochs=8):
    import torch

    gen = torch.Generator(device="cuda").manual_seed(seed)
    for p in net.parameters():
        p.requires_grad_(False)
    for p in net.action.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(net.action.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for bi in np.array_split(rng.permutation(photos), max(1, len(photos) // 256)):
            b = _sub(d, torch.as_tensor(bi, device="cuda"))
            z_s, logp = _rollout(net, b, budget, "sample", gen)
            with torch.no_grad():
                z_g, _ = _rollout(net, b, budget, "greedy")
                adv = (_answer(net, b, z_s) < 25).float() - (_answer(net, b, z_g) < 25).float()
            loss = -(adv * logp).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    for p in net.parameters():
        p.requires_grad_(True)
    return {k: v.clone() for k, v in net.action.state_dict().items()}


def evaluate(net, d, photos, budgets, action_states=None):
    """km per photo for each (budget, policy, rule)."""

    import torch

    out = {}
    with torch.no_grad():
        for bi in np.array_split(np.asarray(photos), max(1, len(photos) // 1024)):
            b = _sub(d, torch.as_tensor(bi, device="cuda"))
            for budget in budgets:
                policies = ["top"] + (["agent"] if action_states and budget in action_states and 0 < budget else [])
                for pol in policies:
                    if pol == "agent":
                        net.action.load_state_dict(action_states[budget])
                    z, _ = _rollout(net, b, budget, "top" if pol == "top" else "greedy")
                    for rule in ("agree", "set"):
                        out.setdefault((budget, pol, rule), []).append(_answer(net, b, z, rule).cpu().numpy())
            z = b["mask"]
            for rule in ("agree", "set"):
                out.setdefault(("all", "all", rule), []).append(_answer(net, b, z, rule).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def apply(ckpt_tag: str, budgets: Sequence[int], apply_tag: str, zoom_rule: str = "gated") -> None:
    """All-of-train answer and action heads on another photo set, per benchmark."""

    tr = load("train", ckpt_tag, zoom_rule=zoom_rule)
    te = load(apply_tag, ckpt_tag, stats=tr["stats"], zoom_rule=zoom_rule)
    n = tr["n"]
    print(f"agent ({ckpt_tag or 'released Pinpoint'}) trained on all of train, applied to {apply_tag} ({te['n']} photos)", flush=True)
    net = fit_answer(tr, np.arange(n), seed=100)
    states = {B: fit_policy(net, tr, np.arange(n), B, seed=100 + B) for B in budgets if B > 0}
    res = evaluate(net, te, np.arange(te["n"]), budgets, states)
    answers = {"Pinpoint (released) top-1": te["base_km"], f"retrained reranker top-1": te["rt_base_km"]}
    for key in [(0, "top", "agree")] + [(B, "agent", r) for B in budgets if B > 0 for r in ("agree", "set")] + [("all", "all", "agree"), ("all", "all", "set")]:
        if key in res:
            answers[f"budget {key[0]}, {key[1]}, {key[2]} pick"] = res[key]
    thresholds = (1, 25, 200, 750, 2500)
    for bench in sorted(set(te["benchmark"])) + ["all"]:
        sel = np.ones(te["n"], bool) if bench == "all" else te["benchmark"] == bench
        print(f"\n {bench} ({sel.sum()} photos): % within " + " / ".join(f"{t} km" for t in thresholds))
        for name, km in answers.items():
            d25 = 100 * (np.mean(km[sel] < 25) - np.mean(te["base_km"][sel] < 25))
            d1 = 100 * (np.mean(km[sel] < 1) - np.mean(te["base_km"][sel] < 1))
            print(f"   {name:34s} " + "  ".join(f"{100 * np.mean(km[sel] < t):5.1f}" for t in thresholds) + f"   ({d25:+.1f} at 25 km, {d1:+.1f} at 1 km)")
    if apply_tag != "dev3k":
        from .strategy_search import _bootstrap

        km = answers.get("budget 2, agent, agree pick", answers.get(("budget all, all, agree pick")))
        c = _bootstrap((km < 25).astype(float) - (te["base_km"] < 25))
        print(f"\n budget-2 agent vs released Pinpoint at 25 km, all photos: {100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]")
    np.savez(hl_out() / f"agent_{apply_tag}_{ckpt_tag or 'released'}{'' if zoom_rule == 'gated' else '_' + zoom_rule}_answers.npz",
             **{k.replace(" ", "_").replace(",", "").replace("(", "").replace(")", ""): v for k, v in answers.items()}, benchmark=te["benchmark"])


def run(ckpt_tag: str, budgets: Sequence[int], ensemble: int = 1, zoom_rule: str = "gated") -> None:
    from .strategy_search import _bootstrap

    tr = load("train", ckpt_tag, zoom_rule=zoom_rule)
    dv = load("dev3k", ckpt_tag, stats=tr["stats"], zoom_rule=zoom_rule)
    n = tr["n"]
    print(f"agent ({ckpt_tag or 'released Pinpoint'}, zoom rule {zoom_rule}): {n} train photos, up to {tr['mask'].shape[1]} areas, "
          f"{tr['mask'].sum().item() / n:.2f} per photo", flush=True)
    folds = np.random.default_rng(0).permutation(n) % 5
    cv: dict = {}
    order = []
    for k in range(5):
        fit, held = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        net = fit_answer(tr, fit, seed=10 * k)
        states = {B: fit_policy(net, tr, fit, B, seed=10 * k + B) for B in budgets if B > 0}
        res = evaluate(net, tr, held, budgets, states)
        for key, v in res.items():
            cv.setdefault(key, []).append(v)
        order.append(held)
    order = np.concatenate(order)
    cv = {key: np.concatenate(v)[np.argsort(order)] for key, v in cv.items()}
    net = fit_answer(tr, np.arange(n), seed=100)
    states = {B: fit_policy(net, tr, np.arange(n), B, seed=100 + B) for B in budgets if B > 0}
    dev = evaluate(net, dv, np.arange(dv["n"]), budgets, states)
    for name, res, d in (("train CV", cv, tr), ("dev3k", dev, dv)):
        base = d["base_km"]
        print(f"\n {name}: released Pinpoint {np.mean(base < 25):.1%} / {np.mean(base < 1):.1%}; "
              f"retrained top-1 {np.mean(d['rt_base_km'] < 25):.1%} / {np.mean(d['rt_base_km'] < 1):.1%}  (25 / 1 km)")
        for key in sorted(res, key=lambda k: (str(k[0]) if k[0] == "all" else f"{k[0]:03d}", k[1], k[2])):
            km = res[key]
            c, c1 = _bootstrap((km < 25).astype(float) - (base < 25)), _bootstrap((km < 1).astype(float) - (base < 1))
            label = f"budget {key[0]}, {key[1]}, {key[2]} pick"
            print(f"   {label:34s} < 25 km {np.mean(km < 25):.1%} ({100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]); "
                  f"< 1 km {np.mean(km < 1):.1%} ({100 * c1[0]:+.1f})", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-tag", default="", help="photo sets on a retrained reranker (pinpoint_set photo-set)")
    parser.add_argument("--budgets", type=int, nargs="*", default=(0, 1, 2, 3))
    parser.add_argument("--zoom-rule", choices=("agree", "soft", "hybrid", "gated"), default="gated",
                        help="a zoomed area's answer: box agreement, soft kernel, kernel inside the box group, or kernel gated to 5 km of the box founder (default; lesson 74)")
    parser.add_argument("--apply-tag", help="apply the all-of-train models to this photo set (full = test set, only when the user asks)")
    args = parser.parse_args(argv)
    if args.apply_tag:
        apply(args.ckpt_tag, args.budgets, args.apply_tag, args.zoom_rule)
    else:
        run(args.ckpt_tag, args.budgets, zoom_rule=args.zoom_rule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
