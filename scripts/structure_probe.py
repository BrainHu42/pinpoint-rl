# Structural simplifications of the answer head (no new inputs), dev3k, setft_own, read + recognise + zoom all (the RL policy ties it, lesson 94):
#   training state: mixed (random reveals / zooms, as action_agent.fit_answer) vs full (the state the head is used in);
#   answer: joint 25 km + 1 km options (--one-km 0.5, lesson 111) vs factorized (the area from a 25 km-only head, the point from the 1 km head);
#   seed ensembles (log-softmax averaged).
# Usage: PYTHONPATH=src .venv/bin/python scripts/structure_probe.py [--seeds 3]
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
from geo_search_env.experiment import action_agent as aa  # noqa: E402
from geo_search_env.experiment.hypothesis_agent import _net, _random_zoom, _sub  # noqa: E402


def fit(d, photos, seed, one_km, full, epochs=25):
    torch.manual_seed(seed)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    net = _net(d["pre"].shape[-1] + 2, d["post"].shape[-1], d["cand"].shape[-1]).cuda()
    if one_km:
        net.point = torch.nn.Sequential(torch.nn.LayerNorm(64), torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 2)).cuda()
    batches = max(1, len(photos) // 128)
    opt = torch.optim.AdamW([p for n, p in net.named_parameters() if not n.startswith("action.")], lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * batches, pct_start=0.1)
    rng = np.random.default_rng(seed)
    net.train()
    for _ in range(epochs):
        for bi in np.array_split(rng.permutation(photos), batches):
            b = _sub(d, torch.as_tensor(bi, device="cuda"))
            u = torch.rand(len(bi), 2, device="cuda", generator=gen) < 0.5
            if full:
                u[:] = True
            v = aa.view(b, u[:, 0], u[:, 1])
            z = v["mask"].clone() if full else _random_zoom(v, gen)
            if one_km:
                logp = torch.log_softmax(aa._options(net, v, z), 1)
                okm = lambda r: torch.cat([v["km_unzoom"] < r, (v["km_zoom"] < r) & z], 1) & torch.cat([v["mask"], v["mask"] & z], 1)
                loss = 0.0
                for r, w in ((25.0, 1.0), (1.0, one_km)):
                    ok = okm(r)
                    use = ok.any(1)
                    if use.any():
                        loss = loss - w * torch.logsumexp(logp.masked_fill(~ok, float("-inf")), 1)[use].mean()
                if not torch.is_tensor(loss):
                    continue
            else:
                ok = torch.where(z, v["km_zoom"], v["km_unzoom"]) < 25
                ok &= v["mask"]
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
    return net


@torch.no_grad()
def scores(net, v, z, one_km):
    """log-softmax over the 2h options (one_km) or the h areas."""
    return torch.log_softmax(aa._options(net, v, z) if one_km else net.answer_logits(v, z), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    aa.CKPT_TAG = "setft_own"
    tr = aa._data("train")
    dv = aa._data("dev3k", stats=tr["stats"])
    b = _sub(dv, torch.arange(dv["n"], device="cuda"))
    on = torch.ones(dv["n"], dtype=torch.bool, device="cuda")
    v = aa.view(b, on, on)
    z = v["mask"]
    h = z.shape[1]
    km2 = torch.cat([v["km_unzoom"], v["km_zoom"]], 1)
    km_area = torch.where(z, v["km_zoom"], v["km_unzoom"])
    ar = torch.arange(dv["n"], device="cuda")
    S = {}  # (state, one_km) -> per-seed score tensors
    for state in ("mixed", "full"):
        for w in (0.5, 0.0):
            S[state, w] = [scores(fit(tr, np.arange(tr["n"]), 7 + 100 * s, w, state == "full"), v, z, w) for s in range(a.seeds)]
            print(f"fitted {state} one_km {w}", flush=True)

    def joint(sc):
        return km2.gather(1, sc.argmax(1, keepdim=True)).squeeze(1)

    def area_only(sc):
        return km_area.gather(1, sc.argmax(1, keepdim=True)).squeeze(1)

    def factor(sc_area, sc_pt):
        area = sc_area.argmax(1)
        use_zoom = sc_pt[ar, h + area] > sc_pt[ar, area]
        return torch.where(use_zoom, v["km_zoom"][ar, area], v["km_unzoom"][ar, area])

    def oracle_pt(sc_area):
        area = sc_area.argmax(1)
        return torch.minimum(v["km_zoom"][ar, area], v["km_unzoom"][ar, area])

    ens = lambda xs: torch.stack(xs).mean(0)
    rows = {}
    for state in ("mixed", "full"):
        j, a0 = S[state, 0.5], S[state, 0.0]
        rows[f"{state}: joint 25+1 km (current)"] = [joint(x) for x in j]
        rows[f"{state}: 25 km head only"] = [area_only(x) for x in a0]
        rows[f"{state}: factorized"] = [factor(x, y) for x, y in zip(a0, j)]
        rows[f"{state}: factorized, oracle point"] = [oracle_pt(x) for x in a0]
        rows[f"{state}: joint, {a.seeds}-seed ensemble"] = [joint(ens(j))]
        rows[f"{state}: factorized, {a.seeds}-seed ensemble"] = [factor(ens(a0), ens(j))]
    ref = torch.stack([(k < 25).float() for k in rows["mixed: joint 25+1 km (current)"]]).mean(0).cpu().numpy()
    ref1 = torch.stack([(k < 1).float() for k in rows["mixed: joint 25+1 km (current)"]]).mean(0).cpu().numpy()
    lines = [f"answer-head structure, dev3k ({a.seeds} seeds; read + recognise + zoom all; vs the current head, seed-averaged)"]
    for name, kms in rows.items():
        k = torch.stack(kms).cpu().numpy()
        acc, acc1 = (k < 25).mean(0), (k < 1).mean(0)
        d, d1 = acc - ref, acc1 - ref1
        lines.append(f"  {name:42s} < 25 km {acc.mean():6.1%} ({100 * d.mean():+.2f}, SE {100 * d.std() / np.sqrt(len(d)):.2f})   "
                     f"< 1 km {acc1.mean():6.1%} ({100 * d1.mean():+.2f}, SE {100 * d1.std() / np.sqrt(len(d1)):.2f})   "
                     f"seeds {' '.join(f'{x:.1%}' for x in (k < 25).mean(1))}")
    text = "\n".join(lines)
    print(text, flush=True)
    with open("artifacts/agent/structure_probe_dev3k.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
