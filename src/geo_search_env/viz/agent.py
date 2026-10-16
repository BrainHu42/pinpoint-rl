"""The lesson 109 answer head and RL policies, trained once on all of train (as `action_agent --ckpt-tag setft_own --apply`) and saved, plus a
step-by-step greedy rollout that records what the agent saw and chose at every step.

  PYTHONPATH=src .venv/bin/python -m geo_search_env.viz.agent fit     # -> artifacts/viz/agent_setft_own.pt (~minutes, GPU)
  PYTHONPATH=src .venv/bin/python -m geo_search_env.viz.agent check   # greedy km on full / dev3k vs action_agent_apply_km_setft_own.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

CKPT_TAG = "setft_own"
LAMBDAS = (0.0, 0.005)
OUT = Path("artifacts/viz")
AGENT_PATH = OUT / f"agent_{CKPT_TAG}.pt"


def _aa():
    from ..experiment import action_agent

    action_agent.CKPT_TAG = CKPT_TAG
    action_agent.ONE_KM = 0.0
    return action_agent


def fit() -> None:
    import torch

    aa = _aa()
    tr = aa._data("train")
    n = tr["n"]
    net = aa.fit_answer(tr, np.arange(n), seed=7)
    pols = {lam: aa.fit_policy(net, tr, np.arange(n), lam, seed=11) for lam in LAMBDAS}
    OUT.mkdir(parents=True, exist_ok=True)
    dims = (tr["pre"].shape[-1] + 2, tr["post"].shape[-1], tr["cand"].shape[-1])
    torch.save({"net": net.state_dict(), "dims": dims, "pols": {lam: p.state_dict() for lam, p in pols.items()},
                "stats": {k: tuple(np.asarray(x) for x in v) for k, v in tr["stats"].items()}}, AGENT_PATH)
    print(f"answer head + policies {list(LAMBDAS)} trained on {n} train photos -> {AGENT_PATH}", flush=True)


def load_agent():
    """(net, {lambda: policy}, train stats) from AGENT_PATH."""

    import torch

    from ..experiment.hypothesis_agent import _net

    aa = _aa()
    ck = torch.load(AGENT_PATH, weights_only=False)
    net = _net(*ck["dims"]).cuda()
    net.load_state_dict(ck["net"])
    net.eval()
    pols = {}
    for lam, sd in ck["pols"].items():
        p = aa._policy()
        p.load_state_dict(sd)
        p.eval()
        pols[lam] = p
    return net, pols, ck["stats"]


def trace(net, pol, b, max_steps: int | None = None) -> list[dict]:
    """Greedy rollout of policy `pol` on batch b (as action_agent.rollout 'greedy'), recording per photo and step: the state, the policy's
    action probabilities (zoom per area slot, read, recognise, stop), the action taken and the answer head's area probabilities in that
    state. The last step of every photo is 'stop' (or the step cap); its answer probabilities are the final answer."""

    import torch

    aa = _aa()
    n, h = b["mask"].shape
    zoomed = torch.zeros_like(b["mask"])
    read = torch.zeros(n, dtype=torch.bool, device=b["mask"].device)
    rec = torch.zeros_like(read)
    done = torch.zeros_like(read)
    out = [[] for _ in range(n)]
    with torch.no_grad():
        for _ in range(max_steps or aa.MAX_STEPS):
            logits, v = pol.logits(net, b, zoomed, read, rec)
            probs = torch.softmax(logits, 1).cpu().numpy()
            ans = torch.softmax(net.answer_logits(v, zoomed & v["mask"]), 1).nan_to_num(0.0).cpu().numpy()
            a = logits.argmax(1)
            an = a.cpu().numpy()
            for i in np.flatnonzero(~done.cpu().numpy()):
                act = "zoom" if an[i] < h else ("read", "recognise", "stop")[an[i] - h]
                out[i].append({"read": bool(read[i]), "recognise": bool(rec[i]), "zoomed": zoomed[i].cpu().numpy().tolist(),
                               "p_zoom": probs[i, :h].tolist(), "p_read": float(probs[i, h]), "p_recognise": float(probs[i, h + 1]),
                               "p_stop": float(probs[i, h + 2]), "action": act, "slot": int(an[i]) if act == "zoom" else None,
                               "answer": ans[i].tolist()})
            live = ~done
            is_zoom = a < h
            zoomed = zoomed | (torch.nn.functional.one_hot(a.clamp(max=h - 1), h).bool() & (is_zoom & live)[:, None])
            read = read | ((a == h) & live)
            rec = rec | ((a == h + 1) & live)
            done = done | (a == h + 2)
            if done.all():
                break
        v = aa.view(b, read, rec)
        z = zoomed & v["mask"]
        logits = net.answer_logits(v, z)
        ans = torch.softmax(logits, 1).nan_to_num(0.0).cpu().numpy()
        pick = logits.argmax(1).cpu().numpy()
    for i in range(n):  # the state the answer was given in (after the final action)
        out[i].append({"read": bool(read[i]), "recognise": bool(rec[i]), "zoomed": z[i].cpu().numpy().tolist(), "action": "answer",
                       "slot": int(pick[i]), "answer": ans[i].tolist()})
    return out


def fixed_answer(net, b, read: bool, rec: bool, zoom_all: bool) -> np.ndarray:
    """Answer head probabilities over area slots with a fixed tool set (no policy)."""

    import torch

    aa = _aa()
    n = b["mask"].shape[0]
    r = torch.full((n,), read, dtype=torch.bool, device=b["mask"].device)
    c = torch.full((n,), rec, dtype=torch.bool, device=b["mask"].device)
    v = aa.view(b, r, c)
    z = v["mask"] if zoom_all else torch.zeros_like(v["mask"])
    with torch.no_grad():
        return torch.softmax(net.answer_logits(v, z), 1).nan_to_num(0.0).cpu().numpy()


def check() -> None:
    """The saved agent reproduces lesson 109's numbers (greedy km per photo vs the saved apply run)."""

    import torch

    aa = _aa()
    net, pols, stats = load_agent()
    saved = np.load("artifacts/agent/action_agent_apply_km_setft_own.npz")
    for tag, key in (("full", "full:setft_own"), ("dev3k", "dev3k")):
        d = aa._data(tag, stats=stats, ckpt=CKPT_TAG)
        b = aa._sub(d, torch.arange(d["n"], device="cuda"))
        for lam, pol in pols.items():
            km = aa.rollout(net, pol, b, lam, "greedy")[0].cpu().numpy()
            ref = saved[f"{key}|RL policy, lambda {lam:g}"]
            print(f"{tag} lambda {lam:g}: < 25 / 1 km {np.mean(km < 25):.1%} / {np.mean(km < 1):.1%} (saved run {np.mean(ref < 25):.1%} / "
                  f"{np.mean(ref < 1):.1%}); same answer on {np.mean(np.isclose(km, ref, atol=0.01)):.1%} of photos", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("fit", "check"))
    args = parser.parse_args(argv)
    fit() if args.cmd == "fit" else check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
