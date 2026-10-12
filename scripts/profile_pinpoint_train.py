"""Profile Pinpoint reranker training (pinpoint_train/): Lightning's SimpleProfiler (data wait vs step) plus CUDA-synced timers around the
module's stages. Same arguments as scripts/pinpoint_train.sh; run it through that launcher's python with a few dozen steps:

  PYTHONPATH=pinpoint_train/src:src data/pinpoint-main-src/pinpoint-main/.venv/bin/python scripts/profile_pinpoint_train.py <train.py args>
"""

import sys
import time
from collections import defaultdict

import torch
from lightning.pytorch.profilers import SimpleProfiler

import pinpoint.attention_reranker.train as train_mod
from pinpoint.attention_reranker.lightning_module import AttentionInterpolateLightningModule as M

TIMES: dict[str, list[float]] = defaultdict(list)
STAGES = ("_common_step", "_encode_query_image", "_build_osv_support_token", "_retrieve_candidates_union", "_build_candidate_batch",
          "_rerank_outputs", "forward", "training_step")


def timed(name, fn):
    def wrapper(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        TIMES[name].append(time.perf_counter() - t)
        return out

    return wrapper


import os

THROUGHPUT = os.environ.get("PROFILE_MODE") == "throughput"  # no synced timers or op profiler: steady-state seconds per step only
STAMPS: list[float] = []
if not THROUGHPUT:
    for stage in STAGES:
        if hasattr(M, stage):
            setattr(M, stage, timed(stage, getattr(M, stage)))

# torch.profiler over training steps 30-34: op-level CPU vs CUDA time
_PROF = {"step": 0, "prof": None}
_step = M.training_step


def profiled_step(self, batch, batch_idx):
    _PROF["step"] += 1
    STAMPS.append(time.perf_counter())
    if THROUGHPUT:
        return _step(self, batch, batch_idx)
    if _PROF["step"] == 30:
        _PROF["prof"] = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                               record_shapes=False)
        _PROF["prof"].__enter__()
    out = _step(self, batch, batch_idx)
    if _PROF["step"] == 34 and _PROF["prof"] is not None:
        torch.cuda.synchronize()
        _PROF["prof"].__exit__(None, None, None)
        ka = _PROF["prof"].key_averages()
        print("\n== torch.profiler, steps 30-34, by self CPU time ==")
        print(ka.table(sort_by="self_cpu_time_total", row_limit=25, max_name_column_width=70))
        print("\n== by CUDA time ==")
        print(ka.table(sort_by="self_cuda_time_total", row_limit=20, max_name_column_width=70))
        _PROF["prof"] = None
    return out


M.training_step = profiled_step

import lightning as L


def with_profiler(args, *, logger, callbacks, device_kwargs):
    return L.Trainer(max_steps=args.steps, val_check_interval=args.val_every, check_val_every_n_epoch=None, limit_val_batches=args.val_batches,
                     log_every_n_steps=args.log_every, logger=logger, callbacks=list(callbacks), num_sanity_val_steps=0,
                     enable_checkpointing=False, use_distributed_sampler=False, profiler=SimpleProfiler(), **device_kwargs)


train_mod.create_default_trainer = with_profiler


def report():
    if len(STAMPS) > 30:
        steady = STAMPS[20:]
        print(f"\n== steady state (steps 20-{len(STAMPS)}): {(steady[-1] - steady[0]) / (len(steady) - 1) * 1000:.0f} ms per step ==")
    print("\n== CUDA-synced stage times (skipping the first 5 calls) ==")
    for name, ts in TIMES.items():
        ts = ts[5:] or ts
        print(f"  {name:28s} calls {len(ts):4d}  mean {1000 * sum(ts) / len(ts):8.1f} ms  total {sum(ts):7.1f} s")


if __name__ == "__main__":
    sys.argv[0] = "train.py"
    t0 = time.perf_counter()
    try:
        train_mod.main()
    finally:
        print(f"wall {time.perf_counter() - t0:.0f} s")
        report()
