# Overnight plan (written 2026-10-07; ~9 h unattended under /loop)

You are carrying this out alone; the user is asleep. **Reread this file and the state file at the start of every round.**
It is self-contained: everything you need is here or in the files it names. The user approved the runs listed here by
asking for this plan; **start no long GPU run that is not listed.**

## Where things are
- **Work in the worktree** `/home/brianhu/pinpoint-rl/.claude/worktrees/knowledge-ceiling` (branch
  `worktree-knowledge-ceiling`; enter it with EnterWorktree `path`). Its `artifacts`, `data`, `.venv`, `.venvs`, `.env`
  are symlinks to the main checkout, so outputs land in the shared `artifacts/` and `data/`. All paths below are
  relative to it. Commit there (author `-c user.name="Brian Hu" -c user.email="brainhu42@gmail.com"`, end messages with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`); push only this branch. Never touch `main`, merge, or force-push.
- **State file:** `artifacts/overnight/STATE.md` (create it on the first round). Append-only log: timestamp, step,
  PIDs, outputs, numbers, next action. The first entry records the start time and the **deadline = start + 9 h**.
- Background reading only if a step needs it: `CLAUDE.md` (rules), `KNOWLEDGE_PLAN.md` (the plan this executes),
  `LEARNINGS.md` summary, `LESSONS.md` 1-4 and 50.

## Context (what's known when this was written)
- Question: can a model learn geographic knowledge that visual similarity lacks? Measured as fixes on benchmark photos
  where Pinpoint's reranker top-1 is wrong. Baseline on the full im2gps3k + yfcc4k eval halves (n = 3,795): reranker
  top-1 16.7 / 38.2 % within 1 / 25 km; pool oracle 33.4 / 59.0.
- Step 1 done: `knowledge-a` = Qwen3.5-4B LoRA-SFT on 250k MP16 photos -> "country > region > city"
  (`data/pinpoint/sft/knowledge-a/merged`, trained at 448 px = max_pixels 200704). Its check (`artifacts/knowledge/logs/knowledge_check.log`):
  on photos where the reranker is >= 25 km off it names the right country 53.2 vs 45.5 % (base 4B), region 17.6 vs
  13.2, city 4.1 vs 4.3 (no gain). Only 86.4 % of its answers keep the 3-level format (base 99.6 %).
- Earlier GRPO (lessons 1-3) only reached reranker parity (16.9 / 37.9): GRPO sharpens what the model knows, it
  doesn't discover. The new ingredient is the knowledge model.
- **Ceiling check** (`scripts/knowledge_ceiling.sh`, launched 04:00, ~45-60 min, log
  `artifacts/knowledge/logs/knowledge_ceiling.log`; it ends with "all done"): does knowledge-a's knowledge pick better
  among the retrieval candidates than the reranker, with no training? For knowledge-a and the base 4B (control):
  - `knowledge_check rule`: move to the best-ranked candidate (within top K) whose place labels match the model's
    answer down to a level; variant chosen on MP16 dev (1,000 bucket-99 photos), every variant shown on the full eval halves.
  - `name_score report` (dev, CV) and `transfer` (combiner of rank + candidate-name log-probabilities fitted on dev,
    applied once to the eval halves): change vs reranker top-1 with 95 % CI, fixes and breaks at 25 km; and the
    within-photo AUC of each name score on dev (base 4B scored 0.50-0.53 in lesson 50 = no signal).

## Round procedure (every /loop round)
1. Read this file, then `artifacts/overnight/STATE.md`. Note the clock and the deadline.
2. Check what is running: the PIDs in STATE (`kill -0`), log tails, `nvidia-smi`, `free -g`.
3. Running and healthy: append a one-line progress note (step, training step / photos done, key metric), then
   ScheduleWakeup 1200-1800 s. Finished: verify the outputs (counts, 0 failures, numbers sane), record the numbers in
   STATE, start the next step. Failed: read the log, fix, retry; **at most 2 retries per step**, then take that step's fallback.
4. Short interactive steps (coding, smoke tests): keep working in the round; wake in 270 s if waiting on something short.
5. End every round with STATE updated so a fresh session could continue from it alone.

## Hard rules
- Long jobs run in a memory-capped scope: `systemd-run --user --scope --quiet -p MemoryMax=24G -p MemorySwapMax=0 ...`
  (30 GB RAM, no swap; an uncapped job crashed the machine). Launch with `setsid nohup ... > log 2>&1 &` and record the PID.
- One GPU job at a time. Stop our processes by PID only (never `pkill -f`). Before starting a GPU job, `nvidia-smi` must show < 1 GB used.
- Never train on the benchmarks. Choose every rule, checkpoint, threshold and hyperparameter on MP16 data (dev / val),
  never on benchmark numbers. Benchmarks are measured once per final model.
- Use the full eval halves (3,795) for results; small subsets only for plumbing.
- A and B must get the same recipe, data, image size and hours. If time is short, shorten both equally; never run A without B.
- No Gemini / OpenRouter calls, no wikimedia, no new research directions beyond this file.

## Step 0: wait for and read the ceiling check
- Don't use the GPU until the ceiling log says "all done". If it died, rerun the failed part once (the script is
  idempotent per stage; reruns of `name_score score` overwrite their outputs).
- Record in STATE, for knowledge-a and base: the dev-chosen rule and its full-half change at 1 / 25 km (CI), fixes,
  breaks; the transfer combiner's change (CI), fixes, breaks; the best dev AUC at country / region / city level.
- **Decision (fixed before seeing the results):** GO to GRPO if either
  (a) for knowledge-a, the transfer combiner or the dev-chosen rule improves top-1 < 25 km on the full eval halves by
  >= +0.5 pts with the CI lower bound > 0, **and** by more than the base 4B's same number; or
  (b) knowledge-a's best dev AUC at country or region level is >= base's + 0.03 (the knowledge separates candidates even
  if a crude scorer can't turn it into fixes; GRPO with the photo may).
  Otherwise NO-GO: skip to "NO-GO" below.

## GO: GRPO A (from knowledge-a) vs B (from base 4B)

**Prep (target <= 1.5 h, CPU / short GPU).** Commit each piece.
1. Base as an init: `grpo_train --init X` loads `data/pinpoint/sft/X/merged`. Create `data/pinpoint/sft/base/merged` as a
   symlink to `data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Check that
   `grpo_train merge` works with it (it may copy tokenizer files from the init).
2. Image size: `grpo_train.py` uses `sft_train.MAX_PIXELS` (786432) for the processor and the colocated vLLM. Add
   `--max-pixels` (default unchanged) and use **200704 for both runs** (knowledge-a's training size). Same for serving at eval
   (`scripts/full_eval.sh` hard-codes 786432: add an env override).
3. Hard-photo training data `artifacts/grpo/grpo_hard.jsonl` from `artifacts/sft/sft_retrieval.jsonl` (bucket 99,
   34,530 train / 3,837 val rows, split by photographer; the prompt lists ~17 candidates in retrieval order with place names and
   asks for "Country: .. Region: .. City: .. Candidate: n" + JSON coordinates; `meta` has `nearest_candidate_km`,
   `reranker_top1`, `lat`, `lon`). Keep train rows with `nearest_candidate_km < 25` (solvable). Among those keep all whose
   reranker top-1 is >= 25 km from the truth, plus an equal-size random sample (seed 0) of those where it is < 25 km (so
   "always switch" isn't rewarded). Keep val rows unchanged. Record the counts. This filtering is the lesson-3 / Geo-R fix
   for groups where all 8 samples get the same reward.
4. Smoke test both inits: `grpo_train train --init knowledge-a --run smoke-a --data artifacts/grpo/grpo_hard.jsonl
   --max-steps 2 --max-pixels 200704` (env `.venvs/grpo`, `PYTHONPATH=src`, in the memory scope), same for `--init base`.
   On GPU OOM: `--batch 4 --accumulation 16` (same effective batch) for **both** runs.
5. Step-0 check (~30 min): serve each init with vLLM (as in `scripts/knowledge_ceiling.sh`, max_pixels 200704) and run
   300 val rows of `grpo_hard.jsonl`: answer parse rate (the reward's `parse_coordinates`), greedy < 25 km, best-of-8 at
   T = 1.0. Find or write the smallest tool for this (see how `scripts/eval_run.sh` scores MP16 val). Record the numbers.
   Expectation: if A's best-of-8 isn't above B's, A is unlikely to beat B; continue anyway (the run is the real test),
   but say so in the report. If a parse rate is < 30 %, GRPO may get no signal: still run, and watch the first 20 steps'
   reward; if it stays ~0 for 30 steps, stop that run and report.
6. Benchmark eval plumbing: check that `scripts/full_eval.sh <run> <prompt> 0.0` (calls `pivot_diagnostics vlm_sampling
   --full-eval`) builds the same retrieval prompt for the eval halves on this machine. Run it on ~20 photos with the
   base init. If it can't be made to work within 1 h, still train, and evaluate what you can (MP16 val) — note it.

**Runs.** hours per run H = min(2.75, (time left - 1.75 h) / 2), rounded down to 0.25 h. If H < 1.5, don't start GRPO:
write the report with the prep results and the recommended command for tomorrow.
- Run A: `grpo_train train --init knowledge-a --run grpo-know-a --data artifacts/grpo/grpo_hard.jsonl --hours H
  --max-pixels 200704` (defaults: lr 5e-6, KL beta 0.04, T 1.0, 8 samples; it logs entropy and auto-stops on collapse).
  Then `grpo_train merge --init knowledge-a --run grpo-know-a`.
- Run B: identical with `--init base --run grpo-base-b`.
- While running, record every ~30 min: step, mean reward, entropy, the in-training MP16-val accuracy (every 50 steps).
- Use the final adapter for both (no checkpoint picking on benchmarks).

**Eval (~40 min).** Greedy (T = 0) on the full eval halves for grpo-know-a and grpo-base-b, at max_pixels 200704; if time
allows, also the two inits (shows what RL added). Per model: top-1 < 1 / 25 / 200 km; vs the reranker top-1 per photo:
fixes (reranker >= 25 km off, model < 25 km) and breaks; **A minus B** paired bootstrap 95 % CI at 1 and 25 km;
parse-failure rate. Compare with the ceiling check's combiner (the bar A should clear).

## NO-GO
No GRPO tonight. CPU-only diagnosis (<= 1 h), then the report:
- On the full eval halves, among photos where the reranker is >= 25 km off and some pool candidate is < 25 km: how often
  does knowledge-a name the right country / region (from `artifacts/knowledge/check_knowledge-a.json` and
  `artifacts/place_labels/candidates_full.json`), and how often is the right candidate the *only* pool candidate in that
  region (the case where region knowledge alone could fix it)? This separates "the model lacks the knowledge" from
  "knowledge at that level can't separate the candidates".
- Then stop the loop (ScheduleWakeup `stop: true`) after the report. Don't fill the night with other runs.

## Morning report (both branches; finish >= 30 min before the deadline)
1. `artifacts/overnight/REPORT.md`, <= 40 lines: decision and why (numbers), what ran, results table with CIs, fixes /
   breaks, A - B, what failed or was skipped, recommended next step with its command.
2. Add lesson 52 to `LESSONS.md` and a one-line entry to the `LEARNINGS.md` summary; update the status line in
   `KNOWLEDGE_PLAN.md`. Commit on the branch and push the branch.
3. Stop any of our servers by PID, confirm the GPU is idle, and stop the loop.
