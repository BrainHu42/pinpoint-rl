# Track V handoff (2026-10-23, 11:25 AM Pacific)

Track V (`VALUE_PLAN.md`) moves to the track R session at the user's request. Branch `worktree-value-agent`, worktree
`.claude/worktrees/value-agent`. Lessons V1-V5 are in `LESSONS.md`; everything below is the work after V5.

## Running jobs (let them finish; nothing else is running)
| PID | Command (from the worktree root, memory-capped scope) | Expected finish | Output |
|---|---|---|---|
| 105569 | `PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.value_agent --no-baseline --gain` | ~11:35 AM Pacific | `artifacts/value/value_agent_step1_gain.log`, `artifacts/value/value_agent_setft_own_gain.npz` |
| 105574 | `... value_agent --no-baseline --gain --listwise 1` | ~11:45 AM Pacific | `artifacts/value/value_agent_step1_gainlw.log`, `artifacts/value/value_agent_setft_own_gain_lw1.npz` |

Each job trains the Q network for 3 seeds (7, 8, 9; 30 epochs each) on train and evaluates dev3k and commons26 dev. Read it with
`grep -v epoch <log>`: per set, fixed recipes and the policy at each lambda (% < 25 / 1 km, cost units, zooms / read / recognise), then
"return - lambda x units". These runs are Q only (`--no-baseline`). Compare them with the REINFORCE rows of the first run below
(same seeds 7-9, same data; the RL rows are in `artifacts/value/value_agent_step1.log`, the km arrays in `artifacts/value/value_agent_setft_own.npz`,
keys `"{set}|{seed}|{RL|Q}|{row label}"`).

## Step 1 (value network on today's actions): state
Code: `src/geo_search_env/experiment/value_agent.py` (committed). It has one network: the `hypothesis_agent._net` encoder plus per-token commit
outputs, logistic P(< 25 km) / P(< 1 km) for the unzoomed and zoomed points, with value = P25 + 0.5 P1. Information values for zoom k / read /
recognise come from fitted Q iteration with a soft-updated target network: online argmax, target evaluation, commit included. Lambda is an
input, so one network gives the whole cost curve. States are random partial reveals plus the network's own epsilon-greedy rollouts after
epoch 5. The encoder gets gradients only from the commit loss (`--shared` turns that off).
Flags: `--gain` writes information values as the best commit value now (detached) plus a learned gain. `--listwise W` adds W x
`action_agent.fit_answer`'s listwise loss on log commit values. `--extra pe_g siglip_final pe_probe` adds the V5 columns, revealed by
recognise (`aa.N_ML` set). `--no-baseline` skips REINFORCE.

First run (`value_agent_step1.log`, default flags, 3 seeds, vs answer head + REINFORCE at the same seeds), lambda 0, % < 25 / 1 km:
| | dev3k | commons26 dev |
|---|---|---|
| REINFORCE | 47.23 / 30.67 | 24.97 / 7.68 |
| Q | 47.89 / 30.13 (+0.66, SE 0.24 / -0.53) | 25.18 / 7.06 (+0.20 / -0.61) |
Bar (VALUE_PLAN: lambda 0 within +-0.3 of 47.3 / 30.8 on dev3k and of 25.2 on commons26 dev; cost curve at or above REINFORCE's): **not met**.
- 1 km is 0.5-0.6 low. The deficit shows only on rows with zoomed areas (fixed zoom 0 rows match REINFORCE to <= 0.1). Per-option logistic
  values rank an area's unzoomed vs zoomed point worse than the old listwise softmax. `--listwise 1` tests this.
- The cost curve is below REINFORCE's: the Q policy barely reacts to lambda (6.7 units at lambda 0.015 vs REINFORCE 0.6; return 0.528 vs 0.602
  on dev3k). Costs are 0.0025-0.015 per unit while the information values' RMSE is ~0.11, so cost differences drown. `--gain` (advantage
  form) tests this.

Next, as I intended it:
1. Read the two logs. If `--gain --listwise 1` gives 1 km within 0.3 of REINFORCE and a cost curve at or above it, run the full comparison
   with the baseline once more for the record (~40 min):
   `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 env PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.value_agent --gain --listwise 1`
   Then write lesson V6 in `LESSONS.md`.
2. If the cost curve still lags: give the information heads lambda-specific targets with less noise (realised commit reward at s' instead of
   target-net values, or more on-policy states at high lambda), or fall back as VALUE_PLAN says (keep the current head; steps 2-4 run on it).
3. Then the same with the V5 columns (`--extra pe_g siglip_final pe_probe`): the RL policy's cost curve with the new columns was the
   open item of V5.

## Pending with the user
- **Regime idea (the user's brainstorm, not approved as a run):** separate models for the < 25 km and > 25 km regimes, across the whole system,
  not just the head. Today every stage is trained for 25 / 1 km, and the pipeline ties the released model at 200 / 750 / 2500 km (test
  62.9 / 75.5 / 86.9). I proposed two cheap checks on dev3k + commons26 dev (~1-2 h, cached data) and offered to run them after step 1. Neither
  has been started:
  (a) whether the coarse loss is in selection or in candidates: on photos where the head is unsure, coarse recall (any pool candidate / probe
  top-k cell < 750 km) vs the current answer's < 750 km rate;
  (b) a gated switch with no new models: answer low-confidence photos with the PE-G region probe's top cell or the pool-mass mode; gain at
  200 / 750 / 2500 km vs 25 km hits lost, per gate threshold.
  Also noted: lesson 133's cue tables were judged only between the head's top two areas at 25 km, never for picking a country when nothing is
  close. A multi-threshold commit head (P(< r) for r in 1 / 25 / 200 / 750 / 2500 plus hedge options) would be the one-network form.
- I told track R I'd run the V5 answer head on its PE-G candidates once they're written as extra pool rows / areas for train, dev3k and commons26 dev.

## Key files
- V5: `scripts/second_model_gate.py` (nodes embed / gate / head / siglip; `head --tags dev3k commons26dev --configs none pe_g pe_g+siglip_final`
  is the trained-head loop), `scripts/second_model_where.py`, launchers `scripts/second_model{,_train,_head,_commons}.sh`. Column files are
  `artifacts/value/{pe_g,siglip_final,pe_probe,dino_7b}_cols_{train,dev3k,commons26dev}_setft_own.npz` (dino_7b: train / dev3k only; pe_probe
  built by track R). They plug in through `hypothesis_agent.EXTRA_FILES` with `action_agent.N_ML = 4 + 4 x k`.
- Step 1: `src/geo_search_env/experiment/value_agent.py`, logs and npz files `artifacts/value/value_agent_*`.
- Probes V1-V4: `scripts/name_miss_probe.py`, `reverse_text_check.py`, `siglip_names_probe.py`, `gallery_text_check.py` / `gallery_text.sh`.
- REINFORCE reference: `artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz`.
- Combined head (lesson 140, step A / B on commons26 dev): another session's work on `worktree-error-viewer` (`combined_head.py`), outputs
  `artifacts/agent/combined_head_*.txt`. Track V didn't touch it; VALUE_PLAN says to merge it from there, not fork it.

## Uncommitted
Nothing apart from this note (`.venvs`, `data` are untracked by design). Artifacts are under `artifacts/` (not tracked).
