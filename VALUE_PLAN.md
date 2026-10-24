# Track V: one calibrated value network over tool actions and answers, on a growing simulator (rewritten 2026-10-23)

Read `LEARNINGS.md` first; lesson N is in `docs/archive/LESSONS_1-137.md#lN`, lessons R1.. / S1.. / V1.. in `LESSONS.md`. Lessons from this
track go in `LESSONS.md` as V6, V7, ... The 2026-10-21 version of this plan (reader-first) is in git history (`git show 4ab2d9e:VALUE_PLAN.md`).

## Idea
- **One network, Q(state, action), prices every action in reward units.** Answers are actions: "commit to candidate k, point option j" has
  value P(< 25 km) + W x P(< 1 km), calibrated. A tool action (zoom, read, recognise, ...) has the expected final return after it, minus its
  cost. Policy = argmax Q; stopping is automatic (commit when no tool is worth its cost). Lambda (the price of tool use) is an input, so one
  network gives the accuracy-vs-cost curve. A new tool is one more output, not new columns plus a retrained policy (today: a softmax answer
  head plus REINFORCE per lambda, `action_agent.py`).
- **The simulator is every tool's cached outcome per photo.** Trying any action in training is free, so states are sampled freely (random
  reveals plus the network's own rollouts) and there is no offline-RL coverage problem. Costs come from each tool's measured latency (live-tool
  costs, not compute on our cache).
- **Where accuracy comes from: the evidence and the actions, not the controller.** The head's form is saturated (R11, 126). What moved the
  number recently is new evidence per candidate: V5's PE-G + SigLIP2 + PE-G region probe columns, +2.7 pts < 25 km on dev3k (49.8) and
  commons26 dev (27.7). The controller's job is to call the right evidence when its value depends on what was already seen (main point 6:
  where a policy can beat fixed recipes).
- **Calibrated values make the answer metric an input too.** With P(< r) per answer for r in 1 / 25 / 200 / 750 / 2500 km, the answer is
  argmax sum_r w_r P(< r). It names the exact place when confident and hedges toward a region when not (the user's regime idea, 2026-10-23,
  as one network). Every stage today is trained for 25 / 1 km, and the pipeline ties the released Pinpoint at 200 / 750 / 2500 km (test
  62.9 / 75.5 / 86.9 vs 60.7 / 75.5 / 86.7, lesson 109). The head's loss never sees photos without a right option (~38% of dev3k).
- **Paper story this builds toward:** image geolocation as value-based tool use: what to look at, when to stop, what to answer, at any
  budget and any distance scale, from one calibrated value function. Closest prior work: classification with costly features (Janisch,
  Pevný, Lisý, AAAI 2019: DQN with "classify as k" actions) and active feature acquisition. New here: candidates that appear mid-episode,
  real retrieval / VLM tools with real costs, open-world answers, and multi-scale commits.

## Where it stands (2026-10-23)
- V1-V4: text probes (misreads, reverse geocoding, SigLIP2 names, gallery OCR): no-go; reads on MP16-like photos have < 1 pt to fix.
- V5: a second image model per area passes (PE-G + SigLIP2 + region probe columns, +2.66 dev3k / +2.70 commons26 dev over the head).
- Step 1 done (lesson V6, `value_agent.py --gain --mc --listwise 2`): lambda 0 matches REINFORCE (dev3k 47.40 / 30.56 vs 47.23 / 30.67,
  commons26 dev 25.08 / 7.68 vs 24.97 / 7.68); the cost curve is level with REINFORCE's within noise (dev3k -0.6 at the cheapest end,
  commons26 dev +1.8 there), not above it. The value network replaces the head + REINFORCE at no loss; it adds no accuracy by itself.
- Step 2 run once (lesson V6, `--split`): the policy only ties a simple gate on dev3k (<= +0.3 at matched cost, bar 0.5) and loses 1.4-2.2
  pts to it on commons26 dev (it keeps MP16's zoom habit; zooms hurt on commons26); open-loop control about equal (sequence barely matters).
  The gate "read + MegaLoc + SigLIP2 + probe, PE-G only where the top two commit values are within 0.02-0.2, no zoom" is the cheap recipe to
  beat: commons26 dev 26.85-27.47% at 3.8-6.5 units vs 27.33% for everything at 17.1. Open: Commons training states, a step cap above the
  number of actions.

- Step 3a done (lesson V7): the gated switch to the region probe passes on commons26 dev (net +1.84 at 750 km) and misses on dev3k
  (+0.83); the right region is in the pool on ~90% of unsure photos vs 52-61% for the answer (oracle room +11-12 pts at 750 km). 3b next.

- Step 3b done (lesson V8): P(< r) outputs + hedge answers; under the table's weights +3.6 / +4.0 at 750 km, +2.9 / +4.7 at 200 km
  (dev3k / commons26 dev), 25 km -0.08 / +0.41, 1 km -2.0 / -0.7; bars met. Calibrated on dev3k, overconfident on commons26 dev.

- Step 4 stopped (lesson V9): Pinpoint boxes around the hedge answers add right candidates (dev3k +2.6 net of nearby-truth luck, bar
  3; commons26 dev, auxiliary, +3.5), but the boxes' own scores pick a right box less often than random; needs an independent picker
  (PE-G columns on restricted candidates). Weight sweep: wc 0.25 lifts dev3k 25 km +1.06 with +3.2 at 750 km, 1 km -1.2; fix it on train CV.

- V5 columns in the multi-scale network (lesson V10): +2.4 at 25 km on dev3k over V8 (+2.9 commons26 dev), but overconfident (ECE 0.20)
  and the hedges' coarse gain is lost; likely the train columns exist only for photos with mixed right / wrong options. Next: columns for
  the other ~16k train photos (~6 h GPU, user's go-ahead) or a cheaper cause check first.

## Ruled out by past lessons: not in this plan
| Not doing | Why (lessons) |
|---|---|
| Head architecture / training variants, ensembles, more MP16 rows, for accuracy | R11, 126, R8 (8x data +0.5) |
| Pool-derived constrained searches (country / region / box) as actions | S1-S2: right regions found, results can't pick them, no switch nets > 0 |
| Restricted-search verification of text places as a new signal | S3: AUC 0.944 but adds nothing over the ranker |
| Imagery near a candidate (YFCC, Commons live / GLDv2, Mapillary, OSV-5M) | 135, 136, 138, R7, R13; GLDv2 columns sub-threshold (R10) |
| A local VLM choosing / judging / scoring candidates (A/B, PMI, profiles, cue tables between two areas) | 84, 86, 106, 114, 128, 131-133, 137 |
| VLM attributes, descriptions, map feature checks | 21, 77, 85 |
| Conjunction / conditional search, rankers, overrides over the 4B's *current* reads | 79, R12 (+0.8 at best), 119-121 |
| Reverse geocoding; SigLIP2 text-side names; OCR of the index photos | V2, V3, V4 |
| More zooms, extra pool areas, more MegaLoc photos per area | 70-71 (2 zooms = full gain), 112, 97 |
| New 1 km refinement features | 91, 107, 112, R10 (doesn't transfer to dev3k) |
| RL / GRPO on the VLM's free text; the VLM as policy | 87, 96, 137 (silence, collapse); out of scope here |
| Offline Wikipedia, YFCC word statistics, photographer / author signals | 115, 92, 110; no photographer signal (user) |
These close one implementation each, not the idea. An item comes back only as a new action whose value plausibly depends on earlier outcomes,
with its own pre-declared bar (e.g. cue tables were judged only between the head's top two areas at 25 km, never for a coarse answer).

## Steps (each with a pre-declared bar; dev3k + commons26 dev, 3 seeds; test only when the user asks)
**Step 1. The value network on today's actions (zoom k / read / recognise / commit).** `value_agent.py`. Bar: lambda 0 within +-0.3 of
answer head + REINFORCE at the same seeds (dev3k 47.23 / 30.67, commons26 dev 24.97 / 7.68, % < 25 / 1 km) and a return-vs-cost curve at or
above REINFORCE's. Candidates: `--gain --mc`. For 1 km prefer a zoomed-point logit written as the unzoomed one plus a learned difference (keeps
values calibrated) over `--listwise` (a ranking loss bends values away from probabilities, which the stop rule relies on). If it fails, keep the
current head and REINFORCE; steps 2-5 still run, with the value network only for the multi-scale answer.

**Step 2. Split "recognise" into one action per image model (cached; no new extraction).** Actions: MegaLoc, PE-G, SigLIP2 cosine, PE-G region
probe (DINOv3-7B only where cached). Columns from `artifacts/value/*_cols_*` (V5). Costs: per-model latency to embed the query plus 12 photos
per area on the L40S, in units relative to one zoom. Bar: (a) lambda 0 within +-0.3 of the V5 head with all columns revealed (dev3k 49.82 /
32.02, commons26 dev 27.70 / 7.92); (b) the policy beats the best fixed recipe and the best simple gate (e.g. PE-G only when MegaLoc's top-two
margin is small) by >= 0.5 pts < 25 km at matched cost, or matches them with <= 60% of the cost; (c) an open-loop control (the same action set
chosen before seeing any result) is worse, so sequence matters. This is the first test of "the controller adds something".

**Step 3. Multi-scale answers.** (3a, ~1-2 h, cached) Headroom: on photos where the head is unsure, coarse recall (any pool candidate / probe
top-k cell < 750 km) vs the current answer's < 750 km rate, and a gated switch to the probe's top cell or the pool-mass mode: gain at 200 / 750 /
2500 km vs 25 km hits lost. Stop if the best switch adds < 1 pt at 750 km net of 25 km losses. (3b) Commit heads P(< r) for r in 1 / 25 / 200 /
750 / 2500 km on the step 1 network (labels from ground truth on train), plus hedge commits (probe top cells, pool-mass mode, area centroids);
answer = argmax sum_r w_r P(< r). Bar: with weights (25 km 1, 1 km 0.5) no worse than step 1 (-0.3); with weights including the coarse
thresholds, >= +1 pt at 750 km (or the mean of 200 / 750 / 2500) on dev3k and commons26 dev, with 25 km >= -0.3.

**Step 4. An action that creates candidates where retrieval failed.** Lesson 110's dev3k misses: nothing within 25 km retrieved 25.9%, a right
candidate in the pool but in no area 10.4% (extra pool areas are ruled out, 112). Action: restricted Pinpoint search inside one of the PE-G
region probe's top cells (classifier cells, not pool-derived; S1-S2's failure was that results couldn't pick the right region, so measure
that directly). Gate first (cached searches for the probe's top 3 cells on dev3k + commons26 dev): new right areas (< 25 km) on photos with no
right area, net of luck (the same areas against another photo's truth), >= 3 pts; then as an action in the step 1 network, >= +1 pt < 25 km.
Below the gate: record why and stop this step.

**Step 5. Re-reading text (the old plan's steps 2-4; GPU-gated).** On commons26 only: re-read a text box from a full-resolution crop (4B), or
escalate it to Qwen3.5-9B / Qwen3.6-35B-A3B FP8; searches run after every read; re-read only when a search fails or is ambiguous. Gate first:
>= 3 pts of commons26 dev in new right candidates net of luck (lesson V1 expects this to be tight: misspellings cover <= 1.3 pts). Needs ~35
GB of GPU for the 35B, which conflicts with track R's B2 embed: run only with the user's go-ahead in a free slot. If it passes: actions
read / re-read box i / escalate box i in the network, judged at matched reader calls vs fixed recipes and gates (>= 1 pt or <= 60% of calls).

**Then**: the primary table (original protocol, 1 / 25 / 200 / 750 / 2500 km) and commons26 test, only when the user asks.

## Exploration: one model per distance regime (user, 2026-10-24)
Every trained part of the system serves one scale: Pinpoint's retriever and reranker learned "same place within 25 km", the answer head
(25 + 0.5 x 1 km). The questions differ by scale: 1 km is "same spot" (viewpoint, landmark), 25 km "which town", 200-750 km "which region"
(architecture, vegetation, language, road furniture), 2500 km "which climate / continent". The evidence differs too, and so may the data:
OSV-5M failed as 25 km candidates (domain gap) but street views are good at telling countries apart. V8 shared one encoder across five
P(< r) outputs and moved only the answer step. Open question: do models trained per scale, on that scale's evidence and data, combined by
the calibrated value network (expected score under the chosen weights), beat one model at every scale? Experts trained for different scales
are also partly independent of each other, which ties this to probe A (independence predicts value).

Where scale already showed up: true region restriction lifts Pinpoint 33.4 -> 61.3% < 25 km, true country +7.8 (lesson 64); nested country /
region / place proposals best-of-16 +6.1 but one sample -19.6 (83); V9's boxes around region-probe cells hold right places (+2.6 net of
nearby luck on dev3k) yet were picked by Pinpoint's 25 km score, a fine-scale judge on a coarse question. Region knowledge didn't help where
retrieval fails at 25 km (52); nobody checked it at 200 / 750 km.

Ideas, cheapest first (each a probe of the question, not a system change):
- **R1. Regime mismatch in V9 (minutes, cached).** Pick the box by a coarse judge (the probe's cell probability, probe / pool mass, V8's
  P(< 750 km) for the hedge) instead of the restricted top-1's score, then answer with the box's restricted top-1. Tests whether "can't
  verify" was the wrong-scale judge. Compare to random and to the area answer, net of nearby-truth luck.
- **R2. Shared vs separate answer models (~45 min per run).** Train V8's network with one radius at a time (r = 1 / 25 / 750 km) and score
  each on its own column against the shared five-output network. Separate wins at its own r -> sharing costs; shared wins -> scales help each
  other (likely at 25 km, via regional support, V9's coarse-weight tie-break).
- **R3. Past "no" signals judged at coarse scales (cached features).** Text cues vs country tables (`cue_check`), the 4B's country guess
  (right on 53%), clue-judge outputs, knowledge-SFT region answers, OSV-5M neighbours: each closed at 25 km. Add them as columns to the
  multi-scale network's hedges / areas and measure 200 / 750 / 2500 km only. A signal that is useless at 25 km may be what a region expert needs.
- **R4. A region expert from more data (GPU hours; coordinate with track R).** The PE-G region probe trained on MP16 + OSV-5M (5M street
  views, country-diverse), or a PE-G projection trained with positives < 750 km (track R's R17 recipe at a coarse radius). Judge: probe alone
  at 750 / 2500 km and as hedges in V8. Then the coarse-to-fine cascade: region expert -> Pinpoint restricted to its top regions (lesson 64's
  +27.9 if the region is right) -> the 25 km head picks within.
- **R5. A 1 km expert.** Same-spot evidence (MegaLoc, keypoint geometry on the area's photos, 12 per area cached) trained only to separate
  < 1 km from 1-25 km within the right area; V8 loses 1 km to coarse weights, a separate expert could give it back.
- **R6. Per-scale tool values (cached, from V8).** Which tool's info value rises with which weight: if zoom pays at 1 km, PE-G at 25 km and
  the probe at 750 km, the controller has a real decision (finding 6 of PAPER_NOTES: tools need outcome- or goal-dependent value).
Paper angle: "scale-specific experts, arbitrated by a calibrated value network under the scoring rule" (mixture of experts by decision theory).
Start with R1 and R2 (cached / short); R4 needs the user's go-ahead (GPU hours beside track R).

## How every step is judged
- Costs from measured latency, the same units for every action; report return - lambda x units and accuracy vs mean cost.
- Baselines at matched budget: fixed recipes, simple heuristic gates, the open-loop control, and the previous step's system. A gain from more
  information alone is the evidence's, not the policy's; report them apart.
- Never fit targets with hindsight: every choice in a target rollout is the network's, made without the truth.
- Choose on dev3k (and commons26 dev for reading); the bar is beyond what we already have (main point: measure against the current pipeline).

## Coordination (shared machine)
- Branch `worktree-value-agent`; outputs in `artifacts/value/`, new tags; never overwrite `artifacts/agent/*`,
  `artifacts/pinpoint_train/setft_own` or the released index. The combined head (`combined_head.py`) lives on `worktree-error-viewer`: merge it from
  there, don't fork it.
- GPU: track R's B2 PE-G embed has priority (nightly, sometimes by day, ~11 GB); stay under 20 GB without asking; never stop its PIDs. RAM:
  `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`. The machine stops at 02:00 UTC (7 PM Pacific) and wipes NVMe.
- Report to the user directly; a heads-up before any major run; runs over ~2 h or anything on the test set only with the user's go-ahead.
  Lessons in the V section of `LESSONS.md`; commit and push after each lesson.
