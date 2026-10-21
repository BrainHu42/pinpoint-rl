# Track V: one value function over tool actions and answers, with a reader that re-reads (2026-10-21)

Read `LEARNINGS.md` first; lesson N is in `docs/archive/LESSONS_1-137.md#lN`, lessons R1.. / S1.. in `LESSONS.md` (R11-R13 on branch
`worktree-retrieval-track`, 140 on `worktree-error-viewer` until merged). Lessons from this track go in `LESSONS.md` as V1, V2, ...

## Why
- **Two learned parts, two objectives, and the policy never answers.** Today a supervised answer head over pre-clustered areas
  (`hypothesis_agent._net`, hand-built columns per tool, trained on random reveal patterns) picks the answer, and a separate REINFORCE
  policy (`action_agent.py`) only chooses which columns to reveal. Every new tool needs new columns, and candidates a tool creates mid-episode
  (text places) have to be squeezed into the area tokens (lesson 140).
- **Framing:** one network Q(state, action) over every action. "Commit to candidate k (point option j)" is an action whose value is the expected
  reward [< 25 km] + 0.5 x [< 1 km]; an information action's value is its cost subtracted from the best value reachable after it. The policy takes
  the best action and stops when committing beats every further action after cost. The current head is the commit part of this; the policy
  becomes argmax Q, and the accuracy-vs-cost curve comes from the same network.
- **What the framing alone will not do:** raise accuracy. The head's form is saturated (R11: training on real states +0.26, factorized
  25 / 1 km +0.28, ensembles +0; lesson 126). Its value is one model, one objective, and candidates that can appear during an episode.
- **Where the accuracy has to come from: what the reader reads.** On commons26 dev the pipeline (25.2% < 25 km) is wrong on 160 photos (16.4%
  of dev) whose text names the place; a text candidate < 25 km exists for 54% of them, lesson 119's ranker picks it for 21%. A look at 14 of
  them: small text misread or invented (a sign "Gregorio Aznárez" read as "Gregorio Arzate", 0 matches, guessed Argentina; a hospital's
  name on a small ground sign missed and "Chiang Mai University Hospital" invented 603 km off), names read right but ambiguous alone, and
  knowledge cases (out of scope). R12 concluded the limit on commons26 is what the 4B reads, not how its reads are searched. Re-reading the
  text box at full resolution when its search fails, or with a larger reader, is an action whose value depends on the last outcome: the
  setting where a policy can beat fixed recipes (main point 6).
- **Scope:** commons26 (fresh photos); expect small gains (< 1 pt) on dev3k / im2gps3k / yfcc4k, measured, not assumed:
  - On MP16-like photos text sits where retrieval already succeeds (main point 3): names the 4B reads geocode within 25 km on 0.5% of
    misses vs 10% of hits (lesson 110); a right search result on photos no area covers 1.3% of dev3k vs 7.0% of commons26 dev (lesson 123).
  - Resolution: dev3k (MP16) photos are 640 px on the long side (0.31 MP), yfcc4k 500 px (0.17 MP), im2gps3k 1024 px (0.79 MP), all within
    the 4B's 0.79 MP budget, so it already sees every pixel and a crop re-read adds none (only upscaling). commons26 photos are 1280-1707 px
    (1.23 MP median), downscaled ~35% today: crops bring real pixels there.
  - Why read names don't place the wrong photos (lesson V1): on dev3k (6.5 pts of photos) the names are generic or not the photo's place.
    A specific name (<= 3 countries) exists near the truth on 0.3 pts, misreads <= 0.3 pts, and 56% of "elsewhere" names aren't in the
    verbatim text at all (inferred lists such as Swiss cities). On commons26 dev: specific right names 2.6 pts (luck 0.1), misreads <= 1.3.
  - This track does not target dev3k's wrong-area gap (area oracle 61.7).

## Ruled out by past lessons: not in this plan
| Not doing | Why (lessons) |
|---|---|
| Head architecture / training variants, ensembles, more MP16 rows, for accuracy | R11, 126, R8 (8x data +0.5) |
| Pool-derived constrained searches (country / region / box) as actions | S1-S2: right regions found, results can't pick them, no switch nets > 0 |
| Restricted-search verification of text places as a new signal | S3: AUC 0.944 but adds nothing over the ranker |
| Imagery near a candidate (YFCC, Commons live / GLDv2, Mapillary, OSV-5M) | 135, 136, 138, R7, R13; GLDv2 columns sub-threshold (R10) |
| A local VLM choosing / judging / scoring candidates (A/B, PMI, profiles, cue tables) | 84, 86, 106, 114, 128, 131-133, 137 |
| VLM attributes, descriptions, cue questions (language, driving side), map feature checks | 21, 77, 85, 133 |
| Conjunction / conditional search, rankers, overrides over the 4B's *current* reads | 79, R12 (+0.8 at best), 119-121 |
| Reverse geocoding: each area's nearby names fuzzy-matched against the photo's text | V2 (AUC 0.60-0.66; switches +0.31 / -1.00 at best) |
| More zooms, extra pool areas, more MegaLoc photos per area | 70-71 (2 zooms = full gain), 112, 97 |
| New 1 km refinement features | 91, 107, 112, R10 (doesn't transfer to dev3k) |
| RL / GRPO on the VLM's free text; the VLM as policy | 87, 96, 137 (silence, collapse); out of scope here |
| Offline Wikipedia, YFCC word statistics, photographer / author signals | 115, 92, 110; no photographer signal (user) |
| A reader comparison on dev3k / MP16 | 88 (27B found fewer right names than the 4B), 123 |

What is new here: value-based control with commit actions (never built; lesson 94 used REINFORCE over reveals); targeted full-resolution
re-reads of the photo's own text boxes on commons26 (lesson 7's crop rewrite was MP16, before the head; commons26 never had crop reads);
Qwen3.5-9B / Qwen3.6-35B-A3B as readers on commons26 (the 27B was only tried on dev3k); reader escalation decided from search outcomes, judged
at matched reader-call budget.

## Steps (cheap before expensive; each has a pre-declared bar)
**Step 1. Value network on today's actions (refactor, no gain expected).** Replace the answer head + REINFORCE with one Q network on the same
cached reveals (zoom k / read / recognise / commit (area, point option)); reuse the `_net` encoder with a commit output per token and an
output per information action. Commit values by regression on outcomes; information values by fitted Q iteration (target: minus cost plus
the max over the next state's values, from a target network). Every outcome is cached, so states are sampled freely (random partial reveal
sets plus the network's own rollouts): no offline-RL coverage problem. Never fit targets from hindsight search over outcomes (it sees the truth).
Bar: lambda 0 within +-0.3 of the current pipeline on dev3k (47.3 / 30.8 with `--one-km 0.5`) and commons26 dev (25.2), and a cost curve at
or above REINFORCE's. If it fails, keep the current head; steps 2-4 still run on it.

**Step 2. Do better reads exist? (the go / no-go for everything after).** commons26 dev, 977 photos (the 282 whose text names the place reported
apart). Readers: the 4B (current), Qwen3.5-9B, Qwen3.6-35B-A3B FP8. Reads: (a) the full photo with a box per string (Qwen grounding JSON);
(b) each box re-read from a full-resolution crop; (c) the 0.6-crop grid of `observe crops` as a box-free fallback. Every string goes through
the lesson 119 four-source search unchanged. Measure, per reader and read type, right candidates (< 25 km) on pipeline-wrong photos that the
current 4B candidate list lacks, net of luck (the same candidates against another photo's truth, lesson 120), and the lesson 119 ranker's
top-1 on them. Bar: >= 3 pts of dev net of luck in new right candidates (lesson 121 converted a +4.2 perfect switch into +0.9, so less can't
reach 1 pt). Also log each reader's latency per call: these become the action costs. Below the bar: stop the track and record why.
Scope check (measure, not choose): the winning reader on dev3k's wrong photos with a read place name (6.5 pts), same metric, to test the
"< 1 pt off commons26" expectation.
Expect it to be tight (lesson V1): of the 160 wrong commons26 dev photos whose text names the place, a read name that is a close misspelling
of a name near the truth covers <= 13 (1.3 pts), so crop re-reads alone can't reach 3 pts. The rest has to come from the 25 with names found
nowhere (garbled or not in the gazetteers), the 39 where text was read but no place name extracted, and the larger readers.

**Step 3. Everything at once.** Cache every step 2 action on commons26 dev, the Commons training pool (`commons_train`, built by the combined-head
session: consume, don't rebuild) and a 10k MP16 train sample (missing-indicators elsewhere). Commit values (the step 1 network, or lesson 140's head
if step 1 failed) see all results. Bar: >= +1 pt < 25 km on commons26 dev over lesson 140's head (uploader folds, Commons rows x3, 3 seeds; SE
~0.6), dev3k no worse than -0.3.

**Step 4. The outcome-dependent policy.** Actions: read (4B full photo), re-read box i (4B crop), escalate box i (the step 2 winner), zoom k, recognise,
commit; searches run automatically after every read. Costs from step 2's latencies (a reader call >> a search). Policy = argmax Q, lambda swept.
Compare at matched reader calls: fixed recipes (read once; read + re-read every box; everything), gates (re-read only strings with 0 or
> 1,000 matches; escalate only when the re-read still fails), and an open-loop control (same actions chosen before seeing any result).
Bar: >= 1 pt over the best recipe / gate at the same budget, or the same accuracy with <= 60% of its calls, on commons26 dev; and the
open-loop control worse (sequence matters). Then the primary table (original protocol) and commons26 test only when the user asks.

## Coordination (shared machine, same rules as the other briefs)
- Own worktree / branch (`worktree-value-agent`); outputs in `artifacts/value/`, new tags; never overwrite `artifacts/agent/*`,
  `artifacts/pinpoint_train/setft_own` or the released index. The combined head (`combined_head.py`) lives on `worktree-error-viewer`:
  merge it from there, don't fork it.
- GPU (L40S, 46 GB): vLLM readers only when `nvidia-smi` shows track R idle (35B-A3B FP8 needs ~35 GB weights). RAM: `systemd-run --user --scope
  -p MemoryMax=12G -p MemorySwapMax=0` by default. The machine stops at 02:00 UTC and wipes NVMe (`text_search stage` after a restart).
- Lessons in the V section of `LESSONS.md`; touch `LEARNINGS.md` / `CLAUDE.md` only for this track's lines. Confirm long runs with the user;
  test set only when asked.
