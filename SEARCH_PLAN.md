# Track S: an RL agent whose searches create candidates (2026-10-20)

One of two tracks run in parallel sessions (the other: `RETRIEVAL_PLAN.md`). Read `LEARNINGS.md` first; lesson N is in
`docs/archive/LESSONS_1-137.md#lN`. Lessons from this track go in `LESSONS.md` under "Search-agent track" as S1, S2, ...

## Why
- Today's RL policy only saves cost: its actions (zoom into an existing area, read, recognise) are cheap, independent and reveal fixed
  features, so "take everything" ties it (lessons 94, 98). An RL policy earns its place when the next action depends on what the last one
  returned. This is the RL framing the project keeps (CLAUDE.md, Direction).
- Where to search is worth a lot: Pinpoint restricted to the true region reaches 61.3% < 25 km on dev3k vs 33.4 unrestricted, the true
  country +7.8 (lesson 64). Retrieval proposes regions far better than the 4B (best of 16 +19.0 vs +6.1, lesson 84).
- The losses this targets (dev3k, lessons 129-130): the head picks a wrong area on 14.4 pts and its wrong picks are cross-region (median
  381 km from the truth); 11.8 pts have a right candidate at noise-level weight that never becomes an area; some of the 26.5 pts of retrieval
  misses may surface under a region constraint, since the unconstrained search ranks them out.

## What is known (and what this must do differently)
- Zooming inside existing areas (25 km boxes): 2 zooms carry the full gain, and a learned zoom policy adds +0-0.3 over a fixed rule
  (lessons 70, 71). Extra areas from lower-ranked pool candidates +0.3 (lesson 112); region candidates with map / text checks +0.3 (lesson 85).
  These added candidates the head could not convert: added recall is worth nothing without evidence that tells the head to trust it.
- So the new ingredients are (1) constraints at coarser scales and outside the current areas (country / region of pool candidates via
  `artifacts/place_labels/mp16_country_region.npz`: 204 countries, 3,743 regions; boxes; exclusion of explored rows), (2) the *result* of a
  search as evidence (how well the restricted top-1 scores, agreement inside the restricted list, MegaLoc of the restricted top vs the query,
  score drop vs the unconstrained search), and (3) outcome-dependent sequences: coarse to fine (country -> region -> box), move on when a
  region's evidence is weak, stop when one is confirmed.

## Tools in the repo
- `tools/pinpoint.py`: `Pinpoint.search(embeddings, constraints)` with `Constraint` box / country / region / near / rows. none / box / country /
  region stream the full index per batch (`rank`); near / rows use `rank_subset` on a GPU-resident index (~10.5 GB; same top-1 within 1 km as
  box on 98%). The query photographer's photos are always excluded: under the primary protocol (`PINPOINT_KEEP_OWN=1`) check how the tool
  treats them and match the pipeline's rule. Measure the search speed first (`rank_subset` loops per query).
- `experiment/action_agent.py`: the current policy to reuse. State = a frozen answer net's per-area encoding with unrevealed columns
  masked, plus the query glance; actions zoom k / read / recognise / stop, <= 8 steps; return [answer < 25 km] - lambda x cost; REINFORCE with
  a self-critical greedy baseline; every outcome precomputed so a rollout only flips masks. `hypotheses.restricted_search` (a zoom = full
  Pinpoint restricted to a +-25 km box around a centre), `zoom_agent.py`.

## First steps (cheap before expensive)
1. **Search-tree ceiling with a luck control** (dev3k): a discrete constraint vocabulary from the photo's own pool (countries and regions of
   the top ~46 candidates; boxes around candidates outside every area). Run the restricted searches and measure how often some restricted top-1
   is within 25 km beyond the current area oracle (61.7), per loss bucket (wrong area / never an area / retrieval miss). Control for luck as in
   lesson 120: the same vocabulary scored against another photo's truth. Report searches per photo needed for the ceiling.
2. **Is a search's result evidence?** On the regions from step 1, does the restricted-search output separate right from wrong regions
   better than what the answer head already has (Pinpoint support AUC ~0.8, lesson 127), and on the photos where the head is wrong? This is
   the go / no-go before any RL: without it a sequential agent finds right regions the head can't pick (lessons 111-112 again).
3. **The agent**: state = the constraints searched so far and summaries of their results; actions = search a constraint generated from the
   current candidates (country / region / box / exclude-explored), answer, stop; cost per search; reward [< 25 km] + W x [< 1 km] - lambda x
   cost. search(photo, constraint) is deterministic, so cache it for the vocabulary on ~12k train photos + dev3k and replay. A small
   set-transformer policy first; the 4B as policy only if outcome dependence is shown to pay.

## Go / no-go
- >= 1 pt at 25 km on dev3k beyond the current pipeline (47.3 / 30.8 with `--one-km 0.5`), at matched search budget, vs fixed recipes (search
  the top-k regions by pool mass) and heuristic gates at the same cost (judge with realistic live-tool costs). Then commons26 dev.
- Show that sequence matters: the same action set with a policy that cannot see outcomes should do worse.
- Choose everything on train / dev3k; test set only when the user asks. Confirm long runs with the user first.

## Coordination with track R (shared machine; both briefs say the same)
- Work in your own worktree / branch (`search-track`); track R uses `retrieval-track`. Both start from main with this brief merged.
- Write to `artifacts/search/` and new tags; never overwrite shared caches (`artifacts/agent/*`, `artifacts/pinpoint_train/setft_own`, the
  released index). Use the current retriever (released) and reranker (`setft_own`), passed as parameters, so track R's can be swapped in.
- GPU (L40S, 46 GB): track S keeps <= 16 GB (resident index ~10.5 GB + reranker), track R <= 28 GB (a reranker fine-tune needs ~27 GB). vLLM or
  anything larger only after checking `nvidia-smi` that the other track is idle.
- RAM (30 GB, no swap): run long jobs with `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0` by default. A job that needs up to
  20G first checks `systemctl --user list-units --type=scope` and `free -g`, and never overlaps another job above 12G.
- CPU: track R's reranker fine-tune is CPU-bound and will slow search jobs while it runs.
- The machine stops at 02:00 UTC daily and wipes NVMe: checkpoint long jobs, re-stage the index after a restart.
- Lessons only in your own `LESSONS.md` section; touch `LEARNINGS.md` / `CLAUDE.md` only for your own track's lines.

## Status (2026-10-20): closed offline; the result feeds the tool-use design (user's decision)
Lessons S1-S3 (`LESSONS.md`; code `search_tree.py`, `text_verify.py`; outputs `artifacts/search/`). Step 3 (the agent) was not built.
- **S1, ceiling** (dev3k): ~37 constraints per photo from its own pool reach +14.0 pts beyond the area oracle, but a same-country photo > 100 km
  away gets +7.7 by luck: net ~+6, ~+2.4 at 8 searches.
- **S2, evidence** (1.31M cached searches on all train photos, dev3k): the restricted result lifts right-vs-wrong AUC among alternatives
  0.71 -> 0.76, yet the best-scored alternative is right on 6% of photos (some alternative 14-25%) and no switch nets > 0, even in hindsight;
  MegaLoc adds nothing. Pool-derived constraints all contain a look-alike by construction, so their search results can't verify them.
- **S3, independent proposals** (commons26 dev): places proposed by read text verify much better (restricted-search AUC 0.944 vs 0.847 for
  pool boxes on the same photos), but add nothing over the text ranker or the pipeline: where text is right and the pipeline wrong (116
  photos), the right place has few MP16 references (index photos within 25 km e^7.2 vs e^8.9) and verifies worse than the wrong pick.

### What carries into the tool-use agent
- **Propose from evidence retrieval didn't produce; spend verification there.** Pool-derived candidates are not worth a verification call
  (S2); text-proposed places are (S3). On commons26 dev a right text candidate exists on 35.0% of photos, 16.8% where the pipeline is wrong;
  a perfect choice among them is +4.2 pts (lesson 121).
- **Verification needs reference imagery at the candidate.** Offline, Pinpoint restricted to a +-25 km box is the check (batched ~1.5-3 ms;
  one streamed query ~1.2 s; a box on the GPU-resident index 2.6 ms). It is strong where the gallery is dense and blind where it is sparse,
  which is where text fixes live. The live tool to add is "photos at (lat, lon)": Street View / Places / recent Commons or Flickr images near
  a candidate, compared with MegaLoc / SigLIP2. Old Flickr (YFCC) and OSV-5M were already too sparse or off-view (lessons 135, 136).
- **The outcome dependence an RL policy can use:** the index density around a candidate (`Pinpoint.rows_near`) says when the free offline
  check is blind; the costly live fetch is worth calling only then, and only for independently proposed places. Read -> propose -> offline
  check -> live fetch if blind -> answer or next candidate.
- **Test set for a live verifier:** the 116 commons26 dev photos with a right text candidate and a wrong pipeline answer; the verifier pays if
  it ranks the right place above the ranker's wrong pick on well over half of them (offline checks: 16-37%), at matched call budget.
- **Gotchas found:** the full-commons26 MegaLoc caches (`megaloc_queries_commons26_setft_own.npy`, `megaloc_commons26_setft_own.npz`) are
  blank (pre-bd2d12d); only the commons26dev ones are valid. Small classifiers need enough optimizer steps (`search_tree._fit`, >= 2,000).
