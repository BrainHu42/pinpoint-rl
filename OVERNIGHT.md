# Autonomous research loop 3: landmarks, business names, and the first agent tool (2026-10-07, 5 h)

You are running alone under /loop. **Reread this file and the journal at the start of every round.** The plan is open-ended: it sets
the goal, what's known, the resources, the guardrails and starting ideas. **You decide what to try, in what order, and when to change
course.** Follow the evidence. The user approved this kind of work by asking for it; stay within the guardrails and the budget.

## Mission
Pursue the three next steps from the last session (`OVERNIGHT2_REPORT.md`, lessons 54-56), all toward `AGENT_DIRECTION.md`: an agent
that geolocates like a professional, with retrieval as the candidate generator and tools that add constraints.
1. **Landmark matching as a tool.** About 12% of the photos retrieval fails on show a recognisable landmark (castle, museum interior,
   cloister). Can a tool recognise it and point to the right candidate?
2. **Business-name search that tolerates misreadings.** About 15% show a readable business or organisation name. Overture POI search with
   exact names gave +0.2 (dev + val); misreads ("Ketter" for "Tepper") and missing entries ("Seg World") lose matches. Can fuzzy matching
   and better candidates recover them?
3. **Rank + text as the first agent tool.** It's the one confirmed gain: +0.4 [+0.2, +0.6] within 25 km on the eval halves, 18 fixes / 3 breaks.
   Combine the tools that work into one precise chooser. If time allows, take a first step toward an agent that uses tool results,
   e.g. a one-step GRPO run with the tool results in the prompt vs the same prompt without.

**North-star metric:** change in top-1 within 25 km (and 1 km) vs the reranker top-1, with fixes and breaks, especially on the fixable photos
(reranker >= 25 km off, best-ranked candidate in the true region < 25 km: dev 102, val 106, eval halves 413). Clear null results count.

## What's known (details: `LEARNINGS.md` summary; `LESSONS.md` 22-26, 39-41, 47-49 for exemplar / comparator / keypoint matching; 52-56)
- Eval halves (n = 3,795): reranker top-1 16.7 / 38.2 % within 1 / 25 km; pool oracle 33.4 / 59.0. Perfect region choice would add +10.8.
- Constraint tools (lesson 54): text near candidate +0.4 on the eval halves; POI names +0.2, text -> region +0.1, features / rules / country
  guess ~0 (dev + val). Text names the true region on ~5% of fixable photos; reading crops adds text but no fixes.
- Capacity (55): 9B vs 4B right region on fixable photos 16.8 vs 13.9% (n.s.); resolution helps to 786k px, then saturates.
- Audit (56): of 24 dev fixable photos, 58% nothing place-specific, ~15% business name, 12% landmark, 12% country-level objects.
- Earlier visual matching (22-26, 39-41, 47-49): a fine-tuned 4B comparator (query vs exemplar photo of a candidate: same place?) gave +0.7
  to +0.9 at 25 km; zero-shot judges mostly measure exemplar similarity; keypoint matching couldn't separate near-miss look-alikes (AUC 0.52),
  but it was tested on candidates within 25 km of each other, not on landmarks across regions.

## Resources
- **Repo:** this worktree, `/home/brianhu/pinpoint-rl/.claude/worktrees/agent-tools` (branch `worktree-agent-tools`; enter it with EnterWorktree
  `path` if needed). `artifacts`, `data`, `.venv`, `.venvs`, `.env` are symlinks to the main checkout. Code map in `CLAUDE.md`.
- **Built last session (reuse):** `agent_targets.py` (fixable sets), `observe.py` + `scripts/observe.sh` (VLM clues; `crops` node),
  `geo_tools.py` (GeoNames index, tools, `POIs`, CV `report`, `transfer`), `poi_query.py` (DuckDB over Overture places; env `.venvs/overture`),
  `agent_capacity.py`, `agent_vote.py`, `scripts/capacity.sh`. Clues on disk: `artifacts/agent/observe_{dev,val,full}_9b.json` (Qwen3.5-9B;
  fields include `place_names`, often the landmark's name), `observe_{dev,val}_9bcrop.json`, tool features `tools_*_9b.npz`.
- **Photo sets:** splits in `CLAUDE.md`: dev = `artifacts/query_evidence/dev3k/dev.json` (MP16, via `sft_data.MP16Images`), test =
  `full` (by path, remap with `name_score._read`). The `val` tag is test photos: never use it; place labels `artifacts/place_labels/candidates_*.json`.
- **Visual data:** MP16 SigLIP2 gallery embeddings `data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384` (with lat / lon and image ids),
  MP16 images (tar shards, `MP16Images`), neighbour caches `artifacts/strategy_search{,_v6}`. The trained comparator weights are **not** on
  this machine; its code (`comparator_data.py`, `comparator_train.py`, `exemplar_judge.py` helpers) is at git tag `pre-cleanup-2026-10-07`
  (`git show pre-cleanup-2026-10-07:src/geo_search_env/experiment/<file>`). Same-photographer exclusion applies to any exemplar photo.
- **Map data:** GeoNames (`data/pinpoint/geonames`, index `artifacts/agent/geonames_index.npz`), Overture places (`data/pinpoint/overture/places`,
  11 GB, has names, categories, websites, brand). Downloads allowed: public data and open-weight models, up to **40 GB** this session (e.g.
  Overture buildings, Wikidata dumps of landmarks with coordinates, a keypoint matcher such as LightGlue). Log each one in the journal.
- **Models:** Qwen3.5-4B and Qwen3.5-9B (`data/hf/hub/...`), `knowledge-a`. Serve with vLLM as in `scripts/capacity.sh`. Training envs
  `.venvs/sft`, `.venvs/grpo` (GRPO not yet run on this machine).
- **Compute:** one L40S (46 GB), 30 GB RAM, no swap. Deadline: the latest one in the journal.

## Guardrails (hard rules)
- Long jobs in a memory-capped scope (`systemd-run --user --scope --quiet -p MemoryMax=24G -p MemorySwapMax=0 ...`), launched with
  `setsid nohup ... > log 2>&1 &`; record PIDs; stop them by PID only, never `pkill -f`. One GPU job at a time.
- **Never train on the benchmarks.** Develop and choose everything on MP16 dev / val or bucket-99 train. **Leave the test set (eval halves) alone:**
  no model runs, features, diagnostics or measurements on it unless the user asks (CLAUDE.md). Exemplar / gallery photos: exclude the query's
  photographer and benchmark photos.
- Training only on bucket 99. No Gemini, OpenRouter or other paid APIs; no web or reverse image search for benchmark photos. Public map data is fine.
- Ground-truth labels only; no distillation from another model's answers.
- GPU runs over 3 h need a written reason in the journal. Nothing may start that can't finish 45 min before the deadline.
- Git: commit on this branch (author `-c user.name="Brian Hu" -c user.email="brainhu42@gmail.com"`, messages ending
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`). Don't touch `main`, don't merge, don't force-push.
- Repo style: one module per idea under `src/geo_search_env/experiment/` with the usage line at the top; scripts in `scripts/`.

## Starting ideas (suggestions, not a checklist)
- **Landmarks:**
  - (a) The 9B's `place_names` often name the landmark. Geocode them (GeoNames / Overture / a Wikidata landmark list) and score candidates by distance.
  - (b) Visual: for photos the VLM flags as showing a landmark, compare the query with gallery photos near each candidate. Options: a VLM judge
    restricted to landmark photos; keypoint matching (rigid structures, across regions); or retrain the comparator from the tagged code if you
    judge it worth the hours.
  - First measure how many dev / val fixable photos show a landmark at all, and the tool's precision on those.
- **Business names:** fuzzy matching (edit distance / token overlap) between read text and Overture names near each candidate, instead of exact
  global lookup. Use Overture `categories` and `brand` to down-weight chains. Measure what fuzzy matching adds over exact matching.
- **Combine:** one combiner of rank + text + POI + landmark, chosen on dev + val. Look for a precision rule (switch only on strong, agreeing
  evidence) rather than a big feature set: last session the full set overfit.
- **Agent step (if the tools justify it):** tool results as a short list per candidate in the candidate prompt; GRPO on bucket-99 hard photos,
  A (with tool results) vs B (without), same recipe and hours. Expect setup work: base model as an init needs `data/pinpoint/sft/base/merged`.

## How to work
- Research journal: `artifacts/overnight3/JOURNAL.md` (create it). Each round: time, hypothesis, what ran (commands, PIDs, logs), numbers,
  conclusion and next step. A fresh session must be able to continue from it alone.
- Quick, decisive measurements before big builds; kill ideas without signal on dev; double down on the ones with signal.
- Measure beyond the reranker top-1 and the pool oracle. Check that any gain isn't a leak (same photographer, benchmark photos, labels).

## Round procedure (every /loop round)
1. Reread this file and the journal; note the clock vs the deadline.
2. Check running jobs (PIDs, log tails, `nvidia-smi`, `free -g`). Fix failures (at most 2 retries per job), else move on.
3. Do the next piece of work or launch the next job; append to the journal.
4. **Always end the round with ScheduleWakeup, at most 900 s**, even when a background job will notify. Last session lost 4 h to rounds that
   ended without one.

## End of session (start >= 45 min before the deadline)
1. `artifacts/overnight3/REPORT.md` (<= 50 lines; also committed as `OVERNIGHT3_REPORT.md`): what was tried and why, results with CIs, fixes /
   breaks, what worked, what didn't, recommended next steps with commands.
2. New lessons (57, ...) in `LESSONS.md` with one-line entries in the `LEARNINGS.md` summary; update `AGENT_DIRECTION.md` if the evidence changes
   the direction. Commit.
3. Stop our servers and jobs by PID, confirm the GPU is idle, stop the loop (ScheduleWakeup `stop: true`).
