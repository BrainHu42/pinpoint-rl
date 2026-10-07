# Overnight research loop: toward a geolocation agent that reasons like a professional (2026-10-07, ~9 h)

You are running alone overnight under /loop; the user is asleep. **Reread this file and the journal at the start of
every round.** This plan is deliberately open-ended: it gives the goal, what's known, the resources, the guardrails and some
starting ideas. **You decide what to try, in what order, and when to change course.** Follow the evidence. The user
approved this kind of work by asking for it; stay within the guardrails and the budget.

## Mission
Make progress on `AGENT_DIRECTION.md` (read it first, it's short): an agent that geolocates like a professional geolocator.
It reads text, applies rule-like clues, queries map data for features that co-occur near each candidate, geocodes
names, then eliminates and verifies. Visual retrieval (Pinpoint) is only the candidate generator. The research question:
**can constraint-style evidence through tools fix the photos where visual similarity fails?** And can a model learn
(SFT / RL on ground truth) to use such tools?

**North-star metric:** change in top-1 accuracy within 25 km (and 1 km) vs the reranker top-1, with fixes and breaks,
especially on the "fixable" photos: reranker top-1 >= 25 km off, but the best-ranked candidate in the true region is < 25 km.
A good night ends with a clear, honest answer about at least one tool idea, with numbers. A working pipeline the next
session can build on also counts. A null result, clearly measured, is a good result too.

## What's known (details: `LEARNINGS.md` summary, `LESSONS.md` 1-4, 8, 35, 50-53)
- Full im2gps3k + yfcc4k eval halves (n = 3,795): reranker top-1 16.7 / 38.2 % within 1 / 25 km; pool oracle 33.4 / 59.0.
- Every similarity-style lever (choosers, SFT/GRPO, evidence, comparators, knowledge SFT) lands at -0.5 to +1.3 pts.
- 792 photos have the top-1 wrong with a right candidate in the pool. Knowing the true region would fix 413 of them
  (+10.8 pts at 25 km): true city +8.4, true country +3.3. The 413: 280 yfcc4k / 133 im2gps3k; 77 % are > 200 km
  off; the top-1 is in the right country for 62 % (US 140); the first true-region candidate is rank 7+ for 49 %.
- Neither the base 4B nor the knowledge-SFT model knows these regions: right region ~9-11 % on the 413, vs ~50 % on
  photos the reranker gets right. This isn't a training-coverage gap (lesson 53), so the clues have to come from elsewhere.
- Text helps where present (lesson 4). GeoNames name lookup needs fuzzy matching (lesson 6). About a third of hard photos
  have nothing place-specific (lesson 35): expect a ceiling.
- Earlier "evidence" work fed retrieval more *similarity* (search queries, attribute matching, concept similarity) and
  got nothing (11-21, 51). Constraint queries ("which candidates have a river + church + rail bridge within 2 km?",
  "which countries write like this?") are untested.

## Resources
- **Repo:** this worktree, `/home/brianhu/pinpoint-rl/.claude/worktrees/agent-tools` (branch `worktree-agent-tools`). Enter it with
  EnterWorktree `path` if your session isn't already in it. `artifacts`, `data`, `.venv`, `.venvs`, `.env` are symlinks to
  the main checkout. Code map in `CLAUDE.md`. Run modules as `PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.<m>`.
- **Photo sets** (`artifacts/query_evidence/{dev,val,full}/dev.json`): dev = 1,000 and val = 1,000 MP16 bucket-99 photos (images
  via `sft_data.MP16Images`); full = the 3,795 eval-half photos (paths under `/data/pinpoint/...`; remap with `paths.DATA_ROOT`, as
  `name_score._read` does). Each has the truth, the reranker-ordered pool (~17-19 candidates) and the top-10 candidates with names.
  Place labels (country / region / locality / neighbourhood) for the truth and every pool candidate:
  `artifacts/place_labels/candidates_{dev,val,full}.json` (about 98 % labelled).
- **Training data:** `artifacts/sft/sft_retrieval.jsonl` (bucket 99, 34,530 train / 3,837 val, photographer split; candidate prompt
  with names, `meta` with the reranker top-1 and truth). `grpo_train.py` (TRL GRPO with colocated vLLM, distance reward,
  env `.venvs/grpo`), `sft_train.py` (env `.venvs/sft`). Neither has been run on this machine yet (expect setup issues: base model
  as an init needs `data/pinpoint/sft/base/merged` -> the snapshot; image size is `sft_train.MAX_PIXELS`).
- **Models:** Qwen3.5-4B (`data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) and
  `data/pinpoint/sft/knowledge-a/merged` (knowledge SFT, 448 px). Serve with vLLM as in `scripts/knowledge_ceiling.sh` (copy its
  env exports: `C_INCLUDE_PATH`, `VLLM_USE_FLASHINFER_SAMPLER=0`, PATH to `.venvs/vllm/bin`). Thinking off, no MTP.
- **Map data on disk:** GeoNames `data/pinpoint/geonames/allCountries.txt` (12M entries: names, alternate names, feature class / code,
  coordinates; classes H water, T terrain, S spots / buildings, L areas, P populated places, R roads / rails, V vegetation),
  `featureCodes_en.txt`, `countryInfo.txt` (per country: languages, TLD, currency, phone code, postcode format, neighbours),
  `landmarks.npz` (1.2M class S/L entries only, see `evidence_test.landmarks`).
- **Downloads allowed:** public data and open-weight models, up to **60 GB in total tonight** (243 GB free). Examples:
  Overture places / buildings / divisions (`aws s3 cp --no-sign-request` or https from `overturemaps-us-west-2`; the DuckDB env
  `.venvs/overture` doesn't exist on this machine, so create one with uv if needed). Also Qwen3.5-9B or another open VLM from
  Hugging Face if model capacity is the question (set `HF_HOME=data/hf`). Log every download (what, size, why) in the journal.
- **Compute:** one L40S (46 GB), 30 GB RAM, no swap. Deadline: the latest one recorded in the journal (restarts may set a new budget).

## Guardrails (hard rules)
- Long jobs run in a memory-capped scope (`systemd-run --user --scope --quiet -p MemoryMax=24G -p MemorySwapMax=0 ...`), launched
  with `setsid nohup ... > log 2>&1 &`; record PIDs. Stop our processes by PID only, never `pkill -f`. One GPU job at a time.
- **Never train on the benchmarks.** Develop and choose everything (prompts, tools, thresholds, rules, combiners, checkpoints)
  on MP16 dev / val or bucket-99 train. The full eval halves are a test set: **at most 3 full-set measurements tonight**, each of a
  method fixed beforehand; report every one, including the bad ones. Same-photographer gallery exclusion stays as is.
- Training only on bucket 99 (`sft_retrieval.jsonl` and the dev / val pools are bucket 99). No Gemini, OpenRouter or other paid
  APIs, no web search for benchmark photos (contamination), no reverse image search. Public map data is fine.
- Ground-truth labels only; no distillation from another model's answers.
- Single GPU runs longer than 3 h need a clear reason written in the journal first. Don't start anything that can't finish 45 min
  before the deadline.
- Git: commit your code on this branch (author `-c user.name="Brian Hu" -c user.email="brainhu42@gmail.com"`, messages ending
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`). Don't touch `main`, don't merge, don't force-push. Pushing will fail
  (no credentials); that's fine.
- Match the repo style: one module per idea under `src/geo_search_env/experiment/` with the usage line at the top, scripts in `scripts/`.

## Starting ideas (suggestions, not a checklist)
- **The development target first:** build the "fixable" set for dev and val with the same definition, so tools can be judged on
  dev photos like the 413. `knowledge_oracle.py` has the logic.
- **Observation extraction:** the base 4B reads each photo into structured JSON: verbatim text, script / language, driving side,
  plate and sign style, a list of checkable features from a fixed vocabulary, named places. Try full resolution (786432 px) and crops
  for small text. Measure extraction quality on dev before building on it.
- **Constraint tools over GeoNames** (and Overture if worth the download):
  - `features_near(candidate, features, km)`: an index over chosen GeoNames classes (H / T / S / R) with a map from visual words to
    feature codes. Weight rare features more than common ones.
  - `geocode(text, near=candidate)`: fuzzy match of read text against names and alternate names within reach of each candidate.
  - `country_rules`: languages / TLD / phone / currency / postcode format from `countryInfo`, plus driving side, giving allowed countries.
- **Measure the tools as candidate scorers without training:** rank + tool features in the listwise combiner (`name_score._fit` /
  `transfer` pattern), CV on dev, checked on val. Count fixes and breaks on the dev / val fixable photos. Ablate each tool. Ask where
  each tool is right: on the photos retrieval fails on, or only where it already works (lessons 17-19, 52)?
- **Toward the agent, if a tool shows signal:**
  - a one-step agent: tool results in the candidate prompt, GRPO A (with tools) vs B (same prompt without), bucket-99 hard photos,
    same recipe and hours;
  - or a multi-turn prototype where the model chooses which tool to call.
- **Capacity questions:** does a larger open VLM (e.g. 9B) read text, features or regions much better on the dev fixable photos?

## How to work
- Keep a **research journal** at `artifacts/overnight2/JOURNAL.md` (create it): for each round, append the time, what you're testing and
  why (a hypothesis), what ran (commands, PIDs, logs), the numbers, and what you concluded / what's next. A fresh session must be
  able to continue from this file alone.
- Prefer quick, decisive measurements (minutes to an hour) before building big things. Kill ideas that show no signal on dev.
  Double down on ones that do.
- Measure beyond what we have: the reranker top-1 and the pool oracle, not current greedy (`CLAUDE.md`).
- When you find something real, check it isn't a leak (benchmark photos, same photographer, labels derived from the answer).

## Round procedure (every /loop round)
1. Reread this file and the journal; note the clock vs the deadline.
2. Check running jobs (PIDs, log tails, `nvidia-smi`, `free -g`). If one failed, fix it (at most 2 retries per job), else move on.
3. Do the next piece of work, or launch the next job. Append to the journal.
4. ScheduleWakeup: 1200-1800 s while a long job runs; 270 s if waiting on something short; keep working within the round when
   coding or analysing.

## End of night (start >= 45 min before the deadline)
1. `artifacts/overnight2/REPORT.md` (<= 50 lines, also committed as `OVERNIGHT2_REPORT.md`): what you tried and why, results with CIs
   (dev / val and any full-set measurements), fixes / breaks, what worked, what didn't, recommended next steps with commands.
2. New lessons (54, 55, ...) in `LESSONS.md` with one-line entries in the `LEARNINGS.md` summary; update `AGENT_DIRECTION.md` if the
   evidence changes the direction. Commit.
3. Stop our servers and jobs by PID, confirm the GPU is idle, stop the loop (ScheduleWakeup `stop: true`).
