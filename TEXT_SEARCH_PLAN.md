# Plan: a text-search tool and an agent trained to use it (draft for review, 2026-10-08)

## Goal
Fix Pinpoint's mistakes on photos with searchable text or a nameable place (domains, business and gallery signs, event banners,
landmarks). By eye these are about 5-7.5% of its mistakes (your test-set estimate; lesson 62 on dev3k). The agent reads the photo,
writes search queries, reads the results and decides whether to keep Pinpoint's answer or switch. Bar: **>= 1 pt within 25 km over
Pinpoint's top-1 on dev3k** (33.4%, so >= 34.4%), with fixes clearly outnumbering breaks. The test set stays untouched unless you ask.

## Where we are
**The tool exists:** `text_search.py` (branch `worktree-text-search`). `text_search(query)` sorts the query into one of three kinds:
- **Websites:** matched against Overture business websites; a country-code ending also gives a country hint.
- **Phone numbers:** matched against Overture business phones; an international prefix gives a country hint.
- **Names:** looked up as a phrase in GeoNames (24.7M names) and Overture (81.5M places).

Hits are grouped into 25 km clusters and returned as short text. The photo's own Flickr entry and its photographer are always
excluded. Searches take about 21 ms on average and 170 ms at worst, reading SQLite FTS5 from the NVMe drive.

**First dev3k measurement** (9B/27B phrases already produced; GeoNames + Overture; no YFCC; the tool picks without the truth):

| | Real phrases | Shuffled (chance) |
|---|---|---|
| Pick right < 25 km | 364 of 1,155 picks | 4 |
| Search-fixable (Pinpoint wrong, pick right) | **60 (2.0 pts)** | 2 |
| Traps (Pinpoint right, pick wrong) | 328 | 364 |
| Right place anywhere in the results, on Pinpoint's mistakes | **172 (5.7 pts)** | |

What it shows:
- The signal is real.
- Always switching would lose.
- The decision (when to trust a search) and choosing among results are where the gain is.
- The ceiling is above the 1-pt bar.

## Plan

### Step 1. Finish the tool (about half a day, CPU)
- **YFCC100M titles and tags:** `build-yfcc` once the other session's extraction finishes. It covers events and informal landmark
  names that GeoNames and Overture miss (e.g. "San Silvestre Toledana").
- **Postcodes:** GeoNames postal-code file (small download).
- **Rebuild:** for the corrected chain flag, so chains can be marked or down-weighted.
- **Optional, only if step 2 shows misreadings matter:** fuzzy matching (re-score near-matches).
- **Measure each source:** re-run `report` with each source switched on or off, to see what each adds.

### Step 2. Labels on train and dev (about a day; GPU for reading, CPU for searching)
1. The 4B reads every train photo (35,359) and dev photo and writes up to 5 queries: visible text, names, domains, phones.
   The 4B, not the 9B, because it is the model we will train.
2. Search every query and label each photo:
   - **search-fixable:** Pinpoint >= 25 km off, the tool's blind pick < 25 km;
   - **trap:** Pinpoint right, the pick far away;
   - **neither.**
3. Checks before trusting the labels:
   - shuffled-phrase control (done once; repeat with the 4B);
   - a by-eye audit of about 50 fixable labels: is the text visible, and is the match the place it names?
   - the YFCC title of each MP16 photo (joined on the Flickr id), to tell reading failures from lookup failures.
4. Output: the training mix, plus the dev3k ceiling with the 4B's own queries.

**Go/no-go:** the 4B's own queries must keep a ceiling well above 1 pt on dev3k (today 2.0 pts blind, 5.7 pts anywhere in the
results, with the 9B / 27B's phrases).

### Step 3. Headroom check (about half a day, GPU)
- **Setup:** the base 4B in the multi-turn loop (photo + Pinpoint's pool with its scores; tools `text_search` and `answer`;
  up to 3 searches), 8-32 samples per dev3k photo.
- **Measure:** best of N vs Pinpoint (fixes and breaks), and how often the model searches and how often it switches.

**Go/no-go:** best of 8 clearly above Pinpoint (target >= 3 pts within 25 km). GRPO sharpens what the model can already
sometimes do; it doesn't create it (`LEARNINGS.md` lesson 3). If the headroom is low, fix the prompt and result format first, or
seed with step 4.

### Step 4. Warm start: rejection-sampling SFT (about half a day, GPU)
- **What it learns from:** the 4B's own rollouts on train photos, keeping those judged good by ground truth only (no Gemini):
  - rollouts that beat Pinpoint on search-fixable photos;
  - rollouts that keep Pinpoint's answer on traps and on photos with nothing searchable.
- **Training:** LoRA SFT (`sft_train.py`).
- **Measure on dev3k:** fixes and breaks. This is also the fallback result if RL doesn't help.

### Step 5. GRPO (1-2 days, GPU; confirm with you before the run)
- **Trainer:** TRL 1.14 `GRPOTrainer` with `environment_factory`, which supports multi-turn tool calls (experimental in this
  version). Tool outputs are excluded from the loss.
- **Reward:** the existing `geo_reward` on the final answer, a small cost per search, and Pinpoint's top-1 plus a penalty for a
  malformed answer.
- **Prompt mix:** about one third search-fixable, one third traps or Pinpoint-right photos with text, one third random.
  Groups where all samples get the same reward are dropped.
- **Settings:** the safe ones from lesson 3 (lr 5e-6, KL 0.04, T = 1.0), with entropy watched (the earlier run collapsed at lr 2e-5).
- **Keep RL only if** it beats step 4 on dev3k.

### Step 6. Decision
- Report fixes, breaks and the change within 1 / 25 km vs Pinpoint on dev3k, with bootstrap intervals.
- Write a lesson.
- Run the frozen method once on the test set **only if you ask**.

## Decisions for you
1. **More training photos.** Train has about 1-2k search-fixable photos (estimate). Options:
   - pull extra geotagged YFCC100M photos outside MP16, picked by tags such as sign / shop / banner / station (Pinpoint never
     trained on them; benchmark photos and their photographers excluded; image download not yet checked);
   - or stay on bucket 99 only.
2. **Live web search.** Offline only for training (free, reproducible, no risk of finding the photo's own page), with web search
   added later for the deployed agent? Or bring it in now (needs a search API key and leak guards)?
3. **Scope.** Text and named places only (this plan), or let the same environment also take the landmark tools from lessons
   58-61 now?

## Risks
- **Most photos have no clue:** lesson 57 found 58% of fixable photos had nothing place-specific, so the ceiling is a few points at best.
- **Breaks:** confident wrong searches (328 traps today). If RL doesn't learn the decision, the gain disappears.
- **The 4B reads less than the 9B / 27B** (lesson 56: 13.9 vs 16.8% right region): its queries may lower the ceiling (step 2 measures this).
- **Coverage:** small businesses and events missing from the offline sources (YFCC and, later, web search address this).
- **Infrastructure:** multi-turn GRPO with images is new here: `environment_factory` is experimental, and prompts grow with each tool turn.
