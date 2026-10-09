# Plan: a text-search tool and an agent trained to use it (draft for review, 2026-10-08)

> **Status (2026-10-09): agent training (steps 2-5) is on hold; a direction decision is pending.**
> - **The design holds:** Pinpoint restricted to the true region reaches +27.9 pts on dev3k (lesson 64).
> - **No evidence source supplies the right region where Pinpoint's is wrong:**
>   - text search with a VLM judge: +0.3 (lesson 65);
>   - the VLM's own country guess: worse than Pinpoint's;
>   - domains / phone numbers: 21 photos;
>   - read-text language: barely predictive;
>   - number plates: 1.5% of photos, no better than Pinpoint;
>   - choosing among the pool's regions by restricted reranker score: -0.1.
> - **Explicit clues are rare on these personal-Flickr photos.** See "Options" and the checks at the end.

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

## Target design: Pinpoint as a tool (agreed 2026-10-08)
- **Pinpoint runs once:** its pool (<= 12 candidates) and scores go into the prompt.
- **The agent decides:** Qwen3.5-4B with LoRA sees the photo, the pool and Pinpoint's confidence. Its default answer is
  Pinpoint's top-1, so on photos without a clue the system is Pinpoint.
- **Tools:**
  - `pinpoint_search(country / region / box)`: Pinpoint's retrieval and reranker restricted to an area. This is the key one:
    a clue only has to give the region, and visual retrieval finds the place inside it, beyond the original pool. 62% of
    Pinpoint's far misses are the right country but the wrong region (lessons 52-54, measured with the old reranker).
    Restricting retrieval to the true region raised it from 38.4 to 64.4% < 25 km on im2gps3k (lesson 5, old retrieval).
  - `text_search(query, near=)`: built.
  - `zoom(crop)`.
  - Later: landmark tools, map features, rule tables, live web search when deployed.
- **The answer must come from a tool result** (a candidate or a text-search place). Free guesses are almost always wrong
  (lesson 4: 11 of 270).
- **Not a replacement for the reranker:** every LLM chooser so far landed near parity.

**Check before building the agent (step 0):** rerun Pinpoint on dev3k restricted to the true country / region (the ceiling) and
to the country / region of the text tool's blind pick (realistic); count fixes and breaks (`pinpoint_constrained.py`). If the
pick's region fixes clearly more than the pick's own coordinate (60), this design is the one.

**Step 0 results (2026-10-08, dev3k; Pinpoint 33.4% < 25 km):**

| Pinpoint restricted to | Photos | < 25 km | Change | Fixes / breaks |
|---|---|---|---|---|
| true country | 2,914 | 41.2% | **+7.8** [+6.7, +8.9] | 255 / 21 |
| true region | 2,901 | **61.3%** | **+27.9** [+26.2, +29.5] | 859 / 23 |
| text pick's country | 1,126 | 27.7% | -5.8 | 14 / 187 |
| text pick's region | 1,122 | 25.7% | -7.7 | 38 / 270 |
| (text pick's own coordinate) | 1,155 | 24.5% | -8.9 | 60 / 328 |

- **The ceiling is large:** knowing the region and letting Pinpoint search inside it reaches 61.3%, above the oracle over
  Pinpoint's own pool (50.3%), with almost no breaks. So the design is right.
- **The text picks don't supply right regions yet:** gating them on Pinpoint's confidence never turns positive (best -0.1).
- **So the bottleneck is reliable region evidence.** Next, find the subset of clues whose region is right more often than
  Pinpoint's (by clue type, specificity, agreement), and let the agent learn when to trust them.

**Region-evidence breakdown (`region_evidence.py`, dev3k).**
- **Setup:** 13,510 searches of the 9B / 27B phrases (GeoNames + Overture, no YFCC). Pinpoint's top-1 region is right on
  42.8% of photos.
- **No group of clues beats Pinpoint where they disagree.** Counted on (photo, region) pairs that differ from Pinpoint's region:

| Clue group | Query right | Pinpoint right |
|---|---|---|
| all | 119 | 2,724 |
| landmark names (27B) | 20 | 153 |
| text read (9B) | 83 | 1,971 (region right on only 8% of searches) |
| exact name, <= 5 matches, share >= 0.9 | 28 | 124 |
| 2+ queries agree, share >= 0.9 (best) | 33 | 106 |
| domain or phone | 4 | 36 |

- **Clues are right mostly where Pinpoint is already right:** with 3+ agreeing queries the region is right 56% of the time,
  but on disagreements Pinpoint still wins 124 to 23. This is the same pattern as lessons 17-19.
- **Upper bound with a perfect selector:** a query names the true region while Pinpoint's region is wrong on 119 photos;
  restricting Pinpoint there fixes 68 (+2.3 pts). That's the most these clues can give, and no simple rule gets near it.

**Eye check of those 119 photos (2026-10-08).**
- **Method:** sorted by each photo's best query, compared with the 811 trap photos (a query's region differs from Pinpoint's,
  Pinpoint right, query wrong), and looked at 5 contact sheets (45 photos).
- **Wins:** 38 are luck: a generic word ("CHIPS", "PAPER", "©2009") whose top cluster happened to fall in the right region.
  29 are names read in the photo: a shop shutter "Dr Sanchis Bergon", "R-KIOSKI", the Antietam plaque, "International
  Stadium Yokohama", "Porto de Manaus", "City Island". 52 are names the model recalled: Glastonbury, VDNKh, the Catacombs of
  Domitilla, Huangshan, Mount Rundle.
- **Traps outnumber wins in both real-clue groups,** counting specific results (cluster share >= 0.3):
  - **Text read in the photo:** 369 wrong vs 72 right. Text often names somewhere else: brands (KIRKLAND, KONA), shirts
    ("MICHIGAN"), posters for other cities ("Cartagena de Indias" in Bogotá, "Hochschule Wismar" in Groningen).
  - **Names the model recalled:** 691 wrong vs 72 right. These are look-alike guesses: Atlanta airport for San Jose's,
    Lake Champlain for Lake Starnberg, Cape Spear for Prince Edward Island, the Twelve Apostles in Egypt.
- **What separates them is visible, not countable:** whether the text belongs to the place (a shop sign, a street or station
  sign, a plaque) or travels (a brand, clothing, a poster), and whether a landmark is recognised from its own visible
  name or only from appearance. An agent that sees the photo could judge this. Counts and match statistics can't.

**With YFCC100M titles + tags added (48.5M geotagged photos, `yfcc.db`; same dev3k phrases):**

| | GeoNames + Overture | + YFCC |
|---|---|---|
| Blind pick: search-fixable / traps | 60 / 328 | 65 / 337 |
| Right place anywhere in the results, on Pinpoint's mistakes | 172 (5.7 pts) | **231 (7.7 pts)** |
| Region differs from Pinpoint's: query right / Pinpoint right | 119 / 2,724 | 148 / 2,721 |
| Best rule (2+ queries agree, share >= 0.9) | 33 / 106 | 47 / 195 |
| Perfect region selector + restricted Pinpoint | 68 fixes (+2.3) | **82 fixes (+2.7)** |

- **YFCC adds coverage:** events and informal names. "San Silvestre Toledana", absent before, now points to Toledo.
- **It doesn't add precision:** no rule beats Pinpoint where they disagree; the best is about 1 to 3.
- **The 147 wins:** 53 luck, 39 read in the photo, 55 recalled by the model.

**VLM clue judge (`clue_judge.py`, dev3k).**
- **Setup:** for each of the 4,591 specific clues (cluster share >= 0.3), the VLM sees the photo, the clue and where the search
  points, and is asked: is the name on a sign / building / plaque, or a landmark it recognises (Yes), or could it refer to
  somewhere else, such as a brand, clothing, a poster or a look-alike (No)?
- **Precision flips at high confidence:** counted on (photo, region) pairs that differ from Pinpoint's region:

| P(Yes) >= | 4B: clue right / Pinpoint right | 9B: clue right / Pinpoint right |
|---|---|---|
| 0 (all) | 95 / 896 | 95 / 896 |
| 0.5 | 26 / 36 | 27 / 49 |
| 0.7 | 17 / 14 | 13 / 15 |
| 0.8 | 14 / 6 | 11 / 9 |
| 0.9 | 8 / 1 | 8 / 1 |

  With the threshold chosen on one half and counted on the other: 4B 14 / 6, 9B 9 / 7. The 9B is no better than the 4B.
- **End to end (4B, P(Yes) >= 0.8, 25 photos switched):**
  - answering with the clue's own coordinate: **+0.3 pts** [+0.0, +0.5], 12 fixes / 4 breaks;
  - restricting Pinpoint to the clue's region: +0.1 (7 / 4).
- **At P(Yes) >= 0.5 (74 photos):** -0.1 to -0.3.
- **Conclusion:** the judge works: on accepted clues it turns about 1 right to 9 wrong into about 3 to 1. But few right clues
  exist. Specific named places are better answered by their own coordinate than by a region restriction. Text search with
  these readers adds about +0.3 pts on dev3k, far below the 1-pt bar.

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

## Options: where right-region evidence could come from (draft for discussion, 2026-10-08)
Knowing the region is worth +27.9 pts and the country +7.8 (lesson 64). Text supplies the right region on only ~150 dev3k photos, which is
about +0.3 pts in practice (lesson 65). Each option below has a cheap first check on dev3k.

| # | Option | Ceiling / why | First check (dev3k) | Risk |
|---|---|---|---|---|
| 1 | **Country first.** Restrict Pinpoint to the country the photo's clues imply: the VLM's country guess, script / language, driving side, plates, domain and phone prefixes | True country +7.8 with only 21 breaks; country cues are far commoner and easier than region cues | Pinpoint restricted to the 9B's country guess (already in `observe_dev3k_9b*.json`), gated by the judge's confidence; one GPU pass, ~15 min | Lesson 55 found country rules ~0 *as candidate scores*; restriction is a different use but the VLM's country may be wrong exactly where Pinpoint's is |
| 2 | **Choose among the pool's regions with restricted search.** For each region in Pinpoint's pool, restricted Pinpoint gives that region's best candidate; the agent compares them | Restricted search per pool region raises the oracle beyond the pool's 50.3% | Oracle over pool regions with per-region restricted top-1; whether any signal (reranker score, the VLM, a comparator) picks better than Pinpoint | Choosing between look-alike regions is the visual-similarity problem lessons 22-29 hit (comparator +0.9) |
| 3 | **Region-level rule clues:** regional plate codes, area codes, route shields, regional chains (`GEOLOCATION_CLUES.md` sections 1, 5, 8) | Constraints that don't depend on appearance | VLM reads plates / route numbers / area codes on Pinpoint's dev3k mistakes; count how often present and right | Lesson 57: most hard photos show nothing place-specific; regional codes rarer still |
| 4 | **Build the agent now** with Pinpoint-as-tool + text search + the judge, and let RL combine them | Tools are complementary; the judge shows the visual decision is learnable | Base-4B headroom (best of 8 vs Pinpoint) in the multi-turn loop | No tool clears the bar alone, and GRPO sharpens rather than discovers (lesson 3) |
| 5 | **Live web search** for the deployed agent | Coverage of right clues is the limit lesson 65 found | Needs a search API key; on Flickr-based dev the photo's own page can leak | Can't be measured cleanly offline |

**Option 1, first checks (`vlm_country.py`, dev3k; truth / Pinpoint countries from the Overture labels):**
- **The VLM's holistic country guess is worse than Pinpoint's:**
  - 9B greedy: right on 62.4% of the photos it answers, where Pinpoint is right on 77.1% of the same photos. Where they
    disagree: VLM right 145, Pinpoint right 461. Even in Pinpoint's least confident fifth: 40 vs 63.
  - 8 samples per photo, the modal answer: 4B right 46.9% and 9B 54.6%, against Pinpoint 68.4%.
  - Where all 8 samples agree and differ from Pinpoint: 4B 10 vs 19, 9B 47 vs 47.
  - A guess from appearance carries the same information as retrieval, only less of it.
- **Explicit clues are rare or noisy on these photos (personal Flickr):**

| Explicit clue (as a constraint) | Photos | Truth allowed | Pinpoint not allowed | Of those, Pinpoint wrong |
|---|---|---|---|---|
| Country-code domain | 20 | 75% | 6 | 4 |
| Phone prefix | 1 | | 0 | |
| Language of text the 9B read | 999 | 80% | 191 | 133 (70%, vs 67% baseline) |
| Same, excluding English | 342 | 80% | 77 | 53 |

  Domains and phone numbers almost never appear. Read-text language is often wrong (20% of the time the true country isn't
  allowed: brands, foreign text), so a violation barely signals a Pinpoint mistake. Number plates are not measured yet.

**Recommendation:** 1 first. It's the cheapest, uses data already on disk, and has the largest clean ceiling with few breaks. If
it shows signal, 2 extends it to regions; 4 follows once some tool clears +1 pt on dev3k by itself.

**Options 2 and 3, first checks (2026-10-09, dev3k):**
- **Option 2, choosing among the pool's regions (`pinpoint_constrained regions`):**
  - **Setup:** Pinpoint restricted to each (country, region) in the photo's own pool; 5.6 regions per photo, 16,870 searches.
  - **Oracle:** over Pinpoint + the per-region top-1s it is 44.9%, below the pool's own oracle (50.3%). The true region often isn't
    in the pool, and keeping only one candidate per region drops good ones.
  - **Selection:** switching to the highest restricted reranker score when it beats Pinpoint's top-1: -0.1 [-0.9, +0.7]
    (1,865 switched, 66 fixes / 70 breaks). Reranker scores across regions carry nothing new.
- **Option 3, number plates (`plates.py`, 9B):**
  - **Presence:** a plate is reported on 44 of 3,000 photos (1.5%), 27 of Pinpoint's 1,997 mistakes.
  - **Country:** right 34 vs Pinpoint 36; where they differ, plate right 4 vs Pinpoint right 6.
  - **Region:** given on 5 photos.
- **Conclusion:** with options 1-3 checked, every explicit clue we can extract (text, domains, phones, language, plates) and
  every holistic one (the VLM's country, reranker scores) gives the right region less often than Pinpoint does where they
  disagree, or is too rare to matter. On dev3k (and so likely on im2gps3k / yfcc4k) a clue-based agent has about +0.3-0.5 pts
  to work with.

**Direction choices (for discussion):**
1. **Build the agent anyway, as the platform for tool use and live APIs.** Pinpoint as a tool, text search, the clue judge,
   landmarks. Accept that the tools sum to about +0.5 pts on these benchmarks; the value is the deployed system, not the
   benchmark number.
2. **Retarget the evaluation to photos where explicit clues are common** (street-level: OSV-5M's test split, Mapillary-style
   imagery, or the deployed use case). The professional-geolocator approach has something to work with there. Pinpoint's
   retriever was trained on OSV-5M, so a held-out street-level set would be needed.
3. **Look for region evidence that isn't explicit and isn't appearance:** the open research question (e.g. co-occurring map
   features near each candidate). Earlier tries (lessons 51, 55) added about 0, because they re-encoded retrieval.
