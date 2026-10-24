# Learnings (summary, 2026-10-20)

The main points of lessons 1-137 on one page. Each claim links to its lesson in the archive (`docs/archive/LESSONS_1-137.md`, every
number and module there; the previous summary is `docs/archive/LEARNINGS_2026-10-15.md`). New lessons go in `LESSONS.md` from 138.
Numbers are % of photos within 1 / 25 km of the truth.

**How to read this.** A lesson closes one implementation, not an idea. The "why" statements below are our best explanations, not laws:
an idea that has a concrete reason to escape one of them is worth a test even if a lesson looks similar. Open the archive to check the
exact setup of a prior attempt, not to rule a direction out up front.

## Where we are
The pipeline (`action_agent.py`, lessons [109], [111]): Pinpoint's reranker fine-tuned on its own data without photographer exclusion
(`setft_own`, [108]) -> candidate areas (agreement clusters + clue places) -> an answer head over area tokens, with costly actions (zoom =
Pinpoint restricted to an area, read = 4B clues + text search + clue judge, recognise = MegaLoc same-place similarity) chosen by a REINFORCE
policy. Test, eval halves, original protocol (photographer's photos kept):

| % < 1 / 25 km | both | im2gps3k | yfcc4k |
|---|---|---|---|
| Released Pinpoint | 28.9 / 46.4 | 21.8 / 48.9 | 33.5 / 44.8 |
| Pipeline, RL lambda 0 ([109]) | 29.1 / 51.2 | 22.4 / 52.8 | 33.3 / 50.2 |
| + `--one-km 0.5` ([111], [112]) | 31.6 / 50.6 | 23.2 / 52.4 | 37.1 / 49.5 |

200 / 750 / 2500 km move 0-2 pts. dev3k: released 27.4 / 42.3, pipeline 28.5 / 47.9, `--one-km` 30.8 / 47.3. commons26 test (fresh
2026 photos, continent-balanced): released 4.6 / 16.8, pipeline 6.1 / 24.2, the gain mostly from reading ([113]).

**What moved the number**, all retrieval-side (gain at 25 km, train CV / dev3k): agreement among Pinpoint's candidates +1.0 / +2.0 ([68]);
areas as hypotheses with restricted search +2.9 / +3.8 ([69]); fine-tuning Pinpoint's reranker +3.1 / +3.3 ([70], [74]); an answer head over
areas on it +4.9 / +5.4 ([71]); MegaLoc per area +0.8 / +0.5-1.1 ([93]); clue reading +0.5 / +1.0 ([76]; +4.4 on commons26, [113]).

**Where the pipeline loses** (dev3k, 47.3% < 25 km; [129], [130]):
- *Retrieval miss*, 26.5 pts: no pool candidate within 25 km. 24.6 of them are photos without the photographer's own photo in the pool
  (59% of dev3k; pipeline 33.7% there vs 66.6% with one). Weak where the gallery is sparse (< 100 photos within 25 km: 21.5%), indoors, nature.
  On commons26 dev retrieval misses are 49.1 pts.
- *Right candidate never an area*, 11.8 pts: it ranks ~9th-29th of ~46 with noise-level reranker weight (2.4% of the softmax), so no
  clustering reaches it ([122]); only better scoring or retrieval does. The reachable ceiling today is the area oracle 61.7, not the pool oracle 73.1.
- *Wrong area chosen*, 14.4 pts: wrong picks are cross-region (median 381 km from the truth), not near-misses ([125]).
- At 1 km, ~17% of the reranker's misses are coverage gaps; in the rest a photo within 1 km is in the index but ranked low ([134]).

## Main points
1. **Structure over retrieval works; re-describing the photo doesn't.** Every gain above came from aggregating, restricting or retraining
   retrieval and from same-place recognition. Choosers, comparators, attributes, map features and photo descriptions land within about
   +-1 pt ([1]-[3], [21]-[26], [41], [77], [89]), and so does every new head input built from the same embeddings or index metadata: area
   formation, hubness, geotag accuracy, area location, graph diffusion ([122], [124], [126], [127]).
2. **Look-alike verification saturates** at AUC ~0.6-0.79 for right vs wrong areas, because retrieval picked the wrong areas for looking like
   the photo. Seen with crops, round trips, more photos per area, fetched references, patch matchers, pair discriminants and diffusion along
   visual edges ([90], [97], [99]-[101], [103], [127], [135]).
3. **On MP16, text and model knowledge sit where retrieval already succeeds** ([17], [52], [53], [78], [79], [123]). On fresh photos (commons26)
   reading is the largest gain, and there the limit is *ranking* the candidates, not finding them: some span-search cluster is within 25 km
   on 39% of commons26 dev photos vs 19% for the top result ([115]-[121]).
4. **Local open models lack calibrated place knowledge beyond retrieval.** Qwen 4B to Qwen2.5-VL-72B and Gemma 3 27B, in every format tried
   (choice, evidence or reasoning first, name PMI, Wikipedia profiles in the prompt, photo-cue tables) and trained on ground truth, are near
   chance on the head's mistakes, and their confident switches are mostly breaks ([84], [88], [106], [114], [128], [131]-[133], [137]). Sonnet
   given only town names did +0.94 at 25 km with calibrated confidence ([105]), so the signal exists. Paid LLMs are ruled out.
5. **The right place's own photos look like "the same kind of scene nearby", not the same spot** ([90], [95]). External imagery is limited by
   coverage of the exact view: old Flickr photos rarely show the query's view ([135]), nor do the 20 nearest live Commons / Mapillary
   photos, even at the true spot (AUC 0.63 / 0.52; a covered view is matched cleanly, [138]). OSV-5M sits on roads, not photo spots ([136]). New
   nearby YFCC photos would fix at most 1-3% at 1 km ([134]). Same-era photos don't help either: on dev3k close calls the 50 nearest YFCC
   photos at the right candidate beat those at the wrong pick on 39% (0.44-0.47 at equal counts), and clean matches are commoner at the wrong
   pick (142).
6. **RL over today's actions only buys cost**: the policy ties "take every action", because the actions are cheap, independent and reveal fixed
   features ([94], [98]). RL on open-model free text learns silence ([87], [96]); GRPO on a 2-way one-token answer collapses (use the exact
   expected reward, [137]).
7. **Training data distribution matters**: photos Pinpoint's encoder trained on are easier, so train only on bucket 99 ([54], [75]).
   Photographer exclusion helped the photographer-excluded metric and hurt the original protocol ([74], [108]).

## Open, untested or under-tested
- **Retrieval itself** (the largest loss, 26.5 + 11.8 pts): we retrained only Pinpoint's reranker; its retriever (an MLP tower over frozen
  SigLIP2 embeddings, `pinpoint_train/src/pinpoint/contrastive_retrieval/`) and index are as released (a SigLIP2 adapter in the old setup gave
  ~+1 recall, [38]). 44.2M geotagged YFCC100M photos are not in MP16, so Pinpoint's encoder never saw them: possible training data. A
  full-index MegaLoc search was never built ([75], [95]).
- **Outcome-dependent, sequential actions** (the next check depends on the last result), the setting where an RL policy can beat fixed
  recipes; today's actions are not ([94]). Track S (lessons S1-S3 in `LESSONS.md`, closed offline 2026-10-20): constrained searches over a
  photo's own pool find right regions (net of luck ~+6 pts beyond the area oracle) that their results can't verify, since every pool region
  holds a look-alike; text-proposed places verify much better (AUC 0.94 vs 0.85) but are blind where the gallery is sparse, which is where text
  fixes the pipeline. Live imagery near the candidate doesn't verify them either ([138], `TOOL_USE_DESIGN.md`).
- **Sub-threshold gains were never stacked**: crops +0.4 ([89]), region hypotheses +0.3 ([85]), extra areas +0.3 and local refinement +0.3
  ([112]), diffusion seeds +0.4 ([127]), MegaLoc 1 km chooser +0.6 at 1 km ([107]), rare OSM names +1.4 and text override +0.9 on commons26
  dev ([118], [121]).
- **Model coverage**: thinking-on readers and geo-tuned open VLMs are untried; knowledge SFT was one 1-epoch run with the loss still falling ([52]).
- Physical constraints not tried: skyline vs elevation model, sun / shadow with capture time.

## Method rules
- Measure a change beyond the current pipeline and the pool oracle, not a weaker baseline; >= 1 pt at 25 km on dev3k (SE ~0.26) ([10]).
- Choose scorers, thresholds and combiners on dev / train; pre-declare thresholds ([41], [105]).
- Report fixes and breaks. A switch needs > 50% precision at the base rate of mistakes ([79]).
- Judges: restrict output to the answer tokens, count failures, test both A/B orders ([40], [99], [106]).
- Best-of-N ceilings are upper bounds, not forecasts ([83], [84]). A candidate list's oracle includes luck: check against another photo's
  truth ([120]). The pool oracle overstates what is reachable; use the area oracle ([130]).
- Leak guards for fetched references: drop the photo, near-duplicates and by default the photographer (yfcc4k photos are YFCC photos).

## Engineering facts
- RAM 30 GB, no swap: long jobs in a capped cgroup (CLAUDE.md). Load each `np.load` NpzFile array once (each key access re-reads it).
- Pinpoint fine-tune ~400 ms / step (CPU-bound), 6,000 steps ~65 min; training against the shipped index needs the "mp16" source alias ([70], [74]).
- vLLM 0.30: Python 3.12 headers on `C_INCLUDE_PATH`, `VLLM_USE_FLASHINFER_SAMPLER=0`, no MTP; serve with `scripts/clue_judge.sh` (`PIXELS=0` /
  `MM_KWARGS='{}'` for non-Qwen models, `MAX_LEN`, `SERVE_EXTRA`); force JSON with `response_format` json_schema; thinking off via
  `chat_template_kwargs` (HF processors: `enable_thinking=False`).
- MegaLoc: DINOv2 ViT-B/14 + SALAD projected to 8,448-d (chunks are not clusters, [100]); ~190 photos / s. Benchmark images must not go through the
  MP16 tar reader (commons26 MegaLoc was blank until bd2d12d).
- A kNN graph of all 4.12M MP16 photos takes 12 min on the L40S (`graph_diffusion.py`, [127]).
- The machine stops daily at 02:00 UTC and wipes the NVMe drive (`text_search stage`, `tools/maps` stage after a restart).

[1]: docs/archive/LESSONS_1-137.md#l1
[3]: docs/archive/LESSONS_1-137.md#l3
[10]: docs/archive/LESSONS_1-137.md#l10
[17]: docs/archive/LESSONS_1-137.md#l17
[21]: docs/archive/LESSONS_1-137.md#l21
[26]: docs/archive/LESSONS_1-137.md#l26
[38]: docs/archive/LESSONS_1-137.md#l38
[40]: docs/archive/LESSONS_1-137.md#l40
[41]: docs/archive/LESSONS_1-137.md#l41
[52]: docs/archive/LESSONS_1-137.md#l52
[53]: docs/archive/LESSONS_1-137.md#l53
[54]: docs/archive/LESSONS_1-137.md#l54
[64]: docs/archive/LESSONS_1-137.md#l64
[68]: docs/archive/LESSONS_1-137.md#l68
[69]: docs/archive/LESSONS_1-137.md#l69
[70]: docs/archive/LESSONS_1-137.md#l70
[71]: docs/archive/LESSONS_1-137.md#l71
[74]: docs/archive/LESSONS_1-137.md#l74
[75]: docs/archive/LESSONS_1-137.md#l75
[76]: docs/archive/LESSONS_1-137.md#l76
[77]: docs/archive/LESSONS_1-137.md#l77
[78]: docs/archive/LESSONS_1-137.md#l78
[79]: docs/archive/LESSONS_1-137.md#l79
[83]: docs/archive/LESSONS_1-137.md#l83
[84]: docs/archive/LESSONS_1-137.md#l84
[85]: docs/archive/LESSONS_1-137.md#l85
[87]: docs/archive/LESSONS_1-137.md#l87
[88]: docs/archive/LESSONS_1-137.md#l88
[89]: docs/archive/LESSONS_1-137.md#l89
[90]: docs/archive/LESSONS_1-137.md#l90
[93]: docs/archive/LESSONS_1-137.md#l93
[94]: docs/archive/LESSONS_1-137.md#l94
[95]: docs/archive/LESSONS_1-137.md#l95
[96]: docs/archive/LESSONS_1-137.md#l96
[97]: docs/archive/LESSONS_1-137.md#l97
[98]: docs/archive/LESSONS_1-137.md#l98
[99]: docs/archive/LESSONS_1-137.md#l99
[100]: docs/archive/LESSONS_1-137.md#l100
[101]: docs/archive/LESSONS_1-137.md#l101
[103]: docs/archive/LESSONS_1-137.md#l103
[105]: docs/archive/LESSONS_1-137.md#l105
[106]: docs/archive/LESSONS_1-137.md#l106
[107]: docs/archive/LESSONS_1-137.md#l107
[108]: docs/archive/LESSONS_1-137.md#l108
[109]: docs/archive/LESSONS_1-137.md#l109
[111]: docs/archive/LESSONS_1-137.md#l111
[112]: docs/archive/LESSONS_1-137.md#l112
[113]: docs/archive/LESSONS_1-137.md#l113
[114]: docs/archive/LESSONS_1-137.md#l114
[115]: docs/archive/LESSONS_1-137.md#l115
[118]: docs/archive/LESSONS_1-137.md#l118
[120]: docs/archive/LESSONS_1-137.md#l120
[121]: docs/archive/LESSONS_1-137.md#l121
[122]: docs/archive/LESSONS_1-137.md#l122
[123]: docs/archive/LESSONS_1-137.md#l123
[124]: docs/archive/LESSONS_1-137.md#l124
[125]: docs/archive/LESSONS_1-137.md#l125
[126]: docs/archive/LESSONS_1-137.md#l126
[127]: docs/archive/LESSONS_1-137.md#l127
[128]: docs/archive/LESSONS_1-137.md#l128
[129]: docs/archive/LESSONS_1-137.md#l129
[130]: docs/archive/LESSONS_1-137.md#l130
[131]: docs/archive/LESSONS_1-137.md#l131
[133]: docs/archive/LESSONS_1-137.md#l133
[134]: docs/archive/LESSONS_1-137.md#l134
[135]: docs/archive/LESSONS_1-137.md#l135
[136]: docs/archive/LESSONS_1-137.md#l136
[137]: docs/archive/LESSONS_1-137.md#l137
[138]: LESSONS.md
