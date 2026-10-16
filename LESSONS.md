# Lessons (condensed)

One entry per numbered lesson: the claim, the deciding numbers, the module. The full text with every number is at git tag
`lessons-full-2026-10-15` (`git show lessons-full-2026-10-15:LESSONS.md`); the summary is `LEARNINGS.md`. Add new lessons here in the same
form (2-4 lines). Modules no longer in the tree: lessons 1-51 at tag `pre-cleanup-2026-10-07`, 1-53 (one-step reranker) at commit 57918cb,
55-67 and 77-79 at tag `pre-tool-consolidation`; plans cited (AGENT_DIRECTION, HYPOTHESIS_PLAN, PINPOINT_SET_PLAN, TEXT_SEARCH_PLAN) at tag
`plans-2026-10-15`.

Read with these corrections:
- **Lessons 1-53 used the old reference**: our one-step reranker (top-1 16.7 / 38.2 % within 1 / 25 km on the eval halves) and its ~17-candidate
  pool, not Pinpoint's reranker (lesson 54). Their gains are relative to that baseline and were never re-measured.
- **"val" in lessons 8-55 is 1,000 benchmark eval-half photos**, not MP16 (found in lesson 55): numbers fitted on "dev + val" touched test photos.
- **Photographer protocol**: lessons 54-97 exclude the query photographer's photos from the index; since 2026-10-15 the primary benchmark keeps
  them (the original Pinpoint protocol, lesson 98).
- **No paid LLMs** (user, 2026-10-15): Gemini (lessons 6-12) is a benchmark only, and the Claude readers of lessons 80 and 102-105 are diagnostics,
  not pipeline components.

## A. Choosing among one-shot retrieval results with the 4B (Sep 2026; old reference)
1. **Selection is saturated**: SFT and GRPO land at reranker parity; model and reranker agree on > 90% of photos.
2. **SFT reaches parity, not beyond**, mostly by learning to copy the reranker (#1 copied 41 -> 67%); 7x more data didn't help.
3. **GRPO sharpens, doesn't discover**: best-of-8 -> greedy +3-4; high lr / no KL collapses entropy.
4. **The VLM's edge is readable text** (64.9 vs 60.9 < 25 km there); it loses on generic photos; off-list guesses are almost always wrong.
5. **Where to search isn't the bottleneck**: retrieval inside the VLM's region scores below the VLM's own answer.
6. **Offline geocoding needs fuzzy matching** (exact GeoNames matches 55% of confident names).
7. **Rewriting the query (text, crops) adds no missing candidates**: < 2 pts of oracle; the gap is choosing.
8. **Per-candidate evidence (landmarks, exemplar photos) helps an untrained chooser ~+1.3.**
9. **The cap is the chooser's knowledge**: Gemini Flash closes ~2/3 of the gap the 4B can't (300 subset).
10. **Method: go/no-go against what you already have** (the shown-list oracle, not greedy); decide on full sets, not the 300 subset.
11. **Base-4B search queries + evidence photos don't help** (queries re-fetch the candidates; the model ignores right evidence).
12. **Geotagged Wikipedia is a good backend, the 4B's queries are the limit**: Gemini's names +7.3 oracle pts, the 4B's +1; 27B ~2x the 4B.
13. **Stage 1: extra retrieval over the pool adds ~1 pt; the 4B's queries no more than extra whole-image hits.**
14. **Search evidence discriminates (AUC 0.58-0.61) but adds ~0 to a combiner.**
15. **A learned chooser can't use it either** (all within +-0.5; shuffled control equal).
16. **A 27B searcher doubles the oracle gain (+2-3) but a simple chooser still gains nothing.**
17. **Why: evidence is right where retrieval is already right** (near the truth on 33-43% of pool-hit photos vs 3-7% of misses).
18. **Text screen fails**: 27% of photos have text, the same share where retrieval fails; mostly generic strings.
19. **An 80M-place name index (Overture) doesn't rescue text**: few choosing photos have it (cap ~2.7 pts, no signal).
20. **Set audit**: clean; flaws = Flickr "unavailable" placeholders (3% of yfcc4k) and ~4-5% near-duplicates (lift everyone alike).
21. **VLM photo attributes are accurate but redundant**: wrong candidates match the photo's attributes as well as right ones.
22. **Zero-shot "same place?" judge: informative per candidate, no top-1 gain** (+0.1-0.2).
23. **Its skill is mostly exemplar similarity**: adds nothing beyond similarity.
24. **Fine-tuned comparator (query + exemplar -> same place?): first small positive**, +0.6-0.8 at 25 km (`comparator-a`).
25. **Pairwise and other judge variants don't beat it.**
26. **Training it for 25 km doesn't either**; the signal saturates ~+0.5-0.7.
27. **Model size alone doesn't make a zero-shot chooser beat the reranker** (4B/9B -5, 27B -2 to -3).
28. **Nearby Wikipedia text per candidate helps zero-shot choosers a little, not enough.**
29. **Comparator gain holds on the benchmarks**: +0.7 [+0.3, +1.2] at 25 km (im2gps3k +1.4, yfcc4k +0.3).
30. **The gallery isn't thin; going deeper than the pool doesn't help** (deep recall sits where precision is lowest).
31. **Region-diversified pools add ~2 oracle pts; region classification is the bottleneck.**
32. **Bigger region heads / more data barely move region accuracy** (~43% top-1 from frozen SigLIP2).
33. **The models' own answers add nothing beyond reranker + comparator** (learnable headroom ~+0.7).
34. **Kinds of places near a candidate (Overture categories) don't discriminate.**
35. **By eye, ~20% of the choosing gap is recoverable from the photo** (~2.5-3 pts): near-misses and generic photos dominate.
36. **Coarse thresholds have as much headroom; our evidence doesn't help there either.**
37. **The 4B's own place guess is far weaker than retrieval at every scale** (25 km 19.4 vs 38.8).
38. **A geo-aware adapter on SigLIP2 adds ~+1 pt of candidate recall** (look-alike-only negatives are a trap).
39. **Several exemplars per candidate lift the comparator a little** (+0.3-0.6, all on im2gps3k).
40. **3x pairs with several exemplars (`comparator-b`) gains ~+0.3**; scoring trap: restrict output to the answer tokens, check failed counts.
41. **Benchmark test of `comparator-b`: +0.9 [+0.3, +1.6]** (combiner trained on dev); the dev-chosen rule +0.4. Comparator line ends ~+1 pt.
42. **Wikimedia set: harder, same headroom, comparator gives nothing** (set not on this machine).
43. **Near-misses (1-25 km) are the largest target; feature-only local refinement gets ~+0.8 at 1 km.** Trap: index NpzFile arrays once.
44. **A near-band comparator (`comparator-c`) raises AUC to ~0.8 but 1 km moves < 1 pt.**
45. **4x near-band data (`comparator-d`) doesn't move it.**
46. **Better combiners don't help: the signal is the limit.**
47. **Map search reaches the truth (70-82% within 100-300 photos) but nothing tells it apart** (keypoints match *what*, not *where*).
48. **4 exemplars per local candidate: AUC 0.84, top-1 still ~+1**; comparator line closed.
49. **No verifier separates the right place among the search's look-alikes** (AUC ~0.55-0.6): map search has nothing to steer by.
50. **Knowledge-grounded choosing: the base 4B knows too little** (name-score AUC 0.50-0.53); labels from Overture divisions.
51. **Concept queries as similarity scores around candidates are redundant.**
52. **Knowledge SFT (`knowledge-a`) adds country / region knowledge, but not where retrieval fails**; true-region knowledge would be worth +10.8.
53. **Retrieval's failures are hard for the VLM too, not under-covered** in its training data.

## B. Pinpoint's reranker as the reference; clues and tools (Oct 7-9)
54. **Pinpoint's attention reranker is the reference** (exact port, `models/pinpoint_reranker.py`). Eval halves top-1 / pool oracle, 1 / 25 km:
    photographer excluded 17.5 / 37.2, oracle 30.1 / 54.7; kept (original protocol) 28.9 / 46.4 (yfcc4k 33.5 / 44.8; im2gps3k unchanged:
    no shared photographers). Its index favours photos its retriever trained on, so training stays on bucket 99. Tie order matters.
55. **Constraint tools from photo clues are precise but rare**: text matched to GeoNames near a candidate +0.2 (clean); "val" correction.
56. **Capacity / resolution barely move region knowledge on hard photos** (9B +2.9 region; 786k px helps the 4B a little).
57. **What retrieval's failures contain**: nothing place-specific 58%, business names ~15%, landmarks 12%, cultural objects 12%.
58. **Landmark / object identification by crop search over MP16 doesn't work** (hits are the same kind of thing elsewhere).
59. **A 27B names more landmarks but too few right ones** (city right 1.7% of mistakes): +0.2.
60. **A DINOv2 instance descriptor barely helps landmark search** (+0.2).
61. **Deciding by place doesn't rescue crop search.**
62. **The landmark / text target exists (~5% / ~7.5% of mistakes); recognition and Pinpoint's pool cap the tool.**
63. **YFCC100M names extracted** (`data/yfcc100m/geo.npz`, 48.5M geotagged photos); they locate the same few photos (+0.1).
64. **Pinpoint restricted to the true region: 33.4 -> 61.3% < 25 km (+27.9)**; true country +7.8. A clue only has to name the region.
65. **Text search gives the right region too rarely**; a VLM clue judge fixes precision but leaves +0.3.
66. **A fine-tuned clue judge lifts read-text clues to +0.7, then stalls** (the clues are the limit).
67. **Locatability: 47% of photos carry nothing place-specific**; ceiling from content + retrieval ~42-48% < 25 km. Physical-constraint
    elimination dropped (the top-1 and the right candidate essentially never differ physically).
68. **Agreement among Pinpoint's candidates: +1.0 train / +2.0 dev3k** (`pool_agreement.py`); the first clean +1.
69. **Hypothesis testing: +2.9 / +3.5 train, +3.8 / +4.6 dev3k** (areas + clue places + restricted search + learned set comparator).
    Remaining misses: the right hypothesis has the weaker retrieval evidence (~70%).

## C. Look-zoom-rerank and the merged pipeline (Oct 12-13)
70. **Retraining Pinpoint's reranker is the lever: +3.1 / +3.3** (train CV / dev3k; wide pools, photographer exclusion, set head;
    `pinpoint_train/`); area model over zoomed states +4.4 / +4.9; RL zoom agent: 2 zooms = the full gain. Trap: source alias for leave-one-out.
71. **Merged pipeline (`hypothesis_agent.py`): areas on the retrained reranker +4.9 / +5.4**; the zoom policy adds +0-0.3 over a fixed rule.
72. **Soft kernels instead of the agreement rule are free; learned location distributions don't help** (~10-pt coverage gap not recoverable).
73. **Merged pipeline on the test set: 42.1% < 25 km, +4.9** (photographer excluded); ties the previous pipeline. With the gated pick
    (lesson 74): +4.8 at 25 km, +0.8 at 1 km.
74. **Gated final pick (kernel peak within 5 km of the founder) is the default**; longer lists don't help (ranking, not length); the reranker
    fine-tune saturates at ~+3 (photographer exclusion is its key ingredient; wide lists without it +0.3 / +0.5). ~400 ms / step, CPU-bound.
75. **Cross-fitted labels don't help**: photos Pinpoint's encoder trained on are easier than evaluation; only bucket 99 is in-distribution.
76. **Clue reading as a costly action** (`read_agent.py`): always read +0.5 / +1.0; a learned read policy keeps +0.5 / +0.8 with 20% of reads.
    What's in the photo decides whether reading pays, not retrieval confidence.

## D. Tools, readers and evidence beyond retrieval (Oct 13-15)
77. **Map checks of features the 4B sees add nothing**: Pinpoint's wrong areas have them too (beach 92% right vs 78% wrong).
78. **OSM name search inside areas is redundant** with the text-search clue evidence (fixable ceiling 0.4-0.8%).
79. **Names as a candidate generator: oracle +1.3, learned switch +0**; hard constraints (domains, phones, co-located names) are right where
    retrieval is already right.
80. **A stronger reader (Claude, 40 photos where no area is right) hits 4 / 40 within 25 km vs 0 for the 4B**, mostly by recognition and
    regional knowledge, not text; every name it produced resolves in our corpus. (Diagnostic; no paid LLMs.)
81. **Tools consolidated into `tools/`** (pinpoint, maps, web); landmark matching and the map / name checks archived.
82. **Gate for query RL**: novel pass@16 9.7% / pass@1 2% on fixable photos (~3 pts ceiling, ~0.4 at pass@1).
83. **Nested country / region / place proposals: best of 16 +6.1, single sample -19.6**; verification is the bottleneck.
84. **Retrieval proposes regions far better than the 4B** (best of 16 +19.0 vs +6.1); the 4B is confidently wrong where it disagrees.
85. **Region candidates + map / text checks in the answer head: +0.3 at most**; the head gets 38.7% where the best of its areas is 51.7%.
86. **The 4B reading each area's YFCC titles / tags picks near chance** (21.7% vs top area 36.2%; position bias).
87. **GRPO on the 4B's clue / region output learns silence, not precision** (pilot -0.37, zero-shot -1.13). Full run `clue-grpo-full`
    stopped at step 950 (reward flat).
88. **Qwen3.8-27B-FP8 as the clue reader finds fewer right names / countries than the 4B** (3.6 vs 5.1% of top-1 mistakes).
89. **Crop views vs each area's photos add nothing** (+0.0 / +0.4, `crop_look.py`).
90. **Round-trip (reverse-rank) verification: AUC 0.66 vs 0.655 forward**; the right area's photos are "same kind of scene nearby".
91. **Refining to the next distance bucket: oracle +11-17 pts, local chooser +-0.1** (signal at chance).
92. **YFCC word-location statistics for read text: +0.1** over the existing clue features.
93. **MegaLoc (visual place recognition) per area: +0.8-1.0 train CV, +0.5-1.1 dev3k** in the answer head (`megaloc_gate.py`); the largest
    evidence gain since lesson 70.
94. **RL action policy over zoom / read / recognise with costs** (`action_agent.py`): at or above every fixed combination; best 39.7% dev3k
    (+6.3 over Pinpoint). Per unit, recognise is the most efficient action. **Current best pipeline.**
95. **MegaLoc as a proposer fails** (right new places on ~1% of photos): a verifier only.
96. **GRPO on free-text search queries learns silence** (precision 26 -> 64%, new right places 0.6 -> 0.3%).
97. **MegaLoc over 64 photos per area instead of 12: +0.1**; inside an area it is worse than the restricted top-1 at 1 km (15.5 vs 17.0%).
98. **Action agent on the test set** (`action_agent --apply`): photographer excluded 43.5% < 25 km (+6.2), 19.1% < 1 km (+1.6); original
    protocol (`PINPOINT_KEEP_OWN=1`, `setft_aown`) 48.2-48.7% (+1.8 to +2.3) but 1 km 25.5-28.6% vs 28.9 (yfcc4k 27.3 vs 33.5): the pipeline
    was trained with the photographer excluded. The RL policy ties take-everything (43.5 vs 43.4): its value is cost.

## E. Choosing between the head's top two areas like an expert (Oct 15; `EXPERT_PLAN.md`)
Setup for 99-106: the answer head's top two areas; "decisive" = exactly one within 25 km (200 of 500 dev3k pairs; head right on 85%).
99. **Contrast with fetched YFCC references, 4B reader** (`contrast_gate.py`): position-biased (first place 81%), 67 vs 83% for the head on
    consistent decisive pairs; fetched references beat Pinpoint's own photos as evidence (67 vs 57%).
100. **Our own comparator over MegaLoc matches to fetched references adds nothing** (AUC 0.821 vs 0.819; `contrast_model.py`). The "cluster"
    cosines first reported were chunks of a linearly projected descriptor, i.e. global matching only.
101. **A learned patch-level matcher (frozen MegaLoc DINOv2 patches, late interaction) adds nothing**: stacked weight ~0, end to end -1.6
    (`patch_matcher.py`).
102. **Sonnet with anonymous reference photos**: right 12 / 20 where the head is wrong, 17 / 20 where right; confident only on landmarks the head
    already gets. (Diagnostic.)
103. **A pair-specific discriminant between the two areas' photos adds nothing** (AUC 0.62 alone vs similarity 0.58; `pair_discriminant.py`).
104. **Sonnet told the candidates' town names** (not photos): hard constraints are rare (2 / 20 far mistakes); at confidence >= 0.7, 6 fixes /
    0 breaks on 20 + 20 train pairs at all distances. (Diagnostic.)
105. **The same on dev3k (202 pairs): 8 fixes / 3 breaks at >= 0.7, +0.94 pts at 25 km** (p ~ 0.11); at chance on the head's mistakes overall,
    the gain is its calibrated confident cases. The first gain beyond the head since lesson 94, but from a paid LLM (now ruled out).
106. **Local readers can't do it** (`names_judge.py`, same 202 pairs, first-token P(A), both orders): the 4B is right on 55% of decisive pairs
    (head 85%) and position-biased (62%); Qwen3.8-27B-FP8 60.5%, unbiased. At every confidence threshold their switches are mostly breaks
    (best: 4B -0.4, 27B -0.9 pts). Evidence first + stated confidence (Sonnet's format, `NAME=27b_reason`): 71.4% overall, but right on
    only 37% of the head's mistakes; confident switches 7 fixes / 22 breaks (-2.8). Calibration, not just accuracy, is what Sonnet had.
107. **MegaLoc local chooser for 1 km** (`refine_gate megaloc` + `gate --radius 25`; dev3k, Pinpoint restricted to 25 km around the
    retrained top-1, ~43 candidates; top-1 < 1 km 17.9%, list oracle 29.9%): MegaLoc cosine per candidate (+ relative to the list's best,
    best match among candidates within 1 km, MegaLoc-weighted density). Reward-trained listwise chooser (2-fold x 3 seeds): linear +0.33 /
    +0.90 / +0.70 pts at 1 km (SE ~0.3; 49-54 fixes vs 27-43 breaks), MLP +0.40 / +0.77 / +0.80; the same choosers without MegaLoc +0.0-0.3
    (lesson 91). 25 km unchanged. MegaLoc alone is worse than the reranker (argmax 15.7% vs 17.9%; switching on its lead nets negative).
    Real but below the 1-pt bar: ~+0.6 at 1 km, measured against Pinpoint's top-1, not the action agent's answer.
108. **Original protocol on dev3k; retrained reranker without photographer exclusion** (`PINPOINT_KEEP_OWN=1` now drops only the query photo
    for MP16 queries; run `setft_own` = setft_a's recipe with `EXCL=0`). dev3k top-1 within 1 / 25 km under the original protocol: released
    27.4 / 42.3 (test: 28.9 / 46.4, so dev3k is a fair proxy); setft_a 25.1 / 41.3 (-2.3 / -1.0): trained with the photographer masked, it
    passes over the photographer's own near-identical photos (top-1 own 13.9 vs 17.2%; on the 132 photos where only released picks one,
    7.6 vs 73.5% within 1 km), though it is still +0.5 / +1.3 on the other photos. setft_own wide pool 28.3 / 44.3: +0.9 [+0.1, +1.7] /
    +2.0 [+1.2, +2.8] over released. The base for retraining the pipeline under the primary protocol.
    Pipeline stages on setft_own (original protocol, no MegaLoc / RL yet; `hypothesis_agent --ckpt-tag setft_own`): dev3k top-1 44.3 / 28.3,
    area oracle (K = 5) 60.3%; answer head with all zooms 46.1 / 29.0 (set pick) or 46.7 / 27.6 (agree pick) = +3.8 / +1.6 or +4.4 / +0.2
    over released under the same protocol (42.3 / 27.4). The script's "released Pinpoint" line is the photographer-excluded baseline.
109. **Test set, original protocol, pipeline retrained without photographer exclusion** (`action_agent --ckpt-tag setft_own --apply
    full:setft_own`, the user asked 2026-10-16; trained on all of train). Both halves, % within 1 / 25 / 200 / 750 / 2500 km: released
    28.9 / 46.4 / 60.7 / 75.5 / 86.7; setft_own top-1 29.6 / 47.2 / 61.3 / 75.2 / 86.9; RL lambda 0 29.1 / 51.2 / 62.9 / 75.5 / 86.9
    (+0.2 / +4.8 [+3.9, +5.8]); RL lambda 0.005 30.9 / 50.5 / 62.9 / 76.0 / 87.0 (+2.0 [+1.2, +2.8] / +4.1). im2gps3k lambda 0
    22.4 / 52.8 / 66.4 / 78.9 / 90.3 (+4.0 at 25 km); yfcc4k 33.3 / 50.2 / 60.6 / 73.3 / 84.8 (+5.3; lambda 0.005 35.9 at 1 km, +2.4).
    vs lesson 98 (setft_a, same protocol): 25 km 48.2 -> 51.2, 1 km 25.5 -> 29.1-30.9; the yfcc4k 1 km loss is gone. Zooms move answers off
    exact matches (1 km falls with zooms); recognise (MegaLoc) is the action that helps at both distances. dev3k: lambda 0 28.5 / 47.9, 0.005 29.0 / 47.1.
110. **Error analysis of the lesson 109 system on dev3k** (original protocol, RL lambda 0, 47.9% < 25 km). Misses (52.1%): D nothing within
    25 km retrieved 25.9% of dev3k (63% of them > 750 km off; by eye mostly people, pets, macros, sky, interiors); A a right area exists but
    the head picks another 13.8% (the right one 2nd-5th in 81%); C a right candidate in the pool but in none of the <= 5 areas 10.4%
    (median rank 18); B right candidate inside a wrong-centred area 2.1%. Place names the 4B already reads geocode within 25 km on 0.5% of
    misses vs 10% of hits. Right own-photographer candidates exist in 43% of A and 35% of C, but author agreement in the pool adds nothing
    beyond the reranker (re-rank on train: 44.3 -> 44.2-44.4% on dev3k). 1 km: vs the retrained top-1 the pipeline fixes 129 and breaks
    123 (72% of breaks have the photographer's own photo in the pool); best-of(pipeline, top-1) would be 32.6 vs 28.5%; a confidence gate
    on the top-1's share recovers +0.3.
111. **1 km-aware answer and extra-area ceiling** (dev3k, original protocol, setft_own). `action_agent --one-km W`: the answer picks an area
    and one of its two points (the global pool's best candidate near the area, often the exact match, or the zoomed kernel peak); the
    answer loss and the RL return add W x [< 1 km]. dev3k % < 25 / 1 km: lesson 109 lambda 0 47.9 / 28.5, lambda 0.005 47.1 / 29.0;
    W = 0.5 lambda 0 **47.3 / 30.8** (-0.6 / +2.3), lambda 0.005 47.0 / 30.2; W = 1 lambda 0 46.9 / 30.6, 0.005 46.9 / 30.6 (best-of oracle
    32.6 at 1 km, lesson 110). Extra areas for C errors (ceiling): seeding 1 / 2 / 3 areas at the top-scored pool candidates outside every
    area adds a right area on 1.9 / 3.7 / 4.7% of dev3k misses (MegaLoc-ranked seeds 1.4 / 2.0 / 2.6%); with the head converting low-ranked
    areas rarely, expect ~+0.5-1 pt; not built.
112. **Lesson 111 on the test set, extra areas, MegaLoc local refinement** (original protocol, setft_own). Test (eval halves), W = 0.5
    lambda 0: both 31.6 / 50.6 / 62.7 / 75.1 / 86.6 (vs released +2.7 [+1.9, +3.6] at 1 km, +4.2 [+3.3, +5.2] at 25 km; lesson 109 29.1 / 51.2);
    im2gps3k 23.2 / 52.4; yfcc4k 37.1 / 49.5 (+3.6 / +4.7). dev3k % < 25 / 1 km, reference W = 0.5 lambda 0 47.3 / 30.8:
    - extra areas (`hypotheses run --k 8 --min-rank 5`, `HYP_VARIANT=more8`: Pinpoint areas ranked 5-7 through restricted search, features,
      zoom, MegaLoc): RL lambda 0 47.6 / 30.7 (+0.3 / -0.1), lambda 0.005 46.7 / 29.9; fixed "everything" 47.6 / 31.0; train CV +0.3.
    - MegaLoc local refinement of each area's two answer points (`local_refine.py`: candidates within 25 km of the point, listwise MLP over
      score / density / MegaLoc columns, fitted on 12k train photos; option level +0.63 at 1 km; `action_agent --refine`): RL lambda 0
      47.2 / 31.1 (-0.1 / +0.3), lambda 0.005 46.9 / 30.9.
    Both below the 1-pt bar; not adopted. Lessons 110-112: the remaining misses are retrieval's (D) and the head's ranking of areas it has (A).
113. **commons26** (5,023 Commons photos taken in 2026; dev 977 / test 4,046 by uploader; scorer weights the six continents equally,
    uploader-clustered SEs; `pinpoint_reference commons26`, `photo_sets select --source commons26`, then the test chain with tag `commons26`;
    `scripts/commons26_score_km.py`). No Overture divisions on this machine, so the clue judge keeps every clue (no truth-region filter).
    Released Pinpoint, test, continent-balanced % < 1 / 25 / 200 / 750 / 2500 km: 4.6 / 16.8 / 33.7 / 58.4 / 78.1 (dev 6.0 / 21.5 / 38.2 /
    62.7 / 78.3): no same-place photos in MP16. Lesson 111 pipeline (setft_own, W = 0.5, RL lambda 0), test **6.1 / 24.2 / 39.2 / 60.8 / 78.4**
    (+1.5 / +7.4 / +5.5 / +2.4 / +0.3; plain mean 5.9 / 23.9 vs 4.7 / 17.5), dev 7.2 / 26.7 / 42.0 / 63.1 / 79.4; retrained top-1 alone
    18.1 at 25 km. The gain is reading: read alone +4.4 at 25 km, recognise (MegaLoc) +0.0, zooms +0.1-0.5 (all 5,023 photos). By the 27B's
    tier at 25 km: landmark 25.3 -> 36.6, city 26.0 -> 40.7, region 16.6 -> 19.4, none 6.9 -> 8.3.
114. **27B named-area choice as head evidence: +0.2** (`names_feature.py`, `scripts/names_gate.py`; dev3k, setft_own). Qwen3.8-27B-FP8 sees
    the photo and all of its areas named "near <town> (lat, lon), country" and picks a letter (first-token P, both orders averaged; ~2.4
    photos/s). Its pick is right (centre < 25 km) on 18.3% of photos with > 1 area vs first area 39.5% (any 59.1%); position-biased toward
    B / C, orders agree on 41%. Signal sits on the top area (P right 0.40 vs wrong 0.16 at rank 0), little below it (0.29 vs 0.24 at rank 2).
    Listwise linear chooser, 5-fold CV x 3 seeds on dev3k: area features 44.8-45.0%, + MegaLoc 46.9-47.0, + MegaLoc + names 47.0-47.2.
    Stopped during the train pass (6.3k of 66k answers kept in names_train_setft_own_27b.jsonl; resumable) at the user's request.
115. **Where reading loses on commons26 dev** (977 photos; `scripts/commons26_read_funnel.py`, `scripts/commons26_wiki_probe.py`). The 4B reads text /
    names on 52%, writes queries on 90%; a text-search result < 25 km exists for 19.0%; a right clue judged yes 10.7%; final 24.8% (retrained
    top-1 19.7%). Photos whose text names the place (place_text, 29%): read 98%, a right result only 47.5% (~15% of dev lost at search);
    found-but-lost 5.2% of dev (judge said no on 25 of 51, never judged 9, judged yes but not chosen 17). Precision of a search < 25 km:
    read place names 33%, queries / text ~5%. The 4B's country guess is right on 53% (place_text 76%). Offline Wikipedia (25 languages,
    top-3 hits with coordinates, same names + queries) is redundant: a hit < 25 km on 56.5% of photos the read chain already finds, 0.9% of
    the rest; tier "none" 0.4% findable by any search. The limit is the queries (misreads, ambiguous common names), not the search backend.
116. **Search recall is limited by literal phrase matching** (`read_constrain.py`, `scripts/commons26_span_rules.py`; commons26 dev). text_search needs
    every word of a phrase, so the 4B's descriptive queries ("Karel Hynek Mácha monument Prague") return nothing. Choosing among all 8
    clusters of each phrase: oracle 24.3% vs top 19.0%; by the 4B's country 19.3, Pinpoint area countries 19.9, near an area 20.1. Searching
    every 1-3 word span (not only generic words; 29.7k searches for 884 photos): some span's top cluster < 25 km on 24.9% (place_text 55.0 vs
    47.5), any span cluster 39.4% (place_text 73.4 vs 58.2, landmark 62.7 vs 46.7). A hand-made cross-span agreement pick is worse than
    the phrase top (15.9%): the recall is there, the ranking isn't.
117. **OSM (tools.maps) on commons26 dev** (`scripts/commons26_osm_probe.py global`; OSM names + GeoNames + Overture, top 10, no Flickr). Some result
    < 25 km: phrases 20.4% (top result 15.5%), phrases + spans 31.4%, vs text_search 24.3% / 39.4%; OSM finds where text_search (phrases +
    spans) doesn't on 4.0% (place_text 5.0%); either 43.4% (place_text 78.4%). Not in the read chain so far because on MP16 it was
    redundant (lessons 77-79); on fresh Commons photos it adds recall, but the gap remains ranking.
118. **OSM names near Pinpoint's areas as an area check** (`scripts/commons26_osm_near.py`, `_report.py`; commons26 dev). Fast form: one global FTS lookup
    per phrase / span (OSM + places.db, <= 2,000 matches each), then keep matches < 25 km of each area: 95 s for dev vs ~55 min for per-area
    `maps.query(near=)` (box scans for common names). Area 1 right 21.0%, any area 35.8%. A name matching <= 20 places worldwide found near an
    area ranked 2+: right 12.6% vs 3.4% without (n 269 / 3,463). Rule "most-supported area by names with <= 5 matches, else area 1": +1.4 pts
    of dev (18% of photos flagged); looser thresholds lose (<= 100: -0.1, <= 4,000: -2.8). Not yet tested beyond the head's clue features.
119. **Learned ranker over four-source text-search candidates** (`candidate_ranker.py`; commons26 dev, grouped 5-fold CV by uploader). Every 4B phrase
    and 1-3 word span searched separately in GeoNames, Overture, Flickr titles / tags and OSM names (30.9k queries, 9 min), clustered per (query,
    source), merged per photo within 25 km, top 64 by support kept (some candidate right 35.0%; uncapped ~43-47%). Listwise ranker with a 'none'
    option over 37 features (query specificity, agreement across queries / sources, Pinpoint areas and mass, 4B country, town population):
    top candidate right 20.7% linear / 19.1% MLP vs phrase top cluster 19.0% and a support heuristic 15.4%; well calibrated (confidence >= 0.7:
    86 photos, 84% right). But its right picks are the pipeline's: top right & pipeline wrong 4.8% of dev, pipeline right & top wrong 8.9%.
    Confidence switch (threshold from other folds): 24.8 -> 25.5% (+0.7, SE 0.7; landmark +3.3, city +3.8, region -1.6). Source ablations:
    dropping Overture / Flickr / OSM each costs ~1.3-1.5 pts of top-1 and ~1 pt of switch gain; GeoNames is redundant (its names are in Overture / OSM).
    The candidates that would fix the pipeline (16.5% of dev under the cap) are not the ones a ranker can pick without knowing the pipeline's answer.
120. **Why the candidate ranker (lesson 119) barely helps** (`scripts/commons26_ranker_diag.py`; commons26 dev). (1) Much of the headroom was luck:
    with all ~200 candidates per photo, some candidate lies < 25 km of a *random other photo's* truth on 12.2% (tier none 9.7 vs 15.5% for the
    real truth); the 64 cap cuts chance to 3.7% while real recall drops 45.5 -> 35.0% (net of chance ~33 vs ~31). (2) On the 16.5% of photos
    with a right candidate and a wrong pipeline answer, the right candidate's evidence is weaker than the ranker's wrong top on every feature
    (queries 1.5 vs 2.0 log, sources 2.5 vs 3.2, top-of-a-query 57 vs 89%, near area 1 4 vs 44%, Pinpoint mass 0.04 vs 0.30); it ranks 1st on
    29% (86% on pipeline-right photos), median P 0.02. The ranker correctly learns that Pinpoint agreement + many agreeing queries = right, which
    the pipeline already knows. (3) Rare matches don't separate: 52% of right candidates have a query with <= 5 matches, but so do wrong tops
    (dates, single Flickr titles, generic descriptions). (4) Not mostly geometry: the wrong top is 25-100 km off on 19%, 100-300 km on 35%.
    The fixable photos are ones where text evidence is thin and contradicts retrieval; a better ranker can't find them, better reading or
    evidence that the pipeline is wrong (its own confidence) might.
121. **Text override with the pipeline's own confidence: below the bar; offline text-search path closed** (`candidate_ranker.py override`;
    `action_agent --apply` now also saves each answer's probability and area centre, results unchanged). commons26 dev: the ranker's top
    candidate disagrees with the pipeline's area on 56.7% of photos; there it is right 7.6% vs the pipeline 14.1%; a perfect switch reaches
    24.8 -> 29.0% (+4.2, 41 photos). The pipeline's confidence doesn't locate the fixes (where it is < 0.3, 549 photos, the pipeline is right
    9.1% and a disagreeing ranker top 6.1%). Switches, thresholds from the other folds: ranker conf > a and pipeline conf < b +0.9 (SE 0.6; 23
    fixes / 14 breaks), logistic over (ranker conf, pipeline conf, text support of the pipeline's area, distance, Pinpoint mass) +0.5, ranker
    conf alone +0.9. Below the agreed +1 stop rule; not tested on dev3k. The λ0 policy reads 73% of commons26 photos (lesson 113's +4.4 comes from those).
