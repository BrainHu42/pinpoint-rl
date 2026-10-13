# Directions as of 2026-10-05 (moved from LESSONS.md; superseded by archive/KNOWLEDGE_PLAN.md, then AGENT_DIRECTION.md)

1. **Geometric verification as a tool** (never tried). Keypoint matching + RANSAC inliers (SuperPoint / DISK + LightGlue, pretrained, no training) between the
   query and 2-4 exemplars per local candidate. Near-binary when the same structure is visible and silent otherwise, which is what flipping near-misses without
   breaking exact photos needs (today ~8 fixes per 5 breaks). Go test, a few hours on dev: within-photo AUC on the subset where it fires, and the 1 km gain of
   the lesson-46 combiner with inlier features; go if > +2 at 1 km on the benchmarks. Weak on nature / indoor / no-overlap photos.
2. **Candidate recall in the near band** (pairs with 1). Search the whole gallery within 25 km of the top-1, not the top-300 neighbours: the ceiling goes from
   59% to up to 93% of near-misses. Only worth it with a precise scorer, since a noisy one breaks more as candidates grow.
3. **RL over tools** (the project's framing, still untested for stage 2): a policy that decides which candidates to verify (comparator, matcher, search) under a
   budget and when to keep the top-1. Do it after 1 gives a signal stronger than AUC 0.8; RL cannot create a signal the tools don't have.
4. **Scale the comparator** (9B / 27B LoRA) only if 1-3 show the 4B pipeline is the bottleneck.
Pending when this was written: 4-exemplar scoring of the local candidates with comparator-d (`scripts/near_multi_exemplar.sh`, log
`artifacts/query_evidence/logs/near_multi_exemplar.log`, results `artifacts/query_evidence/near_miss_combiners_comparator-d.json`), queued behind other GPU jobs.
Expected +1.0 to +1.4 at 1 km; below +1.3 closes the comparator line.
Not promising (tested): more comparator data, combiner tweaks, zero-shot choosers up to 27B, new text / attribute evidence (stage 1), deeper pools.
