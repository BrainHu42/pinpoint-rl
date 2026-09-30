# Multi-Corpus Candidate-Location Experiment

## Question

Does OSV-5M surface useful geographic hypotheses that complement the frozen
Pinpoint MP16-Pro retrieval baseline before any search-agent training?

This is an offline acquisition diagnostic. It does not use live Mapillary and
does not expose retrieved corpus images to a policy.

## Comparison

Use the existing 10-query calibration and 40-query report cohort. Generate
location proposals from:

1. MP16-Pro retrieval;
2. OSV-5M retrieval through the frozen Pinpoint image tower; and
3. a source-alternating union of both.

Raw retrieval hits are clustered within 1 km. Each public result contains only
a coordinate, source, source-specific rank and score, support count, and
cluster dispersion. It contains no retrieved image, thumbnail, or asset ID.

Evaluate each source at 10, 25, and 50 candidate locations. Measure candidate
recall within 1 km, 5 km, and 25 km of private truth, plus whether a candidate
makes independently eligible OSV-5M evidence reachable within a 5 km local
search.

## Integrity controls

- Remove all assets from pilot-query sequences from the OSV-5M retrieval world.
- Retain the prior feasibility pilot's duplicate and same-capture exclusions.
- Do not compare MP16-Pro and OSV-5M similarity scores during fusion.
- Keep coordinates and eligibility out of retrieval scoring.
- Store private truth metrics separately from public candidate-location cards.

## Gate

Proceed to a fixed location-search policy only if the combined top-50 proposal
set makes evidence reachable for at least 30% of queries with available
independent evidence and improves on MP16-Pro top-50 by at least 10 percentage
points.
