# Multi-Corpus Candidate-Location Results

## Outcome

The offline acquisition gate passed. Candidate locations derived from OSV-5M
made independent evidence reachable much more often than locations from the
frozen MP16-Pro baseline.

| Source | Location budget | Evidence reachable within 5 km |
| --- | ---: | ---: |
| MP16-Pro | 10 | 0 / 20 (0%) |
| MP16-Pro | 25 | 1 / 20 (5%) |
| MP16-Pro | 50 | 3 / 20 (15%) |
| OSV-5M | 10 | 3 / 20 (15%) |
| OSV-5M | 25 | 6 / 20 (30%) |
| OSV-5M | 50 | 10 / 20 (50%) |
| Equal MP16/OSV union | 10 | 2 / 20 (10%) |
| Equal MP16/OSV union | 25 | 3 / 20 (15%) |
| Equal MP16/OSV union | 50 | 6 / 20 (30%) |

The predeclared union gate required at least 30% conditional reachability and a
10-percentage-point improvement over MP16-Pro. The 25/25 union reached 30%,
versus 15% for MP16-Pro top 50, and therefore passed exactly at the threshold.

OSV-5M alone was the strongest source. Equal source allocation reduced its
top-50 reachability from 50% to 30%, so the result supports adaptive source and
depth allocation rather than a permanently equal split. MP16-Pro top 50 found
one reachable case that OSV-5M top 50 did not, but that MP16-only case was below
the top-25 MP16 allocation used by the equal union.

## Candidate contract

The experiment retrieved corpus images internally but emitted only geographic
location proposals. Public cards contain coordinates, corpus provenance,
source-local rank and score, cluster support, and dispersion. They contain no
retrieved image pixels, thumbnails, or asset identifiers.

## Next experiment

Run a fixed offline policy that searches the OSV-5M simulator around these
locations. Compare OSV-only, MP16-only, equal union, and OSV-heavy allocation at
matched location and opening budgets. This must demonstrate that reachable
evidence can be acquired and selected safely before training a controller.

Artifacts: `artifacts/candidate_locations/v1/`.
