# Project Goal

Train a small search controller on an audited, API-shaped OSV-5M simulator and
test whether its learned geographic-search strategy transfers to bounded live
Mapillary search.

Starting from ranked coordinates produced by a frozen upstream geolocator, the
controller decides where to inspect local coverage, where to search, which
Mapillary results to open, and when to stop. It must either select the camera
coordinate of an opened result or retain the designated upstream baseline.

The primary outcome is all-query accuracy within 1 kilometer at matched
acquisition cost. Fixed search, adaptive scripted search, supervised imitation,
and offline RL are compared on held-out snapshot episodes before frozen-policy
live evaluation.

Flickr, MP16-Pro, global semantic archive search, text search, Wikimedia, maps,
aerial imagery, and unrestricted live training are outside the baseline scope.

See [`docs/mapillary_baseline_plan.md`](docs/mapillary_baseline_plan.md) for the
predeclared experiment and [`PLAN.md`](PLAN.md) for the implementation gates.
