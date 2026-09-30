# Compatibility entrypoint for running and scoring baseline experiment policies.
# Usage: python -m geo_search_env.runner --public-episodes fixtures/public/episodes.jsonl --private-labels fixtures/private/labels.jsonl --policy adaptive_search --output /tmp/mapillary-run

"""Stable CLI and import path backed by :mod:`geo_search_env.experiment.runner`."""

from .experiment.runner import (
    load_private_labels,
    load_public_episodes,
    main,
    run_corpus_episodes,
    run_episodes,
    score_traces,
    summarize,
    write_run,
)

__all__ = [name for name in globals() if not name.startswith("_")]


if __name__ == "__main__":
    raise SystemExit(main())
