from __future__ import annotations

import json
from pathlib import Path

from geo_search_env import (
    BudgetConfig,
    Coordinate,
    InitialCandidate,
    PublicEpisode,
    SnapshotWorld,
    SyntheticFixtureMatcher,
)

ROOT = Path(__file__).resolve().parents[1]


def episode(*, credits: int = 16) -> PublicEpisode:
    return PublicEpisode(
        "ep",
        "query",
        (InitialCandidate("baseline", Coordinate(0, 0), 1, 0.5),),
        "baseline",
        BudgetConfig(credits=credits, max_nonterminal_actions=8),
    )


def fixture_world() -> SnapshotWorld:
    return SnapshotWorld.from_dict(json.loads((ROOT / "fixtures/public/snapshot.json").read_text()))


def fixture_matcher() -> SyntheticFixtureMatcher:
    return SyntheticFixtureMatcher.from_dict(json.loads((ROOT / "fixtures/public/matcher.json").read_text()))
