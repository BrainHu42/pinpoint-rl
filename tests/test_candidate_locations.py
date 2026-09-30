from geo_search_env.core.contracts import Coordinate
from geo_search_env.experiment.candidate_locations import (
    LocationProposal,
    RankedCoordinate,
    cluster_candidate_locations,
    merge_candidate_locations,
)


def test_cluster_candidate_locations_emits_locations_not_asset_records():
    rows = [
        RankedCoordinate(Coordinate(0.0, 0.0), 0.9, 1),
        RankedCoordinate(Coordinate(0.0, 0.001), 0.8, 2),
        RankedCoordinate(Coordinate(10.0, 10.0), 0.7, 3),
    ]

    proposals = cluster_candidate_locations(rows, source="osv5m", radius_m=500)

    assert len(proposals) == 2
    assert proposals[0].support_count == 2
    assert proposals[0].source == "osv5m"


def test_merge_candidate_locations_round_robins_and_deduplicates():
    def proposal(latitude: float, longitude: float, source: str, rank: int) -> LocationProposal:
        return LocationProposal(Coordinate(latitude, longitude), source, rank, 0.5, 1, 0.0)

    merged = merge_candidate_locations(
        [proposal(0.0, 0.0, "mp16", 1), proposal(20.0, 20.0, "mp16", 2)],
        [proposal(0.0, 0.001, "osv5m", 1), proposal(30.0, 30.0, "osv5m", 2)],
        limit=3,
        radius_m=500,
    )

    assert [(item.source, item.source_rank) for item in merged] == [
        ("mp16", 1),
        ("osv5m", 2),
        ("mp16", 2),
    ]
