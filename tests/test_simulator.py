import unittest

from geo_search_env import Action, InMemoryCorpusStore, OSVSimulatorTools, SearchEnvironment, SyntheticFixtureMatcher
from tests.helpers import episode
from geo_search_env import Coordinate, ReferenceAsset, SnapshotWorld


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        references = tuple(
            ReferenceAsset(f"mapillary:r{i}", (f"thumb:{i}",), Coordinate(0, i * 0.001), "fixture", "fixture-camera", sequence_id=f"s{i}")
            for i in range(1, 4)
        )
        self.world = SnapshotWorld("osv-fixture-v1", references)
        self.backend = OSVSimulatorTools(InMemoryCorpusStore(self.world), SyntheticFixtureMatcher({}, default_score=0.5), page_size=2)

    def test_pagination_is_deterministic_and_request_bound(self):
        env = SearchEnvironment(self.backend)
        env.reset(episode())
        first = env.step(Action.search_near("baseline", 1000))
        self.assertEqual(len(first.observation.latest_tool_response.search_results), 2)
        cursor = first.observation.latest_tool_response.next_cursor
        second = env.step(Action.search_near("baseline", 1000, cursor))
        self.assertEqual(len(second.observation.latest_tool_response.search_results), 1)
        wrong = env.step(Action.search_near("baseline", 900, cursor))
        self.assertEqual(wrong.error.code, "invalid_cursor")
        self.assertEqual(wrong.action_cost, 4)

    def test_store_reports_1km_availability(self):
        store = InMemoryCorpusStore(self.world)
        self.assertEqual(len(store.eligible_references_near(Coordinate(0, 0), 1000)), 3)

    def test_coverage_is_exact_for_audited_catalog(self):
        env = SearchEnvironment(self.backend)
        env.reset(episode())
        transition = env.step(Action.inspect_coverage("baseline", 1000))
        summary = transition.observation.latest_tool_response.coverage_summary
        self.assertEqual((summary.approximate_image_count, summary.approximate_sequence_count), (3, 3))
        self.assertEqual(summary.tiles_queried, 0)
        self.assertFalse(summary.is_approximate)


if __name__ == "__main__":
    unittest.main()
