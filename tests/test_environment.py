import unittest

from geo_search_env import Action, Coordinate, InitialCandidate, LocalSnapshotTools, PublicEpisode, ReferenceAsset, SearchEnvironment, SnapshotWorld, SyntheticFixtureMatcher
from tests.helpers import episode


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.reference = ReferenceAsset("mapillary:r", ("thumb", "image"), Coordinate(0, 0.005), "fixture", "fixture-camera", sequence_id="s")
        world = SnapshotWorld("world", (self.reference,))
        matcher = SyntheticFixtureMatcher({("query", "mapillary:r"): 0.9}, default_score=0)
        self.environment = SearchEnvironment(LocalSnapshotTools(world, matcher))
        self.observation = self.environment.reset(episode())

    def test_search_open_and_finish(self):
        inspected = self.environment.step(Action.inspect_coverage("baseline", 1000))
        self.assertEqual(inspected.action_cost, 1)
        self.assertEqual(inspected.observation.latest_tool_response.coverage_summary.approximate_image_count, 1)
        self.assertFalse(inspected.observation.latest_tool_response.coverage_summary.is_approximate)
        searched = self.environment.step(Action.search_near("baseline", 1000))
        self.assertIsNone(searched.error)
        self.assertEqual(searched.action_cost, 4)
        expanded = self.environment.step(Action.search_near("mapillary:r", 1000))
        self.assertIsNone(expanded.error)
        opened = self.environment.step(Action.open_results(("mapillary:r",)))
        self.assertEqual(opened.action_cost, 2)
        self.assertIn("mapillary:r", opened.observation.valid_candidate_ids)
        finished = self.environment.step(Action.finish("mapillary:r"))
        self.assertEqual(finished.termination_state.value, "finished")

    def test_discovery_and_budget_are_enforced(self):
        result = self.environment.step(Action.open_results(("missing",)))
        self.assertEqual(result.error.code, "asset_not_discovered")
        self.assertEqual(result.action_cost, 0)
        low = SearchEnvironment(LocalSnapshotTools(SnapshotWorld("empty", ()), SyntheticFixtureMatcher({}, default_score=0)))
        low.reset(episode(credits=2))
        transition = low.step(Action.search_near("baseline", 1000))
        self.assertEqual(transition.error.code, "unaffordable_action")

    def test_invalid_finish_falls_back(self):
        result = self.environment.step(Action.finish("unknown"))
        self.assertEqual(result.error.code, "invalid_final_candidate")
        self.assertTrue(self.environment.get_trace().final_selection.used_fallback)


if __name__ == "__main__":
    unittest.main()
