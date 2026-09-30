import unittest

from geo_search_env import Action, Coordinate, GroundTruthRecord, LocalSnapshotTools, ReferenceAsset, SearchEnvironment, SnapshotWorld, SyntheticFixtureMatcher, score_episode
from tests.helpers import episode


class ScoringTests(unittest.TestCase):
    def test_success_cost_availability_and_harm_are_private(self):
        good = ReferenceAsset("mapillary:good", ("thumb",), Coordinate(0, 0.001), "fixture", "fixture-camera")
        bad = ReferenceAsset("mapillary:bad", ("thumb",), Coordinate(0, 0.02), "fixture", "fixture-camera")
        world = SnapshotWorld("world", (good, bad))
        matcher = SyntheticFixtureMatcher({}, default_score=0.9)
        env = SearchEnvironment(LocalSnapshotTools(world, matcher))
        public_episode = episode()
        env.reset(public_episode)
        env.step(Action.search_near("baseline", 3000))
        env.step(Action.open_results(("mapillary:good",)))
        env.step(Action.finish("mapillary:good"))
        result = score_episode(env.get_trace(), GroundTruthRecord("ep", Coordinate(0, 0.001)), eligible_world=world)
        self.assertTrue(result.success_1000m)
        self.assertEqual(result.acquisition_cost, 6)
        self.assertEqual(result.available_reference_count_1000m, 1)
        self.assertEqual(result.acquired_reference_count_1000m, 1)

    def test_harmful_refinement_has_threshold_definition(self):
        bad = ReferenceAsset("mapillary:bad", ("thumb",), Coordinate(0, 0.02), "fixture", "fixture-camera")
        world = SnapshotWorld("world", (bad,))
        env = SearchEnvironment(LocalSnapshotTools(world, SyntheticFixtureMatcher({}, default_score=0.9)))
        env.reset(episode())
        env.step(Action.search_near("baseline", 3000))
        env.step(Action.open_results(("mapillary:bad",)))
        env.step(Action.finish("mapillary:bad"))
        result = score_episode(env.get_trace(), GroundTruthRecord("ep", Coordinate(0, 0)), eligible_world=world)
        self.assertTrue(result.baseline_success_1000m)
        self.assertTrue(result.harmful_refinement)


if __name__ == "__main__":
    unittest.main()
