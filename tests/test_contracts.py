import unittest

from geo_search_env import Action, ActionKind, BackendCapabilities, BudgetConfig, Coordinate, CoverageSummary, InitialCandidate, PublicEpisode
from geo_search_env.contracts import ContractError


class ContractTests(unittest.TestCase):
    def test_only_baseline_actions_round_trip(self):
        actions = (
            Action.search_near("candidate", 1000, "cursor"),
            Action.inspect_coverage("candidate", 1000),
            Action.open_results(("asset",)),
            Action.finish("candidate"),
        )
        self.assertEqual({item.kind for item in actions}, set(ActionKind))
        self.assertEqual(tuple(Action.from_dict(item.to_dict()) for item in actions), actions)
        with self.assertRaises(ValueError):
            Action.from_dict({"tool": "search_images", "arguments": {"query": "x"}})

    def test_coverage_summary_round_trip(self):
        summary = CoverageSummary("candidate", 1000, 12, 3, "100", "200", 0.25, 4, True)
        self.assertEqual(CoverageSummary.from_dict(summary.to_dict()), summary)

    def test_public_episode_rejects_private_data_and_legacy_budget(self):
        valid = PublicEpisode("ep", "q", (InitialCandidate("p", Coordinate(0, 0), 1),), "p", BudgetConfig())
        self.assertEqual(PublicEpisode.from_dict(valid.to_dict()), valid)
        private = valid.to_dict() | {"query_coordinate": {"latitude": 0, "longitude": 0}}
        with self.assertRaises(ContractError):
            PublicEpisode.from_dict(private)
        legacy = valid.to_dict()
        legacy["budget"]["verify_match_cost"] = 2
        with self.assertRaises(ContractError):
            PublicEpisode.from_dict(legacy)

    def test_capabilities_require_exact_baseline_surface(self):
        from geo_search_env.contracts import ActionCapability

        with self.assertRaises(ContractError):
            BackendCapabilities((ActionCapability(ActionKind.FINISH),), ())


if __name__ == "__main__":
    unittest.main()
