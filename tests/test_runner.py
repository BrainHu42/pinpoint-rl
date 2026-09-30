import json
from pathlib import Path
import tempfile
import unittest

from geo_search_env import OSVSimulatorTools
from geo_search_env.runner import load_private_labels, load_public_episodes, run_episodes, score_traces, summarize, write_run
from geo_search_env.scoring import RewardConfig
from tests.helpers import ROOT, fixture_matcher, fixture_world


class RunnerTests(unittest.TestCase):
    def test_fixture_policies_run_and_adaptive_improves_accuracy(self):
        episodes = load_public_episodes(ROOT / "fixtures/public/episodes.jsonl")
        labels = load_private_labels(ROOT / "fixtures/private/labels.jsonl")
        world = fixture_world()
        baseline = score_traces(run_episodes(episodes, OSVSimulatorTools(world, fixture_matcher()), "baseline_only"), labels, world, RewardConfig())
        adaptive_traces = run_episodes(episodes, OSVSimulatorTools(world, fixture_matcher()), "adaptive_search")
        adaptive = score_traces(adaptive_traces, labels, world, RewardConfig())
        self.assertGreater(summarize(adaptive)["accuracy_1000m"], summarize(baseline)["accuracy_1000m"])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            summary = write_run(output, adaptive_traces, adaptive, configuration={"fixture": True})
            self.assertEqual(summary["results"]["episode_count"], len(episodes))
            self.assertNotIn("query_coordinate", (output / "traces.jsonl").read_text())
            self.assertIn("query_coordinate", json.loads((ROOT / "fixtures/private/labels.jsonl").read_text().splitlines()[0]))


if __name__ == "__main__":
    unittest.main()
