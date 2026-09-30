import unittest

import numpy as np

from geo_search_env import Coordinate, PinpointRetrievalBaseline, RetrievalCandidate


class RetrievalBaselineContractTests(unittest.TestCase):
    def test_candidate_contract_has_rank_score_and_coordinate(self):
        candidate = RetrievalCandidate(1, Coordinate(10, 20), 0.75, 42)
        self.assertEqual(candidate.rank, 1)
        self.assertEqual(candidate.coordinate, Coordinate(10, 20))
        self.assertEqual(candidate.retrieval_index, 42)

    def test_retrieval_candidates_become_ranked_episode_anchors(self):
        model = PinpointRetrievalBaseline.__new__(PinpointRetrievalBaseline)
        model.predict_candidates = lambda embedding, top_k: (
            RetrievalCandidate(1, Coordinate(10, 20), -0.5, 4),
            RetrievalCandidate(2, Coordinate(30, 40), 0.75, 8),
        )
        candidates = model.initial_candidates(object(), 2)
        self.assertEqual([item.candidate_id for item in candidates], ["retrieval:1", "retrieval:2"])
        self.assertEqual([item.rank for item in candidates], [1, 2])
        self.assertEqual([item.confidence for item in candidates], [0.25, 0.875])
        self.assertEqual(model.predict(object()), Coordinate(10, 20))

    def test_projection_delegates_to_the_named_frozen_source_adapter(self):
        class Runtime:
            def encode_embeddings(self, embeddings, *, source):
                self.call = (embeddings, source)
                return np.array([[0.0, 1.0]], dtype=np.float32)

        model = PinpointRetrievalBaseline.__new__(PinpointRetrievalBaseline)
        model._runtime = Runtime()
        value = np.array([1.0, 2.0], dtype=np.float32)
        projected = model.project_image_embeddings(value, source="osv5m")
        self.assertEqual(model._runtime.call[1], "osv5m")
        np.testing.assert_array_equal(model._runtime.call[0], value)
        np.testing.assert_array_equal(projected, [[0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
