import json
import unittest

from geo_search_env import Action, HttpResponse, LiveMapillaryTools, MapillaryApiClient, SearchEnvironment, SyntheticFixtureMatcher
from tests.helpers import episode


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


class FakeAcquirer:
    def __init__(self):
        self.calls = []

    def acquire_reference(self, asset_id, source_url):
        self.calls.append((asset_id, source_url))


def response(value, status=200):
    return HttpResponse(status, json.dumps(value).encode(), {})


class LiveApiTests(unittest.TestCase):
    def _tools(self, transport):
        client = MapillaryApiClient(transport, "secret", user_agent="research@example.com", minimum_interval_s=0)
        return LiveMapillaryTools(client, SyntheticFixtureMatcher({}, default_score=0.8), FakeAcquirer(), page_size=2)

    def test_bbox_search_filters_radius_opens_and_hides_secrets(self):
        transport = FakeTransport([response({
            "data": [
                {"id": "near", "computed_geometry": {"coordinates": [0.001, 0]}, "thumb_256_url": "https://x.fbcdn.net/a.jpg", "thumb_1024_url": "https://x.fbcdn.net/b.jpg", "creator": {"username": "u"}, "sequence": "s"},
                {"id": "far", "computed_geometry": {"coordinates": [1, 1]}, "thumb_256_url": "https://x.fbcdn.net/c.jpg"},
            ],
            "paging": {},
        })])
        tools = self._tools(transport)
        env = SearchEnvironment(tools)
        env.reset(episode())
        searched = env.step(Action.search_near("baseline", 1000))
        cards = searched.observation.latest_tool_response.search_results
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].coordinate_provenance, "provider-computed-camera-geometry")
        self.assertEqual(cards[0].sequence_id, "s")
        self.assertNotIn("secret", json.dumps(searched.observation.to_dict()))
        opened = env.step(Action.open_results((cards[0].asset_id,)))
        self.assertEqual(opened.observation.match_scores[0].score, 0.8)

    def test_provider_errors_are_structured_and_charged(self):
        env = SearchEnvironment(self._tools(FakeTransport([response({}, status=429)])))
        env.reset(episode())
        result = env.step(Action.search_near("baseline", 1000))
        self.assertEqual(result.error.code, "provider_rate_limited")
        self.assertEqual(result.action_cost, 4)

    def test_client_repr_redacts_token(self):
        client = MapillaryApiClient(FakeTransport([]), "very-secret", user_agent="research@example.com", minimum_interval_s=0)
        self.assertNotIn("very-secret", repr(client))

    def test_coverage_aggregates_and_deduplicates_public_tiles(self):
        tile = {
            "image": {"features": [{
                "id": "image-1",
                "geometry": {"type": "Point", "coordinates": [0.001, 0]},
                "properties": {"captured_at": 200, "is_pano": True, "sequence_id": "sequence-1"},
            }]},
            "sequence": {"features": [{
                "id": "sequence-1",
                "geometry": {"type": "LineString", "coordinates": [[0, 0], [0.001, 0]]},
                "properties": {},
            }]},
        }
        transport = FakeTransport([HttpResponse(200, b"tile", {}) for _ in range(4)])
        client = MapillaryApiClient(
            transport,
            "secret",
            user_agent="research@example.com",
            minimum_interval_s=0,
            tile_decoder=lambda body, zoom, x, y: tile,
        )
        tools = LiveMapillaryTools(client, SyntheticFixtureMatcher({}, default_score=0.8), FakeAcquirer())
        env = SearchEnvironment(tools)
        env.reset(episode())
        transition = env.step(Action.inspect_coverage("baseline", 1000))
        summary = transition.observation.latest_tool_response.coverage_summary
        self.assertEqual(summary.approximate_image_count, 1)
        self.assertEqual(summary.approximate_sequence_count, 1)
        self.assertEqual(summary.panorama_fraction, 1.0)
        self.assertEqual(summary.tiles_queried, 4)
        self.assertTrue(summary.is_approximate)
        self.assertTrue(all(call[0].startswith("https://tiles.mapillary.com/maps/vtp/mly1_public/2/14/") for call in transport.calls))
        self.assertTrue(all(call[1]["params"] == {"access_token": "secret"} for call in transport.calls))

    def test_coverage_tile_cap_fails_before_network_and_is_not_charged(self):
        transport = FakeTransport([])
        client = MapillaryApiClient(
            transport,
            "secret",
            user_agent="research@example.com",
            minimum_interval_s=0,
            tile_decoder=lambda body, zoom, x, y: {},
            max_coverage_tiles=1,
        )
        tools = LiveMapillaryTools(client, SyntheticFixtureMatcher({}, default_score=0.8), FakeAcquirer())
        env = SearchEnvironment(tools)
        env.reset(episode())
        transition = env.step(Action.inspect_coverage("baseline", 1000))
        self.assertEqual(transition.error.code, "coverage_window_too_large")
        self.assertEqual(transition.action_cost, 0)
        self.assertEqual(transport.calls, [])


if __name__ == "__main__":
    unittest.main()
