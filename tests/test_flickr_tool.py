import io
import json

import pytest

from geo_search_env.tools import flickr


def _photo(pid, owner, lat, lon, acc=16, uploaded=1_700_000_000, alias=""):
    return {"id": str(pid), "owner": owner, "pathalias": alias, "latitude": str(lat), "longitude": str(lon), "accuracy": str(acc),
            "dateupload": str(uploaded), "datetaken": "2023-01-01 10:00:00", "title": "t", "tags": "", "url_m": ""}


@pytest.fixture
def fake_api(monkeypatch):
    calls = []
    body = {"stat": "ok", "photos": {"total": "3", "photo": [
        _photo(1, "a@N01", 48.14, 17.11), _photo(2, "b@N01", 48.15, 17.10, alias="bob"), _photo(3, "c@N01", 40.4, -3.7, uploaded=1_800_000_000)]}}

    def urlopen(url, timeout=None):
        calls.append(url)
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(flickr.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("FLICKR_KEY", "k")
    monkeypatch.setattr(flickr, "MIN_INTERVAL", 0.0)
    return calls


def test_calls_are_cached_canonically_and_replayed_offline(tmp_path, fake_api):
    tool = flickr.FlickrSearch(tmp_path)
    assert len(tool.search("Galéria  Nedbalka").photos) == 3
    assert len(tool.search("galéria nedbalka").photos) == 3  # same canonical call: from the cache
    assert len(fake_api) == 1 and "api_key=k" in fake_api[0]
    offline = flickr.FlickrSearch(tmp_path, offline=True)
    assert offline.search("GALÉRIA NEDBALKA").total == 3
    with pytest.raises(flickr.CacheMiss):
        offline.search("something never fetched")
    assert offline.usage()["calls, total"] == 1


def test_guard_hides_the_photo_its_owner_and_later_uploads(tmp_path, fake_api):
    tool = flickr.FlickrSearch(tmp_path)
    kept = lambda g: [p.photo_id for p in tool.search("x", guard=g).photos]  # noqa: E731
    assert kept(flickr.Guard(photo_ids={1})) == [2, 3]
    assert kept(flickr.Guard(owners={"bob"})) == [1, 3]  # path alias, as in a Commons credit URL
    assert kept(flickr.Guard(before="2025-01-01")) == [1, 2]  # 1.8e9 is 2027


def test_boxes_round_outward():
    box = [float(v) for v in flickr.FlickrSearch.search_params("x", (48.144, 17.112, 1.0))["bbox"].split(",")]
    assert box[0] <= 17.112 - 1 / 74.0 and box[2] >= 17.112 + 1 / 74.0 and box[1] <= 48.135 and box[3] >= 48.153
