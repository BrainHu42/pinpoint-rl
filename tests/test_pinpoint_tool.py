from geo_search_env.experiment.pinpoint_tool import areas, pick


def test_agreement_beats_a_lone_top_score():
    # candidate 0 scores highest but stands alone; candidates 1-3 sit within a few km of each other
    scores = [1.0, 0.9, 0.85, 0.8]
    latlon = [[48.0, 2.0], [45.0, 7.0], [45.01, 7.01], [45.02, 6.99]]
    out = areas(scores, latlon, ["Paris", "Turin", "Turin", "Turin"])
    assert out[0].label == "Turin" and sorted(out[0].members) == [1, 2, 3]
    assert out[1].members == [0]
    assert pick(scores, latlon, "best") == 1  # Pinpoint's best member of the winning area
    assert abs(sum(a.support for a in out) - 1) < 1e-9


def test_single_area_keeps_pinpoints_top1():
    scores = [0.5, 0.4]
    latlon = [[40.0, -3.7], [40.05, -3.7]]
    assert pick(scores, latlon, "best") == 0
