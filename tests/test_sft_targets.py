from geo_search_env.experiment.pivot_diagnostics import parse_coordinates
from geo_search_env.experiment.sft_data import build_example


OPTIONS = [("Sydney, New South Wales, Australia", -33.8688, 151.2093), ("Lima, Lima, Peru", -12.0464, -77.0428)]
LABELS = ("Peru", "Lima", "Lima")


def test_target_copies_a_candidate_within_1km_and_parses_back_as_displayed():
    prompt, target, meta = build_example(OPTIONS, (-12.0431, -77.0452), LABELS)  # ~0.45 km from candidate 2
    assert "Candidate: 2" in target and meta["case"] == "copy <1km"
    assert parse_coordinates(target) == (-12.046, -77.043)  # the candidate's coordinates as the prompt shows them
    assert "(-12.046, -77.043)" in prompt


def test_target_uses_truth_when_no_candidate_is_within_1km():
    _, target, meta = build_example(OPTIONS, (-12.1500, -77.0200), LABELS)  # ~12 km from candidate 2
    assert "Candidate: 2" in target and meta["case"] == "candidate 1-25km"
    assert parse_coordinates(target) == (-12.15, -77.02)
    _, target, meta = build_example(OPTIONS, (48.8566, 2.3522), ("France", "Île-de-France", "Paris"))
    assert "Candidate: none" in target and parse_coordinates(target) == (48.857, 2.352)
