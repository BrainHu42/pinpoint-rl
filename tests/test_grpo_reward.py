from geo_search_env.experiment.grpo_train import geo_reward


def test_reward_scores_distance_bands_and_zeroes_unparseable_answers():
    answer = lambda lat, lon: [{"role": "assistant", "content": f'Candidate: none\n```json\n{{"lat": {lat}, "lon": {lon}}}\n```'}]
    completions = [answer(-12.046, -77.043), answer(-12.15, -77.02), answer(48.857, 2.352), [{"role": "assistant", "content": "Lima"}]]
    near, town, far, unparsed = geo_reward(completions, lat=[-12.0431] * 4, lon=[-77.0452] * 4)  # ~0.4 km, ~12 km, ~10,000 km
    assert near > 1.99 and 1.49 < town < 1.5 and far < 0.01 and unparsed == 0.0
