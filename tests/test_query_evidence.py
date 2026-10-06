from geo_search_env.experiment.query_evidence import parse_queries


def test_parse_queries_keeps_distinct_nonempty_strings_in_order():
    answer = 'Sure.\n```json\n{"queries": ["red bridge", "", "red bridge", "tram on street", "a", "b"]}\n```'
    assert parse_queries(answer) == ["red bridge", "tram on street", "a"]
    assert parse_queries("no json here") == []
