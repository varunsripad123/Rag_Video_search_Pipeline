from src.api.models import SearchOptions, SearchRequest


def test_search_request_defaults():
    payload = SearchRequest(query="hello")
    assert payload.options.expand is True
    assert payload.options.top_k == 5
    assert payload.history == []
