"""Tests for the TMDB poster lookup. The network is always faked here."""

import json

import pytest

from src.engine.posters import IMAGE_BASE
from src.engine.posters import PosterClient
from src.engine.posters import first_poster_path
from src.engine.posters import search_title

FAKE_KEY = "0123456789abcdef0123456789abcdef"


class FakeTmdb:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        return self.payload


@pytest.fixture
def cache_path(tmp_path):
    return tmp_path / "posters.json"


def test_trailing_article_and_year_are_cleaned():
    assert search_title("Godfather, The (1972)") == "The Godfather"
    assert search_title("Star Wars (1977)") == "Star Wars"


def test_alternative_titles_in_brackets_are_dropped():
    assert search_title("Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)") == "Shanghai Triad"


def test_no_key_means_no_request_and_no_poster(cache_path):
    fake = FakeTmdb({"results": [{"poster_path": "/x.jpg"}]})
    client = PosterClient(None, cache_path, fetch=fake)
    assert client.enabled is False
    assert client.poster_url("Star Wars (1977)", 1977) is None
    assert fake.urls == []


def test_a_found_poster_becomes_a_full_image_url(cache_path):
    fake = FakeTmdb({"results": [{"poster_path": "/abc.jpg"}]})
    client = PosterClient(FAKE_KEY, cache_path, fetch=fake)
    assert client.poster_url("Star Wars (1977)", 1977) == IMAGE_BASE + "/abc.jpg"


def test_the_search_sends_the_cleaned_title_and_year(cache_path):
    fake = FakeTmdb({"results": []})
    PosterClient(FAKE_KEY, cache_path, fetch=fake).poster_url("Godfather, The (1972)", 1972)
    assert "query=The+Godfather" in fake.urls[0]
    assert "year=1972" in fake.urls[0]


def test_each_film_is_fetched_only_once(cache_path):
    fake = FakeTmdb({"results": [{"poster_path": "/abc.jpg"}]})
    client = PosterClient(FAKE_KEY, cache_path, fetch=fake)
    client.poster_url("Star Wars (1977)", 1977)
    client.poster_url("Star Wars (1977)", 1977)
    assert len(fake.urls) == 1


def test_the_cache_survives_a_restart(cache_path):
    fake = FakeTmdb({"results": [{"poster_path": "/abc.jpg"}]})
    PosterClient(FAKE_KEY, cache_path, fetch=fake).poster_url("Star Wars (1977)", 1977)
    second = FakeTmdb({"results": []})
    restarted = PosterClient(FAKE_KEY, cache_path, fetch=second)
    assert restarted.poster_url("Star Wars (1977)", 1977) == IMAGE_BASE + "/abc.jpg"
    assert second.urls == []


def test_a_missing_poster_is_cached_too(cache_path):
    """Otherwise every page load would re-ask TMDB about the same obscure film."""
    fake = FakeTmdb({"results": []})
    client = PosterClient(FAKE_KEY, cache_path, fetch=fake)
    assert client.poster_url("Nothing (1990)", 1990) is None
    assert client.poster_url("Nothing (1990)", 1990) is None
    assert len(fake.urls) == 1


def test_a_network_failure_degrades_to_no_poster(cache_path):
    client = PosterClient(FAKE_KEY, cache_path, fetch=FakeTmdb(error=OSError("down")))
    assert client.poster_url("Star Wars (1977)", 1977) is None
    assert client.last_error == "OSError"


def test_the_key_is_never_written_to_the_cache(cache_path):
    fake = FakeTmdb({"results": [{"poster_path": "/abc.jpg"}]})
    PosterClient(FAKE_KEY, cache_path, fetch=fake).poster_url("Star Wars (1977)", 1977)
    assert FAKE_KEY not in cache_path.read_text(encoding="utf-8")


def test_a_failure_does_not_leak_the_key(cache_path):
    """Errors from urllib can quote the URL, and the URL carries the key."""
    error = OSError("failed fetching https://x/?api_key=" + FAKE_KEY)
    client = PosterClient(FAKE_KEY, cache_path, fetch=FakeTmdb(error=error))
    client.poster_url("Star Wars (1977)", 1977)
    assert FAKE_KEY not in str(client.last_error)


def test_a_corrupt_cache_file_is_ignored(cache_path):
    cache_path.write_text("{not json", encoding="utf-8")
    client = PosterClient(FAKE_KEY, cache_path, fetch=FakeTmdb({"results": []}))
    assert client.cache == {}


def test_results_without_a_poster_are_skipped():
    payload = {"results": [{"poster_path": None}, {"poster_path": "/second.jpg"}]}
    assert first_poster_path(payload) == "/second.jpg"
    assert first_poster_path({}) == ""
