"""Film posters from The Movie Database (TMDB).

Optional. With no API key the app shows genre tags instead, and nothing else changes.

Each film is looked up once and the answer - including "no poster found" - is cached to
``data/posters.json``, so a page of recommendations costs at most ten requests the first
time and none after. ``data/`` is gitignored, so the cache never reaches the repository.

The key is sent to TMDB and nowhere else. It is never printed, logged or written to the
cache, and a failed request is reported without the URL, because the URL contains it.

Attribution: TMDB's terms require any product using their API to say so. The app carries
the notice in ``TMDB_NOTICE``.
"""

import json
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request

SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
TIMEOUT_SECONDS = 6
TMDB_NOTICE = "This product uses the TMDB API but is not endorsed or certified by TMDB."
YEAR_IN_TITLE = re.compile(r"\s*\(\s*(\d{4})\s*\)\s*")
ARTICLE_SUFFIX = re.compile(r",\s*(The|A|An)$")


def search_title(label):
    """Turn a MovieLens title into what TMDB expects: "Godfather, The (1972)" -> "The Godfather".

    Alternative titles in a second set of brackets are dropped, since TMDB matches on the
    main title and the extra words only hurt the search.
    """
    text = YEAR_IN_TITLE.sub(" ", str(label)).strip()
    bracket = text.find("(")
    if bracket > 0:
        text = text[:bracket].strip()
    match = ARTICLE_SUFFIX.search(text)
    if match:
        text = match.group(1) + " " + text[: match.start()].strip()
    return text


def fetch_json(url):
    """GET a URL and parse JSON, verifying certificates with a certifi fallback."""
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        if not isinstance(error.reason, ssl.SSLCertVerificationError):
            raise
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS, context=context) as response:
            return json.loads(response.read().decode("utf-8"))


class PosterClient:
    """Look up and cache poster URLs by title and year."""

    def __init__(self, api_key, cache_path, fetch=fetch_json):
        self.api_key = api_key
        self.cache_path = cache_path
        self.fetch = fetch
        self.cache = self.load_cache()
        self.last_error = None

    @property
    def enabled(self):
        return self.api_key is not None and str(self.api_key).strip() != ""

    def load_cache(self):
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}

    def save_cache(self):
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=0), encoding="utf-8")

    def poster_url(self, label, year):
        """A full image URL, or None when there is no key, no match, or no network."""
        cache_key = search_title(label) + "|" + str(year)
        if cache_key in self.cache:
            return full_url(self.cache[cache_key])
        if not self.enabled:
            return None

        query = {"api_key": str(self.api_key).strip(), "query": search_title(label)}
        if int(year) > 0:
            query["year"] = str(int(year))
        url = SEARCH_URL + "?" + urllib.parse.urlencode(query)
        try:
            payload = self.fetch(url)
        except Exception as error:
            # Deliberately not the URL or the message: either can carry the key.
            self.last_error = type(error).__name__
            return None

        path = first_poster_path(payload)
        self.cache[cache_key] = path
        self.save_cache()
        return full_url(path)


def first_poster_path(payload):
    results = payload.get("results", [])
    for result in results:
        path = result.get("poster_path")
        if path:
            return path
    return ""


def full_url(path):
    if path is None or path == "":
        return None
    return IMAGE_BASE + path
