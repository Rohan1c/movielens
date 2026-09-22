"""Pieces every view shares: the engine, posters, poster rows and chart styling.

The engine is fitted once per server process and cached, so moving between pages or
clicking a button never retrains anything. The first load after starting the server is
the slow one.
"""

import html
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config import load_config
from src.engine.engine import RecommendationEngine
from src.engine.posters import TMDB_NOTICE
from src.engine.posters import PosterClient
from src.engine.posters import search_title
from src.plots.style import DASHED_MODELS
from src.plots.style import MODEL_LABELS

# Chart colours. The report figures use deep colours chosen for white paper, which nearly
# disappear on a dark background, so these are the same hues lifted for contrast.
MODEL_COLOURS = {
    "global_mean": "#8A8F98",
    "user_mean": "#A3A7AE",
    "item_mean": "#C3C6CB",
    "most_popular": "#E0A33B",
    "content": "#4FB3A9",
    "item_knn": "#6F9BEB",
    "item_knn_ranking": "#6F9BEB",
    "mf": "#E05D44",
    "mf_ranking": "#E05D44",
    "hybrid": "#B08BE0",
    "hybrid_frontier": "#B08BE0",
}
DASHED = DASHED_MODELS

MODEL_NOTES = {
    "global_mean": "predicts the overall average for everything",
    "user_mean": "your average rating, for every film",
    "item_mean": "each film's average rating",
    "most_popular": "the most-rated films - same list for everyone",
    "content": "similar genres, title words and decade to what you liked",
    "item_knn": "films rated like the ones you liked, by people who rated both",
    "item_knn_ranking": "same, tuned for ranking",
    "mf": "matrix factorisation - learns taste dimensions from everyone's ratings",
    "mf_ranking": "same, tuned for ranking",
    "hybrid": "content and matrix factorisation blended, weighted by how much you've rated",
    "hybrid_frontier": "content and matrix factorisation at a fixed blend",
}

DEFAULT_METHODS = ["mf", "item_knn", "content", "most_popular"]

CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1400px; }
.method { margin: 1.6rem 0 0.5rem; display: flex; align-items: baseline; gap: 0.6rem;
          flex-wrap: wrap; }
.method-name { font-weight: 600; font-size: 1.05rem; }
.method-note { color: #9A9BA0; font-size: 0.85rem; }
.method-score { color: #9A9BA0; font-size: 0.85rem; margin-left: auto; }
.strip { display: flex; gap: 14px; overflow-x: auto; padding-bottom: 8px; }
.film { flex: 0 0 124px; }
.film img, .film .blank { width: 124px; height: 186px; border-radius: 4px; object-fit: cover;
        display: block; background: #2A2B30; }
.film .blank { display: flex; align-items: center; justify-content: center;
        text-align: center; padding: 10px; font-size: 12px; color: #C8C8C8;
        box-sizing: border-box; line-height: 1.3; }
.film.liked img, .film.liked .blank { box-shadow: 0 0 0 3px #4CAF7D; }
.film .name { font-size: 12.5px; margin-top: 6px; line-height: 1.25; max-height: 2.5em;
        overflow: hidden; }
.film .sub { font-size: 11.5px; color: #9A9BA0; margin-top: 1px; }
.film .liked-tag { color: #4CAF7D; }
.small { color: #9A9BA0; font-size: 0.85rem; }
</style>
"""


@st.cache_resource(show_spinner="Loading the models (about a minute the first time)...")
def get_engine():
    return RecommendationEngine.from_config(load_config(), cache=True)


def read_tmdb_key():
    """The key from .streamlit/secrets.toml or the environment, or None. Never displayed."""
    try:
        value = st.secrets.get("TMDB_API_KEY")
    except Exception:
        value = None
    if value is None:
        value = os.environ.get("TMDB_API_KEY")
    return value


@st.cache_resource
def get_posters():
    return PosterClient(read_tmdb_key(), REPOSITORY_ROOT / "data" / "posters.json")


@st.cache_data
def load_result(name):
    return pd.read_csv(REPOSITORY_ROOT / "results" / name)


def model_label(name):
    return MODEL_LABELS.get(name, name)


def film_card(film, posters, caption):
    """A poster with the title the way a person writes it and the year underneath.

    MovieLens stores "Usual Suspects, The (1995)"; the card shows "The Usual Suspects"
    over "1995", which is how every film app presents a title.
    """
    image = posters.poster_url(film["title"], film["year"])
    title = html.escape(search_title(film["title"]))
    if int(film["year"]) > 0:
        caption = str(int(film["year"])) + " &middot; " + caption
    if image is None:
        picture = '<div class="blank">' + title + "</div>"
    else:
        picture = '<img src="' + html.escape(image) + '" alt="' + title + '" loading="lazy">'
    css_class = "film"
    if film.get("hit"):
        css_class = "film liked"
        caption = '<span class="liked-tag">liked it later</span>'
        if int(film["year"]) > 0:
            caption = str(int(film["year"])) + " &middot; " + caption
    return ('<div class="' + css_class + '" title="' + title + '">' + picture
            + '<div class="name">' + title + '</div><div class="sub">' + caption
            + "</div></div>")


def poster_strip(films, posters, caption_for):
    cards = []
    for film in films:
        cards.append(film_card(film, posters, caption_for(film)))
    return '<div class="strip">' + "".join(cards) + "</div>"


def predicted_caption(film):
    return format(film["predicted"], ".1f") + " predicted"


def method_rows(results, names, posters, show_hits):
    """One row of posters per method, the way a streaming app lays out its rows."""
    films = []
    for name in names:
        for film in results[name]:
            films.append((film["title"], film["year"]))
    posters.prefetch(films)

    for name in names:
        rows = results[name]
        score = ""
        if show_hits:
            hits = 0
            for film in rows:
                if film.get("hit"):
                    hits = hits + 1
            score = str(hits) + " of " + str(len(rows)) + " liked later"
        st.markdown(
            '<div class="method"><span class="method-name">' + html.escape(model_label(name))
            + '</span><span class="method-note">' + html.escape(MODEL_NOTES.get(name, ""))
            + '</span><span class="method-score">' + score + "</span></div>"
            + poster_strip(rows, posters, predicted_caption),
            unsafe_allow_html=True,
        )


def style_chart(figure, height=380):
    """Dark styling for a plotly figure, applied in one place."""
    figure.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=36, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6E6E6"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    figure.update_xaxes(gridcolor="#2E2F34", zerolinecolor="#3A3B40")
    figure.update_yaxes(gridcolor="#2E2F34", zerolinecolor="#3A3B40")
    return figure


def footer(posters):
    text = ("Recommendations come from the same trained models the project evaluates. "
            "The catalogue is MovieLens 100K, so it stops at 1998.")
    if posters.enabled:
        text = text + " " + TMDB_NOTICE
    st.markdown('<p class="small" style="margin-top:2.5rem">' + html.escape(text) + "</p>",
                unsafe_allow_html=True)
