"""Smoke tests for the demo app, run headlessly through Streamlit's AppTest.

These need the processed dataset and fitted models, so they are skipped on a bare clone.
They check that every page renders without raising and that the main interactions work -
not how anything looks.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"

pytest.importorskip("streamlit")
if not (ROOT / "data" / "processed" / "train.parquet").exists():
    pytest.skip("needs the processed dataset; run scripts/preprocess.py", allow_module_level=True)

from streamlit.testing.v1 import AppTest

# Streamlit puts the entry script's folder on the path when it serves the app, which is how
# the views import shared. The same is needed when the app runs inside a test.
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

TIMEOUT = 600
VIEWS = ["views/recommend.py", "views/users.py", "views/results.py", "views/experiments.py",
         "views/about.py"]
TOY_STORY = 1
STAR_WARS = 50


def open_app(view=None):
    app = AppTest.from_file(str(APP / "Home.py"), default_timeout=TIMEOUT)
    app.run()
    if view is not None:
        app.switch_page(view)
        app.run()
    return app


def rendered_html(app):
    parts = []
    for element in app.markdown:
        parts.append(element.value)
    return "\n".join(parts)


@pytest.mark.parametrize("view", VIEWS)
def test_every_view_renders_without_error(view):
    app = open_app(view)
    assert not app.exception, [error.value for error in app.exception]


def test_recommend_is_the_landing_page():
    app = open_app()
    assert app.title[0].value == "MovieLens recommender"


def test_recommend_asks_for_films_before_recommending():
    app = open_app()
    assert any("add a few movies" in info.value for info in app.info)


def test_rated_films_produce_poster_rows():
    app = open_app()
    app.multiselect[0].set_value([TOY_STORY, STAR_WARS]).run()
    assert not app.exception
    html = rendered_html(app)
    assert 'class="method"' in html
    assert 'class="strip"' in html


def test_rated_films_are_not_recommended_back():
    app = open_app()
    app.multiselect[0].set_value([TOY_STORY, STAR_WARS]).run()
    html = rendered_html(app)
    assert "Toy Story" not in html.split('class="method"', 1)[1]


def test_an_example_fills_in_films_and_clear_empties_them():
    app = open_app()
    example = [button for button in app.button if button.label == "Sci-fi"][0]
    example.click().run()
    assert not app.exception
    assert len(app.session_state["films"]) > 0
    assert 'class="strip"' in rendered_html(app)
    clear = [button for button in app.button if button.label == "Clear"][0]
    clear.click().run()
    assert app.session_state["films"] == []


def test_browse_users_shows_hits_for_a_chosen_user():
    app = open_app("views/users.py")
    app.number_input[0].set_value(42).run()
    assert not app.exception
    assert "liked later" in rendered_html(app)


def test_results_redraw_for_other_measures():
    app = open_app("views/results.py")
    app.selectbox[0].set_value("rmse").run()
    app.selectbox[1].set_value("gini").run()
    assert not app.exception


def test_experiments_describe_results_from_the_csvs():
    """The sentences are generated, so they must name a real model from the results."""
    app = open_app("views/experiments.py")
    text = "\n".join(element.value for element in app.markdown)
    assert "most accurate personalised method" in text
