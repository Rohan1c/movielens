"""Tests for the recommendation engine and model forking."""

import numpy as np
import pandas as pd
import pytest

from src.data.splitting import temporal_split
from src.engine.engine import SESSION_USER
from src.engine.engine import RecommendationEngine
from src.engine.engine import ratings_frame
from tests.test_models import MODEL_FACTORIES

ENGINE_MODELS = ["most_popular", "content", "item_knn", "mf"]


@pytest.fixture
def engine(config, synthetic_ratings, synthetic_items):
    config.values["models"]["mf"]["n_epochs"] = 3
    train, validation, test = temporal_split(synthetic_ratings, config)
    return RecommendationEngine(config, train, validation, test, synthetic_items,
                                model_names=ENGINE_MODELS, cache=False)


# ---------------------------------------------------------------- forking

@pytest.fixture(params=sorted(MODEL_FACTORIES.keys()))
def fitted(request, config, synthetic_ratings, synthetic_items):
    return MODEL_FACTORIES[request.param](config, synthetic_items).fit(synthetic_ratings)


def test_folding_into_a_fork_leaves_the_original_untouched(fitted, synthetic_ratings):
    """The engine forks on every request; a leak would grow the trained model forever."""
    before = fitted.predict(2, fitted.candidate_items(2))
    fork = fitted.fork()
    fork.fold_in(SESSION_USER, pd.DataFrame({"item_id": [1, 2], "rating": [5.0, 1.0]}))
    assert SESSION_USER in fork.known_users
    assert SESSION_USER not in fitted.known_users
    assert np.allclose(fitted.predict(2, fitted.candidate_items(2)), before)


def test_the_same_session_id_can_be_reused_across_forks(fitted):
    ratings = pd.DataFrame({"item_id": [1, 2], "rating": [5.0, 1.0]})
    first = fitted.fork()
    first.fold_in(SESSION_USER, ratings)
    second = fitted.fork()
    second.fold_in(SESSION_USER, ratings)
    assert np.array_equal(first.recommend(SESSION_USER, 5), second.recommend(SESSION_USER, 5))


def test_a_fork_shares_fitted_arrays_rather_than_copying_them(config, synthetic_ratings):
    from src.models.mf import MatrixFactorization

    model = MatrixFactorization(config, n_epochs=3).fit(synthetic_ratings)
    assert model.fork().item_factors is model.item_factors


# ---------------------------------------------------------------- engine

def test_engine_fits_the_requested_models(engine):
    assert sorted(engine.models.keys()) == sorted(ENGINE_MODELS)


def test_existing_user_gets_every_models_picks(engine):
    results = engine.recommend_existing(1, k=5)
    assert sorted(results.keys()) == sorted(ENGINE_MODELS)
    for rows in results.values():
        assert len(rows) == 5
        assert [row["rank"] for row in rows] == [1, 2, 3, 4, 5]


def test_hits_are_the_users_held_out_likes(engine):
    results = engine.recommend_existing(1, k=10)
    liked = engine.likes_after(1)
    for rows in results.values():
        for row in rows:
            assert row["hit"] == (row["item_id"] in liked)


def test_an_unknown_existing_user_is_rejected(engine):
    with pytest.raises(ValueError, match="not in the training data"):
        engine.recommend_existing(424242)


def test_a_new_person_gets_recommendations(engine):
    results = engine.recommend_new({1: 5.0, 2: 4.0, 3: 1.0}, k=5)
    for rows in results.values():
        assert len(rows) == 5
        rated = {1, 2, 3}
        assert rated.isdisjoint({row["item_id"] for row in rows})


def test_new_person_requests_do_not_accumulate(engine):
    for _ in range(3):
        engine.recommend_new({1: 5.0, 2: 1.0}, k=3)
    for model in engine.models.values():
        assert SESSION_USER not in model.known_users


def test_an_empty_request_is_rejected(engine):
    with pytest.raises(ValueError, match="at least one"):
        engine.recommend_new({})


def test_rows_carry_what_a_front_end_needs(engine):
    row = engine.recommend_existing(1, k=1)["most_popular"][0]
    for field in ["rank", "item_id", "title", "year", "genres", "predicted", "ratings", "hit"]:
        assert field in row
    assert isinstance(row["genres"], list)


def test_search_finds_titles_by_word(engine):
    matches = engine.search("movie 1")
    assert len(matches) > 0
    for item_id in matches:
        assert "movie" in engine.title(item_id).lower()


def test_catalogue_is_ordered_most_rated_first(engine):
    counts = engine.catalogue()["ratings"].tolist()
    assert counts == sorted(counts, reverse=True)


def test_history_is_highest_rated_first(engine):
    ratings = engine.history(1)["rating"].tolist()
    assert ratings == sorted(ratings, reverse=True)


def test_exported_history_is_matched_and_scaled(engine):
    export = pd.DataFrame({"Name": ["Movie 1", "Movie 2", "Nothing Like It"],
                           "Rating": [4.5, 7.0, 3.0]})
    ratings, unmatched = engine.match_export(export)
    assert set(ratings.keys()) == {1, 2}
    assert ratings[2] == 5.0
    assert unmatched == ["Nothing Like It"]


def test_ratings_frame_shape():
    frame = ratings_frame({3: 4.0, 1: 5.0})
    assert list(frame.columns) == ["item_id", "rating"]
    assert len(frame) == 2
