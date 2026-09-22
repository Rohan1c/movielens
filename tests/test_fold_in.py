"""Tests for fold-in: adding a user to a trained model without refitting.

The property that matters is consistency with training. If folding a user in produced
different recommendations from training on that same user, the app would be serving
something other than the models the report evaluates. For every model whose per-user state
has a closed form, the two must agree exactly.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.mf import MatrixFactorization
from tests.test_models import MODEL_FACTORIES

NEW_USER = 99999
EXACT_MODELS = [
    "global_mean", "user_mean", "item_mean", "most_popular",
    "content", "item_knn", "item_knn_ranking",
]


@pytest.fixture(params=sorted(MODEL_FACTORIES.keys()))
def fitted(request, config, synthetic_ratings, synthetic_items):
    model = MODEL_FACTORIES[request.param](config, synthetic_items)
    return request.param, model.fit(synthetic_ratings)


def history_of(ratings, user_id):
    return ratings[ratings["user_id"] == user_id][["item_id", "rating"]].reset_index(drop=True)


def test_every_model_supports_fold_in(fitted, synthetic_ratings):
    _, model = fitted
    model.fold_in(NEW_USER, history_of(synthetic_ratings, 1))
    assert NEW_USER in model.known_users


def test_a_folded_user_gets_recommendations(fitted, synthetic_ratings):
    _, model = fitted
    model.fold_in(NEW_USER, history_of(synthetic_ratings, 1))
    recommendations = model.recommend(NEW_USER, 5)
    assert len(recommendations) == 5


def test_a_folded_users_rated_films_are_excluded(fitted, synthetic_ratings):
    _, model = fitted
    history = history_of(synthetic_ratings, 1)
    model.fold_in(NEW_USER, history)
    rated = set(history["item_id"].tolist())
    assert set(model.recommend(NEW_USER, 10).tolist()).isdisjoint(rated)


def test_folded_predictions_are_finite_and_in_range(fitted, synthetic_ratings):
    _, model = fitted
    model.fold_in(NEW_USER, history_of(synthetic_ratings, 1))
    predictions = model.predict(NEW_USER, model.candidate_items(NEW_USER))
    assert np.all(np.isfinite(predictions))
    assert np.all(predictions >= model.rating_min)
    assert np.all(predictions <= model.rating_max)


def test_folding_in_an_existing_id_is_refused(fitted, synthetic_ratings):
    """Overwriting a training user would silently change what the report measured."""
    _, model = fitted
    with pytest.raises(ValueError, match="already known"):
        model.fold_in(1, history_of(synthetic_ratings, 1))


def test_fold_in_matches_training_for_closed_form_models(
    config, synthetic_ratings, synthetic_items
):
    """Fold in user 1's own history under a new id: the answer must be identical."""
    for name in EXACT_MODELS:
        model = MODEL_FACTORIES[name](config, synthetic_items).fit(synthetic_ratings)
        model.fold_in(NEW_USER, history_of(synthetic_ratings, 1))
        candidates = model.candidate_items(1)
        assert np.allclose(
            model.predict(1, candidates), model.predict(NEW_USER, candidates)
        ), name
        assert np.array_equal(model.recommend(1, 10), model.recommend(NEW_USER, 10)), name


def test_fold_in_leaves_training_users_untouched(config, synthetic_ratings, synthetic_items):
    for name in sorted(MODEL_FACTORIES.keys()):
        model = MODEL_FACTORIES[name](config, synthetic_items).fit(synthetic_ratings)
        before = model.predict(2, model.candidate_items(2))
        model.fold_in(NEW_USER, history_of(synthetic_ratings, 1))
        after = model.predict(2, model.candidate_items(2))
        assert np.allclose(before, after), name


def test_several_users_can_be_folded_in(fitted, synthetic_ratings):
    _, model = fitted
    model.fold_in(NEW_USER, history_of(synthetic_ratings, 1))
    model.fold_in(NEW_USER + 1, history_of(synthetic_ratings, 2))
    assert len(model.recommend(NEW_USER, 3)) == 3
    assert len(model.recommend(NEW_USER + 1, 3)) == 3


def test_films_outside_the_catalogue_are_ignored(fitted):
    _, model = fitted
    ratings = pd.DataFrame({"item_id": [1, 2, 424242], "rating": [5.0, 4.0, 5.0]})
    model.fold_in(NEW_USER, ratings)
    assert np.all(np.isfinite(model.predict(NEW_USER, np.array([3, 4]))))


def test_mf_fold_in_learns_the_new_users_taste(config):
    """Two taste groups; a folded-in user who shares group one's taste must lean that way.

    Training ran SGD over everyone at once while fold-in solves one regression, so the
    numbers are not expected to match training exactly - the direction of the taste is.
    """
    rows = []
    for user_id in range(1, 21):
        if user_id <= 10:
            liked = [1, 2, 3]
            disliked = [4, 5, 6]
        else:
            liked = [4, 5, 6]
            disliked = [1, 2, 3]
        for item_id in liked:
            rows.append((user_id, item_id, 5.0, 100 + item_id))
        for item_id in disliked:
            rows.append((user_id, item_id, 1.0, 100 + item_id))
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    model = MatrixFactorization(config, n_factors=4, learning_rate=0.02, n_epochs=200)
    model.fit(ratings)

    new = pd.DataFrame({"item_id": [1, 2, 4, 5], "rating": [5.0, 5.0, 1.0, 1.0]})
    model.fold_in(NEW_USER, new)
    liked_score, disliked_score = model.predict(NEW_USER, np.array([3, 6]))
    assert liked_score > disliked_score + 1.0


def test_mf_fold_in_with_no_known_films_falls_back_to_the_item_bias(
    config, synthetic_ratings
):
    model = MatrixFactorization(config, n_epochs=5).fit(synthetic_ratings)
    model.fold_in(NEW_USER, pd.DataFrame({"item_id": [424242], "rating": [5.0]}))
    positions = model.rating_matrix.item_positions(np.array([1, 2]))
    expected = model.clip(model.global_mean + model.item_bias[positions])
    assert np.allclose(model.predict(NEW_USER, np.array([1, 2])), expected)
