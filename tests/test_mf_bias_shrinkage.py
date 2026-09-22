"""Tests for frequency-aware bias shrinkage in matrix factorisation.

Plain per-rating L2 shrinks every bias by the same factor, 1 / (1 + lambda), whatever the
number of ratings behind it. These tests pin down the replacement: a thinly supported
item has to be pulled towards zero much harder than a well supported one.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.mf import MatrixFactorization


@pytest.fixture
def uneven_support():
    """Item 1 has thirty ratings of 5; item 2 has a single rating of 5.

    Both have the same raw mean. Any sensible shrinkage has to trust item 1 far more.
    Filler items keep the global mean well below 5 so there is a deviation to shrink.
    """
    rows = []
    timestamp = 100
    for user_id in range(1, 31):
        rows.append((user_id, 1, 5.0, timestamp))
        rows.append((user_id, 3, 2.0, timestamp + 1))
        rows.append((user_id, 4, 2.0, timestamp + 2))
        timestamp = timestamp + 3
    rows.append((1, 2, 5.0, timestamp))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def fit(config, ratings, beta):
    """Train to convergence so the comparison is between fixed points.

    A one-rating item gets a single SGD update per epoch, so at the project's usual
    learning rate it is still a long way short of its fixed point after a few dozen
    epochs. That slow convergence partly hides the plain-L2 defect for items rated once -
    on the real data it is items with 6-20 ratings that end up with the largest biases -
    so these tests use a larger step and enough epochs for every bias to settle.
    """
    model = MatrixFactorization(
        config, n_factors=2, learning_rate=0.05, n_epochs=300, bias_shrinkage=beta
    )
    return model.fit(ratings)


def bias_of(model, item_id):
    return float(model.item_bias[model.rating_matrix.item_position[item_id]])


def test_zero_keeps_plain_l2(config, uneven_support):
    model = fit(config, uneven_support, 0.0)
    assert np.allclose(model.item_bias_reg, model.regularisation)
    assert np.allclose(model.user_bias_reg, model.regularisation)


def test_penalty_scales_inversely_with_support(config, uneven_support):
    model = fit(config, uneven_support, 10.0)
    heavy = model.item_bias_reg[model.rating_matrix.item_position[1]]
    light = model.item_bias_reg[model.rating_matrix.item_position[2]]
    assert heavy == pytest.approx(10.0 / 30.0)
    assert light == pytest.approx(10.0 / 1.0)


def test_plain_l2_barely_distinguishes_one_rating_from_thirty(config, uneven_support):
    """The defect: at convergence a single 5-star rating earns nearly the same bias."""
    model = fit(config, uneven_support, 0.0)
    ratio = bias_of(model, 2) / bias_of(model, 1)
    assert ratio > 0.7


def test_shrinkage_pulls_the_single_rating_item_towards_zero(config, uneven_support):
    plain = fit(config, uneven_support, 0.0)
    shrunk = fit(config, uneven_support, 10.0)
    assert abs(bias_of(shrunk, 2)) < abs(bias_of(plain, 2))
    assert bias_of(shrunk, 2) < 0.5 * bias_of(shrunk, 1)


def test_a_well_supported_item_is_still_predicted_high(config, uneven_support):
    """Judged on the prediction, not the bias alone.

    When the bias is penalised harder the latent factors pick up part of a strong,
    consistent item effect, so the bias value by itself shrinks further than the n/(n+beta)
    formula suggests. What matters is that the model still predicts the item well.
    """
    shrunk = fit(config, uneven_support, 10.0)
    assert shrunk.predict(30, np.array([1]))[0] > 4.0


def test_the_well_supported_item_now_ranks_first(config, uneven_support):
    """With equal raw means, the item with more evidence should be the recommendation."""
    model = fit(config, uneven_support, 10.0)
    scores = model.predict(30, np.array([1, 2]))
    assert scores[0] > scores[1]


def test_different_shrinkage_does_not_share_a_cache_entry(config, uneven_support, tmp_path):
    config.values["paths"]["fits_dir"] = str(tmp_path / "fits")
    MatrixFactorization(config, n_epochs=3, bias_shrinkage=0.0, cache=True).fit(uneven_support)
    other = MatrixFactorization(config, n_epochs=3, bias_shrinkage=10.0, cache=True)
    other.fit(uneven_support)
    assert other.loaded_from_cache is False


def test_shrinkage_is_read_from_the_config_when_not_given(config):
    config.values["models"]["mf"]["bias_shrinkage"] = 7.0
    assert MatrixFactorization(config).bias_shrinkage == pytest.approx(7.0)


def test_a_config_without_the_key_defaults_to_plain_l2(config):
    config.values["models"]["mf"].pop("bias_shrinkage", None)
    assert MatrixFactorization(config).bias_shrinkage == 0.0
