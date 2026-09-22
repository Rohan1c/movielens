"""The recommendation engine: the evaluated models, served to real people.

Everything else in this repository measures the models. This module uses them. It fits
each registered model once, then answers two kinds of request:

``recommend_existing(user_id)``
    Recommendations for one of the 943 MovieLens users, with the films that user went on
    to rate highly marked, so a person can see which picks were right.

``recommend_new(ratings)``
    Recommendations for someone who is not in the dataset at all, from a handful of films
    they have rated. Each model is forked and the new person folded in, so nothing is
    retrained and the fitted models are never modified.

The models are built through the same registry the experiments use and fitted on the same
training split, so the engine serves exactly the models the results table describes. It
does not produce any reported number.
"""

import numpy as np
import pandas as pd

from src.data.splitting import load_splits
from src.evaluation.ranking import item_popularity
from src.evaluation.ranking import relevant_items_by_user
from src.experiments.recommend import build_item_lookup
from src.experiments.registry import MODEL_ORDER
from src.experiments.registry import build_model
from src.experiments.run_main import load_items
from src.experiments.run_qualitative import match_history
from src.experiments.run_qualitative import normalise_title

SESSION_USER = 1000000
DEFAULT_K = 10


class RecommendationEngine:
    """Fitted models plus the lookups a front end needs."""

    def __init__(self, config, train, validation, test, items, model_names=None,
                 cache=True):
        self.config = config
        self.train = train
        self.test = test
        self.items = items
        if model_names is None:
            model_names = list(MODEL_ORDER)
        self.model_names = list(model_names)
        self.lookup = build_item_lookup(items)
        self.popularity = item_popularity(train)
        threshold = float(config.section("evaluation")["relevance_threshold"])
        self.held_out_likes = relevant_items_by_user(test, threshold)
        self.years = {}
        for _, row in items.iterrows():
            self.years[int(row["item_id"])] = int(row["release_year"])

        self.models = {}
        for name in self.model_names:
            model = build_model(name, config, items, validation=validation, cache=cache)
            self.models[name] = model.fit(train)

    @classmethod
    def from_config(cls, config, model_names=None, cache=True):
        train, validation, test = load_splits(config.path("processed_dir"))
        return cls(config, train, validation, test, load_items(config),
                   model_names=model_names, cache=cache)

    # -- catalogue -------------------------------------------------------------

    def title(self, item_id):
        entry = self.lookup.get(int(item_id))
        if entry is None:
            return "item " + str(item_id)
        return entry["label"]

    def genres(self, item_id):
        entry = self.lookup.get(int(item_id))
        if entry is None:
            return []
        return list(entry["genres"])

    def catalogue(self):
        """Every rateable film, most rated first, for a picker in the front end."""
        rows = []
        for item_id in self.lookup.keys():
            rows.append({
                "item_id": item_id,
                "title": self.title(item_id),
                "ratings": int(self.popularity.get(item_id, 0)),
            })
        frame = pd.DataFrame(rows)
        return frame.sort_values(["ratings", "item_id"], ascending=[False, True])

    def search(self, query, limit=20):
        """Films whose title contains every word of the query, most rated first."""
        words = normalise_title(query).split()
        catalogue = self.catalogue()
        matches = []
        for _, row in catalogue.iterrows():
            key = normalise_title(row["title"])
            found = True
            for word in words:
                if word not in key:
                    found = False
                    break
            if found:
                matches.append(int(row["item_id"]))
            if len(matches) >= limit:
                break
        return matches

    # -- users -----------------------------------------------------------------

    def users(self):
        return sorted(int(value) for value in self.train["user_id"].unique())

    def history(self, user_id):
        rated = self.train[self.train["user_id"] == int(user_id)].copy()
        rated["title"] = [self.title(item) for item in rated["item_id"]]
        return rated.sort_values(["rating", "item_id"], ascending=[False, True])

    def likes_after(self, user_id):
        return set(self.held_out_likes.get(int(user_id), set()))

    # -- recommendations -------------------------------------------------------

    def describe(self, model, user_id, k, hits):
        recommendations = model.recommend(user_id, k)
        predictions = model.predict(user_id, recommendations)
        rows = []
        for position, item_id in enumerate(recommendations):
            item_id = int(item_id)
            rows.append({
                "rank": position + 1,
                "item_id": item_id,
                "title": self.title(item_id),
                "year": self.years.get(item_id, 0),
                "genres": self.genres(item_id),
                "predicted": float(predictions[position]),
                "ratings": int(self.popularity.get(item_id, 0)),
                "hit": item_id in hits,
            })
        return rows

    def recommend_existing(self, user_id, k=DEFAULT_K):
        if int(user_id) not in set(self.users()):
            raise ValueError("user " + str(user_id) + " is not in the training data")
        hits = self.likes_after(user_id)
        results = {}
        for name in self.model_names:
            results[name] = self.describe(self.models[name], user_id, k, hits)
        return results

    def recommend_new(self, ratings, k=DEFAULT_K):
        """Recommendations for someone outside the dataset.

        ``ratings`` maps item id to a 1-5 rating. Each model is forked before the new
        person is folded in, so repeated requests never accumulate inside the trained
        models.
        """
        frame = ratings_frame(ratings)
        if len(frame) == 0:
            raise ValueError("rate at least one film to get recommendations")
        results = {}
        for name in self.model_names:
            model = self.models[name].fork()
            model.fold_in(SESSION_USER, frame)
            results[name] = self.describe(model, SESSION_USER, k, set())
        return results

    def match_export(self, history):
        """Match an exported watch history, such as Letterboxd's, onto the catalogue."""
        matched, unmatched = match_history(history, self.items, 1.0)
        ratings = {}
        for _, row in matched.iterrows():
            rating = row["rating"]
            if rating is None or pd.isna(rating):
                rating = 4.0
            ratings[int(row["item_id"])] = float(np.clip(rating, 1.0, 5.0))
        return ratings, unmatched


def ratings_frame(ratings):
    rows = []
    for item_id, rating in ratings.items():
        rows.append({"item_id": int(item_id), "rating": float(rating)})
    return pd.DataFrame(rows, columns=["item_id", "rating"])
