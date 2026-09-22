"""Recommendations for someone who is not in the dataset.

Nothing is retrained. Each model is forked and the visitor folded in from the films they
rate here - the same per-user computation training would have done for them.
"""

import pandas as pd
import streamlit as st

import shared

engine = shared.get_engine()
posters = shared.get_posters()

STARS = ["1", "2", "3", "4", "5"]
EXAMPLES = {
    "Sci-fi": ["Star Wars", "Empire Strikes Back", "Terminator, The", "Alien", "Blade Runner"],
    "Drama": ["Schindler's List", "Shawshank Redemption", "Casablanca", "Secrets & Lies",
              "Dead Man Walking"],
    "Comedy": ["Clerks", "Groundhog Day", "Mrs. Doubtfire", "Ace Ventura: Pet Detective",
               "Dumb & Dumber"],
    "Horror": ["Scream", "Shining, The", "Nightmare on Elm Street, A", "Carrie", "Alien"],
}


def resolve(titles):
    found = []
    for title in titles:
        matches = engine.search(title, limit=1)
        if len(matches) > 0 and matches[0] not in found:
            found.append(matches[0])
    return found


def use_example(name):
    ids = resolve(EXAMPLES[name])
    st.session_state["films"] = ids
    for item_id in ids:
        st.session_state["stars_" + str(item_id)] = "5"


def clear_films():
    st.session_state["films"] = []


def load_export():
    upload = st.session_state.get("export")
    if upload is None:
        return
    try:
        history = pd.read_csv(upload)
        ratings, unmatched = engine.match_export(history)
    except Exception as error:
        st.session_state["export_message"] = ("Couldn't read that file (" + type(error).__name__
                                              + "). Use ratings.csv from the Letterboxd export.")
        return
    current = list(st.session_state.get("films", []))
    for item_id, rating in ratings.items():
        if item_id not in current:
            current.append(item_id)
        stars = int(round(rating))
        stars = max(1, min(5, stars))
        st.session_state["stars_" + str(item_id)] = str(stars)
    st.session_state["films"] = current
    st.session_state["export_message"] = ("Found " + str(len(ratings)) + " of your "
                                          + str(len(history)) + " films in the catalogue.")
    st.session_state["export_unmatched"] = unmatched


st.title("MovieLens recommender")
st.write("Rate a few movies you've seen and see what each method in this project would "
         "recommend. The models have never seen you - you're added on the fly, so nothing is "
         "retrained and nothing you enter is saved.")

# Buttons in a horizontal container size to their labels and wrap on a narrow window,
# where fixed columns squeezed "Drama" into one letter per line.
examples = st.container(horizontal=True, horizontal_alignment="left",
                        vertical_alignment="center", gap="small")
examples.markdown('<span class="small">Try an example:</span>', unsafe_allow_html=True)
for name in EXAMPLES.keys():
    examples.button(name, on_click=use_example, args=(name,))
examples.button("Clear", on_click=clear_films, type="tertiary")

catalogue = engine.catalogue()
titles = dict(zip(catalogue["item_id"], catalogue["title"]))


def title_of(item_id):
    return titles.get(item_id, str(item_id))


left, right = st.columns([2, 1])
with left:
    picked = st.multiselect("Movies you've seen", options=list(catalogue["item_id"]),
                            format_func=title_of, key="films",
                            placeholder="Start typing a title...")
with right:
    st.file_uploader("Or upload your Letterboxd ratings.csv", type=["csv"], key="export",
                     on_change=load_export,
                     help="On Letterboxd: Settings > Data > Export your data, then upload "
                          "ratings.csv from the zip.")
    message = st.session_state.get("export_message")
    if message is not None:
        st.caption(message + " The catalogue stops at 1998, so newer films won't match.")
        unmatched = st.session_state.get("export_unmatched", [])
        if len(unmatched) > 0:
            with st.expander("Not found (" + str(len(unmatched)) + ")"):
                st.write(", ".join(unmatched[:200]))

ratings = {}
if len(picked) > 0:
    rating_columns = st.columns(4)
    for position, item_id in enumerate(picked):
        key = "stars_" + str(item_id)
        # Seeded through session state rather than a default value: an example or an upload
        # may already have set it, and giving both makes Streamlit print a warning.
        if key not in st.session_state:
            st.session_state[key] = "4"
        with rating_columns[position % 4]:
            choice = st.select_slider(title_of(item_id), options=STARS, key=key)
            ratings[item_id] = float(choice)

methods = st.multiselect("Methods", options=engine.model_names, default=shared.DEFAULT_METHODS,
                         format_func=shared.model_label)

if len(ratings) == 0:
    st.info("Pick an example above or add a few movies to get started.")
elif len(methods) == 0:
    st.info("Pick at least one method.")
else:
    results = engine.recommend_new(ratings, k=10)
    shared.method_rows(results, methods, posters, show_hits=False)
    if len(ratings) < 5:
        st.caption("With only " + str(len(ratings)) + " rating(s) there isn't much to go on "
                   "- the picks get more personal as you add more.")

shared.footer(posters)
