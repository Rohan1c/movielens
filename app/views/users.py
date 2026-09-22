"""One of the 943 real users: what they rated, what they went on to like, who guessed it."""

import random

import plotly.graph_objects as go
import streamlit as st

import shared

engine = shared.get_engine()
posters = shared.get_posters()
users = engine.users()


def pick_random_user():
    st.session_state["user"] = random.choice(users)


if "user" not in st.session_state:
    st.session_state["user"] = 1

st.title("Browse users")
st.write("Every user's most recent ratings were hidden from the models. Pick someone to see "
         "what they rated before, what they rated highly afterwards, and which methods "
         "recommended those films. Films they went on to like are outlined in green.")

controls = st.columns([1, 1, 4])
controls[0].number_input("User", min_value=min(users), max_value=max(users), step=1, key="user")
controls[1].markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
controls[1].button("Random", on_click=pick_random_user, use_container_width=True)
user_id = int(st.session_state["user"])

history = engine.history(user_id)
liked = engine.likes_after(user_id)
st.markdown('<p class="small">' + str(len(history)) + " ratings before the cut-off, averaging "
            + format(float(history["rating"].mean()), ".1f") + " stars. "
            + str(len(liked)) + " of their later ratings were 4 or 5 stars.</p>",
            unsafe_allow_html=True)


def films_from(frame, limit):
    films = []
    for _, row in frame.head(limit).iterrows():
        item_id = int(row["item_id"])
        films.append({"title": engine.title(item_id), "year": engine.years.get(item_id, 0),
                      "rating": float(row["rating"]), "hit": False})
    return films


def stars_caption(film):
    return format(film["rating"], ".0f") + " stars"


favourites = films_from(history, 10)
later = engine.test[(engine.test["user_id"] == user_id) & (engine.test["item_id"].isin(liked))]
later = films_from(later.sort_values(["rating", "item_id"], ascending=[False, True]), 10)
posters.prefetch([(film["title"], film["year"]) for film in favourites + later])

st.subheader("Rated highly before the cut-off")
st.markdown(shared.poster_strip(favourites, posters, stars_caption), unsafe_allow_html=True)
st.subheader("Rated highly afterwards (hidden from the models)")
if len(later) == 0:
    st.caption("None - this user didn't give anything 4 or 5 stars after the cut-off.")
else:
    st.markdown(shared.poster_strip(later, posters, stars_caption), unsafe_allow_html=True)

st.subheader("What each method recommended")
methods = st.multiselect("Methods", options=engine.model_names, default=shared.DEFAULT_METHODS,
                         format_func=shared.model_label)
results = engine.recommend_existing(user_id, k=10)
shared.method_rows(results, methods, posters, show_hits=True)

st.subheader("Hits for this user, every method")
names = list(engine.model_names)
counts = []
for name in names:
    hits = 0
    for film in results[name]:
        if film["hit"]:
            hits = hits + 1
    counts.append(hits)
figure = go.Figure(go.Bar(x=[shared.model_label(name) for name in names], y=counts,
                          marker_color=[shared.MODEL_COLOURS.get(n, "#E05D44") for n in names],
                          text=counts, textposition="outside"))
figure.update_yaxes(title="liked later, out of 10", rangemode="tozero")
figure.update_xaxes(tickangle=-25)
st.plotly_chart(shared.style_chart(figure, 320), use_container_width=True)
st.caption("One user is one data point. Averaged over everyone, most-popular gets the most "
           "hits - see Results.")
shared.footer(posters)
