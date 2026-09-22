"""What the project is and how the test works."""

import html

import streamlit as st

import shared

engine = shared.get_engine()

st.title("About")
st.write("This is a course project comparing ways of recommending movies. The point isn't "
         "to build the best recommender - it's to run several approaches through exactly "
         "the same test and see where each one does well and where it falls down.")

st.subheader("The test")
st.markdown("""
- **Data:** MovieLens 100K - 100,000 ratings of 1,682 movies by 943 people, collected in
  1997-98. Everyone rated at least 20 movies.
- **Hidden ratings:** each person's most recent 20% of ratings were hidden from every
  method. Hiding the most recent ones, rather than a random sample, stops a method from
  "seeing the future" - the Experiments page shows how much a random split would have
  flattered every method.
- **Scoring:** a recommendation counts as a hit if the person rated it 4 or 5 stars later.
  Methods are also scored on how close their predicted ratings are (RMSE), how much of the
  catalogue they ever recommend, and how mainstream their picks are.
- **Same rules for everyone:** every method goes through the same evaluation code, and a
  test checks that none of them gets special handling.
""")

st.subheader("The methods")
for name in engine.model_names:
    st.markdown("**" + html.escape(shared.model_label(name)) + "** - "
                + html.escape(shared.MODEL_NOTES.get(name, "")))

st.subheader("Recommending for new people")
st.write("The methods were trained on the 943 users. When you rate movies on the Recommend "
         "page nothing is retrained: each trained model is copied and you're added to the "
         "copy, working out your taste from your ratings while everything it learned about "
         "the movies stays fixed. For most methods that gives exactly the result retraining "
         "would; for matrix factorisation it solves for you directly instead of re-running "
         "training. The copy is discarded afterwards.")

st.subheader("Limits")
st.write("The catalogue stops in 1998, so newer movies can't be recommended or matched from "
         "a Letterboxd export. Posters, where shown, come from TMDB and are only decoration - "
         "they play no part in any recommendation.")
shared.footer(shared.get_posters())
