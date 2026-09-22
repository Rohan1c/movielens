"""Entry point for the demo app.

    streamlit run app/Home.py

Sets up the page once and routes to the views in app/views/.
"""

import streamlit as st

import shared

st.set_page_config(page_title="MovieLens recommender", layout="wide")
st.markdown(shared.CSS, unsafe_allow_html=True)

navigation = st.navigation([
    st.Page("views/recommend.py", title="Recommend", default=True),
    st.Page("views/users.py", title="Browse users"),
    st.Page("views/results.py", title="Results"),
    st.Page("views/experiments.py", title="Experiments"),
    st.Page("views/about.py", title="About"),
])
navigation.run()
