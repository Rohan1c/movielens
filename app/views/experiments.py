"""The four supporting experiments.

The sentences describing each result are built from the CSVs, not written by hand, so a
re-run that changes a result changes the description with it. An earlier version of this
page hard-coded claims that a later fix to matrix factorisation made false.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import shared
from src.models.hybrid import weight_for_history

PERSONALISED = ["content", "item_knn", "item_knn_ranking", "mf", "mf_ranking", "hybrid",
                "hybrid_frontier"]

st.title("Experiments")
cold, hybrid, tuning, split = st.tabs(["Cold start", "Hybrid", "Tuning", "Train/test split"])

with cold:
    frame = shared.load_result("cold_start.csv")
    rmse_at_one = frame[(frame["metric"] == "rmse") & (frame["history"] == 1)
                        & frame["model"].isin(PERSONALISED)].set_index("model")["value"]
    best = rmse_at_one.idxmin()
    worst = rmse_at_one.idxmax()
    st.write("200 users had their history cut down to 1, 3, 5, 10 and 20 ratings, everything "
             "was retrained at each level, and only those users were scored. With a single "
             "rating the most accurate personalised method was **"
             + shared.model_label(best) + "** (RMSE " + format(rmse_at_one[best], ".3f")
             + ") and the least accurate was **" + shared.model_label(worst) + "** (RMSE "
             + format(rmse_at_one[worst], ".3f") + ").")
    options = {"rmse": "RMSE (lower is better)", "precision_at_k": "precision@10",
               "coverage": "coverage@10"}

    def cold_label(key):
        return options[key]

    metric = st.radio("Measure", list(options.keys()), format_func=cold_label, horizontal=True)
    figure = go.Figure()
    for name in frame["model"].drop_duplicates():
        selected = frame[(frame["model"] == name) & (frame["metric"] == metric)]
        if metric != "rmse":
            selected = selected[selected["k"] == 10]
        selected = selected.sort_values("history")
        dash = "solid"
        if name in shared.DASHED:
            dash = "dash"
        figure.add_trace(go.Scatter(x=selected["history"], y=selected["value"],
                                    mode="lines+markers", name=shared.model_label(name),
                                    line=dict(color=shared.MODEL_COLOURS.get(name), dash=dash)))
    figure.update_xaxes(type="log", title="ratings kept per user", tickvals=[1, 3, 5, 10, 20])
    figure.update_yaxes(title=options[metric])
    st.plotly_chart(shared.style_chart(figure, 460), use_container_width=True)

with hybrid:
    weights = shared.load_result("hybrid_weights.csv")
    frontier = shared.load_result("hybrid_frontier.csv")
    intervals = shared.load_result("bootstrap_ci.csv")
    pure_cf = frontier[frontier["weight"] == 0.0].iloc[0]
    peak = frontier.loc[frontier["precision_at_10"].idxmax()]
    gain = peak["precision_at_10"] / pure_cf["precision_at_10"] - 1.0
    test_gap = intervals[(intervals["kind"] == "difference")
                         & (intervals["metric"] == "precision_at_10")
                         & intervals["model_a"].isin(["hybrid", "mf"])
                         & intervals["model_b"].isin(["hybrid", "mf"])]
    significant = False
    if len(test_gap) > 0 and str(test_gap.iloc[0]["excludes_zero"]) == "True":
        significant = True
    if significant:
        verdict = "and on the test set the difference from plain matrix factorisation is real."
    else:
        verdict = ("but on the test set the difference from plain matrix factorisation is not "
                   "statistically significant - the gain that holds up is catalogue coverage.")
    st.write("Blending content-based and matrix factorisation scores. On validation, the best "
             "blend (" + format(peak["weight"], ".1f") + " content) scored "
             + format(gain, "+.0%") + " precision@10 over pure matrix factorisation, "
             + verdict)
    left, right = st.columns(2)
    with left:
        intercept = float(weights["fitted_intercept"].iloc[0])
        slope = float(weights["fitted_slope"].iloc[0])
        histories = [1, 2, 5, 10, 20, 50, 100, 200, 400, 700]
        curve = []
        for history in histories:
            curve.append(weight_for_history(history, intercept, slope))
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=histories, y=curve, mode="lines", name="fitted",
                                    line=dict(color=shared.MODEL_COLOURS["hybrid"])))
        figure.add_trace(go.Scatter(x=weights["median_history"], y=weights["best_weight"],
                                    mode="markers", name="best per group",
                                    marker=dict(size=9, color="#E6E6E6")))
        figure.update_xaxes(type="log", title="ratings the user has")
        figure.update_yaxes(title="share given to content", range=[0, 1])
        figure.update_layout(title="Content weight by history length")
        st.plotly_chart(shared.style_chart(figure), use_container_width=True)
    with right:
        labels = []
        for value in frontier["weight"]:
            labels.append(format(value, ".1f"))
        figure = go.Figure(go.Scatter(
            x=frontier["coverage_at_10"], y=frontier["precision_at_10"],
            mode="lines+markers+text", text=labels, textposition="top center",
            line=dict(color=shared.MODEL_COLOURS["hybrid"]), marker=dict(size=8)))
        figure.update_xaxes(title="coverage@10")
        figure.update_yaxes(title="precision@10")
        figure.update_layout(title="Blend weight vs precision and coverage (validation)")
        st.plotly_chart(shared.style_chart(figure), use_container_width=True)

with tuning:
    st.write("Each dot is one hyperparameter setting for the same method. If a setting that "
             "predicts ratings better also ranks better, the dots slope down to the right.")
    columns = st.columns(2)
    sweeps = [("tuning_knn.csv", "Item-kNN", "item_knn"),
              ("tuning_mf.csv", "Matrix factorisation", "mf")]
    for column, (name, title, model) in zip(columns, sweeps):
        frame = shared.load_result(name)
        rmse = frame[frame["metric"] == "rmse"].set_index("variant")["value"]
        precision = frame[(frame["metric"] == "precision_at_k") & (frame["k"] == 10)]
        precision = precision.set_index("variant")["value"].reindex(rmse.index)
        r = float(np.corrcoef(rmse, precision)[0, 1])
        if r > 0.2:
            reading = "the two goals disagree - better ratings, worse rankings"
        elif r < -0.2:
            reading = "the two goals agree"
        else:
            reading = "no clear relationship"
        figure = go.Figure(go.Scatter(x=rmse, y=precision, mode="markers",
                                      marker=dict(size=8, color=shared.MODEL_COLOURS[model]),
                                      text=list(rmse.index),
                                      hovertemplate="%{text}<br>RMSE %{x:.4f}<br>"
                                                    "P@10 %{y:.4f}<extra></extra>"))
        figure.update_xaxes(title="validation RMSE")
        figure.update_yaxes(title="validation precision@10")
        figure.update_layout(title=title + " (" + str(len(rmse)) + " settings)")
        with column:
            st.plotly_chart(shared.style_chart(figure), use_container_width=True)
            st.caption("r = " + format(r, ".2f") + ": " + reading + ".")

with split:
    frame = shared.load_result("split_ablation.csv")
    st.write("The same pipeline run twice: once hiding each user's most recent ratings, once "
             "hiding a random sample. The random version lets later ratings leak into "
             "training, which makes every method look better than it is.")
    options = {"rmse": "RMSE", "precision_at_k": "precision@10"}

    def split_label(key):
        return options[key]

    metric = st.radio("Measure", list(options.keys()), format_func=split_label,
                      horizontal=True, key="split_metric")
    selected = frame[frame["metric"] == metric]
    if metric != "rmse":
        selected = selected[selected["k"] == 10]
    figure = go.Figure()
    for split_name, colour in [("temporal", "#E05D44"), ("random", "#6B6C72")]:
        rows = selected[selected["split"] == split_name]
        names = []
        for model in rows["model"]:
            names.append(shared.model_label(model))
        figure.add_trace(go.Bar(x=names, y=rows["value"], name=split_name, marker_color=colour))
    figure.update_layout(barmode="group")
    figure.update_xaxes(tickangle=-25)
    st.plotly_chart(shared.style_chart(figure, 420), use_container_width=True)

shared.footer(shared.get_posters())
