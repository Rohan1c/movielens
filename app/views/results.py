"""The main comparison. Every number is read from the committed CSVs."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import shared

METRICS = {
    "rmse": ("RMSE", "rating error - lower is better", False),
    "mae": ("MAE", "rating error - lower is better", False),
    "precision_at_k": ("Precision@k", "share of the top k the user liked later", True),
    "recall_at_k": ("Recall@k", "share of what they liked that made the top k", True),
    "coverage": ("Coverage@k", "share of the catalogue that shows up in anyone's top k", True),
    "popularity_percentile": ("Popularity@k", "how mainstream the recommendations are", True),
    "gini": ("Gini@k", "how unevenly recommendations are spread across films", True),
}
PER_K = ["precision_at_k", "recall_at_k", "coverage", "popularity_percentile", "gini"]
INTERVAL_METRICS = {"rmse": "RMSE", "precision_at_10": "precision@10"}

frame = shared.load_result("main_results.csv")
order = list(frame["model"].drop_duplicates())


def metric_name(key):
    return METRICS[key][0]


def series(metric, k):
    selected = frame[frame["metric"] == metric]
    if metric in PER_K:
        selected = selected[selected["k"] == k]
    return selected.set_index("model")["value"].reindex(order)


st.title("Results")
st.write("All eleven methods, scored the same way on the same held-out ratings. Pick two "
         "measures to plot against each other.")

controls = st.columns([2, 2, 1])
x_metric = controls[0].selectbox("Horizontal", list(METRICS.keys()),
                                 index=list(METRICS).index("popularity_percentile"),
                                 format_func=metric_name)
y_metric = controls[1].selectbox("Vertical", list(METRICS.keys()),
                                 index=list(METRICS).index("precision_at_k"),
                                 format_func=metric_name)
k = controls[2].radio("k", [5, 10, 20], index=1, horizontal=True)

x_values = series(x_metric, k)
y_values = series(y_metric, k)
figure = go.Figure()
for name in order:
    figure.add_trace(go.Scatter(
        x=[x_values[name]], y=[y_values[name]], mode="markers+text",
        marker=dict(size=13, color=shared.MODEL_COLOURS.get(name, "#E05D44")),
        text=[shared.model_label(name)], textposition="top center",
        hovertemplate=shared.model_label(name) + "<br>%{x:.4f}, %{y:.4f}<extra></extra>",
    ))
figure.update_layout(showlegend=False)
figure.update_xaxes(title=METRICS[x_metric][0] + " (" + METRICS[x_metric][1] + ")")
figure.update_yaxes(title=METRICS[y_metric][0] + " (" + METRICS[y_metric][1] + ")")
if not METRICS[y_metric][2]:
    figure.update_yaxes(autorange="reversed")
st.plotly_chart(shared.style_chart(figure, 480), use_container_width=True)
correlation = float(np.corrcoef(x_values.to_numpy(), y_values.to_numpy())[0, 1])
st.caption("r = " + format(correlation, ".2f") + " across these " + str(len(order))
           + " methods. When the vertical measure is lower-is-better the axis is flipped, so "
           "up always means better. The default view is the main finding: precision tracks "
           "how mainstream the recommendations are.")

st.subheader("Full table")
table = frame[(frame["metric"].isin(["rmse", "mae"])) | (frame["k"] == k)]
table = table.pivot_table(index="model", columns="metric", values="value").reindex(order)
table = table[[key for key in METRICS.keys() if key in table.columns]]
table.columns = [METRICS[key][0] for key in table.columns]
table.index = [shared.model_label(name) for name in table.index]
st.dataframe(table.style.format("{:.4f}"), use_container_width=True)

st.subheader("Which differences are real")
st.write("Users were resampled 1,000 times. If the 95% interval for a difference doesn't "
         "cross zero, the gap is unlikely to be luck.")
intervals = shared.load_result("bootstrap_ci.csv")
intervals = intervals[intervals["kind"] == "difference"]
against_popular = (intervals["model_a"] == "most_popular") | (intervals["model_b"] == "most_popular")
against_mf = ((intervals["model_a"] == "mf") | (intervals["model_b"] == "mf")) & (
    intervals["model_a"].isin(["item_knn", "hybrid", "hybrid_frontier", "mf_ranking"])
    | intervals["model_b"].isin(["item_knn", "hybrid", "hybrid_frontier", "mf_ranking"]))
rows = []
for _, row in intervals[against_popular | against_mf].iterrows():
    if str(row["excludes_zero"]) == "True":
        verdict = "yes"
    else:
        verdict = "no"
    rows.append({
        "Comparison": shared.model_label(row["model_a"]) + " vs " + shared.model_label(row["model_b"]),
        "Measure": INTERVAL_METRICS.get(row["metric"], row["metric"]),
        "Difference": round(float(row["difference"]), 4),
        "95% interval": "[" + format(row["lower"], ".4f") + ", " + format(row["upper"], ".4f") + "]",
        "Real": verdict,
    })
st.dataframe(rows, use_container_width=True, hide_index=True)
shared.footer(shared.get_posters())
