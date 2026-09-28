"""Figures: interactive Plotly HTML for exploration, matplotlib PNG for the report.

Each figure is produced from the persisted artefacts of the earlier stages, so the
figure stage can be re-run on its own without retraining anything.
"""

from __future__ import annotations

import functools
import json
import logging
import re

import matplotlib

matplotlib.use("Agg")  # headless: no display is available in CI or on a server

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve

from . import (
    clean,
    cluster_stations,
    collect_lamp,
    collect_v3,
    config,
    features,
    model_delay,
)

log = logging.getLogger(__name__)

PLOTLY_TEMPLATE = "plotly_white"
FIGURES = config.FIGURES_DIR

#: Cluster colours reused across figures so a station cluster means the same thing
#: everywhere in the report.
CLUSTER_PALETTE = px.colors.qualitative.Bold


def _write_html(figure: go.Figure, name: str) -> str:
    """Write a self-contained interactive figure.

    The Plotly runtime is referenced from a CDN rather than inlined: embedding it
    per figure would add ~3 MB to each file for no benefit.
    """
    path = FIGURES / f"{name}.html"
    figure.write_html(path, include_plotlyjs="cdn", full_html=True)
    log.info("wrote %s", path.name)
    return path.name


def _write_png(figure, name: str, dpi: int = 150) -> str:
    path = FIGURES / f"{name}.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    log.info("wrote %s", path.name)
    return path.name


@functools.lru_cache(maxsize=1)
def _clean_table() -> pd.DataFrame:
    """Genuine arrivals from the clean table, read once per figure run.

    Callers must treat it as read-only; every current caller filters or
    aggregates into a new frame.
    """
    frame = clean.load()
    return frame[clean.is_arrival(frame)]


@functools.lru_cache(maxsize=1)
def _station_hour_median_delay() -> pd.DataFrame:
    """Station x hour median delay, rows ordered from most to least late."""
    pivot = _clean_table().pivot_table(
        index="station_name", columns="scheduled_hour",
        values="delay_seconds", aggfunc="median",
    ).reindex(columns=range(24))
    return pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]


def _period_columns(frame: pd.DataFrame) -> list[str]:
    """The ``p00``..``p47`` half-hour share columns of a demand-profile table."""
    return [c for c in frame.columns if re.fullmatch(r"p\d{2}", str(c))]


def _sample(frame: pd.DataFrame, n: int, seed: int = 506) -> pd.DataFrame:
    """Randomly subsample a frame for plotting.

    Plotly embeds every plotted point in the HTML file. A violin or box trace over
    millions of rows produces a 60 MB document, so the distribution figures are
    drawn from a bounded sample -- the shape of a distribution is thoroughly
    captured well below the full row count.
    """
    if len(frame) <= n:
        return frame
    return frame.sample(n, random_state=seed)


def _downsample_curve(x, y, limit: int = 400):
    """Thin a dense curve to at most ``limit`` points for plotting."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) <= limit:
        return x, y
    step = -(-len(x) // limit)  # ceiling division, so the result is <= limit
    index = np.arange(0, len(x), step)
    # Always keep the final point so the curve ends where it should (a ROC curve
    # must reach (1, 1); striding alone usually skips it).
    if index[-1] != len(x) - 1:
        index = np.append(index, len(x) - 1)
    return x[index], y[index]


# ---------------------------------------------------------------------------
# Network / reliability figures
# ---------------------------------------------------------------------------
def figure_station_map() -> tuple[go.Figure, str]:
    """Map stations coloured by reliability cluster and sized by daily entries."""
    clusters = cluster_stations.load_clusters()
    try:
        stations = collect_v3.load_stations()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"station coordinates unavailable: {exc}") from exc

    merged = clusters.merge(
        stations[["stop_name", "latitude", "longitude"]],
        left_on="station_name", right_on="stop_name", how="left",
    ).dropna(subset=["latitude", "longitude"])
    if merged.empty:
        raise RuntimeError("no stations could be matched to coordinates")

    merged["total_entries"] = merged.get("total_entries", pd.Series(np.nan)).fillna(0)
    size = merged["total_entries"].clip(lower=1)
    merged["_size"] = 6 + 26 * (size / size.max()) ** 0.5

    figure = px.scatter_map(
        merged,
        lat="latitude",
        lon="longitude",
        color="reliability_cluster",
        size="_size",
        hover_name="station_name",
        hover_data={
            "mean_delay": ":.0f",
            "on_time_rate": ":.2f",
            "total_entries": ":,.0f",
            "latitude": False,
            "longitude": False,
            "_size": False,
        },
        color_discrete_sequence=CLUSTER_PALETTE,
        zoom=10.5,
        height=720,
        title="MBTA subway stations by reliability cluster "
              "(marker size = mean daily gated entries)",
        labels={"reliability_cluster": "Reliability", "mean_delay": "Mean delay (s)",
                "on_time_rate": "On-time rate", "total_entries": "Daily entries"},
        map_style="carto-positron",
    )
    figure.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=0, r=0, t=60, b=0))
    return figure, "station_map"


def figure_delay_heatmap() -> tuple[go.Figure, str]:
    """Station x hour heatmap of median delay, ordered by overall lateness."""
    pivot = _station_hour_median_delay()

    figure = go.Figure(go.Heatmap(
        z=pivot.to_numpy(),
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index,
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(title="Median<br>delay (s)"),
        hovertemplate="%{y}<br>%{x}<br>median delay %{z:.0f}s<extra></extra>",
    ))
    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Median arrival delay by station and scheduled hour "
              "(red = late, blue = early)",
        xaxis_title="Scheduled hour of day",
        yaxis_title="",
        height=max(600, 16 * len(pivot)),
        margin=dict(l=180, r=40, t=70, b=50),
    )
    return figure, "delay_heatmap"


def figure_delay_by_route() -> tuple[go.Figure, str]:
    """Delay distribution by route, clipped to a readable window."""
    frame = _clean_table()
    subset = frame[frame["delay_seconds"].between(-600, 1800)]
    subset = _sample(subset, 30_000)
    figure = px.violin(
        subset,
        x="route_id",
        y="delay_seconds",
        color="route_id",
        box=True,
        points=False,
        color_discrete_sequence=CLUSTER_PALETTE,
        height=560,
        title="Arrival delay distribution by route (clipped to -10 to +30 minutes)",
        labels={"route_id": "Route", "delay_seconds": "Delay (seconds)"},
    )
    figure.add_hline(y=300, line_dash="dash", line_color="firebrick",
                     annotation_text="5-minute lateness threshold")
    figure.update_layout(template=PLOTLY_TEMPLATE, showlegend=False)
    return figure, "delay_by_route"


def figure_headways() -> tuple[go.Figure, str]:
    """Observed headway distribution by route versus the scheduled headway."""
    frame = _clean_table()
    subset = frame.dropna(subset=["headway_trunk_seconds", "scheduled_headway_trunk"])
    subset = subset[subset["headway_trunk_seconds"].between(0, 1800)]
    subset = _sample(subset, 20_000)

    figure = go.Figure()
    for route in sorted(subset["route_id"].unique()):
        part = subset.loc[subset["route_id"] == route, "headway_trunk_seconds"]
        figure.add_trace(go.Box(
            y=part, name=route, boxmean=True, marker_color=None,
        ))
    figure.add_hline(y=float(subset["scheduled_headway_trunk"].median()),
                     line_dash="dash", line_color="grey",
                     annotation_text="median scheduled headway")
    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Observed headway by route (bunched service shows as a long upper tail)",
        yaxis_title="Headway (seconds)",
        xaxis_title="Route",
        height=560,
        showlegend=False,
    )
    return figure, "headways"


# ---------------------------------------------------------------------------
# Model figures
# ---------------------------------------------------------------------------
def figure_ablation() -> tuple[go.Figure, str]:
    """Test MAE as feature groups are added cumulatively."""
    ablation = pd.read_csv(model_delay.ABLATION_PATH)
    ablation = ablation.sort_values("n_features")
    baseline = float(ablation.iloc[0]["mae_seconds"])

    figure = go.Figure(go.Bar(
        x=ablation["features"],
        y=ablation["mae_seconds"],
        text=[f"{v:.0f}s" for v in ablation["mae_seconds"]],
        textposition="outside",
        marker_color=["#c44e52" if v > 200 else "#4c72b0" for v in ablation["mae_seconds"]],
    ))
    figure.add_hline(y=baseline, line_dash="dot", line_color="grey",
                     annotation_text=f"schedule-only baseline ({baseline:.0f}s)")
    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Ablation: test MAE as feature groups are added (histogram gradient boosting)",
        yaxis_title="Mean absolute error (seconds)",
        xaxis_title="Feature set",
        height=560,
    )
    return figure, "ablation"


def figure_importance() -> tuple[go.Figure, str]:
    """Permutation importance of the best regression model."""
    importance = pd.read_csv(model_delay.IMPORTANCE_PATH).head(15)
    importance = importance.sort_values("importance_mae")

    figure = go.Figure(go.Bar(
        x=importance["importance_mae"],
        y=importance["feature"],
        orientation="h",
        error_x=dict(type="data", array=importance["importance_std"]),
        marker_color="#4c72b0",
        hovertemplate="%{y}<br>+%{x:.1f}s MAE when shuffled<extra></extra>",
    ))
    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Permutation importance: added test MAE when a feature is shuffled",
        xaxis_title="Increase in MAE (seconds)",
        yaxis_title="",
        height=560,
        margin=dict(l=200, r=40, t=70, b=50),
    )
    return figure, "feature_importance"


def figure_predicted_vs_actual(sample: int = 25_000) -> tuple[go.Figure, str]:
    """Predicted versus actual delay with a calibrated residual view."""
    predictions = pd.read_parquet(model_delay.PREDICTIONS_PATH)
    subset = predictions.sample(min(sample, len(predictions)), random_state=model_delay.SEED)
    clipped = subset[subset["delay_seconds"].between(-900, 2400)
                     & subset["predicted_delay"].between(-900, 2400)]

    figure = px.density_heatmap(
        clipped, x="delay_seconds", y="predicted_delay",
        nbinsx=90, nbinsy=90, color_continuous_scale="Viridis",
        height=620,
        title="Predicted versus actual arrival delay (density of test observations)",
        labels={"delay_seconds": "Actual delay (seconds)",
                "predicted_delay": "Predicted delay (seconds)"},
    )
    limit = 2400
    figure.add_trace(go.Scatter(
        x=[-900, limit], y=[-900, limit], mode="lines",
        line=dict(color="firebrick", dash="dash"), name="perfect prediction",
    ))
    figure.update_layout(template=PLOTLY_TEMPLATE)
    return figure, "predicted_vs_actual"


def figure_roc_pr() -> tuple[go.Figure, str]:
    """ROC and precision-recall curves for the lateness classifiers."""
    predictions = pd.read_parquet(model_delay.CLF_PREDICTIONS_PATH)
    y_true = predictions["late"].astype(int).to_numpy()

    figure = go.Figure()
    for column in predictions.columns:
        if not column.startswith("proba_"):
            continue
        name = column.removeprefix("proba_")
        scores = predictions[column].to_numpy()
        fpr, tpr, _ = roc_curve(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        # A ROC curve over 170k rows carries 170k vertices; thinning it keeps the
        # HTML small without changing the curve's appearance.
        fpr_t, tpr_t = _downsample_curve(fpr, tpr)
        recall_t, precision_t = _downsample_curve(recall, precision)
        figure.add_trace(go.Scatter(x=fpr_t, y=tpr_t, mode="lines",
                                    name=f"{name} (ROC)"))
        figure.add_trace(go.Scatter(x=recall_t, y=precision_t, mode="lines",
                                    name=f"{name} (PR)", line=dict(dash="dot")))

    figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                line=dict(color="grey", dash="dash"),
                                name="chance"))
    base_rate = float(y_true.mean())
    figure.add_hline(y=base_rate, line_dash="dot", line_color="grey",
                     annotation_text=f"late base rate ({base_rate:.1%})")
    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        title="ROC (solid) and precision-recall (dotted) for lateness classification",
        xaxis_title="False positive rate  /  Recall",
        yaxis_title="True positive rate  /  Precision",
        height=620,
    )
    return figure, "roc_pr"


def figure_error_by_hour_and_route() -> tuple[go.Figure, str]:
    """Where the model's error concentrates."""
    metrics = model_delay.load_metrics()
    analysis = metrics["error_analysis"]

    by_hour = pd.DataFrame(analysis["by_hour_band"])
    by_route = pd.DataFrame(analysis["by_route"])

    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=by_hour["hour_band"], y=by_hour["mae"], name="by hour band",
        marker_color="#4c72b0",
        customdata=by_hour[["n"]], hovertemplate="%{x}<br>MAE %{y:.0f}s<br>n=%{customdata[0]}<extra></extra>",
    ))
    figure.add_trace(go.Bar(
        x=by_route["route_id"], y=by_route["mae"], name="by route",
        marker_color="#dd8452",
        customdata=by_route[["n"]], hovertemplate="%{x}<br>MAE %{y:.0f}s<br>n=%{customdata[0]}<extra></extra>",
    ))
    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Mean absolute error by time-of-day band and by route",
        yaxis_title="Mean absolute error (seconds)",
        height=520, barmode="group",
    )
    return figure, "error_breakdown"


def figure_demand_profiles() -> tuple[go.Figure, str]:
    """Mean demand shape for each demand cluster across the 48 half-hour periods."""
    path = config.PROCESSED_DIR / "demand_profiles.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run the `cluster` stage first")
    profiles = pd.read_parquet(path)
    # Group on the numeric id and label with the name, so distinct clusters stay
    # distinct even if two of them were given the same descriptive name.
    profiles["label"] = (profiles["demand_cluster_id"].astype(str)
                         + ": " + profiles["demand_cluster"].astype(str))

    period_columns = _period_columns(profiles)
    grouped = profiles.groupby("label")[period_columns].mean()
    hours = [int(c[1:]) / 2 for c in period_columns]

    figure = go.Figure()
    for cluster, row in grouped.iterrows():
        figure.add_trace(go.Scatter(
            x=hours, y=row.to_numpy() * 100, mode="lines+markers", name=cluster,
            hovertemplate="%{x:.1f}h<br>%{y:.2f}% of daily entries<extra></extra>",
        ))
    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Ridership demand shape by station cluster (share of daily gated entries)",
        xaxis_title="Hour of day",
        yaxis_title="Share of daily entries (%)",
        height=560,
    )
    return figure, "demand_profiles"


def figure_cluster_scatter() -> tuple[go.Figure, str]:
    """Mean delay versus daily entries, coloured by demand cluster."""
    clusters = cluster_stations.load_clusters()
    if "demand_cluster" not in clusters.columns:
        raise ValueError("demand clustering not available")

    subset = clusters.dropna(subset=["demand_cluster", "total_entries"])
    figure = px.scatter(
        subset,
        x="total_entries", y="mean_delay",
        color="demand_cluster", symbol="reliability_cluster",
        hover_name="station_name",
        color_discrete_sequence=CLUSTER_PALETTE,
        height=620,
        title="Station demand versus mean delay, coloured by demand cluster "
              "and shaped by reliability cluster",
        labels={"total_entries": "Mean daily gated entries",
                "mean_delay": "Mean delay (seconds)",
                "demand_cluster": "Demand cluster",
                "reliability_cluster": "Reliability cluster"},
    )
    figure.add_hline(y=0, line_dash="dash", line_color="grey")
    figure.update_layout(template=PLOTLY_TEMPLATE)
    return figure, "cluster_scatter"


def figure_live_vs_historical() -> tuple[go.Figure, str] | None:
    """Compare live-collected predictions against the historical delay profile.

    Returns ``None`` when the live collector has not been run, because there is
    nothing honest to plot in that case.
    """
    try:
        live = collect_v3.live_prediction_delays()
    except FileNotFoundError:
        return None
    if live.empty:
        return None

    revisions = live["arrival_revision_seconds"].dropna()
    revisions = revisions[revisions.abs() <= 1800]
    if revisions.empty:
        return None

    frame = _clean_table()
    # The historical column has millions of rows; plotting all of them would embed
    # every point in the HTML. A probability-density histogram is well estimated
    # from a large sample.
    historical = _sample(frame, 100_000)["delay_seconds"].clip(-600, 1800)

    figure = go.Figure()
    figure.add_trace(go.Histogram(
        x=historical, name="historical observed delay",
        opacity=0.6, marker_color="#4c72b0", histnorm="probability density",
    ))
    figure.add_trace(go.Histogram(
        x=revisions, name="live prediction revision between polls", opacity=0.6,
        marker_color="#dd8452", histnorm="probability density",
    ))
    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="overlay",
        title="Live API prediction volatility versus the historical delay distribution",
        xaxis_title="Seconds",
        yaxis_title="Density",
        height=560,
    )
    return figure, "live_vs_historical"


# ---------------------------------------------------------------------------
# matplotlib companions for the README
# ---------------------------------------------------------------------------
def png_ablation() -> str:
    ablation = pd.read_csv(model_delay.ABLATION_PATH).sort_values("n_features")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#c44e52" if v > 200 else "#4c72b0" for v in ablation["mae_seconds"]]
    ax.bar(ablation["features"], ablation["mae_seconds"], color=colors)
    for i, value in enumerate(ablation["mae_seconds"]):
        ax.text(i, value + 6, f"{value:.0f}", ha="center", fontsize=8)
    ax.set_ylabel("Test MAE (seconds)")
    ax.set_title("Feature ablation: test MAE by feature set")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    return _write_png(fig, "ablation")


def png_importance() -> str:
    importance = pd.read_csv(model_delay.IMPORTANCE_PATH).head(12).sort_values("importance_mae")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance["feature"], importance["importance_mae"],
            xerr=importance["importance_std"], color="#4c72b0")
    ax.set_xlabel("Increase in test MAE when shuffled (seconds)")
    ax.set_title("Permutation importance of the delay model")
    fig.tight_layout()
    return _write_png(fig, "feature_importance")


def png_predicted_vs_actual() -> str:
    predictions = pd.read_parquet(model_delay.PREDICTIONS_PATH)
    subset = predictions[predictions["delay_seconds"].between(-900, 2400)
                         & predictions["predicted_delay"].between(-900, 2400)]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.hexbin(subset["delay_seconds"], subset["predicted_delay"],
              gridsize=60, cmap="viridis", bins="log")
    ax.plot([-900, 2400], [-900, 2400], "r--", linewidth=1)
    ax.set_xlabel("Actual delay (seconds)")
    ax.set_ylabel("Predicted delay (seconds)")
    ax.set_title("Predicted versus actual arrival delay")
    fig.colorbar(ax.collections[0], ax=ax, label="Observations per cell (log scale)")
    fig.tight_layout()
    return _write_png(fig, "predicted_vs_actual")


def png_demand_profiles() -> str:
    path = config.PROCESSED_DIR / "demand_profiles.parquet"
    profiles = pd.read_parquet(path)
    profiles["label"] = (profiles["demand_cluster_id"].astype(str)
                         + ": " + profiles["demand_cluster"].astype(str))
    period_columns = _period_columns(profiles)
    grouped = profiles.groupby("label")[period_columns].mean()
    hours = [int(c[1:]) / 2 for c in period_columns]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cluster, row in grouped.iterrows():
        ax.plot(hours, row.to_numpy() * 100, marker="o", markersize=3, label=cluster)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Share of daily entries (%)")
    ax.set_title("Ridership demand shape by station cluster")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _write_png(fig, "demand_profiles")


def png_delay_heatmap() -> str:
    pivot = _station_hour_median_delay()

    fig, ax = plt.subplots(figsize=(11, 12))
    limit = float(np.nanpercentile(np.abs(pivot.to_numpy()), 95)) or 300.0
    mesh = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r",
                     vmin=-limit, vmax=limit)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=5)
    ax.set_xlabel("Scheduled hour of day")
    ax.set_title("Median arrival delay by station and hour (seconds)")
    fig.colorbar(mesh, ax=ax, label="Median delay (s)", shrink=0.6)
    fig.tight_layout()
    return _write_png(fig, "delay_heatmap")


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run() -> dict:
    """Render every figure that the available artefacts support."""
    config.ensure_dirs()
    html: list[str] = []
    png: list[str] = []
    skipped: dict[str, str] = {}

    interactive = [
        figure_station_map,
        figure_delay_heatmap,
        figure_delay_by_route,
        figure_headways,
        figure_ablation,
        figure_importance,
        figure_predicted_vs_actual,
        figure_roc_pr,
        figure_error_by_hour_and_route,
        figure_demand_profiles,
        figure_cluster_scatter,
        figure_live_vs_historical,
    ]
    for builder in interactive:
        try:
            result = builder()
        except Exception as exc:  # noqa: BLE001 - one missing input must not stop the rest
            skipped[builder.__name__] = str(exc)
            log.warning("skipping %s: %s", builder.__name__, exc)
            continue
        if result is None:
            skipped[builder.__name__] = "no live data collected yet"
            continue
        figure, name = result
        html.append(_write_html(figure, name))

    for builder in (png_ablation, png_importance, png_predicted_vs_actual,
                    png_demand_profiles, png_delay_heatmap):
        try:
            png.append(builder())
        except Exception as exc:  # noqa: BLE001
            skipped[builder.__name__] = str(exc)
            log.warning("skipping %s: %s", builder.__name__, exc)

    manifest = {"html": html, "png": png, "skipped": skipped}
    (FIGURES / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("figures: %d interactive, %d static, %d skipped",
             len(html), len(png), len(skipped))
    return manifest
