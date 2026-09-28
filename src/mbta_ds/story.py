"""The project for readers who will never open a notebook: charts and plain words.

``report.html`` is written for the marker: every table, every metric. This stage
writes ``reports/story.html``, one page that answers the questions a rider or a
non-technical reader would ask, in the order they would ask them:

1. How late do trains actually run, line by line?
2. When in the day is it worst?
3. Why can a train's delay be predicted at all? (It barely changes stop to stop.)
4. How good are the predictions, next to simple rules of thumb?
5. Do they survive a snowstorm?
6. What does the model pay attention to?
7. Can it warn about sudden big delays?
8. Which stations see the most late trains?

Every chart is interactive (hover for values), has a one-line takeaway above it,
and a "show the numbers" table beneath it. The same four key charts are also
saved as PNGs (``reports/figures/story_*.png``) so the README can show them on
GitHub, where scripts do not run.

Colours come from one validated palette: an ordinal blue ramp for "how late", two
categorical hues (blue, orange) for comparisons, grey for context. Dark mode uses
steps chosen for the dark surface, swapped in by a few lines of script.
"""

from __future__ import annotations

import html
import json
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version
from plotly.subplots import make_subplots

from . import clean, config

log = logging.getLogger(__name__)

OUT_PATH = config.REPORTS_DIR / "story.html"
FIGURES = config.FIGURES_DIR
SPRING = config.DATA_DIR / "processed"
WINTER = config.DATA_DIR / "runs" / "winter" / "processed"
WINTER_WEATHER = config.DATA_DIR / "runs" / "winter" / "weather" / "boston_hourly.parquet"

# ---------------------------------------------------------------------------
# Palette (validated: blue/orange pass every CVD and contrast check in both modes)
# ---------------------------------------------------------------------------
LIGHT = {"surface": "#fcfcfb", "page": "#f9f9f7", "text": "#0b0b0b", "text2": "#52514e",
         "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
         "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "context": "#c3c2b7",
         # Ordinal "how late" ramp, least to most; the lightest step still clears 2:1.
         "ramp": ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
         "seq": [[0, "#e6f0fc"], [0.5, "#3987e5"], [1, "#0d366b"]]}
DARK = {"surface": "#1a1a19", "page": "#0d0d0d", "text": "#ffffff", "text2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70", "context": "#52514e",
        # On a dark surface "more" is lighter: the ramp runs dark to light.
        "ramp": ["#184f95", "#2a78d6", "#6da7ec", "#b7d3f6"],
        "seq": [[0, "#1f2a38"], [0.5, "#2a78d6"], [1, "#cde2fb"]]}
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

#: Plain names for the model's inputs.
PLAIN_FEATURES = {
    "prev_delay_1": "How late it was at the last stop",
    "prev_dwell_seconds": "How long it waited at the last platform",
    "station_name": "Which station it is arriving at",
    "scheduled_travel_time": "Scheduled time to the next stop",
    "scheduled_seconds_ahead": "How far ahead the prediction is",
    "prev_travel_time_seconds": "How long the last stretch took",
    "line_late_share_15m": "How late the rest of the line is",
    "prev_delay_2": "How late it was two stops back",
    "leader_delay": "How late the train in front is",
}
BANDS = (("On time (under 1 min late)", -np.inf, 60), ("1-5 min late", 60, 300),
         ("5-10 min late", 300, 600), ("10+ min late", 600, np.inf))
SERVICE_HOURS = list(range(5, 24)) + [0, 1]


# ---------------------------------------------------------------------------
# Data: one tidy table per chart, read from what earlier stages saved
# ---------------------------------------------------------------------------
def _arrivals() -> pd.DataFrame:
    frame = clean.load()
    return frame[clean.is_arrival(frame)]


def delay_bands(arrivals: pd.DataFrame) -> pd.DataFrame:
    """Share of each line's arrivals in each lateness band, most-late line last."""
    edges = [b[1] for b in BANDS] + [np.inf]
    band = pd.cut(arrivals["delay_seconds"], edges, labels=[b[0] for b in BANDS], right=False)
    table = pd.crosstab(arrivals["route_id"].astype(str), band, normalize="index")
    # A band no arrival fell into is still a band: keep it, at 0%.
    table = table.reindex(columns=[b[0] for b in BANDS], fill_value=0.0)
    late = table[BANDS[2][0]] + table[BANDS[3][0]]
    return table.loc[late.sort_values().index]


def late_by_hour(arrivals: pd.DataFrame, min_arrivals: int = 200) -> pd.DataFrame:
    """Line x hour share of arrivals 5+ minutes late; thin cells left blank."""
    late = arrivals["delay_seconds"] >= config.LATE_THRESHOLD_SECONDS
    grouped = late.groupby([arrivals["route_id"].astype(str), arrivals["scheduled_hour"]])
    share = grouped.mean().unstack()
    share = share.where(grouped.size().unstack() >= min_arrivals)
    share = share.reindex(columns=SERVICE_HOURS)
    return share.loc[share.mean(axis=1).sort_values(ascending=False).index]


def example_journeys(arrivals: pd.DataFrame, route: str = "Orange") -> pd.DataFrame:
    """Three real end-to-end trips, typical of trains ending ~1, ~5 and ~10 min late.

    From the last week's weekdays, in one direction, among runs that call at every
    station. For each target, the trip with the median amount of stop-to-stop change
    is chosen, so the picture is typical rather than tidy.
    """
    rows = arrivals[(arrivals["route_id"].astype(str) == route) & (arrivals["direction_id"] == 0)]
    rows = rows[rows["service_date"] >= sorted(rows["service_date"].unique())[-7]]
    rows = rows[~rows["is_weekend"].astype(bool)]
    full = rows.groupby(clean.RUN_KEY)["stop_id"].transform("size")
    rows = rows[full == full.max()].sort_values([*clean.RUN_KEY, "stop_sequence"])
    runs = rows.groupby(clean.RUN_KEY, sort=False)
    summary = pd.DataFrame({
        "final": runs["delay_seconds"].last(),
        "wiggle": runs["delay_seconds"].apply(lambda s: s.diff().abs().mean()),
    })
    picks = []
    for target in (60, 300, 600):
        near = summary[(summary["final"] - target).abs() <= 90]
        if near.empty:
            continue
        pick = (near["wiggle"] - near["wiggle"].median()).abs().idxmin()
        picks.append(pick)
    out = []
    for pick in picks:
        trip = runs.get_group(pick)
        start = pd.to_datetime(trip["arrival_local"].iloc[0])
        out.append(pd.DataFrame({
            "train": f"{start:%a %d %b}, {start:%I:%M %p}".replace(", 0", ", "),
            "station": trip["station_name"].astype(str).to_numpy(),
            "delay_minutes": trip["delay_seconds"].to_numpy() / 60,
        }))
    frame = pd.concat(out, ignore_index=True)
    frame.attrs["route"] = route
    return frame


def prediction_error() -> pd.DataFrame:
    """Average error of the model next to three rules of thumb (spring test period)."""
    table = pd.read_csv(SPRING / "model_comparison.csv").set_index("model")["mae_seconds"]
    return pd.DataFrame({"method": [
        "Assume every train is on time",
        "Use the usual delay for that line and hour",
        "Assume it stays as late as it is now",
        "Our model",
    ], "error_seconds": [table["baseline_zero"], table["baseline_route_hour_mean"],
                         table["baseline_persistence"], table["hist_gradient_boosting_change"]]})


def winter_days() -> pd.DataFrame | None:
    """Per-day error in the winter test period, with that day's snowfall."""
    path = WINTER / "predictions.parquet"
    if not path.exists():
        return None
    p = pd.read_parquet(path)
    p["model_error"] = (p["predicted_delay"] - p["delay_seconds"]).abs()
    p["guess_error"] = (p["persistence_delay"] - p["delay_seconds"]).abs()
    days = p.groupby("service_date_parsed")[["model_error", "guess_error"]].mean()
    days.index = pd.to_datetime(days.index)
    if WINTER_WEATHER.exists():
        w = pd.read_parquet(WINTER_WEATHER)
        snow = w.groupby(pd.to_datetime(w["timestamp"]).dt.normalize())["snowfall_cm"].sum()
        days["snow_cm"] = snow.reindex(days.index).fillna(0)
    else:
        days["snow_cm"] = 0.0
    return days


def importance(top: int = 6) -> pd.DataFrame:
    table = pd.read_csv(SPRING / "feature_importance.csv").head(top)
    table["label"] = table["feature"].map(PLAIN_FEATURES).fillna(table["feature"])
    return table[["label", "importance_mae"]]


def early_warning() -> pd.DataFrame | None:
    """Share of sudden big delays flagged in advance, at 50% alarm precision."""
    rows = []
    for season, folder in (("Spring", SPRING), ("Winter", WINTER)):
        path = folder / "tail_metrics.json"
        if not path.exists():
            continue
        for r in json.loads(path.read_text(encoding="utf-8"))["warning"]:
            if r["method"] == "model" and r["subset"] == "onsets":
                rows.append({"season": season, "stops_ahead": r["horizon_stops"],
                             "caught": r["recall_at_50pct_precision"], "cases": r["positives"]})
    return pd.DataFrame(rows) if rows else None


def late_stations(arrivals: pd.DataFrame, top: int = 12, min_arrivals: int = 2_000) -> pd.DataFrame:
    late = (arrivals["delay_seconds"] >= config.LATE_THRESHOLD_SECONDS)
    g = late.groupby(arrivals["station_name"].astype(str)).agg(["mean", "size"])
    g = g[g["size"] >= min_arrivals].sort_values("mean", ascending=False).head(top)
    return g.rename(columns={"mean": "late_share", "size": "arrivals"})


# ---------------------------------------------------------------------------
# Plotly: layout and theming
# ---------------------------------------------------------------------------
def _theme(trace, **by_mode):
    """Attach light/dark values for this trace; the page script swaps them.

    ``by_mode`` maps a restyle key (``"marker.color"``) to a ``(light, dark)`` pair.
    """
    trace.meta = {"theme": {mode: {k: [v[i]] for k, v in by_mode.items()}
                            for i, mode in enumerate(("light", "dark"))}}
    return trace


def _layout(fig: go.Figure, height: int, **extra) -> go.Figure:
    axis = dict(gridcolor=LIGHT["grid"], linecolor=LIGHT["axis"], zeroline=False,
                ticklabelstandoff=8,
                tickfont=dict(color=LIGHT["muted"], size=12), title_font=dict(color=LIGHT["text2"]))
    extra.setdefault("margin", dict(l=8, r=16, t=40 if fig.layout.showlegend is not False
                                    and len(fig.data) > 1 else 8, b=8))
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=LIGHT["text2"]),
        hoverlabel=dict(font=dict(family=FONT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None,
                    traceorder="normal",
                    font=dict(color=LIGHT["text2"])),
        **extra)
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def _labels_above(fig: go.Figure, labels, values) -> go.Figure:
    """Category labels on their own line above each bar, and room for end values.

    Tick labels beside a bar squeeze the plot to nothing on a phone; above the bar
    they need no width at all.
    """
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_xaxes(range=[0, max(values) * 1.15])
    for label in labels:
        fig.add_annotation(x=0, xref="paper", y=label, yref="y", text=html.escape(str(label)),
                           showarrow=False, xanchor="left", yanchor="bottom", yshift=8,
                           font=dict(color=LIGHT["text2"], size=12))
    return fig


def fig_delay_bands(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, band in enumerate(table.columns):
        fig.add_trace(_theme(go.Bar(
            y=table.index, x=table[band] * 100, name=band, orientation="h",
            marker=dict(color=LIGHT["ramp"][i], line=dict(color=LIGHT["surface"], width=2)),
            hovertemplate="%{y}: %{x:.0f}% " + band.lower() + "<extra></extra>"),
            **{"marker.color": (LIGHT["ramp"][i], DARK["ramp"][i]),
               "marker.line.color": (LIGHT["surface"], DARK["surface"])}))
    fig = _layout(fig, 90 + 38 * len(table), barmode="stack", bargap=0.35)
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title="Share of arrivals")
    return fig


def fig_late_by_hour(share: pd.DataFrame) -> go.Figure:
    labels = [f"{h % 12 or 12}{'am' if h % 24 < 12 else 'pm'}" for h in share.columns]
    fig = go.Figure(_theme(go.Heatmap(
        z=share.to_numpy() * 100, x=labels, y=share.index, colorscale=LIGHT["seq"],
        zmin=0, xgap=2, ygap=2, hoverongaps=False,
        colorbar=dict(ticksuffix="%", thickness=10, outlinewidth=0, tickfont=dict(color=LIGHT["muted"])),
        hovertemplate="%{y}, %{x}: %{z:.0f}% of arrivals 5+ min late<extra></extra>"),
        **{"colorscale": (LIGHT["seq"], DARK["seq"]),
           "colorbar.tickfont.color": (LIGHT["muted"], DARK["muted"])}))
    fig = _layout(fig, 60 + 34 * len(share))
    fig.update_xaxes(showgrid=False, title="Scheduled hour")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def _minute_range(frame: pd.DataFrame) -> list[float]:
    """At least 0-12 minutes, and below zero only if a train ran early."""
    return [min(0.0, frame["delay_minutes"].min() - 0.5), max(12.0, frame["delay_minutes"].max() + 1)]


def fig_journeys(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    order = list(dict.fromkeys(frame["station"]))
    keys = ("blue", "orange", "aqua")
    for i, (train, trip) in enumerate(frame.groupby("train", sort=False)):
        k = keys[i % 3]
        fig.add_trace(_theme(go.Scatter(
            x=trip["station"], y=trip["delay_minutes"], name=train, mode="lines+markers",
            line=dict(color=LIGHT[k], width=2), marker=dict(size=8, color=LIGHT[k],
                                                            line=dict(color=LIGHT["surface"], width=2)),
            hovertemplate="%{x}: %{y:.1f} min late<extra>" + train + "</extra>"),
            **{"line.color": (LIGHT[k], DARK[k]), "marker.color": (LIGHT[k], DARK[k]),
               "marker.line.color": (LIGHT["surface"], DARK["surface"])}))
    fig = _layout(fig, 400, margin=dict(l=8, r=16, t=40, b=8))
    fig.update_xaxes(categoryorder="array", categoryarray=order, tickangle=-45, showgrid=False)
    # Headroom matters: on a 0-2 minute axis, 30-second wiggles look dramatic.
    fig.update_yaxes(title="Minutes late", range=_minute_range(frame))
    return fig


def fig_prediction_error(table: pd.DataFrame) -> go.Figure:
    ours = table["method"] == "Our model"
    light = [LIGHT["blue"] if o else LIGHT["context"] for o in ours]
    dark = [DARK["blue"] if o else DARK["context"] for o in ours]
    fig = go.Figure(_theme(go.Bar(
        y=table["method"], x=table["error_seconds"], orientation="h",
        marker=dict(color=light), text=[f"{v:.0f} s" for v in table["error_seconds"]],
        textposition="outside", textfont=dict(color=LIGHT["text"]), cliponaxis=False,
        hovertemplate="%{y}: wrong by %{x:.1f} seconds on average<extra></extra>"),
        **{"marker.color": (light, dark), "textfont.color": (LIGHT["text"], DARK["text"])}))
    fig = _layout(fig, 60 + 56 * len(table), bargap=0.62, showlegend=False)
    fig.update_xaxes(title="Average error (seconds)")
    fig.update_yaxes(autorange="reversed")
    return _labels_above(fig, table["method"], table["error_seconds"])


def fig_winter(days: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.3, 0.7],
                        vertical_spacing=0.08)
    fig.add_trace(_theme(go.Bar(
        x=days.index, y=days["snow_cm"], name="Snowfall", showlegend=False,
        marker=dict(color=LIGHT["context"]),
        hovertemplate="%{x|%a %d %b}: %{y:.1f} cm of snow<extra></extra>"),
        **{"marker.color": (LIGHT["context"], DARK["context"])}), row=1, col=1)
    for column, name, k in (("guess_error", "Assume it stays as late as it is now", "orange"),
                            ("model_error", "Our model", "blue")):
        fig.add_trace(_theme(go.Scatter(
            x=days.index, y=days[column], name=name, mode="lines+markers",
            line=dict(color=LIGHT[k], width=2),
            marker=dict(size=8, color=LIGHT[k], line=dict(color=LIGHT["surface"], width=2)),
            hovertemplate="%{x|%a %d %b}: wrong by %{y:.0f} s on average<extra>" + name + "</extra>"),
            **{"line.color": (LIGHT[k], DARK[k]), "marker.color": (LIGHT[k], DARK[k]),
               "marker.line.color": (LIGHT["surface"], DARK["surface"])}), row=2, col=1)
    fig = _layout(fig, 420, hovermode="x unified")
    fig.update_yaxes(title="Snow (cm)", row=1, col=1)
    fig.update_yaxes(title="Average error (s)", rangemode="tozero", row=2, col=1)
    return fig


def fig_importance(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure(_theme(go.Bar(
        y=table["label"], x=table["importance_mae"], orientation="h",
        marker=dict(color=LIGHT["blue"]), text=[f"+{v:.0f} s" for v in table["importance_mae"]],
        textposition="outside", textfont=dict(color=LIGHT["text"]), cliponaxis=False,
        hovertemplate="Hide \"%{y}\" and predictions get %{x:.0f} s worse<extra></extra>"),
        **{"marker.color": (LIGHT["blue"], DARK["blue"]),
           "textfont.color": (LIGHT["text"], DARK["text"])}))
    fig = _layout(fig, 60 + 56 * len(table), bargap=0.62, showlegend=False)
    fig.update_xaxes(title="Extra error (seconds)")
    fig.update_yaxes(autorange="reversed")
    return _labels_above(fig, table["label"], table["importance_mae"])


def fig_early_warning(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for season, k in (("Spring", "blue"), ("Winter", "orange")):
        rows = table[table["season"] == season]
        if rows.empty:
            continue
        x = [f"{s} stop{'s' if s > 1 else ''} before" for s in rows["stops_ahead"]]
        fig.add_trace(_theme(go.Bar(
            x=x, y=rows["caught"] * 100, name=season, marker=dict(color=LIGHT[k]),
            text=[f"{v:.0%}" for v in rows["caught"]], textposition="outside",
            textfont=dict(color=LIGHT["text"]), cliponaxis=False,
            hovertemplate="%{x}, " + season.lower() + ": %{y:.0f}% flagged<extra></extra>"),
            **{"marker.color": (LIGHT[k], DARK[k]), "textfont.color": (LIGHT["text"], DARK["text"])}))
    fig = _layout(fig, 300, barmode="group", bargap=0.72, bargroupgap=0.12)
    fig.update_yaxes(ticksuffix="%", title="Sudden big delays flagged", rangemode="tozero")
    return fig


def fig_stations(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure(_theme(go.Bar(
        y=table.index, x=table["late_share"] * 100, orientation="h",
        marker=dict(color=LIGHT["blue"]),
        text=[f"{v:.0%}" for v in table["late_share"]], textposition="outside",
        textfont=dict(color=LIGHT["text"]), cliponaxis=False,
        hovertemplate="%{y}: %{x:.0f}% of arrivals 5+ min late<extra></extra>"),
        **{"marker.color": (LIGHT["blue"], DARK["blue"]),
           "textfont.color": (LIGHT["text"], DARK["text"])}))
    fig = _layout(fig, 60 + 50 * len(table), bargap=0.6, showlegend=False)
    fig.update_xaxes(ticksuffix="%", title="Arrivals 5+ min late")
    fig.update_yaxes(autorange="reversed")
    return _labels_above(fig, table.index, table["late_share"] * 100)


# ---------------------------------------------------------------------------
# Static PNGs for the README (GitHub runs no scripts)
# ---------------------------------------------------------------------------
def _mpl_axes(width=8.0, height=4.0):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(LIGHT["surface"])
    ax.set_facecolor(LIGHT["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LIGHT["axis"])
    ax.tick_params(colors=LIGHT["muted"], length=0)
    ax.grid(axis="x", color=LIGHT["grid"], linewidth=1)
    ax.set_axisbelow(True)
    plt.rcParams["font.family"] = "sans-serif"
    return fig, ax


def _save(fig, name: str) -> str:
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path.name


def png_delay_bands(table: pd.DataFrame) -> str:
    fig, ax = _mpl_axes(8, 0.5 + 0.42 * len(table))
    left = np.zeros(len(table))
    for i, band in enumerate(table.columns):
        values = table[band].to_numpy() * 100
        ax.barh(table.index, values, left=left, height=0.55, color=LIGHT["ramp"][i],
                edgecolor=LIGHT["surface"], linewidth=2, label=band)
        left += values
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.tick_params(axis="y", colors=LIGHT["text2"])
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9,
              labelcolor=LIGHT["text2"], handlelength=1, handleheight=1)
    ax.set_title("How late do trains run? Share of arrivals, by line", loc="left",
                 color=LIGHT["text"], pad=28, fontsize=12)
    return _save(fig, "story_delay_bands")


def png_journeys(frame: pd.DataFrame) -> str:
    fig, ax = _mpl_axes(9, 4)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=LIGHT["grid"], linewidth=1)
    order = list(dict.fromkeys(frame["station"]))
    pos = {s: i for i, s in enumerate(order)}
    for (train, trip), k in zip(frame.groupby("train", sort=False), ("blue", "orange", "aqua")):
        x = trip["station"].map(pos)
        ax.plot(x, trip["delay_minutes"], color=LIGHT[k], linewidth=2, marker="o", markersize=5,
                markeredgecolor=LIGHT["surface"], markeredgewidth=1.5, label=train)
    ax.set_xticks(range(len(order)), order, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(*_minute_range(frame))
    ax.set_ylabel("Minutes late", color=LIGHT["text2"])
    ax.legend(frameon=False, fontsize=9, labelcolor=LIGHT["text2"], loc="upper left")
    ax.set_title(f"Delays stick: real {frame.attrs['route']} Line trains, stop by stop",
                 loc="left", color=LIGHT["text"], fontsize=12)
    return _save(fig, "story_journeys")


def png_prediction_error(table: pd.DataFrame) -> str:
    fig, ax = _mpl_axes(8, 2.4)
    colors = [LIGHT["blue"] if m == "Our model" else LIGHT["context"] for m in table["method"]]
    ax.barh(table["method"], table["error_seconds"], height=0.5, color=colors)
    for y, v in enumerate(table["error_seconds"]):
        ax.text(v + 4, y, f"{v:.0f} s", va="center", fontsize=10, color=LIGHT["text"])
    ax.invert_yaxis()
    ax.tick_params(axis="y", colors=LIGHT["text2"])
    ax.set_xlabel("Average error in seconds (shorter is better)", color=LIGHT["text2"])
    ax.set_title("How far off is each way of guessing a train's delay?", loc="left",
                 color=LIGHT["text"], fontsize=12)
    return _save(fig, "story_prediction_error")


def png_winter(days: pd.DataFrame) -> str:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 2.4], "hspace": 0.12})
    fig.patch.set_facecolor(LIGHT["surface"])
    for ax in (top, bottom):
        ax.set_facecolor(LIGHT["surface"])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(LIGHT["axis"])
        ax.tick_params(colors=LIGHT["muted"], length=0)
        ax.grid(axis="y", color=LIGHT["grid"], linewidth=1)
        ax.set_axisbelow(True)
    top.bar(days.index, days["snow_cm"], color=LIGHT["context"], width=0.7)
    top.set_ylabel("Snow (cm)", color=LIGHT["text2"])
    for column, name, k in (("guess_error", "Assume it stays as late as it is now", "orange"),
                            ("model_error", "Our model", "blue")):
        bottom.plot(days.index, days[column], color=LIGHT[k], linewidth=2, marker="o",
                    markersize=5, markeredgecolor=LIGHT["surface"], markeredgewidth=1.5, label=name)
    bottom.set_ylim(bottom=0)
    bottom.set_ylabel("Average error (s)", color=LIGHT["text2"])
    bottom.legend(frameon=False, fontsize=9, labelcolor=LIGHT["text2"], loc="upper left")
    bottom.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    top.set_title("Winter, day by day: snowfall (top) and prediction error (bottom)", loc="left",
                  color=LIGHT["text"], fontsize=12)
    return _save(fig, "story_winter")


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
PAGE_CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--text:#0b0b0b;--text2:#52514e;
--muted:#898781;--line:rgba(11,11,11,.10);--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--page:#0d0d0d;--surface:#1a1a19;--text:#fff;--text2:#c3c2b7;--line:rgba(255,255,255,.10);--accent:#3987e5}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--text:#fff;
--text2:#c3c2b7;--line:rgba(255,255,255,.10);--accent:#3987e5}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text);font:16px/1.6 %(font)s}
main{max-width:860px;margin:0 auto;padding:40px 16px 80px}
h1{font-size:2rem;line-height:1.2;margin:0 0 12px;letter-spacing:-.01em}
.lede{color:var(--text2);font-size:1.1rem;margin:0 0 28px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:0 0 40px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px}
.tile .v{font-size:clamp(1.3rem,3.2vw,1.7rem);white-space:nowrap;font-weight:600;line-height:1.1}
.tile .l{color:var(--text2);font-size:.9rem;margin-top:6px}
section{background:var(--surface);border:1px solid var(--line);border-radius:14px;
padding:22px 20px 14px;margin:0 0 20px}
.q{color:var(--muted);font-size:.8rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;margin:0}
h2{font-size:1.3rem;line-height:1.3;margin:4px 0 6px}
.why{color:var(--text2);margin:0 0 12px}
details{margin:6px 0 4px;color:var(--text2);font-size:.9rem}
summary{cursor:pointer;color:var(--accent)}
table{border-collapse:collapse;margin-top:8px;font-variant-numeric:tabular-nums;width:100%%}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}
.table-wrap{overflow-x:auto}
footer{color:var(--muted);font-size:.85rem;margin-top:32px}
footer a{color:var(--accent)}
""" % {"font": FONT}

THEME_JS = """
const PAL = %(pal)s;
function isDark(){const t=document.documentElement.dataset.theme;
  return t ? t==="dark" : matchMedia("(prefers-color-scheme: dark)").matches;}
function paint(){const dark=isDark(), p=dark?PAL.dark:PAL.light, mode=dark?"dark":"light";
  document.querySelectorAll(".js-plotly-plot").forEach(gd=>{
    const lay={"font.color":p.text2,"legend.font.color":p.text2};
    Object.keys(gd.layout).filter(k=>/^[xy]axis\\d*$/.test(k)).forEach(k=>{
      lay[k+".gridcolor"]=p.grid; lay[k+".linecolor"]=p.axis;
      lay[k+".tickfont.color"]=p.muted; lay[k+".title.font.color"]=p.text2;});
    (gd.layout.annotations||[]).forEach((a,i)=>{lay["annotations["+i+"].font.color"]=p.text2;});
    Plotly.relayout(gd,lay);
    gd.data.forEach((tr,i)=>{if(tr.meta&&tr.meta.theme){Plotly.restyle(gd,tr.meta.theme[mode],[i]);}});
  });}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",paint);
new MutationObserver(paint).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
window.addEventListener("load",paint);
"""


def _numbers(frame: pd.DataFrame, fmt: dict | None = None) -> str:
    """A "show the numbers" table, so no value is only readable as a colour or a length."""
    body = frame.to_html(border=0, escape=True, formatters=fmt, na_rep="–")
    return f"<details><summary>Show the numbers</summary><div class='table-wrap'>{body}</div></details>"


def _section(question: str, answer: str, why: str, fig: go.Figure, numbers: str) -> str:
    chart = fig.to_html(full_html=False, include_plotlyjs=False,
                        config={"displayModeBar": False, "responsive": True})
    return (f"<section><p class='q'>{html.escape(question)}</p><h2>{html.escape(answer)}</h2>"
            f"<p class='why'>{why}</p>{chart}{numbers}</section>")


def build() -> dict:
    """Write reports/story.html and the README PNGs from the saved artefacts."""
    config.ensure_dirs()
    arrivals = _arrivals()
    bands, hours = delay_bands(arrivals), late_by_hour(arrivals)
    journeys, errors = example_journeys(arrivals), prediction_error()
    winter, top, warning = winter_days(), importance(), early_warning()
    stations = late_stations(arrivals)

    on_time = float((arrivals["delay_seconds"] < config.LATE_THRESHOLD_SECONDS).mean())
    ours = errors.set_index("method")["error_seconds"]
    model, guess = ours["Our model"], ours["Assume it stays as late as it is now"]
    winter_mae = None
    if (WINTER / "model_comparison.csv").exists():
        w = pd.read_csv(WINTER / "model_comparison.csv").set_index("model")["mae_seconds"]
        winter_mae = w.get("hist_gradient_boosting_change")
    start, end = pd.to_datetime(arrivals["service_date_parsed"]).agg(["min", "max"])

    steps = journeys.assign(step=journeys.groupby("train", sort=False)["delay_minutes"].diff())
    jump = steps.loc[steps["step"].abs().idxmax()]
    tiles = [
        (f"{len(arrivals) / 1e6:.1f}M", f"train arrivals studied, {start:%d %b} – {end:%d %b %Y}"),
        (f"{on_time:.0%}", "of arrivals are less than 5 minutes late"),
        (f"{model:.0f} seconds", "average error when predicting a train's delay at its next stop"),
        (f"{guess / model:.0f}× better", f"than assuming a train stays as late as it is ({guess:.0f} s)"),
    ]
    if winter_mae is not None:
        tiles.append((f"{winter_mae:.0f} seconds", "average error in winter, snowstorms included"))
    tile_html = "".join(f"<div class='tile'><div class='v'>{html.escape(v)}</div>"
                        f"<div class='l'>{html.escape(l)}</div></div>" for v, l in tiles)

    worst_line, best_line = bands.index[-1], bands.index[0]
    peak = hours.stack().idxmax()
    peak_label = f"{peak[1] % 12 or 12}{'am' if peak[1] < 12 else 'pm'}"
    worst_two = " and ".join(hours.index[:2])
    pct = {c: (lambda v: f"{v:.0%}") for c in bands.columns}

    sections = [
        _section("How late do trains run?",
                 f"Most arrivals are close to on time; the {worst_line} line is late most often",
                 f"Each bar is one line's arrivals, split by how late they were against the "
                 f"timetable. The stronger the blue, the later. The {best_line} line keeps "
                 f"closest to schedule.",
                 fig_delay_bands(bands), _numbers(bands, pct)),
        _section("When is it worst?",
                 f"{worst_two} run late at every hour; the single worst hour is {peak_label} "
                 f"on the {peak[0]} line",
                 "Each square is one line in one hour of the day, shaded by the share of trains "
                 "arriving 5 or more minutes late. The stronger the blue, the more late trains; blank squares "
                 "had too few trains to say.",
                 fig_late_by_hour(hours),
                 _numbers((hours * 100).round(0).rename(columns=lambda h: f"{h}:00"))),
        _section("Why can delays be predicted at all?",
                 "A late train stays late: its delay barely changes from one stop to the next",
                 f"Real {journeys.attrs['route']} Line trains from the last week of June, end to "
                 "end: one that ran nearly on time, one about 5 minutes late and one about 10. "
                 "The lines are mostly flat. The one big step, "
                 f"{abs(jump['step']):.0f} minutes at {html.escape(jump['station'])}, is the "
                 "kind of sudden change nothing upstream warns about. "
                 "That is the pattern the model learns: start from how late the train is now, "
                 "then predict the small change to the next stop.",
                 fig_journeys(journeys),
                 _numbers(journeys.pivot_table(index="station", columns="train",
                                               values="delay_minutes", sort=False).round(1))),
        _section("How good are the predictions?",
                 f"Our model is off by {model:.0f} seconds on average, {guess / model:.0f}× better "
                 "than the best rule of thumb",
                 "Each bar is a way of guessing how late a train will be at its next stop, scored "
                 "on three weeks of trains the model never saw. The two simple averages are "
                 "wrong by over four minutes; \"it stays as late as it is\" is a good rule, and "
                 "the model cuts its error by two thirds.",
                 fig_prediction_error(errors),
                 _numbers(errors.set_index("method").round(1))),
    ]
    if winter is not None:
        sections.append(_section(
            "Does it hold up in a snowstorm?",
            "Yes: on the worst storm day the model stayed far closer than the simple rule",
            "Every day of the last three weeks of February 2026, including the 23 February "
            "storm. Top: snowfall. Bottom: how far off each method was that day. An earlier "
            "version of the model broke down on storm days, when trains ran hours behind the "
            "timetable; predicting the change from stop to stop, not the delay itself, fixed it.",
            fig_winter(winter),
            _numbers(winter.round(1).rename(columns={
                "model_error": "Our model (s)", "guess_error": "Simple rule (s)",
                "snow_cm": "Snow (cm)"}).rename(index=lambda d: f"{d:%a %d %b}"))))
    sections.append(_section(
        "What does the model pay attention to?",
        "Overwhelmingly, how late the train already is",
        "Each bar shows how much worse the predictions get if the model is not allowed to see "
        "that piece of information. Weather, service alerts and ridership barely register; "
        "the train's own recent delay dominates everything.",
        fig_importance(top),
        _numbers(top.set_index("label").round(1).rename(columns={"importance_mae": "Extra error (s)"}))))
    if warning is not None:
        one = warning[(warning["stops_ahead"] == 1) & (warning["season"] == "Spring")]["caught"]
        sections.append(_section(
            "Can it warn about sudden big delays?",
            f"Sometimes: it flags about {one.iloc[0]:.0%} of them one stop in advance" if len(one)
            else "Sometimes",
            "The hard case is a train running on time that suddenly ends up 10+ minutes late "
            "(a breakdown, a medical emergency). Bars show the share of those the model flags "
            "in advance while keeping at least half of its warnings correct. Further ahead, "
            "it catches far fewer: most sudden disruptions give no sign in this data.",
            fig_early_warning(warning),
            _numbers(warning.set_index(["season", "stops_ahead"]).round(3))))
    sections.append(_section(
        "Which stations see the most late trains?",
        f"{stations.index[0]} tops the list",
        f"Share of arrivals 5 or more minutes late at each station (stations with at least "
        f"2,000 arrivals). A station's figure reflects every line that calls there.",
        fig_stations(stations),
        _numbers(stations.assign(late_share=(stations["late_share"] * 100).round(1))
                 .rename(columns={"late_share": "Late (%)", "arrivals": "Arrivals"}))))

    pal = json.dumps({mode: {k: p[k] for k in ("text2", "muted", "grid", "axis")}
                      for mode, p in (("light", LIGHT), ("dark", DARK))})
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Will My Train Be Late?</title>
<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>
<style>{PAGE_CSS}</style></head>
<body><main>
<h1>Will my train be late?</h1>
<p class="lede">What {len(arrivals) / 1e6:.1f} million MBTA subway arrivals say about when the T
runs late, and how well it can be predicted. No maths required: every chart answers one
question, and you can hover over it for exact values.</p>
<div class="tiles">{tile_html}</div>
{''.join(sections)}
<footer>Data: MBTA LAMP performance archive, MBTA gated-station entries, Open-Meteo
weather. Built by <code>python -m mbta_ds.cli story</code>. The full technical results,
with every table and method, are in <a href="report.html">report.html</a>.</footer>
</main><script>{THEME_JS % {"pal": pal}}</script></body></html>"""
    OUT_PATH.write_text(page, encoding="utf-8")

    pngs = [png_delay_bands(bands), png_journeys(journeys), png_prediction_error(errors)]
    if winter is not None:
        pngs.append(png_winter(winter))
    log.info("story -> %s (%d charts, %d PNGs)", OUT_PATH, len(sections), len(pngs))
    return {"path": str(OUT_PATH), "sections": len(sections), "png": pngs}
