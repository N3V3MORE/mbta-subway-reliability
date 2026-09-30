"""Report stage: one self-contained HTML page plus plain CSV tables.

Everything is rebuilt from the artefacts the earlier stages persisted, so the
report regenerates identically on any machine with the same data. The page has
no external dependencies: styles are inline and the static figures are embedded,
so it opens offline in any browser. (The interactive Plotly figures it links to
load their library from a CDN.)
"""

from __future__ import annotations

import base64
import html
import json
import logging

import numpy as np
import pandas as pd

from . import cluster_stations, config, model_delay, model_tail

log = logging.getLogger(__name__)

OUT_PATH = config.REPORTS_DIR / "report.html"

#: Internal model key -> the name a reader sees in the report and its tables.
MODEL_LABELS = {
    "hist_gradient_boosting_change": "Gradient boosting on the change since the last stop",
    "hist_gradient_boosting_run_time": "Gradient boosting correcting the running-time lookup (candidate)",
    "hist_gradient_boosting_mae": "Gradient boosting (absolute-error loss)",
    "hist_gradient_boosting": "Gradient boosting (squared-error loss)",
    "decision_tree": "Decision tree",
    "ridge": "Ridge regression",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "knn": "k-nearest neighbours",
    "baseline_persistence": "Baseline: previous stop's delay",
    "baseline_route_hour_mean": "Baseline: route x hour average",
    "baseline_run_time": "Baseline: departure + usual running time",
    "baseline_zero": "Baseline: always on time",
    "baseline_route_hour_rate": "Baseline: route x hour late rate",
    "baseline_always_ontime": "Baseline: never late",
}

CSS = """
body{font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;max-width:980px;margin:0 auto;
padding:24px 16px;color:#1d2330;background:#fbfbfd}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 8px;border-bottom:1px solid #dde1ea;
padding-bottom:4px}p.lead{color:#4a5263;margin-top:0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}
.card{background:#fff;border:1px solid #dde1ea;border-radius:8px;padding:12px 14px}
.card b{display:block;font-size:24px}.card span{color:#4a5263;font-size:13px}
table{border-collapse:collapse;width:100%;margin:8px 0 4px;font-size:13.5px;background:#fff}
th,td{border:1px solid #dde1ea;padding:5px 8px;text-align:right}th{background:#f0f2f7}
td:first-child,th:first-child{text-align:left}.note{color:#4a5263;font-size:13px}
img{max-width:100%;border:1px solid #dde1ea;border-radius:6px;background:#fff}
.fail{color:#b3261e;font-weight:600}.ok{color:#1d7a3a}
"""


#: Columns whose values live in 0..1 and need three decimals to be told apart.
THREE_DECIMALS = {"R²", "F1", "Precision", "Recall", "ROC-AUC", "PR-AUC", "Accuracy",
                  "Coverage", "Coverage before calibration", "Share not >5 min late", "value"}


def _cell(value, decimals: int) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.{decimals}f}"
    return html.escape(str(value))


def _table(frame: pd.DataFrame, digits: int = 1) -> str:
    """Render a frame as an HTML table: escaped text, rounded numbers, blank gaps."""
    places = [3 if c in THREE_DECIMALS else digits for c in frame.columns]
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    body = "".join("<tr>" + "".join(f"<td>{_cell(v, p)}</td>" for v, p in zip(row, places)) + "</tr>"
                   for row in frame.itertuples(index=False))
    return f"<table><tr>{head}</tr>{body}</table>"


def _date(value) -> str:
    """20260607 -> 2026-06-07."""
    return pd.to_datetime(str(value), format="%Y%m%d").strftime("%Y-%m-%d")


def _image(name: str) -> str:
    path = config.FIGURES_DIR / f"{name}.png"
    if not path.exists():
        return f'<p class="note">({name}.png not generated)</p>'
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img alt="{name}" src="data:image/png;base64,{data}">'


def _label_models(frame: pd.DataFrame, overrides: dict | None = None) -> pd.DataFrame:
    labels = {**MODEL_LABELS, **(overrides or {})}
    return frame.assign(model=frame["model"].map(labels).fillna(frame["model"]))


def tables() -> dict[str, pd.DataFrame]:
    """Every result table, with readable column names, ready to show or export."""
    metrics = model_delay.load_metrics()
    reg = _label_models(pd.DataFrame(metrics["regression"]))
    # The classifier's boosted model uses log-loss, not either regression loss.
    clf = _label_models(pd.DataFrame(metrics["classification"]),
                        {"hist_gradient_boosting": "Gradient boosting"})
    clusters = cluster_stations.load_clusters()
    backtest = pd.DataFrame(metrics["backtest"])
    backtest[["test_start", "test_end"]] = backtest[["test_start", "test_end"]].map(_date)
    out = {
        "regression": reg[["model", "mae_seconds", "rmse_seconds", "r2", "bias_seconds"]].rename(columns={
            "model": "Model", "mae_seconds": "MAE (s)", "rmse_seconds": "RMSE (s)", "r2": "R²",
            "bias_seconds": "Bias (s)"}),
        "classification": clf[["model", "f1", "precision", "recall", "roc_auc", "pr_auc", "accuracy"]].rename(
            columns={"model": "Model", "f1": "F1", "precision": "Precision", "recall": "Recall",
                     "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "accuracy": "Accuracy"}),
        "backtest": backtest.rename(columns={
            "test_start": "Test from", "test_end": "Test to", "train_days": "Training days",
            "n_test": "Arrivals scored", "persistence_mae": "Baseline MAE (s)",
            "run_time_mae": "Run-time baseline MAE (s)",
            "model_mae": "Model MAE (s)", "improvement_pct": "Improvement (%)"}),
        "horizons": pd.DataFrame(metrics["horizons"]).rename(columns={
            "horizon_stops": "Stops ahead", "n_test": "Arrivals scored",
            "persistence_mae": "Baseline MAE (s)", "run_time_mae": "Run-time baseline MAE (s)",
            "own_train_mae": "Own train only MAE (s)",
            "with_other_trains_mae": "+ other trains MAE (s)"}),
        "ablation": pd.DataFrame(metrics["ablation"])[
            ["features", "n_features", "mae_seconds", "mae_seed_spread"]].rename(columns={
                "features": "Feature set", "n_features": "Features", "mae_seconds": "MAE (s)",
                "mae_seed_spread": "Seed spread (s)"}),
        "error_by_route": pd.DataFrame(metrics["error_analysis"]["by_route"]).rename(columns={
            "route_id": "Route", "mae": "MAE (s)", "median_abs_error": "Median error (s)", "n": "Arrivals"}),
        "error_by_day_type": pd.DataFrame(metrics["error_analysis"]["by_day_type"]).rename(columns={
            "day_type": "Route-day", "mae": "MAE (s)", "median_abs_error": "Median error (s)",
            "n": "Arrivals"}),
        "stations": clusters[[c for c in ("station_name", "reliability_cluster", "mean_lateness",
                                          "median_lateness", "on_time_rate", "demand_cluster",
                                          "total_entries") if c in clusters]].rename(columns={
            "station_name": "Station", "reliability_cluster": "Reliability cluster",
            "mean_lateness": "Mean lateness (s)", "median_lateness": "Median lateness (s)",
            "on_time_rate": "Share not >5 min late", "demand_cluster": "Demand cluster",
            "total_entries": "Mean daily entries"}),
        "cleaning": pd.read_csv(config.PROCESSED_DIR / "clean_ledger.csv").rename(
            columns={"step": "Cleaning step", "rows_removed": "Rows removed"}),
        "validation": pd.read_csv(config.PROCESSED_DIR / "validation.csv"),
    }
    try:
        tail = model_tail.load_metrics()
    except FileNotFoundError:
        return out
    out["early_warning"] = pd.DataFrame(tail["warning"]).rename(columns={
        "horizon_stops": "Stops ahead", "method": "Method", "subset": "Trains",
        "n": "Arrivals", "positives": "10+ min late", "pr_auc": "PR-AUC",
        "recall_at_50pct_precision": "Recall"})
    out["ranges"] = pd.DataFrame(tail["ranges"]).rename(columns={
        "horizon_stops": "Stops ahead", "day_type": "Route-days", "coverage": "Coverage",
        "coverage_uncorrected": "Coverage before calibration",
        "conformal_margin_seconds": "Calibration widening (s)",
        "median_width_seconds": "Median band width (s)",
        "median_abs_error_seconds": "Median error (s)", "pinball_q90": "Pinball loss, 90th pct"})
    return out


#: What "late" means everywhere in the report (see ``clean.lateness``).
LATE_DEFINITION = (
    "<p>\"Late\" is judged the way riders feel it: how much longer than planned they "
    "waited for the train, i.e. the gap behind the train in front minus the scheduled gap "
    "(the timetable where no gap is recorded). Delay against the timetable drifts on "
    "frequent lines, where trains are paired with timetable slots in order.</p>"
)

def build() -> dict:
    """Write ``reports/report.html`` and ``reports/tables/*.csv``."""
    config.ensure_dirs()
    t = tables()
    for name, frame in t.items():
        frame.to_csv(config.TABLES_DIR / f"{name}.csv", index=False)

    metrics = model_delay.load_metrics()
    window = json.loads(config.WINDOW_PATH.read_text(encoding="utf-8"))
    best = metrics["best_regression_metrics"]
    persistence = next(r for r in metrics["regression"] if r["model"] == "baseline_persistence")
    clf = metrics["best_classification_metrics"]
    horizon = pd.DataFrame(metrics["horizons"])
    far = horizon.iloc[-1]
    validation = t["validation"]
    failed = validation[validation["hard"] & ~validation["ok"]]

    cards = [
        (f"{best['mae_seconds']:.1f} s", "typical error predicting the next stop's delay (MAE)"),
        (f"{100 * (1 - best['mae_seconds'] / persistence['mae_seconds']):.0f}%",
         "better than \"same as the last stop\""),
        (f"{clf['f1']:.3f}", "F1 for \"more than 5 min late?\""),
        (f"{100 * (1 - far['with_other_trains_mae'] / far['persistence_mae']):.0f}%",
         f"better than the baseline {int(far['horizon_stops'])} stops ahead"),
        (f"{len(validation) - len(failed)}/{len(validation)}", "data-validity checks passed"),
    ]
    links = "".join(f'<li><a href="figures/{p.name}">{p.stem.replace("_", " ")}</a></li>'
                    for p in sorted(config.FIGURES_DIR.glob("*.html")))
    # (title, body) pairs; numbering is added below so sections can be inserted freely.
    sections = [
        ("Predicting the next stop's delay",
         "<p>MAE is the average size of the error in seconds; lower is better. Test set: every "
         f"arrival after {_date(metrics['split']['cutoff_service_date'])} "
         f"({metrics['split']['n_test']:,} arrivals), none of which the models saw in training.</p>"
         + _table(t["regression"])),
        ("Does it hold on other weeks?",
         "<p>The model refit before each two-week window and scored on it. Consistent improvement "
         "across windows means the result is not one lucky test period.</p>" + _table(t["backtest"])),
        ("Predicting further ahead",
         "<p>How well can a train's delay be predicted several stops before it gets there? "
         "\"Other trains\" adds what the train ahead and the rest of the line were doing at the "
         "time of the prediction.</p>" + _table(t["horizons"])),
    ]
    if "early_warning" in t:
        sections += [
            ("Will it be 10+ minutes late?",
             LATE_DEFINITION +
             "<p>PR-AUC measures how well each method ranks the trains that will be 10+ minutes "
             "late above the rest (1 = perfect; a random guess scores the share that are late). "
             "\"Recall\" is the share of those trains caught while keeping at least half of the "
             "alarms correct.</p><p><b>Onsets</b> are the hard cases: trains under 5 minutes late "
             "at the time of the prediction that end up 10+ minutes late. Almost every other late "
             "train is already late, which any method can spot.</p>"
             + _table(t["early_warning"], 3)
             + "<p>The predicted probabilities can be taken at face value: when the model says "
             "30%, about that share of trains really are 10+ minutes late.</p>"
             + _image("calibration")),
            ("Prediction ranges",
             "<p>Instead of a single number, a range the delay should fall inside 80% of the time "
             "(10th to 90th percentile). \"Coverage\" is how often it actually did.</p>"
             + _table(t["ranges"])),
        ]
    sections += [
        ("Which information helps",
         "<p>Error as groups of features are added. Differences smaller than the seed spread are "
         "noise.</p>" + _table(t["ablation"]) + _image("ablation")),
        ("Where the errors are",
         _table(t["error_by_route"]) + _table(t["error_by_day_type"])
         + f"<p class='note'>A route-day is \"disrupted\" when at least "
         f"{model_delay.DISRUPTED_SHARE:.0%} of its arrivals kept riders waiting 10+ minutes "
         "longer than planned (a typical route-day has under 3%). Those few days carry a large "
         "share of the error.</p>" + _image("predicted_vs_actual")),
        ("Will the train be more than 5 minutes late?",
         LATE_DEFINITION + _table(t["classification"], 3)),
        ("Stations", _image("delay_heatmap") + _table(t["stations"])),
        ("Data quality",
         "<p>How many rows each cleaning rule removed, and every automated check with its "
         "measured value.</p>" + _table(t["cleaning"], 0) + _table(
             validation.assign(result=validation.apply(
                 lambda r: "pass" if r["ok"] else ("FAIL" if r["hard"] else "info"), axis=1))
             [["source", "check", "value", "op", "limit", "result"]], 4)),
        ("Interactive figures", f"<ul>{links}</ul>"),
    ]
    head = (f"<h1>MBTA subway delays: results</h1><p class='lead'>Service dates {window['start']} "
            f"to {window['end']}. Regenerate with <code>make all</code> (or <code>.\\make.ps1 "
            f"all</code>). Every table below is also saved as CSV in <code>reports/tables/</code>."
            "</p><div class='cards'>" + "".join(f"<div class='card'><b>{v}</b><span>{k}</span></div>"
                                              for v, k in cards) + "</div>")
    body = head + "".join(f"<h2>{i}. {html.escape(title)}</h2>{content}"
                          for i, (title, content) in enumerate(sections, 1))
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>MBTA delay results</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    OUT_PATH.write_text(page, encoding="utf-8")
    log.info("report -> %s (%d tables in %s)", OUT_PATH, len(t), config.TABLES_DIR)
    return {"report": str(OUT_PATH), "tables": sorted(t)}
