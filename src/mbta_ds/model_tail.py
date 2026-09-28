"""Tail models: will the train be 10+ minutes late, and what range to expect?

The delay model predicts the *typical* outcome, which is exactly what fails on bad
days. This stage asks the two questions a rider has when things go wrong, k stops
before the train reaches their station:

* **Early warning** -- the probability the train is more than 10 minutes late on
  arrival. About 9% of arrivals are, but nearly all of those trains are *already*
  that late (delay is sticky), so any method catches them. The honest test is
  **onsets**: trains under 5 minutes late now that end up 10+ minutes late (1,254
  one stop ahead, 4,097 five stops ahead in the test period). They are scored apart.
* **Ranges** -- a 10th-90th percentile band for the delay, judged by how often the
  truth falls inside (it should be ~80%) and how wide the band is.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

from . import config, features, model_delay as md
from .progress import Progress

log = logging.getLogger(__name__)

METRICS_PATH = config.PROCESSED_DIR / "tail_metrics.json"

BIG_DELAY_SECONDS = 600               # "10+ minutes late"
ONSET_BELOW_SECONDS = config.LATE_THRESHOLD_SECONDS   # "on time now": under 5 min late
HORIZONS = (1, 5)
QUANTILES = (0.1, 0.5, 0.9)
#: Recall is reported at this precision: of the trains flagged, half really are 10+ late.
TARGET_PRECISION = 0.5
CALIBRATION_EDGES = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
#: Share of the latest training days held back to calibrate the ranges.
CONFORMAL_FRACTION = 0.2


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def recall_at_precision(y: np.ndarray, score: np.ndarray, precision: float) -> float:
    """Largest share of positives caught while keeping precision >= ``precision``."""
    if not np.any(y):
        return float("nan")
    p, r, _ = precision_recall_curve(y, score)
    ok = p >= precision
    return float(r[ok].max()) if ok.any() else 0.0


def warning_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    return {"n": int(len(y)), "positives": int(np.sum(y)),
            "pr_auc": float(average_precision_score(y, score)) if np.any(y) else float("nan"),
            "recall_at_50pct_precision": recall_at_precision(y, score, TARGET_PRECISION)}


def calibration_table(y: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """Predicted probability vs how often it actually happened, per probability bin."""
    bins = pd.cut(proba, CALIBRATION_EDGES, include_lowest=True)
    table = pd.DataFrame({"y": y, "p": proba, "bin": bins}).groupby("bin", observed=True).agg(
        arrivals=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    return table.reset_index().assign(bin=lambda t: t["bin"].astype(str))


def band_metrics(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict:
    """Coverage and width of a prediction band."""
    return {"coverage": float(np.mean((y >= low) & (y <= high))),
            "median_width_seconds": float(np.median(high - low))}


def conformal_margin(y: np.ndarray, low: np.ndarray, high: np.ndarray,
                     coverage: float) -> float:
    """Split-conformal widening for a quantile band (CQR, Romano et al. 2019).

    Each held-out row scores how far the truth fell outside its band (negative
    when inside). Widening every band by the right quantile of those scores gives
    the requested coverage on data like the held-out rows, whatever the quantile
    models got wrong.
    """
    scores = np.maximum(low - y, y - high)
    level = min(1.0, np.ceil((len(scores) + 1) * coverage) / len(scores))
    return float(np.quantile(scores, level))


def pinball_loss(y: np.ndarray, predicted: np.ndarray, q: float) -> float:
    """The loss quantile regression minimises; lower is better."""
    diff = y - predicted
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


# ---------------------------------------------------------------------------
# One horizon
# ---------------------------------------------------------------------------
def evaluate_horizon(k: int, *, quick: bool) -> dict:
    """Fit and score the early-warning and range models for ``k`` stops ahead."""
    frame = features.build(horizon=k, no_cache=True,
                           max_rows=features.QUICK_MAX_ROWS if quick else None)
    frame["big_delay"] = (frame["delay_seconds"] > BIG_DELAY_SECONDS).astype(int)
    split = md.temporal_split(frame)
    train = features.stratified_sample(split.train, md.TRAIN_ROWS // 4 if quick else md.TRAIN_ROWS,
                                       md.SEED)
    test = split.test
    numeric, categorical = md._columns(train)
    X = test[numeric + categorical]
    y = test["big_delay"].to_numpy()
    onset = (test["prev_delay_1"] < ONSET_BELOW_SECONDS).to_numpy()

    # Averaged over three seeds: with ~1,250 onsets, a single seed's onset PR-AUC
    # ranged from 0.11 to 0.22 on identical data, which is noise, not signal.
    seeds = md.PROBE_SEEDS[:1] if quick else md.PROBE_SEEDS

    def classifier(rows: pd.DataFrame, target: str, seed: int) -> np.ndarray:
        model = HistGradientBoostingClassifier(
            categorical_features="from_dtype", max_iter=150 if quick else 300,
            learning_rate=0.08, early_stopping=True, random_state=seed)
        return md._fit(model, rows, numeric, categorical, target).predict_proba(X)[:, 1]

    per_seed = [classifier(train, "big_delay", seed) for seed in seeds]
    proba = np.mean(per_seed, axis=0)
    # Two ways of aiming at onsets directly, scored like the rest: train only on
    # trains under 5 minutes late now, or target "loses 5+ minutes from here".
    on_time = train[train["prev_delay_1"] < ONSET_BELOW_SECONDS]
    train = train.assign(loses_5min=((train["delay_seconds"] - train["prev_delay_1"])
                                     > ONSET_BELOW_SECONDS).astype(int))
    scores = {
        "model": proba,
        **{f"model, single seed {seed}": p for seed, p in zip(seeds, per_seed)},
        "model trained on on-time trains only": classifier(on_time, "big_delay", md.SEED),
        "model targeting a 5+ minute loss": classifier(train, "loses_5min", md.SEED),
        # Baselines rank trains by one signal each; higher means "more likely late".
        "baseline: how late the train is now": test["prev_delay_1"].to_numpy(dtype=float),
        "baseline: how late the line is now": test["line_late_share_15m"].fillna(0).to_numpy(),
    }
    warning = [{"horizon_stops": k, "method": name, "subset": subset,
                **warning_metrics(y[mask], score[mask])}
               for name, score in scores.items()
               for subset, mask in (("all arrivals", np.ones_like(onset)), ("onsets", onset))]

    delay = test["delay_seconds"].to_numpy()
    # Quantiles of the change since the last stop, shifted by the known delay
    # there, are quantiles of the delay itself -- without the trees' range cap.
    # They are fitted on the earlier training days; the latest are held back to
    # calibrate the band's width (split conformal).
    dates = np.sort(train["service_date"].unique())
    calibration_start = dates[int(len(dates) * (1 - CONFORMAL_FRACTION))]
    fit_rows = train[train["service_date"] < calibration_start]
    calibration = train[train["service_date"] >= calibration_start]
    models = {level: md._fit(md.ChangeRegressor(md._boost(quick, loss="quantile", quantile=level)),
                             fit_rows, numeric, categorical, "delay_seconds")
              for level in QUANTILES}
    q = {level: m.predict(X) for level, m in models.items()}
    cal_cols = calibration[numeric + categorical]
    margin = conformal_margin(calibration["delay_seconds"].to_numpy(),
                              models[0.1].predict(cal_cols), models[0.9].predict(cal_cols),
                              QUANTILES[-1] - QUANTILES[0])
    low, high = q[0.1] - margin, q[0.9] + margin
    days = md.day_type(test)
    ranges = [{"horizon_stops": k, "day_type": label,
               **band_metrics(delay[m], low[m], high[m]),
               "coverage_uncorrected": band_metrics(delay[m], q[0.1][m], q[0.9][m])["coverage"],
               "conformal_margin_seconds": margin,
               "median_abs_error_seconds": float(np.median(np.abs(delay[m] - q[0.5][m]))),
               "pinball_q90": pinball_loss(delay[m], q[0.9][m], 0.9)}
              for label, m in (("all", np.ones(len(test), bool)), ("normal", days == "normal"),
                               ("disrupted", days == "disrupted"))]

    calibration = calibration_table(y, scores["model"]).assign(horizon_stops=k)
    return {"warning": warning, "ranges": ranges, "calibration": calibration.to_dict("records"),
            "base_rate": float(y.mean()), "onsets": int((y.astype(bool) & onset).sum())}


def run(*, quick: bool = False) -> dict:
    """Stage entry point: every horizon, persisted for the figures and report."""
    config.ensure_dirs()
    results: dict[str, list] = {"warning": [], "ranges": [], "calibration": [], "summary": []}
    horizons = HORIZONS[:1] if quick else HORIZONS
    with Progress(len(horizons), "tail models") as progress:
        for k in horizons:
            out = evaluate_horizon(k, quick=quick)
            for key in ("warning", "ranges", "calibration"):
                results[key] += out[key]
            results["summary"].append({"horizon_stops": k, "base_rate": out["base_rate"],
                                       "onsets": out["onsets"]})
            model_all = next(r for r in out["warning"]
                             if r["method"] == "model" and r["subset"] == "all arrivals")
            band = next(r for r in out["ranges"] if r["day_type"] == "all")
            progress.step(f"{k} stop(s) ahead",
                          extra=f"PR-AUC {model_all['pr_auc']:.3f}  band coverage {band['coverage']:.0%}")
    METRICS_PATH.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    log.info("tail metrics -> %s", METRICS_PATH.name)
    return {"horizons": list(horizons)}


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"{METRICS_PATH} missing; run the `tail` stage first")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
