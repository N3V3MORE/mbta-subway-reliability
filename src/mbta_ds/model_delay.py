"""Track A: predict subway arrival delay and lateness.

Two supervised tasks on the same feature table:

* **Regression** -- predict ``delay_seconds`` at each stop.
* **Classification** -- predict ``late`` (more than 5 minutes behind schedule).

Evaluation discipline
---------------------
* **Temporal split**, never a random one. Trains on the earlier 75% of service
  dates and tests on the later 25% -- the question a rider actually asks ("given
  what happened up to yesterday, what happens tomorrow?").
* **Three baselines** are always reported, including delay persistence (predict
  the previous stop's delay). Beating persistence is the real bar here.
* **Ablations** over the declared feature groups quantify what each data source
  contributes, so feature choice is justified with numbers.
* Every preprocessing step lives inside a scikit-learn ``Pipeline``, so it is fit
  on training rows only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import config, features
from .progress import Progress

log = logging.getLogger(__name__)

METRICS_PATH = config.PROCESSED_DIR / "model_metrics.json"
TREND_DIAGNOSTIC_PATH = config.PROCESSED_DIR / "trend_diagnostic.csv"
COMPARISON_PATH = config.PROCESSED_DIR / "model_comparison.csv"
ABLATION_PATH = config.PROCESSED_DIR / "ablation.csv"
PREDICTIONS_PATH = config.PROCESSED_DIR / "predictions.parquet"
CLF_PREDICTIONS_PATH = config.PROCESSED_DIR / "classifier_predictions.parquet"
IMPORTANCE_PATH = config.PROCESSED_DIR / "feature_importance.csv"
BACKTEST_PATH = config.PROCESSED_DIR / "backtest.csv"
HORIZON_PATH = config.PROCESSED_DIR / "horizons.csv"

TEST_FRACTION = 0.25
SEED = 506
#: Training rows per fit, sampled evenly across service dates. Test sets are
#: never sampled: every arrival in the test period is scored.
TRAIN_ROWS = 450_000
#: A route-day where at least this share of arrivals ran more than 10 minutes
#: late is reported separately as "disrupted". On a typical route-day 6% do (15%
#: on Blue and Red), so 20% flags roughly the worst eighth of route-days.
DISRUPTED_SHARE = 0.20
#: Walk-forward backtest: refit before each of the last N windows of this length.
BACKTEST_FOLDS, BACKTEST_DAYS = 4, 14
#: How many stops ahead the horizon experiment predicts.
HORIZONS = (1, 3, 5, 10)
#: KNN is O(n) per prediction and random forests are O(trees x rows); both are
#: capped so `make all` stays in the minutes-not-hours range.
KNN_SAMPLE = 20_000
RF_SAMPLE = 150_000
#: The monotone window-index feature investigated by `run_calendar_diagnostic`.
TREND_FEATURE = "days_since_window_start"
#: The ablation probe is averaged over these seeds: one seed alone moves a row's
#: MAE by up to 2.5 s, more than most feature groups are worth.
PROBE_SEEDS = (SEED, 1, 2)

#: Consume pandas ``category`` columns and NaN natively; no encoding or imputing.
NATIVE_MODELS = (HistGradientBoostingRegressor, HistGradientBoostingClassifier)
#: Distance- and gradient-based models, which need standardised inputs.
SCALED_MODELS = (Ridge, LogisticRegression, KNeighborsRegressor)


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------
@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: object

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)


def temporal_split(frame: pd.DataFrame, test_fraction: float = TEST_FRACTION) -> Split:
    """Split by service date so the test set is strictly in the future.

    Splitting on unique *dates* rather than rows keeps whole days together -- a
    row-level split would place some stops of a trip in train and others in test,
    leaking the trip's own delay path across the boundary.
    """
    dates = np.sort(frame["service_date"].unique())
    if len(dates) < 4:
        raise ValueError(
            f"need at least 4 service dates to split temporally, got {len(dates)}. "
            "Collect a longer window (e.g. `--days 90`)."
        )
    cut_index = max(1, min(int(len(dates) * (1 - test_fraction)), len(dates) - 1))
    cutoff = dates[cut_index - 1]

    train = frame[frame["service_date"] <= cutoff].copy()
    test = frame[frame["service_date"] > cutoff].copy()
    log.info(
        "temporal split at %s: train %s rows (%s..%s), test %s rows (%s..%s)",
        cutoff, f"{len(train):,}", dates[0], cutoff,
        f"{len(test):,}", dates[cut_index], dates[-1],
    )
    return Split(train=train, test=test, cutoff=cutoff)


def _usable_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """Drop feature columns that are absent or entirely missing in this window.

    ``station_entries_trend`` needs 14 days of ridership history, so on a short
    window it is legitimately empty.
    """
    present = [c for c in columns if c in frame.columns]
    empty = [c for c in present if frame[c].isna().all()]
    if empty:
        log.warning("dropping %d all-missing feature(s): %s", len(empty), empty)
    return [c for c in present if c not in empty]


def _columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """The ``(numeric, categorical)`` feature columns usable in ``frame``."""
    numeric = _usable_columns(frame, [c for g in features.FEATURE_GROUPS.values() for c in g])
    return numeric, [c for c in features.CATEGORICAL_FEATURES if c in frame.columns]


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def make_preprocessor(
    numeric: list[str],
    categorical: list[str],
    *,
    scale: bool,
    one_hot: bool = True,
) -> ColumnTransformer:
    """Build the fit-on-train-only preprocessing step.

    ``one_hot=False`` is for histogram gradient boosting, which takes pandas
    ``category`` columns and NaN natively, so both pass through untouched. Leaving
    NaN in place matters: a missing ``prev_delay_1`` means "second stop of the
    trip", and median-imputing it erased that signal (test MAE 44.2 s -> 43.7 s).

    Otherwise numeric columns are median-imputed (and standardised if ``scale``)
    and categoricals one-hot encoded into a *dense* matrix: scikit-learn's trees
    are ~4x slower on the sparse alternative, for identical splits.
    """
    if not one_hot:
        return ColumnTransformer(
            [("num", "passthrough", numeric), ("cat", "passthrough", categorical)],
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [("num", Pipeline(numeric_steps), numeric),
         ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20),
          categorical)],
        verbose_feature_names_out=False,
        sparse_threshold=0.0,
    )


def _fit(model, train: pd.DataFrame, numeric: list[str], categorical: list[str],
         target: str):
    """Fit one candidate on ``train``; learned models are wrapped in a pipeline.

    Baselines are fit as-is. Random forests see at most :data:`RF_SAMPLE` rows,
    since impurity gains on 400k rows are not worth the order-of-magnitude cost.
    """
    if isinstance(model, (RandomForestRegressor, RandomForestClassifier)) and len(train) > RF_SAMPLE:
        train = train.sample(RF_SAMPLE, random_state=SEED)
    return _pipeline(model, numeric, categorical).fit(
        train[numeric + categorical], train[target].to_numpy(dtype=float))


def _pipeline(model, numeric: list[str], categorical: list[str]):
    """Wrap a learned model in its preprocessing; baselines are returned as-is.

    A :class:`ChangeRegressor` keeps the raw columns (it reads the previous
    stop's delay itself) and wraps its inner model instead.
    """
    if isinstance(model, BASELINES):
        return model
    if isinstance(model, ChangeRegressor):
        return ChangeRegressor(_pipeline(model.estimator, numeric, categorical))
    return Pipeline([
        ("prep", make_preprocessor(numeric, categorical,
                                   scale=isinstance(model, SCALED_MODELS),
                                   one_hot=not isinstance(model, NATIVE_MODELS))),
        ("model", model),
    ])


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
class ZeroRegressor:
    """Predict on-time (0 seconds of delay) for every row."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class AlwaysOnTimeClassifier:
    """Majority-class baseline: never predicts "late"."""

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class PersistenceRegressor:
    """Predict the previous stop's delay; fall back to its median when unknown.

    The strongest trivial baseline, because delay is highly autocorrelated along
    a trip.
    """

    def __init__(self, column: str = "prev_delay_1"):
        self.column = column
        self.fill_ = 0.0

    def fit(self, X, y=None):
        if self.column in X.columns and X[self.column].notna().any():
            self.fill_ = float(pd.to_numeric(X[self.column]).median())
        return self

    def predict(self, X):
        if self.column not in X.columns:
            return np.zeros(len(X))
        return pd.to_numeric(X[self.column]).fillna(self.fill_).to_numpy()


class PersistenceClassifier:
    """Predict "late" iff the previous stop was already late.

    The bar the learned classifiers must clear: because delay is autocorrelated
    along a trip, a high F1 on its own means little -- the lift over this matters.
    """

    def __init__(self, column: str = "prev_delay_1", threshold: float = 300.0):
        self.column = column
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        prev = pd.to_numeric(X[self.column]).to_numpy(dtype=float)
        proba = np.where(np.isnan(prev), 0.5,
                         np.clip(0.5 + (prev - self.threshold) / 3600.0, 0.0, 1.0))
        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        # Unknown previous delay counts as "not late" (NaN > threshold is False).
        return (pd.to_numeric(X[self.column]).to_numpy(dtype=float) > self.threshold).astype(int)


class GroupMeanRegressor:
    """Predict the training mean of the target within ``(route, hour)`` groups."""

    def __init__(self, keys: tuple[str, ...] = ("route_id", "scheduled_hour")):
        self.keys = keys

    def _index(self, X) -> pd.MultiIndex:
        return pd.MultiIndex.from_arrays([X[k].astype(str).to_numpy() for k in self.keys])

    def fit(self, X, y):
        y = pd.Series(np.asarray(y, dtype=float), index=self._index(X))
        self.global_ = float(y.mean())
        self.table_ = y.groupby(level=list(range(len(self.keys)))).mean()
        return self

    def predict(self, X):
        return self.table_.reindex(self._index(X)).fillna(self.global_).to_numpy()


class GroupRateClassifier(GroupMeanRegressor):
    """Predict the training lateness rate within ``(route, hour)`` groups."""

    def predict_proba(self, X):
        rate = GroupMeanRegressor.predict(self, X)
        return np.column_stack([1 - rate, rate])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ChangeRegressor(BaseEstimator, RegressorMixin):
    """Predict how much delay a train gains or loses since its last stop.

    The target is ``delay - persistence``, and the prediction adds persistence
    back. A tree model can only output values it saw in training, so asked for
    the delay itself it caps out: in winter, trains logged hours "late" on storm
    days were predicted at a fraction of that while persistence copied them
    forward exactly. The change since the last stop stays small whatever the
    delay is, so this framing never runs out of range.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.base_ = PersistenceRegressor().fit(X)
        self.estimator_ = clone(self.estimator).fit(X, np.asarray(y, dtype=float) - self.base_.predict(X))
        return self

    def predict(self, X):
        return self.base_.predict(X) + self.estimator_.predict(X)


def _change_boost(quick: bool) -> ChangeRegressor:
    """The headline model: absolute-error boosting on the change in delay."""
    return ChangeRegressor(_boost(quick, loss="absolute_error"))


BASELINES = (ZeroRegressor, AlwaysOnTimeClassifier, PersistenceRegressor,
             PersistenceClassifier, GroupMeanRegressor)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def _boost(quick: bool, **overrides) -> HistGradientBoostingRegressor:
    params = dict(categorical_features="from_dtype", max_iter=200 if quick else 400,
                  learning_rate=0.08, max_leaf_nodes=63, early_stopping=True,
                  validation_fraction=0.1, random_state=SEED)
    return HistGradientBoostingRegressor(**{**params, **overrides})


def regression_models(quick: bool, full: bool = False) -> dict[str, object]:
    """Return the regression candidate set for the requested run size.

    ``hist_gradient_boosting_mae`` optimises absolute error, the metric the
    project is scored on; the squared-error version chases the long right tail
    at the expense of the typical arrival. ``hist_gradient_boosting_change`` is
    the same model predicting the change since the last stop (see
    :class:`ChangeRegressor`). The random forest is opt-in via
    ``full`` because its cost is super-linear in the training rows.
    """
    candidates: dict[str, object] = {
        "baseline_zero": ZeroRegressor(),
        "baseline_persistence": PersistenceRegressor(),
        "baseline_route_hour_mean": GroupMeanRegressor(),
        "ridge": Ridge(alpha=1.0, random_state=SEED),
        "decision_tree": DecisionTreeRegressor(max_depth=12 if quick else 18,
                                               min_samples_leaf=20, random_state=SEED),
        "hist_gradient_boosting": _boost(quick),
        "hist_gradient_boosting_mae": _boost(quick, loss="absolute_error"),
        "hist_gradient_boosting_change": _change_boost(quick),
    }
    if full:
        candidates["random_forest"] = RandomForestRegressor(
            n_estimators=100, max_samples=0.4, min_samples_leaf=20, max_depth=24,
            n_jobs=-1, random_state=SEED,
        )
    return candidates


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Regression metrics, including a robust variant that ignores the tail."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    keep = np.abs(y_true) <= 3600  # exclude the flagged tail for the robust view
    return {
        "mae_seconds": float(mean_absolute_error(y_true, y_pred)),
        "rmse_seconds": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mae_seconds_trimmed": float(mean_absolute_error(y_true[keep], y_pred[keep]))
        if keep.any() else float("nan"),
        "bias_seconds": float(np.mean(y_pred - y_true)),
        "n": int(len(y_true)),
    }


def run_regression(split: Split, *, quick: bool, full: bool = False
                   ) -> tuple[pd.DataFrame, dict, str]:
    """Fit every regression candidate and return the comparison table."""
    numeric, categorical = _columns(split.train)
    cols = numeric + categorical
    y_test = split.test["delay_seconds"].to_numpy()

    rows: list[dict] = []
    fitted: dict[str, object] = {}
    candidates = regression_models(quick, full=full)
    with Progress(len(candidates) + full, "regression models") as progress:
        for name, model in candidates.items():
            fitted[name] = _fit(model, split.train, numeric, categorical, "delay_seconds")
            metrics = evaluate_regression(y_test, fitted[name].predict(split.test[cols]))
            rows.append({"model": name, **metrics})
            progress.step(name, extra=f"MAE {metrics['mae_seconds']:.1f}s  R2 {metrics['r2']:.3f}")

        if full:
            # KNN is O(n) per prediction, so it trains and is scored on bounded
            # samples; the flag keeps the comparison honest.
            train = split.train.sample(min(KNN_SAMPLE, split.n_train), random_state=SEED)
            test = split.test.sample(min(KNN_SAMPLE, split.n_test), random_state=SEED)
            fitted["knn"] = _fit(KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1),
                                 train, numeric, categorical, "delay_seconds")
            metrics = evaluate_regression(test["delay_seconds"], fitted["knn"].predict(test[cols]))
            rows.append({"model": "knn", "evaluated_on_subsample": True, **metrics})
            progress.step("knn", extra=f"MAE {metrics['mae_seconds']:.1f}s (on {metrics['n']:,} rows)")

    comparison = pd.DataFrame(rows).sort_values("mae_seconds").reset_index(drop=True)
    return comparison, fitted, comparison.iloc[0]["model"]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classification_models(quick: bool, full: bool = False) -> dict[str, object]:
    candidates: dict[str, object] = {
        "baseline_always_ontime": AlwaysOnTimeClassifier(),
        "baseline_persistence": PersistenceClassifier(),
        "baseline_route_hour_rate": GroupRateClassifier(),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "decision_tree": DecisionTreeClassifier(max_depth=12 if quick else 18,
                                                min_samples_leaf=20, random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            categorical_features="from_dtype", max_iter=150 if quick else 300,
            learning_rate=0.08, early_stopping=True, random_state=SEED,
        ),
    }
    if full:
        candidates["random_forest"] = RandomForestClassifier(
            n_estimators=100, max_samples=0.4, min_samples_leaf=20, max_depth=24,
            n_jobs=-1, random_state=SEED,
        )
    return candidates


def evaluate_classification(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Threshold-free and thresholded classification metrics."""
    y_true = np.asarray(y_true, dtype=int)
    predicted = (proba >= 0.5).astype(int)
    both_classes = len(np.unique(y_true)) > 1
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "n": int(len(y_true)),
        "base_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, proba)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)) if both_classes else float("nan"),
    }


def run_classification(split: Split, *, quick: bool, full: bool = False
                       ) -> tuple[pd.DataFrame, dict, dict[str, np.ndarray]]:
    """Fit every classification candidate.

    Returns the comparison table, the fitted models, and each model's test-set
    probabilities (kept so the ROC / PR artefact does not re-score every model).
    """
    numeric, categorical = _columns(split.train)
    y_test = split.test["late"].astype(int).to_numpy()

    rows: list[dict] = []
    fitted: dict[str, object] = {}
    probabilities: dict[str, np.ndarray] = {}
    candidates = classification_models(quick, full=full)
    with Progress(len(candidates), "classifiers") as progress:
        for name, model in candidates.items():
            fitted[name] = _fit(model, split.train, numeric, categorical, "late")
            probabilities[name] = fitted[name].predict_proba(split.test[numeric + categorical])[:, 1]
            metrics = evaluate_classification(y_test, probabilities[name])
            rows.append({"model": name, **metrics})
            progress.step(name, extra=f"F1 {metrics['f1']:.3f}  AUC {metrics['roc_auc']:.3f}  "
                                      f"acc {metrics['accuracy']:.3f}")

    comparison = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    return comparison, fitted, probabilities


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------
ABLATION_ORDER = (
    ("schedule", ("schedule",)),
    ("+ route/station", ("schedule", "categorical")),
    ("+ calendar", ("schedule", "categorical", "calendar")),
    ("+ delay propagation", ("schedule", "categorical", "calendar", "propagation")),
    ("+ demand", ("schedule", "categorical", "calendar", "propagation", "demand")),
    ("+ weather", ("schedule", "categorical", "calendar", "propagation", "demand",
                   "weather")),
    ("+ alerts", ("schedule", "categorical", "calendar", "propagation", "demand",
                  "weather", "alerts")),
    ("+ other trains", ("schedule", "categorical", "calendar", "propagation", "demand",
                        "weather", "alerts", "network")),
    ("+ same train's last trip", ("schedule", "categorical", "calendar", "propagation", "demand",
                                  "weather", "alerts", "network", "vehicle")),
)


def _probe(split: Split, numeric: list[str], categorical: list[str], *, quick: bool,
           cache: dict) -> dict:
    """Test metrics of the fixed ablation probe (squared-error boosting).

    The probe is held fixed so every feature set is judged by the same model, and
    its metrics are averaged over :data:`PROBE_SEEDS` (one seed in ``quick`` mode),
    with the seed-to-seed MAE spread reported so small differences can be judged.
    Results are memoised by feature *set*: the time-trend diagnostic repeats four
    of the ablation's configurations, which would otherwise be refit.
    """
    numeric = _usable_columns(split.train, numeric)
    key = (frozenset(numeric), frozenset(categorical))
    if key not in cache:
        runs = pd.DataFrame([
            evaluate_regression(
                split.test["delay_seconds"],
                _fit(_boost(quick, max_iter=120 if quick else 250, max_leaf_nodes=31,
                            random_state=seed),
                     split.train, numeric, categorical, "delay_seconds")
                .predict(split.test[numeric + categorical]))
            for seed in (PROBE_SEEDS[:1] if quick else PROBE_SEEDS)
        ])
        cache[key] = {"n_features": len(numeric) + len(categorical), **runs.mean().to_dict(),
                      "mae_seed_spread": runs["mae_seconds"].max() - runs["mae_seconds"].min()}
    return cache[key]


def run_ablation(split: Split, *, quick: bool, cache: dict | None = None) -> pd.DataFrame:
    """Measure test MAE as feature groups are added cumulatively."""
    cache = {} if cache is None else cache
    rows: list[dict] = []
    with Progress(len(ABLATION_ORDER), "ablation") as progress:
        for label, groups in ABLATION_ORDER:
            columns = features.all_feature_columns(groups=groups)
            categorical = [c for c in columns if c in features.CATEGORICAL_FEATURES]
            numeric = [c for c in columns if c not in categorical]
            metrics = _probe(split, numeric, categorical, quick=quick, cache=cache)
            rows.append({"features": label, **metrics})
            progress.step(label, extra=f"{metrics['n_features']} feats  "
                                       f"MAE {metrics['mae_seconds']:.1f}s")

    ablation = pd.DataFrame(rows)
    ablation["mae_improvement_vs_schedule_only_seconds"] = (
        ablation.iloc[0]["mae_seconds"] - ablation["mae_seconds"]
    )
    return ablation


def run_calendar_diagnostic(split: Split, *, quick: bool, cache: dict | None = None) -> dict:
    """Isolate the effect of the monotone time-trend feature on test error.

    ``days_since_window_start`` is a monotone index of the analysis window: under
    a temporal split its training values span [0, 66] and its test values [67, 89],
    so the two ranges share no values. A tree that splits on it extrapolates off
    the end of its training data, and no in-training validation exposes that. It
    is therefore not a model feature; this diagnostic re-adds it to measure the
    effect with the trend, without it, and alone.
    """
    cache = {} if cache is None else cache
    schedule = list(features.FEATURE_GROUPS["schedule"])
    propagation = list(features.FEATURE_GROUPS["propagation"])
    without_trend = list(features.FEATURE_GROUPS["calendar"])
    calendar = without_trend + [TREND_FEATURE]
    categorical = list(features.CATEGORICAL_FEATURES)

    configs = (
        ("schedule only", schedule, []),
        ("+ route/station", schedule, categorical),
        ("+ calendar (with time trend)", schedule + calendar, categorical),
        ("+ calendar (time trend removed)", schedule + without_trend, categorical),
        ("+ time trend alone", schedule + [TREND_FEATURE], categorical),
        ("+ propagation (with time trend)", schedule + propagation + calendar, categorical),
        ("+ propagation (trend removed)", schedule + propagation + without_trend, categorical),
    )
    rows: list[dict] = []
    with Progress(len(configs), "time-trend diagnostic") as progress:
        for label, numeric, cats in configs:
            metrics = _probe(split, numeric, cats, quick=quick, cache=cache)
            rows.append({"configuration": label, "includes_time_trend": TREND_FEATURE in numeric,
                         **metrics})
            progress.step(label, extra=f"MAE {metrics['mae_seconds']:.1f}s  R2 {metrics['r2']:.3f}")

    train_values = pd.to_numeric(split.train[TREND_FEATURE]).dropna()
    test_values = pd.to_numeric(split.test[TREND_FEATURE]).dropna()
    ranges = {
        "feature": TREND_FEATURE,
        "train_min": float(train_values.min()),
        "train_max": float(train_values.max()),
        "test_min": float(test_values.min()),
        "test_max": float(test_values.max()),
        "shared_values": len(set(train_values) & set(test_values)),
    }
    log.info("  %s: train [%.0f, %.0f] vs test [%.0f, %.0f], %d shared values",
             TREND_FEATURE, ranges["train_min"], ranges["train_max"],
             ranges["test_min"], ranges["test_max"], ranges["shared_values"])

    pd.DataFrame(rows).to_csv(TREND_DIAGNOSTIC_PATH, index=False)
    return {"table": rows, "trend_feature_range": ranges}


# ---------------------------------------------------------------------------
# Error analysis and importance
# ---------------------------------------------------------------------------
def day_type(frame: pd.DataFrame) -> np.ndarray:
    """Label each row's route-day "disrupted" or "normal" (see DISRUPTED_SHARE)."""
    late_share = (frame["delay_seconds"] > 600).groupby(
        [frame["route_id"].astype(str), frame["service_date"]]).transform("mean")
    return np.where(late_share >= DISRUPTED_SHARE, "disrupted", "normal")


def error_analysis(test: pd.DataFrame, predictions: np.ndarray) -> dict:
    """Break test error down by route, hour band and weather condition."""
    frame = test[["route_id", "scheduled_hour"]].copy()
    frame["abs_error"] = np.abs(predictions - test["delay_seconds"].to_numpy())
    frame["hour_band"] = pd.cut(
        frame["scheduled_hour"], bins=[-1, 5, 9, 15, 19, 23],
        labels=["overnight", "am_peak", "midday", "pm_peak", "evening"],
    )
    frame["condition"] = np.select(
        [test["is_snow"].fillna(0) > 0, test["precip_mm"].fillna(0) > 0],
        ["snow", "precip"], "dry",
    )
    # A handful of bad days dominates the average, so they are reported apart.
    frame["day_type"] = day_type(test)

    def _agg(by: str) -> list[dict]:
        out = frame.groupby(by, observed=True)["abs_error"].agg(["mean", "median", "count"])
        out = out.reset_index().set_axis([by, "mae", "median_abs_error", "n"], axis=1)
        return out.sort_values("mae", ascending=False).round(2).to_dict("records")

    return {"by_route": _agg("route_id"), "by_hour_band": _agg("hour_band"),
            "by_condition": _agg("condition"), "by_day_type": _agg("day_type")}


# ---------------------------------------------------------------------------
# Robustness: walk-forward backtest and prediction horizon
# ---------------------------------------------------------------------------
def _mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(predicted) - actual)))


def run_backtest(frame: pd.DataFrame, *, quick: bool) -> pd.DataFrame:
    """Refit before each of the last few 14-day windows and score that window.

    One train/test cut-off gives one number; several show how much the result
    depends on *which* weeks happen to be the test set.
    """
    dates = np.sort(frame["service_date"].unique())
    numeric, categorical = _columns(frame)
    folds = 2 if quick else BACKTEST_FOLDS
    rows: list[dict] = []
    with Progress(folds, "walk-forward backtest") as progress:
        for i in range(folds, 0, -1):
            window = dates[len(dates) - i * BACKTEST_DAYS: len(dates) - (i - 1) * BACKTEST_DAYS]
            test = frame[frame["service_date"].isin(window)]
            train = features.stratified_sample(frame[frame["service_date"] < window[0]],
                                               TRAIN_ROWS // 4 if quick else TRAIN_ROWS, SEED)
            y = test["delay_seconds"].to_numpy()
            model = _fit(_change_boost(quick), train, numeric, categorical, "delay_seconds")
            rows.append({
                "test_start": str(window[0]), "test_end": str(window[-1]),
                "train_days": int(train["service_date"].nunique()), "n_test": len(test),
                "persistence_mae": _mae(PersistenceRegressor().fit(train).predict(test), y),
                "model_mae": _mae(model.predict(test[numeric + categorical]), y),
            })
            progress.step(f"{window[0]}..{window[-1]}",
                          extra=f"MAE {rows[-1]['model_mae']:.1f}s vs {rows[-1]['persistence_mae']:.1f}s")
    table = pd.DataFrame(rows)
    table["improvement_pct"] = 100 * (1 - table["model_mae"] / table["persistence_mae"])
    return table


def run_horizons(*, quick: bool, cutoff) -> pd.DataFrame:
    """Predict k stops ahead, with and without the other trains on the line.

    One stop ahead, "same as the last stop" is hard to beat because delay barely
    changes stop to stop. Further ahead there is room for a model to add value,
    and it is where a rider actually needs a prediction.
    """
    rows: list[dict] = []
    horizons = HORIZONS[:2] if quick else HORIZONS
    network = set(features.FEATURE_GROUPS["network"])
    with Progress(len(horizons), "prediction horizon") as progress:
        for k in horizons:
            frame = features.build(horizon=k, no_cache=True,
                                   max_rows=features.QUICK_MAX_ROWS if quick else None)
            train = features.stratified_sample(frame[frame["service_date"] <= cutoff],
                                               TRAIN_ROWS // 4 if quick else TRAIN_ROWS, SEED)
            test = frame[frame["service_date"] > cutoff]
            numeric, categorical = _columns(train)
            y = test["delay_seconds"].to_numpy()
            row = {"horizon_stops": k, "n_test": len(test),
                   "persistence_mae": _mae(PersistenceRegressor().fit(train).predict(test), y)}
            for label, cols in (("own_train_mae", [c for c in numeric if c not in network]),
                                ("with_other_trains_mae", numeric)):
                model = _fit(_change_boost(quick), train, cols, categorical, "delay_seconds")
                row[label] = _mae(model.predict(test[cols + categorical]), y)
            rows.append(row)
            progress.step(f"{k} stop(s) ahead",
                          extra=f"persistence {row['persistence_mae']:.1f}s  own {row['own_train_mae']:.1f}s  "
                                f"+others {row['with_other_trains_mae']:.1f}s")
    return pd.DataFrame(rows)


def compute_importance(pipeline, test: pd.DataFrame, numeric: list[str],
                       categorical: list[str], *, sample: int = 8_000) -> pd.DataFrame:
    """Permutation importance on a test subsample.

    Used rather than impurity importance, which is biased toward high-cardinality
    features -- and this model has a 124-level station feature.
    """
    subset = test.sample(min(sample, len(test)), random_state=SEED)
    result = permutation_importance(
        pipeline, subset[numeric + categorical], subset["delay_seconds"].to_numpy(),
        n_repeats=5, random_state=SEED, n_jobs=-1, scoring="neg_mean_absolute_error",
    )
    return (pd.DataFrame({"feature": numeric + categorical,
                          "importance_mae": result.importances_mean,
                          "importance_std": result.importances_std})
            .sort_values("importance_mae", ascending=False).reset_index(drop=True))


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(*, quick: bool = False, full: bool = False) -> dict:
    """Run both tasks and persist every artefact the report and figures need."""
    config.ensure_dirs()
    frame = features.load()
    split = temporal_split(frame)
    # Train on an even sample of the earlier dates; score every later arrival.
    split.train = features.stratified_sample(split.train, TRAIN_ROWS // 4 if quick else TRAIN_ROWS, SEED)
    numeric, categorical = _columns(split.train)
    X_test = split.test[numeric + categorical]

    with Progress(7, "train stage") as stage:
        log.info("--- regression ---")
        reg_comparison, reg_fitted, _ = run_regression(split, quick=quick, full=full)
        stage.step("regression")

        log.info("--- classification ---")
        clf_comparison, _, clf_probabilities = run_classification(split, quick=quick, full=full)
        stage.step("classification")

        probe_cache: dict = {}
        log.info("--- ablation ---")
        ablation = run_ablation(split, quick=quick, cache=probe_cache)
        stage.step("ablation")

        log.info("--- calendar / time-trend diagnostic ---")
        trend_diagnostic = run_calendar_diagnostic(split, quick=quick, cache=probe_cache)
        stage.step("time-trend diagnostic")

        # The best *learned* model is reported and analysed, even if a baseline
        # happened to win outright.
        learned = reg_comparison[~reg_comparison["model"].str.startswith("baseline_")]
        chosen_name = learned.iloc[0]["model"]
        chosen = reg_fitted[chosen_name]
        test_predictions = chosen.predict(X_test)
        analysis = error_analysis(split.test, test_predictions)

        log.info("--- feature importance (%s) ---", chosen_name)
        importance = compute_importance(chosen, split.test, numeric, categorical)
        stage.step("importance and artefacts")

        log.info("--- walk-forward backtest ---")
        backtest = run_backtest(frame, quick=quick)
        del frame  # the horizon experiment builds its own tables
        stage.step("backtest")

        log.info("--- prediction horizon ---")
        horizons = run_horizons(quick=quick, cutoff=split.cutoff)
        stage.step("horizons")

    # Persist everything the figures and README depend on.
    predictions = split.test[
        ["service_date_parsed", "trip_id", "stop_id", "station_name", "route_id",
         "trunk_route_id", "scheduled_hour", "stop_sequence",
         "delay_seconds", "late", "prev_delay_1"]
    ].assign(predicted_delay=test_predictions,
             persistence_delay=reg_fitted["baseline_persistence"].predict(X_test))
    predictions["abs_error"] = (predictions["predicted_delay"] - predictions["delay_seconds"]).abs()
    predictions.to_parquet(PREDICTIONS_PATH, index=False)

    reg_comparison.to_csv(COMPARISON_PATH, index=False)
    ablation.to_csv(ABLATION_PATH, index=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    backtest.to_csv(BACKTEST_PATH, index=False)
    horizons.to_csv(HORIZON_PATH, index=False)
    split.test[["service_date_parsed", "late"]].assign(
        **{f"proba_{name}": p for name, p in clf_probabilities.items()}
    ).to_parquet(CLF_PREDICTIONS_PATH, index=False)

    chosen_metrics = learned.iloc[0].to_dict()
    metrics = {
        "split": {
            "cutoff_service_date": str(split.cutoff),
            "n_train": split.n_train,
            "n_test": split.n_test,
            "test_fraction": TEST_FRACTION,
            "train_dates": [str(split.train["service_date"].min()),
                            str(split.train["service_date"].max())],
            "test_dates": [str(split.test["service_date"].min()),
                           str(split.test["service_date"].max())],
        },
        "regression": reg_comparison.to_dict("records"),
        "classification": clf_comparison.to_dict("records"),
        "best_regression_model": chosen_name,
        "best_regression_metrics": chosen_metrics,
        "best_classification_model": clf_comparison.iloc[0]["model"],
        "best_classification_metrics": clf_comparison.iloc[0].to_dict(),
        "ablation": ablation.to_dict("records"),
        "trend_diagnostic": trend_diagnostic,
        "error_analysis": analysis,
        "top_features": importance.head(15).to_dict("records"),
        "backtest": backtest.to_dict("records"),
        "horizons": horizons.to_dict("records"),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    log.info("best regression: %s (MAE %.1fs)", chosen_name, chosen_metrics["mae_seconds"])
    return {
        "best_model": chosen_name,
        "best_mae_seconds": float(chosen_metrics["mae_seconds"]),
        "best_r2": float(chosen_metrics["r2"]),
        "n_train": split.n_train,
        "n_test": split.n_test,
    }


def load_metrics() -> dict:
    """Load the persisted metrics bundle."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"{METRICS_PATH} missing; run the `train` stage first")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
