"""Track A: predict subway arrival delay and lateness.

Two supervised tasks on the same feature table:

* **Regression** -- predict ``delay_seconds`` at each stop, against the timetable.
* **Classification** -- predict ``late``: riders wait more than 5 minutes longer
  than planned for the train (``clean.lateness``).

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
SEED = config.SEED
#: Training rows per fit, sampled evenly across service dates. Test sets are
#: never sampled: every arrival in the test period is scored.
TRAIN_ROWS = 450_000
#: A route-day where at least this share of arrivals kept riders waiting 10+
#: minutes longer than planned is reported separately as "disrupted". On a typical
#: route-day 2.6% do, so 8% flags about the worst 5% of spring route-days (18% of
#: winter's, storms included).
DISRUPTED_SHARE = 0.08
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
#: The ablation probe is averaged over these seeds, and their spread is reported,
#: because most feature groups are worth less than one seed's noise.
PROBE_SEEDS = (SEED, 1, 2)

#: The models reported and analysed as the result, fixed in advance. Picking
#: whichever candidate scores best on the test period would make that score an
#: optimistic, selected-on-test estimate; every candidate is still reported.
HEADLINE_REGRESSOR = "hist_gradient_boosting_change"
HEADLINE_CLASSIFIER = "hist_gradient_boosting"

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


def _columns(frame: pd.DataFrame, *, lateness: bool = False) -> tuple[list[str], list[str]]:
    """The ``(numeric, categorical)`` feature columns usable in ``frame``.

    ``lateness`` adds the rider-lateness group, for the models judged on it; the
    timetable-delay regression leaves it out.
    """
    groups = [g for g in features.FEATURE_GROUPS
              if lateness or g not in features.LATENESS_ONLY_GROUPS]
    numeric = _usable_columns(frame, [c for g in groups for c in features.FEATURE_GROUPS[g]])
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
    NaN in place matters: a missing ``prev_delay_2`` means "first stop after the
    origin", which median imputation would erase.

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


#: Share of the latest training service dates held out to decide when boosting
#: stops. A random row split (scikit-learn's default) puts stops of one trip on
#: both sides, which flatters the validation loss.
EARLY_STOP_FRACTION = 0.1


def _early_stopping(model) -> bool:
    inner = model.estimator if isinstance(model, (ChangeRegressor, RunTimeChangeRegressor)) else model
    return isinstance(inner, NATIVE_MODELS) and inner.early_stopping is True


def temporal_validation(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Split off the latest :data:`EARLY_STOP_FRACTION` of service dates, or ``None``
    when there are too few dates to hold any out."""
    dates = np.sort(train["service_date"].unique())
    held = max(1, int(round(len(dates) * EARLY_STOP_FRACTION)))
    if len(dates) < 5:
        return None
    cut = dates[-held]
    return train[train["service_date"] < cut], train[train["service_date"] >= cut]


def _fit(model, train: pd.DataFrame, numeric: list[str], categorical: list[str],
         target: str):
    """Fit one candidate on ``train``; learned models are wrapped in a pipeline.

    Baselines are hand-written rules fit on the whole frame (see :func:`_inputs`).
    Random forests see at most :data:`RF_SAMPLE` rows, since impurity gains on
    400k rows are not worth the order-of-magnitude cost. Boosted models stop early
    on the latest training dates (:func:`temporal_validation`).
    """
    if isinstance(model, BASELINES):
        return model.fit(train, train[target].to_numpy(dtype=float))
    if isinstance(model, (RandomForestRegressor, RandomForestClassifier)) and len(train) > RF_SAMPLE:
        train = train.sample(RF_SAMPLE, random_state=SEED)
    # The lookup corrector reads the departure and timetable columns itself.
    cols = slice(None) if isinstance(model, RunTimeChangeRegressor) else numeric + categorical
    pipeline = _pipeline(model, numeric, categorical)
    split = temporal_validation(train) if _early_stopping(model) else None
    if split is None:
        return pipeline.fit(train[cols], train[target].to_numpy(dtype=float))
    fit_rows, val_rows = split
    val = {"X_val": val_rows[cols], "y_val": val_rows[target].to_numpy(dtype=float)}
    if not isinstance(pipeline, (ChangeRegressor, RunTimeChangeRegressor)):
        val = {f"model__{k}": v for k, v in val.items()}
    return pipeline.fit(fit_rows[cols], fit_rows[target].to_numpy(dtype=float), **val)


def _needs_frame(model) -> bool:
    """Candidates that read columns beyond the feature set (all known before arrival)."""
    return isinstance(model, BASELINES + (RunTimeChangeRegressor,))


def _inputs(model, frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """What a fitted candidate predicts from: learned models see only their
    feature columns; baselines may also read the prediction moment and the
    timetable (``known_at``, ``scheduled_epoch``), both known before arrival."""
    return frame if _needs_frame(model) else frame[cols]


def _pipeline(model, numeric: list[str], categorical: list[str]):
    """Wrap a learned model in its preprocessing; baselines are returned as-is.

    A :class:`ChangeRegressor` keeps the raw columns (it reads the previous
    stop's delay itself) and wraps its inner model instead.
    """
    if isinstance(model, BASELINES):
        return model
    if isinstance(model, ChangeRegressor):
        return ChangeRegressor(_pipeline(model.estimator, numeric, categorical))
    if isinstance(model, RunTimeChangeRegressor):
        return RunTimeChangeRegressor(_pipeline(model.estimator, numeric + [LOOKUP_FEATURE], categorical),
                                      columns=numeric + categorical, folds=model.folds)
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
    """Predict "late" iff the train was already late at its previous stop.

    The bar the learned classifiers must clear: because delay is autocorrelated
    along a trip, a high F1 on its own means little -- the lift over this matters.
    """

    def __init__(self, column: str = "prev_lateness_1", threshold: float = 300.0):
        self.column = column
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        prev = pd.to_numeric(X[self.column]).to_numpy(dtype=float)
        below_half = np.nextafter(0.5, 0.0)
        proba = np.where(np.isnan(prev), below_half,
                         np.clip(0.5 + (prev - self.threshold) / 3600.0, 0.0, 1.0))
        # Evaluation labels scores >= 0.5 as late. Keep its decision identical
        # to predict(), including equality, missing values and rounding near it.
        proba = np.where(prev > self.threshold, np.maximum(proba, 0.5),
                         np.minimum(proba, below_half))
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

    def fit(self, X, y, X_val=None, y_val=None):
        self.base_ = PersistenceRegressor().fit(X)
        change = np.asarray(y, dtype=float) - self.base_.predict(X)
        extra = {}
        if X_val is not None:
            # The early-stopping set is judged on the same target: the change.
            extra = {"X_val": X_val,
                     "y_val": np.asarray(y_val, dtype=float) - self.base_.predict(X_val)}
            if isinstance(self.estimator, Pipeline):
                extra = {f"model__{k}": v for k, v in extra.items()}
        self.estimator_ = clone(self.estimator).fit(X, change, **extra)
        return self

    def predict(self, X):
        return self.base_.predict(X) + self.estimator_.predict(X)


class RunTimeRegressor:
    """Departure time plus the usual running time to this stop.

    The textbook transit arrival-prediction baseline. At the
    prediction moment ``known_at`` the train has just left an earlier stop, so
    predict its arrival as that moment plus the median time trains took from there
    to this stop in training (per route, direction and stop). Unlike persistence it
    uses the departure time, which the learned models also know (through the
    previous dwell) -- and which matters most at a run's second stop, after the
    train has waited at the origin.
    """

    keys = ("route_id", "direction_id", "stop_id")

    def _index(self, X) -> pd.MultiIndex:
        return pd.MultiIndex.from_arrays([X[k].astype(str).to_numpy() for k in self.keys])

    def fit(self, X, y):
        run = pd.Series(X["scheduled_epoch"].to_numpy(dtype=float) + np.asarray(y, dtype=float)
                        - X["known_at"].to_numpy(dtype=float), index=self._index(X))
        self.fill_ = float(run.median())
        self.table_ = run.groupby(level=list(range(len(self.keys)))).median()
        # Without a prediction moment there is no departure to add to: persist.
        self.fallback_ = PersistenceRegressor().fit(X)
        return self

    def predict(self, X):
        run = self.table_.reindex(self._index(X)).fillna(self.fill_).to_numpy()
        predicted = X["known_at"].to_numpy(dtype=float) + run - X["scheduled_epoch"].to_numpy(dtype=float)
        return np.where(np.isnan(predicted), self.fallback_.predict(X), predicted)


#: The run-time lookup's prediction, given to the correcting model as a feature.
LOOKUP_FEATURE = "run_time_lookup"


class RunTimeChangeRegressor(BaseEstimator, RegressorMixin):
    """The run-time lookup, plus a learned correction to it.

    :class:`ChangeRegressor` corrects persistence, which does not know when the
    train left its last stop; this corrects :class:`RunTimeRegressor`, which does.
    The lookup is also passed in as a feature. Its training-row predictions are
    cross-fitted by service date: a median fitted on the very rows it predicts
    already contains their answers, and the correction would learn from that.
    """

    def __init__(self, estimator, columns=None, folds: int = 5):
        self.estimator = estimator
        self.columns = columns
        self.folds = folds

    def _with_lookup(self, X, lookup):
        return X[self.columns].assign(**{LOOKUP_FEATURE: lookup})

    def _out_of_fold(self, X, y) -> np.ndarray:
        dates = X["service_date"].astype(str).to_numpy()
        unique = np.unique(dates)
        if len(unique) < 2:
            return RunTimeRegressor().fit(X, y).predict(X)
        fold_of = dict(zip(unique, np.arange(len(unique)) % min(self.folds, len(unique))))
        fold = np.array([fold_of[d] for d in dates])
        out = np.empty(len(X))
        for f in np.unique(fold):
            held = fold == f
            out[held] = RunTimeRegressor().fit(X[~held], y[~held]).predict(X[held])
        return out

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y, dtype=float)
        self.base_ = RunTimeRegressor().fit(X, y)
        lookup = self._out_of_fold(X, y)
        extra = {}
        if X_val is not None:
            val_lookup = self.base_.predict(X_val)
            extra = {"X_val": self._with_lookup(X_val, val_lookup),
                     "y_val": np.asarray(y_val, dtype=float) - val_lookup}
            if isinstance(self.estimator, Pipeline):
                extra = {f"model__{k}": v for k, v in extra.items()}
        self.estimator_ = clone(self.estimator).fit(self._with_lookup(X, lookup), y - lookup, **extra)
        return self

    def predict(self, X):
        lookup = self.base_.predict(X)
        return lookup + self.estimator_.predict(self._with_lookup(X, lookup))


def _change_boost(quick: bool) -> ChangeRegressor:
    """The headline model: absolute-error boosting on the change in delay."""
    return ChangeRegressor(_boost(quick, loss="absolute_error"))


BASELINES = (ZeroRegressor, AlwaysOnTimeClassifier, PersistenceRegressor,
             PersistenceClassifier, GroupMeanRegressor, RunTimeRegressor)


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
        "baseline_run_time": RunTimeRegressor(),
        "ridge": Ridge(alpha=1.0, random_state=SEED),
        "decision_tree": DecisionTreeRegressor(max_depth=12 if quick else 18,
                                               min_samples_leaf=20, random_state=SEED),
        "hist_gradient_boosting": _boost(quick),
        "hist_gradient_boosting_mae": _boost(quick, loss="absolute_error"),
        "hist_gradient_boosting_change": _change_boost(quick),
        "hist_gradient_boosting_run_time": RunTimeChangeRegressor(_boost(quick, loss="absolute_error")),
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
                   ) -> tuple[pd.DataFrame, dict]:
    """Fit every regression candidate; return the comparison table and the fits.

    Deliberately no "best model": the headline is :data:`HEADLINE_REGRESSOR`,
    fixed in advance, never the candidate that happened to score best on test.
    """
    numeric, categorical = _columns(split.train)
    cols = numeric + categorical
    y_test = split.test["delay_seconds"].to_numpy()

    rows: list[dict] = []
    fitted: dict[str, object] = {}
    candidates = regression_models(quick, full=full)
    with Progress(len(candidates) + full, "regression models") as progress:
        for name, model in candidates.items():
            fitted[name] = _fit(model, split.train, numeric, categorical, "delay_seconds")
            metrics = evaluate_regression(y_test, fitted[name].predict(_inputs(model, split.test, cols)))
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
    return comparison, fitted


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
    numeric, categorical = _columns(split.train, lateness=True)
    y_test = split.test["late"].astype(int).to_numpy()

    rows: list[dict] = []
    fitted: dict[str, object] = {}
    probabilities: dict[str, np.ndarray] = {}
    candidates = classification_models(quick, full=full)
    with Progress(len(candidates), "classifiers") as progress:
        for name, model in candidates.items():
            fitted[name] = _fit(model, split.train, numeric, categorical, "late")
            probabilities[name] = fitted[name].predict_proba(
                _inputs(model, split.test, numeric + categorical))[:, 1]
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
    """Test metrics of the fixed ablation probe (the headline change model, smaller).

    Without the propagation group there is no previous delay to add back, so the
    probe falls back to predicting the delay itself (:class:`PersistenceRegressor`
    returns zero when its column is absent).

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
                _fit(ChangeRegressor(_boost(quick, max_iter=120 if quick else 250,
                                            max_leaf_nodes=31, loss="absolute_error",
                                            random_state=seed)),
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
    late_share = (frame["lateness_seconds"] > 600).groupby(
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


def run_start_comparison(test: pd.DataFrame, predictions: dict[str, np.ndarray]) -> list[dict]:
    """MAE at a run's first predictable stop versus every later stop.

    Trains wait at the origin, and the source records no dwell there, so at the
    next stop persistence and the learned models do not know when the train left;
    the run-time baseline does. Reported so that gap is measured, not asserted.
    """
    start = (test["stop_index"] == 1).to_numpy()
    y = test["delay_seconds"].to_numpy()
    return [{"model": name, "run_start_mae": _mae(p[start], y[start]),
             "later_stops_mae": _mae(p[~start], y[~start]), "run_start_share": float(start.mean())}
            for name, p in predictions.items()]


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
            X = test[numeric + categorical]
            model = _fit(_change_boost(quick), train, numeric, categorical, "delay_seconds")
            level = _fit(_boost(quick, loss="absolute_error"), train, numeric, categorical,
                         "delay_seconds")
            predicted, level_pred = model.predict(X), level.predict(X)
            persistence = PersistenceRegressor().fit(train).predict(test)
            run_time = RunTimeRegressor().fit(train, train["delay_seconds"]).predict(test)
            corrected = _fit(RunTimeChangeRegressor(_boost(quick, loss="absolute_error")), train,
                             numeric, categorical, "delay_seconds").predict(test)
            # Arrivals within an hour of the timetable: is the change model's win
            # only on the few rows logged hours off schedule?
            within = np.abs(y) <= 3600
            rows.append({
                "test_start": str(window[0]), "test_end": str(window[-1]),
                "train_days": int(train["service_date"].nunique()), "n_test": len(test),
                "persistence_mae": _mae(persistence, y),
                "run_time_mae": _mae(run_time, y),
                "run_time_model_mae": _mae(corrected, y),
                "model_mae": _mae(predicted, y),
                "level_model_mae": _mae(level_pred, y),
                "persistence_mae_within_1h": _mae(persistence[within], y[within]),
                "model_mae_within_1h": _mae(predicted[within], y[within]),
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
    key = ["service_date", "trip_id", "stop_id"]
    common = None   # reference arrivals scorable at the longest horizon
    with Progress(len(horizons), "prediction horizon") as progress:
        # Compare each horizon with its overlap with the longest-horizon test set.
        # Timing exclusions mean shorter horizons need not contain every reference arrival.
        for k in sorted(horizons, reverse=True):
            frame = features.build(horizon=k, no_cache=True,
                                   max_rows=features.QUICK_MAX_ROWS if quick else None)
            train = features.stratified_sample(frame[frame["service_date"] <= cutoff],
                                               TRAIN_ROWS // 4 if quick else TRAIN_ROWS, SEED)
            test = frame[frame["service_date"] > cutoff]
            numeric, categorical = _columns(train)
            y = test["delay_seconds"].to_numpy()
            row = {"horizon_stops": k, "n_test": len(test),
                   "persistence_mae": _mae(PersistenceRegressor().fit(train).predict(test), y),
                   "run_time_mae": _mae(RunTimeRegressor().fit(train, train["delay_seconds"]).predict(test), y)}
            ids = pd.MultiIndex.from_frame(test[key].astype(str))
            if common is None:
                common = ids
            same = ids.isin(common)
            row["n_common"] = int(same.sum())
            row["persistence_mae_common"] = _mae(
                PersistenceRegressor().fit(train).predict(test[same]), y[same])
            for label, cols in (("own_train_mae", [c for c in numeric if c not in network]),
                                ("with_other_trains_mae", numeric)):
                model = _fit(_change_boost(quick), train, cols, categorical, "delay_seconds")
                predicted = model.predict(test[cols + categorical])
                row[label] = _mae(predicted, y)
                if label == "with_other_trains_mae":
                    row["with_other_trains_mae_common"] = _mae(predicted[same], y[same])
            rows.append(row)
            progress.step(f"{k} stop(s) ahead",
                          extra=f"persistence {row['persistence_mae']:.1f}s  own {row['own_train_mae']:.1f}s  "
                                f"+others {row['with_other_trains_mae']:.1f}s")
    return pd.DataFrame(rows).sort_values("horizon_stops").reset_index(drop=True)


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
        reg_comparison, reg_fitted = run_regression(split, quick=quick, full=full)
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

        # The headline model is fixed in advance, not chosen by test score.
        chosen_name = HEADLINE_REGRESSOR
        chosen = reg_fitted[chosen_name]
        test_predictions = chosen.predict(X_test)
        analysis = error_analysis(split.test, test_predictions)
        analysis["by_run_start"] = run_start_comparison(split.test, {
            name: reg_fitted[name].predict(_inputs(reg_fitted[name], split.test, numeric + categorical))
            for name in ("baseline_persistence", "baseline_run_time", chosen_name,
                         "hist_gradient_boosting_run_time")})

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
         "delay_seconds", "lateness_seconds", "late", "prev_delay_1"]
    ].assign(predicted_delay=test_predictions,
             persistence_delay=reg_fitted["baseline_persistence"].predict(X_test),
             level_predicted_delay=reg_fitted["hist_gradient_boosting_mae"].predict(X_test))
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

    chosen_metrics = reg_comparison.set_index("model").loc[chosen_name].to_dict()
    clf_metrics = clf_comparison.set_index("model").loc[HEADLINE_CLASSIFIER].to_dict()
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
        "best_classification_model": HEADLINE_CLASSIFIER,
        "best_classification_metrics": clf_metrics,
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
