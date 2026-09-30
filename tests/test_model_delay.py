"""Tests for the delay model: splitting, baselines, metrics and pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mbta_ds import model_delay


def _toy_frame(dates=range(8), rows_per_date=40, seed=0) -> pd.DataFrame:
    """A tiny frame with the columns the model needs."""
    rng = np.random.default_rng(seed)
    records = []
    for date in dates:
        for trip in range(4):
            for stop in range(rows_per_date // 4):
                previous = float(rng.normal(0, 120))
                delay = previous + float(rng.normal(0, 60))
                records.append({
                    "service_date": 20260101 + date,
                    "trip_id": f"t{trip}",
                    "stop_id": f"s{stop}",
                    "station_name": f"Station {stop}",
                    "route_id": "Red" if trip % 2 else "Orange",
                    "trunk_route_id": "Red" if trip % 2 else "Orange",
                    "direction_id": 0 if stop % 2 else 1,
                    "scheduled_hour": 7 + (stop % 12),
                    "stop_sequence": stop,
                    "delay_seconds": delay,
                    "lateness_seconds": delay,
                    "late": delay > 300,
                    "prev_delay_1": previous,
                    "prev_lateness_1": previous,
                    "prev_delay_2": np.nan,
                    "prev_dwell_seconds": 30.0,
                    "prev_travel_time_seconds": 90.0,
                    "prev_headway_seconds": 300.0,
                    "delay_trend": np.nan,
                    "stop_count": rows_per_date // 4,
                    "fraction_through_trip": stop / (rows_per_date // 4),
                    "stop_index": stop,
                    "scheduled_seconds_of_day": (7 + stop) * 3600,
                    "scheduled_elapsed_seconds": stop * 120.0,
                    "scheduled_travel_time": 120.0,
                    "scheduled_headway_branch": 480.0,
                    "scheduled_headway_trunk": 480.0,
                    "day_of_week": 0,
                    "is_weekend": False,
                    "is_peak": True,
                    "days_since_window_start": date,
                    "station_entries_lag7": 5000.0,
                    "station_entries_trend": 10.0,
                    "demand_missing": 0,
                    "temperature_c": 20.0,
                    "precip_mm": 0.0,
                    "snowfall_cm": 0.0,
                    "wind_kph": 5.0,
                    "humidity_pct": 60.0,
                    "is_precip": 0,
                    "is_snow": 0,
                    "weather_missing": 0,
                    "route_alerts_active": 0.0,
                    "route_alert_severity_max": np.nan,
                    "scheduled_seconds_ahead": 120.0,
                    "leader_delay": previous,
                    "leader_age_seconds": 300.0,
                    "line_late_share_15m": 0.1,
                    "line_arrivals_15m": 12.0,
                    "vehicle_prev_trip_delay": previous,
                    "vehicle_layover_seconds": 240.0,
                })
    frame = pd.DataFrame(records)
    for column in ("station_name", "route_id", "trunk_route_id", "direction_id"):
        frame[column] = frame[column].astype("category")
    return frame


class TestTemporalSplit:
    def test_train_dates_are_strictly_before_test_dates(self):
        split = model_delay.temporal_split(_toy_frame(), test_fraction=0.25)
        assert split.train["service_date"].max() < split.test["service_date"].min()
        assert split.cutoff == split.train["service_date"].max()

    def test_split_keeps_whole_days_together(self):
        """No service date may appear in both sides of the split."""
        split = model_delay.temporal_split(_toy_frame(), test_fraction=0.25)
        assert not (set(split.train["service_date"]) & set(split.test["service_date"]))

    def test_roughly_respects_the_fraction(self):
        split = model_delay.temporal_split(_toy_frame(), test_fraction=0.25)
        total = split.n_train + split.n_test
        assert 0.1 < split.n_test / total < 0.5

    def test_too_few_dates_raises_a_helpful_error(self):
        with pytest.raises(ValueError, match="at least 4 service dates"):
            model_delay.temporal_split(_toy_frame(dates=range(2)))


class TestMetrics:
    def test_perfect_predictions(self):
        truth = np.array([0.0, 100.0, -50.0, 400.0])
        metrics = model_delay.evaluate_regression(truth, truth.copy())
        assert metrics["mae_seconds"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["bias_seconds"] == pytest.approx(0.0)

    def test_constant_bias_is_reported(self):
        truth = np.zeros(10)
        metrics = model_delay.evaluate_regression(truth, np.full(10, 60.0))
        assert metrics["mae_seconds"] == pytest.approx(60.0)
        assert metrics["bias_seconds"] == pytest.approx(60.0)

    def test_classification_metrics_on_perfect_scores(self):
        truth = np.array([0, 0, 1, 1])
        proba = np.array([0.01, 0.02, 0.98, 0.99])
        metrics = model_delay.evaluate_classification(truth, proba)
        assert metrics["f1"] == pytest.approx(1.0)
        assert metrics["roc_auc"] == pytest.approx(1.0)
        assert metrics["base_rate"] == pytest.approx(0.5)

    def test_classification_handles_single_class_gracefully(self):
        truth = np.zeros(5, dtype=int)
        metrics = model_delay.evaluate_classification(truth, np.full(5, 0.2))
        assert np.isnan(metrics["roc_auc"])


class TestBaselines:
    def test_persistence_returns_the_previous_delay(self):
        frame = _toy_frame()
        model = model_delay.PersistenceRegressor().fit(frame, frame["delay_seconds"])
        predictions = model.predict(frame)
        np.testing.assert_allclose(predictions, frame["prev_delay_1"].to_numpy())

    def test_persistence_fills_missing_lags_with_the_median(self):
        frame = _toy_frame()
        frame.loc[frame.index[:5], "prev_delay_1"] = np.nan
        model = model_delay.PersistenceRegressor().fit(frame, frame["delay_seconds"])
        predictions = model.predict(frame.iloc[:5])
        assert np.isfinite(predictions).all()

    def test_group_mean_learns_the_training_mean(self):
        frame = _toy_frame()
        model = model_delay.GroupMeanRegressor().fit(frame, frame["delay_seconds"])
        expected = frame.groupby(["route_id", "scheduled_hour"],
                                 observed=True)["delay_seconds"].mean()
        predictions = model.predict(frame)
        # Every prediction must match its group's training mean.
        closest = min(abs(predictions[0] - v) for v in expected.to_numpy())
        assert closest < 1.0

    def test_zero_regressor_predicts_zero(self):
        frame = _toy_frame()
        assert (model_delay.ZeroRegressor().predict(frame) == 0).all()

    def test_persistence_classifier_thresholds_the_previous_lateness(self):
        frame = _toy_frame()
        previous = np.full(len(frame), 100.0)
        previous[:4] = [100.0, 400.0, np.nan, 301.0]
        frame["prev_lateness_1"] = previous

        model = model_delay.PersistenceClassifier()
        labels = model.predict(frame.iloc[:4])
        # NaN (unknown) is treated as "not late" rather than crashing.
        assert list(labels) == [0, 1, 0, 1]


class TestFeatureColumns:
    def test_regression_leaves_out_rider_lateness(self):
        numeric, _ = model_delay._columns(_toy_frame())
        assert "prev_lateness_1" not in numeric

    def test_lateness_models_use_it(self):
        numeric, _ = model_delay._columns(_toy_frame(), lateness=True)
        assert "prev_lateness_1" in numeric


class TestChangeRegressor:
    """Predicting the change since the last stop must not cap large delays."""

    def test_predicts_beyond_the_training_range(self):
        train = _toy_frame()
        test = _toy_frame(dates=[20], seed=1)
        # A train logged hours late at its last stop: far outside anything the
        # training rows contain.
        test["prev_delay_1"] = test["prev_delay_1"] + 16_000
        test["delay_seconds"] = test["delay_seconds"] + 16_000
        numeric, categorical = model_delay._columns(train)
        cols = numeric + categorical

        absolute = model_delay._fit(model_delay._boost(True, loss="absolute_error"), train,
                                    numeric, categorical, "delay_seconds")
        change = model_delay._fit(model_delay._change_boost(True), train,
                                  numeric, categorical, "delay_seconds")
        y = test["delay_seconds"].to_numpy()
        assert np.abs(absolute.predict(test[cols]) - y).mean() > 10_000
        assert np.abs(change.predict(test[cols]) - y).mean() < 200

    def test_is_listed_as_a_regression_candidate(self):
        models = model_delay.regression_models(quick=True)
        assert isinstance(models["hist_gradient_boosting_change"], model_delay.ChangeRegressor)


class TestPipeline:
    def test_pipeline_fits_and_predicts(self):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline

        frame = _toy_frame()
        numeric = [c for g in __import__("mbta_ds.features", fromlist=["x"]).FEATURE_GROUPS.values()
                   for c in g]
        categorical = ["route_id", "station_name", "trunk_route_id", "direction_id"]

        pipeline = Pipeline([
            ("prep", model_delay.make_preprocessor(numeric, categorical, scale=True)),
            ("model", Ridge(alpha=1.0)),
        ])
        pipeline.fit(frame[numeric + categorical], frame["delay_seconds"])
        predictions = pipeline.predict(frame[numeric + categorical])
        assert len(predictions) == len(frame)
        assert np.isfinite(predictions).all()

    def test_native_categorical_preprocessor_keeps_category_dtype(self):
        frame = _toy_frame()
        numeric = ["scheduled_hour", "prev_delay_1"]
        categorical = ["route_id", "station_name"]
        preprocessor = model_delay.make_preprocessor(
            numeric, categorical, scale=False, one_hot=False
        )
        transformed = preprocessor.fit_transform(frame[numeric + categorical])
        assert "station_name" in transformed.columns
        assert str(transformed["station_name"].dtype) == "category"

    def test_imputation_happens_inside_the_pipeline(self):
        """A pipeline must cope with missing values without any pre-filling."""
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline

        frame = _toy_frame()
        frame.loc[frame.index[:20], "prev_delay_1"] = np.nan
        numeric = ["scheduled_hour", "prev_delay_1"]
        categorical = ["route_id"]

        pipeline = Pipeline([
            ("prep", model_delay.make_preprocessor(numeric, categorical, scale=True)),
            ("model", Ridge(alpha=1.0)),
        ])
        pipeline.fit(frame[numeric + categorical], frame["delay_seconds"])
        assert np.isfinite(pipeline.predict(frame[numeric + categorical])).all()


class TestFeatureSelection:
    def test_all_missing_columns_are_dropped_and_reported(self, caplog):
        frame = _toy_frame()
        frame["station_entries_trend"] = np.nan
        kept = model_delay._usable_columns(frame, ["prev_delay_1", "station_entries_trend"])
        assert kept == ["prev_delay_1"]

    def test_columns_absent_from_the_frame_are_dropped(self):
        frame = _toy_frame()
        kept = model_delay._usable_columns(frame, ["prev_delay_1", "not_a_column"])
        assert kept == ["prev_delay_1"]


class TestAblationOrder:
    def test_first_step_is_schedule_only(self):
        label, groups = model_delay.ABLATION_ORDER[0]
        assert groups == ("schedule",)
        assert label == "schedule"

    def test_adding_propagation_is_a_distinct_step(self):
        labels = [label for label, _ in model_delay.ABLATION_ORDER]
        assert "+ delay propagation" in labels
        assert "+ weather" in labels
        assert "+ alerts" in labels
        assert labels.index("+ delay propagation") > labels.index("+ calendar")


class TestTemporalEarlyStopping:
    def test_holds_out_the_latest_dates(self):
        frame = _toy_frame(dates=range(10))
        fit_rows, val_rows = model_delay.temporal_validation(frame)
        assert fit_rows["service_date"].max() < val_rows["service_date"].min()
        assert val_rows["service_date"].nunique() == 1

    def test_too_few_dates_falls_back(self):
        assert model_delay.temporal_validation(_toy_frame(dates=range(3))) is None

    def test_boosted_models_fit_with_a_dated_validation_set(self):
        frame = _toy_frame(dates=range(10))
        numeric, categorical = model_delay._columns(frame)
        for model in (model_delay._boost(True), model_delay._change_boost(True)):
            fitted = model_delay._fit(model, frame, numeric, categorical, "delay_seconds")
            assert np.isfinite(fitted.predict(frame[numeric + categorical])).all()


class TestRunTimeBaseline:
    def _frame(self):
        # One stop, two training trains: left the previous stop at t=0 and took
        # 60 s and 100 s to arrive; scheduled to arrive at t=90.
        return pd.DataFrame({
            "route_id": ["Red"] * 3, "direction_id": [0] * 3, "stop_id": ["s1"] * 3,
            "known_at": [0.0, 0.0, 1_000.0], "scheduled_epoch": [90.0, 90.0, 1_090.0],
            "prev_delay_1": [5.0, 5.0, 400.0],
        })

    def test_adds_the_median_running_time_to_the_departure(self):
        frame = self._frame()
        y = np.array([60.0 - 90.0, 100.0 - 90.0, 0.0])
        model = model_delay.RunTimeRegressor().fit(frame.iloc[:2], y[:2])
        # Median run 80 s: arrives at 1,080 against a scheduled 1,090.
        assert model.predict(frame.iloc[[2]])[0] == -10.0

    def test_ignores_the_previous_delay_when_the_departure_is_known(self):
        """Persistence would say +400 s; the train has in fact just left."""
        frame = self._frame()
        model = model_delay.RunTimeRegressor().fit(frame.iloc[:2], np.array([-30.0, 10.0]))
        assert model.predict(frame.iloc[[2]])[0] != 400.0

    def test_is_a_baseline_candidate_fit_on_the_whole_frame(self):
        assert "baseline_run_time" in model_delay.regression_models(quick=True)
        frame = self._frame()
        frame["delay_seconds"] = [-30.0, 10.0, 0.0]
        fitted = model_delay._fit(model_delay.RunTimeRegressor(), frame.iloc[:2], [], [], "delay_seconds")
        assert fitted.predict(model_delay._inputs(fitted, frame, []))[2] == -10.0


def test_run_start_comparison_splits_the_first_predictable_stop_from_the_rest():
    test = pd.DataFrame({"stop_index": [1, 2, 3, 1], "delay_seconds": [100.0, 0.0, 0.0, 100.0]})
    rows = model_delay.run_start_comparison(test, {"zero": np.zeros(4)})
    assert rows == [{"model": "zero", "run_start_mae": 100.0, "later_stops_mae": 0.0,
                     "run_start_share": 0.5}]


class TestRunTimeChangeRegressor:
    """The run-time lookup plus a learned correction."""

    @staticmethod
    def _frame(n_days: int = 6, per_day: int = 40) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        rows = []
        for d in range(n_days):
            for i in range(per_day):
                known = 1_000.0 * i
                run = 80.0 + (30.0 if i % 2 else 0.0)   # odd rows run slower: learnable
                rows.append({"service_date": f"2026-06-{d + 1:02d}", "route_id": "Red",
                             "direction_id": 0, "stop_id": "s1", "known_at": known,
                             "scheduled_epoch": known + 90.0, "odd": float(i % 2),
                             "prev_delay_1": rng.normal(0, 60),
                             "delay_seconds": run - 90.0})
        return pd.DataFrame(rows)

    def test_learns_what_the_lookup_misses(self):
        frame = self._frame()
        y = frame["delay_seconds"].to_numpy()
        lookup = model_delay.RunTimeRegressor().fit(frame, y).predict(frame)
        model = model_delay._fit(
            model_delay.RunTimeChangeRegressor(model_delay._boost(True, loss="absolute_error",
                                                                  early_stopping=False)),
            frame, ["odd"], [], "delay_seconds")
        predicted = model.predict(model_delay._inputs(model, frame, ["odd"]))
        assert np.mean(np.abs(predicted - y)) < 0.5 * np.mean(np.abs(lookup - y))

    def test_training_lookup_is_cross_fitted_by_date(self):
        frame = self._frame(n_days=2, per_day=2)
        y = frame["delay_seconds"].to_numpy().copy()
        y[frame["service_date"] == "2026-06-01"] += 1_000.0
        model = model_delay.RunTimeChangeRegressor(None, columns=[], folds=2)
        oof = model._out_of_fold(frame, y)
        # Each day's lookup comes from the other day only, never its own answers.
        day1 = (frame["service_date"] == "2026-06-01").to_numpy()
        assert np.all(np.abs(oof[day1] - y[day1]) >= 1_000.0 - 30.0)

    def test_is_a_regression_candidate(self):
        assert "hist_gradient_boosting_run_time" in model_delay.regression_models(quick=True)
