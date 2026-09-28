"""Tests for the clustering stage.

These use synthetic blobs with a known structure, so a failure means the
clustering logic broke rather than the data changing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mbta_ds import cluster_stations


@pytest.fixture
def blobs() -> tuple[np.ndarray, np.ndarray]:
    """Three well-separated 2-D blobs with known labels."""
    rng = np.random.default_rng(506)
    centres = np.array([[0.0, 0.0], [8.0, 8.0], [-8.0, 8.0]])
    points, labels = [], []
    for index, centre in enumerate(centres):
        points.append(centre + rng.normal(0, 0.4, size=(30, 2)))
        labels.extend([index] * 30)
    return np.vstack(points), np.array(labels)


class TestKMeansSweep:
    def test_recovers_the_planted_cluster_count(self, blobs):
        points, _ = blobs
        solution, diagnostics = cluster_stations.kmeans_sweep(points, name="test")
        assert solution.k == 3
        assert solution.silhouette > 0.8
        assert len(diagnostics) == len(list(cluster_stations.K_RANGE))

    def test_diagnostics_report_inertia_decreasing_with_k(self, blobs):
        points, _ = blobs
        _, diagnostics = cluster_stations.kmeans_sweep(points, name="test")
        inertias = [d["inertia"] for d in diagnostics]
        # Inertia must fall monotonically as k grows.
        assert all(a >= b for a, b in zip(inertias, inertias[1:]))

    def test_is_deterministic(self, blobs):
        points, _ = blobs
        first, _ = cluster_stations.kmeans_sweep(points, name="test")
        second, _ = cluster_stations.kmeans_sweep(points, name="test")
        np.testing.assert_array_equal(first.labels, second.labels)

    def test_agrees_with_ward_clustering(self, blobs):
        points, _ = blobs
        solution, _ = cluster_stations.kmeans_sweep(points, name="test")
        comparison = cluster_stations.compare_algorithms(points, solution)
        assert comparison["ward"]["adjusted_rand_vs_kmeans"] > 0.9

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            cluster_stations.kmeans_sweep(np.zeros((1, 2)), name="tiny")


class TestReduceDimensions:
    def test_standardises_and_reduces(self):
        rng = np.random.default_rng(0)
        matrix = rng.normal(0, 1, size=(40, 12)) * np.arange(1, 13)
        reduced, pca = cluster_stations.reduce_dimensions(matrix, name="test")
        assert reduced.shape[0] == 40
        assert reduced.shape[1] <= cluster_stations.MAX_COMPONENTS
        # Standardisation means no single raw feature should dominate component 1.
        assert pca.explained_variance_ratio_.sum() <= 1.0 + 1e-9

    def test_output_is_centred(self):
        rng = np.random.default_rng(1)
        matrix = rng.normal(5, 3, size=(50, 6))
        reduced, _ = cluster_stations.reduce_dimensions(matrix, name="test")
        assert np.allclose(reduced.mean(axis=0), 0, atol=1e-8)


class TestClusterNaming:
    def test_reliability_names_are_ordered_by_mean_delay(self):
        """The label assigned to a cluster must be monotone in its mean delay.

        The label list is sampled evenly across however many clusters were found,
        so the contract is about ordering, not about specific strings.
        """
        profile = pd.DataFrame(
            {"mean_delay": [10.0, 500.0, 200.0]},
            index=["A", "B", "C"],
        )
        cluster_of = np.array([0, 1, 2])  # A -> 0, B -> 1, C -> 2
        names = cluster_stations.name_reliability_clusters(profile, cluster_of)

        # Walk the stations from least to most delayed and read off the label of
        # each one's cluster; those label positions must increase monotonically.
        ranking = profile["mean_delay"].sort_values()
        ordered = [cluster_of[profile.index.get_loc(station)] for station in ranking.index]
        positions = [cluster_stations.RELIABILITY_LABELS.index(names[c]) for c in ordered]

        assert positions == sorted(positions)
        assert positions == [0, 2, 4]
        assert names[0] == cluster_stations.RELIABILITY_LABELS[0]
        assert names[1] == cluster_stations.RELIABILITY_LABELS[-1]

    def test_clusters_ahead_of_the_timetable_are_not_called_reliable(self):
        profile = pd.DataFrame({"mean_delay": [-60.0, -20.0, 150.0]}, index=["A", "B", "C"])
        names = cluster_stations.name_reliability_clusters(profile, np.array([0, 1, 2]))
        assert names[0] == "runs early"
        assert names[1] == "runs early (1)"   # still unique
        assert names[2] == cluster_stations.RELIABILITY_LABELS[-1]

    def test_demand_names_reflect_the_peak_period(self):
        columns = [f"p{i:02d}" for i in range(48)]
        # Cluster 0 peaks in the morning (periods 12-19), cluster 1 in the
        # afternoon (32-39). Band intensity is measured against each band's share
        # of the clock, so a 4-hour band holding 100% of the day's entries scores
        # 1 / (8/48) = 6.
        morning = np.zeros(48)
        morning[12:20] = 1 / 8
        afternoon = np.zeros(48)
        afternoon[32:40] = 1 / 8
        profile = pd.DataFrame([morning, afternoon], columns=columns, index=["M", "A"])

        names = cluster_stations.name_demand_clusters(profile, np.array([0, 1]))
        assert names[0] == "morning-peaked"
        assert names[1] == "afternoon-peaked"

    def test_flat_profile_is_labelled_all_day(self):
        columns = [f"p{i:02d}" for i in range(48)]
        # A perfectly even profile scores exactly 1.0 in every band, which is below
        # the peak threshold, so it must not claim a band.
        flat = np.full(48, 1 / 48)
        morning = np.zeros(48)
        morning[12:20] = 1 / 8
        profile = pd.DataFrame([flat, morning], columns=columns, index=["F", "M"])

        names = cluster_stations.name_demand_clusters(profile, np.array([0, 1]))
        assert names[0].startswith("all-day steady")
        assert names[1] == "morning-peaked"

    def test_evening_heavy_profile_is_labelled(self):
        columns = [f"p{i:02d}" for i in range(48)]
        evening = np.zeros(48)
        evening[40:] = 1 / 8
        flat = np.full(48, 1 / 48)
        profile = pd.DataFrame([evening, flat], columns=columns, index=["E", "F"])

        names = cluster_stations.name_demand_clusters(profile, np.array([0, 1]))
        assert names[0] == "evening-heavy"
        assert names[1].startswith("all-day steady")

    def test_names_are_unique_across_clusters(self):
        """Two clusters must never end up with the same name.

        Names are used as group keys downstream, so a collision would silently
        merge two distinct clusters.
        """
        rng = np.random.default_rng(3)
        profile = pd.DataFrame(
            rng.random((6, 48)), columns=[f"p{i:02d}" for i in range(48)]
        )
        names = cluster_stations.name_demand_clusters(profile, np.arange(6))
        assert len(set(names.values())) == len(names)

    def test_band_shares_sum_to_one_per_cluster(self):
        columns = [f"p{i:02d}" for i in range(48)]
        morning = np.zeros(48)
        morning[12:20] = 1 / 8
        flat = np.full(48, 1 / 48)
        profile = pd.DataFrame([morning, flat], columns=columns)
        shares = cluster_stations.demand_band_shares(profile, np.array([0, 1]))
        assert np.allclose(shares.sum(axis=1), 1.0)

    def test_band_intensity_is_one_for_an_evenly_spread_profile(self):
        columns = [f"p{i:02d}" for i in range(48)]
        flat = pd.DataFrame([np.full(48, 1 / 48)], columns=columns)
        intensity = cluster_stations.demand_band_intensity(
            cluster_stations.demand_band_shares(flat, np.array([0]))
        )
        # Equal travel in every hour must score exactly 1.0 regardless of band
        # length -- this is the property that stops the 6-hour night band from
        # being labelled a peak.
        assert np.allclose(intensity.to_numpy(), 1.0)


class TestReliabilityMatrix:
    def test_pivots_station_by_hour_and_fills_gaps(self):
        frame = pd.DataFrame({
            "station_name": ["A", "A", "B"],
            "scheduled_hour": [8, 9, 8],
            "delay_seconds": [100.0, 300.0, -50.0],
            "late": [False, True, False],
        })
        matrix = cluster_stations.reliability_matrix(frame)
        assert "h08" in matrix.columns
        assert "h09" in matrix.columns
        assert matrix.loc["A", "h08"] == pytest.approx(100.0)
        # Station B has no hour-9 service, so it inherits the network value for
        # that hour rather than being filled with a fake zero delay.
        assert matrix.loc["B", "h09"] == pytest.approx(300.0)
        assert matrix.loc["B", "h08"] == pytest.approx(-50.0)

    def test_reports_summary_statistics(self):
        frame = pd.DataFrame({
            "station_name": ["A"] * 4,
            "scheduled_hour": [8, 8, 9, 9],
            "delay_seconds": [0.0, 0.0, 600.0, 600.0],
            "late": [False, False, True, True],
        })
        matrix = cluster_stations.reliability_matrix(frame)
        assert matrix.loc["A", "on_time_rate"] == pytest.approx(0.5)
        assert matrix.loc["A", "n_observations"] == 4
        assert matrix.loc["A", "mean_delay"] == pytest.approx(300.0)


class TestDemandCoverageFilter:
    """The shape of a near-empty station profile is an artefact, not a pattern.

    Normalising each station profile to sum to 1 means dividing by a near-zero
    total, which turns a handful of faregate taps into an extreme-looking shape.
    """

    def _raw(self) -> pd.DataFrame:
        good_hours = list(range(24)) * 2          # all 48 half-hour periods
        return pd.DataFrame({
            "station_name": ["Good"] * 48 + ["Sparse"] * 3,
            "period_hour": good_hours + [8, 9, 10],
            "period_minute": [0, 30] * 24 + [0, 0, 0],
            "gated_entries": [100.0] * 48 + [1.0, 1.0, 1.0],
            "service_date": ["2026-01-01"] * 51,
        })

    def test_low_coverage_stations_are_excluded(self, monkeypatch):
        monkeypatch.setattr(cluster_stations.collect_ridership, "load_raw", self._raw)
        shape, coverage = cluster_stations.demand_matrix()
        assert "Good" in shape.index
        assert "Sparse" not in shape.index

    def test_coverage_table_records_the_decision(self, monkeypatch):
        monkeypatch.setattr(cluster_stations.collect_ridership, "load_raw", self._raw)
        _, coverage = cluster_stations.demand_matrix()
        assert coverage.loc["Good", "clustered"]
        assert not coverage.loc["Sparse", "clustered"]
        assert coverage.loc["Sparse", "periods_observed"] == 3
        assert coverage.loc["Sparse", "total_entries"] == 3.0

    def test_kept_profiles_are_normalised(self, monkeypatch):
        monkeypatch.setattr(cluster_stations.collect_ridership, "load_raw", self._raw)
        shape, _ = cluster_stations.demand_matrix()
        period_columns = [c for c in shape.columns if c.startswith("p")]
        assert np.allclose(shape[period_columns].sum(axis=1), 1.0)

    def test_stations_are_renamed_by_stop_id_to_match_the_delay_data(self, monkeypatch):
        """"State Street" in ridership is "State" in the delay data."""
        raw = self._raw().assign(stop_id=["place-state"] * 48 + ["place-x"] * 3)
        monkeypatch.setattr(cluster_stations.collect_ridership, "load_raw", lambda: raw)
        names = pd.Series({"place-state": "State"})
        shape, coverage = cluster_stations.demand_matrix(names)
        assert "State" in shape.index
        assert "Good" not in shape.index
        # Stations without a mapping keep their ridership name.
        assert "Sparse" in coverage.index

    def test_interchange_lines_are_added_and_absent_periods_count_as_zero(self, monkeypatch):
        """Park Street's Red and Green halves add up; a silent half-hour is zero."""
        day ={"period_hour": [h for h in range(24) for _ in (0, 1)],
               "period_minute": [0, 30] * 24}
        red = pd.DataFrame({**day, "station_name": "Park", "route_or_line": "Red",
                            "gated_entries": 10.0, "service_date": "2026-01-01"})
        green = red.assign(route_or_line="Green")
        # Day two: only the 08:00 half-hour reported (the rest had no taps).
        quiet = red.iloc[[16]].assign(service_date="2026-01-02")
        raw = pd.concat([red, green, quiet], ignore_index=True)
        monkeypatch.setattr(cluster_stations.collect_ridership, "load_raw", lambda: raw)
        shape, _ = cluster_stations.demand_matrix()
        # Day one: 48 periods x (10 + 10) = 960. Day two: 10. Mean daily = 485.
        assert shape.loc["Park", "total_entries"] == pytest.approx(485.0)

    def test_the_thresholds_cannot_exclude_a_full_day(self):
        # A station observed across the whole day with real volume must pass.
        assert cluster_stations.MIN_PERIODS_OBSERVED <= 48
        assert cluster_stations.MIN_TOTAL_ENTRIES > 0


class TestCrossTrackTest:
    def test_detects_a_strong_association(self):
        joined = pd.DataFrame({
            "reliability_cluster": ["late"] * 10 + ["on time"] * 10,
            "demand_cluster": ["busy"] * 10 + ["quiet"] * 10,
            "mean_delay": [500.0] * 10 + [10.0] * 10,
        })
        result = cluster_stations.cross_track_test(joined)
        assert result["chi2_p_value"] < 0.01
        assert result["anova_p_value"] < 0.01
        assert set(result["mean_delay_by_demand_cluster"]) == {"busy", "quiet"}

    def test_handles_a_single_cluster_without_crashing(self):
        joined = pd.DataFrame({
            "reliability_cluster": ["only"] * 6,
            "demand_cluster": ["only"] * 6,
            "mean_delay": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        result = cluster_stations.cross_track_test(joined)
        assert result["chi2_p_value"] is None
