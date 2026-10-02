"""Tests for the figure helpers that shape data before it is plotted."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mbta_ds import viz


class TestDownsampleCurve:
    def test_short_curves_are_untouched(self):
        x, y = viz._downsample_curve([0, 0.5, 1], [0, 0.7, 1], limit=10)
        assert list(x) == [0, 0.5, 1]
        assert list(y) == [0, 0.7, 1]

    def test_thinned_curve_still_ends_at_its_last_point(self):
        """A thinned ROC curve must still reach (1, 1)."""
        x = np.linspace(0, 1, 1001)
        thin_x, thin_y = viz._downsample_curve(x, x, limit=400)
        assert len(thin_x) <= 401  # the limit, plus the kept endpoint
        assert thin_x[0] == 0 and thin_x[-1] == 1
        assert thin_y[-1] == 1


class TestPeriodColumns:
    def test_only_half_hour_share_columns_are_selected(self):
        frame = pd.DataFrame(columns=["p00", "p47", "pca_1", "label", "p1", "total_entries"])
        assert viz._period_columns(frame) == ["p00", "p47"]


class TestMarkerSizes:
    def test_a_run_without_ridership_gets_one_finite_size(self):
        """The holdout run has no entries column; its map used to fail on NaN sizes."""
        sizes = viz._marker_sizes(None, 4)
        assert sizes.shape == (4,) and np.isfinite(sizes).all()
        assert len(set(sizes)) == 1

    def test_ungated_stations_get_the_smallest_marker(self):
        sizes = viz._marker_sizes(pd.Series([np.nan, 100.0, 10_000.0]), 3)
        assert np.isfinite(sizes).all()
        assert sizes[0] < sizes[1] < sizes[2]


def test_skip_reasons_do_not_name_the_local_checkout():
    exc = FileNotFoundError(f"{viz.config.ROOT / 'data' / 'x.parquet'} missing")
    assert str(viz.config.ROOT) not in viz._reason(exc)
