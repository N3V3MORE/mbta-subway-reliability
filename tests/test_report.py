"""Tests for the report's table rendering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mbta_ds import report


def test_tables_escape_text_and_round_numbers():
    frame = pd.DataFrame({"Station": ["<Park & Street>"], "MAE (s)": [24.3571], "n": [3]})
    rendered = report._table(frame, digits=1)
    assert "&lt;Park &amp; Street&gt;" in rendered
    assert "<td>24.4</td>" in rendered
    assert "<td>3</td>" in rendered


def test_missing_values_render_blank_not_as_none():
    rendered = report._table(pd.DataFrame({"x": [np.nan], "y": [None]}))
    assert rendered.count("<td></td>") == 2
    assert "None" not in rendered


def test_counts_get_separators_and_scores_three_decimals():
    rendered = report._table(pd.DataFrame({"Arrivals": [872054], "R²": [0.9553]}))
    assert "<td>872,054</td>" in rendered
    assert "<td>0.955</td>" in rendered


def test_service_dates_are_readable():
    assert report._date(20260607) == "2026-06-07"


def test_model_ids_get_readable_names():
    frame = pd.DataFrame({"model": ["baseline_persistence", "something_new"]})
    labelled = report._label_models(frame)["model"].tolist()
    assert labelled == ["Baseline: previous stop's delay", "something_new"]
    relabelled = report._label_models(pd.DataFrame({"model": ["hist_gradient_boosting"]}),
                                      {"hist_gradient_boosting": "Gradient boosting"})
    assert relabelled["model"].tolist() == ["Gradient boosting"]
