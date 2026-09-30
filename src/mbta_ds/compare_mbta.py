"""Score our predictions the way the MBTA scores its own countdown clocks.

The MBTA publishes the weekly accuracy of its real-time subway predictions
("Rapid Transit and Bus Prediction Accuracy Data", MBTA open data portal). A
prediction is put in a bin by how far ahead it was made, and counts as accurate
if the train arrives inside that bin's window:

====== ============================
bin    accurate if the train arrives
====== ============================
0-3    60 s early to 60 s late
3-6    90 s early to 120 s late
6-12   150 s early to 210 s late
12-30  240 s early to 360 s late
====== ============================

Our models predict from the moment a train leaves a stop, k stops before the
target (``known_at``). Seconds until arrival are then ``scheduled_epoch +
delay - known_at``, predicted and actual, and the same bins and windows apply.

The comparison is not like for like, and the report says so: the MBTA's clocks
predict in real time, refreshed every few seconds, including for trains still
waiting at a terminal; ours are made once per departure, after the fact, from
cleaned data. Optional stage (``python -m mbta_ds.cli compare-mbta``): it
downloads the MBTA file, so it is not part of ``make all``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config, features
from . import model_delay as md
from .http import download, make_session
from .progress import Progress

log = logging.getLogger(__name__)

URL = ("https://www.arcgis.com/sharing/rest/content/items/"
       "155ab68df00145cabddfb90377201b0e/data")
RAW_PATH = config.RAW_DIR / "mbta_prediction_accuracy.csv"
OUT_PATH = config.PROCESSED_DIR / "mbta_comparison.csv"

#: (label, lower bound s, upper bound s, allowed early s, allowed late s)
BINS = (("0-3 min", 0, 180, 60, 60), ("3-6 min", 180, 360, 90, 120),
        ("6-12 min", 360, 720, 150, 210), ("12-30 min", 720, 1800, 240, 360))
#: Stops ahead to predict. Enough to fill the 12-30 minute bin.
HORIZONS = (1, 2, 3, 4, 6, 8, 10, 12, 15)


def accuracy_by_bin(predicted_s: np.ndarray, actual_s: np.ndarray) -> pd.DataFrame:
    """Share of predictions inside the MBTA's window, per bin of predicted seconds ahead."""
    predicted_s, actual_s = np.asarray(predicted_s, float), np.asarray(actual_s, float)
    late = actual_s - predicted_s
    rows = []
    for label, low, high, early, allowed_late in BINS:
        inside = (predicted_s >= low) & (predicted_s < high)
        ok = inside & (late >= -early) & (late <= allowed_late)
        rows.append({"bin": label, "predictions": int(inside.sum()), "accurate": int(ok.sum())})
    return pd.DataFrame(rows)


def mbta_accuracy(start, end, session=None) -> pd.DataFrame:
    """The MBTA's subway accuracy per bin, over the weeks inside ``start``..``end``.

    Each row of the file is a week labelled by its Friday, covering the seven
    service days before it, so a week counts if all of those days are in range.
    """
    download(URL, RAW_PATH, session or make_session(), skip_if_exists=False)
    data = pd.read_csv(RAW_PATH)
    data = data[data["mode"] == "subway"].assign(weekly=pd.to_datetime(data["weekly"]))
    first_day = data["weekly"] - pd.Timedelta(days=7)
    weeks = data[(first_day >= pd.Timestamp(start)) & (data["weekly"] - pd.Timedelta(days=1)
                                                        <= pd.Timestamp(end))]
    table = (weeks.groupby("bin")[["num_predictions", "num_accurate_predictions"]].sum()
             .rename(columns={"num_predictions": "predictions",
                              "num_accurate_predictions": "accurate"}).reset_index())
    table["weeks"] = weeks["weekly"].nunique()
    return table


def our_accuracy(*, quick: bool) -> tuple[pd.DataFrame, object, object]:
    """Accuracy per bin for the lookup, the headline model and the lookup corrector."""
    split = md.temporal_split(features.load())
    cutoff, test_end = split.cutoff, split.test["service_date"].max()
    del split
    predicted: dict[str, list] = {}
    actual: list = []
    horizons = HORIZONS[:3] if quick else HORIZONS
    with Progress(len(horizons), "MBTA comparison") as progress:
        for k in horizons:
            frame = features.build(horizon=k, no_cache=True,
                                   max_rows=features.QUICK_MAX_ROWS if quick else None)
            train = features.stratified_sample(frame[frame["service_date"] <= cutoff],
                                               md.TRAIN_ROWS // 4 if quick else md.TRAIN_ROWS, md.SEED)
            test = frame[(frame["service_date"] > cutoff) & frame["known_at"].notna()]
            numeric, categorical = md._columns(train)
            y = test["delay_seconds"].to_numpy(dtype=float)
            ahead = test["scheduled_epoch"].to_numpy(dtype=float) - test["known_at"].to_numpy(dtype=float)
            actual.append(ahead + y)
            candidates = {"lookup": md.RunTimeRegressor(),
                          "headline model": md._change_boost(quick),
                          "lookup + correction": md.RunTimeChangeRegressor(
                              md._boost(quick, loss="absolute_error"))}
            for name, model in candidates.items():
                fitted = md._fit(model, train, numeric, categorical, "delay_seconds")
                p = fitted.predict(md._inputs(fitted, test, numeric + categorical))
                predicted.setdefault(name, []).append(ahead + p)
            progress.step(f"{k} stop(s) ahead", extra=f"{len(test):,} test arrivals")
    actual_s = np.concatenate(actual)
    tables = [accuracy_by_bin(np.concatenate(p), actual_s).assign(source=name)
              for name, p in predicted.items()]
    return pd.concat(tables, ignore_index=True), cutoff, test_end


def run(*, quick: bool = False) -> dict:
    """Both sides over the same test period, with overall accuracy per source."""
    config.ensure_dirs()
    ours, cutoff, test_end = our_accuracy(quick=quick)
    start = pd.Timestamp(str(cutoff)) + pd.Timedelta(days=1)
    theirs = mbta_accuracy(start, pd.Timestamp(str(test_end)))
    if theirs.empty:
        log.warning("the MBTA file has no complete week inside %s..%s yet", start.date(), test_end)
    table = pd.concat([theirs.drop(columns="weeks").assign(source="MBTA countdown clocks"), ours],
                      ignore_index=True)
    table["accuracy"] = table["accurate"] / table["predictions"].where(table["predictions"] > 0)
    table.to_csv(OUT_PATH, index=False)
    overall = table.groupby("source")[["accurate", "predictions"]].sum()
    return {"test_period": [str(start.date()), str(test_end)],
            "mbta_weeks": int(theirs["weeks"].max()) if len(theirs) else 0,
            "overall_accuracy": (overall["accurate"] / overall["predictions"]).round(3).to_dict(),
            "path": str(OUT_PATH)}
