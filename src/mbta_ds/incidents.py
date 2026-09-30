"""Minute-level incident features from the MBTA alerts archive, and a test of them.

The feature table already has two alert columns (``route_alerts_active``,
``route_alert_severity_max``), counted per clock hour and only from the first
full hour after an alert was raised: an alert raised at 08:05 is invisible until
09:00, by which time most subway delays are over. This module asks whether alerts
help when used at the moment of prediction, with what they say:

``incident_active``         service alerts in force on the line
``incident_minutes_since``  minutes since the newest of them was raised
``incident_delay_minutes``  the delay it announces ("Delays of about 15 minutes")
``incident_suspension``     service suspended, reduced or replaced by shuttles
``incident_technical``      caused by a technical problem (signals, a disabled train)
``incident_medical``        a medical emergency
``incident_police``         police activity
``incident_at_station``     the alert names the station the train is heading to

**Nothing from the future.** Alerts are edited while they run ("about 20
minutes" becomes "about 30"). The archive keeps each version with the time it
was made (``last_modified_datetime``), so a prediction at time *t* sees each
alert as it stood at *t*: from the version's own timestamp until the next
version, the alert's close, or its active period's end, whichever is first.

``run`` is an optional experiment (``python -m mbta_ds.cli incidents``): it
trains the headline model and the lookup corrector with and without these
columns and scores them overall and while an alert is in force; then does the
same for the 10+ minute early warning, where sudden disruptions are the target.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier

from . import collect_lamp, config, features
from . import model_delay as md
from . import model_tail as mt

log = logging.getLogger(__name__)

OUT_PATH = config.PROCESSED_DIR / "incident_experiment.csv"
WARNING_PATH = config.PROCESSED_DIR / "incident_warning.csv"

COLUMNS = ("incident_active", "incident_minutes_since", "incident_delay_minutes",
           "incident_suspension", "incident_technical", "incident_medical",
           "incident_police", "incident_at_station")

_ALERT_COLUMNS = ["id", "cause", "effect", "header_text.translation.text", "created_datetime",
                  "closed_datetime", "last_modified_datetime", "active_period.end_datetime",
                  "informed_entity.route_id", "informed_entity.stop_id"]
#: Effects that take trains out of service, rather than slowing them.
SUSPENSION_EFFECTS = ("NO_SERVICE", "REDUCED_SERVICE", "DETOUR", "SUSPENSION", "SHUTTLE")
#: An alert version with no recorded end is assumed over after this long.
MAX_VERSION_HOURS = 6
_DELAY = re.compile(r"[Dd]elays? of (?:about |up to |approximately |over |around )?(\d+)\s*min")


def announced_delay(text) -> float:
    """Minutes of delay an alert's headline announces, 0 when it names none."""
    match = _DELAY.search(text) if isinstance(text, str) else None
    return float(match.group(1)) if match else 0.0


def versions(alerts: pd.DataFrame, stop_parent: dict[str, str]) -> pd.DataFrame:
    """One row per alert version and line: when it was known, until when, and what it said.

    Times stay local and zone-less, as in the archive.
    """
    alerts = alerts[alerts["informed_entity.route_id"].isin(config.SUBWAY_ROUTES)
                    & ~alerts["effect"].isin(features.NON_SERVICE_ALERT_EFFECTS)].copy()
    if alerts.empty:
        return pd.DataFrame(columns=["route_id", "start", "end", "created", "delay", "suspension",
                                     "cause", "stations"])
    for col in ("created_datetime", "closed_datetime", "last_modified_datetime", "active_period.end_datetime"):
        alerts[col] = pd.to_datetime(alerts[col], errors="coerce")
    closed = alerts.groupby("id")["closed_datetime"].min()
    alerts["parent"] = alerts["informed_entity.stop_id"].map(
        lambda s: None if pd.isna(s) else stop_parent.get(str(s), str(s)))

    keys = ["id", "last_modified_datetime", "informed_entity.route_id"]
    table = (alerts.dropna(subset=["last_modified_datetime"])
             .groupby(keys, sort=False)
             .agg(created=("created_datetime", "min"), cause=("cause", "last"), effect=("effect", "last"),
                  text=("header_text.translation.text", "last"), active_end=("active_period.end_datetime", "max"),
                  closed=("closed_datetime", "max"),
                  stations=("parent", lambda s: frozenset(s.dropna())))
             .reset_index()
             .rename(columns={"last_modified_datetime": "start", "informed_entity.route_id": "route_id"}))
    # The row that closes an alert is its end, not a version in force.
    table = table[table["closed"].isna()].sort_values(["id", "route_id", "start"])
    following = table.groupby(["id", "route_id"])["start"].shift(-1)
    cap = table["start"] + pd.Timedelta(hours=MAX_VERSION_HOURS)
    end = pd.concat([following, table["id"].map(closed), table["active_end"], cap], axis=1).min(axis=1)
    table = table.assign(end=end, delay=table["text"].map(announced_delay),
                         suspension=table["effect"].isin(SUSPENSION_EFFECTS))
    table["created"] = table["created"].fillna(table["start"])
    return table[table["end"] > table["start"]][
        ["route_id", "start", "end", "created", "delay", "suspension", "cause", "stations"]
    ].reset_index(drop=True)


def _local_minutes(epoch: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    local = pd.to_datetime(epoch, unit="s", utc=True).dt.tz_convert(config.SERVICE_TZ).dt.tz_localize(None)
    return ((local - origin) // pd.Timedelta(minutes=1)).to_numpy(dtype="float64")


def add_features(frame: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Add :data:`COLUMNS`, read at each row's prediction moment (``known_at``)."""
    out = frame.copy()
    minute = pd.Series(np.nan, index=out.index)
    ok = out["known_at"].notna()
    origin = (pd.to_datetime(out["service_date"].astype("int64").astype(str), format="%Y%m%d").min()
              - pd.Timedelta(days=1))
    minute[ok] = _local_minutes(out.loc[ok, "known_at"], origin)
    n = int(np.nanmax(minute.to_numpy())) + 2 if ok.any() else 1
    for col in COLUMNS:
        out[col] = 0.0
    out["incident_minutes_since"] = np.nan
    if table.empty or not ok.any():
        return out

    to_min = lambda t: ((t - origin) // pd.Timedelta(minutes=1)).to_numpy()
    table = table.assign(s=np.clip(to_min(table["start"]), 0, n), e=np.clip(to_min(table["end"]), 0, n),
                         c=to_min(table["created"]))
    table = table[table["e"] > table["s"]]
    route = out["route_id"].astype(str).to_numpy()
    at = minute.fillna(-1).to_numpy().astype(int)
    station = out["parent_station"].astype(str).to_numpy()
    for line, rows in table.groupby("route_id"):
        grids = {name: np.zeros(n) for name in ("active", "delay", "suspension", "technical", "medical", "police")}
        newest = np.full(n, -np.inf)
        named: dict[str, np.ndarray] = {}
        for r in rows.itertuples(index=False):
            span = slice(int(r.s), int(r.e))
            grids["active"][span] += 1
            grids["delay"][span] = np.maximum(grids["delay"][span], r.delay)
            grids["suspension"][span] = np.maximum(grids["suspension"][span], float(r.suspension))
            for flag, cause in (("technical", "TECHNICAL_PROBLEM"), ("medical", "MEDICAL_EMERGENCY"),
                                ("police", "POLICE_ACTIVITY")):
                if r.cause == cause:
                    grids[flag][span] = 1.0
            newest[span] = np.maximum(newest[span], r.c)
            for s in r.stations:
                named.setdefault(s, np.zeros(n, bool))[span] = True
        mine = (route == line) & (at >= 0)
        idx = at[mine]
        for name, grid in grids.items():
            out.loc[mine, "incident_delay_minutes" if name == "delay" else f"incident_{name}"] = grid[idx]
        since = idx - newest[idx]
        out.loc[mine, "incident_minutes_since"] = np.where(np.isfinite(since), since, np.nan)
        hit = np.array([named[s][i] if s in named else False for s, i in zip(station[mine], idx)])
        out.loc[mine, "incident_at_station"] = hit.astype(float)
    return out


def _score(test: pd.DataFrame, predicted: np.ndarray) -> dict:
    y = test["delay_seconds"].to_numpy()
    during = (test["incident_active"] > 0).to_numpy()
    err = np.abs(predicted - y)
    return {"mae": float(err.mean()), "mae_during_incidents": float(err[during].mean()),
            "mae_otherwise": float(err[~during].mean()), "share_during_incidents": float(during.mean())}


def warning_test(k: int, table: pd.DataFrame, *, quick: bool) -> list[dict]:
    """The 10+ minute early warning k stops ahead, with and without incident columns.

    Scored on every arrival and on onsets (trains under 5 minutes late now), as in
    :mod:`model_tail`, averaging the same seeds.
    """
    frame = add_features(features.build(horizon=k, no_cache=True,
                                        max_rows=features.QUICK_MAX_ROWS if quick else None), table)
    frame["big_delay"] = (frame["lateness_seconds"] > mt.BIG_DELAY_SECONDS).astype(int)
    split = md.temporal_split(frame)
    train = features.stratified_sample(split.train, md.TRAIN_ROWS // 4 if quick else md.TRAIN_ROWS, md.SEED)
    test = split.test
    numeric, categorical = md._columns(train, lateness=True)
    base = [c for c in numeric if c not in COLUMNS]
    y = test["big_delay"].to_numpy()
    onset = (test["prev_lateness_1"] < mt.ONSET_BELOW_SECONDS).to_numpy()
    during = (test["incident_active"] > 0).to_numpy()
    rows = []
    for label, cols in (("without incidents", base), ("with incidents", base + list(COLUMNS))):
        proba = np.mean([md._fit(HistGradientBoostingClassifier(
            categorical_features="from_dtype", max_iter=150 if quick else 300, learning_rate=0.08,
            early_stopping=True, random_state=seed), train, cols, categorical, "big_delay")
            .predict_proba(test[cols + categorical])[:, 1]
            for seed in (md.PROBE_SEEDS[:1] if quick else md.PROBE_SEEDS)], axis=0)
        for subset, mask in (("all arrivals", np.ones_like(onset)), ("onsets", onset),
                             ("onsets during incidents", onset & during)):
            rows.append({"horizon_stops": k, "features": label, "subset": subset,
                         **mt.warning_metrics(y[mask], proba[mask])})
    return rows


def run(*, quick: bool = False) -> dict:
    """Train with and without the incident columns; score overall and during incidents."""
    config.ensure_dirs()
    stops = pd.read_parquet(config.PROCESSED_DIR / "trips_clean.parquet", columns=["stop_id", "parent_station"])
    table = versions(collect_lamp.load_alerts(columns=_ALERT_COLUMNS),
                     dict(zip(stops["stop_id"].astype(str), stops["parent_station"].astype(str))))
    frame = add_features(features.load(), table)
    split = md.temporal_split(frame)
    split.train = features.stratified_sample(split.train, md.TRAIN_ROWS // 4 if quick else md.TRAIN_ROWS, md.SEED)
    numeric, categorical = md._columns(split.train)
    base = [c for c in numeric if c not in COLUMNS]
    rows = []
    for label, cols in (("without incidents", base), ("with incidents", base + list(COLUMNS))):
        for name, model in (("headline model", md._change_boost(quick)),
                            ("lookup + correction", md.RunTimeChangeRegressor(md._boost(quick, loss="absolute_error")))):
            fitted = md._fit(model, split.train, cols, categorical, "delay_seconds")
            predicted = fitted.predict(md._inputs(fitted, split.test, cols + categorical))
            rows.append({"model": name, "features": label, **_score(split.test, predicted)})
            log.info("%s, %s: %s", name, label, {k: round(v, 2) for k, v in rows[-1].items() if isinstance(v, float)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT_PATH, index=False)
    del frame, split
    warning = pd.DataFrame([row for k in (1, 5) for row in warning_test(k, table, quick=quick)])
    warning.to_csv(WARNING_PATH, index=False)
    return {"path": str(OUT_PATH), "rows": result.to_dict("records"),
            "warning": warning.to_dict("records")}
