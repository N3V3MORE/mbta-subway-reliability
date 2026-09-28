# MBTA Subway Reliability & Ridership Analytics

**CS 506 (Data Science Tools and Applications) — final project**

Predicting when the T runs late, and what kind of stations the network is made of.
This repository implements the full data-science lifecycle — collection, cleaning,
feature extraction, visualisation and modelling — on real MBTA open data.

Two tracks share one dataset builder:

* **Track A — arrival delay.** Predict subway arrival delay in seconds, and
  classify whether a train will be more than 5 minutes late. Then cluster stations
  by how reliably they are served.
* **Track B — demand.** Cluster stations by the *shape* of their ridership through
  the day, and test whether reliability and demand type are related.

**Headline result.** Histogram gradient boosting trained on absolute error predicts
arrival delay with a mean absolute error of **23.2 seconds**, a **54% improvement**
on the strongest baseline (delay persistence, 50.1 s), steady at 48–56% across
four separate test periods. Lateness classification
reaches **F1 0.964 / ROC-AUC 0.997**. Ten stops ahead, the model is still 27%
better than persistence.

**Headline finding.** Almost all of the predictive power comes from one feature:
the train's delay at its previous stop. Delays are *sticky*: a train that is five
minutes late at one stop is almost exactly five minutes late at the next, on every
line. Everything else — weather, mode, service alerts, the schedule itself — is
worth an order of magnitude less. The project also documents a feature that
**raised test error by 70%**, four defects in the source data that had been
quietly shaping earlier results (§5), and a ridership calculation that halved the
network's busiest stations until it was caught (§8).

---

## 1. How to build and run the code

Requires **Python 3.11 or newer**. Nothing else — no API key, no manual downloads.
Every data source is either public or keyless by default.

```bash
make all          # install, fetch data, validate, train, cluster, plot, write the report
```

That is the whole thing. It runs the stages below in order, and when it finishes
**open `reports/report.html`**: every result on one page, readable offline in any
browser, with every table also saved as CSV in `reports/tables/`. On a 16-core
laptop the first run takes roughly 15–20 minutes (the download is ~400 MB; the
training stage alone is ~9 minutes). Individual stages:

```bash
make setup        # install Python dependencies
make data         # download + clean 90 days, build features, validate the data
make live         # poll the MBTA V3 API and record live snapshots (~10 min)
make model        # train and evaluate the delay models (Track A)
make model-full   # as above, plus the costly random forest and KNN
make cluster      # cluster stations by reliability and demand (Track B)
make figures      # render interactive HTML figures and static PNGs
make report       # write reports/report.html and reports/tables/*.csv
make test         # run the test suite (142 tests, no network required)
```

**Reproducing the exact numbers on another computer.** `requirements.txt` accepts
compatible newer library versions, which can shift results in the last decimal.
For an exact reproduction, install the pinned versions these results were made
with, then run the pipeline:

```bash
python -m pip install -r requirements-lock.txt
make all
```

The same cached data and pinned versions give byte-identical results (§12).

Useful knobs:

```bash
make DAYS=180 data        # fetch a longer analysis window
```

**Windows.** `make` is not installed on a stock Windows machine, so `make.ps1`
exposes the same targets:

```powershell
.\make.ps1 all
.\make.ps1 data -Days 180
```

You can also skip `make` entirely and call the pipeline directly:

```bash
PYTHONPATH=src python -m mbta_ds.cli collect --days 90
PYTHONPATH=src python -m mbta_ds.cli all --days 90
```

### Progress output

Long stages report progress so a multi-minute run never looks like a hang. Two
levels are printed: a `[n/total]` counter for the stage as a whole, and one per
inner loop with the running metric and an ETA.

```
INFO mbta_ds.progress: train stage: 0/7
INFO mbta_ds.progress: [5/7] regression models decision_tree  (elapsed 22s, eta 9s)  MAE 29.8s  R2 0.971
INFO mbta_ds.progress: [2/4] walk-forward backtest 20260520..20260602  (elapsed 36s, eta 36s)  MAE 23.3s vs 48.7s
INFO mbta_ds.progress: [3/4] prediction horizon 5 stop(s) ahead  (elapsed 2m 40s, eta 53s)  persistence 104.5s  own 72.5s  +others 71.4s
```

Counting is built on the logging framework (`src/mbta_ds/progress.py`) rather than a
progress-bar dependency, so the output lands in the same stream as everything else
and is captured by a redirect. The same `Progress` helper is reused by every slow
loop: model fitting, ablations, the k-sweep in clustering, feature joins, and the
bulk downloads. Raise verbosity or dial the interval with `Progress(log_every=...)`.

### Environment and API key

The project runs with **no API key at all**. The bulk historical data is public,
and the live collector stays under the anonymous rate limit (3 requests/minute
against a 20/minute allowance).

A key is optional and only raises the rate limit:

```bash
cp .env.example .env      # then put your key in MBTA_API_KEY
```

`.env` is gitignored and never committed. Keys are free from
<https://api-v3.mbta.com/portal>.

### What gets produced

| Path | Contents |
|---|---|
| `data/processed/trips_clean.parquet` | tidy trip-stop table with computed delay |
| `data/processed/clean_ledger.csv` | how many rows each cleaning step removed |
| `data/processed/validation.csv` | every data-validity check, its value and limit |
| `data/processed/features.parquet` | the leakage-safe modelling matrix |
| `data/processed/model_comparison.csv` | every regression candidate scored |
| `data/processed/ablation.csv` | test MAE per feature group |
| `data/processed/trend_diagnostic.csv` | the time-trend experiment |
| `data/processed/station_clusters.parquet` | per-station cluster assignments |
| **`reports/report.html`** | **every result on one self-contained page — start here** |
| `reports/tables/*.csv` | each results table, with readable column names |
| `reports/figures/*.html` | interactive figures (open in a browser) |
| `reports/figures/*.png` | static versions, embedded in the report |

---

## 2. Project goals

Stated so they can be measured:

1. **Predict subway arrival delay** at a given stop to within a mean absolute
   error materially below the delay-persistence baseline, evaluated on a future
   time period never seen in training.
2. **Classify lateness** (>5 minutes behind schedule) with F1 and ROC-AUC above
   that same baseline.
3. **Quantify which data sources actually help**, by ablating feature groups
   rather than asserting their value.
4. **Cluster the 125 subway stations** into interpretable reliability and demand
   types, and test whether the two typologies are related.

All four are met, and the fourth produces a **null result that is reported as
such** (§8).

---

## 3. Data sources

Five independent sources. Every endpoint below was verified live during
development; nothing is assumed to exist.

| # | Source | What it provides | Access |
|---|---|---|---|
| 1 | **LAMP subway on-time performance** | One Parquet file per service date, one row per `trip_id` × `stop_id`, with the *observed* stop timestamp **and** the *scheduled* arrival/departure time, plus dwell time and headways. Coverage 2019-09-15 → present. | `performancedata.mbta.com` — public, no key |
| 2 | **MBTA V3 API** | Live `/predictions`, `/vehicles`, `/alerts`; also station names and coordinates for the map, which the static table lacks. | `api-v3.mbta.com` — key optional |
| 3 | **Gated station entries** | Faregate validations per station per 30-minute period. The demand signal. | ArcGIS FeatureServer — public |
| 4 | **Open-Meteo historical archive** | Hourly temperature, precipitation, snowfall, wind, humidity for Boston. | `archive-api.open-meteo.com` — keyless |
| 5 | **LAMP alerts archive** | Archived GTFS-Realtime alerts with `cause`, `effect`, `severity`, text, and affected route/stop. | `performancedata.mbta.com` — public |

**Why LAMP rather than the live API for training?** The V3 API is a *snapshot*: it
tells you where trains are now, not how late they were on 400 previous days. The
LAMP archive is the retrospective record, and pairing observed against scheduled
times is what makes delay a *label* rather than an inference. The live API is still
used, for two things it alone can do: station coordinates, and the live collector.

### Two coverage traps, both handled in code

* **The ridership service is shorter than the performance archive.** Probing it
  showed coverage ending 2026-06-30 while LAMP ran to 2026-09-27. Rather than
  hard-code dates, `collect_ridership.coverage()` queries the live bounds and the
  collection stage intersects all sources into **one shared window**, persisted to
  `data/analysis_window.json`. Every later stage reads that file, so no stage can
  silently disagree about which dates are under analysis. This window is
  **2026-04-02 → 2026-06-30 (90 service days)**.
* **56 of 125 stations have no gated entries at all.** Green Line surface stops
  west of Kenmore and parts of the Green Line Extension have no faregates. That
  absence is informative, so it is flagged (`demand_missing`) rather than imputed
  to zero.

### Data volumes actually collected

| Source | Volume |
|---|---|
| Performance records | 3,906,187 rows across 90 daily Parquet files |
| Ridership | 282,237 half-hourly rows → 6,964 station-days (72 gated stations) |
| Weather | 2,160 hourly rows |
| Alerts archive | full GTFS-Realtime alert history |
| Station coordinates | 125 stations from the V3 API |
| **Live capture** | **5 polls: 3,932 predictions, 281 vehicle positions, 50 alerts** |

---

## 4. Repository structure

```
.
├── README.md                  ← this document (the final report)
├── makefile                   ← required build entry point
├── make.ps1                   ← Windows equivalent (no `make` needed)
├── requirements.txt           ← compatible version ranges
├── requirements-lock.txt      ← exact versions, for byte-identical reproduction
├── pytest.ini                 ← puts src/ on the path so `pytest` just works
├── .env.example               ← copy to .env to add an API key (optional)
├── src/mbta_ds/               one module per pipeline stage, in run order:
│   ├── config.py              paths, endpoints, window manifest, .env loader
│   ├── http.py                one retrying session + atomic cached downloads
│   ├── progress.py            [n/total] counters with elapsed time and ETA
│   ├── collect_lamp.py        performance archive, stops table, alerts
│   ├── collect_ridership.py   gated entries (coverage probe + parallel paging)
│   ├── collect_weather.py     Open-Meteo hourly weather
│   ├── collect_v3.py          live poller, station coordinates
│   ├── clean.py               delay computation and the ten cleaning rules
│   ├── features.py            the leakage-safe design matrix, any horizon
│   ├── validate.py            25 data-validity checks across every source
│   ├── model_delay.py         Track A: models, ablation, backtest, horizons
│   ├── cluster_stations.py    Track B: reliability and demand clustering
│   ├── viz.py                 interactive Plotly HTML + matplotlib PNG
│   ├── report.py              reports/report.html + reports/tables/*.csv
│   └── cli.py                 `python -m mbta_ds.cli <stage>`
├── tests/                     142 tests, network-free, synthetic fixtures
├── reports/                   everything a person reads (regenerated by `make all`)
│   ├── report.html            all results on one offline page
│   ├── tables/                each results table as CSV
│   └── reports/figures/               interactive HTML and static PNG figures
└── data/                      raw downloads + processed tables (gitignored, regenerable)
```

---

## 5. Data cleaning

Delay is computed as **observed stop time − scheduled arrival time**, anchored at
**local midnight of the service date**:

```
delay_seconds = stop_timestamp − (midnight_local(service_date) + scheduled_arrival_time)
```

Anchoring at local midnight is what makes the MBTA's 3 AM service-day boundary
work without special cases: post-midnight trips carry GTFS times above 86400
seconds (23,971 of 316,427 rows in a five-day sample did), and adding the raw GTFS
time to the previous calendar date's midnight resolves them correctly, including
across the DST transition. This is asserted directly by the tests.

### The rules, and the evidence behind them

Every rule came from inspecting real data, not from assumption:

**1. Exclude `ADDED-*` and `NONREV-*` trips — 652,192 rows (16.7%).**
`trip_id` has three namespaces. Regular numeric trips have a textbook delay
distribution. `ADDED-*` trips are created at runtime and `NONREV-*` trips are
non-revenue moves; **neither has a meaningful entry in the published schedule**, so
their apparent "delay" is an artefact of a bad match:

| namespace | rows | median delay | mean delay |
|---|---|---|---|
| numeric (kept) | 259,343 | **+63 s** | +153 s |
| `ADDED-*` | 47,156 | −9,817 s | −13,198 s |
| `NONREV-*` | 1,494 | −18,734 s | −22,519 s |

They are excluded not because they are outliers but because the quantity being
predicted is *undefined* for them. Keeping them would have dragged mean delay to
−2,001 s and made 13% of rows look absurd.

**2. Drop rows missing an observation or a schedule — 11,837 rows (0.30%).**
A delay needs both sides of the subtraction.

**3. De-duplicate on `(service_date, trip_id, stop_id)` — 3,028 rows (0.08%).**
Critically, the key **includes `service_date`**: `trip_id` is reused across days
(2,123 of 13,305 trip ids appear on more than one date), so grouping on `trip_id`
alone silently mixes days. An early version of this analysis did exactly that and
produced nonsense lag features — see the tests in `test_features.py` that now pin
the behaviour down.

**4. Flag, do not delete, implausible values.** Delays beyond ±1 hour (0.29% of
rows) and dwell times beyond 30 minutes are flagged (`delay_outlier`,
`dwell_implausible`) rather than dropped, so the report can quantify the tail the
models are fighting instead of hiding it. Dwell outliers are set to null.

**5. Resolve station names.** The LAMP data dictionary says `parent_station` holds
a stop *name*; it actually holds a stop **id** (e.g. `place-alfcl`). Names are
joined from the static stops table. That table carries every schedule version since
2019 (750 of them), and stations get renamed, so the **latest** version's name is
used. Taking the first row per id, as an earlier version did, produced 2019 names
("Packards Corner", "Science Park") that no longer match the V3 API, and silently
dropped three stations from the map.

**6. Order each trip by scheduled time, not `stop_sequence` — 3,830 trips reordered.**
`stop_sequence` is not a reliable order. On 3,830 trips (2%), mostly Green-E into
Heath Street, the trip's **final** stop is labelled `stop_sequence == 1`. Sorting by
it put the end of the trip first, so the next stop's "previous delay" was really
the delay at the *end* of the trip — information from the future. The scheduled
arrival time is monotone by construction, and ordering by it removes every
schedule inversion (3,907 → 0) and 65% of the apparent time-travel (5,703 → 1,992).
The stop numbers also step by 10 on most lines (up to 710 on a 32-stop trip), which
had made `fraction_through_trip` range up to 670 instead of 0–1.

**7. Keep one row per scheduled visit — 483 rows.** At Kenmore, Ashmont and
Mattapan, one arrival is sometimes recorded under two platform ids. The second copy
became its own "previous stop", handing the model the answer.

**8. Drop whole trips running more than 30 minutes early — 435 trips, 6,355 rows.**
Some trips sit at a steady offset of about an hour *ahead* of schedule at every
stop (one Green-D trip: −4,051 s at Kenmore, −4,141 s at Riverside, varying by
under a minute). On a line where trains cannot overtake, a trip 30 minutes early
would have passed the 3–5 trains scheduled in front of it. These are vehicles
matched to the wrong timetable entry, so their "delay" is undefined, like rule 1.
Trips 5–15 minutes early are kept: they are common on Green-E out of Heath Street,
where departures are not held to the clock (§8). Very *late* trips are kept too; they cluster
on known disruption days, and a late train does not need to overtake anything.

**9. Flag arrivals timestamped before the previous stop — 1,636 rows (0.05%).**
A train cannot reach stop *k*+1 before stop *k*. These rows (`time_inconsistent`)
stay in the table but are not used as delays.

**10. Split trips where the train changes, and don't score each train's first stop
— 188,153 rows.** Trains reach the origin platform and wait: the median "delay"
there is **−190 s**, 15% of origins look more than 10 minutes early, and the model's
error on them was **307 s** (6% of test rows, 35% of all error). On 300 Green Line
trips, a *different vehicle* finishes the trip (the record switches trains at, say,
Copley). The delay jumps a median 68 s at that hand-over, against 24 s between
ordinary stops, so each vehicle's stretch is its own "run" and lag features never
reach across trains. The first stop of each run is kept as the source of the next
stop's lag features, but `clean.is_arrival` excludes it wherever delay is
*measured*: model targets, clustering and figures. Left in, origins also made
terminals look punctual (§8). (A further 1,328 trips change their recorded
`stop_count` partway through with the *same* train; that is bookkeeping in the
source, the delays run on smoothly, and those trips are left whole.)

### Result

Delay statistics describe genuine arrivals (rules 9 and 10 excluded).

| | |
|---|---|
| Raw rows | 3,906,187 |
| **Clean rows** | **3,232,292** |
| Arrival rows (delay measured) | 3,042,503 |
| Median delay | **+56 s** |
| Mean delay | +129.4 s |
| 5th–95th percentile | −421 s → +832 s |
| Share more than 5 min late | 21.4% |
| Flagged as outliers (beyond ±1 h) | 0.24% |

**Every derived value was re-derived independently** on a random sample of real
rows, using different code, and matched exactly: delay (to the second, with
Python's own `zoneinfo`), both lag features in scheduled order, the weather joined
to the scheduled hour, the 7-calendar-day demand lag, and the count of alerts in
force.

**Where values are missing, and why.** The gaps in the arrival table are
structural, not errors. Branch headways are missing only on lines with no
branches (100% on Blue, Orange and Mattapan; under 1% elsewhere). Dwell time is
missing at the trip origin (99.8%), where there is no previous dwell. In the
feature table, `station_entries_lag7` is missing for the 32% of rows at ungated
stations, flagged by `demand_missing`. The boosted models take every gap as
"missing" natively, and the others impute a training-set median inside the
pipeline.

That distribution — a median within a minute of schedule, a long right tail — is
what a functioning transit system looks like, which is the sanity check that the
arithmetic is right.

### Validation

`make data` ends with a `validate` stage (`src/mbta_ds/validate.py`) that runs **25
checks** across the delay table, feature table, ridership and weather, and fails
the build if a hard one fails. For example: no service date missing from the
window, no schedule running backwards inside a trip, no duplicate visits, no trip
matched to the wrong timetable, recorded travel time equal to arrival minus
departure (100% of rows), weather hourly with no gaps, humidity within 0–100%, no
target column among the features. All 25 pass.
One *informational* check counts route-days running below 60% of their usual
volume: **79**. Those are planned shutdowns, visible in the data — the whole Green
Line on 30 May–5 June, Green-C on 6–17 May, and the Red Line at 5–20% of normal on
several weekends. They are real service changes, not errors, so they are reported
rather than removed.

---

## 6. Feature extraction

The central discipline is **causal ordering**: a feature for the arrival at stop
*k* may only use information that exists before the train reaches stop *k*.

* Lag features come from the **previous stops of the same trip**, in scheduled-time
  order (§5 rule 6), produced with `groupby(["service_date", "trip_id"]).shift(1)` —
  never from the current or later rows, and never across a day boundary.
* **Targets are genuine arrivals.** Trip origins and impossible timestamps (§5
  rules 9–10) feed the next stop's lags but are never themselves predicted, so every
  target row has a previous-stop delay.
* Demand is joined from a **lagged 7-day rolling window** over *calendar* days,
  never the same day's total, which is only known once the day is over.
* **Alerts count only from the first full hour after they start.** The archive's
  alerts are reactive, not published in advance: the median gap between an alert's
  creation and the start of its active period is zero. Bucketing an alert into the
  hour it started would let one raised at 08:50 inform an 08:10 arrival. Elevator
  and escalator outages (`ACCESSIBILITY_ISSUE`, ~93% of subway alerts) are excluded
  because they say nothing about train running.
* **No target encoding is done ahead of time.** Categorical identifiers reach the
  model as raw categories, and every imputation, scaling and encoding step lives
  inside a scikit-learn `Pipeline` that is fit on training rows only.
* Weather is joined on the *scheduled* hour. This is a **nowcast, not a forecast**,
  and is called out in the limitations.

**Ordering bug caught during development.** The first version subsampled rows
*before* computing lags. That corrupted every lag feature — `shift(1)` returned the
previous *surviving* stop rather than the previous stop — and inflated the missing
rate from 5.8% to 29%. The pipeline now computes row-wise features on the full clean
table and subsamples afterwards. The reason is documented in `features.build`.

| Group | Features |
|---|---|
| `schedule` | `stop_sequence`, `stop_count`, `fraction_through_trip`, `scheduled_seconds_of_day`, `scheduled_hour`, `scheduled_elapsed_seconds`, `scheduled_travel_time`, `scheduled_headway_branch`, `scheduled_headway_trunk` |
| `propagation` | `prev_delay_1`, `prev_delay_2`, `prev_dwell_seconds`, `prev_travel_time_seconds`, `prev_headway_seconds`, `delay_trend`, `scheduled_seconds_ahead` |
| `calendar` | `day_of_week`, `is_weekend`, `is_peak` |
| `demand` | `station_entries_lag7`, `station_entries_trend`, `demand_missing` |
| `weather` | `temperature_c`, `precip_mm`, `snowfall_cm`, `wind_kph`, `humidity_pct`, `is_precip`, `is_snow`, `weather_missing` |
| `alerts` | `route_alerts_active`, `route_alert_severity_max` |
| `network` | `leader_delay`, `leader_age_seconds`, `line_late_share_15m`, `line_arrivals_15m` |
| `categorical` | `route_id`, `trunk_route_id`, `direction_id`, `station_name` |

**The `network` group: other trains on the line.** Delay does not only travel along
a trip; it travels down a line, because trains queue behind each other. For every
arrival the model predicts, these features describe the line **as it stood at the
moment of prediction** — when the train left its previous stop — and nothing later:
the delay of the last train to reach the target station and how long ago it did,
and the share of the line's arrivals in the previous 15 minutes that were more than
5 minutes late. Branches that share track count as one line (every Green train
queues at Park Street). The lookups are `merge_asof` joins that only look strictly
backwards in time, and the tests pin down that a train arriving *after* the
prediction moment is never used.

**Prediction horizon.** `features.build(horizon=k)` builds the same table for
predicting delay *k* stops ahead: every `prev_*` feature then describes the stop *k*
back, `scheduled_seconds_ahead` is the scheduled time still to run, and the network
features are taken at the moment the train left that stop. §7 uses this to ask how
far ahead delay can be predicted.

Groups are declared in code (`features.FEATURE_GROUPS`) so the trainer can ablate
them declaratively, and the build fails loudly if a declared feature is ever not
produced — a guard added after `scheduled_headway_trunk` was silently missing from
the download list. `days_since_window_start` is still built, but only for the
diagnostic in §7. It is deliberately **not** a model feature, for the reason §7
explains.

**Modelling matrix:** 3,042,503 rows × 39 features — every predictable arrival in the
window, built in about 20 seconds. The trainer fits on a seeded sample of 450,000
earlier-period rows, **stratified by service date** so every day is represented, and
scores on *every* later arrival. An earlier version sampled 600k rows before
splitting, so its test set was a sample too, and results moved by a few tenths of a
second whenever the sample was redrawn.

---

## 7. Track A — modelling delay

### Evaluation design

* **Temporal split, never random.** Train on the earlier 75% of *service dates*,
  test on the later 25%. A random row split would put some stops of a trip in train
  and others in test, leaking that trip's own delay path across the boundary. It
  also matches the question a rider asks: given what happened up to yesterday, what
  happens tomorrow?
* **Every test arrival is scored.** Training uses a date-stratified sample; the test
  set is never sampled.
* **Three baselines always reported**, including **delay persistence** — just
  predict the previous stop's delay. Beating persistence is the real bar here, and
  it is not automatic.
* **Several test periods, not one** (the walk-forward backtest below), so a result
  cannot hinge on which weeks happened to be held out.

| | |
|---|---|
| Cutoff | 2026-06-07 |
| Train | 449,997 rows sampled from 2026-04-02 → 2026-06-07 |
| Test | **all 872,054 arrivals**, 2026-06-08 → 2026-06-30 |

### Regression: predicting delay in seconds

| Model | MAE (s) | RMSE (s) | R² | Bias (s) |
|---|---|---|---|---|
| **Gradient boosting, absolute-error loss** | **23.2** | 103.3 | 0.955 | −6.3 |
| Gradient boosting, squared-error loss | 27.7 | 104.0 | 0.955 | +0.5 |
| Decision tree | 29.8 | **82.7** | **0.971** | +0.5 |
| Ridge regression | 40.8 | 101.9 | 0.956 | 0.0 |
| *Baseline: persistence* | *50.1* | *140.6* | *0.917* | *−22.7* |
| *Baseline: route × hour mean* | *275.0* | *468.2* | *0.080* | *−24.8* |
| *Baseline: predict on time* | *286.0* | *509.1* | *−0.088* | *−144.5* |

**Goal 1 met: 23.2 s vs 50.1 s persistence — a 54% reduction in error.** The two
naïve baselines are *actively harmful* (R² ≈ 0): subway delay is not explained by
route and hour at all. Only persistence is a real competitor.

**Train on the metric you are scored on.** The same boosted model drops from 27.7 s
to 23.2 s simply by minimising absolute rather than squared error. Squared error
chases the rare 30-minute meltdowns at the expense of the typical arrival. The
trade-off is visible in the table: the absolute-error model is best on MAE, while
the decision tree, which fits large delays more closely, has by far the lowest RMSE.
The absolute-error model also has a −6.3 s bias, because it predicts the typical
(median) arrival rather than the mean, which the long right tail pulls upwards.

**These numbers are not comparable with earlier versions of this report** (51.1 s
vs 73.5 s), because the test set changed, each time by removing rows whose "delay"
was not a real delay, and finally by scoring every test arrival instead of a sample.
Measured step by step:

| Step | MAE (s) |
|---|---|
| Earlier model, all test rows | 51.1 |
| Earlier model, genuine arrivals only (origins removed, §5 rule 10) | 34.4 |
| This version, same arrival rows (trip order fixed, MAE loss, native missing values) | 26.4 |
| This version, mismatched trips also removed (§5 rule 8) | 24.1 |
| This version, runs split at vehicle hand-overs (§5 rule 10); a new 600k sample | 24.4 |
| This version, `network` features added; trained on 450k rows, **all** test arrivals scored | **23.2** |

The genuine modelling improvement, measured on identical rows, is 34.4 s → 26.4 s
(23%). The rest is the test set no longer containing rows that had no correct answer.

### Does it hold on other weeks? The walk-forward backtest

The model is refit four times, each time on everything before a two-week window,
and scored on that window:

| Test window | Training days | Arrivals scored | Persistence MAE (s) | Model MAE (s) | Improvement |
|---|---|---|---|---|---|
| 6 May – 19 May | 34 | 440,054 | 50.7 | 24.8 | 51% |
| 20 May – 2 Jun | 48 | 418,364 | 48.7 | 23.3 | 52% |
| 3 Jun – 16 Jun | 62 | 441,119 | 51.1 | 26.7 | 48% |
| 17 Jun – 30 Jun | 76 | 536,987 | 49.5 | 21.8 | 56% |

**The improvement is steady at 48–56% in every window**, including the first, trained
on only 34 days. The headline is not the product of one convenient test period.

### Predicting further ahead

Predicting one stop ahead is the easiest version of the problem, because delay
barely changes from stop to stop. A rider cares about the stop they are going to,
often several stops away. With `features.build(horizon=k)` the model predicts delay
*k* stops ahead, knowing only what had happened when the train left the stop *k*
back:

| Stops ahead | Arrivals scored | Persistence MAE (s) | Own train only (s) | + other trains (s) | Improvement |
|---|---|---|---|---|---|
| 1 | 872,054 | 50.1 | 23.9 | **23.2** | 54% |
| 3 | 768,460 | 79.6 | 50.5 | **49.8** | 37% |
| 5 | 665,390 | 104.5 | 72.5 | **71.4** | 32% |
| 10 | 422,387 | 159.7 | 117.6 | **116.4** | 27% |

Two results. **The model stays well ahead of persistence at every horizon**:
predicting a train's delay ten stops out, it is wrong by about two minutes, where
"same as now" is wrong by nearly three. And **the other trains on the line help, but
only a little** — about a second at every horizon. What the train ahead did is
already mostly reflected in your own train's recent delay, because the two are held
up by the same things. (Each horizon scores the arrivals that are at least *k* stops
into their run, so the row counts differ.)

### Classification: will it be more than 5 minutes late?

| Model | F1 | Precision | Recall | ROC-AUC | PR-AUC | Accuracy |
|---|---|---|---|---|---|---|
| **Histogram gradient boosting** | **0.964** | 0.975 | 0.953 | **0.997** | **0.993** | **0.984** |
| Decision tree | 0.953 | 0.963 | 0.943 | 0.987 | 0.980 | 0.979 |
| *Baseline: persistence* | *0.933* | *0.955* | *0.912* | *0.984* | *0.975* | *0.971* |
| Logistic regression | 0.933 | 0.950 | 0.917 | 0.991 | 0.980 | 0.970 |
| *Baseline: route × hour rate* | *0.001* | *0.280* | *0.000* | *0.656* | *0.341* | *0.774* |
| *Baseline: always on time* | *0.000* | *0.000* | *0.000* | *0.500* | *0.226* | *0.774* |

**Goal 2 met, modestly.** The base rate is **22.6%** late, so accuracy alone is
uninformative: the "always on time" baseline scores 77.4% accuracy and 0.0 F1. PR-AUC
is the metric that matters at this base rate, and the best model reaches **0.993**
against a 0.226 floor. But persistence alone already reaches F1 0.933, so the
learned classifier's lift (to 0.964) is real but small: whether a train is more
than five minutes late is mostly decided by whether it already was.

### Ablation: what actually helps

Test MAE as feature groups are added cumulatively. The probe is a fixed
squared-error boosted model, so every feature set is judged by the same learner.
Each row is the **mean of three seeds**: with a single seed, a row moved by up to
2.5 s between runs, more than most feature groups are worth.

| Feature set | n | Test MAE (s) | Δ | Seed spread (s) |
|---|---|---|---|---|
| schedule only | 9 | 267.0 | — | 0.9 |
| + route/station | 13 | 264.6 | −2.4 | 0.5 |
| + calendar | 16 | 265.4 | +0.8 | 0.6 |
| + delay propagation | 23 | **30.7** | **−234.7** | 1.6 |
| + demand | 26 | 31.1 | +0.4 | 1.7 |
| + weather | 34 | 29.6 | −1.5 | 0.3 |
| + alerts | 36 | 29.0 | −0.6 | 1.2 |
| + other trains | 40 | 28.5 | −0.5 | 0.9 |

A Δ smaller than the seed spread beside it is noise, and so is anything that flips
between runs: before test sets were scored in full, redrawing the 600k-row sample
moved demand's Δ from +0.3 s to +1.8 s and weather's from −1.8 s to −3.6 s. **Only
delay propagation's effect is unambiguous.** The others are "a couple of seconds
either way".

**Goal 3 met, and it produced four findings worth reporting.**

**Delay propagation is essentially the whole model.** Everything before it is
worth 265 s; adding it drops MAE to 30.7 s. Permutation importance agrees
decisively:

| Feature | Increase in MAE when shuffled |
|---|---|
| `prev_delay_1` | **+360.0 s** |
| `prev_delay_2` | +34.0 s |
| `prev_dwell_seconds` | +21.0 s |
| `station_name` | +18.5 s |
| `scheduled_travel_time` | +10.4 s |

One feature is an order of magnitude more important than any other. This is a
**true but unglamorous** result, and the honest conclusion is that the interesting
modelling problem is not "what causes delay" but "how does delay propagate through
a trip" (§8 shows why: delay barely changes from stop to stop).

**Weather helps a little, and only one model uses it.** The ablation probe gains
1.5–3.6 s from weather depending on the run, and it helped in every seed tried. But
in the final absolute-error model, shuffling any weather feature changes MAE by at
most 0.1 s. The defensible claim is "at most a couple of seconds, in a snow-free
window".

**Alerts: from harmful to neutral once made causal.** An earlier version joined each
alert to the hour it *started* in, and counted elevator outages; adding it made test
MAE *worse* by 3.8 s. With service-affecting alerts counted only from the first full
hour after they start (§6), the effect ranges from −0.9 s to +0.9 s across five
seeds. In other words, no reliable effect, and no longer a leak.

**A feature that raised test error by 70%.** An earlier version of the calendar
group included `days_since_window_start`, and adding that group *raised* test MAE
from 265 s to 449 s. That is not a bug, and isolating it is worth the space. The
feature is a monotone index of the analysis window. Under a temporal split its
training values span [0, 66] and its test values span [67, 89] — the two ranges
**share no values whatsoever**. It has been removed from the model's features, and
the diagnostic below re-adds it explicitly to reproduce the effect:

| Configuration | Test MAE (s) | R² | Seed spread (s) |
|---|---|---|---|
| schedule + route/station | 264.6 | 0.118 | 0.5 |
| + calendar **with** time trend | **449.2** | **−0.936** | **11.2** |
| + calendar, trend removed | 265.4 | 0.133 | 0.6 |
| + time trend **alone** | 309.5 | −0.206 | 13.5 |
| + propagation, with trend | 29.7 | 0.950 | 1.5 |
| + propagation, trend removed | 30.7 | 0.931 | 1.6 |

The seed spread tells the story. With the trend, the model's error swings by
11–14 s depending only on which rows its early-stopping check holds out; without
it, by under 2 s. With delay propagation present, the trend *seems* to help by
1.0 s, but that is inside its own 1.5 s spread, and an earlier run of this pipeline
found it hurting by 1.8 s. A feature whose effect changes sign between runs cannot
be trusted in production.

A tree that splits on that feature is extrapolating off the end of its training
data, and because the boost's internal validation split is drawn from *within* the
training window, nothing in training exposes the problem. The lesson generalises:
**any monotone time index is dangerous under a temporal split**, and an ablation
that only ever improves is a sign the failure modes have not been probed. The
experiment is not a throwaway — it lives in `model_delay.run_calendar_diagnostic`
and its output is committed to `data/processed/trend_diagnostic.csv`.

### Where the error lives

| By route | MAE (s) | | By time band | MAE (s) | | By condition | MAE (s) |
|---|---|---|---|---|---|---|---|
| By route | MAE (s) | | By time band | MAE (s) | | By condition | MAE (s) |
|---|---|---|---|---|---|---|---|
| Blue | 53.4 | | PM peak | 29.4 | | precipitation | 24.0 |
| Green-E | 27.5 | | midday | 24.8 | | dry | 23.0 |
| Green-D | 26.0 | | evening | 20.7 | | | |
| Red | 25.0 | | AM peak | 18.9 | | | |
| Orange | 8.5 | | overnight | 18.4 | | | |

**Normal days and bad days.** A route-day counts as *disrupted* when at least 20% of
its arrivals ran more than 10 minutes late. That is deliberately strict: on a typical
route-day 6% already do (about 15% on Blue and Red), so a looser cut-off would call
most days "disrupted".

| Route-day | Share of test arrivals | MAE (s) | Median error (s) |
|---|---|---|---|
| normal | 89% | **19.3** | 6.2 |
| disrupted (worst ~12% of route-days) | 11% | **54.8** | 8.1 |

On a normal day the model's average error is under 20 seconds. The disrupted days,
about one arrival in nine, produce over a quarter of all the error, which is why
predicting disruptions is the obvious next problem (§14).

The Blue Line is by far the hardest to predict and the Orange Line the easiest. The
Blue Line's *median* error is only 5.3 s, so its high MAE comes from a handful of
very large misses: half of all >1-hour delays in the window fall on 8 route-days,
four of them Blue Line days in the test period (9, 16, 29 and 30 June). Error is a
little higher in precipitation (24.0 s vs 23.0 s), but the window runs April to June
and snowfall is essentially absent (maximum 0.42 cm, present on 0.3% of rows). That
is a genuine limitation of the analysis period, not evidence that weather never
matters — see §11.

---

## 8. Track B — clustering stations

Both matrices are standardised, reduced with PCA (the SVD of the centred matrix —
the course's SVD/LSA material in practice), and clustered with **k-means++** with
*k* chosen by silhouette across k = 2…8. Results are cross-checked against **Ward
hierarchical**, **Gaussian mixture** and **DBSCAN**, with the agreement between
methods reported as an adjusted Rand index so the clusters are not an artefact of
one algorithm's assumptions.

### Reliability clusters

Station × hour-of-day matrix of **median** arrival delay (median, not mean, so a
handful of meltdowns cannot dominate a station's profile), plus summary reliability
statistics. PCA: 6 components capture 92.2% of variance, 69% in the first alone.

| | |
|---|---|
| k | **2** (silhouette 0.468, Davies–Bouldin 0.820) |
| runs early | **25 stations** — median delay −49 s |
| most delay-prone | **100 stations** — median delay +69 s |
| Ward ARI | **0.89** |
| Gaussian mixture ARI | 0.01 |
| DBSCAN | 7 clusters, 11 noise points (ARI 0.34) |

**The "reliable" cluster used to be mostly terminals.** An earlier version found a
31-station "most reliable" cluster. It was measuring the trip-origin artefact (§5
rule 10): 17% of that cluster's rows were trains *waiting to depart*, against 3% in
the other cluster, and 11 of the 15 terminal-heavy stations landed in it. At
Braintree, trains arriving are a median **4.5 minutes late**, but thousands of
departure waits made the station look early. With origins excluded, the split is
different and more interesting. The early cluster is **Green-E to Heath Street,
the Green Line Extension and the whole Mattapan line**, where trains reach stations
ahead of the timetable. It is named "runs early" rather than "reliable", because for
a rider an early train is a missed train.

**Agreement is mixed, and the reason is shape, not doubt.** Ward agrees closely
(ARI 0.89). The Gaussian mixture does not (0.01): with one component
carrying 69% of the variance, the data is close to one-dimensional, and the mixture
separates a tight core from a spread-out tail rather than early from late. DBSCAN
fragments the delay-prone stations into small dense groups. The two-way split is a
reasonable summary of "early vs late", not a sharp natural boundary.

### Demand clusters

Station × 48-half-hour **shape** profile from gated entries, normalised per station
so the clustering captures *when* people travel rather than how many do. PCA: 6
components capture 87.4% of variance.

| | |
|---|---|
| k | **2** (silhouette 0.459, Davies–Bouldin 0.847) |
| morning-peaked | **40 stations** — sharp 08:00 peak (6.6% of the day), then a decline |
| afternoon-peaked | **31 stations** — 17:00 peak (6.7%), with a secondary 08:00 bump |
| Ward ARI / GMM ARI | 0.84 / 0.89 |
| DBSCAN | **finds no clusters** (68 of 71 stations classified as noise) |

**A calculation bug that halved the busiest stations.** The ridership feed splits
each interchange station's entries across its lines. An earlier version *averaged*
the halves instead of adding them, so Downtown Crossing, Park Street, South
Station, North Station, State, Government Center and Haymarket all showed exactly
half their real entries (Downtown Crossing: 6,123 a day instead of 12,227). That
shrank the seven busiest markers on the station map and misplaced them on the
entries-vs-delay chart. It also averaged each half-hour only over days when it
had taps, which overstated quiet periods by ~3% at a typical station and up to 2×
at sparse ones. Daily entries are now lines *summed*, divided by every day the
station reported.

**A data-quality exclusion, and why it was necessary.** An earlier run of this
analysis produced a third cluster containing exactly one station, whose profile
spiked to 16.7% and back to zero six times a day. That is not a travel pattern: the
station is **`Longwood`, which has 7 rows in the entire 90-day window — one faregate
entry per day.** Because each profile is normalised to sum to 1, dividing by a
near-zero total turns measurement noise into an extreme-looking shape. Stations are
now required to have at least 24 of the 48 half-hour periods observed *and* at least
500 total entries before their shape is clustered; every other station has 40–48
periods, so the threshold separates cleanly. The exclusion is logged and recorded in
`cluster_metrics.json` rather than applied silently.

**DBSCAN finding no structure is itself a result.** The two clusters are a
*gradient*, not separated islands — morning share varies continuously from 5% to
51% across stations. Density-based clustering is the right tool for finding islands
and the wrong one here, and reporting that is more informative than tuning `eps`
until it produces an answer the data does not support.

**Naming note.** Cluster names are derived, not asserted. A band is called
"peaked" when it carries at least 1.15× its proportional share of the *clock* — a
comparison against band duration is what makes this fair, since the 6-hour night
band would otherwise always look busy. Cluster names are also guaranteed unique,
because they are used as group keys downstream.

### Cross-track result: no association

| Test | Statistic | p |
|---|---|---|
| χ²: are the two typologies independent? | 1.39 (df 1) | **0.238** |
| One-way ANOVA: does mean delay differ by demand cluster? | F = 0.64 | **0.428** |

**Goal 4 met, with a negative answer.** Mean arrival delay by demand cluster is
morning-peaked 183.2 s, afternoon-peaked 156.0 s, and the ANOVA finds that gap
consistent with chance; the χ² finds no association either. How busy a station is,
and when, does **not** predict how punctually it is served. (With the origin
artefact still in the data, the χ² had been borderline at p = 0.052. Removing a
measurement error removed the hint of a relationship.) This is a legitimate finding
and is reported as one — a project that only reports its positive results is not
reporting honestly.

**The join between the two typologies is by stop id, not by name.** The ridership
feed and the delay data name some stations differently ("State Street" vs "State",
"Mattapan Line" vs "Mattapan"), so an earlier name-based join silently left those
stations out of this test. Joining on the shared `place-…` id brings in every
subway station that has faregates (69, up from 67). Only the two Silver Line
stations remain unmatched, correctly, because they are not in the subway delay data.

### Other findings from the data

**Delays are sticky.** Regressing each arrival's delay on the previous stop's gives
a slope of **0.95–1.00 on every line** except Mattapan (0.78): a train five minutes
late stays five minutes late. Mid-trip, trains gain or lose only a few seconds per
stop, and only 3.4% of hops recover more than a minute. The timetable has almost no
slack to absorb a delay, which is exactly why persistence is so hard to beat.

**The Red Line loses time on the way into its terminals.** The median change in
delay on the final hop of a Red Line trip is **+27 s** (mean +58 s), against about
0 s per hop mid-trip, consistent with trains being held outside a busy terminal.
Green-D shows a larger final-hop mean (+49 s), but that comes from trips *cut short*
at Reservoir (the yard, +911 s), Eliot or Fenway during disruptions, not from
terminal congestion.

**Green-E trains out of Heath Street are not running to the clock.** Towards
Medford/Tufts, Green-E trains look a median 4 minutes early on weekdays and 8
minutes early on weekends, all the way along the line. An earlier version of this
report blamed a padded weekend timetable. Checking the timetable disproved that:

| | Weekday | Weekend |
|---|---|---|
| Scheduled Lechmere → Medford/Tufts (April–June data) | 13.0 min | 12.0 min |
| Actual Lechmere → Medford/Tufts | 12.3 min | 12.2 min |
| Published gap between Heath St departures (MBTA V3 API, Oct 2026) | 8 min | 10 min |
| Median departure from Heath St vs its scheduled time | **−3.5 min** | **−5.5 min** |

The running times match the timetable almost exactly, so nothing is padded. The
offset is there from the very first stop, and it is about **half the gap between
scheduled trains** on both day types (correlation 0.77 with the scheduled gap across
hours). At the second stop, the middle half of "delays" spans 6 minutes, nearly the
whole gap, against 1.7 minutes for the same line in the other direction. That is the
signature of departures that are not held to timetable slots, paired afterwards
with the *next* slot. It would follow from headway-based dispatching at Heath
Street, or from how the source pairs trains with scheduled trips; this data cannot
tell those apart. Either way, for these trips "delay versus the timetable" measures
the pairing, not lateness. The "runs early" station cluster (above) is partly this
artefact on the Heath Street branch, and genuine early running on the Mattapan line.

**Planned shutdowns are plain in the record** (§5, Validation): 79 route-days at
under 60% of normal volume, including the whole Green Line for a week. The model is
trained and tested across them without special handling, which is one reason the
tail is hard to predict (§11).

---

## 9. Visualisations

`make figures` writes **11 interactive HTML figures and 5 static PNGs**; running
`make live` first adds a twelfth (`live_vs_historical.html`) that compares live API
behaviour against the historical record. The interactive figures are self-contained
pages (the Plotly runtime is loaded from a CDN); open any of them in a browser.

| Figure | What it shows |
|---|---|
| `station_map.html` | Every station plotted geographically, coloured by reliability cluster and sized by daily entries |
| `delay_heatmap.html/png` | Station × hour median delay — where and when lateness concentrates |
| `delay_by_route.html` | Delay distribution per route with the 5-minute threshold marked |
| `headways.html` | Observed headway per route vs the scheduled median |
| `ablation.html/png` | Test MAE per feature group, including the calendar regression |
| `feature_importance.html/png` | Permutation importance of the delay model |
| `predicted_vs_actual.html/png` | Density of predicted against actual delay on the test set |
| `roc_pr.html` | ROC and precision-recall curves for every classifier |
| `error_breakdown.html` | MAE by time-of-day band and by route |
| `demand_profiles.html/png` | Mean demand shape per demand cluster across the day |
| `cluster_scatter.html` | Station demand vs mean delay, coloured and shaped by both typologies |
| `live_vs_historical.html` | Live API prediction volatility against the historical delay distribution |

**Visual design decisions, stated so they can be defended:**

* **Violin rather than box** for delay by route, because the delay distribution is
  strongly bimodal near zero and a box plot's quartiles would hide that shape.
* **Heatmap rather than line chart** for station × hour, because the question is
  two-dimensional and a line per station would be unreadable at 125 stations.
* **Median rather than mean** in the heatmap, so a few severe disruptions do not
  recolour a station.
* **Permutation importance rather than a tree's built-in importance**, because
  impurity importance is biased towards high-cardinality features and this model has
  a 124-level station feature that would otherwise dominate the ranking spuriously.
* Distribution and curve figures are drawn from bounded samples. Plotly embeds every
  plotted point; an unsampled violin over 3.2M rows produced a **61 MB** HTML file.
  Reducing that to a 30,000-row sample brings it to 0.6 MB with no visible change in
  the distribution.

### Live data collection

`make live` polls `/predictions`, `/vehicles` and `/alerts` for the eight subway
routes and appends timestamped snapshots as JSONL — line-appendable and crash-safe,
which matters for a process intended to run in the background. The capture above
(5 polls, ~40 s apart) is genuine and used for one thing the historical archive
cannot provide: **how much a predicted arrival time moves between polls.** Across
2,921 repeated observations, the median revision is **0 s** and the mean **1.5 s**
(σ 13.8 s, largest 82 s) — the MBTA's live predictions are stable over ~40-second
intervals. Five polls is a small sample; `make live` runs longer for a sturdier
estimate.

---

## 10. Testing and continuous integration

```bash
make test        # 142 tests, ~10 seconds, no network access
```

`.github/workflows/test.yml` runs the suite on Python 3.11 and 3.12 for every push
and pull request, then checks that every pipeline stage imports cleanly.

The tests are **deliberately network-free** and build small synthetic frames with
the same columns and quirks as the real sources, so CI needs no API key and does not
break when MBTA publishes new data. The behaviours they pin down are the ones most
likely to regress silently:

* the local-midnight schedule anchor, **including the DST transition day**;
* post-midnight trips with GTFS times above 86400 s;
* exclusion of `ADDED-*` / `NONREV-*` trips;
* de-duplication on the full `(service_date, trip_id, stop_id)` key;
* **lag features never crossing service dates or trips** (the bug that produced
  nonsense in an early version);
* the leakage guard: no target column may appear in the feature set;
* temporal splits placing all train dates strictly before all test dates, with no
  service date in both;
* clustering recovering a planted structure, and agreement between k-means and Ward;
* band-intensity arithmetic that stops the long night band being labelled a peak;
* cluster names being unique across clusters;
* **alerts never being visible before they start**, elevator outages being ignored,
  and duplicate archive rows counting once;
* demand lags running over **calendar days**, so a station's missing day does not
  shift what "the previous seven days" means;
* station names resolving to the **latest** schedule version, with the platform-id
  fallback staying row-aligned on a filtered, shuffled index;
* ridership stations joining to delay stations **by stop id**, not by name;
* the monotone time index staying **out** of the model's features;
* trips ordered by **scheduled time**, so a final stop mislabelled
  `stop_sequence == 1` never becomes anyone's "previous stop";
* one row per scheduled visit, impossible timestamps flagged, and trip origins
  used as lag sources but never as targets;
* `fraction_through_trip` staying within 0–1;
* the validation checks passing on clean data and failing on broken data;
* clusters that run ahead of the timetable being named "runs early", not "reliable".
* a trip finished by a different vehicle splitting into two runs, with no lag
  reaching across the hand-over;
* the "train ahead" lookup ignoring any train that arrives after the prediction
  moment, trains in the other direction, and other lines; and the 15-minute
  line-lateness window counting only that window;
* horizon-*k* features reaching exactly *k* stops back;
* the report escaping text, formatting counts and scores, and naming models readably;
* whole trips an hour ahead of schedule being dropped as mismatched, while trips a
  few minutes early are kept;
* interchange stations' ridership summed across lines, with silent half-hours
  counted as zero taps.

---

## 11. Limitations and failure cases

Stated plainly, because the rubric asks for them and because they are the honest
boundaries of what this analysis supports.

1. **The window contains essentially no snow.** April–June 2026 means the weather
   features cannot show the effect they would have in winter: snowfall is non-zero
   on only 0.3% of rows and never exceeds 0.42 cm, and weather's measured effect is
   a few seconds at most and model-dependent (§7). Any claim that "weather doesn't matter" would be wrong; the
   correct claim is "it does not matter in a snow-free spring window".
2. **Weather is a nowcast, not a forecast.** Features are joined on the *actual*
   weather at the scheduled hour, which would not be available at prediction time
   for a genuinely forward-looking system.
3. **Delay persistence is a very strong baseline.** The learned model halves its
   MAE, but it is largely re-weighting a single signal rather than discovering
   structure, because delays barely change from stop to stop (§8). For the yes/no
   lateness question the lift over persistence is small (F1 0.933 → 0.964).
4. **The longest stop-level delay is the least predictable part.** The median
   absolute error is ~6 s while the MAE is ~24 s, so the mean is driven by a tail
   the model does not capture. Predicting the *tail* is the open problem.
5. **The headline model under-predicts large delays, by design.** Trained on
   absolute error, it predicts the *typical* outcome (bias −6.3 s). Its large
   misses lean towards under-prediction: 5,493 predictions more than 5 minutes too
   early against 2,639 more than 5 minutes too late. Of 77,407 test arrivals more
   than 10 minutes late, 692 were predicted under 5 minutes late: trains that
   *start* losing time at this stop, with nothing upstream to signal it. An on-time
   train was called 10 minutes late only 13 times in 674,547. If catching big delays
   matters more than the typical error, the decision tree (lowest RMSE) is the
   better choice. It only outputs its leaves' constant values, so its predictions
   form horizontal bands in the same plot.
6. **The demand clusters are a gradient, not separated groups.** DBSCAN finds no
   structure at all, and Ward agreement is 0.84.
   The two clusters are a useful summary of a continuum, not a discovered taxonomy,
   and one station (`Longwood`, 1 entry/day) is excluded from shape clustering
   entirely for insufficient coverage.
7. **56 of 125 stations have no demand signal at all**, so `station_entries_lag7` is
   missing for 32% of rows. The boosted models take the gap as missing, the others
   impute the training median, and all see the `demand_missing` flag; any demand
   effect is estimated from the gated subset only.
8. **Ridership counts are unscaled faregate taps.** They exclude non-tapping riders
   and fare evasion, include employee taps, and are not captured when gates are held
   open. Multi-line stations are split across lines using CTPS survey factors. They
   are **not total ridership**, and the clustering is a clustering of *entry
   behaviour*, not of people.
9. **Models train on a sample.** Each fit uses 450,000 date-stratified training
   rows rather than all 2.2M, to keep the pipeline to minutes; every test arrival
   is scored. Training on every row was tested on an earlier version of the target
   and improved MAE by under 2%.
10. **Some source-data defects can be flagged but not repaired.** 1,636 arrivals
    are timestamped before the previous stop, and are excluded from targets. Green-E
    trips out of Heath Street are compared against timetable slots they do not run
    to (§8), so their "delay" measures the pairing, not lateness. The model learns
    that offset rather than correcting it. For such services, a *headway* measure
    (time since the previous train) would be the fairer yardstick.
11. **`trip_id` is reused across service dates.** Any future analysis that groups on
    `trip_id` alone will silently mix days — the mistake is documented and tested,
    but the underlying data-model wrinkle remains.
12. **`ADDED-*` trips are excluded entirely.** These are real service that real
    riders experienced, so the model is trained on scheduled service only.

---

## 12. Reproducibility

* **Everything is regenerable from the repository.** `make all` rebuilds every
  artefact from public sources. Processed data and downloaded caches are gitignored.
* **One shared analysis window**, persisted to `data/analysis_window.json` by the
  collection stage and read by every later stage, so no stage can quietly analyse a
  different date range.
* **Seeded throughout.** `SEED = 506` is used for sampling, every model and every
  clustering call.
* **Collection is cached and incremental.** Re-running `make data` skips files
  already downloaded; downloads are written atomically via a `.part` file so an
  interrupted run never leaves a truncated file that a later run treats as valid.
  Only data the pipeline actually reads is fetched: of the four LAMP static tables,
  only `stops` is used, so `routes`, `trips` and `stop_times` (~2.6 GB between them,
  2.3 GB of it `stop_times`) are not downloaded.
* **Deterministic stages.** Given the same cached data, the cleaning, feature,
  modelling and clustering stages reproduce identical outputs: rebuilding the
  features and retraining from the same cache gives a byte-identical
  `model_metrics.json` (3,232,292 clean rows, 3,042,503 feature rows).
* **Speed.** The training stage takes ~9 minutes on a 16-core laptop CPU, most of it
  the robustness work (three seeds per ablation row, four backtest refits, four
  prediction horizons); `--quick` cuts it to about a minute. The decision trees get
  a dense rather than sparse one-hot matrix (4× faster, identical splits), and the
  ablation and time-trend diagnostic share a cache, so a feature set is only ever
  fitted once. A GPU was benchmarked (XGBoost on CUDA, RTX 5060 Ti) and is *not*
  used: at 430k training rows it was no faster than the CPU, and it only pulled
  ahead (2.2×) on the full 2.3M-row table, where accuracy improved by under 2%.
* **Four thresholds are data-derived and documented** rather than tuned silently:
  the demand-coverage floor, the band-intensity factor for cluster naming, and the
  row caps for the costly models.
* **The default run is bounded** so `make all` finishes in minutes rather than
  hours. `make DAYS=180 data` and `make model-full` scale it up.

### Pipeline stages

| Stage | Command | Purpose |
|---|---|---|
| Collect | `collect --days 90` | Intersect all sources into one window and download |
| Live | `live --minutes 10` | Poll the V3 API into JSONL snapshots |
| Clean | `clean --refresh` | Compute delay, apply the ten cleaning rules, write the ledger |
| Features | `features --refresh` | Build the leakage-safe matrix, run the ablations' inputs |
| Validate | `validate` | 25 data-validity checks; fails the build on a hard failure |
| Train | `train [--full]` | Track A: models, ablation, diagnostic, backtest, horizons (~9 min) |
| Report | `report` | `reports/report.html` and `reports/tables/*.csv` from the saved results |
| Cluster | `cluster` | Track B: reliability and demand clustering, cross-tests |
| Figures | `figures` | Render interactive HTML + PNG figures (12 with live data, 11 without) |

---

## 13. Rubric map

| Criterion (points) | Where |
|---|---|
| Build & run instructions (`makefile`) — 10 | §1, `makefile`, `make.ps1` |
| Repository well organised — 5 | §4 |
| Results reproducible — 5 | §12 |
| Data collection documented — 5 | §3 |
| Sources identified and justified — 5 | §3 (including why LAMP over the live API) |
| Collection implemented in code — 5 | `collect_*.py`, live capture in §9 |
| Cleaning described — 5 | §5 |
| Missing / noisy / inconsistent handling — 5 | §5 rules 1–10 and Validation, ledger in `clean_ledger.csv`, checks in `validation.csv` |
| Cleaning implemented in code — 5 | `clean.py` + `validate.py`, with tests in `test_clean.py` and `test_validate.py` |
| Features defined and explained — 5 | §6 |
| Features appropriate — 5 | §6, ablation in §7 |
| Training procedure described — 5 | §7 |
| Model choice appropriate — 10 | §7 (three baselines incl. persistence; why trees win) |
| Evaluation strategy appropriate — 5 | §7 (temporal split, leakage guards) |
| Limitations and failure cases — 5 | §11 |
| Visualisations clear — 5 | §9 (all figures labelled, decisions defended) |
| Visualisations reveal insights — 5 | §9 |
| Visualisations support results — 5 | §7–§9 |
| Tests + GitHub workflow | §10, `.github/workflows/test.yml` |

---

## 14. What I would do next

Done since the first version of this list: scoring every test arrival, a
walk-forward backtest, reporting normal and disrupted days separately, features
describing the other trains on the line, and predicting several stops ahead (§7).

1. **Predict disruptions, not the average.** Disrupted route-days are 11% of arrivals
   and over a quarter of the error. Model the chance of a delay over 10 minutes in the
   next half hour directly, or give ranges ("likely 2–4 min, 10% chance of 15+")
   with quantile regression.
2. **Measure high-frequency lines by headway.** For Green-E out of Heath Street, delay
   against the timetable measures the pairing, not lateness (§8). "Minutes until the
   next train" is the fairer target there.
3. **Benchmark against the MBTA's own countdown clocks.** The live API publishes its
   predictions with an uncertainty band, and per-car crowding. Neither is archived, so
   this needs a collector left running for a few weeks (`make live` on a schedule).
4. **A winter window.** Re-run on December–February to actually test the weather
   features, which this window cannot.
5. **Forecast the weather.** Replace the nowcast with a forecast API to make the
   feature legitimate at prediction time.
6. **Add bus data.** LAMP publishes bus events too (7.4 GB), and buses share road
   congestion with the Green Line's street-running branches.
7. **Turn it into a rider-facing artefact.** "Should I leave now?" is a
   decision-theoretic problem on top of the horizon predictions.
