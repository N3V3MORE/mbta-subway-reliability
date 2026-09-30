# Mini Boston: the T, as it actually ran

A 3D map of this project's results, built on the
[Mini Delhi](https://github.com/N3V3MORE/mini-metro-delhi) map app (React, MapLibre, Vite) and fed by the
pipeline in `src/mbta_ds`.

* **NETWORK** stands every station up as a column. Its height is the metric you
  pick: the share of arrivals more than 5 minutes late, the extra wait on a bad
  day (90th percentile) or mean daily faregate entries. Its colour is the
  station's reliability cluster from Track B. Click a station for its numbers and
  its demand through the day.
* **REPLAY** plays back real service days, train by train: a spring weekday, the
  23 February 2026 snowstorm and a day from the untouched July–September holdout.
  Each train is drawn only between stops where the data observed it. Colour and
  column height show how late it was for riders at its last stop. Click a train
  to see the model's forecast of its next-stop delay against the timetable,
  beside the rule of thumb ("it stays as late as it is") and what happened.
  **Ride** follows it at street level.

"Late" means riders waited longer than planned: the gap behind the train in
front, minus the scheduled gap. Delay against the timetable is not used for it,
because on frequent lines trains are paired with timetable slots in order, and a
line running slightly sparse drifts far "behind" while riders see near-normal
service (see the main README, §5).

## Run it

Needs Node.js 20 or later. The data in `public/data/` is committed, so the map runs
straight from a clone:

```bash
npm install
npm run dev
```

`make map` (or `.\make.ps1 map` on Windows) re-exports the data from the pipeline
first and then starts the map.

## Where the data comes from

`python -m mbta_ds.cli export-map` writes:

| File | Contents |
| --- | --- |
| `network.json` | lines and colours, one track geometry per branch (MBTA's canonical V3 shapes), each station's reliability and demand results |
| `replay-<date>.json` | every observed trip that day: station, arrival, time at the stop, lateness for riders, delay against the timetable, and the model's prediction of that delay |

Replay days are chosen from each run's test period, so every forecast shown was
made by a model that never saw that day. Choose others with
`--day 2026-06-12 --day winter:2026-02-20`.

## Checks

```bash
npm run lint
npm test
npm run build
```

Placing every train on a service day takes about 0.01 ms per frame: trips are
sorted by start time, and each stop's distance along its track is resolved once
when the day loads.
