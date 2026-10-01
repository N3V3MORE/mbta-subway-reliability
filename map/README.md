# Mini Boston: the T, as it actually ran

The project's interface: a 3D map of the MBTA subway and every result from the
pipeline in `src/mbta_ds`, in one app (React, MapLibre, Vite). It began from the
[Mini Delhi](https://github.com/N3V3MORE/mini-metro-delhi) map app.

The map is always on screen; a panel docked beside it changes with the view, and
takes the room that view needs: more for reading, the charts and the station
table, less where the map is the subject. Hover anything on the map for its
numbers; a key in its corner says what height and colour mean.

* **Story** (the landing page) tells what the project found, in short chapters:
  how late trains run, whether it can be predicted, where time is lost, what does
  not help, what cannot be foreseen, and whether it holds up. Each has one number
  and one chart, all from `results.json`, and as a chapter scrolls into view the
  map changes to show it.
* **Network** stands every station up as a column. Its height is the measure you
  pick: the share of arrivals more than 5 minutes late, the extra wait on a bad
  day (90th percentile) or mean daily faregate entries. Its colour is the
  station's reliability cluster from Track B. The panel ranks stations by that
  measure; select one for its numbers and its entries through the day.
* **Replay** plays back real service days, train by train: a spring weekday, the
  23 February 2026 snowstorm and a day from the untouched July–September holdout.
  Each train is drawn only between stops where the data observed it, coloured and
  raised by how late it was for riders at its last stop. The transport bar under
  the map shows how many trains ran through the day and marks the MBTA's alerts
  where they were raised. Select a train to follow it stop by stop: when it
  arrived, how far off the timetable it was, and the model's forecast for each
  stop against what happened. **Ride along** follows it at street level.
* **Time lost** shows where trains lose time against each place's own good runs:
  each stretch of track coloured by the time lost moving, a column at each station
  for the time lost standing at its platforms, per train or per day. The panel
  ranks the places, splits each line's loss into moving and standing, and for
  winter and the holdout lists what got worse than in spring.
* **Results** compares the model with the rules of thumb across all three
  periods, then, for the period you pick: error one to ten stops ahead,
  fortnight by fortnight, how late each line runs and when, and what the model
  relies on. Hovering a line in a chart brings it forward on the map. Every chart
  has its numbers behind "Show the numbers".
* **Stations** is every station in a sortable, filterable table; a row finds the
  station on the map.
* **About** says what "late" means, where the data comes from, and how the
  cleaning and data checks went.

Every view has its own link (`#/network/place-pktrm`, `#/replay/2026-02-23/<trip>`),
so a station or a train can be shared.

"Late" means riders waited longer than planned: the gap behind the train in
front, minus the scheduled gap. Delay against the timetable is not used for it,
because on frequent lines trains are paired with timetable slots in order, and a
line running slightly sparse drifts far "behind" while riders see near-normal
service (see REPORT.md, §5).

## Run it

Needs Node.js 20.19+ or 22.12+. The data in `public/data/` is committed, so the app runs
straight from a clone:

```bash
npm install
npm run dev
```

`make map` (or `.\make.ps1 map` on Windows) re-exports the data from the pipeline
first and then starts the app.

## Where the data comes from

`python -m mbta_ds.cli export-map` writes:

| File | Contents |
| --- | --- |
| `network.json` | lines and colours, one track geometry per branch (MBTA's canonical V3 shapes), each station's reliability and demand results |
| `replay-<date>.json` | every observed trip that day: station, arrival, time at the stop, lateness for riders, delay against the timetable, and the model's prediction of that delay; plus the MBTA's alerts that day |
| `results.json` | each built period's results (spring, winter, the July–September holdout): every method's error, error by horizon and by fortnight, lateness by line and hour, early warning, time lost per stretch and platform, cleaning and data checks |

Replay days are chosen from each run's test period, so every forecast shown was
made by a model that never saw that day. Choose others with
`--day 2026-06-12 --day winter:2026-02-20`. A period whose run has not been built
is left out of `results.json`.

## Design notes

The palette is warm charcoal with one amber accent. Chart series keep one
identity everywhere (amber for the model, violet for "stays as late as it is",
green for the running-time lookup), and they, the lateness ramp and the
reliability pair were each checked for colour-blind separation and contrast on
the panel surface. Line colours are the MBTA's own and always carry the line's
name beside them.

## Checks

```bash
npm run lint
npm test
npm run build
```

Placing every train on a service day takes about 0.01 ms per frame: trips are
sorted by start time, and each stop's distance along its track is resolved once
when the day loads.
