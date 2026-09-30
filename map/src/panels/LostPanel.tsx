import { useMemo } from 'react'
import { ArrowLeft, DoorOpen, Route } from 'lucide-react'
import { Legend, TableView } from '../components/charts'
import { directionName, seconds } from '../data/labels'
import { LOST_METRICS, type LostMetric, lostPlaces, MOVING, type Place, placeName, STANDING, valueOf } from '../data/lost'
import type { PreparedNetwork } from '../data/network'
import type { LostPlatform, LostStretch, Period, Results } from '../types'

const LIST_LENGTH = 15

const trainMinutes = (value: number) => `${Math.round(value).toLocaleString('en-US')} train-min`
const formatValue = (metric: LostMetric, value: number) => (metric === 'train' ? seconds(value) : `${Math.round(value)} min`)
const pct = (value: number) => `${Math.round(value * 100)}%`

export function LostPanel({ metric, network, onMetric, onPeriod, onSelect, periodId, results, selectedId }: {
  metric: LostMetric
  network: PreparedNetwork
  onMetric: (metric: LostMetric) => void
  onPeriod: (id: string) => void
  onSelect: (id: string | undefined) => void
  periodId: string
  results: Results
  selectedId?: string
}) {
  const periods = results.periods.filter((p) => p.lost)
  const period = periods.find((p) => p.id === periodId) ?? periods[0]
  const places = useMemo(() => lostPlaces(period), [period])
  if (!period?.lost) {
    return (
      <header className="panel-head">
        <h1>Time lost</h1>
        <p>Not exported yet. Run <code>python -m mbta_ds.cli segments</code>, then <code>export-map</code>.</p>
      </header>
    )
  }
  const selected = places.find((p) => p.id === selectedId)
  if (selected) return <PlaceDetail network={network} onBack={() => onSelect(undefined)} place={selected} />

  const s = period.lost.summary
  const ranked = [...places].sort((a, b) => valueOf(b, metric) - valueOf(a, metric)).slice(0, LIST_LENGTH)
  const top = ranked.length ? valueOf(ranked[0], metric) : 1
  const spring = results.periods.find((p) => p.id === 'spring')

  return (
    <>
      <header className="panel-head">
        <h1>Time lost</h1>
        <p>
          Every stretch of track and every platform, measured against its own good runs (its fastest tenth). Time beyond a
          good run is time lost: moving slower than usual, or standing longer.
        </p>
      </header>

      {periods.length > 1 ? (
        <div className="period-bar">
          <div aria-label="Period" className="segmented" role="radiogroup">
            {periods.map((p) => (
              <button aria-checked={p.id === period.id} key={p.id} onClick={() => onPeriod(p.id)} role="radio" type="button">{p.label}</button>
            ))}
          </div>
          <p className="hint">{period.note}</p>
        </div>
      ) : null}

      <section className="panel-section">
        <dl className="stat-grid is-three">
          <div><dt>Lost a day</dt><dd>{Math.round(s.lost_minutes_per_day).toLocaleString('en-US')}</dd><small>train-minutes, whole network</small></div>
          <div><dt>Lost moving</dt><dd>{pct(s.running_share)}</dd><small>the other {pct(1 - s.running_share)} standing at platforms</small></div>
          <div><dt>Worst tenth of places</dt><dd>{pct(s.top_tenth_share)}</dd><small>of all time lost</small></div>
        </dl>
        <ul className="finding-list">
          <li>
            The last stretch into a terminal costs a typical train <b>{seconds(s.into_terminal.median_lost)}</b> of running
            time, against {seconds(s.into_terminal.elsewhere_median_lost)} elsewhere, most likely waiting outside for a free platform.
          </li>
          <li>
            {s.per_visit.peak_mean < 1.2 * s.per_visit.offpeak_mean ? 'Rush hour barely matters: a' : 'Rush hour costs more: a'} train
            loses <b>{seconds(s.per_visit.peak_mean)}</b> per stretch or platform at the weekday peaks, against{' '}
            {seconds(s.per_visit.offpeak_mean)} the rest of the time.
          </li>
        </ul>
      </section>

      <section className="panel-section">
        <h2>By line</h2>
        <p className="hint">Average seconds a train loses on each stretch it runs, and at each platform.</p>
        <ByLine lines={s.by_line} />
      </section>

      <section className="panel-section">
        <div aria-label="Measure" className="segmented" role="radiogroup">
          {(Object.keys(LOST_METRICS) as LostMetric[]).map((id) => (
            <button aria-checked={metric === id} key={id} onClick={() => onMetric(id)} role="radio" type="button">{LOST_METRICS[id].label}</button>
          ))}
        </div>
        <p className="hint">{LOST_METRICS[metric].detail}. On the map, track is coloured by time lost moving; columns rise with time lost at platforms.</p>
        <Legend items={[{ color: MOVING, label: 'Moving (track)', shape: 'line' }, { color: STANDING, label: 'Standing (platforms)' }]} />
        <h2>{metric === 'train' ? 'Where a train loses the most' : 'What costs the network the most'}</h2>
        <ol className="rank-list">
          {ranked.map((place, i) => (
            <li key={place.id}>
              <button onClick={() => onSelect(place.id)} type="button">
                <span className="rank">{i + 1}</span>
                <span className="rank-name">
                  {place.kind === 'stretch' ? <Route aria-label="Track" size={13} /> : <DoorOpen aria-label="Platforms" size={13} />}
                  {placeName(network, place)}
                </span>
                <span className="rank-value">{formatValue(metric, valueOf(place, metric))}</span>
                <span aria-hidden="true" className="rank-bar">
                  <i style={{ background: place.kind === 'stretch' ? MOVING : STANDING, width: `${(valueOf(place, metric) / top) * 100}%` }} />
                </span>
              </button>
            </li>
          ))}
        </ol>
        <TableView
          columns={['Place', 'Lines', 'Per train', 'Per day', 'Trains']}
          rows={[...places].sort((a, b) => valueOf(b, metric) - valueOf(a, metric)).map((p) => [
            placeName(network, p), p.lines.join(', '), seconds(p.mean), trainMinutes(p.perDay), p.trains.toLocaleString('en-US')])}
        />
      </section>

      {period.id !== 'spring' && spring?.lost ? <Changes network={network} onSelect={onSelect} period={period} spring={spring} /> : null}
    </>
  )
}

function ByLine({ lines }: { lines: { line: string; platform_mean: number; running_mean: number }[] }) {
  const rows = [...lines].sort((a, b) => (b.running_mean + b.platform_mean) - (a.running_mean + a.platform_mean))
  const top = Math.max(...rows.map((r) => r.running_mean + r.platform_mean))
  return (
    <div className="stacked">
      <Legend items={[{ color: MOVING, label: 'Moving' }, { color: STANDING, label: 'Standing' }]} />
      {rows.map((row) => (
        <div className="stacked-row" key={row.line} title={`${row.line}: ${seconds(row.running_mean)} moving, ${seconds(row.platform_mean)} standing`}>
          <span className="bar-label">{row.line}</span>
          <span className="stacked-track" style={{ width: `${((row.running_mean + row.platform_mean) / top) * 100}%` }}>
            <span style={{ background: MOVING, flexGrow: row.running_mean }} />
            <span style={{ background: STANDING, flexGrow: row.platform_mean }} />
          </span>
          <span className="stacked-value">{seconds(row.running_mean + row.platform_mean)}</span>
        </div>
      ))}
      <TableView columns={['Line', 'Moving', 'Standing']} rows={rows.map((r) => [r.line, seconds(r.running_mean), seconds(r.platform_mean)])} />
    </div>
  )
}

/** Places that get worse per train in this period than in spring. */
function Changes({ network, onSelect, period, spring }: { network: PreparedNetwork; onSelect: (id: string) => void; period: Period; spring: Period }) {
  const before = new Map(lostPlaces(spring).map((p) => [p.id, p]))
  const worse = lostPlaces(period)
    .flatMap((p) => { const b = before.get(p.id); return b ? [{ change: p.mean - b.mean, from: b.mean, place: p }] : [] })
    .sort((a, b) => b.change - a.change)
    .slice(0, 6)
  return (
    <section className="panel-section">
      <h2>Worse than in spring</h2>
      <p className="hint">Places where a train loses the most extra time in this period, against the same place in spring.</p>
      <ol className="rank-list">
        {worse.map(({ change, from, place }) => (
          <li key={place.id}>
            <button onClick={() => onSelect(place.id)} type="button">
              <span className="rank">+</span>
              <span className="rank-name">{placeName(network, place)}</span>
              <span className="rank-value">{seconds(from)} → {seconds(place.mean)}</span>
              <span aria-hidden="true" className="rank-bar"><i style={{ background: place.kind === 'stretch' ? MOVING : STANDING, width: `${Math.min(100, (change / worse[0].change) * 100)}%` }} /></span>
            </button>
          </li>
        ))}
      </ol>
    </section>
  )
}

function PlaceDetail({ network, onBack, place }: { network: PreparedNetwork; onBack: () => void; place: Place }) {
  const name = (id: string) => network.stationById.get(id)?.name ?? id
  return (
    <>
      <header className="panel-head">
        <button className="back-button" onClick={onBack} type="button"><ArrowLeft size={15} /> All places</button>
        <p className="tag-row"><span className="chip">{place.kind === 'stretch' ? 'Track between stations' : 'Platforms'}</span></p>
        <h1>{placeName(network, place)}</h1>
        <p>{place.trains.toLocaleString('en-US')} train visits · {trainMinutes(place.perDay)} lost a day · {seconds(place.mean)} per train on average</p>
      </header>
      {place.parts.map((part) => {
        const stretch = 'from' in part ? part as LostStretch : undefined
        return (
          <section className="panel-section" key={`${part.line}-${part.dir}-${stretch?.from ?? (part as LostPlatform).station}`}>
            <h2>{stretch ? `${name(stretch.from)} → ${name(stretch.to)}` : `${part.line} Line, ${directionName(part.line, part.dir)}`}</h2>
            <p className="hint">{stretch ? `${part.line} Line, ${directionName(part.line, part.dir)}${stretch.terminal ? ', into the terminal' : ''}` : 'Time standing at the platform'}</p>
            <dl className="stat-grid">
              <div><dt>Good run</dt><dd>{seconds(part.good)}</dd><small>its fastest tenth</small></div>
              <div><dt>Typical train loses</dt><dd>{seconds(part.median)}</dd><small>median</small></div>
              <div><dt>1 train in 10 loses</dt><dd>{seconds(part.p90)}</dd><small>or more</small></div>
              <div><dt>Every day</dt><dd>{Math.round(part.perDay)} min</dd><small>{part.trains.toLocaleString('en-US')} trains in the window</small></div>
            </dl>
          </section>
        )
      })}
    </>
  )
}
