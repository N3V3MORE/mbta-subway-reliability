import { ArrowLeft, RotateCcw, Train, TriangleAlert } from 'lucide-react'
import { formatClock, formatDate, formatPercent } from '../data/format'
import type { PreparedNetwork } from '../data/network'
import { activeIncidents, DELAY_BANDS, type PreparedReplay, type PreparedTrip } from '../data/replay'
import { LineTag } from '../components/LineTag'
import { TripStrip } from '../components/TripStrip'
import type { Incident } from '../types'

export function ReplayPanel({ dayIndex, network, onDay, onRide, onSeconds, onTrip, replay, riding, seconds, trip }: {
  dayIndex: number
  network: PreparedNetwork
  onDay: (index: number) => void
  onRide: () => void
  onSeconds: (seconds: number) => void
  onTrip: (tripId: string | undefined) => void
  replay: PreparedReplay | null
  riding: boolean
  seconds: number
  trip?: PreparedTrip
}) {
  if (trip) return <TripDetail network={network} onBack={() => onTrip(undefined)} onRide={onRide} onSeconds={onSeconds} riding={riding} seconds={seconds} trip={trip} />
  const day = network.replays[dayIndex]
  if (!day) {
    return (
      <header className="panel-head">
        <h1>Replay</h1>
        <p>No replay days were exported. Run <code>python -m mbta_ds.cli export-map</code> after training.</p>
      </header>
    )
  }
  const active = replay ? activeIncidents(replay.incidents, seconds) : []

  return (
    <>
      <header className="panel-head">
        <h1>Replay</h1>
        <p>Real service days, train by train, as the MBTA’s records observed them. Pick a train on the map to follow it.</p>
      </header>

      <section className="panel-section">
        <div aria-label="Service day" className="day-list" role="radiogroup">
          {network.replays.map((item, index) => (
            <button aria-checked={index === dayIndex} key={item.file} onClick={() => onDay(index)} role="radio" type="button">
              <strong>{item.label}</strong>
              <span>{formatDate(item.date, 'short')}</span>
              <span className="day-late">{formatPercent(item.late)} late</span>
            </button>
          ))}
        </div>
        <p className="hint">{day.note}</p>
        <dl className="stat-grid is-three">
          <div><dt>Trips</dt><dd>{day.trips.toLocaleString('en-US')}</dd></div>
          <div><dt>Model error</dt><dd>{Math.round(day.maeModel)} s</dd><small>next stop, on average</small></div>
          <div><dt>Rule of thumb</dt><dd>{Math.round(day.maePersistence)} s</dd><small>“stays as late as it is”</small></div>
        </dl>
      </section>

      <section className="panel-section">
        <h2>Train colour</h2>
        <p className="hint">How much longer than planned riders waited for it at its last stop. Taller columns are later trains.</p>
        <ul className="key-list is-inline">
          {DELAY_BANDS.map((band) => <li key={band.label}><i style={{ background: band.color }} />{band.label}</li>)}
        </ul>
      </section>

      <section className="panel-section">
        <h2><TriangleAlert size={14} /> MBTA alerts {active.length ? `in effect at ${formatClock(seconds)}` : 'this day'}</h2>
        {replay ? <AlertList incidents={active.length ? active : replay.incidents} network={network} onSeconds={onSeconds} /> : <p className="hint">Loading…</p>}
      </section>
    </>
  )
}

function AlertList({ incidents, network, onSeconds }: { incidents: Incident[]; network: PreparedNetwork; onSeconds: (seconds: number) => void }) {
  if (!incidents.length) return <p className="hint">No service alerts were raised this day.</p>
  return (
    <ul className="alert-list">
      {incidents.map((incident) => (
        <li key={incident.id}>
          <button onClick={() => onSeconds(incident.start)} type="button">
            <span className="alert-time">{formatClock(incident.start)}</span>
            <span className="alert-body">
              <span className="tag-row">{incident.lines.map((id) => <LineTag id={id} key={id} line={network.lineById.get(id)} />)}</span>
              {incident.text}
            </span>
          </button>
        </li>
      ))}
    </ul>
  )
}

function TripDetail({ network, onBack, onRide, onSeconds, riding, seconds, trip }: {
  network: PreparedNetwork
  onBack: () => void
  onRide: () => void
  onSeconds: (seconds: number) => void
  riding: boolean
  seconds: number
  trip: PreparedTrip
}) {
  const inService = seconds >= trip.start && seconds <= trip.end
  // Both scored on the same stops: those the model made a forecast for.
  const stops = trip.p.flatMap((p, i) => (p === null || i === 0 ? [] : [i]))
  const scored = stops.map((i) => Math.abs(trip.p[i]! - trip.d[i]))
  const persistence = stops.map((i) => Math.abs(trip.d[i - 1] - trip.d[i]))
  const mean = (values: number[]) => values.reduce((sum, v) => sum + v, 0) / (values.length || 1)
  return (
    <>
      <header className="panel-head">
        <button className="back-button" onClick={onBack} type="button"><ArrowLeft size={15} /> Day overview</button>
        <p className="tag-row"><LineTag id={trip.line} line={network.lineById.get(trip.line)} /><span className="chip">Trip {trip.id}</span></p>
        <h1>To {trip.dest}</h1>
        <p>{formatClock(trip.start)}–{formatClock(trip.end)} · {trip.s.length} observed stops</p>
      </header>
      <section className="panel-section">
        <dl className="stat-grid is-three">
          <div><dt>Model error</dt><dd>{scored.length ? `${Math.round(mean(scored))} s` : '—'}</dd><small>on this trip</small></div>
          <div><dt>Rule of thumb</dt><dd>{persistence.length ? `${Math.round(mean(persistence))} s` : '—'}</dd><small>same stops</small></div>
          <div><dt>Status</dt><dd className="is-small">{inService ? 'Running' : seconds < trip.start ? 'Not yet left' : 'Arrived'}</dd></div>
        </dl>
        <button className={`primary-button${riding ? ' is-active' : ''}`} disabled={!inService && !riding} onClick={onRide} type="button">
          {riding ? <RotateCcw size={15} /> : <Train size={15} />}
          {riding ? 'Back to the whole network' : 'Ride along'}
        </button>
      </section>
      <section className="panel-section is-flush">
        <TripStrip network={network} onSeconds={onSeconds} seconds={seconds} trip={trip} />
      </section>
    </>
  )
}
