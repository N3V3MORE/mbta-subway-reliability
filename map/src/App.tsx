import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import { MapPin, Pause, Play, RotateCcw, Train, TriangleAlert, X } from 'lucide-react'
import { StationSearch } from './components/StationSearch'
import { formatClock, formatDate, formatDelay, formatPercent } from './data/format'
import {
  isInterchange, loadNetwork, loadReplay, METRICS, type MetricId, type PreparedNetwork, RELIABILITY_COLORS, UNCLUSTERED_COLOR,
} from './data/network'
import { activeIncidents, DELAY_BANDS, type PreparedReplay, type PreparedTrip, stopAt, vehiclesAt } from './data/replay'
import type { Mode, Selection } from './map/NetworkMap'
import type { Incident, Station } from './types'

const NetworkMap = lazy(() => import('./map/NetworkMap').then((module) => ({ default: module.NetworkMap })))

/** Replay speeds, in service seconds per real second. */
const SPEEDS = [60, 300, 900]
const MORNING_PEAK = 8 * 3_600

type Timing = { end: number; start: number }

function ModeTabs({ mode, onChange }: { mode: Mode; onChange: (mode: Mode) => void }) {
  return (
    <nav aria-label="Map views" className="mode-tabs">
      {(['network', 'replay'] as const).map((item) => (
        <button aria-pressed={mode === item} className={mode === item ? 'is-active' : ''} key={item} onClick={() => onChange(item)} type="button">
          {item.toUpperCase()}
        </button>
      ))}
    </nav>
  )
}

function Swatch({ color }: { color: string }) {
  return <i aria-hidden="true" className="swatch" style={{ backgroundColor: color }} />
}

function NetworkLegend({ metric, network, onMetric }: { metric: MetricId; network: PreparedNetwork; onMetric: (metric: MetricId) => void }) {
  const clusters = [...new Set(network.stations.map((station) => station.reliability).filter((name): name is string => name !== null))]
  return (
    <aside aria-label="Station columns" className="note">
      <span className="mono kicker">STATIONS · {formatDate(network.window[0], 'short')} – {formatDate(network.window[1], 'short')}</span>
      <div aria-label="Column height" className="segmented" role="group">
        {(Object.keys(METRICS) as MetricId[]).map((id) => (
          <button aria-pressed={metric === id} key={id} onClick={() => onMetric(id)} type="button">{METRICS[id].label}</button>
        ))}
      </div>
      <p>Column height: {METRICS[metric].detail.toLowerCase()}.</p>
      <ul aria-label="Column colour: reliability cluster" className="legend">
        {clusters.map((name) => <li key={name}><Swatch color={RELIABILITY_COLORS[name] ?? UNCLUSTERED_COLOR} />{name}</li>)}
      </ul>
    </aside>
  )
}

/** Line colours for an incident's lines, falling back to the text colour. */
function IncidentLines({ incident, network }: { incident: Incident; network: PreparedNetwork }) {
  return <>{incident.lines.map((id) => <Swatch color={network.lineById.get(id)?.color ?? '#f1e9dd'} key={id} />)}</>
}

function IncidentList({ active, network, total }: { active: Incident[]; network: PreparedNetwork; total: number }) {
  return (
    <section aria-label="Incidents in effect" className="incidents">
      <span className="mono kicker"><TriangleAlert size={11} /> MBTA ALERTS IN EFFECT</span>
      {active.length === 0
        ? <p>None at this time. {total} reported this day; the marks above the timeline jump to them.</p>
        : (
          <ul>
            {active.map((incident) => (
              <li key={incident.id}>
                <span className="mono">{formatClock(incident.start)}</span>
                <span><IncidentLines incident={incident} network={network} />{incident.text}</span>
              </li>
            ))}
          </ul>
        )}
    </section>
  )
}

function ReplayNote({ network, replay, dayIndex, seconds }: { dayIndex: number; network: PreparedNetwork; replay: PreparedReplay | null; seconds: number }) {
  const day = network.replays[dayIndex]
  return (
    <aside aria-label="Replay day" aria-live="polite" className="note">
      <span className="mono kicker">OBSERVED · {day.label.toUpperCase()}</span>
      <strong>{formatDate(day.date)}</strong>
      <p>{day.trips.toLocaleString('en-US')} trips · {formatPercent(day.late)} of arrivals over 5 min late for riders. {day.note}</p>
      <p>Next-stop forecast error: model <b>{Math.round(day.maeModel)} s</b>, “stays as late as it is” <b>{Math.round(day.maePersistence)} s</b>.</p>
      <p>Trains are coloured and raised by how much longer than planned riders waited for them at their last stop.</p>
      <ul aria-label="Train colour and column height: extra wait at the last stop" className="legend">
        {DELAY_BANDS.map((band) => <li key={band.label}><Swatch color={band.color} />{band.label}</li>)}
      </ul>
      {replay ? <IncidentList active={activeIncidents(replay.incidents, seconds)} network={network} total={replay.incidents.length} /> : <p>Loading trains…</p>}
    </aside>
  )
}

function ReplayTimeline({ dayIndex, incidents, network, onDay, onPlay, onSeconds, onSpeed, playing, seconds, speed, timing, trains }: {
  dayIndex: number
  incidents: Incident[]
  network: PreparedNetwork
  onDay: (index: number) => void
  onPlay: () => void
  onSeconds: (seconds: number) => void
  onSpeed: () => void
  playing: boolean
  seconds: number
  speed: number
  timing: Timing | null
  trains: number
}) {
  const { end, start } = timing ?? { end: 26 * 3_600, start: 5 * 3_600 }
  return (
    <section aria-label="Replay timeline" className="timeline">
      <div className="timeline-heading">
        <select aria-label="Replay day" onChange={(event) => onDay(Number(event.target.value))} value={dayIndex}>
          {network.replays.map((day, index) => <option key={day.file} value={index}>{day.label} · {day.date}</option>)}
        </select>
        <strong className="mono">{formatClock(seconds)}</strong>
      </div>
      {timing && incidents.length ? (
        <div className="timeline-incidents">
          {incidents.filter((incident) => incident.start >= start && incident.start <= end).map((incident) => (
            <button
              aria-label={`Jump to ${formatClock(incident.start)}: ${incident.text}`}
              key={incident.id}
              onClick={() => onSeconds(incident.start)}
              style={{ left: `${((incident.start - start) / (end - start)) * 100}%` }}
              title={`${formatClock(incident.start)} · ${incident.text}`}
              type="button"
            />
          ))}
        </div>
      ) : null}
      <input aria-label="Replay time" aria-valuetext={formatClock(seconds)} disabled={!timing} max={end} min={start} onChange={(event) => onSeconds(Number(event.target.value))} step={30} type="range" value={seconds} />
      <div aria-hidden="true" className="timeline-ticks">
        {[0, 1, 2, 3, 4].map((i) => <span key={i}>{formatClock(start + ((end - start) * i) / 4)}</span>)}
      </div>
      <div className="timeline-row">
        <button aria-label={playing ? 'Pause replay' : 'Play replay'} className="icon-button" disabled={!timing} onClick={onPlay} type="button">
          {playing ? <Pause fill="currentColor" size={16} /> : <Play fill="currentColor" size={16} />}
        </button>
        <button aria-label="Change replay speed" className="speed-button mono" onClick={onSpeed} type="button">{speed / 60} min/s</button>
        <p>{timing ? `${trains} trains in service. Positions are interpolated only between observed stops.` : 'Loading trains…'}</p>
      </div>
      {timing ? <CurrentIncident incidents={incidents} network={network} seconds={seconds} /> : null}
    </section>
  )
}

/** The latest incident in effect, for screens too narrow for the replay note. */
function CurrentIncident({ incidents, network, seconds }: { incidents: Incident[]; network: PreparedNetwork; seconds: number }) {
  const [latest] = activeIncidents(incidents, seconds)
  if (!latest) return null
  return (
    <p className="timeline-incident">
      <TriangleAlert size={12} /> <span className="mono">{formatClock(latest.start)}</span> <IncidentLines incident={latest} network={network} />{latest.text}
    </p>
  )
}

function DemandSparkline({ profile }: { profile: number[] }) {
  const max = Math.max(...profile)
  const peak = profile.indexOf(max)
  const points = profile.map((share, i) => `${(i / (profile.length - 1)) * 100},${29 - (share / max) * 27}`).join(' ')
  return (
    <figure className="sparkline">
      <figcaption>Entries through the day · busiest {formatClock(peak * 1_800)}–{formatClock((peak + 1) * 1_800)}</figcaption>
      <svg aria-hidden="true" preserveAspectRatio="none" viewBox="0 0 100 30"><polyline points={points} /></svg>
      <div className="timeline-ticks"><span>00:00</span><span>12:00</span><span>24:00</span></div>
    </figure>
  )
}

function StationPanel({ network, onClose, station }: { network: PreparedNetwork; onClose: () => void; station: Station }) {
  const late = METRICS.late.value(station)
  const rows: [string, React.ReactNode][] = [
    ['Late', late === null ? '—' : `${formatPercent(late)} of arrivals over 5 min late`],
    ['Typical', station.median === null ? '—' : `${formatDelay(station.median)} extra wait (median)`],
    ['Bad day', station.p90 === null ? '—' : `${formatDelay(station.p90)} extra wait (1 in 10 worse)`],
    ['Entries', station.entries === null ? 'Not gated' : `${METRICS.entries.format(station.entries)} a day`],
    ['Reliability', station.reliability ? <><Swatch color={RELIABILITY_COLORS[station.reliability] ?? UNCLUSTERED_COLOR} />{station.reliability}</> : '—'],
    ['Demand', station.demand ?? '—'],
  ]
  return (
    <aside aria-label={`Station details for ${station.name}`} className="context-panel">
      <button aria-label="Close station details" className="close-button" onClick={onClose} type="button"><X size={16} /></button>
      <p className="panel-kicker"><MapPin size={14} /> {isInterchange(station) ? 'Interchange' : 'Station'}</p>
      <h2>{station.name}</h2>
      <ul className="line-list">
        {station.lines.map((id) => {
          const line = network.lineById.get(id)
          return line ? <li key={id}><span style={{ backgroundColor: line.color }} />{line.name}</li> : null
        })}
      </ul>
      <dl>{rows.map(([term, value]) => <div key={term}><dt>{term}</dt><dd>{value}</dd></div>)}</dl>
      {station.profile ? <DemandSparkline profile={station.profile} /> : null}
    </aside>
  )
}

function TripPanel({ network, onClose, onRide, riding, seconds, trip }: {
  network: PreparedNetwork
  onClose: () => void
  onRide: () => void
  riding: boolean
  seconds: number
  trip: PreparedTrip
}) {
  const line = network.lineById.get(trip.line)
  const name = (stop: number) => network.stations[trip.s[stop]].name
  const stop = stopAt(trip.a, seconds)
  const next = stop + 1 < trip.s.length ? stop + 1 : undefined
  const inService = seconds >= trip.start && seconds <= trip.end
  const forecast = next === undefined ? null : trip.p[next]
  const off = (value: number) => `off by ${formatDelay(Math.abs(value - trip.d[next!])).slice(1)}`
  return (
    <aside aria-label={`Trip ${trip.id}`} className="context-panel trip-panel" style={{ '--line-color': line?.color } as React.CSSProperties}>
      <button aria-label="Close trip details" className="close-button" onClick={onClose} type="button"><X size={16} /></button>
      <p className="panel-kicker"><Train size={14} /> Observed trip {trip.id}</p>
      <h2>{line?.name ?? trip.line}</h2>
      <dl>
        <div><dt>Toward</dt><dd>{trip.dest}</dd></div>
        <div><dt>Late now</dt><dd>{inService ? `${formatDelay(trip.l[stop])} extra wait at ${name(stop)}` : 'Not in service at this time'}</dd></div>
        {inService && next !== undefined ? (
          <>
            <div><dt>Next</dt><dd>{name(next)}, arrived {formatClock(trip.a[next])} at <b>{formatDelay(trip.d[next])}</b> vs the timetable</dd></div>
            <div><dt>Model</dt><dd>{forecast === null ? 'No forecast for this stop' : `${formatDelay(forecast)}, ${off(forecast)}`}</dd></div>
            <div><dt>Rule of thumb</dt><dd>{formatDelay(trip.d[stop])}, {off(trip.d[stop])}</dd></div>
          </>
        ) : null}
      </dl>
      <button className={`ride-button ${riding ? 'is-active' : ''}`} disabled={!inService && !riding} onClick={onRide} type="button">
        {riding ? <RotateCcw size={15} /> : <Train size={15} />}
        {riding ? 'Return to overview' : 'Ride'}
      </button>
    </aside>
  )
}

function Explorer({ network }: { network: PreparedNetwork }) {
  const [mode, setMode] = useState<Mode>('network')
  const [metric, setMetric] = useState<MetricId>('late')
  const [dayIndex, setDayIndex] = useState(0)
  const [replay, setReplay] = useState<PreparedReplay | null>(null)
  const [seconds, setSeconds] = useState(MORNING_PEAK)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(SPEEDS[0])
  const [selection, setSelection] = useState<Selection | null>(null)
  const [focus, setFocus] = useState<{ stationId: string }>()
  const [rideTripId, setRideTripId] = useState<string>()
  const day = network.replays[dayIndex]
  const timing = useMemo(() => replay ? { end: replay.end, start: replay.start } : null, [replay])

  useEffect(() => {
    if (mode !== 'replay' || !day) return undefined
    let cancelled = false
    setReplay(null)
    loadReplay(network, day).then((loaded) => {
      if (cancelled) return
      setReplay(loaded)
      setSeconds((current) => Math.min(loaded.end, Math.max(loaded.start, current)))
    }, () => undefined)
    return () => { cancelled = true }
  }, [day, mode, network])

  // One animation frame, one step of service time: smooth at any speed.
  useEffect(() => {
    if (!playing || !timing) return undefined
    let frame = 0
    let last = performance.now()
    const tick = (now: number) => {
      const step = (Math.min(250, now - last) / 1_000) * speed
      last = now
      setSeconds((current) => current + step > timing.end ? timing.start : current + step)
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [playing, speed, timing])

  const vehicles = useMemo(() => mode === 'replay' && replay ? vehiclesAt(replay, network.patterns, seconds) : [], [mode, network, replay, seconds])
  // Stations under an alert in effect now, as a key so the map redraws only when the set changes.
  const incidentKey = mode === 'replay' && replay
    ? [...new Set(activeIncidents(replay.incidents, seconds).flatMap((incident) => incident.stations.map((i) => network.stations[i]?.id ?? '')))]
        .filter(Boolean).sort().join(',')
    : ''
  const incidentStations = useMemo(() => (incidentKey ? incidentKey.split(',') : []), [incidentKey])
  const station = selection?.kind === 'station' ? network.stationById.get(selection.id) : undefined
  const trip = selection?.kind === 'trip' ? replay?.tripById.get(selection.id) : undefined

  const reset = () => { setSelection(null); setFocus(undefined); setRideTripId(undefined) }
  const changeMode = (next: Mode) => { reset(); setPlaying(false); setMode(next) }
  const changeDay = (index: number) => { reset(); setDayIndex(index) }

  return (
    <main className="shell">
      <Suspense fallback={<div className="map-loading">Loading the map…</div>}>
        <NetworkMap
          focus={focus}
          incidentStations={incidentStations}
          metric={metric}
          mode={mode}
          network={network}
          onSelect={(next) => { setSelection(next); setFocus(undefined); setRideTripId(undefined) }}
          rideTripId={rideTripId}
          selection={selection}
          vehicles={vehicles}
        />
      </Suspense>

      <header className="app-header">
        <div><strong>MINI BOSTON</strong><span>The T, as it actually ran</span></div>
        <ModeTabs mode={mode} onChange={changeMode} />
      </header>

      <StationSearch lineById={network.lineById} onSelect={(stationId) => { setSelection({ id: stationId, kind: 'station' }); setRideTripId(undefined); setFocus({ stationId }) }} stations={network.stations} />

      {mode === 'network' ? <NetworkLegend metric={metric} network={network} onMetric={setMetric} /> : null}
      {mode === 'replay' && day ? (
        <>
          <ReplayNote dayIndex={dayIndex} network={network} replay={replay} seconds={seconds} />
          <ReplayTimeline
            dayIndex={dayIndex} incidents={replay?.incidents ?? []} network={network} onDay={changeDay} onPlay={() => setPlaying((value) => !value)}
            onSeconds={setSeconds} onSpeed={() => setSpeed((value) => SPEEDS[(SPEEDS.indexOf(value) + 1) % SPEEDS.length])}
            playing={playing} seconds={seconds} speed={speed} timing={timing} trains={vehicles.length}
          />
        </>
      ) : null}

      {station ? <StationPanel network={network} onClose={reset} station={station} /> : null}
      {trip ? <TripPanel network={network} onClose={reset} onRide={() => setRideTripId((current) => current ? undefined : trip.id)} riding={rideTripId === trip.id} seconds={seconds} trip={trip} /> : null}
    </main>
  )
}

export default function App() {
  const [network, setNetwork] = useState<PreparedNetwork>()
  const [error, setError] = useState<string>()
  useEffect(() => { loadNetwork().then(setNetwork, (reason: Error) => setError(reason.message)) }, [])

  if (!network) {
    return (
      <main className="shell">
        <p className="map-loading">{error ? `Could not load the network (${error}). Run: python -m mbta_ds.cli export-map` : 'Loading the network…'}</p>
      </main>
    )
  }
  return <Explorer network={network} />
}
