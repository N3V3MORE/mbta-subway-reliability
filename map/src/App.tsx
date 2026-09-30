import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { StationSearch } from './components/StationSearch'
import { Timeline } from './components/Timeline'
import { loadNetwork, loadReplay, loadResults, type MetricId, type PreparedNetwork } from './data/network'
import { activeIncidents, type PreparedReplay, vehiclesAt } from './data/replay'
import type { Mode, Selection } from './map/NetworkMap'
import { AboutPanel } from './panels/AboutPanel'
import { NetworkPanel } from './panels/NetworkPanel'
import { ReplayPanel } from './panels/ReplayPanel'
import { ResultsPanel } from './panels/ResultsPanel'
import { StationsPanel } from './panels/StationsPanel'
import type { Results } from './types'

const NetworkMap = lazy(() => import('./map/NetworkMap').then((module) => ({ default: module.NetworkMap })))

const PAGES = [
  { id: 'network', label: 'Network' },
  { id: 'replay', label: 'Replay' },
  { id: 'results', label: 'Results' },
  { id: 'stations', label: 'Stations' },
  { id: 'about', label: 'About' },
] as const
type Page = (typeof PAGES)[number]['id']

/** Replay speeds, in service seconds per real second. */
const SPEEDS = [60, 300, 900]
const MORNING_PEAK = 8 * 3_600

/** `#/page/param/sub`: every view has a link (a station, a replay day and train), and the back button works. */
function readHash(): { page: Page; param?: string; sub?: string } {
  const [, page, param, sub] = window.location.hash.split('/')
  const known = PAGES.find((p) => p.id === page)
  const decode = (value?: string) => (value ? decodeURIComponent(value) : undefined)
  return { page: known ? known.id : 'network', param: decode(param), sub: decode(sub) }
}

function useRoute() {
  const [route, setRoute] = useState(readHash)
  useEffect(() => {
    const onHash = () => setRoute(readHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  const go = useCallback((page: Page, param?: string, sub?: string) => {
    const hash = `#/${page}${param ? `/${encodeURIComponent(param)}` : ''}${param && sub ? `/${encodeURIComponent(sub)}` : ''}`
    if (window.location.hash !== hash) window.location.hash = hash
  }, [])
  return [route, go] as const
}

function Workspace({ network, results }: { network: PreparedNetwork; results: Results }) {
  const [{ page, param, sub }, go] = useRoute()
  const [metric, setMetric] = useState<MetricId>('late')
  const [periodId, setPeriodId] = useState(results.periods[0]?.id ?? 'spring')
  const [highlightLine, setHighlightLine] = useState<string>()
  const [focus, setFocus] = useState<{ stationId: string }>()

  // Replay state lives here so leaving the replay and coming back keeps your place.
  const dayIndex = Math.max(0, network.replays.findIndex((day) => day.date === (page === 'replay' ? param : undefined)))
  const day = network.replays[dayIndex]
  const [replay, setReplay] = useState<PreparedReplay | null>(null)
  const [seconds, setSeconds] = useState(MORNING_PEAK)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(SPEEDS[0])
  const tripId = page === 'replay' ? sub : undefined
  const [rideTripId, setRideTripId] = useState<string>()

  const [lastDay, setLastDay] = useState(day?.date)
  useEffect(() => { if (page === 'replay' && day) setLastDay(day.date) }, [day, page])

  const mode: Mode = page === 'replay' ? 'replay' : 'network'
  const stationId = page === 'network' || page === 'stations' ? param : undefined
  const station = stationId ? network.stationById.get(stationId) : undefined

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
  const timing = useMemo(() => (replay ? { end: replay.end, start: replay.start } : null), [replay])
  useEffect(() => {
    if (!playing || !timing || mode !== 'replay') return undefined
    let frame = 0
    let last = performance.now()
    const tick = (now: number) => {
      const step = (Math.min(250, now - last) / 1_000) * speed
      last = now
      setSeconds((current) => (current + step > timing.end ? timing.start : current + step))
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [mode, playing, speed, timing])

  const vehicles = useMemo(() => (mode === 'replay' && replay ? vehiclesAt(replay, network.patterns, seconds) : []), [mode, network, replay, seconds])
  // Stations under an alert in effect now, as a key so the map redraws only when the set changes.
  const incidentKey = mode === 'replay' && replay
    ? [...new Set(activeIncidents(replay.incidents, seconds).flatMap((incident) => incident.stations.map((i) => network.stations[i]?.id ?? '')))]
        .filter(Boolean).sort().join(',')
    : ''
  const incidentStations = useMemo(() => (incidentKey ? incidentKey.split(',') : []), [incidentKey])
  const trip = tripId ? replay?.tripById.get(tripId) : undefined

  // A station reached by link, search, list or the back button is flown to; one
  // clicked on the map is already in view.
  const flyRef = useRef(true)
  useEffect(() => {
    if (stationId && flyRef.current) setFocus({ stationId })
    flyRef.current = true
  }, [stationId])
  const selectStation = (id: string | undefined, fly = true) => {
    flyRef.current = fly
    go(page === 'stations' ? 'stations' : 'network', id)
  }
  const selectTrip = (id: string | undefined) => { go('replay', day?.date, id); setRideTripId(undefined) }
  const changeDay = (index: number) => { setRideTripId(undefined); setPlaying(false); go('replay', network.replays[index].date) }

  const selection: Selection | null = mode === 'replay'
    ? (tripId ? { id: tripId, kind: 'trip' } : null)
    : (stationId ? { id: stationId, kind: 'station' } : null)

  let panel: React.ReactNode
  if (page === 'network') panel = <NetworkPanel metric={metric} network={network} onMetric={setMetric} onSelect={(id) => selectStation(id)} selected={station} />
  else if (page === 'replay') {
    panel = (
      <ReplayPanel
        dayIndex={dayIndex} network={network} onDay={changeDay} onRide={() => setRideTripId((current) => (current ? undefined : tripId))}
        onSeconds={setSeconds} onTrip={selectTrip} replay={replay} riding={!!rideTripId && rideTripId === tripId} seconds={seconds} trip={trip}
      />
    )
  } else if (page === 'results') panel = <ResultsPanel network={network} onHighlightLine={setHighlightLine} onPeriod={setPeriodId} periodId={periodId} results={results} />
  else if (page === 'stations') panel = <StationsPanel network={network} onSelect={(id) => selectStation(id)} selectedId={stationId} />
  else panel = <AboutPanel network={network} results={results} />

  return (
    <div className="app">
      <header className="topbar">
        <a className="wordmark" href="#/network" title="The T, as it actually ran">
          <svg aria-hidden="true" viewBox="0 0 20 20">
            <rect fill="var(--series-persistence)" height="6" rx="1" width="3.6" x="2" y="12" />
            <rect fill="var(--text)" height="10" rx="1" width="3.6" x="8.2" y="8" />
            <rect fill="var(--accent)" height="14" rx="1" width="3.6" x="14.4" y="4" />
          </svg>
          Mini Boston
        </a>
        <nav aria-label="Views" className="nav">
          {PAGES.map((item) => (
            <a aria-current={page === item.id ? 'page' : undefined} href={`#/${item.id}${item.id === 'replay' && lastDay ? `/${lastDay}` : ''}`} key={item.id}
              onClick={() => { if (item.id !== 'replay') setPlaying(false) }}>
              {item.label}
            </a>
          ))}
        </nav>
        <StationSearch lineById={network.lineById} onSelect={(id) => { go('network', id); setFocus({ stationId: id }) }} stations={network.stations} />
      </header>

      <div className="workspace" data-page={page}>
        <aside aria-label={PAGES.find((p) => p.id === page)?.label} className="panel" key={page}>{panel}</aside>
        <main className="stage">
          <Suspense fallback={<div className="map-loading">Loading the map…</div>}>
            <NetworkMap
              focus={focus}
              highlightLine={page === 'results' ? highlightLine : undefined}
              incidentStations={incidentStations}
              metric={metric}
              mode={mode}
              network={network}
              onSelect={(next) => {
                if (mode === 'replay') selectTrip(next?.kind === 'trip' ? next.id : undefined)
                else selectStation(next?.kind === 'station' ? next.id : undefined, false)
              }}
              rideTripId={rideTripId}
              selection={selection}
              vehicles={vehicles}
            />
          </Suspense>
          {mode === 'replay' ? (
            <Timeline
              incidents={replay?.incidents ?? []} onPlay={() => setPlaying((value) => !value)} onSeconds={setSeconds}
              onSpeed={() => setSpeed((value) => SPEEDS[(SPEEDS.indexOf(value) + 1) % SPEEDS.length])}
              playing={playing} replay={replay} seconds={seconds} speed={speed} trains={vehicles.length}
            />
          ) : null}
        </main>
      </div>
    </div>
  )
}

export default function App() {
  const [data, setData] = useState<{ network: PreparedNetwork; results: Results }>()
  const [error, setError] = useState<string>()
  useEffect(() => {
    Promise.all([loadNetwork(), loadResults()]).then(([network, results]) => setData({ network, results }), (reason: Error) => setError(reason.message))
  }, [])

  if (!data) {
    return (
      <div className="app">
        <p className="map-loading">{error ? `Could not load the network (${error}). Run: python -m mbta_ds.cli export-map` : 'Loading the network…'}</p>
      </div>
    )
  }
  return <Workspace network={data.network} results={data.results} />
}
