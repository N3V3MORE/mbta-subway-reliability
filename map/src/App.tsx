import { Github } from 'lucide-react'
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { MapKey } from './components/MapKey'
import { StationSearch } from './components/StationSearch'
import { Timeline } from './components/Timeline'
import { clusterLabel, lineName, seconds as formatSeconds } from './data/labels'
import { loadNetwork, loadReplay, loadResults, METRICS, type MetricId, type PreparedNetwork } from './data/network'
import { activeIncidents, bandOf, DELAY_BANDS, type PreparedReplay, stopAt, vehiclesAt } from './data/replay'
import { type LostMetric, lostPlaces, placeName, stretchGeometry, valueOf } from './data/lost'
import { REPO_URL } from './links'
import type { LostLayer, Mode, Selection } from './map/NetworkMap'
import { AboutPanel } from './panels/AboutPanel'
import { LostPanel } from './panels/LostPanel'
import { NetworkPanel } from './panels/NetworkPanel'
import { ReplayPanel } from './panels/ReplayPanel'
import { ResultsPanel } from './panels/ResultsPanel'
import { StationsPanel } from './panels/StationsPanel'
import { StoryPanel } from './panels/StoryPanel'
import type { StoryView } from './panels/storyChapters'
import type { LonLat, Results } from './types'

const NetworkMap = lazy(() => import('./map/NetworkMap').then((module) => ({ default: module.NetworkMap })))

const PAGES = [
  { id: 'story', label: 'Story' },
  { id: 'network', label: 'Network' },
  { id: 'replay', label: 'Replay' },
  { id: 'lost', label: 'Time lost' },
  { id: 'results', label: 'All results' },
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
  return { page: known ? known.id : 'story', param: decode(param), sub: decode(sub) }
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
  const [focus, setFocus] = useState<{ center?: LonLat; stationId?: string }>()
  const [lostMetric, setLostMetric] = useState<LostMetric>('train')

  // The story sets what the map shows as each chapter is read.
  const [storyView, setStoryView] = useState<StoryView>({ metric: 'entries', mode: 'network' })
  const inStory = page === 'story'

  // Replay state lives here so leaving the replay and coming back keeps your place.
  // The story can ask for a replay day too.
  const wantedDay = page === 'replay' ? param : inStory && storyView.mode === 'replay' ? storyView.replayDate : undefined
  const dayIndex = Math.max(0, network.replays.findIndex((day) => day.date === wantedDay))
  const day = network.replays[dayIndex]
  const [replay, setReplay] = useState<PreparedReplay | null>(null)
  const [seconds, setSeconds] = useState(MORNING_PEAK)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(SPEEDS[0])
  const tripId = page === 'replay' ? sub : undefined
  const [rideTripId, setRideTripId] = useState<string>()

  const [lastDay, setLastDay] = useState(day?.date)
  useEffect(() => { if (page === 'replay' && day) setLastDay(day.date) }, [day, page])

  const mode: Mode = inStory ? storyView.mode : page === 'replay' ? 'replay' : page === 'lost' ? 'lost' : 'network'
  const shownMetric = inStory ? storyView.metric ?? 'late' : metric
  const shownLostMetric = inStory ? storyView.lostMetric ?? 'train' : lostMetric
  // A chapter showing a replay freezes it at its moment.
  useEffect(() => {
    if (!inStory || storyView.seconds === undefined) return
    setPlaying(false)
    setSeconds(storyView.seconds)
  }, [inStory, storyView])

  // "Time lost": every place of the chosen period, scaled to the worst one on the chosen measure.
  const geometry = useMemo(() => stretchGeometry(network.patterns), [network])
  const lostPeriod = results.periods.find((p) => p.id === periodId && p.lost) ?? results.periods.find((p) => p.lost)
  const places = useMemo(() => lostPlaces(lostPeriod), [lostPeriod])
  const lostLayer = useMemo<LostLayer | undefined>(() => {
    if (!places.length) return undefined
    const scale = (kind: string) => Math.max(...places.filter((p) => p.kind === kind).map((p) => valueOf(p, shownLostMetric)), 1e-9)
    const [maxStretch, maxPlatform] = [scale('stretch'), scale('platform')]
    return {
      platforms: places.flatMap((p) => {
        const station = network.stationById.get(p.stations[0])
        return p.kind === 'platform' && station ? [{ coords: station.coords, id: p.id, t: valueOf(p, shownLostMetric) / maxPlatform }] : []
      }),
      stretches: places.flatMap((p) => {
        const coords = geometry.get(p.id)
        return p.kind === 'stretch' && coords ? [{ coords, id: p.id, t: valueOf(p, shownLostMetric) / maxStretch }] : []
      }),
    }
  }, [geometry, shownLostMetric, network, places])
  const placeId = page === 'lost' ? param : undefined
  const selectPlace = (id: string | undefined, fly = true) => {
    go('lost', id)
    const place = places.find((p) => p.id === id)
    if (!place || !fly) return
    const coords = place.kind === 'stretch' ? geometry.get(place.id) : undefined
    setFocus({ center: coords ? coords[Math.floor(coords.length / 2)] : undefined, stationId: coords ? undefined : place.stations[0] })
  }
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
    : mode === 'lost'
      ? (placeId ? { id: placeId, kind: 'place' } : null)
      : (stationId ? { id: stationId, kind: 'station' } : null)

  // What the map says on hover, in the words of the view it is in.
  const describe = (target: Selection): React.ReactNode => {
    if (target.kind === 'station') {
      const s = network.stationById.get(target.id)
      if (!s) return null
      const value = METRICS[shownMetric].value(s)
      return (
        <>
          <b>{s.name}</b>
          <span>{METRICS[shownMetric].label}: {value === null ? 'no data' : METRICS[shownMetric].format(value)}</span>
          {s.reliability ? <span>{clusterLabel(s.reliability)} station</span> : null}
        </>
      )
    }
    if (target.kind === 'trip') {
      const t = replay?.tripById.get(target.id)
      if (!t) return null
      const stop = stopAt(t.a, seconds)
      return (
        <>
          <b>{lineName(network.lineById.get(t.line), t.line)} to {t.dest}</b>
          <span>{DELAY_BANDS[bandOf(t.l[stop])].label} at {network.stations[t.s[stop]]?.name}</span>
          <span>Click to follow it</span>
        </>
      )
    }
    const place = places.find((p) => p.id === target.id)
    if (!place) return null
    return (
      <>
        <b>{placeName(network, place)}</b>
        <span>{formatSeconds(place.mean)} lost per train, on average</span>
        <span>{Math.round(place.perDay).toLocaleString('en-US')} minutes a day, all trains</span>
      </>
    )
  }

  let panel: React.ReactNode
  if (page === 'story') panel = <StoryPanel network={network} onView={setStoryView} results={results} />
  else if (page === 'network') panel = <NetworkPanel metric={metric} network={network} onMetric={setMetric} onSelect={(id) => selectStation(id)} selected={station} />
  else if (page === 'replay') {
    panel = (
      <ReplayPanel
        dayIndex={dayIndex} network={network} onDay={changeDay} onRide={() => setRideTripId((current) => (current ? undefined : tripId))}
        onSeconds={setSeconds} onTrip={selectTrip} replay={replay} riding={!!rideTripId && rideTripId === tripId} seconds={seconds} trip={trip}
      />
    )
  } else if (page === 'lost') {
    panel = (
      <LostPanel
        metric={lostMetric} network={network} onMetric={setLostMetric} onPeriod={setPeriodId} onSelect={(id) => selectPlace(id)}
        periodId={lostPeriod?.id ?? periodId} results={results} selectedId={placeId}
      />
    )
  } else if (page === 'results') panel = <ResultsPanel network={network} onHighlightLine={setHighlightLine} onPeriod={setPeriodId} periodId={periodId} results={results} />
  else if (page === 'stations') panel = <StationsPanel network={network} onSelect={(id) => selectStation(id)} selectedId={stationId} />
  else panel = <AboutPanel network={network} results={results} />

  return (
    <div className="app">
      <header className="topbar">
        <a className="wordmark" href="#/story" title="The T, as it actually ran">
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
        <a aria-label="Source on GitHub" className="repo-link" href={REPO_URL} rel="noreferrer" target="_blank" title="Source on GitHub">
          <Github aria-hidden="true" size={18} />
        </a>
      </header>

      <div className="workspace" data-page={page}>
        <aside aria-label={PAGES.find((p) => p.id === page)?.label} className="panel" key={page}><div className="panel-inner">{panel}</div></aside>
        <main className="stage">
          <Suspense fallback={<div className="map-loading">Loading the map…</div>}>
            <NetworkMap
              describe={describe}
              focus={focus}
              frame={page}
              highlightLine={page === 'results' ? highlightLine : undefined}
              incidentStations={incidentStations}
              lost={lostLayer}
              metric={shownMetric}
              mode={mode}
              network={network}
              onSelect={(next) => {
                if (mode === 'replay') selectTrip(next?.kind === 'trip' ? next.id : undefined)
                else if (mode === 'lost') selectPlace(next?.kind === 'place' ? next.id : undefined, false)
                else selectStation(next?.kind === 'station' ? next.id : undefined, false)
              }}
              rideTripId={rideTripId}
              scrollZoom={!inStory}
              selection={selection}
              vehicles={vehicles}
            />
          </Suspense>
          <MapKey lostMetric={shownLostMetric} metric={shownMetric} mode={mode} />
          {mode === 'replay' && !inStory ? (
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
