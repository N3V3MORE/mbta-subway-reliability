import type { Line, Network, ReplayDay, Results, Station } from '../types'
import { type PreparedReplay, prepareReplay } from './replay'
import { formatDelay, formatPercent } from './format'

export type PreparedNetwork = Network & {
  lineById: Map<string, Line>
  stationById: Map<string, Station>
}

export type MetricId = 'late' | 'p90' | 'entries'

/** What a station column's height can show. */
export const METRICS: Record<MetricId, { detail: string; format: (value: number) => string; label: string; value: (station: Station) => number | null }> = {
  late: {
    detail: 'Share of arrivals where riders waited over 5 min longer than planned',
    format: formatPercent,
    label: 'Late share',
    value: (station) => station.onTime === null ? null : 1 - station.onTime,
  },
  p90: {
    detail: 'Extra wait on a bad day: 1 arrival in 10 is later',
    format: formatDelay,
    label: 'Bad-day wait',
    value: (station) => station.p90,
  },
  entries: {
    detail: 'Mean faregate entries per day; surface stops are ungated',
    format: (value) => Math.round(value).toLocaleString('en-US'),
    label: 'Daily entries',
    value: (station) => station.entries,
  },
}

/** Reliability clusters from Track B. The pair is validated for colour-blind separation. */
export const RELIABILITY_COLORS: Record<string, string> = {
  'less reliable': '#c98500',
  'more reliable': '#9085e9',
}
export const UNCLUSTERED_COLOR = '#8f877b'

/** The Green Line branches share a trunk, so a stop served by B and C is not an interchange. */
const trunkOf = (lineId: string) => lineId.split('-')[0]

export function isInterchange(station: Station) {
  return new Set(station.lines.map(trunkOf)).size > 1
}

function fetchJson<T>(path: string): Promise<T> {
  // Relative to the page, so the build works wherever it is hosted.
  return fetch(`data/${path}`).then((response) => {
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`)
    return response.json() as Promise<T>
  })
}

let networkPromise: Promise<PreparedNetwork> | undefined

export function loadNetwork() {
  networkPromise ??= fetchJson<Network>('network.json').then((network) => ({
    ...network,
    lineById: new Map(network.lines.map((line) => [line.id, line])),
    stationById: new Map(network.stations.map((station) => [station.id, station])),
  }))
  return networkPromise
}

let resultsPromise: Promise<Results> | undefined

/** The results panel's tables. A missing file is an empty result, not an error: the map still works. */
export function loadResults() {
  resultsPromise ??= fetchJson<Results>('results.json').catch(() => ({ crossSeason: null, periods: [] }))
  return resultsPromise
}

const replays = new Map<string, Promise<PreparedReplay>>()

export function loadReplay(network: PreparedNetwork, day: ReplayDay) {
  let replay = replays.get(day.file)
  if (!replay) {
    replay = fetchJson<Parameters<typeof prepareReplay>[0]>(day.file).then((data) => prepareReplay(data, network))
    // A failed download should be retried next time, not cached.
    replay.catch(() => replays.delete(day.file))
    replays.set(day.file, replay)
  }
  return replay
}
