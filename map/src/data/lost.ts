import type { LonLat, LostPlatform, LostStretch, Pattern, Period } from '../types'
import type { PreparedNetwork } from './network'
import { pointAt } from './geometry'

/** What the "time lost" view measures a place by. */
export type LostMetric = 'train' | 'day'

export const LOST_METRICS: Record<LostMetric, { detail: string; label: string }> = {
  train: { detail: 'Average seconds a train loses there, beyond a good run', label: 'Per train' },
  day: { detail: 'Minutes lost there on an average day, all trains added together', label: 'Per day' },
}

/** One place on the map, both directions (and, at a station, every line) together. */
export type Place = {
  /** Stretch: `from|to` in sorted order; platform: the station id. */
  id: string
  kind: 'stretch' | 'platform'
  lines: string[]
  /** Train-weighted mean seconds lost per train. */
  mean: number
  perDay: number
  /** Station ids: two for a stretch, one for a platform. */
  stations: string[]
  trains: number
  /** The individual directions, worst per train first. */
  parts: (LostStretch | LostPlatform)[]
}

const stretchId = (a: string, b: string) => (a < b ? `${a}|${b}` : `${b}|${a}`)

function combine(id: string, kind: Place['kind'], stations: string[], parts: (LostStretch | LostPlatform)[]): Place {
  const trains = parts.reduce((sum, p) => sum + p.trains, 0)
  return {
    id, kind, stations, trains,
    lines: [...new Set(parts.map((p) => p.line))].sort(),
    mean: parts.reduce((sum, p) => sum + p.mean * p.trains, 0) / (trains || 1),
    parts: [...parts].sort((left, right) => right.mean - left.mean),
    perDay: parts.reduce((sum, p) => sum + p.perDay, 0),
  }
}

/** Stretches of track, both directions together; stretches on different lines stay apart. */
export function stretchPlaces(stretches: LostStretch[]): Place[] {
  const groups = new Map<string, LostStretch[]>()
  for (const s of stretches) {
    const key = `${s.line}:${stretchId(s.from, s.to)}`
    groups.set(key, [...(groups.get(key) ?? []), s])
  }
  return [...groups.entries()].map(([key, parts]) => combine(key, 'stretch', [parts[0].from, parts[0].to], parts))
}

/** Platforms, every line and direction at a station together. */
export function platformPlaces(platforms: LostPlatform[]): Place[] {
  const groups = new Map<string, LostPlatform[]>()
  for (const p of platforms) groups.set(p.station, [...(groups.get(p.station) ?? []), p])
  return [...groups.entries()].map(([station, parts]) => combine(station, 'platform', [station], parts))
}

export const valueOf = (place: Place, metric: LostMetric) => (metric === 'train' ? place.mean : place.perDay)

/** The track between two distances along a pattern, as a line. */
export function trackBetween(pattern: Pick<Pattern, 'coords' | 'dist'>, from: number, to: number): LonLat[] {
  const [lo, hi] = from < to ? [from, to] : [to, from]
  const inner = pattern.coords.filter((_, i) => pattern.dist[i] > lo && pattern.dist[i] < hi)
  return [pointAt(pattern, lo), ...inner, pointAt(pattern, hi)]
}

/**
 * The track of every stretch between neighbouring stations, keyed by
 * `line:from|to`. Where branches share track (the Green Line trunk, the Red Line
 * to JFK/UMass), the first pattern found supplies it.
 */
export function stretchGeometry(patterns: Pattern[]): Map<string, LonLat[]> {
  const out = new Map<string, LonLat[]>()
  for (const pattern of patterns) {
    const line = pattern.line.split('-')[0]
    for (let i = 0; i + 1 < pattern.stations.length; i += 1) {
      const key = `${line}:${stretchId(pattern.stations[i], pattern.stations[i + 1])}`
      if (!out.has(key)) out.set(key, trackBetween(pattern, pattern.at[i], pattern.at[i + 1]))
    }
  }
  return out
}

/** "Time lost" colours: moving (track) and standing (platforms), validated as a pair on the panel surface. */
export const MOVING = 'var(--lost-moving)'
export const STANDING = 'var(--lost-standing)'

/** Every stretch and platform of a period, or none when it has no time-lost results. */
export function lostPlaces(period: Pick<Period, 'lost'> | undefined): Place[] {
  if (!period?.lost) return []
  return [...stretchPlaces(period.lost.stretches), ...platformPlaces(period.lost.platforms)]
}

/** "Davis – Alewife" for a stretch (its worst direction), "South Station platforms" for a station. */
export function placeName(network: Pick<PreparedNetwork, 'stationById'>, place: Place) {
  const name = (id: string) => network.stationById.get(id)?.name ?? id
  if (place.kind === 'platform') return `${name(place.stations[0])} platforms`
  const worst = place.parts[0] as LostStretch
  return `${name(worst.from)} – ${name(worst.to)}`
}
