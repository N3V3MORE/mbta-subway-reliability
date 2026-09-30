import type { Incident, LonLat, Network, Pattern, Replay, Trip } from '../types'
import { pointAt } from './geometry'

export type PreparedTrip = Trip & { end: number; leave: number[]; metres: number[]; start: number }

export type PreparedReplay = Omit<Replay, 'incidents' | 'trips'> & {
  end: number
  /** Sorted by start. Files exported before incidents existed have none. */
  incidents: Incident[]
  start: number
  /** Trips sorted by first arrival, so a scan can stop at the first future trip. */
  trips: PreparedTrip[]
  tripById: Map<string, PreparedTrip>
}

export type Vehicle = { coords: LonLat; lateness: number; stop: number; trip: PreparedTrip }

/**
 * Lateness bands, matching the report's: how much longer than planned riders
 * waited for the train. Colours are a validated one-hue ramp.
 */
export const DELAY_BANDS = [
  { color: '#fee7f3', label: 'On time', min: -Infinity },
  { color: '#f2b0d5', label: '1–5 min late', min: 60 },
  { color: '#e574b8', label: '5–10 min late', min: 300 },
  { color: '#d0399c', label: '10+ min late', min: 600 },
] as const

export function bandOf(lateness: number) {
  // Right-closed like the report: exactly 5 minutes is still "1–5 min late".
  let band = 0
  while (band < DELAY_BANDS.length - 1 && lateness > DELAY_BANDS[band + 1].min) band += 1
  return band
}

/** Resolve each stop to metres along its branch once, instead of on every frame. */
export function prepareReplay(replay: Replay, network: Pick<Network, 'patterns' | 'stations'>): PreparedReplay {
  const along = network.patterns.map((pattern) => new Map(pattern.stations.map((id, i) => [id, pattern.at[i]])))
  const trips: PreparedTrip[] = []
  for (const trip of replay.trips) {
    const metres = trip.s.map((s) => along[trip.pattern]?.get(network.stations[s]?.id) ?? Number.NaN)
    // A file exported against a different network cannot be drawn; skip, don't guess.
    if (trip.a.length < 2 || !metres.every(Number.isFinite)) continue
    trips.push({ ...trip, end: trip.a[trip.a.length - 1], leave: trip.a.map((a, i) => a + trip.w[i]), metres, start: trip.a[0] })
  }
  trips.sort((left, right) => left.start - right.start)
  return {
    ...replay,
    incidents: [...(replay.incidents ?? [])].sort((left, right) => left.start - right.start),
    end: Math.max(...trips.map((trip) => trip.end)),
    start: trips[0]?.start ?? 0,
    tripById: new Map(trips.map((trip) => [trip.id, trip])),
    trips,
  }
}

/** Index of the last stop the trip reached by time `t` (its arrivals are increasing). */
export function stopAt(arrivals: number[], t: number) {
  let lo = 0
  let hi = arrivals.length - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (arrivals[mid] <= t) lo = mid
    else hi = mid - 1
  }
  return lo
}

/**
 * Where every train in service was at time `t`. A train stands at a stop until it
 * left, then moves at constant speed to its next observed stop. Its lateness is
 * the one measured at the last stop it reached: what a rider would have known then.
 */
export function vehiclesAt(replay: PreparedReplay, patterns: Pattern[], t: number): Vehicle[] {
  const vehicles: Vehicle[] = []
  for (const trip of replay.trips) {
    if (trip.start > t) break
    if (trip.end < t) continue
    const stop = stopAt(trip.a, t)
    let metres = trip.metres[stop]
    if (stop < trip.a.length - 1 && t > trip.leave[stop]) {
      const phase = (t - trip.leave[stop]) / (trip.a[stop + 1] - trip.leave[stop])
      metres += (trip.metres[stop + 1] - metres) * phase
    }
    vehicles.push({ coords: pointAt(patterns[trip.pattern], metres), lateness: trip.l[stop], stop, trip })
  }
  return vehicles
}

/** How long an alert that was never closed is shown for, in seconds. */
export const OPEN_INCIDENT_SECONDS = 3_600

/** The incident's end, or an hour after it began when the archive has none. */
export const incidentEnd = (incident: Incident) => incident.end ?? incident.start + OPEN_INCIDENT_SECONDS

/** Incidents in effect at time `t`, most recent first. */
export function activeIncidents(incidents: Incident[], t: number): Incident[] {
  return incidents.filter((incident) => incident.start <= t && t < incidentEnd(incident)).reverse()
}
