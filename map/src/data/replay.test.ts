import { describe, expect, it } from 'vitest'
import type { Pattern, Replay, Station } from '../types'
import { columnRing, pointAt } from './geometry'
import { bandOf, prepareReplay, stopAt, vehiclesAt } from './replay'
import { searchStations } from './search'
import { formatClock, formatDelay } from './format'

const station = (id: string, name: string): Station => ({
  coords: [0, 0], demand: null, entries: null, id, lines: ['Red'], median: null, name, onTime: null, p90: null, profile: null, reliability: null,
})

// A straight 1 km track with stations at 0, 400 and 1000 m.
const pattern: Pattern = {
  at: [0, 400, 1_000], coords: [[0, 0], [1, 0], [3, 0]], dist: [0, 500, 1_000], id: 'Red-A', line: 'Red', stations: ['a', 'b', 'c'],
}
const network = { patterns: [pattern], stations: [station('a', 'Alewife'), station('b', 'Davis'), station('c', 'Porter')] }

const replay: Replay = {
  date: '2026-06-10',
  label: 'test',
  summary: { late: 0, maeModel: 0, maePersistence: 0, stops: 6, trips: 2 },
  trips: [
    // Listed out of order on purpose: preparation must sort by first arrival.
    { a: [2_000, 2_100], d: [0, 30], dest: 'Porter', id: 'late', l: [0, 10], line: 'Red', p: [null, 20], pattern: 0, s: [0, 2], w: [0, 0] },
    { a: [1_000, 1_100, 1_300], d: [0, 90, 400], dest: 'Porter', id: 'early', l: [0, 45, 320], line: 'Red', p: [null, 60, 380], pattern: 0, s: [0, 1, 2], w: [20, 50, 0] },
  ],
}

describe('pointAt', () => {
  it('interpolates by distance, not by vertex count', () => {
    expect(pointAt(pattern, 250)).toEqual([0.5, 0])
    expect(pointAt(pattern, 750)).toEqual([2, 0])
  })

  it('clamps to the ends of the track', () => {
    expect(pointAt(pattern, -10)).toEqual([0, 0])
    expect(pointAt(pattern, 5_000)).toEqual([3, 0])
  })
})

describe('columnRing', () => {
  it('is a closed ring of the requested radius', () => {
    const ring = columnRing([0, 0], 110_540)
    expect(ring[0]).toEqual(ring[ring.length - 1])
    expect(Math.max(...ring.map(([, lat]) => lat))).toBeCloseTo(1, 6)
  })
})

describe('replay', () => {
  const prepared = prepareReplay(replay, network)

  it('sorts trips by start and records the service span', () => {
    expect(prepared.trips.map((trip) => trip.id)).toEqual(['early', 'late'])
    expect([prepared.start, prepared.end]).toEqual([1_000, 2_100])
  })

  it('skips trips whose stations are not on their pattern', () => {
    const stale = { ...replay, trips: [{ ...replay.trips[0], s: [0, 7] }] }
    expect(prepareReplay(stale, network).trips).toHaveLength(0)
  })

  it('finds the last stop reached', () => {
    expect(stopAt([10, 20, 30], 10)).toBe(0)
    expect(stopAt([10, 20, 30], 25)).toBe(1)
    expect(stopAt([10, 20, 30], 99)).toBe(2)
  })

  it('holds a train at a stop until it leaves, then moves at constant speed', () => {
    const at = (t: number) => vehiclesAt(prepared, network.patterns, t).map(({ coords, trip }) => [trip.id, coords[0]])
    expect(at(999)).toEqual([])
    expect(at(1_010)).toEqual([['early', 0]]) // dwelling at Alewife
    expect(at(1_060)).toEqual([['early', 0.4]]) // halfway to Davis (200 m)
    expect(at(1_140)).toEqual([['early', 0.8]]) // dwelling at Davis
    expect(at(1_225)).toEqual([['early', 1.8]]) // halfway to Porter (700 m)
    expect(at(1_500)).toEqual([]) // out of service between the two trips
  })

  it('reports the lateness measured at the last stop reached', () => {
    const [vehicle] = vehiclesAt(prepared, network.patterns, 1_200)
    expect([vehicle.stop, vehicle.lateness]).toEqual([1, 45])
  })
})

describe('formatting and bands', () => {
  it('bands lateness like the report (right-closed: exactly 5 min is not late)', () => {
    expect([-30, 60, 61, 300, 301, 600, 601].map(bandOf)).toEqual([0, 0, 1, 1, 2, 2, 3])
  })

  it('formats clock times and signed delays', () => {
    expect(formatClock(8 * 3_600 + 5 * 60 + 59)).toBe('08:05')
    expect(formatClock(25 * 3_600)).toBe('01:00')
    expect(formatDelay(185)).toBe('+3:05')
    expect(formatDelay(-20)).toBe('−0:20')
  })
})

describe('searchStations', () => {
  const stations = [station('1', 'Park Street'), station('2', 'Street Park'), station('3', 'Parker')]

  it('ranks exact, then prefix, then other matches', () => {
    expect(searchStations(stations, 'park').map((s) => s.name)).toEqual(['Park Street', 'Parker', 'Street Park'])
    expect(searchStations(stations, '   ')).toEqual([])
  })
})
