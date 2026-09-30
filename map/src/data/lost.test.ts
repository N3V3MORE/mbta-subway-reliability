import { describe, expect, it } from 'vitest'
import { platformPlaces, stretchGeometry, stretchPlaces, trackBetween } from './lost'
import type { LostStretch, Pattern } from '../types'

const place = { good: 60, median: 0, p90: 0 }
const stretch = (dir: number, from: string, to: string, trains: number, mean: number, perDay: number): LostStretch =>
  ({ ...place, dir, from, line: 'Red', mean, perDay, terminal: false, to, trains })

describe('time lost', () => {
  it('combines both directions of a stretch, weighting by trains', () => {
    const [davis] = stretchPlaces([stretch(1, 'davis', 'alewife', 300, 90, 10), stretch(0, 'alewife', 'davis', 100, 10, 2)])
    expect(davis.id).toBe('Red:alewife|davis')
    expect(davis.mean).toBe(70)
    expect(davis.perDay).toBe(12)
    expect(davis.parts[0].dir).toBe(1)
  })

  it('keeps each line apart at a shared pair of stations but joins them at a platform', () => {
    const orange = { ...stretch(0, 'north', 'haymarket', 10, 5, 1), line: 'Orange' }
    const green = { ...stretch(0, 'north', 'haymarket', 10, 50, 1), line: 'Green' }
    expect(stretchPlaces([orange, green])).toHaveLength(2)
    const platforms = platformPlaces([{ ...place, dir: 0, line: 'Orange', mean: 4, perDay: 1, station: 'north', trains: 10 },
      { ...place, dir: 0, line: 'Green', mean: 8, perDay: 3, station: 'north', trains: 30 }])
    expect(platforms).toHaveLength(1)
    expect(platforms[0].mean).toBe(7)
    expect(platforms[0].lines).toEqual(['Green', 'Orange'])
  })

  it('cuts the track between two stations', () => {
    const pattern = { coords: [[0, 0], [1, 0], [2, 0], [3, 0]], dist: [0, 100, 200, 300] } as Pick<Pattern, 'coords' | 'dist'>
    expect(trackBetween(pattern, 50, 250)).toEqual([[0.5, 0], [1, 0], [2, 0], [2.5, 0]])
    expect(trackBetween(pattern, 250, 50)).toEqual([[0.5, 0], [1, 0], [2, 0], [2.5, 0]])
  })

  it('draws each stretch once, even where branches share track', () => {
    const base = { coords: [[0, 0], [1, 0], [2, 0]] as [number, number][], dist: [0, 100, 200] }
    const patterns: Pattern[] = [
      { ...base, at: [0, 100, 200], id: 'Red-A', line: 'Red', stations: ['a', 'b', 'c'] },
      { ...base, at: [0, 100], id: 'Red-B', line: 'Red', stations: ['a', 'b'] },
    ]
    expect([...stretchGeometry(patterns).keys()]).toEqual(['Red:a|b', 'Red:b|c'])
  })
})
