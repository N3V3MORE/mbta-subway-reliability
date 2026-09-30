import type { LonLat, Pattern } from '../types'

/** The point `metres` along a pattern's track, found by binary search. */
export function pointAt({ coords, dist }: Pick<Pattern, 'coords' | 'dist'>, metres: number): LonLat {
  const target = Math.min(Math.max(metres, 0), dist[dist.length - 1])
  let lo = 1
  let hi = dist.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (dist[mid] < target) lo = mid + 1
    else hi = mid
  }
  const span = dist[lo] - dist[lo - 1]
  const phase = span > 0 ? (target - dist[lo - 1]) / span : 0
  const [x0, y0] = coords[lo - 1]
  const [x1, y1] = coords[lo]
  return [x0 + (x1 - x0) * phase, y0 + (y1 - y0) * phase]
}

const SIDES = 12
const UNIT = Array.from({ length: SIDES + 1 }, (_, i) => {
  const angle = (2 * Math.PI * (i % SIDES)) / SIDES
  return [Math.cos(angle), Math.sin(angle)] as const
})

/** A closed ring approximating a circle of `radius` metres, for 3D columns. */
export function columnRing([lon, lat]: LonLat, radius: number): LonLat[] {
  const dLon = radius / (111_320 * Math.cos((lat * Math.PI) / 180))
  const dLat = radius / 110_540
  return UNIT.map(([x, y]) => [lon + x * dLon, lat + y * dLat])
}
