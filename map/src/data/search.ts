import type { Station } from '../types'

function normalize(value: string) {
  return value.normalize('NFKD').replace(/[̀-ͯ]/g, '').toLowerCase()
}

/** Stations whose name contains every word of the query: exact, then prefix, then the rest. */
export function searchStations(stations: Station[], query: string): Station[] {
  const words = normalize(query).match(/[a-z0-9]+/g) ?? []
  const compactQuery = words.join('')
  if (!compactQuery) return []

  return stations.flatMap((station) => {
    const name = normalize(station.name).replace(/[^a-z0-9]/g, '')
    if (!words.every((word) => name.includes(word))) return []
    const rank = name === compactQuery ? 0 : name.startsWith(compactQuery) ? 1 : 2
    return [{ rank, station }]
  })
    .sort((left, right) => left.rank - right.rank || left.station.name.localeCompare(right.station.name, 'en'))
    .map(({ station }) => station)
}
