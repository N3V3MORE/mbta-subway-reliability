import { useMemo, useState } from 'react'
import { ArrowDown, ArrowUp, Search } from 'lucide-react'
import { formatDelay, formatPercent } from '../data/format'
import { METRICS, type PreparedNetwork, RELIABILITY_COLORS, UNCLUSTERED_COLOR } from '../data/network'
import { searchStations } from '../data/search'
import { LineDots } from '../components/LineTag'
import { clusterLabel, lineName } from '../data/labels'
import type { Station } from '../types'

type Column = { get: (s: Station) => number | string | null; key: string; label: string; numeric?: boolean; render: (s: Station) => React.ReactNode }

const COLUMNS: Column[] = [
  { get: (s) => s.name, key: 'name', label: 'Station', render: (s) => s.name },
  { get: (s) => METRICS.late.value(s), key: 'late', label: 'Late', numeric: true, render: (s) => (s.onTime === null ? '—' : formatPercent(1 - s.onTime)) },
  { get: (s) => s.median, key: 'median', label: 'Typical', numeric: true, render: (s) => (s.median === null ? '—' : formatDelay(s.median)) },
  { get: (s) => s.p90, key: 'p90', label: 'Bad day', numeric: true, render: (s) => (s.p90 === null ? '—' : formatDelay(s.p90)) },
  { get: (s) => s.entries, key: 'entries', label: 'Entries/day', numeric: true, render: (s) => (s.entries === null ? 'Ungated' : METRICS.entries.format(s.entries)) },
  {
    get: (s) => s.reliability, key: 'reliability', label: 'Reliability',
    render: (s) => <><i className="swatch" style={{ background: RELIABILITY_COLORS[s.reliability ?? ''] ?? UNCLUSTERED_COLOR }} />{clusterLabel(s.reliability) ?? '—'}</>,
  },
  { get: (s) => s.demand, key: 'demand', label: 'Demand', render: (s) => clusterLabel(s.demand) ?? '—' },
]

/** Trunk lines for the filter: the Green Line branches filter together. */
const TRUNKS = ['Red', 'Orange', 'Blue', 'Green', 'Mattapan']

export function StationsPanel({ network, onSelect, selectedId }: { network: PreparedNetwork; onSelect: (id: string) => void; selectedId?: string }) {
  const [query, setQuery] = useState('')
  const [trunk, setTrunk] = useState<string>()
  const [sort, setSort] = useState<{ desc: boolean; key: string }>({ desc: true, key: 'late' })

  const rows = useMemo(() => {
    const matched = query.trim() ? searchStations(network.stations, query) : network.stations
    const filtered = trunk ? matched.filter((s) => s.lines.some((id) => id.split('-')[0] === trunk)) : matched
    const column = COLUMNS.find((c) => c.key === sort.key)!
    return [...filtered].sort((a, b) => {
      const [x, y] = [column.get(a), column.get(b)]
      if (x === null) return 1
      if (y === null) return -1
      const order = typeof x === 'number' ? x - (y as number) : String(x).localeCompare(String(y))
      return sort.desc ? -order : order
    })
  }, [network.stations, query, sort, trunk])

  const trunkColor = (id: string) => network.lines.find((line) => line.id.split('-')[0] === id)?.color

  return (
    <>
      <header className="panel-head">
        <h1>Stations</h1>
        <p>Every station’s reliability for riders and its daily entries. Select a row to find it on the map.</p>
      </header>
      <div className="table-tools">
        <label className="search-field">
          <Search size={15} />
          <input onChange={(event) => setQuery(event.target.value)} placeholder="Filter stations" type="search" value={query} />
        </label>
        <div aria-label="Line" className="chip-row" role="radiogroup">
          <button aria-checked={!trunk} onClick={() => setTrunk(undefined)} role="radio" type="button">All</button>
          {TRUNKS.map((id) => (
            <button aria-checked={trunk === id} key={id} onClick={() => setTrunk(trunk === id ? undefined : id)} role="radio" type="button">
              <i style={{ background: trunkColor(id) }} />{id}
            </button>
          ))}
        </div>
      </div>
      <div className="table-scroll station-table">
        <table>
          <thead>
            <tr>
              {COLUMNS.map((column) => (
                <th aria-sort={sort.key === column.key ? (sort.desc ? 'descending' : 'ascending') : undefined} className={column.numeric ? 'is-numeric' : ''} key={column.key} scope="col">
                  <button onClick={() => setSort((s) => ({ desc: s.key === column.key ? !s.desc : !!column.numeric, key: column.key }))} type="button">
                    {column.label}
                    {sort.key === column.key ? (sort.desc ? <ArrowDown size={12} /> : <ArrowUp size={12} />) : null}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((station) => (
              <tr aria-selected={station.id === selectedId} key={station.id} onClick={() => onSelect(station.id)}>
                {COLUMNS.map((column) => (
                  <td className={column.numeric ? 'is-numeric' : ''} key={column.key}>
                    {column.key === 'name' ? (
                      <button className="row-link" onClick={(event) => { event.stopPropagation(); onSelect(station.id) }} title={station.lines.map((id) => lineName(network.lineById.get(id), id)).join(', ')} type="button">
                        <LineDots ids={station.lines} lineById={network.lineById} />{station.name}
                      </button>
                    ) : column.render(station)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 ? <p className="hint">No station matches.</p> : null}
      </div>
    </>
  )
}
