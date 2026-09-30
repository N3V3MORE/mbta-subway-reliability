import { useState } from 'react'
import { ArrowLeft } from 'lucide-react'
import { formatClock, formatDate, formatDelay, formatPercent } from '../data/format'
import { isInterchange, METRICS, type MetricId, type PreparedNetwork, RELIABILITY_COLORS, UNCLUSTERED_COLOR } from '../data/network'
import { LineDots, LineTag } from '../components/LineTag'
import { clusterLabel } from '../data/labels'
import type { Station } from '../types'

const LIST_LENGTH = 12

export function NetworkPanel({ metric, network, onMetric, onSelect, selected }: {
  metric: MetricId
  network: PreparedNetwork
  onMetric: (metric: MetricId) => void
  onSelect: (stationId: string | undefined) => void
  selected?: Station
}) {
  if (selected) return <StationDetail network={network} onBack={() => onSelect(undefined)} station={selected} />
  return <StationRanking metric={metric} network={network} onMetric={onMetric} onSelect={onSelect} />
}

function StationRanking({ metric, network, onMetric, onSelect }: {
  metric: MetricId
  network: PreparedNetwork
  onMetric: (metric: MetricId) => void
  onSelect: (stationId: string) => void
}) {
  const [showAll, setShowAll] = useState(false)
  const { format, value } = METRICS[metric]
  const ranked = network.stations
    .flatMap((station) => { const v = value(station); return v === null ? [] : [{ station, v }] })
    .sort((a, b) => b.v - a.v)
  const top = ranked[0]?.v ?? 1
  const clusters = Object.entries(RELIABILITY_COLORS).map(([name, color]) => ({
    color, count: network.stations.filter((s) => s.reliability === name).length, name,
  }))

  return (
    <>
      <header className="panel-head">
        <h1>Network</h1>
        <p>{network.stations.length} stations, {formatDate(network.window[0], 'short')} to {formatDate(network.window[1], 'short')}. Each column rises with the measure you pick.</p>
      </header>

      <section className="panel-section">
        <div aria-label="Column height" className="segmented" role="radiogroup">
          {(Object.keys(METRICS) as MetricId[]).map((id) => (
            <button aria-checked={metric === id} key={id} onClick={() => onMetric(id)} role="radio" type="button">{METRICS[id].label}</button>
          ))}
        </div>
        <p className="hint">{METRICS[metric].detail}.</p>
        <ul aria-label="Column colour" className="key-list">
          {clusters.map((cluster) => (
            <li key={cluster.name}><i style={{ background: cluster.color }} />{clusterLabel(cluster.name)}<span>{cluster.count} stations</span></li>
          ))}
        </ul>
      </section>

      <section className="panel-section">
        <h2>{metric === 'entries' ? 'Busiest stations' : 'Where riders wait longest'}</h2>
        <ol className="rank-list">
          {(showAll ? ranked : ranked.slice(0, LIST_LENGTH)).map(({ station, v }, i) => (
            <li key={station.id}>
              <button onClick={() => onSelect(station.id)} type="button">
                <span className="rank">{i + 1}</span>
                <span className="rank-name"><LineDots ids={station.lines} lineById={network.lineById} />{station.name}</span>
                <span className="rank-value">{format(v)}</span>
                <span aria-hidden="true" className="rank-bar"><i style={{ background: RELIABILITY_COLORS[station.reliability ?? ''] ?? UNCLUSTERED_COLOR, width: `${(Math.max(0, v) / top) * 100}%` }} /></span>
              </button>
            </li>
          ))}
        </ol>
        {ranked.length > LIST_LENGTH ? (
          <button className="text-button" onClick={() => setShowAll((value) => !value)} type="button">
            {showAll ? 'Show fewer' : `Show all ${ranked.length}`}
          </button>
        ) : null}
      </section>
    </>
  )
}

function StationDetail({ network, onBack, station }: { network: PreparedNetwork; onBack: () => void; station: Station }) {
  const late = METRICS.late.value(station)
  return (
    <>
      <header className="panel-head">
        <button className="back-button" onClick={onBack} type="button"><ArrowLeft size={15} /> All stations</button>
        <h1>{station.name}</h1>
        <p className="tag-row">
          {station.lines.map((id) => <LineTag id={id} key={id} line={network.lineById.get(id)} />)}
          {isInterchange(station) ? <span className="chip">Interchange</span> : null}
        </p>
      </header>

      <section className="panel-section">
        <dl className="stat-grid">
          <div><dt>Late arrivals</dt><dd>{late === null ? '—' : formatPercent(late)}</dd><small>riders waited 5+ min longer than planned</small></div>
          <div><dt>Typical extra wait</dt><dd>{station.median === null ? '—' : formatDelay(station.median)}</dd><small>median, min:sec</small></div>
          <div><dt>Bad day</dt><dd>{station.p90 === null ? '—' : formatDelay(station.p90)}</dd><small>1 arrival in 10 is later</small></div>
          <div><dt>Entries a day</dt><dd>{station.entries === null ? 'Ungated' : METRICS.entries.format(station.entries)}</dd><small>{station.entries === null ? 'no faregates here' : 'mean, faregate taps'}</small></div>
        </dl>
      </section>

      <section className="panel-section">
        <h2>Clusters</h2>
        <dl className="pair-list">
          <div>
            <dt>Reliability</dt>
            <dd><i className="swatch" style={{ background: RELIABILITY_COLORS[station.reliability ?? ''] ?? UNCLUSTERED_COLOR }} />{clusterLabel(station.reliability) ?? 'Not clustered'}</dd>
          </div>
          <div><dt>Demand</dt><dd>{clusterLabel(station.demand) ?? 'No ridership data'}</dd></div>
        </dl>
      </section>

      {station.profile ? (
        <section className="panel-section">
          <h2>Entries through the day</h2>
          <DemandProfile profile={station.profile} />
        </section>
      ) : null}
    </>
  )
}

/** Share of a day's entries per half-hour, one area in the model accent. */
function DemandProfile({ profile }: { profile: number[] }) {
  const max = Math.max(...profile)
  const peak = profile.indexOf(max)
  const points = profile.map((share, i) => `${(i / (profile.length - 1)) * 300},${76 - (share / max) * 70}`)
  return (
    <figure className="profile">
      <svg aria-label={`Busiest half-hour ${formatClock(peak * 1_800)}`} preserveAspectRatio="none" role="img" viewBox="0 0 300 80">
        <line className="grid" x1="0" x2="300" y1="76" y2="76" />
        <path className="profile-area" d={`M0,76 L${points.join(' L')} L300,76 Z`} />
        <polyline className="profile-line" points={points.join(' ')} />
      </svg>
      <div className="profile-axis"><span>00:00</span><span>06:00</span><span>12:00</span><span>18:00</span><span>24:00</span></div>
      <figcaption>Busiest half-hour: {formatClock(peak * 1_800)}–{formatClock((peak + 1) * 1_800)}, {formatPercent(max)} of the day’s entries.</figcaption>
    </figure>
  )
}
