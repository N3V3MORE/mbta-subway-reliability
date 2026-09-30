import { LOST_METRICS, type LostMetric, MOVING, STANDING } from '../data/lost'
import { clusterLabel } from '../data/labels'
import { METRICS, type MetricId, RELIABILITY_COLORS } from '../data/network'
import { DELAY_BANDS } from '../data/replay'
import type { Mode } from '../map/NetworkMap'

/** What height and colour mean on the map right now, in its corner, so nobody has to hunt for it. */
export function MapKey({ lostMetric, metric, mode }: { lostMetric: LostMetric; metric: MetricId; mode: Mode }) {
  if (mode === 'replay') {
    return (
      <aside aria-label="Map key" className="map-key is-replay">
        <p>Trains: colour and height show how much longer riders waited for them at their last stop</p>
        <ul>{DELAY_BANDS.map((band) => <li key={band.label}><i style={{ background: band.color }} />{band.label}</li>)}</ul>
      </aside>
    )
  }
  if (mode === 'lost') {
    return (
      <aside aria-label="Map key" className="map-key">
        <p>{LOST_METRICS[lostMetric].detail}</p>
        <ul>
          <li><i className="is-line" style={{ background: MOVING }} />Track: time lost moving (brighter, wider = more)</li>
          <li><i style={{ background: STANDING }} />Columns: time lost standing at platforms</li>
        </ul>
      </aside>
    )
  }
  return (
    <aside aria-label="Map key" className="map-key">
      <p>Column height: {METRICS[metric].detail.toLowerCase()}</p>
      <ul>{Object.entries(RELIABILITY_COLORS).map(([name, color]) => <li key={name}><i style={{ background: color }} />{clusterLabel(name)} station</li>)}</ul>
    </aside>
  )
}
