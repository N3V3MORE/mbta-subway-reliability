import { useMemo, useRef } from 'react'
import { Pause, Play } from 'lucide-react'
import { formatClock } from '../data/format'
import type { PreparedReplay } from '../data/replay'
import type { Incident } from '../types'

const BIN_SECONDS = 300

/** Trains in service in each five-minute bin: the day's shape, drawn behind the scrubber. */
function inService(replay: PreparedReplay) {
  const bins = new Array(Math.ceil((replay.end - replay.start) / BIN_SECONDS) + 1).fill(0)
  for (const trip of replay.trips) {
    for (let b = Math.floor((trip.start - replay.start) / BIN_SECONDS); b <= Math.floor((trip.end - replay.start) / BIN_SECONDS); b += 1) bins[b] += 1
  }
  return bins
}

/**
 * The replay's transport bar, docked under the map: play, speed, the clock, and a
 * scrubber drawn over the number of trains in service through the day, with the
 * MBTA's alerts marked where they were raised.
 */
export function Timeline({ incidents, onPlay, onSeconds, onSpeed, playing, replay, seconds, speed, trains }: {
  incidents: Incident[]
  onPlay: () => void
  onSeconds: (seconds: number) => void
  onSpeed: () => void
  playing: boolean
  replay: PreparedReplay | null
  seconds: number
  speed: number
  trains: number
}) {
  const trackRef = useRef<HTMLDivElement>(null)
  const bins = useMemo(() => (replay ? inService(replay) : []), [replay])
  const start = replay?.start ?? 5 * 3_600
  const end = replay?.end ?? 26 * 3_600
  const span = end - start
  const at = (t: number) => `${((t - start) / span) * 100}%`
  const peak = Math.max(1, ...bins)
  const area = bins.length > 1
    ? `M0,40 ${bins.map((n, i) => `L${(i / (bins.length - 1)) * 1000},${40 - (n / peak) * 36}`).join(' ')} L1000,40 Z`
    : ''
  const hours = []
  for (let h = Math.ceil(start / 3_600); h * 3_600 <= end; h += 1) hours.push(h * 3_600)

  const seek = (clientX: number) => {
    const box = trackRef.current!.getBoundingClientRect()
    onSeconds(start + Math.min(1, Math.max(0, (clientX - box.left) / box.width)) * span)
  }

  return (
    <section aria-label="Replay controls" className="timeline">
      <div className="timeline-controls">
        <button aria-label={playing ? 'Pause' : 'Play'} className="play-button" disabled={!replay} onClick={onPlay} type="button">
          {playing ? <Pause fill="currentColor" size={16} /> : <Play fill="currentColor" size={16} />}
        </button>
        <div className="timeline-clock">
          <strong>{formatClock(seconds)}</strong>
          <span>{replay ? `${trains} trains running` : 'Loading trains…'}</span>
        </div>
        <button aria-label={`Replay speed: ${speed / 60} minutes of service per second. Change`} className="speed-button" onClick={onSpeed} title="Minutes of service per second" type="button">
          {speed / 60} min<small>/s</small>
        </button>
      </div>
      <div className="timeline-scrub">
        <div className="timeline-alerts">
          {incidents.filter((incident) => incident.start >= start && incident.start <= end).map((incident) => (
            <button
              aria-label={`Alert at ${formatClock(incident.start)}: ${incident.text}`}
              key={incident.id}
              onClick={() => onSeconds(incident.start)}
              style={{ left: at(incident.start) }}
              title={`${formatClock(incident.start)} · ${incident.text}`}
              type="button"
            />
          ))}
        </div>
        <div
          aria-label="Replay time"
          aria-valuemax={end}
          aria-valuemin={start}
          aria-valuenow={Math.round(seconds)}
          aria-valuetext={formatClock(seconds)}
          className="timeline-track"
          onKeyDown={(event) => {
            const step = event.shiftKey ? 3_600 : 300
            if (event.key === 'ArrowRight') onSeconds(Math.min(end, seconds + step))
            if (event.key === 'ArrowLeft') onSeconds(Math.max(start, seconds - step))
          }}
          onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); seek(event.clientX) }}
          onPointerMove={(event) => { if (event.buttons) seek(event.clientX) }}
          ref={trackRef}
          role="slider"
          tabIndex={replay ? 0 : -1}
        >
          <svg aria-hidden="true" preserveAspectRatio="none" viewBox="0 0 1000 40">
            <path className="service-area" d={area} />
          </svg>
          <span className="timeline-played" style={{ width: at(seconds) }} />
          <span className="timeline-head" style={{ left: at(seconds) }} />
        </div>
        <div aria-hidden="true" className="timeline-hours">
          {hours.map((h) => <span key={h} style={{ left: at(h) }}>{(h / 3_600) % 3 === 0 ? formatClock(h) : ''}</span>)}
        </div>
      </div>
    </section>
  )
}
