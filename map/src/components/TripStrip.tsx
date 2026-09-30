import { useEffect, useRef } from 'react'
import { formatClock, formatDelay } from '../data/format'
import { bandOf, DELAY_BANDS, type PreparedTrip, stopAt } from '../data/replay'
import type { PreparedNetwork } from '../data/network'

/**
 * One train's run as a strip map: its stops top to bottom on a rail in the line's
 * colour, filled up to where the train is now. Each stop shows when it arrived,
 * how far off the timetable it was, and the model's forecast for it made one stop
 * earlier. Selecting a stop moves the replay to that arrival.
 */
export function TripStrip({ network, onSeconds, seconds, trip }: {
  network: PreparedNetwork
  onSeconds: (seconds: number) => void
  seconds: number
  trip: PreparedTrip
}) {
  const color = network.lineById.get(trip.line)?.color ?? 'var(--text)'
  const reached = seconds < trip.start ? -1 : stopAt(trip.a, seconds)
  const between = reached >= 0 && reached < trip.a.length - 1 && seconds > trip.leave[reached]
  const currentRef = useRef<HTMLLIElement>(null)
  const firstRef = useRef(true)

  // Follow the train down the list as the replay moves, but open at the top: the header first.
  useEffect(() => {
    if (firstRef.current) { firstRef.current = false; return }
    currentRef.current?.scrollIntoView({ block: 'nearest' })
  }, [reached])

  return (
    <ol className="trip-strip" style={{ '--line-color': color } as React.CSSProperties}>
      {trip.s.map((stationIndex, i) => {
        const passed = i <= reached
        const forecast = trip.p[i]
        const band = DELAY_BANDS[bandOf(trip.l[i])]
        return (
          <li
            className={`${passed ? 'is-passed' : ''}${i === reached ? ' is-current' : ''}${i === reached && between ? ' is-leaving' : ''}`}
            key={`${stationIndex}-${i}`}
            ref={i === reached ? currentRef : undefined}
          >
            <button onClick={() => onSeconds(trip.a[i])} type="button">
              <span className="strip-rail" aria-hidden="true"><i /></span>
              <span className="strip-name">{network.stations[stationIndex].name}</span>
              <span className="strip-time">{formatClock(trip.a[i])}</span>
              <span className="strip-delay" title="Delay against the timetable">{formatDelay(trip.d[i])}</span>
              <span className="strip-detail">
                <i className="band-dot" style={{ background: band.color }} />
                {band.label}
                {forecast === null ? null : <> · forecast {formatDelay(forecast)}, off by {formatDelay(Math.abs(forecast - trip.d[i])).slice(1)}</>}
              </span>
            </button>
          </li>
        )
      })}
    </ol>
  )
}
